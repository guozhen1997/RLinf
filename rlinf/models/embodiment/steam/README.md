# STEAM Value Model

Robust value model for VLA offline RL, built to sidestep the
frame-index shortcut that plagues scalar progress regression. Takes a pair
``(frame_t, frame_{t+k})`` with a fixed language instruction (and optional
proprio state) and predicts whether the pair is **progress** (forward in
a success demo) or **regress** (a rewound counterfactual).

The folder structure mirrors the sibling `value_model/` on purpose —
same file roles, same public entry points, same observation-dict contract
— so downstream code that dispatches on `model_type` slots either
backbone in without extra plumbing.

---

## File layout

```
steam/
├── configuration.py            # SteamConfig
├── modeling_steam.py           # SteamBackbone (SigLIP + Gemma3 + 2-way head)
├── modeling_critic.py          # SteamCriticModel, CriticOutput
├── ensemble_modeling_critic.py # EnsembleSteamCriticModel
├── processing.py               # SteamProcessor / SteamImageProcessor
├── __init__.py                 # get_model factory
└── README.md                   # this file
```

Design highlights:

| Piece | `steam` |
| --- | --- |
| Config | `SteamConfig` (label_smoothing, num_frames_per_pair) |
| Backbone | `SteamBackbone` with **per-frame concat** (ordering-preserving) and **2-way logit head** |
| Loss | **2-class cross-entropy** over `{regress, progress}` with label smoothing |
| Critic output | `predicted_values` = `P(progress)`, `logits` shape `[B, 2]`, `probs` shape `[B, 2]` |
| Image keys | `("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")` — **camera views** are the dict keys; the pair axis is a separate stacked dim inside each tensor |
| Dataset | `PairDataset` (success demo → forward/rewound pair) |

---

## How classification works

The head outputs **two logits** — `[regress_logit, progress_logit]` — and the
critic softmaxes them:

```
probs[:, 0] = P(regress)         # pair walks backward in original time
probs[:, 1] = P(progress)        # forward pair from a success demo
predicted_values = probs[:, 1]   # exposed as "the value" for downstream code
```

Training uses `F.cross_entropy(logits, targets, label_smoothing=cfg.label_smoothing)`
with `targets ∈ {0, 1}` derived from the dataset's `labels ∈ {-1, +1}`.
Math-wise this is equivalent to a single-logit sigmoid head with BCE, but
the two-logit formulation makes the classification semantics visible in
the code and outputs.

---

## Running

```bash
python examples/steam/value/train_steam.py
python examples/steam/value/train_steam.py \
    data.k=6 training.batch_size=64 logging.use_wandb=true

# Multi-GPU DDP:
torchrun --standalone --nproc_per_node=4 \
    examples/steam/value/train_steam.py training.batch_size=64
```

Unit tests for the pair pipeline (no heavy deps):

```bash
python rlinf/data/datasets/steam/pair_dataset.py
```

---

## Data contract (collated batch)

```python
{
    "observation": {
        # Each dict key is a CAMERA VIEW; the value carries both time steps
        # stacked along a new dim=1 (num_frames_per_pair, default 2).
        "images": {
            "base_0_rgb":        Tensor[B, 2, 3, H, W],
            "left_wrist_0_rgb":  Tensor[B, 2, 3, H, W],
            "right_wrist_0_rgb": Tensor[B, 2, 3, H, W],   # zero-filled if missing
        },
        "image_masks": {
            "base_0_rgb":        Tensor[B, 2] bool,
            "left_wrist_0_rgb":  Tensor[B, 2] bool,
            "right_wrist_0_rgb": Tensor[B, 2] bool,
        },
        "tokenized_prompt":      Tensor[B, T],
        "tokenized_prompt_mask": Tensor[B, T],
    },
    "labels": Tensor[B] float32,   # +1 progress, -1 regress
    "episode": Tensor[B],
    "frame_idx_t":  Tensor[B],
    "frame_idx_tk": Tensor[B],
}
```

**Why this shape.** Cameras and pair-frames are orthogonal axes: the vision
encoder runs on every ``(camera, frame)`` tile, per-frame features are
obtained by **mean-pooling across cameras**,
and the pair signal comes from **concatenating the per-frame features in
``(t, t+k)`` order** before the language feature is concatenated and the
2-way head fires. Separating the two axes also makes it cheap to add
more views (wrist / head / external) without touching the head dim.

`SteamCriticModel.forward(observation, labels)` returns a
`CriticOutput` with:

* `loss`: scalar mean cross-entropy.
* `logits`: `[B, 2]` — `[regress_logit, progress_logit]`.
* `probs`: `[B, 2]` — softmax probs.
* `predicted_values`: `[B]` — `P(progress)` slice.
* `hidden_states`: `[B, fusion_hidden_dim * (num_frames_per_pair + 1)]`.
* `cat_acc_best`: scalar classification accuracy.

---

## Config knobs (`examples/steam/value/config/steam_value_model.yaml`)

* `data.k` — forward pair stride (same for positive and negative). The
  negative pair is just a positive pair fed in reversed order, so no
  separate rewind-length knob is needed.
* `model.num_frames_per_pair` — defaults to 2 (one camera slot per frame).
* `model.label_smoothing` — CE label smoothing (paper default 0.05).
* `model.freeze_language_model` — defaults to `true` (single-task, fixed
  instruction).
* `model.include_state_in_prompt` — if `true`, the dataset threads proprio
  state into the prompt via the processor's digitisation path (requires
  the upstream state to already be normalised into `[-1, 1]`).

---

## Diagnostic: time-counter shortcut

The trainer runs `diagnose_time_counter_shortcut` each epoch. It builds
matched pairs that share the `(t, t+k)` offsets but swap the trajectory
that provides `frame_{t+k}`:

```
normal:   (frame_A[t], frame_A[t+k])
shuffled: (frame_A[t], frame_B[t+k])     # B ≠ A, same t+k
```

A healthy model yields non-zero `mean_abs_prob_diff` and `flip_rate`; a
model that has collapsed to reading the frame index yields ~0 on both.

---

## Extension points

* **Tri-state** (progress / static / regress): bump `NUM_CLASSES` in
  `modeling_steam.py` and add a synthesiser for the static class.
* **Multi-`k` mixing**: instantiate one :class:`PairDataset` per ``k`` and
  wrap them with ``ConcatDataset`` (or a weighted mixture).
* **Action input**: add an action branch alongside the image/language
  ones, feed `actions[t:t+k]` through a small MLP projector, and widen the
  fusion concat. The rest of the critic API stays the same.
* **Multi-task language**: drop `freeze_language_model` and rotate
  prompts in the dataset; the processor already supports per-sample
  prompts.
