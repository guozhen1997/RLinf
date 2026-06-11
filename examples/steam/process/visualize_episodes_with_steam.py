# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-episode visualization for the STEAM value model.

Parallel to :mod:`visualize_episodes_with_value`, but for
:class:`SteamCriticModel` / :class:`EnsembleSteamCriticModel`
(steam). The binary critic scores **pairs** ``(frame_t, frame_{t+k})``
with a single scalar **signed progress** in ``[-1, 1]``
(see :meth:`SteamCriticModel._predicted_signed_value`), rather than
a continuous value ``V(o_t)`` — there is no advantage formula or return
normalization here. For ``num_bins == 2`` this equals
``2 · P(progress) - 1``; positive = looks forward, negative = looks
reversed, zero = undecided.

Works with both single-model and ensemble checkpoints via the same
:meth:`SteamCriticModel.from_checkpoint` entry point:

    * **Single** (``ensemble_size=1``): one signed-progress score per pair.
    * **Ensemble** (``ensemble_size>1``): each of the ``E`` members emits
      its own signed-progress score; the wrapper aggregates those into a
      single ``predicted_values`` score using ``inference_mode``
      (``mo`` = member mean, ``wco`` = worst-case over members,
      ``uwo`` = mean − λ·variance). The renderer draws every member
      curve in its own color AND the aggregated ("total") curve as a
      bold overlay so you can see both the per-member spread and the
      aggregated decision simultaneously.

For each selected episode we reuse the canonical training-inference path
(:class:`PairDataset` + :class:`BinaryPairDataCollator`), restricted to
the **positive** pair samples (flat index ``2 * pair_position``), and
run the model once per frame. The resulting per-frame scalar is the
signed-progress score for the stride-``k`` pair (clamped at the episode
boundary); frame ``T-1`` has no valid pair.

Rendered per episode:

    1. Per-pair forward confidence — each ensemble member in its own
       color, plus the ``inference_mode``-aggregated score as a bold
       overlay. *Positive = local forward motion; ~0 = undecided; negative
       = looks reversed.* For a single model the two curves coincide.
    2. Member variance (ensemble only) — raw variance across members.
    3. ``S(t) = Σ_{i<=t} V_i`` where ``V_i`` is the per-frame signed
       progress — cumulative signed progress, drawn for both the
       aggregated score AND each member in matching colors. *Each step
       adds +1 when the model is fully confident it's forward, −1 when
       fully regressing. A successful trajectory grows approximately
       linearly; a stalled / failing one plateaus or dips.*

The stride ``k`` is pulled from ``data.k`` and **must match the training
config** — the model was only trained on that fixed stride and will not
generalize well to other values.

By default every ``model.*`` knob in the YAML is ``null``, which means
"read the value from the checkpoint's ``config.json``". Override on the
CLI (e.g. ``model.inference_mode=mo``) only when you want to deviate
from the trained setting.

Usage:
    cd examples/steam/process
    python visualize_episodes_with_steam.py \\
        visualize.checkpoint_dir=/path/to/steam_ckpt \\
        data.train_data_paths.0.dataset_path=/path/to/dataset \\
        model.tokenizer_path=/path/to/gemma-3-270m \\
        model.vision_repo_id=/path/to/siglip-so400m-patch14-384 \\
        model.language_repo_id=/path/to/gemma-3-270m \\
        +visualize.num_episodes=5 \\
        +visualize.output_dir=./viz_binary_out

    # Explicit episodes:
    ... +visualize.episodes="[0,7,12]"

    # Plots only:
    ... +visualize.no_video=true

    # 4-member ensemble checkpoint — ensemble_size/inference_mode are read
    # from the checkpoint by default; override only to test other modes:
    python visualize_episodes_with_steam.py \\
        visualize.checkpoint_dir=/path/to/ensemble_ckpt \\
        data.train_data_paths.0.dataset_path=/path/to/dataset \\
        model.tokenizer_path=/path/to/gemma-3-270m \\
        model.vision_repo_id=/path/to/siglip-so400m-patch14-384 \\
        model.language_repo_id=/path/to/gemma-3-270m \\
        data.k=8 \\
        'data.camera_keys=[image,wrist_image]' \\
        visualize.num_episodes=10 \\
        visualize.output_dir=./viz_binary_ensemble_out
"""

import gc
import logging
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Rectangle
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent))
sys.path.insert(0, str(_THIS_DIR))

from rlinf.data.datasets.steam import (  # noqa: E402
    BinaryPairDataCollator,
    PairDataset,
    expected_signed_stride,
)
from rlinf.models.embodiment.steam.modeling_critic import (  # noqa: E402
    SteamCriticModel,
)

# ---------------------------------------------------------------------------
# Multi-bin helpers — only exercised when model.config.num_bins > 2
# ---------------------------------------------------------------------------


def _entropy_nats(probs: np.ndarray) -> np.ndarray:
    """Per-row entropy in nats. Works for any ``[..., num_bins]`` shape.

    Uses the standard ``-Σ p·log p`` formula with ``p·log p = 0`` whenever
    ``p == 0`` (matches scipy.stats.entropy). The max value is
    ``log(num_bins)`` for a uniform distribution and 0 when a single bin
    owns the mass.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def _signed_stride_bin_tick_labels(K: int, num_bins: int) -> list[str]:
    """Y-tick labels for the bin-distribution heatmap (``[low, high]``).

    Example K=8, num_bins=8 → ``["[-8, -7]", "[-6, -5]", ..., "[7, 8]"]``.
    """
    strides_per_bin = (2 * K) // num_bins
    half = num_bins // 2
    labels: list[str] = []
    for b in range(num_bins):
        if b < half:
            low = -K + b * strides_per_bin
        else:
            low = 1 + (b - half) * strides_per_bin
        high = low + strides_per_bin - 1
        labels.append(f"[{low:+d}, {high:+d}]" if low != high else f"{low:+d}")
    return labels


def detect_image_keys(sample: dict) -> list[str]:
    """Auto-detect image-like keys in a LeRobot sample (inlined from
    ``visualize_advantage_dataset.detect_image_keys`` to avoid pulling
    in that module's newer ``lerobot.common.datasets`` import path)."""
    keys: list[str] = []
    for key in sample.keys():
        if key.startswith("observation.images.") or key in (
            "image",
            "wrist_image",
            "front_image",
            "left_image",
            "right_image",
            "face_view",
            "left_wrist_view",
            "right_wrist_view",
        ):
            val = sample[key]
            arr = (
                val.cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
            )
            if arr.ndim >= 3:
                keys.append(key)
    return keys


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode selection
# ---------------------------------------------------------------------------


def _select_episodes(total_episodes: int, viz_cfg: DictConfig) -> list[int]:
    """Pick episode indices from explicit list or random sample."""
    episodes = viz_cfg.get("episodes", None)
    if episodes is not None:
        ep_list = [int(e) for e in episodes]
        bad = [e for e in ep_list if e < 0 or e >= total_episodes]
        if bad:
            raise ValueError(
                f"Requested episodes {bad} out of range "
                f"(dataset has {total_episodes} episodes)."
            )
        return sorted(set(ep_list))

    num_episodes = int(viz_cfg.get("num_episodes", 5))
    seed = int(viz_cfg.get("seed", 42))
    if num_episodes <= 0 or num_episodes >= total_episodes:
        return list(range(total_episodes))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(total_episodes, num_episodes, replace=False).tolist())


# ---------------------------------------------------------------------------
# Pair-position bookkeeping (maps episodes ↔ flat PairDataset indices)
# ---------------------------------------------------------------------------


def _episode_to_pair_range(pair_ds: PairDataset, episode_index: int) -> tuple[int, int]:
    """Return the ``[pair_position_start, pair_position_end)`` slice for an episode.

    ``PairDataset.__len__`` is ``2 * num_pair_positions`` (positive + negative per
    anchor). The flat positive-sample index for pair position ``p`` is ``2 * p``,
    which is what we use for inference.
    """
    eligible = pair_ds.eligible_episodes
    try:
        slot = eligible.index(int(episode_index))
    except ValueError as err:
        raise ValueError(
            f"Episode {episode_index} is not in the PairDataset's eligible set "
            f"(eligible={eligible[:10]}{'...' if len(eligible) > 10 else ''}). "
            "This usually means the episode is shorter than "
            f"min_episode_length={pair_ds._min_episode_length} or was filtered "
            "by the only_success flag."
        ) from err

    pair_position_end = int(pair_ds._pair_position_ends[slot])
    pair_position_start = int(pair_ds._pair_position_ends[slot - 1]) if slot > 0 else 0
    return pair_position_start, pair_position_end


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def _score_episode_pairs(
    model: SteamCriticModel,
    pair_ds: PairDataset,
    collator: BinaryPairDataCollator,
    episode_index: int,
    device: str,
    batch_size: int,
    num_workers: int,
    num_bins: int = 2,
    stride_k: int = 0,
) -> dict[int, dict[str, object]]:
    """Run the binary critic on every positive pair of one episode.

    Returns a mapping ``frame_idx_t -> score bundle``. When ``num_bins > 2``
    the bundle also carries the per-frame bin distribution (aggregated +
    per-member), ``E[signed stride]/K`` (aggregated + per-member), and
    per-frame entropy — everything the rich multi-bin plot path needs.
    """
    pp_start, pp_end = _episode_to_pair_range(pair_ds, episode_index)
    # Positive pair at pair_position ``p`` lives at flat index ``2 * p``.
    flat_indices = [2 * p for p in range(pp_start, pp_end)]
    if not flat_indices:
        return {}

    subset = torch.utils.data.Subset(pair_ds, flat_indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        shuffle=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    def _to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device, non_blocking=True)
        if isinstance(x, dict):
            return {k: _to_device(v) for k, v in x.items()}
        return x

    is_multi_bin = num_bins > 2
    if is_multi_bin and stride_k < 1:
        raise ValueError(f"Multi-bin mode requires stride_k >= 1, got {stride_k}.")

    out: dict[int, dict[str, object]] = {}
    for batch in tqdm(
        loader, desc=f"signed progress ep={episode_index}", leave=False, unit="batch"
    ):
        observation = _to_device(batch["observation"])
        result = model.predict(observation)
        signed_progress = result.predicted_values.detach().float().cpu().numpy()
        member_values = result.member_predicted_values.detach().float().cpu().numpy()
        value_mean = result.prediction_mean.detach().float().cpu().numpy()
        value_min = result.prediction_min.detach().float().cpu().numpy()
        value_variance = result.prediction_variance.detach().float().cpu().numpy()
        frame_idx_t = batch["frame_idx_t"].cpu().numpy().tolist()
        if len(signed_progress) != len(frame_idx_t):
            raise RuntimeError(
                f"Predicted {len(signed_progress)} values for {len(frame_idx_t)} pairs"
            )

        # Multi-bin: derive bin-level quantities from the distributions.
        # result.probs shape is [B, num_bins]; result.member_probs, populated
        # by EnsembleCriticOutput under multi-bin, is [E, B, num_bins].
        # Single-model ensemble wrapper with ensemble_size=1 still populates
        # it as [1, B, num_bins] — the viz path tolerates E==1.
        if is_multi_bin:
            agg_probs = result.probs.detach().float().cpu().numpy()  # [B, num_bins]
            if result.member_probs is None:
                raise RuntimeError(
                    "Multi-bin visualization requires EnsembleCriticOutput."
                    "member_probs; got None. Load via "
                    "SteamCriticModel.from_checkpoint which routes "
                    "through the ensemble wrapper even for ensemble_size=1."
                )
            mem_probs = (
                result.member_probs.detach().float().cpu().numpy()
            )  # [E, B, num_bins]
            # Scalar E[s]/K — aggregated and per-member.
            agg_es = expected_signed_stride(agg_probs, stride_k, num_bins)
            agg_es_normalized = agg_es / float(stride_k)
            mem_es = expected_signed_stride(mem_probs, stride_k, num_bins)
            mem_es_normalized = mem_es / float(stride_k)
            # Entropy per frame, aggregated + per-member (then averaged).
            entropy_agg = _entropy_nats(agg_probs)  # [B]
            entropy_members = _entropy_nats(mem_probs)  # [E, B]
            entropy_member_mean = entropy_members.mean(axis=0)  # [B]

        for batch_idx, (ft, p) in enumerate(zip(frame_idx_t, signed_progress)):
            bundle: dict[str, object] = {
                "value": float(p),
                "member_values": member_values[:, batch_idx].tolist(),
                "value_mean": float(value_mean[batch_idx]),
                "value_min": float(value_min[batch_idx]),
                "value_variance": float(value_variance[batch_idx]),
            }
            if is_multi_bin:
                bundle.update(
                    {
                        "aggregated_probs": agg_probs[batch_idx].tolist(),
                        "member_probs": mem_probs[:, batch_idx, :].tolist(),
                        "expected_stride_normalized": float(
                            agg_es_normalized[batch_idx]
                        ),
                        "member_expected_stride_normalized": mem_es_normalized[
                            :, batch_idx
                        ].tolist(),
                        "entropy_aggregated": float(entropy_agg[batch_idx]),
                        "entropy_member_mean": float(entropy_member_mean[batch_idx]),
                    }
                )
            out[int(ft)] = bundle

    return out


# ---------------------------------------------------------------------------
# Animation writer resolution — tolerate missing system ffmpeg.
# ---------------------------------------------------------------------------

_FFMPEG_PATH_CONFIGURED = False


def _ensure_ffmpeg_on_matplotlib() -> None:
    """Point ``animation.ffmpeg_path`` at a usable binary, once per process.

    matplotlib's default ``animation.ffmpeg_path`` is the literal string
    ``"ffmpeg"``, which only resolves when a system-wide ffmpeg is on
    ``$PATH``. In several of our envs (openpi, openvla) ffmpeg is **not**
    installed at the system level — but the ``imageio-ffmpeg`` pip package
    ships a static binary that we can register with matplotlib instead.

    If neither is available, we raise early rather than produce a cryptic
    ``FileNotFoundError: 'ffmpeg'`` deep inside ``anim.save``.
    """
    global _FFMPEG_PATH_CONFIGURED
    if _FFMPEG_PATH_CONFIGURED:
        return

    import shutil

    if shutil.which("ffmpeg"):
        _FFMPEG_PATH_CONFIGURED = True
        return

    try:
        import imageio_ffmpeg

        ff_path = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "No ffmpeg binary found. Install either (a) system ffmpeg "
            "(e.g. `apt-get install ffmpeg`) or (b) the `imageio-ffmpeg` "
            "pip package, which ships a static binary. "
            "Alternatively pass visualize.no_video=true to skip video "
            f"generation. Underlying error: {e}"
        ) from e

    matplotlib.rcParams["animation.ffmpeg_path"] = ff_path
    _FFMPEG_PATH_CONFIGURED = True
    logger.info("Using imageio_ffmpeg ffmpeg binary: %s", ff_path)


# ---------------------------------------------------------------------------
# Progress accumulator
# ---------------------------------------------------------------------------


def _cumulative_signed_progress(signed_progress) -> np.ndarray:
    """Signed forward-progress integrator: ``S(t) = Σ_{i<=t} V_i``.

    Rationale: ``V_t`` is a per-pair local judgment in ``[-1, 1]`` — at
    frame ``t`` the model is asked whether the stride-k pair looks
    forward. A single ``V_t`` alone is not a progress signal; but
    summing ``V_t`` across frames gives an "expected net forward steps"
    curve that climbs for a successful demo and plateaus/drops for a
    stalled one. Matches the previous binary convention ``Σ (2·P - 1)``
    exactly: for ``num_bins == 2`` the new signed-progress score equals
    ``2 · P(progress) - 1``.

    Accepts either a 1D sequence (aggregated signed-progress) or a 2D
    array of shape ``[T, E]`` (per-member). Cumulative sum is along axis 0.
    """
    p = np.asarray(signed_progress, dtype=np.float64)
    return np.cumsum(p, axis=0)


def _member_colors(num_members: int) -> list:
    """Distinct color per ensemble member — used consistently across panels."""
    if num_members <= 10:
        cmap = matplotlib.colormaps.get_cmap("tab10")
        return [cmap(i) for i in range(num_members)]
    cmap = matplotlib.colormaps.get_cmap("viridis")
    return [cmap(i / max(1, num_members - 1)) for i in range(num_members)]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def _to_scalar(x):
    if hasattr(x, "item"):
        return x.item()
    return x


def _compute_value_axis_limits(
    signed_progress: np.ndarray,
    member_signed_progress: np.ndarray,
) -> tuple[float, float]:
    """Choose y-limits that do not clip UWO or per-member traces."""
    arrays = [signed_progress.reshape(-1)]
    if member_signed_progress.size > 0:
        arrays.append(member_signed_progress.reshape(-1))

    lower = min(float(arr.min()) for arr in arrays)
    upper = max(float(arr.max()) for arr in arrays)
    # Native range for signed progress is [-1, 1]; add a small margin.
    if lower >= -1.0 and upper <= 1.0:
        return -1.05, 1.05

    span = max(upper - lower, 1e-6)
    pad = max(0.05, 0.08 * span)
    return lower - pad, upper + pad


def _collect_episode_frames(
    lerobot_ds,
    episode_index: int,
    ep_starts: list[int],
    ep_ends: list[int],
    tasks: dict,
    image_keys: list[str],
    progress_by_frame: dict[int, dict[str, object]],
    num_bins: int = 2,
) -> dict:
    """Gather images + signed-progress per frame for one episode.

    When ``num_bins > 2`` the per-frame bundle from
    :func:`_score_episode_pairs` also carries bin-level distributions;
    this function threads those fields into the returned dict. Terminal
    frames (no valid pair) default to the previous frame's bundle,
    same as the scalar defaulting rule.
    """
    start = int(ep_starts[episode_index])
    end = int(ep_ends[episode_index])

    data = {
        "frames": [],
        "images": {k: [] for k in image_keys},
        "signed_progress": [],
        "signed_progress_mean": [],
        "signed_progress_min": [],
        "signed_progress_variance": [],
        "member_signed_progress": [],
        "task": "",
        "episode_index": episode_index,
        # num_bins lets the plot path branch consistently; 2 = binary
        # (no additional fields); > 2 = multi-bin (new fields below).
        "num_bins": int(num_bins),
    }
    is_multi_bin = num_bins > 2
    if is_multi_bin:
        for key in (
            "aggregated_probs",
            "member_probs",
            "expected_stride_normalized",
            "member_expected_stride_normalized",
            "entropy_aggregated",
            "entropy_member_mean",
        ):
            data[key] = []

    default_member_count = 1
    default_agg_probs: list[float] = []
    default_member_probs: list[list[float]] = []
    default_es_member: list[float] = []
    if progress_by_frame:
        first_score = next(iter(progress_by_frame.values()))
        default_member_count = max(1, len(first_score.get("member_values", [])))
        if is_multi_bin:
            # Use a uniform distribution + zero E[s] + max entropy as the
            # "I have no idea" default so the initial / terminal-frame
            # filler doesn't spoof a confident prediction.
            default_agg_probs = [1.0 / num_bins] * num_bins
            default_member_probs = [
                [1.0 / num_bins] * num_bins for _ in range(default_member_count)
            ]
            default_es_member = [0.0] * default_member_count

    for idx in tqdm(range(start, end), desc=f"Episode {episode_index}", leave=False):
        sample = lerobot_ds[idx]
        frame_idx = int(_to_scalar(sample["frame_index"]))
        data["frames"].append(frame_idx)

        for key in image_keys:
            if key in sample:
                img = _to_numpy(sample[key])
                if img.ndim == 4:
                    img = img[0]
                if img.dtype in (np.float32, np.float64):
                    img = (img * 255).astype(np.uint8)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                data["images"][key].append(img)

        # Last frame of each episode has no valid pair; default to the previous
        # value (or a "neutral" 0.0 signed-progress default if this is the only
        # frame — matches the uniform bin-distribution default used for
        # multi-bin, which also integrates to signed-progress = 0).
        if frame_idx in progress_by_frame:
            score_bundle = progress_by_frame[frame_idx]
            data["signed_progress"].append(float(score_bundle["value"]))
            data["signed_progress_mean"].append(float(score_bundle["value_mean"]))
            data["signed_progress_min"].append(float(score_bundle["value_min"]))
            data["signed_progress_variance"].append(
                float(score_bundle["value_variance"])
            )
            data["member_signed_progress"].append(
                list(score_bundle.get("member_values", [float(score_bundle["value"])]))
            )
            if is_multi_bin:
                data["aggregated_probs"].append(list(score_bundle["aggregated_probs"]))
                data["member_probs"].append(
                    [list(m) for m in score_bundle["member_probs"]]
                )
                data["expected_stride_normalized"].append(
                    float(score_bundle["expected_stride_normalized"])
                )
                data["member_expected_stride_normalized"].append(
                    list(score_bundle["member_expected_stride_normalized"])
                )
                data["entropy_aggregated"].append(
                    float(score_bundle["entropy_aggregated"])
                )
                data["entropy_member_mean"].append(
                    float(score_bundle["entropy_member_mean"])
                )
        elif data["signed_progress"]:
            data["signed_progress"].append(data["signed_progress"][-1])
            data["signed_progress_mean"].append(data["signed_progress_mean"][-1])
            data["signed_progress_min"].append(data["signed_progress_min"][-1])
            data["signed_progress_variance"].append(
                data["signed_progress_variance"][-1]
            )
            data["member_signed_progress"].append(
                data["member_signed_progress"][-1].copy()
            )
            if is_multi_bin:
                data["aggregated_probs"].append(data["aggregated_probs"][-1].copy())
                data["member_probs"].append(
                    [m.copy() for m in data["member_probs"][-1]]
                )
                data["expected_stride_normalized"].append(
                    data["expected_stride_normalized"][-1]
                )
                data["member_expected_stride_normalized"].append(
                    data["member_expected_stride_normalized"][-1].copy()
                )
                data["entropy_aggregated"].append(data["entropy_aggregated"][-1])
                data["entropy_member_mean"].append(data["entropy_member_mean"][-1])
        else:
            data["signed_progress"].append(0.0)
            data["signed_progress_mean"].append(0.0)
            data["signed_progress_min"].append(0.0)
            data["signed_progress_variance"].append(0.0)
            data["member_signed_progress"].append([0.0] * default_member_count)
            if is_multi_bin:
                data["aggregated_probs"].append(list(default_agg_probs))
                data["member_probs"].append([list(m) for m in default_member_probs])
                data["expected_stride_normalized"].append(0.0)
                data["member_expected_stride_normalized"].append(
                    list(default_es_member)
                )
                data["entropy_aggregated"].append(float(np.log(num_bins)))
                data["entropy_member_mean"].append(float(np.log(num_bins)))

        if not data["task"]:
            if "task" in sample:
                data["task"] = str(_to_scalar(sample["task"]))
            elif "task_index" in sample and tasks:
                t_idx = int(_to_scalar(sample["task_index"]))
                data["task"] = tasks.get(t_idx, f"Task {t_idx}")

    return data


def _create_episode_summary_plot(
    ep_data: dict,
    output_path: Path,
    decision_threshold: float = 0.0,
    stride_k: int = 0,
    inference_mode: str = "mo",
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """Sampled frames + per-pair V_t + cumulative signed progress.

    ``V_t`` is the model's signed-progress score for the stride-k pair
    at frame ``t`` — positive = looks forward, negative = looks
    reversed, zero = undecided (see
    :meth:`SteamCriticModel._predicted_signed_value`).

    When ``ep_data["num_bins"] > 2`` the figure gains three extra
    panels below the existing curves: a bin-distribution heatmap, an
    ``E[signed stride] / K`` curve with ensemble band, and a per-frame
    entropy curve. Binary-mode episodes render the exact same layout
    as before.

    """
    frames = ep_data["frames"]
    n_frames = len(frames)
    if n_frames == 0:
        return

    image_keys = [k for k, v in ep_data["images"].items() if len(v) > 0]
    n_cameras = len(image_keys)
    if n_cameras == 0:
        return

    sample_indices = sorted(
        {
            i
            for i in [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
            if i < n_frames
        }
    )

    member_signed_progress = np.asarray(
        ep_data["member_signed_progress"], dtype=np.float64
    )
    has_ensemble = (
        member_signed_progress.ndim == 2 and member_signed_progress.shape[1] > 1
    )
    num_bins = int(ep_data.get("num_bins", 2))
    is_multi_bin = num_bins > 2

    # Binary layout: 2 or 3 curve rows (signed progress, optional variance, cumulative).
    # Multi-bin adds 3 rows (heatmap, E[stride]/K, entropy).
    curve_rows = (3 if has_ensemble else 2) + (3 if is_multi_bin else 0)

    # Per-row heights. Multi-bin heatmap gets 2× height; all others are 1.
    non_heatmap_rows = curve_rows - (3 if is_multi_bin else 0)
    height_ratios = [1] * n_cameras + [1] * non_heatmap_rows
    if is_multi_bin:
        # Heatmap + E[stride] + entropy: [2, 1, 1]
        height_ratios = [1] * n_cameras + [1] * non_heatmap_rows + [2, 1, 1]
    if len(height_ratios) != n_cameras + curve_rows:
        raise RuntimeError(
            f"Internal: height_ratios length {len(height_ratios)} != "
            f"n_cameras + curve_rows = {n_cameras + curve_rows}"
        )

    if figsize[1] < 10 and is_multi_bin:
        # Scale figure height with extra rows so panels don't look squished.
        figsize = (figsize[0], max(figsize[1] + 4, 12))

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        n_cameras + curve_rows,
        len(sample_indices),
        figure=fig,
        height_ratios=height_ratios,
    )

    signed_progress = ep_data["signed_progress"]
    p_arr = np.asarray(signed_progress, dtype=np.float64)
    value_variance = np.asarray(ep_data["signed_progress_variance"], dtype=np.float64)
    cum_signed_progress = _cumulative_signed_progress(signed_progress)
    num_members = (
        int(member_signed_progress.shape[1]) if member_signed_progress.ndim == 2 else 1
    )
    member_cum_signed_progress = (
        _cumulative_signed_progress(member_signed_progress) if has_ensemble else None
    )
    member_colors = _member_colors(num_members) if has_ensemble else []
    value_ymin, value_ymax = _compute_value_axis_limits(p_arr, member_signed_progress)

    for cam_idx, key in enumerate(image_keys):
        cam_name = key.replace("observation.images.", "").replace("_", " ").title()
        for col_idx, frame_idx in enumerate(sample_indices):
            ax = fig.add_subplot(gs[cam_idx, col_idx])
            ax.imshow(ep_data["images"][key][frame_idx])
            ax.axis("off")
            above = signed_progress[frame_idx] >= decision_threshold
            if above:
                rect = mpatches.FancyBboxPatch(
                    (0, 0),
                    1,
                    1,
                    transform=ax.transAxes,
                    boxstyle="round,pad=0",
                    linewidth=4,
                    edgecolor="lime",
                    facecolor="none",
                    zorder=10,
                )
                ax.add_patch(rect)
            if cam_idx == 0:
                title_str = f"t={frames[frame_idx]}"
                title_str += f" (V={signed_progress[frame_idx]:+.2f})"
                if above:
                    title_str = "* " + title_str
                ax.set_title(
                    title_str,
                    fontsize=9,
                    color="green" if above else "black",
                    fontweight="bold" if above else "normal",
                )
            if col_idx == 0:
                ax.text(
                    -0.1,
                    0.5,
                    cam_name,
                    transform=ax.transAxes,
                    fontsize=9,
                    va="center",
                    ha="right",
                    rotation=90,
                )

    # Row 1: per-pair forward confidence — each ensemble member gets its
    # own color, plus the inference_mode-aggregated ("total") score as a
    # bold black overlay so it's clearly distinguishable.
    ax_p = fig.add_subplot(gs[n_cameras, :])
    if has_ensemble:
        for member_idx in range(num_members):
            ax_p.plot(
                frames,
                member_signed_progress[:, member_idx],
                color=member_colors[member_idx],
                alpha=0.75,
                linewidth=1.0,
                label=f"Member {member_idx}",
            )
    aggregated_label = (
        f"Aggregated ({inference_mode.upper()}, stride k={stride_k})"
        if has_ensemble
        else f"{inference_mode.upper()} score (stride k={stride_k})"
    )
    ax_p.plot(
        frames,
        signed_progress,
        color="black",
        linewidth=2.2,
        label=aggregated_label,
    )
    ax_p.axhline(
        y=decision_threshold,
        color="orange",
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
        label=f"Threshold={decision_threshold:+.2f}",
    )
    # Neutral-signed-progress reference at 0: scores above it look forward,
    # below it look reversed.
    ax_p.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    above_mask = p_arr >= decision_threshold
    ax_p.fill_between(
        frames,
        p_arr,
        decision_threshold,
        where=above_mask,
        alpha=0.18,
        color="lime",
        label="Above threshold",
    )
    ax_p.set_ylabel(f"{inference_mode.upper()} score")
    ax_p.set_ylim(value_ymin, value_ymax)
    ax_p.set_xlim(frames[0], frames[-1])
    ax_p.legend(loc="upper left", fontsize=7, ncol=2 if has_ensemble else 1)
    ax_p.grid(True, alpha=0.3)
    ax_p.tick_params(labelbottom=False)

    row_cursor = n_cameras + 1

    cumulative_row = row_cursor
    if has_ensemble:
        ax_v = fig.add_subplot(gs[row_cursor, :], sharex=ax_p)
        ax_v.plot(
            frames,
            value_variance,
            color="tab:red",
            linewidth=1.5,
            label="Member variance",
        )
        ax_v.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_v.set_ylabel("Variance")
        ax_v.set_xlim(frames[0], frames[-1])
        ax_v.legend(loc="upper left", fontsize=8)
        ax_v.grid(True, alpha=0.3)
        ax_v.tick_params(labelbottom=False)
        cumulative_row = row_cursor + 1

    # Final row: cumulative signed progress Σ V_t. Members share colors
    # with the top panel so each trace is easy to follow across both
    # panels; the aggregated (inference_mode) curve is bold purple.
    ax_c = fig.add_subplot(gs[cumulative_row, :], sharex=ax_p)
    if has_ensemble and member_cum_signed_progress is not None:
        for member_idx in range(num_members):
            ax_c.plot(
                frames,
                member_cum_signed_progress[:, member_idx],
                color=member_colors[member_idx],
                alpha=0.55,
                linewidth=0.9,
                label=f"Member {member_idx}",
            )
    ax_c.plot(
        frames,
        cum_signed_progress,
        color="tab:purple",
        linewidth=2.0,
        label=f"Aggregated ({inference_mode.upper()}) Σ V" if has_ensemble else "Σ V",
    )
    ax_c.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    # Ideal "always forward" line for reference (slope = +1).
    ideal = np.arange(len(frames), dtype=np.float64)
    ax_c.plot(
        frames,
        ideal,
        color="lightgray",
        linestyle=":",
        linewidth=1.0,
        label="Ideal always-forward",
    )
    ax_c.set_xlabel("Frame" if not is_multi_bin else "")
    ax_c.set_ylabel("Cumulative signed\nprogress")
    ax_c.set_xlim(frames[0], frames[-1])
    ax_c.legend(loc="upper left", fontsize=7, ncol=2 if has_ensemble else 1)
    ax_c.grid(True, alpha=0.3)
    if is_multi_bin:
        ax_c.tick_params(labelbottom=False)

    # -------------------------------------------------------------------
    # Multi-bin extra panels — heatmap of bin distribution, E[stride]/K,
    # entropy. Each is drawn below the cumulative-progress panel.
    # -------------------------------------------------------------------
    if is_multi_bin:
        heatmap_row = cumulative_row + 1
        expected_stride_row = heatmap_row + 1
        entropy_row = expected_stride_row + 1

        agg_probs_arr = np.asarray(ep_data["aggregated_probs"], dtype=np.float64)
        expected_stride = np.asarray(
            ep_data["expected_stride_normalized"], dtype=np.float64
        )
        member_expected_stride = np.asarray(
            ep_data["member_expected_stride_normalized"], dtype=np.float64
        )
        entropy_agg_arr = np.asarray(ep_data["entropy_aggregated"], dtype=np.float64)
        entropy_member_arr = np.asarray(
            ep_data["entropy_member_mean"], dtype=np.float64
        )
        max_entropy = float(np.log(num_bins))
        tick_labels = _signed_stride_bin_tick_labels(stride_k, num_bins)

        # --- Heatmap: bin × frame distribution ---
        ax_hm = fig.add_subplot(gs[heatmap_row, :], sharex=ax_p)
        # Transpose to [num_bins, T]. origin='lower' puts bin 0 (most
        # regressive) at the bottom so up = more progressive.
        im = ax_hm.imshow(
            agg_probs_arr.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            extent=[frames[0], frames[-1], -0.5, num_bins - 0.5],
            vmin=0.0,
            vmax=1.0,
        )
        ax_hm.set_yticks(np.arange(num_bins))
        ax_hm.set_yticklabels(tick_labels, fontsize=7)
        ax_hm.axhline(
            y=num_bins / 2 - 0.5,
            color="white",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )  # sign split between regress / progress halves
        ax_hm.set_ylabel("Signed stride bin")
        ax_hm.set_title(
            f"Aggregated bin distribution ({inference_mode.upper()})",
            fontsize=9,
        )
        ax_hm.tick_params(labelbottom=False)
        fig.colorbar(im, ax=ax_hm, fraction=0.02, pad=0.01, label="P(bin)")

        # --- E[signed stride] / K curve with ensemble band ---
        ax_es = fig.add_subplot(gs[expected_stride_row, :], sharex=ax_p)
        if has_ensemble and member_expected_stride.ndim == 2:
            for member_idx in range(num_members):
                ax_es.plot(
                    frames,
                    member_expected_stride[:, member_idx],
                    color=member_colors[member_idx],
                    alpha=0.55,
                    linewidth=0.9,
                )
            mem_mean = member_expected_stride.mean(axis=1)
            mem_std = member_expected_stride.std(axis=1, ddof=0)
            ax_es.fill_between(
                frames,
                mem_mean - mem_std,
                mem_mean + mem_std,
                color="gray",
                alpha=0.18,
                label="Member ±σ",
            )
        ax_es.plot(
            frames,
            expected_stride,
            color="black",
            linewidth=2.0,
            label=f"Aggregated ({inference_mode.upper()}) E[s]/K",
        )
        ax_es.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_es.set_ylim(-1.05, 1.05)
        ax_es.set_ylabel("E[signed stride] / K")
        ax_es.set_xlim(frames[0], frames[-1])
        ax_es.legend(loc="upper left", fontsize=7, ncol=2 if has_ensemble else 1)
        ax_es.grid(True, alpha=0.3)
        ax_es.tick_params(labelbottom=False)

        # --- Entropy curve (aggregated + mean-of-member) ---
        ax_h = fig.add_subplot(gs[entropy_row, :], sharex=ax_p)
        ax_h.plot(
            frames,
            entropy_agg_arr,
            color="tab:blue",
            linewidth=1.6,
            label="H(aggregated)",
        )
        ax_h.plot(
            frames,
            entropy_member_arr,
            color="tab:orange",
            linewidth=1.2,
            alpha=0.8,
            label="mean_m H(member_m)",
        )
        ax_h.axhline(
            y=max_entropy,
            color="lightgray",
            linestyle=":",
            linewidth=1.0,
            label=f"Max = ln({num_bins}) ≈ {max_entropy:.2f}",
        )
        ax_h.set_xlabel("Frame")
        ax_h.set_ylabel("Entropy (nats)")
        ax_h.set_xlim(frames[0], frames[-1])
        ax_h.set_ylim(-0.02, max_entropy * 1.05)
        ax_h.legend(loc="upper left", fontsize=7)
        ax_h.grid(True, alpha=0.3)

    task_text = ep_data.get("task", "")[:80]
    final_cum = float(cum_signed_progress[-1]) if len(cum_signed_progress) else 0.0
    suptitle_bits = [
        f"Episode {ep_data['episode_index']} (k={stride_k}, mode={inference_mode.upper()}",
    ]
    if is_multi_bin:
        suptitle_bits.append(f", num_bins={num_bins}")
    suptitle_head = "".join(suptitle_bits) + f"): {task_text}"
    suptitle_body = (
        f"score mean={p_arr.mean():.3f}, "
        f"above-thresh frac={float(above_mask.mean()):.2f}  |  "
        f"variance mean={value_variance.mean():.4f}  |  "
        f"final cumulative progress={final_cum:.1f} / max {len(frames) - 1}"
    )
    if is_multi_bin:
        suptitle_body += (
            f"  |  E[s]/K mean={float(np.mean(expected_stride)):.3f}, "
            f"H_agg mean={float(np.mean(entropy_agg_arr)):.3f} nats"
        )
    fig.suptitle(f"{suptitle_head}\n{suptitle_body}", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _create_episode_video(
    ep_data: dict,
    output_path: Path,
    decision_threshold: float = 0.0,
    stride_k: int = 0,
    inference_mode: str = "mo",
    fps: int = 10,
    figsize: tuple[int, int] = (14, 9),
    dpi: int = 100,
) -> None:
    """Animated episode with camera views, V_t and cumulative signed-progress curves."""
    frames = ep_data["frames"]
    n_frames = len(frames)
    if n_frames == 0:
        return

    image_keys = [k for k, v in ep_data["images"].items() if len(v) > 0]
    n_cameras = len(image_keys)
    if n_cameras == 0:
        logger.warning("No images to render for episode %s", ep_data["episode_index"])
        return

    member_signed_progress = np.asarray(
        ep_data["member_signed_progress"], dtype=np.float64
    )
    has_ensemble = (
        member_signed_progress.ndim == 2 and member_signed_progress.shape[1] > 1
    )
    num_bins = int(ep_data.get("num_bins", 2))
    is_multi_bin = num_bins > 2

    # Base rows: 1 (cameras) + (signed progress) + (variance?) + (cumulative)
    # Multi-bin adds 3 more: heatmap, E[s]/K, entropy.
    grid_rows = (4 if has_ensemble else 3) + (3 if is_multi_bin else 0)

    # Height ratios. Camera row is 2×; heatmap is 2×; all other rows are 1×.
    non_heatmap_rows = grid_rows - (3 if is_multi_bin else 0)
    base_ratios = [2] + [1] * (non_heatmap_rows - 1)
    if is_multi_bin:
        height_ratios = base_ratios + [2, 1, 1]
    else:
        height_ratios = base_ratios
    if len(height_ratios) != grid_rows:
        raise RuntimeError(
            f"Internal: video height_ratios length {len(height_ratios)} != "
            f"grid_rows = {grid_rows}"
        )

    if is_multi_bin and figsize[1] < 12:
        figsize = (figsize[0], figsize[1] + 4)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(
        grid_rows,
        max(n_cameras, 1),
        figure=fig,
        height_ratios=height_ratios,
    )

    camera_axes = []
    for i, key in enumerate(image_keys):
        ax = fig.add_subplot(gs[0, i])
        display_name = key.replace("observation.images.", "").replace("_", " ").title()
        ax.set_title(display_name, fontsize=10)
        ax.axis("off")
        camera_axes.append(ax)

    signed_progress = ep_data["signed_progress"]
    value_variance = np.asarray(ep_data["signed_progress_variance"], dtype=np.float64)
    cum_signed_progress = _cumulative_signed_progress(signed_progress)
    num_members = (
        int(member_signed_progress.shape[1]) if member_signed_progress.ndim == 2 else 1
    )
    member_colors = _member_colors(num_members) if has_ensemble else []
    value_ymin, value_ymax = _compute_value_axis_limits(
        np.asarray(signed_progress, dtype=np.float64),
        member_signed_progress,
    )

    ax_p = fig.add_subplot(gs[1, :])
    if has_ensemble:
        for member_idx in range(num_members):
            ax_p.plot(
                frames,
                member_signed_progress[:, member_idx],
                color=member_colors[member_idx],
                alpha=0.7,
                linewidth=0.9,
                label=f"Member {member_idx}",
            )
    aggregated_label = (
        f"Aggregated ({inference_mode.upper()})"
        if has_ensemble
        else f"{inference_mode.upper()} score"
    )
    ax_p.plot(
        frames,
        signed_progress,
        color="black",
        linewidth=1.8,
        label=aggregated_label,
    )
    ax_p.axhline(
        y=decision_threshold,
        color="orange",
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
        label=f"Threshold={decision_threshold:+.2f}",
    )
    ax_p.set_title(
        f"{inference_mode.upper()} score (stride k={stride_k})",
        fontsize=10,
    )
    ax_p.set_ylabel("Score")
    ax_p.set_xlim(frames[0], frames[-1])
    ax_p.set_ylim(value_ymin, value_ymax)
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc="upper left", fontsize=7, ncol=2 if has_ensemble else 1)
    ax_p.tick_params(labelbottom=False)

    row_cursor = 2

    variance_row = row_cursor if has_ensemble else None
    cumulative_row = row_cursor + (1 if has_ensemble else 0)
    ax_v = None
    if has_ensemble:
        ax_v = fig.add_subplot(gs[variance_row, :], sharex=ax_p)
        ax_v.plot(
            frames,
            value_variance,
            color="tab:red",
            linewidth=1.2,
            label="Member variance",
        )
        ax_v.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_v.set_title("Member variance", fontsize=10)
        ax_v.set_ylabel("Var")
        ax_v.set_xlim(frames[0], frames[-1])
        ax_v.grid(True, alpha=0.3)
        ax_v.legend(loc="upper left", fontsize=8)
        ax_v.tick_params(labelbottom=False)

    ax_c = fig.add_subplot(gs[cumulative_row, :], sharex=ax_p)
    if has_ensemble:
        member_cum_signed_progress = _cumulative_signed_progress(member_signed_progress)
        for member_idx in range(num_members):
            ax_c.plot(
                frames,
                member_cum_signed_progress[:, member_idx],
                color=member_colors[member_idx],
                alpha=0.5,
                linewidth=0.8,
                label=f"Member {member_idx}",
            )
    ax_c.plot(
        frames,
        cum_signed_progress,
        color="tab:purple",
        linewidth=1.4,
        label=f"Aggregated ({inference_mode.upper()}) Σ V" if has_ensemble else "Σ V",
    )
    ideal = np.arange(len(frames), dtype=np.float64)
    ax_c.plot(
        frames,
        ideal,
        color="lightgray",
        linestyle=":",
        linewidth=1.0,
        label="Ideal always-forward",
    )
    ax_c.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_c.set_title("Cumulative signed progress", fontsize=10)
    ax_c.set_xlabel("Frame" if not is_multi_bin else "")
    ax_c.set_ylabel("Σ V")
    ax_c.set_xlim(frames[0], frames[-1])
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="upper left", fontsize=7, ncol=2 if has_ensemble else 1)
    if is_multi_bin:
        ax_c.tick_params(labelbottom=False)

    # Multi-bin extra panels — heatmap, E[s]/K, entropy.
    ax_hm_v = None
    ax_es_v = None
    ax_h_v = None
    hm_cursor = es_marker = h_marker = None
    expected_stride_arr_v = np.empty(0)
    member_expected_stride_arr_v = np.empty(0)
    entropy_agg_arr_v = np.empty(0)
    entropy_member_arr_v = np.empty(0)
    if is_multi_bin:
        agg_probs_arr = np.asarray(ep_data["aggregated_probs"], dtype=np.float64)
        expected_stride_arr_v = np.asarray(
            ep_data["expected_stride_normalized"], dtype=np.float64
        )
        member_expected_stride_arr_v = np.asarray(
            ep_data["member_expected_stride_normalized"], dtype=np.float64
        )
        entropy_agg_arr_v = np.asarray(ep_data["entropy_aggregated"], dtype=np.float64)
        entropy_member_arr_v = np.asarray(
            ep_data["entropy_member_mean"], dtype=np.float64
        )
        max_entropy = float(np.log(num_bins))
        tick_labels = _signed_stride_bin_tick_labels(stride_k, num_bins)

        hm_row = cumulative_row + 1
        ax_hm_v = fig.add_subplot(gs[hm_row, :], sharex=ax_p)
        im = ax_hm_v.imshow(
            agg_probs_arr.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            extent=[frames[0], frames[-1], -0.5, num_bins - 0.5],
            vmin=0.0,
            vmax=1.0,
        )
        ax_hm_v.set_yticks(np.arange(num_bins))
        ax_hm_v.set_yticklabels(tick_labels, fontsize=7)
        ax_hm_v.axhline(
            y=num_bins / 2 - 0.5,
            color="white",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
        ax_hm_v.set_ylabel("Signed stride bin")
        ax_hm_v.set_title(
            f"Aggregated bin distribution ({inference_mode.upper()})", fontsize=10
        )
        ax_hm_v.tick_params(labelbottom=False)
        fig.colorbar(im, ax=ax_hm_v, fraction=0.02, pad=0.01, label="P(bin)")
        # Vertical cursor on the heatmap marking the current frame.
        (hm_cursor,) = ax_hm_v.plot(
            [frames[0], frames[0]],
            [-0.5, num_bins - 0.5],
            color="red",
            linewidth=1.5,
            alpha=0.9,
        )

        ax_es_v = fig.add_subplot(gs[hm_row + 1, :], sharex=ax_p)
        if has_ensemble and member_expected_stride_arr_v.ndim == 2:
            for member_idx in range(num_members):
                ax_es_v.plot(
                    frames,
                    member_expected_stride_arr_v[:, member_idx],
                    color=member_colors[member_idx],
                    alpha=0.55,
                    linewidth=0.9,
                )
            mem_mean = member_expected_stride_arr_v.mean(axis=1)
            mem_std = member_expected_stride_arr_v.std(axis=1, ddof=0)
            ax_es_v.fill_between(
                frames,
                mem_mean - mem_std,
                mem_mean + mem_std,
                color="gray",
                alpha=0.18,
            )
        ax_es_v.plot(
            frames,
            expected_stride_arr_v,
            color="black",
            linewidth=1.6,
            label=f"Aggregated ({inference_mode.upper()}) E[s]/K",
        )
        ax_es_v.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_es_v.set_title("E[signed stride] / K", fontsize=10)
        ax_es_v.set_ylabel("E[s]/K")
        ax_es_v.set_ylim(-1.05, 1.05)
        ax_es_v.set_xlim(frames[0], frames[-1])
        ax_es_v.grid(True, alpha=0.3)
        ax_es_v.legend(loc="upper left", fontsize=7)
        ax_es_v.tick_params(labelbottom=False)
        (es_marker,) = ax_es_v.plot(
            [frames[0]],
            [expected_stride_arr_v[0]],
            "o",
            color="black",
            markersize=7,
        )

        ax_h_v = fig.add_subplot(gs[hm_row + 2, :], sharex=ax_p)
        ax_h_v.plot(
            frames,
            entropy_agg_arr_v,
            color="tab:blue",
            linewidth=1.4,
            label="H(aggregated)",
        )
        ax_h_v.plot(
            frames,
            entropy_member_arr_v,
            color="tab:orange",
            linewidth=1.0,
            alpha=0.8,
            label="mean_m H(member_m)",
        )
        ax_h_v.axhline(
            y=max_entropy,
            color="lightgray",
            linestyle=":",
            linewidth=1.0,
            label=f"Max = ln({num_bins}) ≈ {max_entropy:.2f}",
        )
        ax_h_v.set_xlabel("Frame")
        ax_h_v.set_ylabel("Entropy (nats)")
        ax_h_v.set_xlim(frames[0], frames[-1])
        ax_h_v.set_ylim(-0.02, max_entropy * 1.05)
        ax_h_v.grid(True, alpha=0.3)
        ax_h_v.legend(loc="upper left", fontsize=7)
        (h_marker,) = ax_h_v.plot(
            [frames[0]],
            [entropy_agg_arr_v[0]],
            "o",
            color="tab:blue",
            markersize=7,
        )

    camera_ims = [
        ax.imshow(ep_data["images"][key][0]) for ax, key in zip(camera_axes, image_keys)
    ]

    border_patches = []
    overlay_patches = []
    for ax in camera_axes:
        rect = Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            linewidth=8,
            edgecolor="lime",
            facecolor="none",
            visible=False,
            zorder=10,
            clip_on=False,
        )
        ax.add_patch(rect)
        border_patches.append(rect)
        overlay = Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            linewidth=0,
            edgecolor="none",
            facecolor="lime",
            alpha=0.15,
            visible=False,
            zorder=9,
        )
        ax.add_patch(overlay)
        overlay_patches.append(overlay)

    # Marker tracks the aggregated ("total") signed-progress score.
    # Red-on-black edge so it stands out against member 0's tab:blue line
    # in ensemble renders.
    (marker_p,) = ax_p.plot(
        [frames[0]],
        [signed_progress[0]],
        "o",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=9,
        markeredgewidth=1.5,
    )
    marker_v = None
    if ax_v is not None:
        (marker_v,) = ax_v.plot(
            [frames[0]], [value_variance[0]], "o", color="tab:red", markersize=7
        )
    (marker_c,) = ax_c.plot(
        [frames[0]], [cum_signed_progress[0]], "o", color="tab:purple", markersize=8
    )

    task_text = ep_data.get("task", "")[:60]
    title_head = (
        f"Episode {ep_data['episode_index']} (k={stride_k}, "
        f"mode={inference_mode.upper()}"
        f"{f', num_bins={num_bins}' if is_multi_bin else ''}) - {task_text}"
    )
    title = fig.suptitle(
        f"{title_head}\n"
        f"Frame: {frames[0]}  V={signed_progress[0]:+.3f}  "
        f"Var={value_variance[0]:.4f}  Cum={cum_signed_progress[0]:.1f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.92 if is_multi_bin else 0.88)

    def update(frame_num):
        for im, key in zip(camera_ims, image_keys):
            im.set_array(ep_data["images"][key][frame_num])
        marker_p.set_data([frames[frame_num]], [signed_progress[frame_num]])
        if marker_v is not None:
            marker_v.set_data([frames[frame_num]], [value_variance[frame_num]])
        marker_c.set_data([frames[frame_num]], [cum_signed_progress[frame_num]])
        # Multi-bin cursors.
        if hm_cursor is not None:
            hm_cursor.set_data(
                [frames[frame_num], frames[frame_num]], [-0.5, num_bins - 0.5]
            )
        if es_marker is not None:
            es_marker.set_data([frames[frame_num]], [expected_stride_arr_v[frame_num]])
        if h_marker is not None:
            h_marker.set_data([frames[frame_num]], [entropy_agg_arr_v[frame_num]])

        above = signed_progress[frame_num] >= decision_threshold
        for rect in border_patches:
            rect.set_visible(above)
        for overlay in overlay_patches:
            overlay.set_visible(above)

        indicator = " [FORWARD]" if above else ""
        body_bits = [
            f"Frame: {frames[frame_num]}  V={signed_progress[frame_num]:+.3f}  "
            f"Var={value_variance[frame_num]:.4f}  "
            f"Cum={cum_signed_progress[frame_num]:.1f}"
        ]
        if is_multi_bin:
            body_bits.append(
                f"  E[s]/K={expected_stride_arr_v[frame_num]:+.3f}  "
                f"H={entropy_agg_arr_v[frame_num]:.2f}"
            )
        title.set_text(f"{title_head}\n{''.join(body_bits)}{indicator}")
        animated_items = camera_ims + [marker_p, marker_c]
        if marker_v is not None:
            animated_items.append(marker_v)
        if hm_cursor is not None:
            animated_items.append(hm_cursor)
        if es_marker is not None:
            animated_items.append(es_marker)
        if h_marker is not None:
            animated_items.append(h_marker)
        animated_items += border_patches + overlay_patches + [title]
        return animated_items

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=True)
    _ensure_ffmpeg_on_matplotlib()
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _opt_cast(value, cast):
    """Cast ``value`` to ``cast`` or pass through ``None``.

    Keeps user-left-null YAML fields as ``None`` so that
    :meth:`SteamCriticModel.from_checkpoint` falls back to the value
    persisted in the checkpoint's ``config.json`` instead of silently
    clamping it to the YAML default (e.g. ``ensemble_size=1`` used to
    overwrite a 4-member checkpoint → random weights).
    """
    return None if value is None else cast(value)


def _parse_from_checkpoint_kwargs(cfg: DictConfig) -> dict:
    """Map Hydra config onto SteamCriticModel.from_checkpoint kwargs.

    Leaves ``None``-valued YAML fields as ``None`` — that lets the
    ensemble / aggregation knobs ride on the checkpoint's saved config
    rather than being overwritten by the visualize-side defaults.
    """
    ckpt = cfg.visualize.checkpoint_dir
    if ckpt is None:
        raise ValueError("visualize.checkpoint_dir must be set")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    m = cfg.model
    return {
        "checkpoint_dir": ckpt,
        "tokenizer_path": m.get("tokenizer_path", None),
        "vision_repo_id": m.get("vision_repo_id", None),
        "language_repo_id": m.get("language_repo_id", None),
        "fusion_hidden_dim": _opt_cast(m.get("fusion_hidden_dim"), int),
        "dropout": _opt_cast(m.get("dropout"), float),
        "label_smoothing": _opt_cast(m.get("label_smoothing"), float),
        "num_frames_per_pair": _opt_cast(m.get("num_frames_per_pair"), int),
        "ensemble_size": _opt_cast(m.get("ensemble_size"), int),
        "inference_mode": _opt_cast(m.get("inference_mode"), str),
        "uwo_lambda": _opt_cast(m.get("uwo_lambda"), float),
        "include_state_in_prompt": _opt_cast(m.get("include_state_in_prompt"), bool),
        "max_state_dim": _opt_cast(m.get("max_state_dim"), int),
        "state_discretization_bins": _opt_cast(m.get("state_discretization_bins"), int),
        "max_token_len": _opt_cast(m.get("max_token_len"), int),
    }


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="visualize_steam",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    device = cfg.visualize.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    viz_cfg = cfg.visualize
    output_dir = Path(viz_cfg.get("output_dir", "./viz_binary_episodes"))
    output_dir.mkdir(parents=True, exist_ok=True)
    no_video = bool(viz_cfg.get("no_video", False))
    fps = int(viz_cfg.get("fps", 10))
    ds_index = int(viz_cfg.get("dataset_index", 0))
    batch_size = int(viz_cfg.get("batch_size", 32))
    num_workers = int(viz_cfg.get("num_workers", 4))
    decision_threshold = float(viz_cfg.get("decision_threshold", 0.0))

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Output directory: {output_dir}")

    # --- Load the binary value critic ---
    model = SteamCriticModel.from_checkpoint(
        **_parse_from_checkpoint_kwargs(cfg), device=device
    )
    inference_mode = str(getattr(model.config, "inference_mode", "mo"))

    # The processor created inside from_checkpoint uses the default
    # processor image_keys (base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb).
    # For datasets that use different camera keys (e.g. libero's
    # "image" / "wrist_image") we MUST align the processor's keys with
    # the dataset's, otherwise every real camera is silently dropped.
    camera_keys = tuple(cfg.data.get("camera_keys", ("image", "wrist_image")))
    model.processor.image_processor.image_keys = camera_keys
    logger.info(f"Processor image_keys set to: {camera_keys}")

    # --- Build the pair dataset + collator ---
    # Visualization does not need to know whether the dataset is sft or
    # rollout: we are just rendering trajectories, not training. Hardcode
    # dataset_type="sft" so PairDataset treats every episode as successful
    # and skips the is_success column scan — all episodes long enough to
    # form a stride-k pair become eligible for visualization.
    ds_entry = cfg.data.train_data_paths[ds_index]
    ds_path = ds_entry.dataset_path
    k = int(cfg.data.get("k", 4))
    state_key = str(cfg.data.get("state_key", "state"))
    # Prefer the CLI override, fall back to whatever the just-loaded model was
    # trained with. A null YAML here means "use the checkpoint's value".
    include_state_override = cfg.model.get("include_state_in_prompt", None)
    include_state = (
        bool(include_state_override)
        if include_state_override is not None
        else bool(getattr(model.config, "include_state_in_prompt", False))
    )
    max_state_dim_override = cfg.model.get("max_state_dim", None)
    max_state_dim = (
        int(max_state_dim_override)
        if max_state_dim_override is not None
        else int(getattr(model.config, "max_state_dim", 32))
    )

    logger.info(
        "Building PairDataset: path=%s  stride_k=%d",
        ds_path,
        k,
    )
    logger.info(
        "  Every frame t will be scored against frame (t+%d, clamped to T-1). "
        "This stride MUST match the value-SFT training config; the model "
        "generalizes poorly to other strides.",
        k,
    )
    pair_ds = PairDataset(
        dataset_path=str(ds_path),
        camera_keys=camera_keys,
        k=k,
        include_state=include_state,
        state_max_dim=max_state_dim,
        state_key=state_key,
        dataset_type="sft",
        only_success=True,
    )

    # Same fall-through rule: null in YAML → use the checkpoint's value.
    max_token_len_override = cfg.model.get("max_token_len", None)
    max_length = (
        int(max_token_len_override)
        if max_token_len_override is not None
        else int(getattr(model.config, "max_token_len", 200))
    )
    collator = BinaryPairDataCollator(
        processor=model.processor,
        max_length=max_length,
        train=False,
    )

    lerobot_ds = pair_ds.source.base
    ep_starts = pair_ds.source._ep_starts
    ep_ends = pair_ds.source._ep_ends
    total_episodes = len(ep_starts)
    tasks = pair_ds.source._tasks

    # Restrict episode selection to eligible episodes only — ineligible ones
    # can't form any pair and we'd immediately raise on lookup.
    eligible = pair_ds.eligible_episodes
    logger.info(
        f"Dataset: {total_episodes} episodes, {len(eligible)} eligible for pairs."
    )

    # Filter the random/explicit selection down to eligible episodes.
    raw_selection = _select_episodes(total_episodes, viz_cfg)
    eligible_set = set(eligible)
    episode_indices = [e for e in raw_selection if e in eligible_set]
    skipped = [e for e in raw_selection if e not in eligible_set]
    if skipped:
        logger.warning(
            f"Skipping ineligible episodes (too short / filtered): {skipped[:10]}"
            f"{'...' if len(skipped) > 10 else ''}"
        )
    if not episode_indices:
        raise RuntimeError(
            "No eligible episodes selected — loosen --num-episodes or check "
            "min_episode_length / only_success."
        )
    preview = episode_indices[:20]
    logger.info(
        f"Rendering {len(episode_indices)}/{len(eligible)} eligible episodes: "
        f"{preview}{'...' if len(episode_indices) > 20 else ''}"
    )

    # --- Detect image keys for rendering (from the raw LeRobot sample, not
    # the pair sample, which has a different schema). ---
    image_keys = detect_image_keys(lerobot_ds[int(ep_starts[episode_indices[0]])])
    logger.info(f"Image keys for rendering: {image_keys}")
    if not image_keys:
        logger.warning(
            "No image keys detected — videos/plots will be empty. "
            "Check the dataset schema."
        )

    # Multi-bin detection: the critic's config owns num_bins, the CLI /
    # data config owns K. Branch the viz pipeline on num_bins > 2.
    num_bins = int(getattr(model.config, "num_bins", 2))
    if num_bins > 2:
        logger.info(
            "Multi-bin mode active: num_bins=%d, K=%d → rendering bin-distribution "
            "heatmap, E[s]/K curve, and entropy curve on top of the usual panels.",
            num_bins,
            k,
        )

    # --- Run inference per episode and render ---
    model.eval()
    for ep in tqdm(episode_indices, desc="Rendering"):
        progress_by_frame = _score_episode_pairs(
            model=model,
            pair_ds=pair_ds,
            collator=collator,
            episode_index=ep,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            num_bins=num_bins,
            stride_k=k,
        )
        if not progress_by_frame:
            logger.warning(f"No pairs scored for episode {ep}; skipping.")
            continue

        ep_data = _collect_episode_frames(
            lerobot_ds=lerobot_ds,
            episode_index=ep,
            ep_starts=ep_starts,
            ep_ends=ep_ends,
            tasks=tasks,
            image_keys=image_keys,
            progress_by_frame=progress_by_frame,
            num_bins=num_bins,
        )

        summary_path = output_dir / f"episode_{ep:04d}_summary.png"
        _create_episode_summary_plot(
            ep_data,
            summary_path,
            decision_threshold=decision_threshold,
            stride_k=k,
            inference_mode=inference_mode,
        )
        logger.info(f"  Wrote {summary_path.name}")

        if not no_video:
            video_path = output_dir / f"episode_{ep:04d}.mp4"
            _create_episode_video(
                ep_data,
                video_path,
                decision_threshold=decision_threshold,
                stride_k=k,
                inference_mode=inference_mode,
                fps=fps,
            )
            logger.info(f"  Wrote {video_path.name}")

    # Free model before returning
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
