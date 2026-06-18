STEAM: Ensemble Advantage Modeling for Offline Policy Optimization
==================================================================

Run the **STEAM** pipeline in RLinf. STEAM is an offline policy-optimization
recipe that scores existing data with a **pair-classification progress critic**
and a **deep ensemble**, turning the conservative worst-of-N ensemble estimate
into per-frame advantage labels. Those labels then drive the same
**Classifier-Free Guidance (CFG) training** used by :doc:`RECAP <recap>`.

Like RECAP, STEAM needs no online environment interaction, so it suits
real-robot settings where large-scale online sampling is impractical. The
difference is the value signal: instead of regressing discounted returns, STEAM
learns a **temporal-progress** critic from frame pairs and aggregates an
ensemble of critics to suppress the advantage over-estimation a single predictor
would assign to out-of-distribution rollouts.

Overview
--------

Improve a policy offline (no new rollouts) by scoring existing data with an
ensemble progress critic and steering with classifier-free guidance.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      STEAM (worst-of-N ensemble)

   .. grid-item-card:: Models
      :text-align: center

      SigLIP2 + Gemma3 critic

   .. grid-item-card:: Environments / Data
      :text-align: center

      LeRobot datasets

   .. grid-item-card:: Training
      :text-align: center

      Offline · 2 + CFG stages

| **You'll do:** SFT an ensemble progress critic → compute ensemble advantages → CFG-train the policy → evaluate.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · SigLIP2 + Gemma3 + π₀.₅ checkpoints · LeRobot-format datasets (steps below).

Pipeline
--------

STEAM reuses RECAP's CFG training stage, so a STEAM run is two STEAM-specific
stages followed by RECAP Step 4:

.. code-block:: text

   ┌────────────────────────┐     ┌────────────────────────┐     ┌──────────────────────┐
   │  Step 1                │     │  Step 2                │     │  Step 3 (RECAP CFG)  │
   │  STEAM Value Model SFT │────▶│  Compute Ensemble      │────▶│  CFG Training        │
   │                        │     │  Advantages            │     │                      │
   │  Train an ensemble of  │     │  Worst-of-N ensemble   │     │  Train the policy    │
   │  pair-classification   │     │  signed score → bool   │     │  with classifier-    │
   │  progress critics      │     │  advantage labels      │     │  free guidance       │
   └────────────────────────┘     └────────────────────────┘     └──────────────────────┘

**Core Idea**

1. **Value Model SFT**: Train an ensemble of progress critics (SigLIP2 + Gemma3
   backbone + classifier head). Each member sees a frame pair
   :math:`(o_t, o_{t+k})` and classifies the signed frame stride into bins, so
   the head predicts *temporal progress* rather than a regressed return.

2. **Compute Ensemble Advantages**: For every frame, run all ensemble members on
   the pair :math:`(o_t, o_{t+k})` and aggregate with the **worst-of-N** rule
   (:math:`A = \min_m A_m`), yielding a signed score ``advantage_continuous``
   :math:`\in [-1, 1]`. Frames are then labelled positive/negative under a
   threshold or quantile rule.

3. **CFG Training**: Hand the advantage labels to RECAP's CFG stage — positive
   (high-advantage) samples are conditional inputs and negative samples are
   unconditional inputs, enabling classifier-free guidance for policy
   optimization.

How STEAM Works
---------------

**Pair-classification progress critic**

Each critic consumes a frame pair :math:`(o_t, o_{t+k})` (a SigLIP2 vision
encoder + Gemma3 language model fused per frame) and classifies the **signed
stride** :math:`s \in \{-K, \dots, -1, 1, \dots, K\}` into ``num_bins``
contiguous bins. Bins ``[0, num_bins/2)`` are regressive (negative stride) and
``[num_bins/2, num_bins)`` are progressive. The degenerate ``num_bins == 2``
case is binary progress classification; ``num_bins > 2`` gives a finer signed
stride distribution.

**Signed score**

The per-member prediction is a bin-weighted expectation normalized to
:math:`[-1, 1]`:

.. math::

   \text{signed\_score} = \frac{1}{K}\sum_b p_b \cdot \text{center}(b)

For ``num_bins == 2`` this reduces to :math:`2 \cdot P(\text{progress}) - 1`.

**Worst-of-N ensemble aggregation**

Members agree in-distribution but diverge on out-of-distribution rollouts. STEAM
reduces them with the conservative worst-of-N rule from the STEAM paper —
``predicted_values = min_m signed_score_m`` — so an over-confident single member
cannot inflate the advantage where the ensemble disagrees. Per-member mean / min
/ variance are recorded for diagnostics.

**Advantage labelling**

``advantage_continuous`` (the aggregated signed score) is turned into the boolean
``advantage`` under one of two ``label_mode`` rules:

- ``threshold``: ``advantage = advantage_continuous > positive_threshold`` for
  rollout frames (a signed-score threshold in :math:`[-1, 1]`); sft frames are
  always True (success demos by construction).
- ``quantile``: label the top ``rollout_quantile`` fraction of rollout frames
  True and, when ``expert_quantile`` is set, the top ``expert_quantile`` fraction
  of sft frames True — the two pools are scored independently.

Installation
------------

1. Clone RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

STEAM shares the OpenPI environment with RECAP.

**Option 1: Docker Image**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

   source switch_env openpi

**Option 2: Custom Environment**

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

Download the Model
------------------

The STEAM value model is built from two pretrained backbones:

- **SigLIP2-so400m** (``google/siglip-so400m-patch14-384``): vision encoder
- **Gemma3-270M** (``google/gemma-3-270m``): language model and tokenizer

.. code:: bash

   git lfs install
   git clone https://huggingface.co/google/siglip-so400m-patch14-384
   git clone https://huggingface.co/google/gemma-3-270m

   # Or via huggingface-hub (set HF_ENDPOINT=https://hf-mirror.com for a mirror)
   hf download google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384
   hf download google/gemma-3-270m --local-dir gemma-3-270m

Set the paths in the model config (``examples/value/steam/config/model/steam.yaml``):

.. code:: yaml

   actor:
     model:
       vision_repo_id: /path/to/siglip-so400m-patch14-384
       language_repo_id: /path/to/gemma-3-270m
       tokenizer_path: /path/to/gemma-3-270m

Data Preparation
----------------

STEAM uses datasets in the LeRobot format, categorized into two types:

- **SFT datasets**: Successful trajectories (human demos or trained policies).
- **Rollout datasets**: Online-collected trajectories with both successes and
  failures.

Example dataset configuration:

.. code:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
       - dataset_path: /path/to/rollout_dataset
         type: rollout

.. note::

   Keep ``train_data_paths`` and ``data.k`` consistent between Step 1 and Step 2:
   the advantage computation must score pairs at the same temporal stride the
   critic was trained on.

Pipeline Tag System
~~~~~~~~~~~~~~~~~~~~~

STEAM uses an **advantage tag** to pass data from Step 2 to the CFG stage. Set
Step 2's ``advantage.tag`` and the CFG stage's ``data.advantage_tag`` to the same
value so CFG reads ``meta/advantages_{tag}.parquet``.

Step 1: Value Model SFT
-----------------------

Train the ensemble progress critic. Each member is a SigLIP2 + Gemma3 backbone
with a classifier head; members are cloned from a shared backbone and their value
heads are re-seeded so ensemble variance is a meaningful epistemic signal.

**Configuration**

The config is ``examples/value/steam/config/steam_value_model.yaml``; the model
defaults live in ``config/model/steam.yaml``. Key fields:

.. code:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
     k: 32                       # max signed stride K (pair temporal scale)
     camera_keys: [face_view, left_wrist_view, right_wrist_view]
     prompt: "perform the task"

   actor:
     micro_batch_size: 32
     global_batch_size: 512
     model:
       num_bins: 32              # 2 = binary progress; >2 = multi-bin (even)
       ensemble_size: 3          # number of critics in the ensemble
       fusion_hidden_dim: 512
       freeze_vision_encoder: false
       freeze_language_model: false
       use_gradient_checkpointing: true
     optim:
       lr: 5.0e-5
       value_lr: 5.0e-5

**Key Parameters**

.. list-table::
   :header-rows: 1
   :widths: 32 16 52

   * - Parameter
     - Default
     - Description
   * - ``data.k``
     - ``required``
     - Max signed stride :math:`K`. In multi-bin mode ``2*K`` must be a multiple of ``num_bins``.
   * - ``actor.model.num_bins``
     - ``2``
     - Bin count. ``2`` is binary progress; ``> 2`` (even) is multi-bin signed-stride classification.
   * - ``actor.model.ensemble_size``
     - ``1``
     - Number of ensemble members. ``> 1`` enables worst-of-N aggregation and uncertainty stats.
   * - ``actor.model.fusion_hidden_dim``
     - ``512``
     - Hidden width of the per-frame fusion MLP.
   * - ``actor.model.freeze_vision_encoder``
     - ``false``
     - Freeze the SigLIP2 encoder.
   * - ``actor.model.use_gradient_checkpointing``
     - ``false``
     - Recompute backbone activations in backward (needed for full-backbone + ensemble on 80GB cards).

**Launch Command**

.. code:: bash

   bash examples/value/steam/run_steam_sft.sh steam_value_model

   # Override config fields inline:
   bash examples/value/steam/run_steam_sft.sh steam_value_model data.k=8

**Output**

- Checkpoints under ``logs/steam_sft/{config_name}-{timestamp}/.../checkpoints/global_step_{N}/actor``
- TensorBoard logs

**Key Metrics**

- ``train/actor/loss``: cross-entropy over the signed-stride bins
- ``train/actor/accuracy``: best-bin classification accuracy
- ``train/actor/grad_norm``: gradient norm

Step 2: Compute Ensemble Advantages
-----------------------------------

Run the trained ensemble over every frame and write per-frame advantage labels.

**Configuration**

The config is ``examples/value/steam/process/config/compute_advantages_ensemble.yaml``:

.. code:: yaml

   advantage:
     value_checkpoint: /path/to/steam_value_ensemble/checkpoints/global_step_N/actor
     batch_size: 256
     label_mode: quantile        # required: "threshold" or "quantile"
     rollout_quantile: 0.3       # top 30% of rollout frames labelled True
     expert_quantile: 0.8        # optional: top 80% of sft frames labelled True
     tag: steam_k32_ensemble3_q30

   data:
     model_type: "pi0"
     robot_type: "restock_cola_sm2sm"
     k: 32                       # must match Step 1 data.k
     camera_keys: [face_view, left_wrist_view, right_wrist_view]
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
       - dataset_path: /path/to/rollout_dataset
         type: rollout

**Key Parameters**

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - Parameter
     - Default
     - Description
   * - ``advantage.value_checkpoint``
     - ``required``
     - Path to the Step 1 ensemble checkpoint (``actor`` directory).
   * - ``advantage.label_mode``
     - ``required``
     - ``threshold`` or ``quantile`` (no default — must be set explicitly).
   * - ``advantage.positive_threshold``
     - ``null``
     - Signed-score threshold in :math:`[-1, 1]` (``label_mode=threshold`` only).
   * - ``advantage.rollout_quantile``
     - ``null``
     - Top fraction of rollout frames labelled True (``label_mode=quantile``, required).
   * - ``advantage.expert_quantile``
     - ``null``
     - Top fraction of sft frames labelled True (``label_mode=quantile``, optional).
   * - ``advantage.tag``
     - ``required``
     - Output tag; writes ``meta/advantages_{tag}.parquet``.
   * - ``data.k``
     - ``required``
     - Pair stride; must match the Step 1 training ``data.k``.

**Launch Command**

.. code:: bash

   # Auto-detects #GPUs; single-GPU or torchrun multi-GPU both supported.
   bash examples/value/steam/process/run_compute_advantages_ensemble.sh compute_advantages_ensemble

   # Force a GPU count:
   bash examples/value/steam/process/run_compute_advantages_ensemble.sh compute_advantages_ensemble --nproc 4

**Output Files**

- ``meta/advantages_{tag}.parquet``: per-frame ``advantage`` (bool),
  ``advantage_continuous`` (signed score), ``ensemble_signed_score``, per-member
  values, and ensemble entropy / variance diagnostics.
- ``meta/mixture_config.yaml``: a per-tag entry recording ``label_mode``, the
  applied threshold, ``ensemble_size``, ``num_bins``, and positive counts.

Step 3: CFG Training
--------------------

STEAM advantage parquets share RECAP's schema, so policy optimization reuses the
RECAP CFG stage. Point the CFG config's ``data.advantage_tag`` at the Step 2
``advantage.tag`` and launch:

.. code:: bash

   bash examples/embodiment/run_cfg_sft.sh libero_cfg_openpi \
       data.advantage_tag=steam_k32_ensemble3_q30

See :doc:`RECAP Step 4 <recap>` for the full CFG configuration and parameters.

Advanced Usage
--------------

Merge Ensemble Checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Members trained as separate single-model runs (or extracted from existing
ensembles) can be fused into one ensemble inference checkpoint. Each ``--member``
is a checkpoint path, or ``PATH:idx`` to pull member ``idx`` from an ensemble:

.. code:: bash

   python examples/value/steam/process/merge_steam_ensemble.py \
       --member /path/to/seed1/checkpoints/global_step_5000/actor \
       --member /path/to/seed2/checkpoints/global_step_5000/actor \
       --member /path/to/ensemble/checkpoints/global_step_6000/actor:2 \
       --output /path/to/merged/actor

The merge logic lives in
``rlinf.models.embodiment.steam.checkpoint_merge.merge_ensemble_checkpoints``.

Threshold / Quantile Relabeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To change the labelling threshold without rerunning GPU inference, relabel an
existing advantages parquet (pure CPU — ``advantage_continuous`` is reused):

.. code:: bash

   python examples/value/steam/process/relabel_advantages.py \
       --dataset_paths /path/to/sft_ds /path/to/rollout_ds \
       --source_tag steam_k32_ensemble3_q30 \
       --new_tag steam_k32_ensemble3_q20 \
       --mode quantile --rollout_quantile 0.2

The relabel logic lives in
``rlinf.data.process.steam.relabel.relabel_advantages``.

Visualize Advantages
~~~~~~~~~~~~~~~~~~~~~~

Render distribution, per-member, uncertainty, per-episode, and episode-timeline
diagnostics from an advantages parquet:

.. code:: bash

   python examples/value/steam/process/visualize_advantage.py \
       --dataset /path/to/dataset \
       --tag steam_k32_ensemble3_q30 \
       --output outputs/steam_viz

Visualization and Results
-------------------------

For metric definitions, see :doc:`Training metrics <../../reference/metrics>`.

.. code:: bash

   tensorboard --logdir ./logs --port 6006

File Structure
--------------

Like RECAP, STEAM keeps its pipeline scripts self-contained under ``examples/``
(the inference + labelling strategy that is bound to the model), the model /
dataset code under ``rlinf/models`` and ``rlinf/data/datasets``, and shares the
model-agnostic post-processing with RECAP via ``rlinf/data/process/``:

.. code-block:: text

   examples/value/steam/
   ├── train_steam.py                         # Step 1: value model SFT entry
   ├── run_steam_sft.sh                       # Step 1 launch script
   ├── config/
   │   ├── steam_value_model.yaml
   │   └── model/steam.yaml
   └── process/
       ├── compute_advantages_ensemble.py     # Step 2: ensemble inference +
       │                                      #   two-pool labelling (self-contained)
       ├── merge_steam_ensemble.py            # CLI: merge ensemble checkpoints
       ├── relabel_advantages.py              # CLI: relabel advantages (CPU)
       ├── visualize_advantage.py             # advantage visualization
       ├── run_compute_advantages_ensemble.sh # Step 2 launch script
       └── config/
           └── compute_advantages_ensemble.yaml

   rlinf/
   ├── models/embodiment/steam/                   # critic, ensemble, config, merge
   │   ├── modeling_steam.py / modeling_critic.py
   │   ├── ensemble_modeling_critic.py            # worst-of-N + coerce_to_ensemble
   │   └── checkpoint_merge.py                    # ensemble checkpoint merge
   ├── data/datasets/steam/binning.py             # signed-stride ↔ bin math + entropy
   └── data/process/                              # shared post-processing (RECAP + STEAM)
       ├── advantage.py                           # quantile threshold + boolean label
       ├── mixture_config.py                      # tags[tag] metadata I/O
       └── distributed.py                         # sharded-inference helpers
