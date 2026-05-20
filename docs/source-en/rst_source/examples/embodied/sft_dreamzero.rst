DreamZero Supervised Fine-Tuning
=================================

This document explains how to run DreamZero supervised fine-tuning (SFT) in RLinf, covering the currently supported combinations:

- Dataset formats: **LIBERO**, **LeRobot / OXE DROID**
- Backbones: **WAN2.1**, **WAN2.2**

It is intended as an end-to-end guide from data/model preparation to launch and troubleshooting.


Supported setups
----------------

Recommended configs:

- ``examples/sft/config/libero_sft_dreamzero.yaml``: LIBERO SFT
- ``examples/sft/config/droid_sft_dreamzero.yaml``: LeRobot/OXE DROID SFT (WAN2.1/WAN2.2, depending on ``model_path``)

Common starting points:

- **Continue SFT from released DreamZero weights**: faster and more stable convergence
- **Cold-start from WAN2.2 base model**: more flexible, but requires more data/tuning budget


Training entrypoint
-------------------

Use:

- ``examples/sft/run_vla_sft.sh``

The script runs:

.. code:: bash

   python examples/sft/train_vla_sft.py \
     --config-path examples/sft/config/ \
     --config-name <config_name> \
     runner.logger.log_path=<auto_log_dir>

Logs are written to:

- ``<repo>/logs/<timestamp>/run_embodiment.log``


Environment
-----------

1. Clone RLinf and install DreamZero-related dependencies.
2. Set ``DREAMZERO_PATH`` to your DreamZero repository path.
3. Ensure ``PYTHONPATH`` in ``run_vla_sft.sh`` includes:

   - RLinf repo path
   - DreamZero repo path

4. Use a CUDA/PyTorch/Transformers stack compatible with your current project.


Data preparation
----------------

LIBERO
~~~~~~

- Config: ``libero_sft_dreamzero.yaml``
- Key field: ``data.train_data_paths``

.. code:: yaml

   data:
     train_data_paths: "path/to/datasets/libero"


LeRobot / OXE DROID
~~~~~~~~~~~~~~~~~~~

- Config: ``droid_sft_dreamzero.yaml``
- Key field: ``data.train_data_paths``
- Dataset directory should follow LeRobot layout (e.g., ``meta`` and ``data``).

.. code:: yaml

   data:
     train_data_paths: /path/to/droid_lerobot


Model and weight preparation
----------------------------

Hydra composes ``actor.model`` from defaults (for example ``model/dreamzero_5b@actor.model`` in
``libero_sft_dreamzero_5b.yaml``). Training configs can override any field on ``actor.model``.

**Policy architecture**

- If ``actor.model.model_path`` is set: load ``<model_path>/config.json`` (other architecture fields on
  ``actor.model`` are ignored for the policy config; a warning is logged).
- If ``actor.model.model_path`` is ``null``: use the resolved Hydra ``actor.model`` dict. These paths must be
  non-null: ``tokenizer_path``, ``diffusion_model_pretrained_path``, ``image_encoder_pretrained_path``,
  ``text_encoder_pretrained_path``, ``vae_pretrained_path``.

Preset files under ``examples/sft/config/model/`` (e.g. ``dreamzero_5b.yaml``) are included via Hydra
``defaults``; they are not selected by separate ``dreamzero_*`` override keys.

Data transforms are constructed in Python from ``embodiment_tag``; optional numeric overrides live under
``video_*``, ``state_horizon``, ``action_horizon``, ``max_state_dim``, ``max_action_dim``, ``max_seq_len``, etc. on ``actor.model``
(see ``data_transforms/__init__.py``). The SFT dataloader defaults to Groot **sharded** temporal sampling
(``data.sampling_mode: sharded``, ``data.max_temporal_blocks``) from ``lerobot_sharded.py``; ``action_horizon`` is only for
``DreamTransform`` / WAN (per-block), not the dataset macro stride (24 frames).

Dataset statistics are loaded from the dataset-generated ``metadata.json`` file.
Specify its path with ``metadata_json_path``, or ensure the file exists at
``model_path/experiment_cfg/metadata.json`` (keyed by embodiment tag).

WAN2.1 path
~~~~~~~~~~~

If you use a **full** DreamZero checkpoint directory (for example an official release tree), set ``model_path`` to that
directory. It typically contains:

- ``config.json``
- ``experiment_cfg/metadata.json`` (dataset statistics; ``conf.yaml`` is not read by RLinf)
- ``model.safetensors`` (or sharded safetensors)

When ``model_path`` is set, RLinf loads **config** from ``config.json`` and **metadata** from
``experiment_cfg/metadata.json`` (unless ``metadata_json_path`` is set). **Data transforms** are still assembled in Python from ``embodiment_tag``.

.. code:: yaml

   actor:
     model:
       model_type: "dreamzero"
       model_path: /path/to/models/DreamZero-DROID
       tokenizer_path: /path/to/models/umt5-xxl


WAN2.2 path
~~~~~~~~~~~

WAN2.2 training usually requires:

- WAN2.2 backbone weights (DiT + T5 + VAE)
- WAN2.1 CLIP image encoder weights (not included in WAN2.2-TI2V-5B)
- tokenizer (umt5-xxl)

For WAN2.2, the **architecture** in ``config.json`` (whether from a checkpoint or from the RLinf preset) should be consistent with weights, for example:

- ``diffusion_model_cfg``: ``model_type=ti2v``, ``in_dim=48``, ``out_dim=48``, ``frame_seqlen=50``
- ``vae_cfg``: ``WanVideoVAE38``
- ``image_encoder_pretrained_path`` points to WAN2.1 CLIP weights when loading WAN components

.. code:: yaml

   actor:
     model:
       model_type: "dreamzero"
       model_path: /path/to/models/dreamzero-22
       tokenizer_path: /path/to/hf_models/umt5-xxl


Key DreamZero config fields
---------------------------

Important fields:

- ``actor.model.model_type``: ``dreamzero``
- ``actor.model.model_path``: checkpoint directory (``null`` to use Hydra ``actor.model`` only). When set, loads ``config.json`` from this path; metadata from ``metadata_json_path`` or ``experiment_cfg/metadata.json``. Data transforms are always built in Python (see ``data_transforms/__init__.py``).
- ``actor.model.tokenizer_path``: tokenizer path
- ``actor.model.embodiment_tag``: usually ``oxe_droid`` (DROID) or LIBERO-related tag
- ``data.sampling_mode``: ``sharded`` (Groot language-aware multi-anchor sampling) or ``dense`` (legacy contiguous window).
- ``data.max_temporal_blocks``: macro temporal blocks (optional, default **4**, matches common checkpoints; set to **5** to match official Groot DROID data recipes).
- ``actor.model.action_horizon``: **DreamTransform / WAN per-block** action steps (LIBERO 16, DROID 24). Sharded dataset action length is typically ``24 * max_temporal_blocks`` (e.g. 96), not ``action_horizon * num_chunks``.
- ``actor.model.num_chunks``: legacy dense-mode alias; ignored for sharded offsets when ``max_temporal_blocks`` is set.
- ``actor.model.state_horizon``: state rows per sample (usually 1; one state per macro anchor).
- ``actor.model.num_video_frames``: only used when ``data.sampling_mode: dense`` (sharded mode yields ~``8 * data.max_temporal_blocks + 1`` frames, e.g. 33).
- ``data.sharded_resample_attempts``: retries when sharded sampling returns empty indices (map-style dataloader).
- ``actor.model.droid_view_height`` / ``droid_view_width`` (optional per-view resize for DROID)
- ``actor.model.relative_action``
- ``actor.fsdp_config``

For WAN2.1/WAN2.2 compatibility, avoid hard-coded video sizes in data preprocessing.
Prefer config-based inference or explicit overrides.


Launch training
---------------

Run from repository root:

.. code:: bash

   # LIBERO
   bash examples/sft/run_vla_sft.sh libero_sft_dreamzero

   # LeRobot/DROID (WAN2.1 or WAN2.2 by model_path)
   bash examples/sft/run_vla_sft.sh droid_sft_dreamzero


Monitoring and sanity checks
----------------------------

1. Check ``run_embodiment.log``:

   - stable ``time/step``
   - reasonable ``train/loss``, ``train/action_loss``, ``train/dynamics_loss``

2. TensorBoard:

.. code:: bash

   tensorboard --logdir ./logs --port 6006

3. Early data sanity checks:

   - ``images/state/action`` shapes, dtypes, value ranges
   - valid ratios for ``state_mask/action_mask/text_attention_mask``
   - for WAN2.2, verify input resolution and ``frame_seqlen`` consistency


Common issues
-------------

1. **Missing model weights (No safetensors weights)**

- Check whether ``model.safetensors`` or sharded index exists under ``model_path``
- If using WAN base cold-start, ensure loader supports component-based initialization

2. **WAN2.2 dimension mismatch**

- Verify ``diffusion_model_cfg`` in the effective config (``model_path/config.json`` or Hydra ``actor.model``) matches WAN2.2
- Verify ``vae_cfg`` uses ``WanVideoVAE38``

3. **Abnormally high action loss**

- Check normalization and relative-action settings for conflicts
- Check temporal alignment (horizon/chunks) between data and model config

4. **DROID video size mismatch**

- Avoid hard-coded sizes in code
- Use config-based inference or explicit overrides for WAN2.1/WAN2.2 compatibility


Practical recommendations
-------------------------

- For fast and stable convergence, start from released DreamZero weights.
- For full WAN2.2 adaptation, cold-start is possible but needs larger data and longer training.
- After each config change, run a short trial (e.g., 50-200 steps) to validate shapes/loss/throughput before long runs.

