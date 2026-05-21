DreamZero Supervised Fine-Tuning (SFT)
======================================

This guide explains how to run DreamZero supervised fine-tuning (SFT) in RLinf, from **model and data preparation** through **configuration**, **launching training**, and **troubleshooting**.

Currently supported:

- **Datasets**: LIBERO (LeRobot), LeRobot / OXE DROID
- **Backbones**: WAN2.1 (e.g. DreamZero-DROID 14B), WAN2.2 (e.g. Wan2.2-TI2V-5B cold start)


Supported setups
----------------

Recommended configs (``examples/sft/config/``):

.. list-table::
   :header-rows: 1
   :widths: 22 18 18 42

   * - Config file
     - Dataset / ``embodiment_tag``
     - Hydra preset
     - Typical use
   * - ``libero_sft_dreamzero_14b.yaml``
     - LIBERO / ``libero_sim``
     - ``model/dreamzero_14b``
     - **WAN2.1**: set ``model_path`` to a full checkpoint (e.g. official DreamZero weights); architecture and weights load from ``config.json``; requires ``metadata_json_path`` or ``experiment_cfg/metadata.json``.
   * - ``libero_sft_dreamzero_5b.yaml``
     - LIBERO / ``libero_sim``
     - ``model/dreamzero_5b``
     - **WAN2.2 cold start**: ``model_path: null``; fill Wan2.2-TI2V-5B ``*_pretrained_path`` entries and ``metadata_json_path``.
   * - ``droid_sft_dreamzero_14b.yaml``
     - DROID / ``oxe_droid``
     - ``model/dreamzero_14b``
     - **DROID SFT (WAN2.1)**: ``defaults`` include ``dreamzero_14b``; default ``model_path`` points to DreamZero-DROID; architecture and weights load from ``config.json``. Includes ``relative_action``, ``sampling_mode: multi_anchor``, etc.

Common starting points:

- **Resume from a released checkpoint** (``libero_sft_dreamzero_14b``, ``droid_sft_dreamzero_14b`` with non-null ``model_path``): faster, more stable.
- **WAN2.2 component cold start** (``libero_sft_dreamzero_5b``, ``model_path: null``): more flexible; needs more data and ``metadata.json``.


Environment setup
-----------------

1. Clone the RLinf repository and enter the repo root:

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Use ``requirements/install.sh`` to create and sync a **DreamZero-specific uv virtual environment** (default directory ``.venv``, Python 3.11.14):

.. code:: bash

   # SFT only (offline LeRobot data, no simulation) — recommended
   bash requirements/install.sh embodied --model dreamzero

   # If you also need LIBERO simulation for evaluation
   bash requirements/install.sh embodied --model dreamzero --env libero

Notes:

- The installer: creates/reuses ``.venv`` with ``uv`` → ``uv sync --extra embodied`` → installs ``requirements/embodied/models/dreamzero.txt`` (``lerobot``, ``diffusers``, ``torchcodec``, etc.) → installs ``flash-attn`` on NVIDIA platforms.
- Add ``--use-mirror`` for faster PyPI / Python / GitHub downloads in regions with limited access.
- Custom venv directory: ``--venv <dir>``; skip system deps when already installed: ``--no-root``.

After installation, activate the environment:

.. code:: bash

   source .venv/bin/activate

3. Clone the **DreamZero (Groot)** codebase separately and set ``DREAMZERO_PATH`` to the **Python package root** (must contain the ``groot`` package; CI often uses ``<DreamZero>/dreamzero``):

.. code:: bash

   # Clone the official DreamZero / Groot repo (URL per project release notes)
   export DREAMZERO_PATH=/path/to/DreamZero/dreamzero   # must support `import groot`


Model preparation
-----------------

Hydra composes ``actor.model`` via ``defaults`` (e.g. ``model/dreamzero_5b@actor.model`` in ``libero_sft_dreamzero_5b.yaml``). Training YAML can override any field under ``actor.model``.

**Two ways to load policy architecture**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mode
     - Description
   * - ``model_path`` set
     - Load the **full architecture** from ``<model_path>/config.json``; other architecture fields in YAML are ignored (a warning is logged). Weights load from that directory (``model.safetensors`` or sharded index).
   * - ``model_path: null``
     - Use the resolved Hydra ``actor.model`` dict (or presets like ``model/dreamzero_5b.yaml``). These paths **must be non-null**: ``tokenizer_path``, ``diffusion_model_pretrained_path``, ``image_encoder_pretrained_path``, ``text_encoder_pretrained_path``, ``vae_pretrained_path``.

**Normalization statistics (metadata.json)**

Action/state normalization stats come from a dataset-generated ``metadata.json``. Specify via:

- ``actor.model.metadata_json_path``: explicit path; or
- ``<model_path>/experiment_cfg/metadata.json`` (entry keyed by ``embodiment_tag``)

For cold start without that file in the checkpoint directory, generate it with the toolkit below and set ``metadata_json_path``.

**Data transforms**

Regardless of ``model_path``, the **transform chain is built in Python from ``embodiment_tag``** (see ``rlinf/data/datasets/dreamzero/data_transforms/__init__.py``). Built-in tags:

- ``libero_sim``: LIBERO simulation
- ``oxe_droid``: LeRobot / OXE DROID

WAN2.1: use a full DreamZero checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Point ``actor.model.model_path`` at an official or custom DreamZero directory, typically containing:

- ``config.json``: model architecture
- ``experiment_cfg/metadata.json``: dataset normalization stats (optional; can override with ``metadata_json_path``)
- ``model.safetensors`` (or sharded safetensors)
- ``tokenizer_path``: still required in YAML (e.g. ``umt5-xxl``)

Add ``model/dreamzero_14b@actor.model`` under ``defaults`` (see ``libero_sft_dreamzero_14b.yaml``).

Example (LIBERO + 14B checkpoint):

.. code:: yaml

   defaults:
     - model/dreamzero_14b@actor.model

   actor:
     model:
       model_path: /path/to/models/<your-checkpoint>   # see libero_sft_dreamzero_14b.yaml
       metadata_json_path: /path/to/dataset/metadata.json
       tokenizer_path: /path/to/models/umt5-xxl
       embodiment_tag: "libero_sim"
       action_horizon: 16

Example (DROID; see ``droid_sft_dreamzero_14b.yaml``; ``defaults`` include ``dreamzero_14b``; non-null ``model_path`` loads architecture and weights from the checkpoint):

.. code:: yaml

   actor:
     model:
       model_type: "dreamzero"
       model_path: /path/to/models/DreamZero-DROID
       metadata_json_path: /path/to/dataset/metadata.json
       tokenizer_path: /path/to/models/umt5-xxl
       embodiment_tag: "oxe_droid"
       action_horizon: 24
       relative_action: True

WAN2.2: cold start from components (5B, etc.)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WAN2.2 training typically requires:

- WAN2.2 backbone weights (DiT + T5 + VAE)
- WAN2.1 CLIP image encoder (**not** included in Wan2.2-TI2V-5B; download separately)
- Tokenizer ``google/umt5-xxl``

Add ``model/dreamzero_5b@actor.model`` under ``defaults``, set ``model_path: null``, and fill each ``*_pretrained_path``. Architecture in ``config.json`` or the preset must match the weights, e.g.:

- ``diffusion_model_cfg``: ``model_type=ti2v``, ``in_dim=48``, ``out_dim=48``, ``frame_seqlen=50``
- ``vae_cfg``: ``WanVideoVAE38``
- ``image_encoder_pretrained_path`` points to WAN2.1 CLIP weights

Example (LIBERO + 5B cold start; see ``libero_sft_dreamzero_5b.yaml``):

.. code:: yaml

   defaults:
     - model/dreamzero_5b@actor.model

   actor:
     model:
       model_path: null
       tokenizer_path: google/umt5-xxl
       diffusion_model_pretrained_path: Wan-AI/Wan2.2-TI2V-5B
       image_encoder_pretrained_path: Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
       text_encoder_pretrained_path: Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
       vae_pretrained_path: Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
       metadata_json_path: /path/to/metadata.json
       embodiment_tag: "libero_sim"

Preset files ``examples/sft/config/model/dreamzero_5b.yaml`` (WAN2.2 / 5B) and ``dreamzero_14b.yaml`` (WAN2.1 / 14B) are included via Hydra ``defaults``. Do **not** select them with a separate top-level ``dreamzero_*`` key. Use ``model/dreamzero_14b@actor.model`` for LIBERO/DROID 14B examples and ``model/dreamzero_5b@actor.model`` for LIBERO 5B cold start.


Data preparation
----------------

Data must follow the **LeRobot v2** layout (``meta/``, ``data/``, etc.). Set the dataset root or HuggingFace repo id via ``data.train_data_paths``.

LIBERO
~~~~~~

- Config: ``libero_sft_dreamzero_14b.yaml`` (WAN2.1 / checkpoint) or ``libero_sft_dreamzero_5b.yaml`` (WAN2.2 cold start)
- ``actor.model.embodiment_tag`` must be ``libero_sim``
- Example data path:

.. code:: yaml

   data:
     train_data_paths: physical-intelligence/libero   # or local LeRobot root

LeRobot / OXE DROID
~~~~~~~~~~~~~~~~~~~

- Config: ``droid_sft_dreamzero_14b.yaml``
- ``actor.model.embodiment_tag`` must be ``oxe_droid``
- Use ``data.sampling_mode: multi_anchor`` and ``data.lazy_load: True`` (enabled by default in the example)
- Directory must follow LeRobot DROID layout (including ``meta/modality.json``)

.. code:: yaml

   data:
     train_data_paths: /path/to/droid_lerobot

Generating metadata.json
~~~~~~~~~~~~~~~~~~~~~~~~

For a new dataset or cold start (no ``experiment_cfg/metadata.json``), generate normalization stats for the corresponding ``embodiment_tag`` first:

.. code:: bash

   # LIBERO
   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset libero_sim \
     --dataset-root /path/to/libero \
     --output-metadata /path/to/metadata.json

   # DROID (use --merge for multiple datasets)
   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset oxe_droid \
     --dataset-root /path/to/droid \
     --output-metadata /path/to/metadata.json \
     --merge

Then set ``actor.model.metadata_json_path`` in config (or place the file at ``model_path/experiment_cfg/metadata.json``).


Configuration reference
-----------------------

Configs are managed by Hydra; the entry script is ``examples/sft/train_vla_sft.py``. Below, **data fields (``data.*``)** and **model/training fields (``actor.model.*`` and related)** are explained separately.

Data-related settings (``data``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning and role
   * - ``train_data_paths``
     - LeRobot dataset root or HF ``repo_id``. Determines which episodes / parquet / video files are read.
   * - ``lazy_load``
     - Lazy-load mp4 videos. **Must be ``True`` for ``multi_anchor`` sampling** (otherwise anchor-based frame lookup fails).
   * - ``sampling_mode``
     - ``multi_anchor`` (default, recommended): sample multiple temporal anchors within the same language span (Groot ``lerobot_sharded`` semantics); macro block count comes from ``max_chunk_size``. **``fixed_window``**: contiguous fixed window; use with ``num_video_frames``.
   * - ``multi_anchor_resample_attempts``
     - Retries when multi-anchor sampling returns empty indices (map-style dataloader).
   * - ``video_backend``
     - LeRobot video decoder: ``pyav`` or ``torchcodec``; affects lazy mp4 speed and compatibility. **``torchcodec`` is recommended.**
   * - ``video_tolerance_s``
     - Timestamp tolerance (seconds) between video frames and target times.
   * - ``parquet_cache_size``
     - Max cached parquet episodes; affects memory and I/O.
   * - ``num_workers`` / ``prefetch_factor``
     - DataLoader parallelism and prefetch; affects throughput.

**Temporal alignment (data sampling vs model blocks)**

- Macro temporal block count: ``actor.model.action_head_cfg.config.diffusion_model_cfg.max_chunk_size`` (commonly **4**; official Groot DROID recipes may use **5**).
- ``actor.model.action_horizon``: **per-block action steps in DreamTransform / WAN** (LIBERO 16, DROID 24), not the dataset macro stride.
- Under ``multi_anchor``, dataset action length is roughly ``action_horizon * max_chunk_size`` (e.g. LIBERO 64, DROID 96).
- ``actor.model.num_chunks``: mainly for ``fixed_window``; ``multi_anchor`` uses ``max_chunk_size`` (warns if it differs from ``num_chunks``).
- ``actor.model.num_video_frames``: only for ``sampling_mode: fixed_window``; under ``multi_anchor``, frame count is ``8 * max_chunk_size + 1`` (e.g. 33).

Model and training settings (``actor``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Identity and weight paths**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning and role
   * - ``model_type``
     - Must be ``dreamzero``.
   * - ``model_path``
     - Full checkpoint directory; when non-null, architecture loads from ``config.json`` and weights from that path. ``null`` uses YAML/preset + ``*_pretrained_path`` for cold start.
   * - ``tokenizer_path``
     - UMT5 tokenizer path (required for training and collate).
   * - ``diffusion_model_pretrained_path``
     - Causal DiT (diffusion backbone) pretrained weights; required for cold start.
   * - ``image_encoder_pretrained_path``
     - WAN image encoder; WAN2.2 must point to **WAN2.1 CLIP** weights.
   * - ``text_encoder_pretrained_path``
     - T5 text encoder weights.
   * - ``vae_pretrained_path``
     - VAE weights; WAN2.2 uses ``WanVideoVAE38``.
   * - ``metadata_json_path``
     - Dataset ``metadata.json``; falls back to ``model_path/experiment_cfg/metadata.json`` if unset.
   * - ``embodiment_tag``
     - Selects data transform and collate template: ``libero_sim`` or ``oxe_droid``. **Must match the dataset.**

**Temporal and action shape (must align with data and WAN capacity)**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning and role
   * - ``action_horizon``
     - Action steps per WAN temporal block (LIBERO 16, DROID 24).
   * - ``state_horizon``
     - State rows per sample (usually 1, one state per macro anchor).
   * - ``num_chunks``
     - Contiguous chunks in ``fixed_window`` mode; under ``multi_anchor``, ``max_chunk_size`` applies.
   * - ``num_action_per_block``
     - Align with DiT ``num_action_per_block`` in ``action_head_cfg`` (often equals ``action_horizon``).
   * - ``action_head_cfg...diffusion_model_cfg.max_chunk_size``
     - Multi-anchor macro temporal blocks / Causal DiT capacity; tied to ``data.sampling_mode: multi_anchor``.
   * - ``num_video_frames``
     - Only used in ``fixed_window`` mode.
   * - ``max_action_dim`` / ``max_state_dim`` / ``max_seq_len``
     - Padding limits and max text sequence length in DreamTransform.

**Video size and DROID-specific options**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning and role
   * - ``target_video_height`` / ``target_video_width``
     - WAN policy head target resolution (5B preset e.g. 176×320; override in YAML). Avoid hard-coded sizes in transform code for WAN2.1/WAN2.2 compatibility.
   * - ``droid_view_height`` / ``droid_view_width``
     - (Optional) per-view resize overrides for DROID.
   * - ``relative_action`` / ``relative_action_keys`` / ``relative_action_per_horizon``
     - Relative action settings; DROID often uses ``relative_action: True`` on keys like ``joint_position``.

**Other training options**

- ``precision``: main precision for Actor/optimizer (``fp32`` / ``bf16``). **Recommended: ``fp32``** with ``actor.fsdp_config.mixed_precision`` for mixed precision: ``precision: fp32`` keeps **optimizer states and master weights in FP32** (more stable), while FSDP runs forward/backward matmuls in **BF16** via ``mixed_precision`` (saves memory, faster). Example:

  .. code:: yaml

     actor:
       model:
         precision: fp32
       fsdp_config:
         mixed_precision:
           param_dtype: bf16
           reduce_dtype: bf16
           buffer_dtype: bf16

  Setting ``precision: bf16`` also lowers optimizer state precision and is usually less stable. With FSDP **CPU offload**, keep ``precision: fp32``.
- ``is_lora``: LoRA fine-tuning (DreamZero SFT examples typically use full fine-tuning ``False``).
- ``actor.micro_batch_size`` / ``actor.global_batch_size``: per-GPU micro-batch and global effective batch size.
- ``actor.optim.*``: learning rate, warmup, cosine schedule, etc.
- ``actor.fsdp_config``: FSDP2 sharding, gradient checkpointing; ``mixed_precision`` controls compute/comm dtypes (works with ``actor.model.precision`` above).

**Example config sketch**

.. code:: yaml

   # ---------- data ----------
   data:
     train_data_paths: /path/to/libero
     lazy_load: True
     sampling_mode: multi_anchor
     video_backend: torchcodec
     num_workers: 8

   # ---------- model (resume from checkpoint) ----------
   actor:
     model:
       model_path: /path/to/DreamZero-DROID
       tokenizer_path: /path/to/umt5-xxl
       embodiment_tag: oxe_droid
       action_horizon: 24
       metadata_json_path: /path/to/metadata.json   # if no experiment_cfg/metadata.json


Launch training
---------------

From the repository root:

.. code:: bash

   # LIBERO + WAN2.1 (checkpoint, dreamzero_14b preset)
   bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_14b

   # LIBERO + WAN2.2 (cold start, dreamzero_5b preset)
   bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_5b

   # DROID + WAN2.1 (dreamzero_14b preset; model_path -> DreamZero-DROID)
   bash examples/sft/run_vla_sft.sh droid_sft_dreamzero_14b

Equivalent command:

.. code:: bash

   python examples/sft/train_vla_sft.py \
     --config-path examples/sft/config/ \
     --config-name <config_name> \
     runner.logger.log_path=<auto_log_dir>

Logs:

- ``<repo>/logs/<timestamp>-<config_name>/run_embodiment.log``

Resume training with ``runner.resume_dir`` pointing to a checkpoint directory (field provided in example configs such as ``droid_sft_dreamzero_14b.yaml`` and ``libero_sft_dreamzero_5b.yaml``).


Monitoring and sanity checks
----------------------------

1. Inspect ``run_embodiment.log``: stable ``time/step``; reasonable ``train/loss``, ``train/action_loss``, ``train/dynamics_loss``.

2. TensorBoard:

.. code:: bash

   tensorboard --logdir ./logs --port 6006

3. Check early in the run:

   - ``images`` / ``state`` / ``action`` shapes, dtypes, value ranges
   - Valid ratios for ``state_mask`` / ``action_mask`` / ``text_attention_mask``
   - For WAN2.2: input resolution and ``frame_seqlen`` match ``config.json`` or the preset


Extension: adding a new ``embodiment_tag``
------------------------------------------

To train DreamZero SFT on a **new robot or LeRobot dataset**, add an ``embodiment_tag`` and register the corresponding transforms and metadata tooling in RLinf. Use existing modules as templates:

- ``rlinf/data/datasets/dreamzero/data_transforms/libero_sim.py`` (two views, simple state/action columns)
- ``rlinf/data/datasets/dreamzero/data_transforms/oxe_droid.py`` (three views, ``meta/modality.json`` slicing)

Data flow:

.. code:: text

   LeRobot dataset
        → DreamZeroLeRobotDataset (reads parquet/mp4 via transform keys)
        → ComposedModalityTransform + DreamTransform (normalize, multi-view concat, tokenize)
        → DreamZeroCollator → training

Step 1: Implement the embodiment transform module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create ``rlinf/data/datasets/dreamzero/data_transforms/<your_tag>.py`` implementing ``DreamZeroEmbodimentTransform`` (see ``base.py``), including at least:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Member / method
     - Description
   * - ``TAG``
     - String id; must **exactly match** ``actor.model.embodiment_tag`` and the top-level key in ``metadata.json``.
   * - ``DEFAULT_TAG_MAPPING``
     - ``{TAG: <int>}`` maps to the WAN action head **embodiment projector ID**. When fine-tuning released DreamZero weights, the ID must appear in ``action_loss_embodiment_ids`` in checkpoint ``config.json`` (5B preset includes 17, 21, 26). A **new ID** implies random projector init or model config changes.
   * - ``DEFAULT_ACTION_HORIZON``
     - Default per-block action steps (LIBERO 16, DROID 24); align with ``actor.model.action_horizon``.
   * - ``get_modality_config()``
     - Returns ``ModalityConfig`` for ``video`` / ``state`` / ``action`` / ``language`` (``delta_indices``, ``modality_keys``). ``language`` keys must exist in the dataset. Video/action ``delta_indices`` should match Groot recipes (current code uses video ``range(25)``, action ``range(24)``); mismatch breaks ``multi_anchor`` alignment.
   * - ``get_transform(...)``
     - Build ``Video*`` → ``StateAction*`` → ``ConcatTransform`` → ``DreamTransform``; RLinf's ``DreamTransform`` (``dream_transform.py``) calls the registry for multi-view concat.
   * - ``format_training_prompt(instruction)``
     - T5 prompt prefix describing the multi-view layout (consistent with Groot training templates).
   * - ``concat_multiview_video(images)``
     - Concatenate ``(v, t, c, h, w)`` to ``(1, t, c, H, W)``; layout must match ``format_training_prompt``.

**``modality_keys`` naming** (wired to ``DreamZeroLeRobotDataset``):

- Video: ``video.<short_name>`` (e.g. ``video.image``); short names resolve via ``meta/modality.json`` ``original_key`` or ``info.json`` ``observation.images.*`` / bare column names.
- State/action: ``state.<name>``, ``action.<name>``; with ``meta/modality.json``, use ``start``/``end`` slices; otherwise fallback to full ``observation.state`` / ``action`` columns or heuristics (see ``_build_component_sources`` in ``dreamzero.py``).
- Keys in training YAML must match ``*_concat_order`` in ``ConcatTransform``.

Step 2: Register in RLinf
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit ``rlinf/data/datasets/dreamzero/data_transforms/__init__.py``:

1. ``from ...<your_tag> import YourEmbodimentDataTransform``
2. Add ``YourEmbodimentDataTransform.TAG: YourEmbodimentDataTransform`` to ``_EMBODIMENT_REGISTRY``

If unregistered, ``build_dreamzero_composed_transform`` errors and lists known tags.

Step 3: Generate ``metadata.json``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute normalization stats; the output key must equal ``TAG``:

**Option A (recommended)**: add an entry to ``PRESETS`` in ``toolkits/lerobot/generate_dreamzero_metadata.py`` (mirror ``libero_sim`` / ``oxe_droid``: ``state_key``, ``action_key``, ``video_keys``, ``use_modality_json``), then:

.. code:: bash

   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset <your_tag> \
     --dataset-root /path/to/lerobot_dataset \
     --output-metadata /path/to/metadata.json

**Option B**: use CLI flags without editing the script (``--embodiment-tag``, ``--state-key``, ``--action-key``, ``--video-keys``, ``--use-modality-json``).

Set ``actor.model.metadata_json_path`` in training config (or ``model_path/experiment_cfg/metadata.json``).

Step 4: Author / adjust training config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Copy ``libero_sft_dreamzero_14b.yaml``, ``libero_sft_dreamzero_5b.yaml``, or ``droid_sft_dreamzero_14b.yaml`` and update at least:

.. code:: yaml

   data:
     train_data_paths: /path/to/your_lerobot
     lazy_load: True              # required for multi_anchor with mp4
     sampling_mode: multi_anchor

   actor:
     model:
       embodiment_tag: "<your_tag>"
       metadata_json_path: /path/to/metadata.json
       action_horizon: <match DEFAULT_ACTION_HORIZON>
       # when resuming: verify action_loss_embodiment_ids includes your projector ID
       target_video_height: ...
       target_video_width: ...
       relative_action: ...
       relative_action_keys: [...]

For WAN cold start, add the new ID to ``action_head_cfg.config.action_loss_embodiment_ids`` in ``examples/sft/config/model/dreamzero_5b.yaml`` (or ``dreamzero_14b.yaml``).

Step 5: Validate (short run + data checks)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Run the metadata script alone; confirm ``metadata.json[<your_tag>]`` statistics/modalities match parquet dimensions.
2. Run 50–200 SFT steps; ensure no ``Could not map transform video keys`` or ``embodiment_tag not found in metadata`` errors.
3. Check finite ``train/action_loss``; verify batch ``images`` concat shape and ``embodiment_id`` vs ``DEFAULT_TAG_MAPPING``.

**Pitfall checklist**

- ``embodiment_tag`` string must match in **config**, **metadata.json key**, and Python ``TAG``.
- ``multi_anchor`` + mp4 data: **must** set ``data.lazy_load: True``.
- Dataset action length is ``action_horizon × max_chunk_size``; do not change only one.
- Multi-view **concat order** must match **prompt text** or training signal is wrong.
- Do not change ``DEFAULT_TAG_MAPPING`` integer IDs arbitrarily when fine-tuning official weights.
- Prefer ``target_video_height/width`` or transform-chain resize over hard-coded sizes for WAN2.1/2.2.
- Inference/eval: set ``embodiment_tag`` correctly in ``examples/embodiment/config/*_dreamzero.yaml``.

For inference only (no RLinf code changes) when upstream Groot/DreamZero already supports the tag, ``metadata.json`` and eval config may suffice; **SFT on new data** usually requires the Python registration and transform steps above.


Common issues
-------------

1. **Missing weights (No safetensors weights)**

   - Check ``model.safetensors`` or a sharded index under ``model_path``
   - For cold start, ensure all ``*_pretrained_path`` entries are valid and match the architecture

2. **WAN2.2 dimension mismatch**

   - Verify effective config (``model_path/config.json`` or ``dreamzero_5b`` preset): ``diffusion_model_cfg`` is ti2v, ``in_dim/out_dim=48``, ``vae_cfg`` is ``WanVideoVAE38``
   - Image encoder must use WAN2.1 CLIP paths

3. **metadata.json not found**

   - Run ``toolkits/lerobot/generate_dreamzero_metadata.py`` and set ``metadata_json_path``
   - Confirm JSON contains a key matching ``embodiment_tag``

4. **Abnormally high action_loss**

   - Check normalization stats match the current dataset
   - Check ``relative_action`` settings vs data
   - Align ``action_horizon``, ``max_chunk_size``, and ``sampling_mode``

5. **DROID video size errors**

   - Do not hard-code resolution in code; use ``target_video_height/width`` or ``droid_view_*``

6. **multi_anchor requires lazy_load**

   - Set ``data.lazy_load: True``


Practical recommendations
-------------------------

- For stable convergence, prefer continuing SFT from released DreamZero weights (set ``model_path``).
- Full WAN2.2 adaptation via cold start needs more data and longer training; after config changes, run 50–200 steps to validate shapes and loss.
- Regenerate or update ``metadata.json`` whenever you change datasets or ``embodiment_tag``.
- Do not mix LIBERO and DROID config templates; ``action_horizon``, ``embodiment_tag``, and multi-view concat logic differ.
