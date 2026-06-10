Embodied Evaluation
===================

Introduction
------------

RLinf provides a unified embodied evaluation entry point. It runs parallel rollouts in simulation or on real robots and reports task-level metrics such as success rate.

All evaluation code and configs live under ``evaluations/`` at the repository root:

.. code-block:: text

   evaluations/
   ├── eval_embodied_agent.py   # Main evaluation program
   ├── run_eval.sh              # One-click launcher
   ├── libero/                  # LIBERO eval configs
   ├── robotwin/                # RoboTwin eval configs
   ├── behavior/                # BEHAVIOR-1K eval configs
   ├── maniskill/               # ManiSkill OOD eval configs
   ├── realworld/               # Real-robot eval configs
   └── polaris/                 # PolaRiS eval configs

Evaluation is driven by ``EmbodiedEvalRunner``: the environment worker and rollout worker communicate through Channels and run parallel evaluation under ``env.eval``. Metrics such as ``eval/success_once`` and ``eval/return`` are printed to the terminal and written to log files.

Supported Benchmarks
--------------------

The table below lists benchmarks that have example configs under ``evaluations/`` and can be launched directly with ``run_eval.sh``.

.. list-table::
   :header-rows: 1
   :widths: 18 28 34

   * - Benchmark
     - Task / env preset
     - Example config
   * - LIBERO
     - ``libero_spatial``, ``libero_object``, ``libero_goal``, ``libero_10``, ``libero_90``
     - ``libero/libero_spatial_openpi_pi05_eval.yaml``, etc.
   * - RoboTwin
     - ``robotwin_place_empty_cup``, ``robotwin_adjust_bottle``, ``robotwin_place_shoe``, ``robotwin_click_bell``
     - ``robotwin/robotwin_place_empty_cup_openvlaoft_eval.yaml``, etc.
   * - BEHAVIOR-1K
     - ``behavior_r1pro``
     - ``behavior/behavior_openpi_pi05_eval.yaml``
   * - ManiSkill OOD
     - ``maniskill_ood_template`` (out-of-distribution generalization)
     - ``maniskill/maniskill_ood_openvlaoft_eval.yaml``
   * - RealWorld
     - ``realworld_franka_sft_env``, ``realworld_bin_relocation``
     - ``realworld/realworld_eval.yaml``, ``realworld/realworld_pnp_eval.yaml``, ``realworld/realworld_pnp_eval_dreamzero.yaml``
   * - PolaRiS
     - ``polaris_droid_tapeintocontainer``, ``polaris_droid_movelattecup``, etc.
     - ``polaris/polaris_tapeintocontainer_openpi_pi05_eval.yaml``, ``polaris/polaris_movelattecup_openpi_eval.yaml``

**LIBERO variants:** Standard LIBERO, LIBERO-PRO, and LIBERO-PLUS are supported via environment variables (see **One-Click Launch** below).

**Config fallback:** If ``evaluations/<benchmark>/<config>.yaml`` does not exist, ``run_eval.sh`` falls back to ``examples/embodiment/config/`` with the same config name, so training configs can be reused for evaluation.

Supported Models
----------------

Eval configs reference model presets from ``examples/embodiment/config/model/`` via ``defaults``, and override fields such as ``model_path`` under ``rollout.model``. Models with example configs in ``evaluations/`` today:

.. list-table::
   :header-rows: 1
   :widths: 20 18 42

   * - Model
     - ``model_type``
     - Example config
   * - π₀ / π₀.₅ (OpenPI)
     - ``openpi``
     - ``libero_spatial_openpi_pi05_eval``, ``libero_goal_openpi_eval``, ``robotwin_adjust_bottle_openpi_eval``, etc.
   * - OpenVLA-OFT
     - ``openvla_oft``
     - ``libero_10_openvlaoft_eval``, ``robotwin_place_empty_cup_openvlaoft_eval``, ``maniskill_ood_openvlaoft_eval``, etc.
   * - StarVLA
     - ``starvla``
     - ``libero_spatial_starvla_eval``
   * - DreamZero
     - ``dreamzero``
     - ``libero_spatial_dreamzero_eval``
   * - LingBotVLA
     - ``lingbotvla``
     - ``robotwin_click_bell_lingbotvla_eval``, ``robotwin_place_shoe_lingbotvla_eval``

Environment Setup
-----------------

Evaluation shares the same embodied environment installation flow as training. From the repository root:

.. code-block:: bash

   bash requirements/install.sh embodied --model <model> --env <env>
   source .venv/bin/activate

Choose ``<model>`` and ``<env>`` to match your target benchmark:

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Benchmark
     - Recommended ``--model``
     - Recommended ``--env``
   * - LIBERO
     - ``openpi`` / ``openvla-oft`` / ``starvla`` / ``dreamzero``
     - ``maniskill_libero`` or ``libero``
   * - RoboTwin
     - ``openvla-oft`` / ``openpi`` / ``lingbotvla``
     - ``robotwin``
   * - BEHAVIOR-1K
     - ``openpi``
     - ``behavior``
   * - ManiSkill OOD
     - ``openvla-oft``
     - ``maniskill_libero``
   * - RealWorld
     - ``openpi``
     - ``franka``

**Additional environment variables (per benchmark):**

- **LIBERO:** ``export LIBERO_PATH=/path/to/LIBERO``
- **RoboTwin:** ``export ROBOTWIN_PATH=/path/to/RoboTwin``, ``export ROBOT_PLATFORM=ALOHA``
- **BEHAVIOR-1K:** Set ``OMNIGIBSON_DATA_PATH`` and related OmniGibson paths (see :doc:`../examples/embodied/behavior`)
- **DreamZero:** ``export DREAMZERO_PATH=/path/to/DreamZero``

YAML Configuration
------------------

Eval configs are Hydra YAML files under ``evaluations/<benchmark>/``. The core structure (using ``libero_spatial_openpi_pi05_eval.yaml`` as an example):

.. code-block:: yaml

   defaults:
     - env/libero_spatial@env.eval      # Environment preset
     - model/pi0_5@rollout.model        # Model preset
     - override hydra/job_logging: stdout

   hydra:
     searchpath:
       - file://${oc.env:EMBODIED_PATH}/config/

   runner:
     task_type: embodied_eval   # Must be embodied_eval
     only_eval: True            # Evaluation only, no training
     ckpt_path: null            # Optional: load a .pt checkpoint
     logger:
       log_path: "../results"

   cluster:
     component_placement:
       env,rollout: all          # GPU placement for env and rollout

   env:
     eval:
       total_num_envs: 500       # Number of parallel environments
       rollout_epoch: 1          # Number of eval epochs
       max_episode_steps: 240
       auto_reset: True
       is_eval: True
       video_cfg:
         save_video: True

   rollout:
     generation_backend: "huggingface"
     model:
       model_path: "/path/to/model"   # Required: model weights path
       model_type: "openpi"

**Fields you must customize:**

1. ``rollout.model.model_path``: Local model directory or HuggingFace cache path.
2. Resource-related fields under ``env.eval``: ``total_num_envs``, ``max_episode_steps``, ``assets_path`` (RoboTwin), etc.
3. ``cluster.component_placement``: Adjust ``env`` and ``rollout`` placement for your GPUs.
4. **Real-robot eval:** Configure Franka IP and node topology in ``cluster.node_groups`` (see ``realworld/realworld_eval.yaml``).

**Deriving from a training config:** Copy the matching YAML from ``examples/embodiment/config/`` or ``tests/e2e_tests/embodied/``, remove training sections (``algorithm``, ``actor``, etc.), keep ``env.eval`` and ``rollout``, and set ``runner.task_type: embodied_eval`` and ``runner.only_eval: True``.

One-Click Launch
----------------

Activate your virtual environment from the repository root, then use ``evaluations/run_eval.sh``.

**Option 1: Explicit benchmark**

.. code-block:: bash

   source .venv/bin/activate
   bash evaluations/run_eval.sh <benchmark> <config_name> [hydra_overrides...]

Examples:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval
   bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval
   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval

**Option 2: Auto-infer benchmark**

When the config name starts with ``libero_``, ``robotwin_``, ``behavior_``, etc., the benchmark can be omitted:

.. code-block:: bash

   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval

**Option 3: Hydra overrides on the command line**

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model/RLinf-Pi05-SFT \
     env.eval.total_num_envs=64 \
     runner.ckpt_path=/path/to/checkpoint.pt

**LIBERO-PRO / LIBERO-PLUS:**

.. code-block:: bash

   export LIBERO_TYPE=pro
   export LIBERO_PERTURBATION=all
   bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval

   export LIBERO_TYPE=plus
   export LIBERO_SUFFIX=all
   bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval

**RoboTwin:**

.. code-block:: bash

   export ROBOTWIN_PATH=/path/to/repo/RoboTwin
   export ROBOT_PLATFORM=ALOHA
   bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval

**ManiSkill OOD batch evaluation:**

The ``mani-ood`` mode runs evaluation across multiple OOD scenes sequentially. Set these environment variables first:

.. code-block:: bash

   export EVAL_NAME=my_ood_eval
   export CKPT_PATH=/path/to/checkpoint.pt
   export TOTAL_NUM_ENVS=16
   export EVAL_ROLLOUT_EPOCH=1
   bash evaluations/run_eval.sh mani-ood maniskill_ood_openvlaoft_eval

Logs and Metrics
----------------

- Default log directory: ``logs/<timestamp>-<config_name>/eval_embodiment.log``
- ManiSkill OOD: ``logs/eval/<EVAL_NAME>/<timestamp>-<env_id>-<obj_set>/run_ppo.log``
- Terminal metrics include ``eval/success_once``, ``eval/return``, etc.
- When ``env.eval.video_cfg.save_video: True``, videos are saved under ``<log_path>/video/eval/``

Related Documentation
---------------------

- Per-benchmark setup and training examples: :doc:`../examples/embodied/index`
- Installation details: :doc:`installation`
- Model-specific standalone eval scripts (outside the unified entry): ``toolkits/standalone_eval_scripts/``
