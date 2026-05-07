RL with PolaRiS Simulation Platform
=======================================================

This document provides a complete guide for using the **π05 (OpenPI)** model to perform PPO reinforcement learning on the `PolaRiS <https://github.com/arhanjain/polaris>`_ simulation platform within the RLinf framework.

Environment
-----------

**PolaRiS (Policy Learning and Benchmarking in Realistic Simulated Environments)**

PolaRiS is a high-fidelity robotics simulation platform based on Isaac Sim and Gaussian Splatting rendering. It supports various desktop manipulation tasks and provides realistic visual rendering effects.

- **Simulation Platform**: Based on NVIDIA Isaac Sim
- **Rendering**: Real-time Gaussian Splatting, supporting switching between high-quality (expensive) and fast rendering modes.
- **Observation Space**:
  - External camera (desktop view) RGB image (224×224)
  - Wrist camera RGB image (224×224)
  - Robot proprioceptive state: 7-dim joint positions + 1-dim gripper position (8 dims total)
- **Action Space**: 8-dim continuous action
  - 7-dim joint velocity control
  - 1-dim gripper position control
- **Tasks**: Supports various desktop manipulation tasks, such as:
  - TapeIntoContainer
  - PanClean
  - BlockStackKitchen
  - FoodBussing
  - MoveLatteCup
  - OrganizeTools
- **Episode Length**: Default 30 seconds (15Hz sampling rate = 450 steps)

Algorithm
---------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**
   - Uses GAE (Generalized Advantage Estimation) for advantage estimation
   - Ratio-based policy clipping
   - Value function clipping
   - Entropy regularization

2. **π05 Flow Matching Policy**
   - Based on the OpenPI π05 architecture
   - Flow Matching for action generation (SDE sampling mode)
   - Supports a Value Head for Critic estimation
   - Action Chunking: Generates multiple action steps at once (default 15 steps) and executes them in an open loop.

3. **DROID Data Format**
   - Uses observation key mapping from the DROID dataset format.
   - State encoding: joint positions (7-dim) + gripper position (1-dim)
   - Image encoding: external camera left image + wrist camera left image

Dependency Installation
-----------------------

1. Clone the RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env polaris
   source .venv/bin/activate

Alternatively, you can clone the PolaRiS repository locally first, and then run the script:

.. code:: bash

   git clone --recursive git@github.com:arhanjain/polaris.git
   export POLARIS_PATH=/path/to/polaris
   bash requirements/install.sh embodied --model openpi --env polaris
   source .venv/bin/activate

Dataset Download
----------------

PolaRiS requires the **PolaRiS-Hub** dataset, which contains scene USD files and initial condition configurations.

Please download the dataset from `PolaRiS-Hub <https://github.com/arhanjain/polaris-hub>`_ and place it in a local directory:

.. code:: bash

   # Clone the PolaRiS-Hub dataset
   uvx hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir ./PolaRiS-Hub

After downloading, set the following environment variable to the dataset path:

.. code:: bash

   export POLARIS_DATA_PATH=/path/to/PolaRiS-Hub

Alternatively, you can modify ``env.train.init_params.dataset_path`` and ``env.eval.init_params.dataset_path`` in the configuration YAML file.

Model Download and Conversion
-----------------------------

Before starting the training, you need to download the pre-trained JAX format model and convert it to PyTorch format.

PolaRiS provides several model variants trained on the DROID dataset, stored in Google Cloud Storage (GCS).

**1. Download JAX Checkpoint**

.. code:: bash

   # π0.5 Polaris (Recommended)
   gsutil -m cp -r gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris /path/to/checkpoints/

   # Other available models:
   # π0 Polaris
   # gsutil -m cp -r gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris /path/to/checkpoints/
   # π0 Polaris (100k)
   # gsutil -m cp -r gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_100k_polaris /path/to/checkpoints/
   # π0 Fast Polaris (Note: the current conversion script does not support this model, see explanation below)
   # gsutil -m cp -r gs://openpi-assets/checkpoints/polaris/pi0_fast_droid_jointpos_polaris /path/to/checkpoints/

.. note::

   If you do not need the PolaRiS co-training fine-tuned version, you can also use the DROID Base models:

   .. code:: bash

      # π0.5 Base
      gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_droid_jointpos /path/to/checkpoints/
      # π0 Base
      # gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_droid_jointpos /path/to/checkpoints/
      # π0 Base 100k
      # gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_droid_jointpos_100k /path/to/checkpoints/
      # π0 Fast Base
      # gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_fast_droid_jointpos /path/to/checkpoints/

**2. Convert to PyTorch Format**

The downloaded JAX checkpoint needs to be converted to PyTorch format to be used in RLinf:

.. code:: bash

   cd path/to/polaris/third_party/openpi
   GIT_LFS_SKIP_SMUDGE=1 uv sync
   GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
   source .venv/bin/activate

   # π0.5 Polaris → PyTorch
   python /path/to/polaris/third_party/openpi/examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir /path/to/checkpoints/pi05_droid_jointpos_polaris \
       --config_name pi05_droid_jointpos_polaris \
       --output_path /path/to/pytorch/pi05_droid_jointpos_polaris

   # π0 Polaris → PyTorch
   # python /paht/to/polaris/third_party/openpi/examples/convert_jax_model_to_pytorch.py \
   #     --checkpoint_dir /path/to/checkpoints/pi0_droid_jointpos_polaris \
   #     --config_name pi0_droid_jointpos_polaris \
   #     --output_path /path/to/pytorch/pi0_droid_jointpos_polaris

Here, ``--config_name`` needs to be the **original OpenPI config name** (not the ``actor.model.openpi.config_name`` from the RLinf YAML). The mapping is as follows:

.. list-table:: **Model Checkpoint to Conversion config_name Mapping**
   :header-rows: 1
   :widths: 35 35 30

   * - Model
     - ``--config_name`` (for conversion script)
     - RLinf YAML ``config_name``
   * - π0.5 Polaris
     - ``pi05_droid_jointpos_polaris``
     - ``pi05_droid_polaris``
   * - π0 Polaris
     - ``pi0_droid_jointpos_polaris``
     - ``pi0_droid_polaris``
   * - π0 Polaris (100k)
     - ``pi0_droid_jointpos_100k_polaris``
     - ``pi0_droid_polaris``
   * - π0.5 Base
     - ``pi05_droid``
     - ``pi05_droid_polaris``
   * - π0 Base
     - ``pi0_droid``
     - ``pi0_droid_polaris``
   * - π0 Base (100k)
     - ``pi0_droid_jointpos_100k``
     - ``pi0_droid_polaris``

.. note::

   **π0 Fast Polaris** (``pi0_fast_droid_jointpos_polaris``) uses ``Pi0FASTConfig`` instead of ``Pi0Config``.
   The current conversion script ``convert_jax_model_to_pytorch.py`` does not support this model type.

**3. Configure Model Path**

After conversion, set the PyTorch model path in the YAML configuration file:

.. code-block:: yaml

   rollout:
     model:
       model_path: "/path/to/pytorch/pi05_droid_jointpos_polaris"
   actor:
     model:
       model_path: "/path/to/pytorch/pi05_droid_jointpos_polaris"

Running the Script
------------------

**1. Configuration Files**

PolaRiS currently supports the following training configurations:

- **PPO Training**: ``examples/embodiment/config/polaris_train_ppo_openpi.yaml``
- **Evaluation**: ``examples/embodiment/config/polaris_eval_openpi.yaml``

Each task has an independent environment configuration file located under ``examples/embodiment/config/env/``:

- ``polaris_droid_tapeintocontainer.yaml``
- ``polaris_droid_panclean.yaml``
- ``polaris_droid_blockstackkitchen.yaml``
- ``polaris_droid_foodbussing.yaml``
- ``polaris_droid_movelattecup.yaml``
- ``polaris_droid_organizetools.yaml``

**2. Key Parameter Configuration**

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       actor,rollout: 0
       env: 0,1

You can flexibly configure the GPU allocation for the Actor, Rollout, and Env components.

- **Actor (Training)**: Occupies the most VRAM (weights + gradients + optimizer). It is recommended to place it on the card with the most available memory.
- **Rollout (Inference)**: Only requires model weights and KV Cache.
- **Env (Environment)**: Can share a card with Rollout. The PolaRiS environment requires a GPU for Gaussian Splatting rendering.

.. code-block:: yaml

   actor:
     model:
       num_action_chunks: 15
       action_dim: 8
       openpi:
         config_name: "pi05_droid_polaris"
         num_images_in_input: 2

- ``num_action_chunks: 15``: The model generates 15 action steps at a time.
- ``action_dim: 8``: 7-dim joint velocity + 1-dim gripper position.
- ``config_name: "pi05_droid_polaris"``: Use the PolaRiS configuration with DROID data format.
- ``num_images_in_input: 2``: External camera + wrist camera, total 2 images.

**3. Environment Parameters**

.. code-block:: yaml

   env:
     train:
       init_params:
         open_loop_horizon: 15

``open_loop_horizon`` controls the frequency of high-quality Gaussian Splatting rendering. During the execution of an action chunk, high-quality rendering is performed every ``open_loop_horizon`` steps, while intermediate steps use low-quality rendering to speed up the simulation.

**4. Start Training**

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh polaris_train_ppo_openpi

.. note::

   If you have hardcoded ``POLARIS_DATA_PATH`` in your configuration file, please ensure the path is correct.
   You can also set the environment variable before running:

   .. code-block:: bash

      export POLARIS_DATA_PATH=/path/to/PolaRiS-Hub

**5. Start Evaluation**

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh polaris_eval_openpi

Visualization and Results
-------------------------

**1. TensorBoard Logs**

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**2. Key Monitoring Metrics**

- **Environment Metrics**:

  - ``env/success_once``: Task success rate. It is recommended to use this metric to monitor training effectiveness.
  - ``env/return``: Total return per episode.
  - ``env/episode_len``: Actual number of steps per episode.

- **Training Metrics**:

  - ``train/actor/policy_loss``: PPO policy loss.
  - ``train/critic/value_loss``: Value function loss.
  - ``train/actor/approx_kl``: Approximate KL divergence, for monitoring the magnitude of policy updates.

- **Rollout Metrics**:

  - ``rollout/rewards``: Step-wise rewards.
  - ``rollout/advantages_mean``: Mean of the advantage function.

