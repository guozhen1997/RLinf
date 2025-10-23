RL with RoboTwin Simulator
===========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document provides a comprehensive guide to launching and managing 
**Vision-Language-Action Models (VLAs)** training tasks within the RLinf framework,
focusing on finetuning a VLA model for robotic manipulation in the RoboTwin environment.

The primary objective is to develop a model capable of performing robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions (position, rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy via PPO with environment feedback.

RLinf RoboTwinEnv Environment
-----------------------

**RLinf RoboTwinEnv Environment**

- **Environment**: RLinf framework provides the RoboTwinEnv environment for reinforcement learning training based on the RoboTwin 2.0 simulation platform.
- **Task**: Control a robotic arm to perform various manipulation tasks. RLinf RoboTwinEnv currently supports **20 tasks** (all with dense reward functions implemented), and users can select tasks for training as needed.

  **Place Tasks**
  
  - ``place_empty_cup``: Place an empty cup on a coaster
  - ``place_shoe``: Place a single shoe
  - ``place_dual_shoes``: Place two shoes
  - ``place_fan``: Place a fan
  - ``place_bread_skillet``: Place bread on a skillet
  - ``place_a2b_left``: Place objects from left to right
  - ``place_a2b_right``: Place objects from right to left

  **Pick Tasks**
  
  - ``pick_dual_bottles``: Pick two bottles
  - ``pick_diverse_bottles``: Pick diverse bottles

  **Stack Tasks**
  
  - ``stack_blocks_two``: Stack two blocks
  - ``stack_blocks_three``: Stack three blocks
  - ``stack_bowls_two``: Stack two bowls
  - ``stack_bowls_three``: Stack three bowls

  **Ranking Tasks**
  
  - ``blocks_ranking_rgb``: Rank blocks by RGB color
  - ``blocks_ranking_size``: Rank blocks by size

  **Interaction Tasks**
  
  - ``click_alarmclock``: Click an alarm clock
  - ``click_bell``: Click a bell
  - ``beat_block_hammer``: Beat a block with a hammer
  - ``adjust_bottle``: Adjust bottle position

  .. note::
     More tasks are under continuous development. The RoboTwin platform plans to support over 50 tasks, and dense reward function implementations will gradually extend to all tasks.

- **Observation**: The observation returned by RLinf RoboTwinEnv is a dictionary (dict) containing the following fields:

  - ``images``: Head camera RGB images

    - **Type**: ``torch.Tensor``
    - **Shape**: ``[batch_size, 224, 224, 3]``
    - **Data Type**: ``uint8`` (0-255)
    - **Description**: Head camera images processed with center crop, one image per environment

  - ``wrist_images``: Wrist camera RGB images (optional)
  
    - **Type**: ``torch.Tensor`` or ``None``
    - **Shape**: ``[batch_size, num_wrist_images, 224, 224, 3]`` (if exists)
    - **Data Type**: ``uint8`` (0-255)
    - **Description**: May contain left wrist camera (``left_wrist_image``) and/or right wrist camera (``right_wrist_image``) images, or ``None`` if the task does not require wrist images

  - ``states``: Proprioception information

    - **Type**: ``torch.Tensor``
    - **Shape**: ``[batch_size, 14]``
    - **Data Type**: ``float32``
    - **Description**: Contains end-effector pose information (position and orientation), 14 dimensions total, corresponding to ``proprio_dim=14``

  - ``task_descriptions``: Task description text

    - **Type**: ``List[str]``
    - **Length**: ``batch_size``
    - **Description**: Natural language task descriptions for each environment, e.g., "What action should the robot take to place the empty cup on the coaster?"

- **Action Space**: 14-dimensional continuous action space

  - **Type**: ``torch.Tensor`` or ``numpy.ndarray``
  - **Shape**: ``[batch_size, action_dim]`` or ``[batch_size, horizon, action_dim]``, where ``action_dim=14``
  - **Data Type**: ``float32``
  - **Action Components**:

    - End-effector 3D position control (x, y, z): 3 dimensions
    - 3D rotation control (roll, pitch, yaw): 3 dimensions
    - Gripper control (open/close): 1 dimension
    - Joint position control: 7 dimensions
    - **Total**: 14 dimensions

Dependency Installation
-----------------------

RLinf provides two installation methods: **Docker image** (recommended, simplest) and **manual installation** (using installation scripts).

Method 1: Using Docker Image (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf provides a pre-configured RoboTwin environment Docker image that includes all required dependencies and can be used directly, **skipping all subsequent installation steps**.

.. code-block:: bash

   # Pull the RoboTwin environment image
   docker pull rlinf/robotwin:latest
   
   # Run container (basic usage)
   docker run -it --gpus all rlinf/robotwin:latest
   
   # Run container (mount data directories, recommended)
   docker run -it --gpus all \
     -v /path/to/robotwin_assets:/opt/robotwin_assets \
     -v /path/to/models:/opt/models \
     -v /path/to/results:/opt/results \
     rlinf/robotwin:latest

.. note::
   The Docker image includes:
   
   - RLinf RoboTwinEnv environment
   - ``embodied`` and ``robotwin`` extra dependencies
   - RoboTwin platform-related dependencies
   - Compatibility patches applied
   - Support for OpenVLA, OpenVLA-OFT, and OpenPI models

   **After using the Docker image, you can directly proceed to the** `Model Download`_ **and** `Running Scripts`_ **sections, skipping all subsequent installation steps.**

Method 2: Manual Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to install in a local environment, you can use the following two methods:

**Method 2.1: Using Installation Script (Recommended)**

Use the ``requirements/install.sh`` script with the ``--robotwin`` parameter to install the RoboTwin environment. Replace the first parameter with the corresponding model name (``openvla``, ``openvla-oft``, or ``openpi``) based on the model you want to train:

.. code-block:: bash

   # Create a virtual environment
   uv venv --python=3.11 --name openvla-oft-robotwin
   source ./venv/openvla-oft-robotwin/bin/activate

   # Example with OpenVLA-OFT (for RoboTwin)
   bash requirements/install.sh openvla-oft --robotwin

This script will automatically:

- Install ``embodied`` and ``robotwin`` extra dependencies
- Install RoboTwin platform-related dependencies
- Apply RoboTwin compatibility patches (fixing compatibility issues between sapien and mplib)
- Install dependencies for the corresponding VLA model

**Method 2.2: Fully Manual Installation**

If you want full manual control over the installation process, follow these steps:

.. code-block:: bash

   # Step 1: Create a virtual environment
   uv venv --python=3.11 --name openvla-oft-robotwin
   source ./venv/openvla-oft-robotwin/bin/activate

   # Step 2: Install RLinf base dependencies and RoboTwin extra
   uv pip install -e ".[embodied,robotwin]"
   
   # Step 3: Install system dependencies for embodied environment
   bash requirements/install_embodied_deps.sh
   
   # Step 4: Apply RoboTwin compatibility patches
   bash requirements/patch_sapien_mplib_for_robotwin.sh
   
   # Step 5: Install corresponding dependencies based on the model used (example with OpenVLA-OFT)
   # OpenVLA-OFT:
   uv pip install -r requirements/openvla_oft.txt

.. note::
   **Dependency Conflict Note**: ``mplib==0.2.1`` is required for RoboTwin but conflicts with ManiSkill.
   If you need both ManiSkill and RoboTwin, it is recommended to:
   
   - Use separate virtual environments for each
   - Or install ``embodied`` first, then use the ``robotwin`` extra as needed

.. note::
   **Compatibility Patch Note**: The patch script fixes the following issues:
   
   - Encoding issues in ``sapien/wrapper/urdf_loader.py``
   - Collision detection logic in ``mplib/planner.py``
   
   If using the ``install.sh`` script, patches are automatically applied and do not need to be run manually.

Assets Download
-----------------------

RoboTwin Assets are asset files required by the RoboTwin environment and need to be downloaded from HuggingFace.

.. code-block:: bash

   # 1. Clone RoboTwin repository
   git clone https://github.com/RoboTwin-Platform/RoboTwin.git
   cd RoboTwin
   
   # 2. Download and extract Assets files
   bash script/_download_assets.sh
   
   # 3. Update configuration file paths
   python script/update_embodiment_config_path.py


Model Download
-----------------------

Before starting training, you need to download the corresponding SFT model:

.. code-block:: bash

   # Download OpenVLA-OFT model
   pip install huggingface-hub
   huggingface-cli download RLinf/RLinf-OpenVLAOFT-RoboTwin

After downloading, ensure that the model path is correctly specified in the configuration yaml file (``actor.model.model_dir``).

Running Scripts
-------------------

**1. Key Parameter Configuration**

Using the OpenVLA-OFT model as an example, the following key parameters need to be configured in ``actor.model``:

.. code-block:: yaml

   actor:
     model:
       model_dir: "/path/to/RLinf-OpenVLAOFT-RoboTwin"  # SFT model path
       model_name: "openvla_oft"
       action_dim: 14                                    # RoboTwin action dimension (14D)
       use_proprio: True                                 # Whether to use proprioception information
       proprio_dim: 14                                   # Proprioception dimension (must match action_dim)
       use_film: False                                   # Whether to use FiLM layer
       num_images_in_input: 1                            # Number of input images
       num_action_chunks: 25                             # Number of action chunks
       unnorm_key: "place_empty_cup"                     # Action normalization key (must match the unnorm_key used during SFT training)


**2. Environment Configuration**

In the environment configuration file, the following key parameters need to be set:

.. code-block:: yaml

   env/train: robotwin_single_task
   env/eval: robotwin_single_task
   
   # In env/train/robotwin_single_task.yaml:
   simulator_type: robotwin
   assets_path: "/path/to/robotwin_assets"
   
   task_config:
     task_name: place_empty_cup  # or other task names
     step_lim: 200
     embodiment: [piper, piper, 0.6]
     camera:
       head_camera_type: D435
       wrist_camera_type: D435
       collect_head_camera: true
       collect_wrist_camera: false

**3. Configuration Files**

Supports **OpenVLA-OFT** model with **PPO** algorithm.  
Corresponding configuration files:

- **OpenVLA-OFT + PPO**: ``examples/embodiment/config/robotwin_ppo_openvlaoft.yaml``

**4. Launch Command**

After selecting the configuration, run the following command to start training:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, training OpenVLA-OFT model with PPO in the RoboTwin environment:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh robotwin_ppo_openvlaoft

Visualization and Results
-------------------------

**1. TensorBoard Logs**

.. code-block:: bash

   # Start TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. Video Generation**

Videos from training and evaluation processes are automatically saved. Configuration:

.. code-block:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train  # Training videos
     # or
     video_base_dir: ${runner.logger.log_path}/video/eval   # Evaluation videos


Configuration Details
-----------------------

**Key Configuration Parameters**

1. **Model Configuration**:

   - ``actor.model.model_name: "openvla_oft"``: Use OpenVLA-OFT model
   - ``actor.model.action_dim: 14``: 14-dimensional action space (including proprioception)
   - ``actor.model.use_proprio: True``: Enable proprioception input
   - ``actor.model.proprio_dim: 14``: Proprioception dimension
   - ``actor.model.num_action_chunks: 25``: Number of action chunks

2. **Algorithm Configuration**:

   - ``algorithm.reward_type: chunk_level``: Chunk-level rewards
   - ``algorithm.logprob_type: token_level``: Token-level log probabilities
   - ``algorithm.n_chunk_steps: 8``: Number of steps per chunk

3. **Environment Configuration**:

   - ``env.train.task_config.task_name``: Task name (e.g., ``place_empty_cup``)
   - ``env.train.task_config.embodiment``: Robot configuration
   - ``env.train.task_config.camera``: Camera configuration

For more detailed information about RoboTwin configuration, please refer to the `RoboTwin Configuration Documentation <https://robotwin-platform.github.io/doc/usage/configurations.html>`_.

Important Notes
-----------------------

1. **Resource Paths**: Ensure the ``assets_path`` is correct
2. **Environment Variables**: Ensure the RoboTwin repo path is added to PYTHONPATH, e.g., ``export PYTHONPATH=/opt/robotwin:$PYTHONPATH``
3. **GPU Memory**: The RoboTwin environment may require significant GPU memory, it is recommended to use ``enable_offload: True``
4. **Task Configuration**: Modify parameters in ``task_config`` according to specific tasks

