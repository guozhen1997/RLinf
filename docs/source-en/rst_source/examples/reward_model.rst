Reward Model Workflow
=====================

This document provides a comprehensive guide for the reward model workflow, including data collection, reward model training, and using the trained reward model to replace environment success signals for training RL policies.

Algorithm Support
-----------------

**This workflow supports PPO (Proximal Policy Optimization)** algorithm:

- ✅ **PPO** (Proximal Policy Optimization) - Use config: ``maniskill_ppo_cnn.yaml``

**Key Benefits**:

- Simple and unified workflow
- All scripts and configs are located in ``examples/embodiment/`` directory

Overview
--------

The reward model workflow consists of three main stages:

1. **Data Collection**: Collect trajectories using an initial RL policy (PPO), storing images and success labels
2. **Reward Model Training**: Train a binary classifier to predict success from single frames (algorithm-agnostic)
3. **RL Policy Training**: Use the trained reward model to replace environment success signals for PPO training

**Key Points**:

- The workflow uses PPO algorithm for RL policy training
- Uses unified ``RewardWorker`` for reward computation

Key Files and Components
-------------------------

Data Collection
~~~~~~~~~~~~~~~

**``rlinf/models/reward_model/reward_data_collector.py``**

**Purpose**: Core data collection implementation

- **Main Class**: ``RewardDataCollector``
- **Responsibilities**:
  - Maintains separate trajectory buffers for each parallel environment
  - Collects observations (images), ``success_frame`` labels, and ``success_once`` flags
  - Saves complete trajectories as ``.npy`` files
  - Classifies trajectories as positive/negative based on success criteria
  - Ensures data format consistency (``[C, H, W]`` images, normalized to ``[0, 1]``)

**Key Methods**:

- ``add_env_output(env_output, env_info, is_last_step)``: Add environment output to buffers
- ``save_trajectory(env_idx)``: Save completed trajectory to disk
- ``_extract_image(image, key_offset)``: Extract and format images from observations

**``rlinf/workers/env/env_worker.py``**

**Purpose**: Environment worker that integrates data collection

- **Integration Point**: Initializes ``RewardDataCollector`` in ``__init__`` if ``cfg.reward.collect_data=True``
- **Integration Method**: Calls ``data_collector.add_env_output()`` after each environment interaction in ``interact()`` method
- **Key Logic**: Extracts ``success_frame`` from ``env_output`` and ``success_once`` from ``env_info``

Reward Model Training
~~~~~~~~~~~~~~~~~~~~~

**``examples/embodiment/train_reward_model.py``**

**Purpose**: Training script for reward model

- **Main Function**: ``main(cfg: DictConfig)`` (uses Hydra decorator)
- **Configuration**: Uses Hydra and OmegaConf, config file at ``examples/embodiment/config/train_reward_model.yaml``
- **Responsibilities**:
  - Loads trajectory data from ``positive_dir`` and ``negative_dir`` (from config)
  - Creates ``RewardDataset`` for frame-level training
  - Trains ``BinaryRewardClassifier`` with BCE loss
  - Saves best model checkpoint
  - Optionally visualizes positive samples

**``rlinf/models/reward_model/reward_classifier.py``**

**Purpose**: Reward model architecture definition

- **Main Class**: ``BinaryRewardClassifier``
- **Architecture**:
  - **Encoder**: ``ResNetEncoderWrapper`` (ResNet10 backbone)
  - **Classifier Head**: MLP (Linear → LayerNorm → Tanh → Linear → Sigmoid)
  - **Input**: Single frame image ``[B, C, H, W]``
  - **Output**: Binary logit (success probability via sigmoid)

Reward Model Usage
~~~~~~~~~~~~~~~~~~

**``rlinf/workers/reward/reward_worker.py``**

**Purpose**: Unified reward worker that supports both text-based reasoning tasks and embodied tasks

- **Main Class**: ``RewardWorker``
- **Task Type Detection**: Automatically detects task type (embodied vs text-based) based on config
- **Responsibilities**:
  - For embodied tasks: Loads ``BinaryRewardClassifier`` from checkpoint and computes frame-based rewards
  - For text-based tasks: Supports rule-based rewards (reward model for text tasks not yet implemented)
  - Handles batch processing across parallel environments
  - Replaces environment success signals with model predictions for embodied tasks

Stage 1: Data Collection
-------------------------

Configuration
~~~~~~~~~~~~~

The reward model workflow uses PPO algorithm. Use the config file:

- **PPO**: ``examples/embodiment/config/maniskill_ppo_cnn.yaml``

Example configuration:

.. code-block:: yaml

   reward:
     group_name: "RewardGroup"
     use_reward_model: False  # Disabled during data collection
     collect_data: True  # Enable data collection
     reward_model:
       # These settings are not used during data collection, but should match actor settings
       image_keys: ["base_camera"]
       image_size: [3, 64, 64]
     data_collection:
       # Directory to save positive trajectories (success_once=1 or any success_frame>=0.5)
       positive_dir: "./reward_data/positive"
       # Directory to save negative trajectories (success_once=0 and all success_frame<0.5)
       negative_dir: "./reward_data/negative"
       # Image keys to collect (must match actor.model.image_keys)
       image_keys: ["base_camera"]
       # Maximum number of trajectories to save (None = unlimited)
       max_positive_trajectories: 500
       max_negative_trajectories: 500

Startup Command
~~~~~~~~~~~~~~~

.. code-block:: bash

   # For PPO
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn

   # Or directly with python
   export EMBODIED_PATH="/path/to/RLinf/examples/embodiment"
   export PYTHONPATH="/path/to/RLinf:$PYTHONPATH"
   python examples/embodiment/train_embodied_agent.py \
       --config-path examples/embodiment/config/ \
       --config-name maniskill_ppo_cnn \
       reward.collect_data=True \
       reward.use_reward_model=False

Data Format
~~~~~~~~~~~

Each trajectory is saved as a ``.npy`` file containing a dictionary:

.. code-block:: python

   {
       'images': {
           'base_camera': np.array([T, C, H, W], dtype=np.float32)  # T frames, normalized [0, 1]
       },
       'labels': np.array([T], dtype=np.float32)  # T success_frame labels (0 or 1)
   }

**File Naming**: ``{counter:06d}.npy`` (e.g., ``000000.npy``, ``000001.npy``, ...)

**Save Locations**:

- Positive trajectories: ``{positive_dir}/000000.npy``, ``{positive_dir}/000001.npy``, ...
- Negative trajectories: ``{negative_dir}/000000.npy``, ``{negative_dir}/000001.npy``, ...

Trajectory Classification Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A trajectory is saved to ``positive_dir`` if:

- ``success_once=True`` **OR**
- Any frame has ``success_frame >= 0.5``

Otherwise, it is saved to ``negative_dir``.

Stage 2: Reward Model Training
--------------------------------

Prerequisites
~~~~~~~~~~~~~

1. **Pretrained ResNet10 Encoder** (Optional but recommended):
   - **File**: ``resnet10_pretrained.pt``
   - **Location**: Should be placed in the project root or specified via ``pretrained_encoder_path`` in config
   - **Purpose**: Provides pretrained visual features for better initialization
   - **Note**: If not provided (set to ``null``), the encoder will be randomly initialized

2. **Collected Data**:
   - Positive trajectories in ``positive_dir``
   - Negative trajectories in ``negative_dir``

Configuration
~~~~~~~~~~~~~

The training script uses Hydra and OmegaConf for configuration management. The default configuration file is located at ``examples/embodiment/config/train_reward_model.yaml``.

Default Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   defaults:
     - override hydra/job_logging: stdout

   hydra:
     run:
       dir: .
     output_subdir: null

   # Data paths
   positive_dir: ./reward_data/positive
   negative_dir: ./reward_data/negative

   # Model configuration
   backbone: resnet10  # Currently only resnet10 is supported
   image_key: base_camera
   image_size: [3, 64, 64]  # [C, H, W]
   hidden_dim: 256
   num_spatial_blocks: 8
   pretrained_encoder_path: null  # Path to pretrained encoder weights

   # Training configuration
   batch_size: 32
   epochs: 100
   lr: 1e-4
   num_workers: 4

   # Output configuration
   output_checkpoint: ./checkpoints/reward_model.pt

   # Visualization
   visualize_positive: false
   vis_output_dir: positive_samples_vis

   # Device (set to null for auto-detection)
   device: null  # Will auto-detect: "cuda" if available, else "cpu"

Startup Command
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Using the provided script (modify paths as needed)
   bash examples/embodiment/train_reward_model.sh

   # Or directly with python (from examples/embodiment directory)
   cd examples/embodiment
   python train_reward_model.py

   # Override configuration parameters via command line
   python train_reward_model.py \
       positive_dir=./reward_data/positive \
       negative_dir=./reward_data/negative \
       output_checkpoint=./checkpoints/reward_model.pt \
       pretrained_encoder_path=./resnet10_pretrained.pt \
       image_key=base_camera \
       image_size=[3,64,64] \
       batch_size=128 \
       epochs=30 \
       lr=1e-4 \
       hidden_dim=256 \
       num_spatial_blocks=8 \
       visualize_positive=true \
       vis_output_dir=./positive_samples_vis

Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| Parameter                   | Default                          | Description                                                      | Example                          |
+============================+==================================+==================================================================+==================================+
| ``positive_dir``           | ``./reward_data/positive``       | Directory containing positive ``.npy`` files                    | ``./reward_data/positive``       |
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| ``negative_dir``           | ``./reward_data/negative``       | Directory containing negative ``.npy`` files                    | ``./reward_data/negative``       |
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| ``output_checkpoint``      | ``./checkpoints/reward_model.pt``| Path to save trained model                                    | ``./checkpoints/reward_model.pt``|
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| ``pretrained_encoder_path``| ``null``                         | Path to pretrained ResNet10 encoder                             | ``./resnet10_pretrained.pt``     |
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| ``backbone``               | ``resnet10``                     | Backbone architecture (currently only resnet10)                 | ``resnet10``                     |
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| ``image_key``              | ``base_camera``                  | Image key to use                                                | ``base_camera``                  |
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| ``image_size``             | ``[3, 64, 64]``                  | Image size as ``[C, H, W]``                                     | ``[3, 64, 64]``                  |
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| ``batch_size``             | ``32``                           | Training batch size                                             | ``128``                          |
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| ``epochs``                 | ``100``                          | Number of training epochs                                       | ``30``                           |
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| ``lr``                     | ``1e-4``                         | Learning rate                                                   | ``1e-4``                         |
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| ``hidden_dim``             | ``256``                          | Hidden dimension for classifier                                 | ``256``                          |
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| ``num_spatial_blocks``     | ``8``                            | Number of spatial blocks for pooling                            | ``8``                            |
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| ``num_workers``            | ``4``                            | Number of data loading workers                                   | ``4``                            |
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| ``visualize_positive``     | ``false``                        | Visualize positive samples before training                       | ``true``                         |
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| ``vis_output_dir``         | ``positive_samples_vis``         | Output directory for visualizations                              | ``./show``                       |
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+
| ``device``                 | ``null``                         | Device to train on (auto-detects if null)                       | ``cuda`` or ``cpu``              |
+----------------------------+----------------------------------+------------------------------------------------------------------+----------------------------------+

Training Process
~~~~~~~~~~~~~~~~

1. **Data Loading**:
   - Loads all ``.npy`` files from ``positive_dir`` and ``negative_dir``
   - Expands trajectory-level data to frame-level samples
   - Each sample = ``(trajectory_path, frame_index, label)``
   - Reports statistics:
      - Number of trajectories and frames in each directory
      - Distribution of labels (label=1 vs label=0) in each directory

2. **Model Creation**:
   - Creates ``BinaryRewardClassifier`` with specified parameters
   - If ``pretrained_encoder_path`` is provided in config, loads pretrained ResNet10 weights
   - Encoder weights are frozen by default (``freeze_encoder=True``)

3. **Training Loop**:
   - Uses BCE loss with logits
   - Adam optimizer
   - Saves best model based on training accuracy

Stage 3: Training RL Policy with Reward Model
-----------------------------------------------

Configuration
~~~~~~~~~~~~~

The reward model workflow uses PPO algorithm. Use the config file:

- **PPO**: ``examples/embodiment/config/maniskill_ppo_cnn.yaml``

Example configuration:

.. code-block:: yaml

   reward:
     group_name: "RewardGroup"
     use_reward_model: True  # Enable reward model
     collect_data: False  # Disable data collection during policy training
     reward_model:
       # Path to trained reward model checkpoint
       checkpoint_path: "./checkpoints/reward_model.pt"
       # Image keys to use (must match actor.model.image_keys)
       image_keys: ["base_camera"]
       # Image size [C, H, W] (must match actor.model.image_size)
       image_size: [3, 64, 64]
       # Hidden dimension for classifier (must match training config)
       hidden_dim: 256
       # Number of spatial blocks for pooling (must match training config)
       num_spatial_blocks: 8
       # Path to pretrained ResNet10 encoder weights (optional, used during model creation)
       pretrained_encoder_path: "./resnet10_pretrained.pt"
       # Whether to use pretrained encoder (must match training config)
       use_pretrain: True
       # Whether to freeze encoder weights (typically True)
       freeze_encoder: True
       # Reward type: "binary" (0 or 1) or "continuous" (probability)
       reward_type: "binary"

**Important**: The ``image_keys``, ``image_size``, ``hidden_dim``, ``num_spatial_blocks``, ``use_pretrain``, and ``freeze_encoder`` settings **must match** the settings used during reward model training.

Startup Command
~~~~~~~~~~~~~~~

.. code-block:: bash

   # For PPO
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn

   # Or directly with python
   export EMBODIED_PATH="/path/to/RLinf/examples/embodiment"
   export PYTHONPATH="/path/to/RLinf:$PYTHONPATH"
   python examples/embodiment/train_embodied_agent.py \
       --config-path examples/embodiment/config/ \
       --config-name maniskill_ppo_cnn \
       reward.use_reward_model=True \
       reward.collect_data=False \
       reward.reward_model.checkpoint_path="./checkpoints/reward_model.pt"

Reward Model Loading Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Task Type Detection** (``RewardWorker._is_embodied_task()``):
   - Automatically detects if this is an embodied task based on config
   - For embodied tasks, initializes image-based reward model
   - For text-based tasks, initializes text-based reward handler

2. **Initialization for Embodied Tasks** (``RewardWorker.init_worker()``):
   - Creates ``BinaryRewardClassifier`` with config parameters
   - Loads checkpoint from ``checkpoint_path``
   - Automatically infers ``use_pretrain`` from checkpoint if there's a mismatch
   - Moves model to device (``torch.cuda.current_device()``)
   - Sets model to eval mode

3. **Checkpoint Compatibility**:
   - If ``use_pretrain`` in config doesn't match the training config, the code will automatically infer the correct value from the checkpoint's pooling layer kernel shape
   - Warning messages will be printed if adjustments are made

Reward Computation Process
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Unified Reward Computation** (``RewardWorker.compute_rewards()``):
   - Automatically detects input type (``RolloutResult`` vs ``EmbodiedRolloutResult``)
   - Routes to appropriate processing method based on input type

2. **For Embodied Tasks** (``RewardWorker._compute_embodied_rewards_with_model()``):
   - Receives ``EmbodiedRolloutResult`` from rollout worker
   - Extracts observations (images) from ``transitions`` or ``forward_inputs``
   - Processes each observation:
      - Extracts image using ``_extract_images_from_obs()``
      - Ensures images are in ``[B, C, H, W]`` format
      - Runs forward pass through reward model
      - Applies sigmoid to get success probability
      - Converts to reward based on ``reward_type``:
         - ``"binary"``: ``(probs > 0.5).float()`` → 0 or 1
         - ``"continuous"``: Uses probability directly

3. **Integration with RL Training**:
   - Reward model predictions replace or supplement environment ``success_frame`` signals
   - For PPO: Used for advantage computation and policy updates

Complete Workflow Summary
--------------------------

The complete workflow consists of three stages:

1. **Stage 1: Data Collection** - Collect trajectories using initial RL policy with ``reward.collect_data=True``
2. **Stage 2: Reward Model Training** - Train binary classifier using collected data
3. **Stage 3: RL Policy Training** - Use trained reward model with ``reward.use_reward_model=True``

Complete Workflow Example
-------------------------

.. code-block:: bash

   # Step 1: Collect data
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn \
       reward.collect_data=True \
       reward.use_reward_model=False

   # Step 2: Train reward model
   cd examples/embodiment
   python train_reward_model.py \
       positive_dir=./reward_data/positive \
       negative_dir=./reward_data/negative \
       output_checkpoint=./checkpoints/reward_model.pt

   # Step 3: Train with reward model
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn \
       reward.use_reward_model=True \
       reward.collect_data=False \
       reward.reward_model.checkpoint_path=./checkpoints/reward_model.pt

