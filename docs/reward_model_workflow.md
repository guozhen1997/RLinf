# Reward Model Workflow Documentation

This document provides a comprehensive guide for the reward model workflow, including data collection, reward model training, and using the trained reward model to replace environment success signals for training RL policies. 

## 🎯 Algorithm Support

**This workflow supports both PPO and SAC algorithms** with the same unified approach:

- ✅ **PPO** (Proximal Policy Optimization) - Use config: `maniskill_ppo_cnn.yaml`
- ✅ **SAC** (Soft Actor-Critic) - Use config: `maniskill_sac_cnn.yaml`

**Key Benefits**:
- Same workflow and commands for both algorithms
- Same reward model can be used for both PPO and SAC training
- Only the config file name differs between algorithms
- All scripts and configs are located in `examples/embodiment/` directory

## Table of Contents

1. [Overview](#overview)
2. [Key Files and Components](#key-files-and-components)
3. [Stage 1: Data Collection](#stage-1-data-collection)
4. [Stage 2: Reward Model Training](#stage-2-reward-model-training)
5. [Stage 3: Training RL Policy with Reward Model](#stage-3-training-rl-policy-with-reward-model)
6. [Complete Workflow Summary](#complete-workflow-summary)
7. [Configuration Examples](#configuration-examples)
8. [Common Issues and Solutions](#common-issues-and-solutions)

## Overview

The reward model workflow consists of three main stages and **supports both PPO and SAC algorithms**:

1. **Data Collection**: Collect trajectories using an initial RL policy (PPO or SAC), storing images and success labels
2. **Reward Model Training**: Train a binary classifier to predict success from single frames (algorithm-agnostic)
3. **RL Policy Training**: Use the trained reward model to replace environment success signals (works for both PPO and SAC)

**Key Points**:
- The same reward model can be used for both PPO and SAC training
- The workflow and commands are identical for both algorithms
- Only the config file name differs: `maniskill_ppo_cnn.yaml` vs `maniskill_sac_cnn.yaml`
- Both use the same unified `RewardWorker` for reward computation

## Key Files and Components

### Data Collection

#### `rlinf/models/reward_model/reward_data_collector.py`
**Purpose**: Core data collection implementation
- **Main Class**: `RewardDataCollector`
- **Responsibilities**:
  - Maintains separate trajectory buffers for each parallel environment
  - Collects observations (images), `success_frame` labels, and `success_once` flags
  - Saves complete trajectories as `.npy` files
  - Classifies trajectories as positive/negative based on success criteria
  - Ensures data format consistency (`[C, H, W]` images, normalized to `[0, 1]`)

**Key Methods**:
- `add_env_output(env_output, env_info, is_last_step)`: Add environment output to buffers
- `save_trajectory(env_idx)`: Save completed trajectory to disk
- `_extract_image(image, key_offset)`: Extract and format images from observations

#### `rlinf/workers/env/env_worker.py`
**Purpose**: Environment worker that integrates data collection
- **Integration Point**: Initializes `RewardDataCollector` in `__init__` if `cfg.reward.collect_data=True`
- **Integration Method**: Calls `data_collector.add_env_output()` after each environment interaction in `interact()` method
- **Key Logic**: Extracts `success_frame` from `env_output` and `success_once` from `env_info`

### Reward Model Training

#### `examples/embodiment/train_reward_model.py`
**Purpose**: Training script for reward model
- **Main Function**: `main(cfg: DictConfig)` (uses Hydra decorator)
- **Configuration**: Uses Hydra and OmegaConf, config file at `examples/embodiment/config/train_reward_model.yaml`
- **Responsibilities**:
  - Loads trajectory data from `positive_dir` and `negative_dir` (from config)
  - Creates `RewardDataset` for frame-level training
  - Trains `BinaryRewardClassifier` with BCE loss
  - Saves best model checkpoint
  - Optionally visualizes positive samples

**Key Configuration Parameters**:
- `positive_dir`: Directory containing positive trajectory `.npy` files
- `negative_dir`: Directory containing negative trajectory `.npy` files
- `output_checkpoint`: Path to save trained model (e.g., `./checkpoints/reward_model.pt`)
- `pretrained_encoder_path`: Path to pretrained ResNet10 encoder weights (e.g., `./resnet10_pretrained.pt`)
- `image_key`: Image key to use (default: `base_camera`)
- `image_size`: Image size as `[C, H, W]` (default: `[3, 64, 64]`)
- `batch_size`: Training batch size (default: `32`)
- `epochs`: Number of training epochs (default: `100`)
- `lr`: Learning rate (default: `1e-4`)
- `hidden_dim`: Hidden dimension for classifier (default: `256`)
- `num_spatial_blocks`: Number of spatial blocks for pooling (default: `8`)
- `visualize_positive`: Flag to visualize positive samples (default: `false`)
- `vis_output_dir`: Output directory for visualizations (default: `positive_samples_vis`)
- `device`: Device to train on (default: `null` for auto-detection)

#### `rlinf/models/reward_model/reward_classifier.py`
**Purpose**: Reward model architecture definition
- **Main Class**: `BinaryRewardClassifier`
- **Architecture**:
  - **Encoder**: `ResNetEncoderWrapper` (ResNet10 backbone)
    - Can load pretrained weights from `pretrained_encoder_path`
    - Supports freezing encoder weights (`freeze_encoder=True`)
    - Uses `SpatialLearnedEmbeddings` for pooling
  - **Classifier Head**: MLP (Linear → LayerNorm → Tanh → Linear → Sigmoid)
  - **Input**: Single frame image `[B, C, H, W]`
  - **Output**: Binary logit (success probability via sigmoid)

**Key Parameters**:
- `image_keys`: List of image keys to process
- `image_size`: Image dimensions `[C, H, W]`
- `hidden_dim`: Hidden dimension for classifier
- `num_spatial_blocks`: Number of spatial blocks for pooling
- `pretrained_encoder_path`: Path to pretrained ResNet10 weights
- `use_pretrain`: Whether to use pretrained encoder
- `freeze_encoder`: Whether to freeze encoder weights during training

#### `rlinf/models/embodiment/modules/nature_cnn.py`
**Purpose**: ResNet10 backbone implementation
- **Main Class**: `ResNet10`
- **Pretrained Weights**: Expected at `./resnet10_pretrained.pt` (hardcoded path in `ResNetEncoder`)
- **Note**: When using `ResNetEncoderWrapper` in reward classifier, pretrained weights are loaded from `pretrained_encoder_path` parameter instead

### Reward Model Usage

#### `rlinf/workers/reward/reward_worker.py`
**Purpose**: Unified reward worker that supports both text-based reasoning tasks and embodied tasks
- **Main Class**: `RewardWorker`
- **Task Type Detection**: Automatically detects task type (embodied vs text-based) based on config
  - Checks for `cfg.runner.task_type == "embodied"`
  - Checks for presence of `cfg.env` config
  - Checks if reward_model config contains `image_keys` or `image_size`
- **Responsibilities**:
  - For embodied tasks: Loads `BinaryRewardClassifier` from checkpoint and computes frame-based rewards
  - For text-based tasks: Supports rule-based rewards (reward model for text tasks not yet implemented)
  - Handles batch processing across parallel environments
  - Replaces environment success signals with model predictions for embodied tasks

**Key Methods for Embodied Tasks**:
- `init_worker()`: Initialize and load reward model
  - Automatically detects if this is an embodied task
  - Creates `BinaryRewardClassifier` with config parameters
  - Loads checkpoint from `cfg.reward.reward_model.checkpoint_path`
  - Automatically infers `use_pretrain` from checkpoint if needed
  - Moves model to device and sets to eval mode
- `_compute_embodied_rewards_with_model()`: Compute rewards using model for embodied tasks
  - Extracts images from observations in `EmbodiedRolloutResult`
  - Runs forward pass through model
  - Converts logits to rewards based on `reward_type` (binary/continuous)
- `compute_rewards()`: Unified method that handles both `RolloutResult` and `EmbodiedRolloutResult`
  - Automatically detects input type and routes to appropriate processing

**Key Methods for Text-Based Tasks**:
- `init_worker()`: Initialize rule-based reward or text reward model (if implemented)
- `_compute_rule_based_rewards()`: Compute rewards using rule-based methods
- `compute_batch_rewards_with_model()`: Compute rewards using reward model (not yet implemented)

#### `examples/embodiment/train_embodied_agent.py`
**Purpose**: Main training script for embodied RL agents
- **Integration**: Creates `RewardWorker` if `cfg.reward.use_reward_model=True`
- **Usage**: Can run SAC or PPO training with or without reward model
- **Note**: The unified `RewardWorker` automatically detects embodied tasks and handles them appropriately

#### `rlinf/runners/embodied_runner.py`
**Purpose**: Main runner that orchestrates training
- **Integration**: Initializes reward worker if provided
- **Key Method**: `init_workers()` calls `reward.init_worker()` if reward worker exists

## Stage 1: Data Collection

### Configuration

The reward model workflow works with both PPO and SAC algorithms. Use the appropriate config file:
- **PPO**: `examples/embodiment/config/maniskill_ppo_cnn.yaml`
- **SAC**: `examples/embodiment/config/maniskill_sac_cnn.yaml`

Both configs use the same reward model configuration structure. Example configuration (same structure for both PPO and SAC):

```yaml
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
```

### Startup Command

The same command works for both PPO and SAC - just specify the appropriate config name:

```bash
# For PPO
bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn

# For SAC
bash examples/embodiment/run_embodiment.sh maniskill_sac_cnn

# Or directly with python (PPO example)
export EMBODIED_PATH="/path/to/RLinf/examples/embodiment"
export PYTHONPATH="/path/to/RLinf:$PYTHONPATH"
python examples/embodiment/train_embodied_agent.py \
    --config-path examples/embodiment/config/ \
    --config-name maniskill_ppo_cnn \
    reward.collect_data=True \
    reward.use_reward_model=False

# Or for SAC
python examples/embodiment/train_embodied_agent.py \
    --config-path examples/embodiment/config/ \
    --config-name maniskill_sac_cnn \
    reward.collect_data=True \
    reward.use_reward_model=False
```

### Data Format

Each trajectory is saved as a `.npy` file containing a dictionary:

```python
{
    'images': {
        'base_camera': np.array([T, C, H, W], dtype=np.float32)  # T frames, normalized [0, 1]
    },
    'labels': np.array([T], dtype=np.float32)  # T success_frame labels (0 or 1)
}
```

**File Naming**: `{counter:06d}.npy` (e.g., `000000.npy`, `000001.npy`, ...)

**Save Locations**:
- Positive trajectories: `{positive_dir}/000000.npy`, `{positive_dir}/000001.npy`, ...
- Negative trajectories: `{negative_dir}/000000.npy`, `{negative_dir}/000001.npy`, ...

### Trajectory Classification Logic

A trajectory is saved to `positive_dir` if:
- `success_once=True` **OR**
- Any frame has `success_frame >= 0.5`

Otherwise, it is saved to `negative_dir`.

### Important Notes

1. **Image Format**: Images are automatically converted to `[C, H, W]` format and normalized to `[0, 1]`
2. **Parallel Environments**: Each environment maintains an independent buffer; trajectories are saved when `is_env_done=True`
3. **Data Consistency**: The `image_keys` and `image_size` in `data_collection` must match `actor.model.image_keys` and `actor.model.image_size`

## Stage 2: Reward Model Training

### Prerequisites

1. **Pretrained ResNet10 Encoder** (Optional but recommended):
   - **File**: `resnet10_pretrained.pt`
   - **Location**: Should be placed in the project root or specified via `pretrained_encoder_path` in config
   - **Purpose**: Provides pretrained visual features for better initialization
   - **Note**: If not provided (set to `null`), the encoder will be randomly initialized

2. **Collected Data**:
   - Positive trajectories in `positive_dir`
   - Negative trajectories in `negative_dir`

### Configuration

The training script uses Hydra and OmegaConf for configuration management. The default configuration file is located at `examples/embodiment/config/train_reward_model.yaml`.

#### Default Configuration File

```yaml
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
```

### Startup Command

```bash
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

# Or modify the YAML file directly and run with defaults
python train_reward_model.py
```

### Configuration Parameters

| Parameter | Default | Description | Example |
|-----------|---------|-------------|---------|
| `positive_dir` | `./reward_data/positive` | Directory containing positive `.npy` files | `./reward_data/positive` |
| `negative_dir` | `./reward_data/negative` | Directory containing negative `.npy` files | `./reward_data/negative` |
| `output_checkpoint` | `./checkpoints/reward_model.pt` | Path to save trained model | `./checkpoints/reward_model.pt` |
| `pretrained_encoder_path` | `null` | Path to pretrained ResNet10 encoder | `./resnet10_pretrained.pt` |
| `backbone` | `resnet10` | Backbone architecture (currently only resnet10) | `resnet10` |
| `image_key` | `base_camera` | Image key to use | `base_camera` |
| `image_size` | `[3, 64, 64]` | Image size as `[C, H, W]` | `[3, 64, 64]` |
| `batch_size` | `32` | Training batch size | `128` |
| `epochs` | `100` | Number of training epochs | `30` |
| `lr` | `1e-4` | Learning rate | `1e-4` |
| `hidden_dim` | `256` | Hidden dimension for classifier | `256` |
| `num_spatial_blocks` | `8` | Number of spatial blocks for pooling | `8` |
| `num_workers` | `4` | Number of data loading workers | `4` |
| `visualize_positive` | `false` | Visualize positive samples before training | `true` |  (Optional)
| `vis_output_dir` | `positive_samples_vis` | Output directory for visualizations | `./show` |  (Optional)
| `device` | `null` | Device to train on (auto-detects if null) | `cuda` or `cpu` |

### Training Process

1. **Data Loading**:
   - Loads all `.npy` files from `positive_dir` and `negative_dir`
   - Expands trajectory-level data to frame-level samples
   - Each sample = `(trajectory_path, frame_index, label)`
   - Reports statistics:
     - Number of trajectories and frames in each directory
     - Distribution of labels (label=1 vs label=0) in each directory

2. **Model Creation**:
   - Creates `BinaryRewardClassifier` with specified parameters
   - If `pretrained_encoder_path` is provided in config, loads pretrained ResNet10 weights
   - Encoder weights are frozen by default (`freeze_encoder=True`)

3. **Training Loop**:
   - Uses BCE loss with logits
   - Adam optimizer
   - Saves best model based on training accuracy

4. **Checkpoint Format**:
   ```python
   {
       'epoch': int,
       'model_state_dict': dict,
       'optimizer_state_dict': dict,
       'train_loss': float,
       'train_acc': float
   }
   ```

### Model Save Location

**Default**: Specified by `output_checkpoint` in configuration

**Example Paths**:
- `./checkpoints/reward_model.pt`
- `/path/to/project/toolkits/reward_model/checkpoints/reward_model.pt`

**Note**: The directory will be created automatically if it doesn't exist.

### Visualization (Optional)

If `visualize_positive: true` is set in the configuration:
- All positive samples (label=1) are visualized
- Images are saved to `vis_output_dir`
- Filename format: `{folder}_{traj_id}_frame{frame_idx}.png`
  - Example: `positive_000001_frame049.png`
- Useful for verifying data quality and understanding success patterns

## Stage 3: Training RL Policy with Reward Model

### Configuration

The reward model workflow works with both PPO and SAC algorithms. Use the appropriate config file:
- **PPO**: `examples/embodiment/config/maniskill_ppo_cnn.yaml`
- **SAC**: `examples/embodiment/config/maniskill_sac_cnn.yaml`

Both configs use the same reward model configuration structure. Example configuration (same structure for both PPO and SAC):

```yaml
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
```

**Important**: The `image_keys`, `image_size`, `hidden_dim`, `num_spatial_blocks`, `use_pretrain`, and `freeze_encoder` settings **must match** the settings used during reward model training.

### Startup Command

The same command works for both PPO and SAC - just specify the appropriate config name:

```bash
# For PPO
bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn

# For SAC
bash examples/embodiment/run_embodiment.sh maniskill_sac_cnn

# Or directly with python (PPO example)
export EMBODIED_PATH="/path/to/RLinf/examples/embodiment"
export PYTHONPATH="/path/to/RLinf:$PYTHONPATH"
python examples/embodiment/train_embodied_agent.py \
    --config-path examples/embodiment/config/ \
    --config-name maniskill_ppo_cnn \
    reward.use_reward_model=True \
    reward.collect_data=False \
    reward.reward_model.checkpoint_path="./checkpoints/reward_model.pt"

# Or for SAC
python examples/embodiment/train_embodied_agent.py \
    --config-path examples/embodiment/config/ \
    --config-name maniskill_sac_cnn \
    reward.use_reward_model=True \
    reward.collect_data=False \
    reward.reward_model.checkpoint_path="./checkpoints/reward_model.pt"
```

### Reward Model Loading Process

1. **Task Type Detection** (`RewardWorker._is_embodied_task()`):
   - Automatically detects if this is an embodied task based on config
   - For embodied tasks, initializes image-based reward model
   - For text-based tasks, initializes text-based reward handler

2. **Initialization for Embodied Tasks** (`RewardWorker.init_worker()`):
   - Creates `BinaryRewardClassifier` with config parameters
   - Loads checkpoint from `checkpoint_path`
   - Automatically infers `use_pretrain` from checkpoint if there's a mismatch
   - Moves model to device (`torch.cuda.current_device()`)
   - Sets model to eval mode

3. **Checkpoint Compatibility**:
   - If `use_pretrain` in config doesn't match the training config, the code will automatically infer the correct value from the checkpoint's pooling layer kernel shape
   - Warning messages will be printed if adjustments are made

### Reward Computation Process

1. **Unified Reward Computation** (`RewardWorker.compute_rewards()`):
   - Automatically detects input type (`RolloutResult` vs `EmbodiedRolloutResult`)
   - Routes to appropriate processing method based on input type

2. **For Embodied Tasks** (`RewardWorker._compute_embodied_rewards_with_model()`):
   - Receives `EmbodiedRolloutResult` from rollout worker
   - Extracts observations (images) from `transitions` or `forward_inputs`
   - Processes each observation:
     - Extracts image using `_extract_images_from_obs()`
     - Ensures images are in `[B, C, H, W]` format
     - Runs forward pass through reward model
     - Applies sigmoid to get success probability
     - Converts to reward based on `reward_type`:
       - `"binary"`: `(probs > 0.5).float()` → 0 or 1
       - `"continuous"`: Uses probability directly

3. **Integration with RL Training**:
   - Reward model predictions replace or supplement environment `success_frame` signals
   - For SAC: Used for Q-learning and policy optimization
   - For PPO: Used for advantage computation and policy updates

### Important Notes

1. **Unified RewardWorker**: The `RewardWorker` class now supports both embodied and text-based reasoning tasks. It automatically detects the task type based on configuration and routes to the appropriate processing methods. This unified design ensures backward compatibility with existing text-based tasks while supporting new embodied tasks.

2. **Configuration Consistency**: All reward model parameters (`image_keys`, `image_size`, `hidden_dim`, `num_spatial_blocks`, `use_pretrain`, `freeze_encoder`) must match between training and usage configs

3. **Pretrained Encoder**: The `pretrained_encoder_path` in the config is used during model creation but may not be reloaded if the checkpoint already contains encoder weights. The checkpoint's encoder weights take precedence.

4. **Performance**: Reward model inference adds computational overhead. Consider using `freeze_encoder=True` to reduce memory and computation.

5. **Parallel Environments**: The reward model processes all environments in parallel batches for efficiency.

6. **Task Type Detection**: The worker automatically determines if it's handling an embodied task by checking:
   - `cfg.runner.task_type == "embodied"` 
   - Presence of `cfg.env` configuration
   - Reward model config containing `image_keys` or `image_size`
   If none of these conditions are met, it treats the task as a text-based reasoning task.

## Complete Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Data Collection (Using Environment Success Signals)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  RL Policy Training (Initial Policy)│
        │  - Supports both PPO and SAC        │
        │  - Config: reward.collect_data=True │
        │  - Config: reward.use_reward_model=False │
        │  - Command: run_embodiment.sh [config_name] │
        │    * PPO: run_embodiment.sh maniskill_ppo_cnn │
        │    * SAC: run_embodiment.sh maniskill_sac_cnn │
        │  - RewardDataCollector collects data│
        │  - Saves to positive/negative dirs  │
        └─────────────────────────────────────┘
                              │
                              ▼
        Output: ./reward_data/positive/*.npy
                ./reward_data/negative/*.npy
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Reward Model Training                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  train_reward_model.py               │
        │  - Location: examples/embodiment/    │
        │  - Uses Hydra config: config/train_reward_model.yaml │
        │  - Input: positive_dir, negative_dir (from config) │
        │  - Optional: pretrained_encoder_path (from config) │
        │  - Output: reward_model.pt           │
        │  - Command: python examples/embodiment/train_reward_model.py │
        └─────────────────────────────────────┘
                              │
                              ▼
        Output: ./checkpoints/reward_model.pt
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: RL Policy Training with Reward Model                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  RL Policy Training (SAC/PPO)       │
        │  - Supports both PPO and SAC        │
        │  - Config: reward.use_reward_model=True │
        │  - Config: reward.collect_data=False │
        │  - Config: checkpoint_path set      │
        │  - Command: run_embodiment.sh [config_name] │
        │    * PPO: run_embodiment.sh maniskill_ppo_cnn │
        │    * SAC: run_embodiment.sh maniskill_sac_cnn │
        │  - RewardWorker automatically detects embodied task │
        │  - RewardWorker loads BinaryRewardClassifier │
        │  - Replaces env success signals with model predictions │
        └─────────────────────────────────────┘
```

## Configuration Examples

### Algorithm Support

The reward model workflow works identically for both **PPO** and **SAC** algorithms. The only difference is the config file name:
- **PPO**: `examples/embodiment/config/maniskill_ppo_cnn.yaml`
- **SAC**: `examples/embodiment/config/maniskill_sac_cnn.yaml`

Both configs share the same reward model configuration structure. The examples below use PPO config, but the same structure applies to SAC config.

### Complete Example: `maniskill_ppo_cnn.yaml` (same for `maniskill_sac_cnn.yaml`)

#### Stage 1: Data Collection Config

```yaml
reward:
  group_name: "RewardGroup"
  use_reward_model: False
  collect_data: True
  reward_model:
    image_keys: ["base_camera"]
    image_size: [3, 64, 64]
  data_collection:
    positive_dir: "./reward_data/positive"
    negative_dir: "./reward_data/negative"
    image_keys: ["base_camera"]
    max_positive_trajectories: 500
    max_negative_trajectories: 500
```

#### Stage 2: Reward Model Training (Hydra Configuration)

**Note**: Reward model training is algorithm-agnostic. The same trained model can be used for both PPO and SAC.

```bash
# Using the provided script (modify paths as needed)
bash examples/embodiment/train_reward_model.sh

# Or directly with python (from examples/embodiment directory)
cd examples/embodiment
python train_reward_model.py

# Or override parameters via command line
python train_reward_model.py \
    positive_dir=./reward_data/positive \
    negative_dir=./reward_data/negative \
    output_checkpoint=./checkpoints/reward_model.pt \
    pretrained_encoder_path=./resnet10_pretrained.pt \
    image_key=base_camera \
    image_size=[3,64,64] \
    batch_size=128 \
    epochs=30 \
    hidden_dim=256 \
    num_spatial_blocks=8
```

#### Stage 3: Training with Reward Model Config

**PPO Example** (`maniskill_ppo_cnn.yaml`):

```yaml
reward:
  group_name: "RewardGroup"
  use_reward_model: True
  collect_data: False
  reward_model:
    checkpoint_path: "./checkpoints/reward_model.pt"
    image_keys: ["base_camera"]
    image_size: [3, 64, 64]
    hidden_dim: 256
    num_spatial_blocks: 8
    pretrained_encoder_path: "./resnet10_pretrained.pt"
    use_pretrain: True
    freeze_encoder: True
    reward_type: "binary"
```

**SAC Example** (`maniskill_sac_cnn.yaml`):

The reward model configuration is identical to PPO. The only differences are algorithm-specific parameters:

```yaml
reward:
  group_name: "RewardGroup"
  use_reward_model: True
  collect_data: False
  reward_model:
    checkpoint_path: "./checkpoints/reward_model.pt"
    image_keys: ["base_camera"]
    image_size: [3, 64, 64]
    hidden_dim: 256
    num_spatial_blocks: 8
    pretrained_encoder_path: "./resnet10_pretrained.pt"
    use_pretrain: True
    freeze_encoder: True
    reward_type: "binary"

# SAC-specific algorithm parameters (different from PPO)
algorithm:
  loss_type: embodied_sac
  adv_type: embodied_sac
  gamma: 0.8
  tau: 0.005
  alpha: 0.2
  auto_entropy_tuning: True
  replay_buffer_capacity: 300000
  # ... other SAC-specific settings
```

### Complete Workflow Example (PPO or SAC)

The workflow is identical for both algorithms. Here's a complete example:

```bash
# Step 1: Collect data (works for both PPO and SAC)
# For PPO:
bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn \
    reward.collect_data=True \
    reward.use_reward_model=False

# For SAC:
bash examples/embodiment/run_embodiment.sh maniskill_sac_cnn \
    reward.collect_data=True \
    reward.use_reward_model=False

# Step 2: Train reward model (same for both algorithms)
cd examples/embodiment
python train_reward_model.py \
    positive_dir=./reward_data/positive \
    negative_dir=./reward_data/negative \
    output_checkpoint=./checkpoints/reward_model.pt

# Step 3: Train with reward model
# For PPO:
bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn \
    reward.use_reward_model=True \
    reward.collect_data=False \
    reward.reward_model.checkpoint_path=./checkpoints/reward_model.pt

# For SAC:
bash examples/embodiment/run_embodiment.sh maniskill_sac_cnn \
    reward.use_reward_model=True \
    reward.collect_data=False \
    reward.reward_model.checkpoint_path=./checkpoints/reward_model.pt
```

