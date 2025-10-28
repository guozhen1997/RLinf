RL with Behavior Simulator
==========================

Behavior 1K is a dataset of 1000 manipulation tasks collected from the real world.

Behavior uses IsaacSim as the simulation platform.

Environment
-----------------------

**Behavior Environment**

- **Environment**: Behavior simulation benchmark built on top of *IsaacSim*.
- **Task**: Command a dual-arm R1 Pro robot to perform a variety of household manipulation skills (pick-and-place, stacking, opening drawers, spatial rearrangement).
- **Observation**: Multi-camera RGB images captured by robot-mounted sensors:
  - **Head Camera**: head camera providing 224×224 RGB images for global scene understanding
  - **Wrist Cameras**: Left and right RealSense cameras providing 224×224 RGB images for precise manipulation
- **Action Space**: 23-dimensional continuous actions (a 3-DOF (x,y,rz) set of joints, 4-DOF torso, x2 7-DOF arm, and x2 1-DOF parallel jaw grippers.)

**Data Structure**

- **Task_descriptions**: select from `behavoir-1k` tasks
- **Images**: Multi-camera RGB tensors
  - Head images: ``[batch_size, 3, 224, 224]``
  - Wrist images: ``[batch_size, 2, 3, 224, 224]`` (left and right cameras)


Algorithm
-----------------------------------------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**

   - Advantage estimation using GAE (Generalized Advantage Estimation)

   - Policy clipping with ratio limits

   - Value function clipping

   - Entropy regularization

2. **GRPO (Group Relative Policy Optimization)**

   - For every state / prompt the policy generates *G* independent actions

   - Compute the advantage of each action by subtracting the group’s mean reward.


3. **Vision-Language-Action Model**

   - OpenVLA architecture with multimodal fusion

   - Action tokenization and de-tokenization

   - Value head for critic function

Running the Script
-------------------
**Installation Steps**

1. **Clone Required Repositories**

   .. code-block:: bash
      git clone https://github.com/StanfordVL/BEHAVIOR-1K.git third_party/BEHAVIOR-1K

2. **Download Assets**

   .. code-block:: bash
      cd third_party/BEHAVIOR-1K
      ./setup.sh --omnigibson --bddl --joylo --dataset

3. **Set Environment Variables and Asset Paths**

   .. code-block:: bash
      export OMNIGIBSON_DATASET_PATH=/path/to/third_party/BEHAVIOR-1K/datasets/behavior-1k-assets/
      export OMNIGIBSON_KEY_PATH=/path/to/third_party/BEHAVIOR-1K/datasets/omnigibson.key
      export OMNIGIBSON_ASSET_PATH=/path/to/third_party/BEHAVIOR-1K/datasets/omnigibson-robot-assets/
      export OMNIGIBSON_DATA_PATH=/path/to/third_party/BEHAVIOR-1K/datasets/
      export OMNIGIBSON_HEADLESS=1

4. **Evluate the Environment**
   .. code-block:: bash

      bash examples/embodiment/eval_embodiment.sh behavior_eval

4. **Train the Environment**
   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh behavior_ppo_openvlaoft


Visualization and Results
-------------------------

**1. TensorBoard Logging**

.. code-block:: bash

   # Start TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. Key Metrics Tracked**

- **Training Metrics**:

  - ``actor/loss``: PPO policy loss
  - ``actor/value_loss``: Value function loss
  - ``actor/entropy``: Policy entropy
  - ``actor/grad_norm``: Gradient norm
  - ``actor/lr``: Learning rate

- **Rollout Metrics**:

  - ``rollout/reward_mean``: Average episode reward
  - ``rollout/reward_std``: Reward standard deviation
  - ``rollout/episode_length``: Average episode length
  - ``rollout/success_rate``: Task completion rate

- **Environment Metrics**:

  - ``env/success_rate``: Success rate across environments
  - ``env/step_reward``: Step-by-step reward
  - ``env/termination_rate``: Episode termination rate

**3. Video Generation**

The Behavior environment supports comprehensive video recording with multi-camera views:

.. code-block:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ./logs/video/train

**Video Features**:
- **Multi-camera Layout**: Combines head camera (448×448) and wrist cameras (224×224 each) in a single video
- **Layout**: Head camera on the right, wrist cameras stacked on the left
- **Resolution**: Final video resolution of 448×672 pixels
- **Format**: MP4 format with RGB encoding
- **Info Overlay**: Task descriptions and success metrics overlaid on video frames

**4. WandB Integration**

.. code-block:: yaml

   trainer:
     logger:
       wandb:
         enable: True
         project_name: "RLinf"
         experiment_name: "openvlaoft-Behavior"

**5. Environment Metrics Tracking**

The Behavior environment provides comprehensive metrics tracking:

- **Success Metrics**: Per-episode success rates and cumulative success tracking
- **Episode Information**: Episode lengths, returns, and reward statistics
- **Multi-environment Support**: Metrics tracked across multiple parallel environments
- **Real-time Monitoring**: Success rates, failure rates, and performance indicators
- **Video Integration**: Metrics overlaid on generated videos for visual analysis


**Technical Implementation Details**

The Behavior environment implementation includes several key technical features:

- **Multi-camera Processing**: Automatic extraction and processing of images from multiple camera sensors
- **Task Description Loading**: Dynamic loading of task descriptions from JSONL files with task name mapping
- **Action Processing**: Support for both single-step and chunked action execution
- **Metrics Collection**: Comprehensive tracking of success rates, episode lengths, and performance metrics
- **Video Recording**: Real-time video generation with multi-camera layout and metric overlays
- **Environment Management**: Support for parallel environments with individual metric tracking

For the Behavior experiment, we were inspired by 
`https://github.com/StanfordVL/b1k-baselines.git`, 
with only minor modifications. We thank the authors for releasing their open-source code.
