RL with Behavior Simulator
==========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document provides a comprehensive guide to launching and managing the 
Vision-Language-Action Models (VLAs) training task within the RLinf framework, 
focusing on finetuning a VLA model for robotic manipulation in the Behavior environment. 

The primary objective is to develop a model capable of performing robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions (position, rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy via the PPO with environment feedback.

Environment
-----------------------

**Behavior Environment**

- **Environment**: Behavior simulation benchmark built on top of *OmniGibson* (PhysX physics engine).
- **Task**: Command a dual-arm R1 Pro robot to perform a variety of household manipulation skills (pick-and-place, stacking, opening drawers, spatial rearrangement).
- **Observation**: Multi-camera RGB images captured by robot-mounted sensors:
  - **Head Camera**: ZED camera providing 224×224 RGB images for global scene understanding
  - **Wrist Cameras**: Left and right RealSense cameras providing 224×224 RGB images for precise manipulation
- **Action Space**: Continuous actions for dual-arm manipulation
  - Left arm: 7-dimensional actions (3D position + 3D rotation + gripper control)
  - Right arm: 7-dimensional actions (3D position + 3D rotation + gripper control)
  - Total: 14-dimensional action space for coordinated dual-arm manipulation

**Task Description Format**

.. code-block:: text

   In: What action should the robot take to [task_description]?
   Out: 

**Data Structure**

- **Images**: Multi-camera RGB tensors
  - Head images: ``[batch_size, 3, 224, 224]``
  - Wrist images: ``[batch_size, 2, 3, 224, 224]`` (left and right cameras)
- **Task Descriptions**: Natural-language instructions loaded from JSONL files
- **Actions**: Continuous values for dual-arm control, processed through action tokenization
- **Rewards**: Step-level rewards based on task completion and success metrics
- **Episode Info**: Success rates, episode lengths, and return values tracked per environment

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

**1. Key Parameters Configuration**

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-7
         rollout: 8-15
         actor: 0-15

   rollout:
      pipeline_stage_num: 2

Here you can flexibly configure the GPU count for env, rollout, and actor components.
Using the above configuration, you can achieve pipeline overlap between env and rollout, and sharing with actor.
Additionally, by setting `pipeline_stage_num = 2` in the configuration, you can achieve pipeline overlap between rollout and actor, improving rollout efficiency.

.. code-block:: yaml
   
   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

You can also reconfigure the placement to achieve complete sharing, where env, rollout, and actor components all share all GPUs.

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 8-15

You can also reconfigure the placement to achieve complete separation, where env, rollout, and actor components each use their own GPUs without interference, eliminating the need for offload functionality.

**2. Configuration Files**

We currently support training in multiple environments: **ManiSkill3**, **Behavior**, and **OmniGibson**.

For the **Behavior** environment, we support multiple model architectures with both **PPO** and **GRPO** algorithms:

- **OpenVLA-OFT + PPO**: ``examples/embodiment/config/omnigibson_ppo_openvlaoft.yaml``
- **OpenVLA-OFT + GRPO**: ``examples/embodiment/config/omnigibson_grpo_openvlaoft.yaml``
- **OpenPI + PPO**: ``examples/embodiment/config/omnigibson_ppo_pi0.yaml``

**Environment Configuration Details**:

- **Simulator**: OmniGibson with PhysX physics engine
- **Action Frequency**: 30 Hz (configurable via `action_frequency`)
- **Physics Frequency**: 120 Hz for stable simulation
- **Rendering Frequency**: 30 Hz for visualization
- **Episode Length**: 2000 steps maximum
- **Image Resolution**: 224×224 for all cameras

**3. Launch Commands**

To start training with a chosen configuration, run the following command:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, to train the OpenVLA-OFT model using the PPO algorithm in the Behavior environment, run:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh omnigibson_ppo_openvlaoft


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
         experiment_name: "openvla-behavior"

**5. Environment Metrics Tracking**

The Behavior environment provides comprehensive metrics tracking:

- **Success Metrics**: Per-episode success rates and cumulative success tracking
- **Episode Information**: Episode lengths, returns, and reward statistics
- **Multi-environment Support**: Metrics tracked across multiple parallel environments
- **Real-time Monitoring**: Success rates, failure rates, and performance indicators
- **Video Integration**: Metrics overlaid on generated videos for visual analysis


Behavior Results
~~~~~~~~~~~~~~~~~~~

Furthermore, we trained OpenVLA-OFT in the Behavior environment using the GRPO algorithm. The improvements achieved through our RL fine-tuning are shown below:

.. list-table:: **OpenVLA-OFT model results on Behavior**
   :header-rows: 1

   * - Model
     - `Spatial <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-Behavior-spatial>`_
     - `Goal <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-Behavior-goal>`_
     - `Object <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-Behavior-object>`_
     - `Long <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-Behavior-long>`_
     - Average
   * - OpenVLA-OFT-SFT (one-shot)
     - 56.5%
     - 45.6%
     - 25.6%
     - 9.7%
     - 34.4%
   * - OpenVLA-OFT-RLinf
     - **99.0%**
     - **99.0%**
     - **99.0%**
     - **94.4%**
     - **97.9%**
   * - Improvement
     - +42.5%
     - +53.4%
     - +73.4%
     - +84.7%
     - +63.5%

**Technical Implementation Details**

The Behavior environment implementation includes several key technical features:

- **Multi-camera Processing**: Automatic extraction and processing of images from multiple camera sensors
- **Task Description Loading**: Dynamic loading of task descriptions from JSONL files with task name mapping
- **Action Processing**: Support for both single-step and chunked action execution
- **Metrics Collection**: Comprehensive tracking of success rates, episode lengths, and performance metrics
- **Video Recording**: Real-time video generation with multi-camera layout and metric overlays
- **Environment Management**: Support for parallel environments with individual metric tracking

For the Behavior experiment, we were inspired by 
`SimpleVLA <https://github.com/PRIME-RL/SimpleVLA-RL>`_, 
with only minor modifications. We thank the authors for releasing their open-source code.
