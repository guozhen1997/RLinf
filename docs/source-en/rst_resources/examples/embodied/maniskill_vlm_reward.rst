ManiSkill PPO with VLM Reward Model
===================================

This document provides a complete guide for running ManiSkill PPO in RLinf with an **MLP policy + Qwen3-VL reward model**.
The main reference config is ``examples/embodiment/config/maniskill_ppo_mlp_qwen3vl4b_robochallenge_reward.yaml``.

The main goals of this setup are:

1. **State-based policy learning**: the actor still uses a lightweight ``mlp_policy`` over ``states``.
2. **Visual reward judgement**: the reward worker uses Qwen3-VL to judge image history together with task text.
3. **History-based scoring**: learned rewards are assigned over a short trajectory segment through ``history_buffer``.
4. **RL optimization**: PPO updates the policy with reward-worker outputs.

Environment
-----------

**ManiSkill3 Environment**

- **Environment**: ManiSkill3 simulation platform
- **Task**: robotic manipulation tasks such as ``PickCube-v1``
- **Policy Observation**: ``states``
- **Reward Observation**: ``main_images`` together with task text
- **Action Space**: 7-dimensional continuous actions

**Reward Input Structure**

- **States**: state vectors for ``mlp_policy``
- **Main Images**: main-view image history for the Qwen3-VL reward worker
- **Task Descriptions**: task text descriptions
- **History Buffer**: short video segments organized by ``history_size`` and ``input_interval``

Algorithm
-----------------------------------------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**

   - GAE-based advantage estimation
   - Clipped policy update
   - Critic optimization

2. **MLP Policy**

   - the actor only consumes ``states``
   - policy training stays lightweight

3. **Qwen3-VL Reward Model**

   - the reward worker uses ``HistoryVLMRewardModel``
   - inputs are task text plus short video history
   - outputs are parsed into scalar rewards by a reward parser

Dependency Installation
-----------------------

1. Clone RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker Image**

Use Docker image for the experiment.

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

For VLM reward experiments, first switch to an image-bundled environment with ManiSkill dependencies,
then upgrade ``transformers`` and ``tokenizers`` to the Qwen3-VL-compatible versions:

.. code:: bash

   source switch_env openvla
   uv pip install --upgrade "transformers>=4.57.1,<=4.57.6" "tokenizers>=0.22,<0.23"

**Option 2: Custom Environment**

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.
   bash requirements/install.sh embodied --env maniskill_libero --vlm-reward
   source .venv/bin/activate

Assets Download
----------------

Download the ManiSkill assets by running the following command:

.. code:: bash

   cd <path_to_RLinf>/rlinf/envs/maniskill
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets

Model Download
--------------

Before starting training, prepare the base model and LoRA weights used by the reward worker:

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com

   # Download the base model
   hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /path/to/Qwen3-VL-4B-Instruct

   # Replace this with your own reward-model LoRA directory
   ls /path/to/Qwen3-VL-4B-Instruct_lora

After downloading, make sure the config yaml correctly sets:

- ``reward.model.model_path``
- ``reward.model.lora_path``

If you still need to prepare or fine-tune the Qwen3-VL checkpoint / LoRA used by the reward worker,
please refer to :doc:`/rst_source/examples/embodied/sft_vlm`.

Running the Script
-------------------

**1. Key Parameters Configuration**

.. code-block:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         actor: 0-0
         env: 0-0
         rollout: 0-0
         reward: 0-0

   env:
      train:
         wrap_obs_mode: simple_prompt
         use_full_state: True
         init_params:
            id: PickCube-v1
            obs_mode: rgb

   reward:
      use_reward_model: True
      reward_mode: history_buffer
      history_reward_assign: True
      reward_weight: 1.0
      env_reward_weight: 0.0
      model:
         model_type: history_vlm
         model_path: /path/to/Qwen3-VL-4B-Instruct
         lora_path: /path/to/Qwen3-VL-4B-Instruct_lora
         input_builder_name: simple_robochallenge_input_builder
         reward_parser_name: robochallenge_reward_parser

These parameters matter because:

- ``component_placement.reward`` places the online reward worker.
- ``wrap_obs_mode: simple_prompt`` exposes ``states``, ``main_images``, ``extra_view_images``, and ``task_descriptions`` together.
- ``use_full_state: True`` keeps the actor on ``states`` with ``mlp_policy``.
- ``reward_mode: history_buffer`` means the reward worker scores a short trajectory segment instead of a single frame.
- ``history_reward_assign: True`` back-fills the reward to earlier steps covered by the current history window.

**2. Configuration Files**

You can directly refer to the following configs:

- Main single-view example: ``examples/embodiment/config/maniskill_ppo_mlp_qwen3vl4b_robochallenge_reward.yaml``
- Dual-view variant: ``examples/embodiment/config/maniskill_ppo_mlp_qwen3vl4b_dualview_robochallenge_reward.yaml``
- 8B confidence variant: ``examples/embodiment/config/maniskill_ppo_mlp_qwen3vl8b_confidance_robochallenge_reward.yaml``

**3. Launch Commands**

After choosing a config, start training with:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwen3vl4b_robochallenge_reward

The other two variants can be launched with:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwen3vl4b_dualview_robochallenge_reward
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwen3vl8b_confidance_robochallenge_reward

Visualization and Results
-------------------------

**1. TensorBoard Logging**

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**2. Key Metrics Tracked**

- ``env/success_once``: the most direct success-rate metric to watch.
- ``env/reward``: raw environment step reward.
- ``rollout/rewards``: mixed rollout reward after reward-model integration.
- ``train/actor/policy_loss``: policy optimization status.

Online Call Chain
-----------------

The VLM reward path is:

.. code-block:: text

   train_embodied_agent.py
      -> EmbodiedRunner
      -> EnvWorker(history_buffer)
      -> HistoryManager
      -> EmbodiedRewardWorker
      -> HistoryVLMRewardModel
      -> InputBuilder + Qwen3-VL generate() + RewardParser

The concrete flow is:

1. ``train_embodied_agent.py`` creates ``EmbodiedRewardWorker`` when ``reward.use_reward_model=True``.
2. ``EmbodiedRunner.run`` activates the reward channel once ``global_step >= reward.use_output_step``.
3. ``EnvWorker.get_reward_model_output`` appends observations into ``HistoryManager`` when ``reward_mode="history_buffer"``.
4. ``HistoryManager.build_history_input`` extracts the configured history windows.
5. ``EmbodiedRewardWorker`` instantiates ``HistoryVLMRewardModel`` from ``reward.model.model_type="history_vlm"``.
6. ``HistoryVLMRewardModel.compute_reward`` builds multimodal inputs with the configured ``input_builder_name``, runs ``AutoModelForVision2Seq.generate()``, and parses the generated text with ``reward_parser_name``.
7. ``EnvWorker.compute_bootstrap_rewards`` writes the reward-model output to the current step. If ``history_reward_assign=True``, ``EnvWorker.assign_history_reward`` also back-fills the same reward to earlier steps covered by the current history window.

Current Implementation Notes
----------------------------

- ``reward_threshold`` is configured at the top-level ``reward`` section in these YAML files, but the ``history_vlm`` implementation does not currently apply that threshold during reward inference.
- In the main single-view config, the reward worker only consumes ``main_images`` history. ``extra_view_images`` are still present in env observations, but they are not used by ``simple_robochallenge_input_builder``.
- ``robochallenge_reward_parser`` clamps final rewards into ``[0, 1]``. In practice, this means ``positive`` maps to a positive score while ``negative`` is clipped to ``0`` rather than becoming a signed penalty.
- For the dual-view variant, ``dualview_robochallenge_input_builder`` reads both ``main_images`` and ``extra_view_images`` from ``history_input``. To get true two-view history inputs, both keys need to be recorded in the configured history buffer.
- ``confidence_robochallenge_reward_parser`` also outputs values in ``[0, 1]``. For ``negative`` judgements it returns ``1 - confidence``, so it behaves as a bounded score rather than a signed penalty.
