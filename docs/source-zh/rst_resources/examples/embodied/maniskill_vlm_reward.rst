ManiSkill PPO（基于 VLM Reward Model）
========================================

本文档给出在 RLinf 框架内使用 **MLP policy + Qwen3-VL reward model** 运行 ManiSkill PPO 训练的完整说明。
主要参考配置为 ``examples/embodiment/config/maniskill_ppo_mlp_qwentrend_reward.yaml``。

主要目标是让训练流程具备以下能力：

1. **状态策略学习**：actor 仍然基于 ``states`` 使用轻量 ``mlp_policy`` 生成动作。
2. **视觉奖励判断**：reward worker 使用 Qwen3-VL 对图像历史片段和任务描述进行判断。
3. **历史片段打分**：通过 ``history_buffer`` 模式为一小段轨迹分配 learned reward。
4. **强化学习优化**：结合 PPO 与 reward worker 输出更新策略。

环境
-----------------------

**ManiSkill3 环境**

- **Environment**：ManiSkill3 仿真平台
- **Task**：以 ``PickCube3View-v1`` 为代表的机械臂操作任务
- **Policy Observation**：``states``
- **Reward Observation**：``main_images``、``extra_view_images``，以及任务描述文本
- **Action Space**：8 维连续动作

**Reward 输入结构**

- **States**：供 ``mlp_policy`` 使用的状态向量
- **Main Images**：供 Qwen3-VL reward worker 使用的主视角历史图像
- **Extra View Images**：与主视角同步的第三人称历史图像
- **Task Descriptions**：任务文本描述
- **History Buffer**：按 ``history_size`` 和 ``input_interval`` 组织的短视频片段

算法
-----------------------------------------

**核心算法组件**

1. **PPO（Proximal Policy Optimization）**

   - 使用 GAE 进行优势估计
   - 使用裁剪目标更新 actor
   - 使用 value loss 优化 critic

2. **MLP 策略网络**

   - actor 仅消费 ``states``
   - 推理与训练开销较低

3. **Qwen3-VL Reward Model**

   - reward worker 使用 ``HistoryVLMRewardModel``
   - 输入为任务文本与短视频历史
   - 输出通过 ``reward_parser`` 转为标量 reward

依赖安装
---------------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~

**选项 1：Docker 镜像**

使用 Docker 镜像运行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

对于 VLM reward 实验，请先通过镜像内置的 `switch_env` 工具切换到带 ManiSkill 依赖的环境，
然后将 Qwen3-VL 所需的 ``transformers`` 和 ``tokenizers`` 升级到对应版本：

.. code:: bash

   source switch_env openvla
   uv pip install --upgrade "transformers>=4.57.1,<=4.57.6" "tokenizers>=0.22,<0.23"

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令
   bash requirements/install.sh embodied --env maniskill_libero --vlm-reward
   source .venv/bin/activate

资源下载
----------------

下载 ManiSkill 资源文件：

.. code:: bash

   cd <path_to_RLinf>/rlinf/envs/maniskill
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets

模型下载
--------------

在开始训练之前，你需要准备 reward worker 所需的基础模型和 LoRA 权重：

.. code:: bash

   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com

   # 下载基础模型
   hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /path/to/Qwen3-VL-4B-Instruct

   # Reward LoRA 权重目录请替换成你自己的路径
   ls /path/to/Qwen3-VL-4B-Instruct_lora

下载完成后，请确保在配置 yaml 中正确指定：

- ``reward.model.model_path``
- ``reward.model.lora_path``

如果还需要准备或微调 reward worker 使用的 Qwen3-VL checkpoint / LoRA，请参考
:doc:`/rst_source/examples/embodied/sft_vlm`。

运行脚本
-------------------

**1. 关键参数配置**

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
            id: PickCube3View-v1
            obs_mode: rgb
         use_3rd_view_as_extra: True

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
         gt_success_bonus: 20.0
         input_builder_name: qwentrend_input_builder
         reward_parser_name: qwentrend_reward_parser
         reward_parser_params:
            positive_reward: 1.0
            negative_reward: -0.2
            unclear_reward: 0.0
            invalid_reward: 0.0

这些参数的作用如下：

- ``component_placement.reward`` 用于放置在线 reward worker。
- ``wrap_obs_mode: simple_prompt`` 会同时暴露 ``states``、``main_images``、``extra_view_images`` 和 ``task_descriptions``。
- ``use_full_state: True`` 保持 actor 继续基于 ``states`` 运行 ``mlp_policy``。
- ``use_3rd_view_as_extra: True`` 会把 ManiSkill 的第三视角相机作为 ``extra_view_images`` 暴露出来。
- ``reward_mode: history_buffer`` 表示 reward worker 消费一段历史片段，而不是只看当前帧。
- ``history_reward_assign: True`` 表示当前窗口的 reward 会回填到窗口覆盖的更早几个 step。

**2. 配置文件**

可以直接参考以下配置文件：

- 主 QwenTrend 示例：``examples/embodiment/config/maniskill_ppo_mlp_qwentrend_reward.yaml``

**3. 启动命令**

选择配置后，运行以下命令开始训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwentrend_reward

可视化与结果
-------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**2. 关键监控指标**

- ``env/success_once``：建议优先关注的成功率指标。
- ``env/reward``：环境原始 step-level reward。
- ``rollout/rewards``：混合后的 rollout reward。
- ``train/actor/policy_loss``：策略优化情况。

在线调用链
----------

VLM reward 的主路径如下：

.. code-block:: text

   train_embodied_agent.py
      -> EmbodiedRunner
      -> EnvWorker(history_buffer)
      -> HistoryManager
      -> EmbodiedRewardWorker
      -> HistoryVLMRewardModel
      -> InputBuilder + Qwen3-VL generate() + RewardParser

具体流程为：

1. ``train_embodied_agent.py`` 会在 ``reward.use_reward_model=True`` 时创建 ``EmbodiedRewardWorker``。
2. ``EmbodiedRunner.run`` 会在 ``global_step >= reward.use_output_step`` 后激活 reward channel。
3. ``EnvWorker.get_reward_model_output`` 会在 ``reward_mode="history_buffer"`` 时把当前观测追加进 ``HistoryManager``。
4. ``HistoryManager.build_history_input`` 负责提取配置中的历史窗口。
5. ``EmbodiedRewardWorker`` 会根据 ``reward.model.model_type="history_vlm"`` 实例化 ``HistoryVLMRewardModel``。
6. ``HistoryVLMRewardModel.compute_reward`` 使用 ``input_builder_name`` 构造多模态输入，调用 ``AutoModelForVision2Seq.generate()``，并用 ``reward_parser_name`` 解析生成文本。
7. ``EnvWorker.compute_bootstrap_rewards`` 会把 reward model 输出写到当前 step；如果 ``history_reward_assign=True``，``EnvWorker.assign_history_reward`` 还会把同一个 reward 回填到当前历史窗口覆盖的更早几个 step。

当前实现说明
------------

- 这些 YAML 在顶层 ``reward`` 段配置了 ``reward_threshold``，但当前 ``history_vlm`` 实现并不会在 reward 推理阶段应用这个阈值。
- ``qwentrend_input_builder`` 会从 ``history_input`` 中同时读取 ``main_images`` 和 ``extra_view_images``，因此历史缓冲里需要同时记录这两个键，才能组成同步双视角输入。
- ``qwentrend_reward_parser`` 会按照 ``positive_reward``、``negative_reward``、``unclear_reward`` 和 ``invalid_reward`` 直接映射成带符号标量 reward，不会把输出截断到 ``[0, 1]``。
- ``gt_success_bonus`` 配置在 ``reward.model`` 下，并在 reward model 内部生效，而不是在 reward worker 前端额外注入。
