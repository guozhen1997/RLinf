RL Token：借助视觉-语言-动作模型启动在线强化学习
================================================

**RL Token: Bootstrapping Online RL with Vision-Language-Action Models** 在冻结的 VLA 特征模型之上训练一个轻量级强化学习策略。在 RLinf 的配置和代码中，这套流程简称为 **RLT**。整个流程分为两个阶段：

1. 在示范数据上联合训练 VLA 检查点和 RLT token transformer。
2. 冻结第一阶段得到的特征模型，用提取出的 RLT 状态训练一个轻量级 off-policy actor-critic。

当前仓库中的示例配置面向 Franka peg insertion 和 ManiSkill
``PegInsertionSideWideClearance-v1`` joint-control 仿真。pipeline 本身不绑定
具体任务；只要示范数据、环境配置、动作维度、状态语义和 OpenPI dataconfig 对齐，
就可以复用相同的两阶段结构。

概览
----

RLT 将表示学习和在线 RL 控制拆开。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Stage 1
      :text-align: center

      VLA SFT + RLT token transformer

   .. grid-item-card:: Stage 2
      :text-align: center

      轻量级 off-policy actor-critic

   .. grid-item-card:: 状态
      :text-align: center

      ``z_rl`` + proprio + reference chunk

   .. grid-item-card:: 部署
      :text-align: center

      Franka 真机 / ManiSkill 仿真

| **你将完成：** 准备示范数据 -> 训练 Stage 1 -> 在 Stage 2 中加载 Stage 1 检查点 -> 启动 actor-critic 训练 -> 观察 replay buffer 与任务成功率指标。
| **前置条件：** 安装 OpenPI π₀.₅ checkpoint，并准备 :doc:`Franka 真机环境 <../embodied/franka>` 或 :doc:`Maniskill 仿真环境 <../embodied/maniskill>`。

提供的配置文件
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - 阶段
     - 配置
     - 作用
   * - Franka Stage 1
     - ``examples/sft/config/rlt_sft_openpi_pi05.yaml``
     - 在 Franka 示范数据上联合 SFT π₀.₅ 和 RLT token transformer。
   * - Franka Stage 2
     - ``examples/embodiment/config/rlt_stage2_ac_mlp.yaml``
     - 使用冻结的 Stage 1 特征模型，在真机上训练 RLT actor-critic。
   * - ManiSkill SFT
     - ``examples/sft/config/rlt_maniskill_joint_pi05_sft.yaml``
     - 训练 joint-control OpenPI 基座，作为 ManiSkill RLT Stage 1 的起点。
   * - ManiSkill Stage 1
     - ``examples/sft/config/rlt_stage1_maniskill_joint.yaml``
     - 在 ManiSkill joint 数据上训练 RLT token transformer。
   * - ManiSkill Stage 2
     - ``examples/embodiment/config/rlt_stage2_maniskill_joint_ac.yaml``
     - 使用自动 ``rlt_policy_switch`` 和 transition replay 训练仿真 RLT actor-critic。
   * - Stage 2 模型
     - ``examples/embodiment/config/model/rlt_mlp_policy.yaml``
     - 定义 RLT MLP actor 和 Q-head 的输入输出维度。

安装
----

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~

**方式一：Docker 镜像**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 为提高国内下载速度，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

进入容器后，切换到 OpenPI 虚拟环境：

.. code:: bash

   source switch_env openpi

**方式二：自建环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 `--use-mirror` 到下面的 install.sh 命令

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

RLT 如何工作
------------

Stage 1：学习 RLT 特征模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 从一个 VLA 检查点开始，在同一批示范数据上优化两个目标：

1. 普通 VLA 动作预测目标，日志中记为 ``vla_loss``。
2. RLT token 目标，日志中记为 ``rlt_loss``。

Stage 1 的总损失为：

.. code:: text

   total_loss = rlt_loss + rlt_alpha * vla_loss

OpenPI 模型会暴露 VLA prefix hidden states。RLT token transformer 读取这些 prefix states，并输出紧凑向量 ``z_rl``。Stage 2 使用 ``z_rl`` 作为学习到的 RL 表示，而不是直接在图像观测上训练 actor-critic。

Stage 1 中比较关键的字段：

.. code:: yaml

   rlt:
     train_data_path: /path/to/lerobot_dataset
     base_model_path: /path/to/model
     openpi_repo_id: <openpi_repo_id>
     openpi_config_name: <openpi_config_name>
     action_dim: <action_dim>
     state_indices: [<state_index_0>, <state_index_1>, ...]
     ref_num_action_chunks: <ref_num_action_chunks>
     z_dim: <z_dim>
     num_rl_tokens: <num_rl_tokens>

   actor:
     model:
       openpi:
         use_rlt: True
         rlt_alpha: 1.0
         rlt_image_only: False
         rlt_use_mask: True

``state_indices`` 表示从环境返回的原始状态向量中抽取哪些维度作为 ``proprio``。它的顺序会决定 Stage 1 和 Stage 2 看到的 proprio 输入顺序，因此需要与数据预处理、动作空间和模型配置保持一致；不同机器人或仿真环境应填写各自状态向量中的对应索引。

.. note::

   ManiSkill joint 示例不依赖 ``state_indices`` 手工切片。``pi05_rlt_maniskill_joint``
   dataconfig 将 LeRobot 数据中的 ``state`` 映射到 OpenPI ``observation.state``；
   Stage 2 rollout 也使用 OpenPI 处理后的 ``observation.state`` 作为 ``proprio``。
   这样 Stage 1、Stage 2 和 OpenPI normalization 看到的是同一种状态语义。

Stage 2：训练 Actor-Critic 策略
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 2 冻结 Stage 1 特征模型，只训练轻量级 RLT MLP actor 和 critic。

.. note::

   当前 Stage 2 实现不是标准 maximum-entropy SAC。

rollout 时：

1. 环境返回原始观测和任务元信息。
2. ``rollout.rlt_feature_model`` 运行冻结的 Stage 1 模型，将原始观测转换为：

   - ``z_rl``：紧凑的 RLT 表示。
   - ``proprio``：选中的机器人或仿真状态。
   - ``ref_chunk``：VLA reference action chunk。

3. Stage 2 actor 的输入是 ``ref_chunk``、``z_rl`` 和 ``proprio``。
4. replay buffer 存储 RLT transition：

.. code:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = 实际发送给环境的动作
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

Franka 真机配置中，``keyboard_reward_wrapper: rlt_policy_switch`` 会提供
``rlt_use_actor`` 标记。在操作员按下 ``b`` 之前，实际执行 VLA 的
``ref_chunk``；按下 ``b`` 之后，实际执行 Stage 2 actor 的动作。

ManiSkill joint 配置中，``env.*.rlt_policy_switch`` 使用任务信息自动产生
``rlt_use_actor``、``in_critical_phase`` 和 ``record_transition``。HF rollout
worker 根据这些标记在整 chunk 级别选择 actor action 或 VLA ``ref_chunk``，并把
实际执行的 action 写入 replay。

critic 使用 chunked rewards 上的 TD target。一个 action chunk 内的奖励会先按折扣累计，然后用下一状态 Q 值 bootstrap：

.. code:: text

   target_q = discounted_chunk_reward + gamma ** chunk_horizon * Q_target(next_obs, next_action)

如果 episode 已终止，并且 ``bootstrap_type`` 为 ``standard``，则不会加入 bootstrap 项。

Stage 2 中比较关键的字段：

.. code:: yaml

   algorithm:
     loss_type: rlt_ac
     q_weight: 1.0
     bc_weight: 1.0
     gamma: 0.99
     actor_agg_q: min
     entropy_tuning:
       alpha_type: fixed_alpha
       initial_alpha: 0.0
     rlt_schedule:
       enable: True
       warmup_post_collect_updates: 30000
       train_every_transitions: 5

   rollout:
     collect_transitions: True
     rlt_feature_model:
       model_type: openpi
       model_path: ${rlt.stage1_model_path}
       openpi:
         use_rlt: True

   actor:
     model:
       model_type: rlt_mlp_policy
       action_dim: ${rlt.action_dim}
       num_action_chunks: ${rlt.num_action_chunks}
       ref_num_action_chunks: ${rlt.ref_num_action_chunks}
       z_dim: ${rlt.z_dim}
       proprio_dim: ${rlt.proprio_dim}
       add_q_head: True

Stage 2 的 actor loss 为：

.. code:: text

   actor_loss = -q_weight * Q(obs, pi(obs)) + bc_weight * BC(pi(obs), target_action)

普通 policy step 中，BC target 是 VLA reference action。如果某一步存了
human intervention action，那么这一步的 BC target 会切换成人类动作。ManiSkill
默认关闭 ``expert_takeover``，因此默认路线只使用 VLA reference 和 actor action。

运行当前 Franka 示例
--------------------

数据：采集 Franka 示范并计算归一化统计
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 需要 LeRobot 格式的 Franka 示范数据，数据目录应直接包含
``data/`` 和 ``meta/``。在控制节点上按照 :doc:`Franka 真机文档
<../embodied/franka>` 的数据采集流程准备机器人和目标位姿，并在采集配置中导出
LeRobot 格式数据：

.. code:: yaml

   env:
     data_collection:
       enabled: True
       export_format: "lerobot"

然后启动采集：

.. code:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_data

采集完成后，将 LeRobot 数据集放到训练节点，并为当前 RLT OpenPI dataconfig
计算归一化统计。``repo_id`` 需要与 Stage 1 / Stage 2 配置中的
``rlt.openpi_repo_id`` 保持一致：

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_franka_state7d \
       --repo-id realworld_peg_insertion_rlt_stage1_7d

后续将 Stage 1 配置中的 ``rlt.train_data_path`` 指向该 LeRobot 数据集目录。

Stage 1：训练 RLT 特征模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

启动前先修改 Stage 1 配置中的路径：

.. code:: yaml

   rlt:
     train_data_path: /path/to/lerobot_dataset
     base_model_path: /path/to/model

启动 SFT：

.. code:: bash

   bash examples/sft/run_vla_sft.sh rlt_sft_openpi_pi05

保存出的检查点目录通常形如：

.. code:: text

   logs/<run-name>/checkpoints/global_step_<step>

Stage 2 中需要将这个目录填到 ``rlt.stage1_model_path``。

Stage 2：运行 RLT Actor-Critic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

修改 Stage 2 配置：

.. code:: yaml

   rlt:
     stage1_model_path: /path/to/stage1/checkpoint
     stage1_openpi_repo_id: <stage1_openpi_repo_id>

   cluster:
     node_groups:
       - label: <gpu_node_group>
         node_ranks: <gpu_node_rank>
       - label: <env_node_group>
         node_ranks: <env_node_rank>
         hardware:
           type: <robot_type>
           configs:
             - robot_ip: <robot_ip>

   env:
     train:
       keyboard_reward_wrapper: rlt_policy_switch
       override_cfg:
         target_ee_pose: [<target_x>, <target_y>, <target_z>, <target_roll>, <target_pitch>, <target_yaw>]

在 master node 上启动异步训练：

.. code:: bash

   bash examples/embodiment/run_realworld_async.sh rlt_stage2_ac_mlp

真机 rollout 时，键盘切换逻辑如下：

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 按键
     - 行为
   * - ``b``
     - 当前 episode 从 VLA reference action 切换到 Stage 2 RLT actor。
   * - reset
     - 清空切换状态，下一个 episode 重新从 VLA reference policy 开始。

运行 ManiSkill Joint 示例
-------------------------

ManiSkill joint 示例使用 ``PegInsertionSideWideClearance-v1``、RGB 双视角观测、
Panda 前 9 维 qpos 和 8 维 ``pd_joint_delta_pos`` action。每个 Stage 2 action
是 10-step chunk，语言指令固定为 ``insert the peg in the hole``。

数据：准备 joint-control LeRobot 数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 和 Stage 2 的 OpenPI feature model 使用
``pi05_rlt_maniskill_joint`` dataconfig。LeRobot 数据集至少需要包含：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - 含义
   * - ``image``
     - 主视角 RGB 图像。
   * - ``wrist_image``
     - wrist RGB 图像。
   * - ``state``
     - Panda joint proprio，当前示例使用 9 维状态。
   * - ``actions``
     - 8 维 ``pd_joint_delta_pos`` action。
   * - ``task``
     - 语言指令；也可以通过 ``default_prompt`` 固定为 ``insert the peg in the hole``。

计算归一化统计时，``--config-name`` 和 ``--repo-id`` 要与训练配置保持一致：

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_rlt_maniskill_joint \
       --repo-id maniskill_peginsertionside_joint

.. warning::

   Stage 1 checkpoint、Stage 2 ``rollout.rlt_feature_model`` 和 OpenPI assets
   必须使用同一套 ``norm_stats.json``。如果 SFT 基座和 Stage 2 加载了不同
   ``repo_id`` 下的 norm stats，VLA reference action 的尺度会发生偏移。

可选：训练 ManiSkill OpenPI SFT 基座
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果还没有 joint-control OpenPI 基座，先运行 SFT 配置：

.. code:: bash

   bash examples/sft/run_vla_sft.sh rlt_maniskill_joint_pi05_sft

这个配置使用 ``pi05_rlt_maniskill_joint`` dataconfig 和
``rlt_maniskill_joint`` 数据集路径。训练完成后，将保存出的
``.../checkpoints/global_step_<step>/actor`` 作为 Stage 1 的
``actor.model.model_path``。

Stage 1：训练 ManiSkill RLT 特征模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

修改 ``examples/sft/config/rlt_stage1_maniskill_joint.yaml`` 中的数据路径和
OpenPI SFT checkpoint：

.. code:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/maniskill_peginsertionside_joint
         weight: 1.0

   actor:
     model:
       model_path: /path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_<step>/actor
       openpi:
         config_name: pi05_rlt_maniskill_joint
     openpi_data:
       repo_id: maniskill_peginsertionside_joint
       default_prompt: insert the peg in the hole

启动训练：

.. code:: bash

   bash examples/sft/run_vla_sft.sh rlt_stage1_maniskill_joint

Stage 2 使用这个 Stage 1 actor 目录作为 ``rlt.stage1_model_path``。该目录需要同时
包含 VLA 权重、RLT token transformer 权重和对应的 OpenPI assets。

Stage 2：运行 ManiSkill RLT Actor-Critic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

修改 ``examples/embodiment/config/rlt_stage2_maniskill_joint_ac.yaml`` 中的
Stage 1 checkpoint：

.. code:: yaml

   rlt:
     stage1_model_path: /path/to/rlt_stage1_maniskill_joint/checkpoints/global_step_<step>/actor
     openpi_repo_id: maniskill_peginsertionside_joint
     openpi_config_name: pi05_rlt_maniskill_joint

   env:
     train:
       wrap_obs_mode: rlt_openpi_joint
       rlt_policy_switch:
         enable: True
         task_mode: full_task
         trigger_mode: auto
     eval:
       wrap_obs_mode: rlt_openpi_joint
       rlt_policy_switch:
         enable: True
         task_mode: full_task
         trigger_mode: auto

启动训练：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh rlt_stage2_maniskill_joint_ac

这个配置会启动 actor、rollout 和 ManiSkill env。rollout 侧冻结
``rollout.rlt_feature_model``，只同步和执行 Stage 2 MLP actor。ManiSkill route
在 ``ready_for_online`` 之前执行 VLA ``ref_chunk``；达到
``algorithm.rlt_schedule.warmup_post_collect_updates`` 后，才允许 actor 在自动
critical phase 中接管。

ManiSkill Stage 2 中最需要确认的字段：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 字段
     - 作用
   * - ``algorithm.rlt_schedule.warmup_post_collect_updates``
     - actor 上线前需要完成的 learner update 数。
   * - ``algorithm.rlt_schedule.train_every_transitions``
     - 每新增多少条 transition 增加一次训练预算。
   * - ``algorithm.replay_buffer.min_buffer_size``
     - replay buffer ready 的最小 transition 数。
   * - ``algorithm.replay_buffer.max_num_samples``
     - active replay transition 上限。
   * - ``algorithm.actor_agg_q``
     - actor loss 中使用的 Q 聚合方式，当前 ManiSkill 配置使用 ``min``。
   * - ``env.*.rlt_policy_switch``
     - 自动产生 ``rlt_use_actor``、``in_critical_phase`` 和 ``record_transition``。

Replay Buffer 逻辑
------------------

当 ``loss_type: rlt_ac`` 时，replay buffer 不会把原始图像观测当作 RL 状态存储。
rollout worker 会返回 RLT 特征，learner 侧把这些特征组装成 transition：

.. code:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = 实际发送给环境的 action chunk
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

这意味着：

- 真机切换前的 step 仍然会用于训练。这些 step 的执行动作是 VLA reference action，transition 也会进入同一个 replay buffer。
- 切换后的 step 执行 actor action，并以相同的 RLT observation 格式存储。
- ManiSkill route 在整 chunk 级别选择 actor action 或 VLA ``ref_chunk``，replay 中的 ``action`` 始终是实际执行的动作。
- ManiSkill learner 会把 rollout chunk 切成 1-sample transition trajectory，再放进 RLinf ``TrajectoryReplayBuffer``。
- ``sample_window_size`` 控制从 replay buffer 近期窗口中采样的范围，不需要和 ``max_steps_per_rollout_epoch`` 一致。
- ``max_steps_per_rollout_epoch`` 控制一次 rollout flush 到训练侧之前收集多少环境 step。

监控指标
--------

指标定义见 :doc:`训练指标 <../../reference/metrics>`。RLT 中比较有用的信号包括：

- Stage 1 SFT：

  - ``vla_loss``：OpenPI 动作预测损失。
  - ``rlt_loss``：RLT token 重建 / 压缩损失。

- Stage 2 actor-critic：

  - ``critic/critic_loss``：Q 函数 TD loss。
  - ``actor/actor_loss``：组合后的 ``-Q + BC`` actor 目标。
  - ``q_pi`` 和 ``q_value_*``：actor 和 critic heads 上的 Q 值。
  - ``actor/bc_loss``、``actor/bc_ref_loss``、``actor/bc_human_loss``：BC 正则相关损失。
  - ``train/replay_buffer/size``：当前 replay transitions 数量。
  - ``env/success_once`` 和 ``env/episode_len``：任务结果指标。
  - ``eval/success_once``：固定 eval reset ids 上的成功率。
  - ``rollout/in_critical_phase_rate`` 和 ``rollout/student_control_rate``：ManiSkill 自动 route 是否按预期让 actor 接管。
  - ``rlt_stage2/ready_for_online``、``rlt_stage2/actor_updates_run`` 和 ``rlt_stage2/pending_update_budget``：warmup、actor 更新和 learner backlog 状态。

实践建议
--------

- Stage 1 和 Stage 2 的维度必须保持一致：``action_dim``、``state_indices``、``ref_num_action_chunks``、``z_dim`` 和 ``num_rl_tokens`` 都要对齐。
- ``rollout.rlt_feature_model`` 指向 Stage 1 检查点；``actor.model`` 是 actor-critic worker 会更新的 Stage 2 MLP 策略。
- ``keyboard_reward_wrapper: rlt_policy_switch`` 只在需要人工控制关键阶段切换时使用。
- ManiSkill joint 示例使用 ``env.*.rlt_policy_switch``，不要再使用真机的 keyboard wrapper。
- ManiSkill 的 ``proprio`` 来自 OpenPI processed ``observation.state``。如果新建仿真 dataconfig，需要同时检查数据集 ``state``、OpenPI transform 和 Stage 2 ``proprio_dim``。
- Stage 1、Stage 2 和 checkpoint assets 中的 ``norm_stats.json`` 必须来自同一套数据语义和同一个 ``repo_id``。
