奖励模型工作流
==============

本文档提供了奖励模型工作流的全面指南，包括数据收集、奖励模型训练，以及使用训练好的奖励模型替换环境成功信号来训练 RL 策略。

算法支持
--------

**此工作流支持 PPO (Proximal Policy Optimization)** 算法：

- ✅ **PPO** (Proximal Policy Optimization) - 使用配置：``maniskill_ppo_cnn.yaml``

**主要优势**：

- 简单统一的工作流
- 所有脚本和配置位于 ``examples/embodiment/`` 目录

概述
----

奖励模型工作流包含三个主要阶段：

1. **数据收集**：使用初始 RL 策略（PPO）收集轨迹，存储图像和成功标签
2. **奖励模型训练**：训练一个二分类器，从单帧图像预测成功（算法无关）
3. **RL 策略训练**：使用训练好的奖励模型替换环境成功信号进行 PPO 训练

**关键点**：

- 工作流使用 PPO 算法进行 RL 策略训练
- 使用统一的 ``RewardWorker`` 进行奖励计算

关键文件和组件
--------------

数据收集
~~~~~~~~

**``rlinf/models/reward_model/reward_data_collector.py``**

**用途**：核心数据收集实现

- **主类**：``RewardDataCollector``
- **职责**：
  - 为每个并行环境维护独立的轨迹缓冲区
  - 收集观察（图像）、``success_frame`` 标签和 ``success_once`` 标志
  - 将完整轨迹保存为 ``.npy`` 文件
  - 根据成功标准将轨迹分类为正/负
  - 确保数据格式一致性（``[C, H, W]`` 图像，归一化到 ``[0, 1]``）

**关键方法**：

- ``add_env_output(env_output, env_info, is_last_step)``：将环境输出添加到缓冲区
- ``save_trajectory(env_idx)``：将完成的轨迹保存到磁盘
- ``_extract_image(image, key_offset)``：从观察中提取和格式化图像

**``rlinf/workers/env/env_worker.py``**

**用途**：集成数据收集的环境工作器

- **集成点**：如果 ``cfg.reward.collect_data=True``，在 ``__init__`` 中初始化 ``RewardDataCollector``
- **集成方法**：在 ``interact()`` 方法中每次环境交互后调用 ``data_collector.add_env_output()``
- **关键逻辑**：从 ``env_output`` 提取 ``success_frame``，从 ``env_info`` 提取 ``success_once``

奖励模型训练
~~~~~~~~~~~~

**``examples/embodiment/train_reward_model.py``**

**用途**：奖励模型训练脚本

- **主函数**：``main(cfg: DictConfig)`` （使用 Hydra 装饰器）
- **配置**：使用 Hydra 和 OmegaConf，配置文件位于 ``examples/embodiment/config/train_reward_model.yaml``
- **职责**：
  - 从 ``positive_dir`` 和 ``negative_dir`` 加载轨迹数据（来自配置）
  - 创建 ``RewardDataset`` 进行帧级训练
  - 使用 BCE 损失训练 ``BinaryRewardClassifier``
  - 保存最佳模型检查点
  - 可选地可视化正样本

**``rlinf/models/reward_model/reward_classifier.py``**

**用途**：奖励模型架构定义

- **主类**：``BinaryRewardClassifier``
- **架构**：
  - **编码器**：``ResNetEncoderWrapper``（ResNet10 骨干网络）
  - **分类器头**：MLP（Linear → LayerNorm → Tanh → Linear → Sigmoid）
  - **输入**：单帧图像 ``[B, C, H, W]``
  - **输出**：二分类 logit（通过 sigmoid 的成功概率）

奖励模型使用
~~~~~~~~~~~~

**``rlinf/workers/reward/reward_worker.py``**

**用途**：支持基于文本的推理任务和具身任务的统一奖励工作器

- **主类**：``RewardWorker``
- **任务类型检测**：根据配置自动检测任务类型（具身 vs 基于文本）
- **职责**：
  - 对于具身任务：从检查点加载 ``BinaryRewardClassifier`` 并计算基于帧的奖励
  - 对于基于文本的任务：支持基于规则的奖励（基于文本任务的奖励模型尚未实现）
  - 处理并行环境的批量处理
  - 为具身任务用模型预测替换环境成功信号

阶段 1：数据收集
----------------

配置
~~~~

奖励模型工作流使用 PPO 算法。使用配置文件：

- **PPO**：``examples/embodiment/config/maniskill_ppo_cnn.yaml``

示例配置：

.. code-block:: yaml

   reward:
     group_name: "RewardGroup"
     use_reward_model: False  # 数据收集期间禁用
     collect_data: True  # 启用数据收集
     reward_model:
       # 这些设置在数据收集期间不使用，但应与 actor 设置匹配
       image_keys: ["base_camera"]
       image_size: [3, 64, 64]
     data_collection:
       # 保存正轨迹的目录（success_once=1 或任何 success_frame>=0.5）
       positive_dir: "./reward_data/positive"
       # 保存负轨迹的目录（success_once=0 且所有 success_frame<0.5）
       negative_dir: "./reward_data/negative"
       # 要收集的图像键（必须匹配 actor.model.image_keys）
       image_keys: ["base_camera"]
       # 要保存的最大轨迹数（None = 无限制）
       max_positive_trajectories: 500
       max_negative_trajectories: 500

启动命令
~~~~~~~~

.. code-block:: bash

   # 对于 PPO
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn

   # 或直接使用 python
   export EMBODIED_PATH="/path/to/RLinf/examples/embodiment"
   export PYTHONPATH="/path/to/RLinf:$PYTHONPATH"
   python examples/embodiment/train_embodied_agent.py \
       --config-path examples/embodiment/config/ \
       --config-name maniskill_ppo_cnn \
       reward.collect_data=True \
       reward.use_reward_model=False

数据格式
~~~~~~~~

每个轨迹保存为一个包含字典的 ``.npy`` 文件：

.. code-block:: python

   {
       'images': {
           'base_camera': np.array([T, C, H, W], dtype=np.float32)  # T 帧，归一化到 [0, 1]
       },
       'labels': np.array([T], dtype=np.float32)  # T 个 success_frame 标签（0 或 1）
   }

**文件命名**：``{counter:06d}.npy``（例如，``000000.npy``，``000001.npy``，...）

**保存位置**：

- 正轨迹：``{positive_dir}/000000.npy``，``{positive_dir}/000001.npy``，...
- 负轨迹：``{negative_dir}/000000.npy``，``{negative_dir}/000001.npy``，...

轨迹分类逻辑
~~~~~~~~~~~~

如果满足以下条件，轨迹将保存到 ``positive_dir``：

- ``success_once=True`` **或**
- 任何帧的 ``success_frame >= 0.5``

否则，保存到 ``negative_dir``。

阶段 2：奖励模型训练
---------------------

先决条件
~~~~~~~~

1. **预训练 ResNet10 编码器**（可选但推荐）：
   - **文件**：``resnet10_pretrained.pt``
   - **位置**：应放置在项目根目录或通过配置中的 ``pretrained_encoder_path`` 指定
   - **用途**：为更好的初始化提供预训练视觉特征
   - **注意**：如果未提供（设置为 ``null``），编码器将随机初始化

2. **收集的数据**：
   - ``positive_dir`` 中的正轨迹
   - ``negative_dir`` 中的负轨迹

配置
~~~~

训练脚本使用 Hydra 和 OmegaConf 进行配置管理。默认配置文件位于 ``examples/embodiment/config/train_reward_model.yaml``。

默认配置文件
~~~~~~~~~~~~

.. code-block:: yaml

   defaults:
     - override hydra/job_logging: stdout

   hydra:
     run:
       dir: .
     output_subdir: null

   # 数据路径
   positive_dir: ./reward_data/positive
   negative_dir: ./reward_data/negative

   # 模型配置
   backbone: resnet10  # 目前仅支持 resnet10
   image_key: base_camera
   image_size: [3, 64, 64]  # [C, H, W]
   hidden_dim: 256
   num_spatial_blocks: 8
   pretrained_encoder_path: null  # 预训练编码器权重路径

   # 训练配置
   batch_size: 32
   epochs: 100
   lr: 1e-4
   num_workers: 4

   # 输出配置
   output_checkpoint: ./checkpoints/reward_model.pt

   # 可视化
   visualize_positive: false
   vis_output_dir: positive_samples_vis

   # 设备（设置为 null 以自动检测）
   device: null  # 将自动检测：如果可用则为 "cuda"，否则为 "cpu"

启动命令
~~~~~~~~

.. code-block:: bash

   # 使用提供的脚本（根据需要修改路径）
   bash examples/embodiment/train_reward_model.sh

   # 或直接使用 python（从 examples/embodiment 目录）
   cd examples/embodiment
   python train_reward_model.py

   # 通过命令行覆盖配置参数
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

训练过程
~~~~~~~~

1. **数据加载**：
   - 从 ``positive_dir`` 和 ``negative_dir`` 加载所有 ``.npy`` 文件
   - 将轨迹级数据扩展到帧级样本
   - 每个样本 = ``(trajectory_path, frame_index, label)``
   - 报告统计信息：
      - 每个目录中的轨迹和帧数
      - 每个目录中标签的分布（label=1 vs label=0）

2. **模型创建**：
   - 使用指定参数创建 ``BinaryRewardClassifier``
   - 如果配置中提供了 ``pretrained_encoder_path``，加载预训练 ResNet10 权重
   - 编码器权重默认冻结（``freeze_encoder=True``）

3. **训练循环**：
   - 使用带 logits 的 BCE 损失
   - Adam 优化器
   - 根据训练准确率保存最佳模型

阶段 3：使用奖励模型训练 RL 策略
-----------------------------------

配置
~~~~

奖励模型工作流使用 PPO 算法。使用配置文件：

- **PPO**：``examples/embodiment/config/maniskill_ppo_cnn.yaml``

示例配置：

.. code-block:: yaml

   reward:
     group_name: "RewardGroup"
     use_reward_model: True  # 启用奖励模型
     collect_data: False  # 策略训练期间禁用数据收集
     reward_model:
       # 训练好的奖励模型检查点路径
       checkpoint_path: "./checkpoints/reward_model.pt"
       # 要使用的图像键（必须匹配 actor.model.image_keys）
       image_keys: ["base_camera"]
       # 图像大小 [C, H, W]（必须匹配 actor.model.image_size）
       image_size: [3, 64, 64]
       # 分类器的隐藏维度（必须匹配训练配置）
       hidden_dim: 256
       # 池化的空间块数（必须匹配训练配置）
       num_spatial_blocks: 8
       # 预训练 ResNet10 编码器权重路径（可选，在模型创建期间使用）
       pretrained_encoder_path: "./resnet10_pretrained.pt"
       # 是否使用预训练编码器（必须匹配训练配置）
       use_pretrain: True
       # 是否冻结编码器权重（通常为 True）
       freeze_encoder: True
       # 奖励类型："binary"（0 或 1）或 "continuous"（概率）
       reward_type: "binary"

**重要**：``image_keys``、``image_size``、``hidden_dim``、``num_spatial_blocks``、``use_pretrain`` 和 ``freeze_encoder`` 设置**必须匹配**奖励模型训练期间使用的设置。

启动命令
~~~~~~~~

.. code-block:: bash

   # 对于 PPO
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn

   # 或直接使用 python
   export EMBODIED_PATH="/path/to/RLinf/examples/embodiment"
   export PYTHONPATH="/path/to/RLinf:$PYTHONPATH"
   python examples/embodiment/train_embodied_agent.py \
       --config-path examples/embodiment/config/ \
       --config-name maniskill_ppo_cnn \
       reward.use_reward_model=True \
       reward.collect_data=False \
       reward.reward_model.checkpoint_path="./checkpoints/reward_model.pt"

完整工作流示例
--------------

.. code-block:: bash

   # 步骤 1：收集数据
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn \
       reward.collect_data=True \
       reward.use_reward_model=False

   # 步骤 2：训练奖励模型
   cd examples/embodiment
   python train_reward_model.py \
       positive_dir=./reward_data/positive \
       negative_dir=./reward_data/negative \
       output_checkpoint=./checkpoints/reward_model.pt

   # 步骤 3：使用奖励模型训练
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn \
       reward.use_reward_model=True \
       reward.collect_data=False \
       reward.reward_model.checkpoint_path=./checkpoints/reward_model.pt

