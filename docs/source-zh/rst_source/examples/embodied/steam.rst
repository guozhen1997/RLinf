STEAM：用于离线策略优化的集成优势建模
========================================================

在 RLinf 中运行 **STEAM** 流程。STEAM 是一种离线策略优化方法：用**成对分类的进度评论器（progress critic）**配合**深度集成（deep ensemble）**为已有数据打分，将保守的 worst-of-N 集成估计转化为逐帧的优势标签，再用这些标签驱动与 :doc:`RECAP <recap>` 相同的 **无分类器引导（Classifier-Free Guidance, CFG）训练**。

与 RECAP 一样，STEAM 无需在线环境交互，适合难以大规模在线采样的真实机器人场景。区别在于价值信号：STEAM 不回归折扣回报，而是从帧对中学习**时间进度（temporal progress）**评论器，并通过集成抑制单一预测器在分布外 rollout 上对优势的高估。

概览
----------------------------------------

离线提升策略（无需新采样）：用集成进度评论器为已有数据打分，再以无分类器引导（CFG）进行优化。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      STEAM（worst-of-N 集成）

   .. grid-item-card:: 模型
      :text-align: center

      SigLIP2 + Gemma3 评论器

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      LeRobot 数据集

   .. grid-item-card:: 训练
      :text-align: center

      离线 · 2 阶段 + CFG

| **你将完成：** SFT 一个集成进度评论器 → 计算集成优势 → CFG 训练策略 → 评测。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · SigLIP2 + Gemma3 + π₀.₅ 检查点 · LeRobot 格式数据集（见下文步骤）。

流程
----------------------------------------

STEAM 复用 RECAP 的 CFG 训练阶段，因此一次 STEAM 运行是两个 STEAM 特有阶段加上 RECAP 的 Step 4：

.. code-block:: text

   ┌────────────────────────┐     ┌────────────────────────┐     ┌──────────────────────┐
   │  Step 1                │     │  Step 2                │     │  Step 3 (RECAP CFG)  │
   │  STEAM Value Model SFT │────▶│  Compute Ensemble      │────▶│  CFG Training        │
   │                        │     │  Advantages            │     │                      │
   │  训练一组成对分类的    │     │  worst-of-N 集成有符号 │     │  用无分类器引导      │
   │  进度评论器            │     │  分数 → 布尔优势标签   │     │  训练策略            │
   └────────────────────────┘     └────────────────────────┘     └──────────────────────┘

**核心思路**

1. **Value Model SFT**：训练一组进度评论器（SigLIP2 + Gemma3 backbone + 分类头）。每个成员接收帧对 :math:`(o_t, o_{t+k})`，将有符号的帧步幅分类到若干 bin，因此预测的是**时间进度**而非回归回报。

2. **Compute Ensemble Advantages**：对每一帧，让所有集成成员在帧对 :math:`(o_t, o_{t+k})` 上推理，并以 **worst-of-N** 规则（:math:`A = \min_m A_m`）聚合，得到有符号分数 ``advantage_continuous`` :math:`\in [-1, 1]`，再按阈值或分位数规则将帧标记为正/负。

3. **CFG Training**：将优势标签交给 RECAP 的 CFG 阶段——正样本（高优势）作为条件输入，负样本作为无条件输入，实现 classifier-free guidance 策略优化。

STEAM 工作原理
----------------------------------------

**成对分类的进度评论器**

每个评论器接收帧对 :math:`(o_t, o_{t+k})`\ （SigLIP2 视觉编码器 + Gemma3 语言模型逐帧融合），并将**有符号步幅** :math:`s \in \{-K, \dots, -1, 1, \dots, K\}` 分类到 ``num_bins`` 个连续 bin。``[0, num_bins/2)`` 是回退 bin（负步幅），``[num_bins/2, num_bins)`` 是前进 bin（正步幅）。``num_bins == 2`` 退化为二分类进度判别；``num_bins > 2`` 给出更细的有符号步幅分布。

**有符号分数**

每个成员的预测是归一化到 :math:`[-1, 1]` 的 bin 加权期望：

.. math::

   \text{signed\_score} = \frac{1}{K}\sum_b p_b \cdot \text{center}(b)

对 ``num_bins == 2``，它退化为 :math:`2 \cdot P(\text{progress}) - 1`。

**worst-of-N 集成聚合**

集成成员在分布内一致，但在分布外 rollout 上发散。STEAM 用 STEAM 论文中的保守 worst-of-N 规则聚合——``predicted_values = min_m signed_score_m``——使得在集成产生分歧的地方，过度自信的单个成员无法抬高优势。逐成员的均值 / 最小值 / 方差作为诊断量记录下来。

**优势标注**

``advantage_continuous``\ （聚合后的有符号分数）通过两种 ``label_mode`` 规则之一转化为布尔 ``advantage``：

- ``threshold``：对 rollout 帧 ``advantage = advantage_continuous > positive_threshold``\ （:math:`[-1, 1]` 内的有符号分数阈值）；sft 帧恒为 True（按构造是成功演示）。
- ``quantile``：将 rollout 帧中分数最高的 ``rollout_quantile`` 比例标为 True；当设置了 ``expert_quantile`` 时，再将 sft 帧中最高的 ``expert_quantile`` 比例标为 True——两个池独立打分。

安装
----------------------------------------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 中国大陆用户可使用以下镜像以获得更快下载速度：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~~~~~~~~~~~

STEAM 与 RECAP 共用 OpenPI 环境。

**方式一：Docker 镜像**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

   source switch_env openpi

**方式二：自定义环境**

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

下载模型
----------------------------------------

STEAM 价值模型由两个预训练 backbone 构成：

- **SigLIP2-so400m**\ （``google/siglip-so400m-patch14-384``\ ）：视觉编码器
- **Gemma3-270M**\ （``google/gemma-3-270m``\ ）：语言模型与分词器

.. code:: bash

   git lfs install
   git clone https://huggingface.co/google/siglip-so400m-patch14-384
   git clone https://huggingface.co/google/gemma-3-270m

   # 或使用 huggingface-hub（设置 HF_ENDPOINT=https://hf-mirror.com 使用镜像）
   hf download google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384
   hf download google/gemma-3-270m --local-dir gemma-3-270m

在模型配置（``examples/value/steam/config/model/steam.yaml``\ ）中设置路径：

.. code:: yaml

   actor:
     model:
       vision_repo_id: /path/to/siglip-so400m-patch14-384
       language_repo_id: /path/to/gemma-3-270m
       tokenizer_path: /path/to/gemma-3-270m

数据准备
----------------------------------------

STEAM 使用 LeRobot 格式数据集，分为两类：

- **SFT 数据集**：成功轨迹（人类演示或训练好的策略）。
- **Rollout 数据集**：在线采集的轨迹，同时包含成功与失败。

示例数据配置：

.. code:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
       - dataset_path: /path/to/rollout_dataset
         type: rollout

.. note::

   Step 1 与 Step 2 的 ``train_data_paths`` 和 ``data.k`` 必须保持一致：优势计算必须以评论器训练时相同的时间步幅对帧对打分。

流程 Tag 系统
~~~~~~~~~~~~~~~~~~~~~

STEAM 用 **advantage tag** 将数据从 Step 2 传递到 CFG 阶段。将 Step 2 的 ``advantage.tag`` 与 CFG 阶段的 ``data.advantage_tag`` 设为相同值，CFG 即可读取 ``meta/advantages_{tag}.parquet``。

Step 1：价值模型 SFT
----------------------------------------

训练集成进度评论器。每个成员是 SigLIP2 + Gemma3 backbone 加一个分类头；成员从共享 backbone 克隆而来，并对其 value head 重新设种子，使集成方差成为有意义的认知不确定性信号。

**配置**

配置文件为 ``examples/value/steam/config/steam_model_ensemble1.yaml``\ ；模型默认值在 ``config/model/steam.yaml``\ 。关键字段：

.. code:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
     k: 32                       # 最大有符号步幅 K（帧对时间尺度）
     camera_keys: [face_view, left_wrist_view, right_wrist_view]
     prompt: "perform the task"

   actor:
     micro_batch_size: 32
     global_batch_size: 512
     model:
       num_bins: 32              # 2 = 二分类进度；>2 = 多 bin（偶数）
       ensemble_size: 3          # 集成中评论器数量
       fusion_hidden_dim: 512
       freeze_vision_encoder: false
       freeze_language_model: false
       use_gradient_checkpointing: true
     optim:
       lr: 5.0e-5
       value_lr: 5.0e-5

**关键参数**

.. list-table::
   :header-rows: 1
   :widths: 32 16 52

   * - 参数
     - 默认值
     - 说明
   * - ``data.k``
     - ``required``
     - 最大有符号步幅 :math:`K`。多 bin 模式下 ``2*K`` 必须是 ``num_bins`` 的整数倍。
   * - ``actor.model.num_bins``
     - ``2``
     - bin 数量。``2`` 为二分类进度；``> 2``\ （偶数）为多 bin 有符号步幅分类。
   * - ``actor.model.ensemble_size``
     - ``1``
     - 集成成员数。``> 1`` 启用 worst-of-N 聚合与不确定性统计。
   * - ``actor.model.fusion_hidden_dim``
     - ``512``
     - 逐帧融合 MLP 的隐藏层宽度。
   * - ``actor.model.freeze_vision_encoder``
     - ``false``
     - 是否冻结 SigLIP2 编码器。
   * - ``actor.model.use_gradient_checkpointing``
     - ``false``
     - 反向时重算 backbone 激活（在 80GB 卡上做全 backbone + 集成时需要）。

**启动命令**

.. code:: bash

   bash examples/value/steam/run_steam_sft.sh steam_model_ensemble1

   # 命令行覆盖配置字段：
   bash examples/value/steam/run_steam_sft.sh steam_model_ensemble1 data.k=8

**输出**

- 检查点位于 ``logs/steam_sft/{config_name}-{timestamp}/.../checkpoints/global_step_{N}/actor``
- TensorBoard 日志

**关键指标**

- ``train/actor/loss``：有符号步幅 bin 上的交叉熵
- ``train/actor/accuracy``：最优 bin 分类准确率
- ``train/actor/grad_norm``：梯度范数

Step 2：计算集成优势
----------------------------------------

用训练好的集成对每一帧推理，并写出逐帧优势标签。

**配置**

配置文件为 ``examples/value/steam/process/config/compute_advantages_ensemble.yaml``：

.. code:: yaml

   advantage:
     value_checkpoint: /path/to/steam_value_ensemble/checkpoints/global_step_N/actor
     batch_size: 256
     label_mode: quantile        # 必填："threshold" 或 "quantile"
     rollout_quantile: 0.3       # rollout 帧最高的 30% 标为 True
     expert_quantile: 0.8        # 可选：sft 帧最高的 80% 标为 True
     tag: steam_k32_ensemble3_q30

   data:
     model_type: "pi0"
     robot_type: "restock_cola_sm2sm"
     k: 32                       # 必须与 Step 1 的 data.k 一致
     camera_keys: [face_view, left_wrist_view, right_wrist_view]
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
       - dataset_path: /path/to/rollout_dataset
         type: rollout

**关键参数**

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - 参数
     - 默认值
     - 说明
   * - ``advantage.value_checkpoint``
     - ``required``
     - Step 1 集成检查点路径（``actor`` 目录）。
   * - ``advantage.label_mode``
     - ``required``
     - ``threshold`` 或 ``quantile``\ （无默认值，必须显式设置）。
   * - ``advantage.positive_threshold``
     - ``null``
     - :math:`[-1, 1]` 内的有符号分数阈值（仅 ``label_mode=threshold``\ ）。
   * - ``advantage.rollout_quantile``
     - ``null``
     - rollout 帧标为 True 的最高比例（``label_mode=quantile``\ ，必填）。
   * - ``advantage.expert_quantile``
     - ``null``
     - sft 帧标为 True 的最高比例（``label_mode=quantile``\ ，可选）。
   * - ``advantage.tag``
     - ``required``
     - 输出 tag；写入 ``meta/advantages_{tag}.parquet``。
   * - ``data.k``
     - ``required``
     - 帧对步幅；必须与 Step 1 训练的 ``data.k`` 一致。

**启动命令**

.. code:: bash

   # 自动检测 GPU 数；单卡与 torchrun 多卡均支持。
   bash examples/value/steam/process/run_compute_advantages_ensemble.sh compute_advantages_ensemble

   # 指定 GPU 数：
   bash examples/value/steam/process/run_compute_advantages_ensemble.sh compute_advantages_ensemble --nproc 4

**输出文件**

- ``meta/advantages_{tag}.parquet``：逐帧的 ``advantage``\ （布尔）、``advantage_continuous``\ （有符号分数）、``ensemble_signed_score``\ 、逐成员值，以及集成熵 / 方差等诊断量。
- ``meta/mixture_config.yaml``：每个 tag 一条记录，记录 ``label_mode``\ 、所用阈值、``ensemble_size``\ 、``num_bins`` 和正样本计数。

Step 3：CFG 训练
----------------------------------------

STEAM 优势 parquet 与 RECAP 共享 schema，因此策略优化复用 RECAP 的 CFG 阶段。将 CFG 配置的 ``data.advantage_tag`` 指向 Step 2 的 ``advantage.tag`` 并启动：

.. code:: bash

   bash examples/embodiment/run_cfg_sft.sh libero_cfg_openpi \
       data.advantage_tag=steam_k32_ensemble3_q30

完整的 CFG 配置与参数见 :doc:`RECAP Step 4 <recap>`。

进阶用法
----------------------------------------

合并集成检查点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

作为独立单模型训练（或从已有集成中抽取）的成员，可以融合为一个集成推理检查点。每个 ``--member`` 是一个检查点路径，或用 ``PATH:idx`` 从集成中抽取第 ``idx`` 个成员：

.. code:: bash

   python examples/value/steam/process/merge_steam_ensemble.py \
       --member /path/to/seed1/checkpoints/global_step_5000/actor \
       --member /path/to/seed2/checkpoints/global_step_5000/actor \
       --member /path/to/ensemble/checkpoints/global_step_6000/actor:2 \
       --output /path/to/merged/actor

合并逻辑位于
``rlinf.models.embodiment.value.steam.checkpoint_merge.merge_ensemble_checkpoints``。

阈值 / 分位数重标注
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

若要在不重跑 GPU 推理的情况下改变标注阈值，可对已有优势 parquet 重标注（纯 CPU——复用 ``advantage_continuous``\ ）：

.. code:: bash

   python examples/value/steam/process/relabel_advantages.py \
       --dataset_paths /path/to/sft_ds /path/to/rollout_ds \
       --source_tag steam_k32_ensemble3_q30 \
       --new_tag steam_k32_ensemble3_q20 \
       --mode quantile --rollout_quantile 0.2

重标注逻辑位于
``rlinf.data.process.steam.relabel.relabel_advantages``。

可视化优势
~~~~~~~~~~~~~~~~~~~~~~

从优势 parquet 渲染分布、逐成员、不确定性、逐 episode 以及 episode 时间线等诊断图：

.. code:: bash

   python examples/value/steam/process/visualize_advantage.py \
       --dataset /path/to/dataset \
       --tag steam_k32_ensemble3_q30 \
       --output outputs/steam_viz

可视化与结果
----------------------------------------

指标定义见 :doc:`训练指标 <../../reference/metrics>`。

.. code:: bash

   tensorboard --logdir ./logs --port 6006

文件结构
----------------------------------------

STEAM 在 ``examples/`` 下保留轻量入口，将模型 / 优势逻辑放在 ``rlinf/`` 下：

.. code-block:: text

   examples/value/steam/
   ├── train_steam.py                         # Step 1：价值模型 SFT 入口
   ├── run_steam_sft.sh                       # Step 1 启动脚本
   ├── config/
   │   ├── steam_model_ensemble1.yaml
   │   └── model/steam.yaml
   └── process/
       ├── compute_advantages_ensemble.py     # Step 2：hydra 入口（薄）
       ├── merge_steam_ensemble.py            # CLI：合并集成检查点
       ├── relabel_advantages.py              # CLI：重标注优势（CPU）
       ├── visualize_advantage.py             # 优势可视化
       ├── run_compute_advantages_ensemble.sh # Step 2 启动脚本
       └── config/
           └── compute_advantages_ensemble.yaml

   rlinf/
   ├── models/embodiment/value/steam/                   # 评论器、集成、配置、合并
   │   ├── modeling_steam.py / modeling_critic.py
   │   ├── ensemble_modeling_critic.py            # worst-of-N + coerce_to_ensemble
   │   └── checkpoint_merge.py
   ├── data/datasets/steam/binning.py             # 有符号步幅 ↔ bin 数学
   └── data/process/                              # 与 RECAP 共享的后处理
       ├── advantage.py                           # 分位数阈值 + 布尔标签
       ├── distributed.py                         # 分片推理辅助
       └── steam/                                 # STEAM 专用流程
           ├── inference.py / pipeline.py         # 集成推理 + 编排
           ├── labelling.py / mixture_config.py   # 标注 + 元数据 I/O
           └── relabel.py                         # CPU 重标注核心
