DreamZero 监督微调（SFT）
==========================

本文档介绍如何在 RLinf 中运行 DreamZero 监督微调（SFT），覆盖从**模型与数据准备**、**配置填写**到**启动训练**与排错的完整流程。

当前支持：

- **数据集**：LIBERO（LeRobot）、LeRobot / OXE DROID
- **骨干网络**：WAN2.1（如 DreamZero-DROID 14B）、WAN2.2（如 Wan2.2-TI2V-5B 冷启动）


支持的训练组合
--------------

推荐配置文件（``examples/sft/config/``）：

.. list-table::
   :header-rows: 1
   :widths: 22 18 18 42

   * - 配置文件
     - 数据集 / ``embodiment_tag``
     - Hydra 预设
     - 典型用法
   * - ``libero_sft_dreamzero_14b.yaml``
     - LIBERO / ``libero_sim``
     - ``model/dreamzero_14b``
     - **WAN2.1**：设置 ``model_path`` 指向完整 checkpoint（如官方 DreamZero 权重目录），从 ``config.json`` 加载架构与权重；需 ``metadata_json_path`` 或 ``experiment_cfg/metadata.json``。
   * - ``libero_sft_dreamzero_5b.yaml``
     - LIBERO / ``libero_sim``
     - ``model/dreamzero_5b``
     - **WAN2.2 冷启动**：``model_path: null``，填写 Wan2.2-TI2V-5B 等各 ``*_pretrained_path`` 与 ``metadata_json_path``。
   * - ``droid_sft_dreamzero_14b.yaml``
     - DROID / ``oxe_droid``
     - ``model/dreamzero_14b``
     - **DROID SFT（WAN2.1）**：``defaults`` 含 ``dreamzero_14b``；默认 ``model_path`` 指向 DreamZero-DROID，从 ``config.json`` 加载架构与权重。含 ``relative_action``、``sampling_mode: multi_anchor`` 等 DROID 项。

常见起点：

- **从已发布 checkpoint 继续 SFT**（``libero_sft_dreamzero_14b``、``droid_sft_dreamzero_14b`` 且 ``model_path`` 非空）：收敛更快、更稳定。
- **从 WAN2.2 组件冷启动**（``libero_sft_dreamzero_5b``，``model_path: null``）：更灵活，需更多数据与 ``metadata.json``。


环境准备
--------

1. 克隆 RLinf 仓库并进入根目录：

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 使用 ``requirements/install.sh`` 创建并同步 **DreamZero 专用 uv 虚拟环境**（默认目录 ``.venv``，Python 3.11.14）：

.. code:: bash

   # 仅做 SFT（LeRobot 离线数据，不跑仿真）— 推荐
   bash requirements/install.sh embodied --model dreamzero

   # 若后续还要在 LIBERO 仿真里评测，可一并安装 libero 环境
   bash requirements/install.sh embodied --model dreamzero --env libero

说明：

- 安装过程会：用 ``uv`` 创建/复用 ``.venv`` → ``uv sync --extra embodied`` → 安装 ``requirements/embodied/models/dreamzero.txt``（含 ``lerobot``、``diffusers``、``torchcodec`` 等）→ 在 NVIDIA 平台上安装 ``flash-attn``。
- 国内网络可加 ``--use-mirror`` 加速 PyPI / Python / GitHub 下载。
- 自定义 venv 目录：``--venv <dir>``；无 root 且系统依赖已就绪：``--no-root``。

安装完成后激活环境：

.. code:: bash

   source .venv/bin/activate

3. 单独克隆 **DreamZero（Groot）** 代码库，并设置 ``DREAMZERO_PATH`` 指向其中 **Python 包根目录**（内含 ``groot`` 包，CI 中多为 ``<DreamZero>/dreamzero``）：

.. code:: bash

   # 克隆官方 DreamZero / Groot 仓库（URL 以项目发布说明为准）
   export DREAMZERO_PATH=/path/to/DreamZero/dreamzero   # 须包含 import groot 的包根目录

模型准备
--------

Hydra 通过 ``defaults`` 组合 ``actor.model``（例如 ``libero_sft_dreamzero_5b.yaml`` 中的 ``model/dreamzero_5b@actor.model``）。训练 YAML 可覆盖 ``actor.model`` 下任意字段。

**策略架构的两种加载方式**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 方式
     - 说明
   * - ``model_path`` 已设置
     - 从 ``<model_path>/config.json`` 读取**完整模型架构**；YAML 中其余架构字段会被忽略（会打 warning）。权重从该目录加载（``model.safetensors`` 或分片索引）。
   * - ``model_path: null``
     - 完全使用 Hydra 解析后的 ``actor.model`` 字典（或 ``model/dreamzero_5b.yaml`` 等预设）。以下路径**必须非空**：``tokenizer_path``、``diffusion_model_pretrained_path``、``image_encoder_pretrained_path``、``text_encoder_pretrained_path``、``vae_pretrained_path``。

**归一化统计（metadata.json）**

动作 / 状态的归一化统计来自数据集生成的 ``metadata.json``。指定方式：

- ``actor.model.metadata_json_path``：显式路径；或
- ``<model_path>/experiment_cfg/metadata.json``（按 ``embodiment_tag`` 取对应条目）

若使用冷启动且 checkpoint 目录中没有该文件，需先用下文工具生成并填写 ``metadata_json_path``。

**数据变换（transform）**

无论 ``model_path`` 是否设置，**数据变换链均在 Python 中按 ``embodiment_tag`` 构建**（见 ``rlinf/data/datasets/dreamzero/data_transforms/__init__.py``）。内置标签：

- ``libero_sim``：LIBERO 仿真
- ``oxe_droid``：LeRobot / OXE DROID

WAN2.1：使用完整 DreamZero checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

将 ``actor.model.model_path`` 指向官方或自训的 DreamZero 目录，通常包含：

- ``config.json``：模型架构
- ``experiment_cfg/metadata.json``：数据集归一化统计（可选，也可用 ``metadata_json_path`` 覆盖）
- ``model.safetensors``（或分片 safetensors）
- ``tokenizer_path``：仍须在 YAML 中指定（如 ``umt5-xxl``）

在 ``defaults`` 中引入 ``model/dreamzero_14b@actor.model``（见 ``libero_sft_dreamzero_14b.yaml``）。

示例（LIBERO + 14B checkpoint）：

.. code:: yaml

   defaults:
     - model/dreamzero_14b@actor.model

   actor:
     model:
       model_path: /path/to/models/<your-checkpoint>   # 示例见 libero_sft_dreamzero_14b.yaml
       metadata_json_path: /path/to/dataset/metadata.json
       tokenizer_path: /path/to/models/umt5-xxl
       embodiment_tag: "libero_sim"
       action_horizon: 16

示例（DROID，见 ``droid_sft_dreamzero_14b.yaml``；``defaults`` 含 ``dreamzero_14b``，``model_path`` 非空时从 checkpoint 读架构与权重）：

.. code:: yaml

   actor:
     model:
       model_type: "dreamzero"
       model_path: /path/to/models/DreamZero-DROID
       metadata_json_path: /path/to/dataset/metadata.json
       tokenizer_path: /path/to/models/umt5-xxl
       embodiment_tag: "oxe_droid"
       action_horizon: 24
       relative_action: True

WAN2.2：从组件冷启动（5B 等）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WAN2.2 训练通常需要：

- WAN2.2 骨干权重（DiT + T5 + VAE）
- WAN2.1 CLIP 图像编码器（**不包含**在 Wan2.2-TI2V-5B 包内，需单独下载）
- 分词器 ``google/umt5-xxl``

在 ``defaults`` 中引入 ``model/dreamzero_5b@actor.model``，并设置 ``model_path: null``，填写各 ``*_pretrained_path``。``config.json`` 或预设中的架构需与权重一致，例如：

- ``diffusion_model_cfg``：``model_type=ti2v``，``in_dim=48``，``out_dim=48``，``frame_seqlen=50``
- ``vae_cfg``：``WanVideoVAE38``
- ``image_encoder_pretrained_path`` 指向 WAN2.1 CLIP 权重

示例（LIBERO + 5B 冷启动，见 ``libero_sft_dreamzero_5b.yaml``）：

.. code:: yaml

   defaults:
     - model/dreamzero_5b@actor.model

   actor:
     model:
       model_path: null
       tokenizer_path: google/umt5-xxl
       diffusion_model_pretrained_path: Wan-AI/Wan2.2-TI2V-5B
       image_encoder_pretrained_path: Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
       text_encoder_pretrained_path: Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
       vae_pretrained_path: Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
       metadata_json_path: /path/to/metadata.json
       embodiment_tag: "libero_sim"

预设文件 ``examples/sft/config/model/dreamzero_5b.yaml``（WAN2.2 / 5B）与 ``dreamzero_14b.yaml``（WAN2.1 / 14B）通过 Hydra ``defaults`` 引入，**不要**用单独的顶层 ``dreamzero_*`` 键选择。LIBERO / DROID 的 14B 示例写 ``model/dreamzero_14b@actor.model``，LIBERO 5B 冷启动写 ``model/dreamzero_5b@actor.model``。


数据准备
--------

数据格式为 **LeRobot v2** 布局（含 ``meta/``、``data/`` 等）。训练时通过 ``data.train_data_paths`` 指定数据集根目录或 HuggingFace 数据集 ID。

LIBERO
~~~~~~

- 配置：``libero_sft_dreamzero_14b.yaml``（WAN2.1 / checkpoint）或 ``libero_sft_dreamzero_5b.yaml``（WAN2.2 冷启动）
- ``actor.model.embodiment_tag`` 必须为 ``libero_sim``
- 数据路径示例：

.. code:: yaml

   data:
     train_data_paths: physical-intelligence/libero   # 或本地 LeRobot 根目录

LeRobot / OXE DROID
~~~~~~~~~~~~~~~~~~~

- 配置：``droid_sft_dreamzero_14b.yaml``
- ``actor.model.embodiment_tag`` 必须为 ``oxe_droid``
- 建议 ``data.sampling_mode: multi_anchor``、``data.lazy_load: True``（示例已默认开启）
- 目录需符合 LeRobot DROID 布局（含 ``meta/modality.json`` 等）

.. code:: yaml

   data:
     train_data_paths: /path/to/droid_lerobot

生成 metadata.json
~~~~~~~~~~~~~~~~~~

在新数据集或冷启动（无 ``experiment_cfg/metadata.json``）时，必须先为对应 ``embodiment_tag`` 生成归一化统计：

.. code:: bash

   # LIBERO
   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset libero_sim \
     --dataset-root /path/to/libero \
     --output-metadata /path/to/metadata.json

   # DROID（多数据集可 --merge）
   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset oxe_droid \
     --dataset-root /path/to/droid \
     --output-metadata /path/to/metadata.json \
     --merge

然后在配置中设置 ``actor.model.metadata_json_path``（或放到 ``model_path/experiment_cfg/metadata.json``）。


配置说明
--------

配置文件由 Hydra 管理，入口脚本为 ``examples/sft/train_vla_sft.py``。下面按 **数据相关（``data.*``）** 与 **模型相关（``actor.model.*`` 及训练超参）** 分别说明含义与作用。

数据相关配置（``data``）
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - 含义与作用
   * - ``train_data_paths``
     - LeRobot 数据集根路径或 HF ``repo_id``。决定读哪些 episode / parquet / 视频文件。
   * - ``lazy_load``
     - 是否懒加载 mp4 视频。**``multi_anchor`` 采样模式下必须为 ``True``**（否则无法按锚点随机取帧）。
   * - ``sampling_mode``
     - ``multi_anchor``（默认，推荐）：在同一语言片段内按多个时间锚点采样，与 Groot ``lerobot_sharded`` 语义一致；宏观时间块数由 ``max_chunk_size`` 决定。**``fixed_window``**：连续固定窗口，需配合 ``num_video_frames``。
   * - ``multi_anchor_resample_attempts``
     - ``multi_anchor`` 下若某次采样无有效索引，重试次数（map-style dataloader）。
   * - ``video_backend``
     - LeRobot 视频解码后端：``pyav`` 或 ``torchcodec``，影响懒加载 mp4 的速度与兼容性，推荐使用 ``torchcodec``。
   * - ``video_tolerance_s``
     - 视频时间戳与目标帧时间的容差（秒）。
   * - ``parquet_cache_size``
     - Parquet episode 缓存上限（episode 数），影响内存与 IO。
   * - ``num_workers`` / ``prefetch_factor``
     - DataLoader 并行与预取，影响数据吞吐。

**时间对齐要点（数据采样 vs 模型块）**

- 宏观时间块数来自 ``actor.model.action_head_cfg.config.diffusion_model_cfg.max_chunk_size``（常见 **4**；官方 Groot DROID 配方可为 **5**）。
- ``actor.model.action_horizon`` 是 **DreamTransform / WAN 每个块内的动作步数**（LIBERO 常用 16，DROID 常用 24），不是数据集宏观步长。
- ``multi_anchor`` 下，数据集侧动作序列长度约为 ``action_horizon * max_chunk_size``（如 LIBERO 64、DROID 96）。
- ``actor.model.num_chunks`` 主要用于 ``fixed_window`` 连续分块；``multi_anchor`` 使用 ``max_chunk_size``，若与 ``num_chunks`` 不一致会打 warning。
- ``actor.model.num_video_frames`` 仅在 ``sampling_mode: fixed_window`` 时使用；``multi_anchor`` 下视频帧数为 ``8 * max_chunk_size + 1``（如 33）。

模型与训练相关配置（``actor``）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**标识与权重路径**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - 含义与作用
   * - ``model_type``
     - 固定为 ``dreamzero``。
   * - ``model_path``
     - 完整 checkpoint 目录；非 ``null`` 时从 ``config.json`` 读架构并加载权重。``null`` 时使用 YAML / 预设 + 各 ``*_pretrained_path`` 冷启动。
   * - ``tokenizer_path``
     - UMT5 分词器路径（训练与 collate 均需）。
   * - ``diffusion_model_pretrained_path``
     - 因果 DiT（扩散骨干）预训练权重；冷启动必填。
   * - ``image_encoder_pretrained_path``
     - WAN 图像编码器；WAN2.2 需指向 **WAN2.1 CLIP** 权重。
   * - ``text_encoder_pretrained_path``
     - T5 文本编码器权重。
   * - ``vae_pretrained_path``
     - VAE 权重；WAN2.2 对应 ``WanVideoVAE38``。
   * - ``metadata_json_path``
     - 数据集 ``metadata.json``；未设置则回退到 ``model_path/experiment_cfg/metadata.json``。
   * - ``embodiment_tag``
     - 选择数据变换与 collate 模板：``libero_sim`` 或 ``oxe_droid``。**必须与数据集一致。**

**时序与动作形状（需与数据、WAN 容量一致）**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - 含义与作用
   * - ``action_horizon``
     - 每个 WAN 时间块内的动作步数（LIBERO 16，DROID 24）。
   * - ``state_horizon``
     - 每个样本的状态行数（通常为 1，每个宏观锚点一个状态）。
   * - ``num_chunks``
     - ``fixed_window`` 模式下的连续块数；``multi_anchor`` 下以 ``max_chunk_size`` 为准。
   * - ``num_action_per_block``
     - 与 ``action_head_cfg`` 中 DiT 的 ``num_action_per_block`` 对齐（常等于 ``action_horizon``）。
   * - ``action_head_cfg...diffusion_model_cfg.max_chunk_size``
     - 多锚点宏观时间块数 / Causal DiT 容量；与 ``data.sampling_mode: multi_anchor`` 强相关。
   * - ``num_video_frames``
     - 仅 ``fixed_window`` 有效。
   * - ``max_action_dim`` / ``max_state_dim`` / ``max_seq_len``
     - DreamTransform 填充与文本序列上限。

**视频尺寸与 DROID 特有项**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - 含义与作用
   * - ``target_video_height`` / ``target_video_width``
     - WAN 策略头目标分辨率（5B 预设如 176×320；可在 YAML 覆盖）。避免在 transform 代码里写死尺寸，以兼容 WAN2.1 / WAN2.2。
   * - ``droid_view_height`` / ``droid_view_width``
     - （可选）DROID 各视角 resize 覆盖。
   * - ``relative_action`` / ``relative_action_keys`` / ``relative_action_per_horizon``
     - 是否使用相对动作及作用维度；DROID 常对 ``joint_position`` 等开启 ``relative_action: True``。

**其它模型训练项**

- ``precision``：Actor / Optimizer 侧的主精度设置（``fp32`` / ``bf16``）。**推荐 ``fp32``**，并配合 ``actor.fsdp_config.mixed_precision`` 做混合精度训练：``precision: fp32`` 使 **优化器状态与主参数保持 FP32**（数值更稳），前向/反向的实际矩阵运算由 FSDP 在 ``mixed_precision`` 中降为 **BF16**（省显存、提速）。示例：

  .. code:: yaml

     actor:
       model:
         precision: fp32
       fsdp_config:
         mixed_precision:
           param_dtype: bf16
           reduce_dtype: bf16
           buffer_dtype: bf16

  若将 ``precision`` 设为 ``bf16``，优化器也会以较低精度维护状态，一般不如上述组合稳定。启用 FSDP **CPU offload** 时，通常保持 ``precision: fp32``。
- ``is_lora``：是否 LoRA 微调（DreamZero SFT 示例多为全参 ``False``）。
- ``actor.micro_batch_size`` / ``actor.global_batch_size``：每 GPU 微批与全局有效 batch（需能被 GPU 数整除关系约束）。
- ``actor.optim.*``：学习率、warmup、cosine 等。
- ``actor.fsdp_config``：FSDP2 分片、梯度检查点；``mixed_precision`` 控制计算/通信 dtype（与 ``actor.model.precision`` 配合，见上）。

**配置示例对照**

.. code:: yaml

   # ---------- 数据 ----------
   data:
     train_data_paths: /path/to/libero
     lazy_load: True
     sampling_mode: multi_anchor
     video_backend: torchcodec
     num_workers: 8

   # ---------- 模型（从 checkpoint 继续）----------
   actor:
     model:
       model_path: /path/to/DreamZero-DROID
       tokenizer_path: /path/to/umt5-xxl
       embodiment_tag: oxe_droid
       action_horizon: 24
       metadata_json_path: /path/to/metadata.json   # 若无 experiment_cfg/metadata.json

启动训练
--------

在仓库根目录执行：

.. code:: bash

   # LIBERO + WAN2.1（checkpoint，dreamzero_14b 预设）
   bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_14b

   # LIBERO + WAN2.2（冷启动，dreamzero_5b 预设）
   bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_5b

   # DROID + WAN2.1（dreamzero_14b 预设，model_path 指向 DreamZero-DROID）
   bash examples/sft/run_vla_sft.sh droid_sft_dreamzero_14b

脚本等价于：

.. code:: bash

   python examples/sft/train_vla_sft.py \
     --config-path examples/sft/config/ \
     --config-name <config_name> \
     runner.logger.log_path=<自动生成的日志目录>

日志目录：

- ``<repo>/logs/<时间戳>-<config_name>/run_embodiment.log``

断点续训可设置 ``runner.resume_dir`` 指向 checkpoint 目录。


监控与 sanity check
-------------------

1. 查看 ``run_embodiment.log``：``time/step`` 是否稳定；``train/loss``、``train/action_loss``、``train/dynamics_loss`` 是否合理。

2. TensorBoard：

.. code:: bash

   tensorboard --logdir ./logs --port 6006

3. 开跑后尽早检查：

   - ``images`` / ``state`` / ``action`` 的 shape、dtype、数值范围
   - ``state_mask`` / ``action_mask`` / ``text_attention_mask`` 有效比例
   - WAN2.2 时确认输入分辨率与 ``frame_seqlen`` 与 ``config.json`` 或预设一致


扩展：新增 ``embodiment_tag``
-----------------------------

当要在 **新的机器人 / 新 LeRobot 数据集** 上训练 DreamZero SFT 时，需要新增一个 ``embodiment_tag``，并在 RLinf 中注册对应的数据变换与元数据生成逻辑。建议以现有实现为模板对照修改：

- ``rlinf/data/datasets/dreamzero/data_transforms/libero_sim.py``（双视角、简单 state/action 列）
- ``rlinf/data/datasets/dreamzero/data_transforms/oxe_droid.py``（三视角、``meta/modality.json`` 切片）

整体数据流：

.. code:: text

   LeRobot 数据集
        → DreamZeroLeRobotDataset（按 transform 链里的 keys 读 parquet/mp4）
        → ComposedModalityTransform + DreamTransform（归一化、多视角拼接、tokenize）
        → DreamZeroCollator → 训练

步骤 1：实现 embodiment 变换模块
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``rlinf/data/datasets/dreamzero/data_transforms/`` 下新建 ``<your_tag>.py``，实现 ``DreamZeroEmbodimentTransform`` 协议（见 ``base.py``），至少包含：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 成员 / 方法
     - 说明
   * - ``TAG``
     - 字符串标识，与配置里 ``actor.model.embodiment_tag``、``metadata.json`` 顶层键**完全一致**。
   * - ``DEFAULT_TAG_MAPPING``
     - ``{TAG: <int>}``，映射到 WAN 动作头里的 **embodiment projector ID**。继续微调已有 DreamZero 权重时，ID 须出现在 checkpoint ``config.json`` 的 ``action_loss_embodiment_ids`` 中（如 5B 预设含 17、21、26）；**全新 ID** 需接受 projector 随机初始化或改模型配置。
   * - ``DEFAULT_ACTION_HORIZON``
     - 该 embodiment 默认每块动作步数（LIBERO 16、DROID 24），与 ``actor.model.action_horizon`` 一致。
   * - ``get_modality_config()``
     - 返回 ``video`` / ``state`` / ``action`` / ``language`` 的 ``ModalityConfig``（``delta_indices``、``modality_keys``）。``language`` 的 key 必须在数据集中存在（任务文本列）。视频/动作 ``delta_indices`` 需与 Groot 配方一致（现实现多为 video ``range(25)``、action ``range(24)``），否则 ``multi_anchor`` 时间对齐会错。
   * - ``get_transform(...)``
     - 组装 ``Video*`` → ``StateAction*`` → ``ConcatTransform`` → ``DreamTransform`` 链；``DreamTransform`` 使用 RLinf 子类（``dream_transform.py``），会从 registry 调用多视角拼接。
   * - ``format_training_prompt(instruction)``
     - 为多视角布局生成 T5 文本前缀（须与 Groot 训练模板语义一致）。
   * - ``concat_multiview_video(images)``
     - 将 ``(v, t, c, h, w)`` 拼成 ``(1, t, c, H, W)``；布局须与 ``format_training_prompt`` 描述一致。

**``modality_keys`` 命名约定**（与 ``DreamZeroLeRobotDataset`` 解析逻辑挂钩）：

- 视频：``video.<short_name>``（如 ``video.image``），短名通过 ``meta/modality.json`` 的 ``original_key`` 或 ``info.json`` 的 ``observation.images.*`` / 裸列名解析到真实特征列。
- 状态/动作：``state.<name>``、``action.<name>``；有 ``meta/modality.json`` 时用 ``start``/``end`` 切片；否则回退到 ``observation.state`` / ``action`` 整列或启发式切片（见 ``dreamzero.py`` 中 ``_build_component_sources``）。
- 训练 YAML 里的 ``video.*`` / ``state.*`` / ``action.*`` 必须与 transform 里 ``ConcatTransform`` 的 ``*_concat_order`` 一致。

步骤 2：注册到 RLinf
~~~~~~~~~~~~~~~~~~~~

编辑 ``rlinf/data/datasets/dreamzero/data_transforms/__init__.py``：

1. ``from ...<your_tag> import YourEmbodimentDataTransform``
2. 在 ``_EMBODIMENT_REGISTRY`` 中加入 ``YourEmbodimentDataTransform.TAG: YourEmbodimentDataTransform``

未注册时，``build_dreamzero_composed_transform`` 会报错并列出已有 tag。

步骤 3：生成 ``metadata.json``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

为新数据集计算归一化统计，输出键名必须等于 ``TAG``：

**方式 A（推荐）**：在 ``toolkits/lerobot/generate_dreamzero_metadata.py`` 的 ``PRESETS`` 中增加一项（字段参考 ``libero_sim`` / ``oxe_droid``：``state_key``、``action_key``、``video_keys``、``use_modality_json``），然后：

.. code:: bash

   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset <your_tag> \
     --dataset-root /path/to/lerobot_dataset \
     --output-metadata /path/to/metadata.json

**方式 B**：不改脚本，用手动参数（``--embodiment-tag``、``--state-key``、``--action-key``、``--video-keys``、``--use-modality-json``）。

在训练配置中设置 ``actor.model.metadata_json_path``（或放到 ``model_path/experiment_cfg/metadata.json``）。

步骤 4：编写 / 调整训练配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

复制 ``libero_sft_dreamzero_14b.yaml``、``libero_sft_dreamzero_5b.yaml`` 或 ``droid_sft_dreamzero_14b.yaml``，至少修改：

.. code:: yaml

   data:
     train_data_paths: /path/to/your_lerobot
     lazy_load: True              # multi_anchor 必须为 True（mp4 数据）
     sampling_mode: multi_anchor

   actor:
     model:
       embodiment_tag: "<your_tag>"
       metadata_json_path: /path/to/metadata.json
       action_horizon: <与 DEFAULT_ACTION_HORIZON 一致>
       # 从 checkpoint 继续时核对 action_loss_embodiment_ids 是否包含你的 projector ID
       target_video_height: ...
       target_video_width: ...
       relative_action: ...
       relative_action_keys: [...]

若冷启动 WAN，在 ``examples/sft/config/model/dreamzero_5b.yaml``（或 14b）的 ``action_head_cfg.config.action_loss_embodiment_ids`` 中加入新 ID。

步骤 5：验证（短跑 + 数据检查）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 单独跑 metadata 脚本，确认 ``metadata.json[<your_tag>]`` 中 ``statistics`` / ``modalities`` 维度与 parquet 一致。
2. 用 50–200 step 启动 SFT，检查日志无 ``Could not map transform video keys``、``embodiment_tag not found in metadata`` 等错误。
3. 在 TensorBoard / 日志中确认 ``train/action_loss`` 有限；检查 batch 内 ``images`` 拼接形状、``embodiment_id`` 与 ``DEFAULT_TAG_MAPPING`` 一致。

**易错细节 checklist**

- ``embodiment_tag`` 字符串在三处一致：**配置**、**metadata.json 键**、**Python ``TAG``**。
- ``multi_anchor`` + mp4 数据：**必须** ``data.lazy_load: True``。
- ``action_horizon`` × ``max_chunk_size`` 决定数据集动作长度；勿只改其一。
- 多视角 **拼接顺序** 与 **prompt 文案** 不一致会导致训练信号错乱。
- 继续微调官方权重时，随意改 ``DEFAULT_TAG_MAPPING`` 的整数 ID 会导致 projector 对不上。
- 视频 resize：优先在 transform 链或 ``target_video_height/width`` 配置，避免写死尺寸导致 WAN2.1/2.2 不兼容。
- 推理 / 评测：``examples/embodiment/config/*_dreamzero.yaml`` 中同样需要正确的 ``embodiment_tag``。

若仅推理、不改 RLinf 代码，且 Groot/DreamZero 上游已支持该 tag，有时只需准备 ``metadata.json`` 与评测配置；**SFT 新数据** 则通常必须完成上述 Python 注册与 transform 实现。


常见问题
--------

1. **找不到权重（No safetensors weights）**

   - 检查 ``model_path`` 下是否存在 ``model.safetensors`` 或分片索引
   - 冷启动时确认各 ``*_pretrained_path`` 可访问且与架构匹配

2. **WAN2.2 维度不匹配**

   - 核对有效配置（``model_path/config.json`` 或 ``dreamzero_5b`` 预设）中 ``diffusion_model_cfg`` 是否为 ti2v、``in_dim/out_dim=48``、``vae_cfg`` 为 ``WanVideoVAE38``
   - 图像编码器须使用 WAN2.1 CLIP 路径

3. **metadata.json 找不到**

   - 运行 ``toolkits/lerobot/generate_dreamzero_metadata.py`` 并设置 ``metadata_json_path``
   - 确认 JSON 内包含与 ``embodiment_tag`` 同名的键

4. **action_loss 异常偏高**

   - 检查归一化统计是否与当前数据集一致
   - 检查 ``relative_action`` 与数据是否冲突
   - 核对 ``action_horizon``、``max_chunk_size`` 与 ``sampling_mode`` 是否匹配

5. **DROID 视频尺寸错误**

   - 勿在代码中写死分辨率；使用 ``target_video_height/width`` 或 ``droid_view_*`` 配置项

6. **multi_anchor 报错要求 lazy_load**

   - 设置 ``data.lazy_load: True``


实践建议
--------

- 追求稳定收敛时，优先从已发布的 DreamZero 权重继续 SFT（设置 ``model_path``）。
- 全量适配 WAN2.2 可冷启动，但需更大数据与更长训练；改配置后先用 50–200 step 试跑验证 shape 与 loss。
- 每次更换数据集或 ``embodiment_tag``，务必重新生成或更新 ``metadata.json``。
- LIBERO 与 DROID 的 ``action_horizon``、``embodiment_tag``、多视角拼接逻辑不同，不要混用配置模板。
