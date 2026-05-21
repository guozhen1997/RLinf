DreamZero 监督微调训练
===========================

本文档介绍如何在 RLinf 框架内进行 DreamZero 模型的监督微调（SFT），覆盖当前常见组合：

- 数据集格式：**LIBERO**、**LeRobot/OXE DROID**
- 视觉主干：**WAN2.1**、**WAN2.2**

目标是帮助你从「数据准备 → 权重准备 → 配置编写 → 启动训练 → 排错」完整跑通流程。


常见起训方式：

- **从 DreamZero 已发布权重继续 SFT**：收敛更快，稳定性更好
- **从 WAN2.2 基模冷启动**：灵活但更吃数据规模与调参


训练入口
----------

统一使用脚本：

- ``examples/sft/run_vla_sft.sh``

脚本最终调用：

.. code:: bash

   python examples/sft/train_vla_sft.py \
     --config-path examples/sft/config/ \
     --config-name <配置名> \
     runner.logger.log_path=<自动生成日志目录>

日志会输出到：

- ``<repo>/logs/<timestamp>/run_embodiment.log``


依赖与环境
------------

1. 克隆 RLinf，并安装 DreamZero 相关依赖环境（embodied + dreamzero）
2. 设置 ``DREAMZERO_PATH`` 指向 DreamZero 代码目录（供 transform / model 组件导入）
3. 确保 ``run_vla_sft.sh`` 中 ``PYTHONPATH`` 包含：

   - RLinf 仓库路径
   - DreamZero 仓库路径

4. 推荐使用与项目一致的 CUDA / PyTorch / transformer 版本组合


数据准备
----------

- 配置文件：``libero_sft_dreamzero.yaml``
- 字段：``data.train_data_paths``

示例：

.. code:: yaml

   data:
     train_data_paths: "path/to/datasets"


模型与权重准备
----------------

Hydra 通过 ``defaults`` 组装 ``actor.model``（例如 ``libero_sft_dreamzero_5b.yaml`` 中的 ``model/dreamzero_5b@actor.model``），训练配置可直接覆盖 ``actor.model`` 下任意字段。

**策略架构**

- ``actor.model.model_path`` **已设置**：读取 ``<model_path>/config.json``（``actor.model`` 上其余架构字段不参与策略配置，并会打 warning）。
- ``actor.model.model_path`` 为 **``null``**：使用 Hydra 解析后的 ``actor.model``；以下路径必须非空：``tokenizer_path``、``diffusion_model_pretrained_path``、``image_encoder_pretrained_path``、``text_encoder_pretrained_path``、``vae_pretrained_path``。

``examples/sft/config/model/`` 下的预设（如 ``dreamzero_5b.yaml``）通过 Hydra ``defaults`` 引入，**没有**单独的 ``dreamzero_*`` 覆盖键。

数据变换链在 Python 中按 ``embodiment_tag`` 组装；分辨率、``DreamTransform`` 的 ``state_horizon`` / ``action_horizon``
等可通过 ``actor.model`` 上的 ``video_*``、``state_horizon`` / ``action_horizon``、``max_state_dim`` 等字段覆盖（见 ``rlinf/data/datasets/dreamzero/data_transforms/__init__.py``）。
SFT 数据加载器默认使用 **multi_anchor** 时间采样（``data.sampling_mode: multi_anchor``，与 Groot ``lerobot_sharded`` 语义一致）；宏观时间块数来自 ``diffusion_model_cfg.max_chunk_size``。``action_horizon`` 仅用于 ``DreamTransform`` / WAN。设 ``data.sampling_mode: fixed_window`` 可恢复旧版连续窗口（``action_horizon * num_chunks``）。

统计量来自数据集生成的 metadata.json，通过 ``metadata_json_path`` 指定路径，
或放在 ``model_path/experiment_cfg/metadata.json``（按 ``embodiment_tag`` 键索引）。

WAN2.1 路径（常规）
~~~~~~~~~~~~~~~~~~~~

若使用 **完整** DreamZero checkpoint 目录（例如官方发布目录），将 ``model_path`` 指向该目录。目录中通常包含：

- ``config.json``
- ``experiment_cfg/metadata.json``（数据集统计；RLinf **不**读取 ``conf.yaml``）
- ``model.safetensors``（或分片 safetensors）

设置 ``model_path`` 时，RLinf 从 ``config.json`` 读取 **config**，从 ``experiment_cfg/metadata.json`` 读取 **metadata**（若未设置 ``metadata_json_path``）。**数据变换**仍由 Python 按 ``embodiment_tag`` 组装。

示例：

.. code:: yaml

   actor:
     model:
       model_type: "dreamzero"
       model_path: /path/to/models/DreamZero-DROID
       tokenizer_path: /path/to/models/umt5-xxl


WAN2.2 路径
~~~~~~~~~~~~

WAN2.2 训练通常需要下列组件权重：

- WAN2.2 主干（DiT + T5 + VAE）
- WAN2.1 的 CLIP 图像编码器（WAN2.2-TI2V-5B 不含该文件）
- tokenizer（umt5-xxl）

无论 ``config.json`` 来自 checkpoint 还是 RLinf 预设/覆盖，**架构**需与权重一致，例如：

- ``diffusion_model_cfg``：``model_type=ti2v``、``in_dim=48``、``out_dim=48``、``frame_seqlen=50`` 等
- ``vae_cfg``：``WanVideoVAE38``
- ``image_encoder_pretrained_path``：加载 WAN 组件时指向 WAN2.1 CLIP 权重


关键配置说明（DreamZero）
--------------------------

以下字段在 DreamZero SFT 中最关键：

- ``actor.model.model_type``：固定 ``dreamzero``
- ``actor.model.model_path``：checkpoint 目录（``null`` 则仅用 Hydra ``actor.model``）。已设置时从该路径读 ``config.json``；metadata 来自 ``metadata_json_path`` 或 ``experiment_cfg/metadata.json``。**数据变换链**始终由 Python 按 ``embodiment_tag`` 组装（见 ``rlinf/data/datasets/dreamzero/data_transforms/__init__.py``）。
- ``actor.model.tokenizer_path``：文本 tokenizer 路径
- ``actor.model.embodiment_tag``："oxe_droid" 或 "libero_sim"
- ``data.sampling_mode``：``multi_anchor``（语言段内多锚点采样）或 ``fixed_window``（旧版连续窗口）。
- ``actor.model.action_head_cfg.config.diffusion_model_cfg.max_chunk_size``：multi_anchor 宏观时间块数（与 WAN Causal DiT 一致，常见为 **4**；需与数据配方一致，例如官方 Groot DROID 为 5 时改此字段）。
- ``actor.model.action_horizon``：仅用于 **DreamTransform / WAN 每块**（LIBERO 16，DROID 24）。multi_anchor 数据集动作长度通常为 ``action_horizon * max_chunk_size``（如 LIBERO 64、DROID 96），不是 ``action_horizon * num_chunks``。
- ``actor.model.num_chunks``：fixed_window 模式连续块数；multi_anchor 下与 ``max_chunk_size`` 不一致时仅打警告，偏移以 ``max_chunk_size`` 为准。
- ``actor.model.state_horizon``：每个样本的状态行数（通常为 1，每个宏观锚点一行 state）。
- ``actor.model.num_video_frames``：仅在 ``data.sampling_mode: fixed_window`` 时使用（multi_anchor 约 ``8 * max_chunk_size + 1`` 帧，如 33）。
- ``data.multi_anchor_resample_attempts``：multi_anchor 采样为空时在 map-style dataloader 中的重试次数。
- ``data.video_backend``：lazy 加载 mp4 时使用的 LeRobot 视频解码后端（``pyav``、``torchcodec``，默认 ``pyav``）。
- ``actor.model.droid_view_height`` / ``droid_view_width``：DROID 每路 resize 目标（可选）
- ``actor.model.relative_action``：是否使用相对动作
- ``actor.fsdp_config``：FSDP 训练策略

兼容 WAN2.1/WAN2.2 的数据尺寸建议：

- 不要在代码里写死单一路径的分辨率
- 通过模型配置自动推断，或在 yaml 中显式覆盖（例如 per-view resize 高宽）


启动训练
----------

在仓库根目录执行：

.. code:: bash

   # LIBERO
   bash examples/sft/run_vla_sft.sh libero_sft_dreamzero
   # DROID
   bash examples/sft/run_vla_sft.sh droid_sft_dreamzero


监控与检查
------------

1. 观察 ``run_embodiment.log``：

   - ``time/step`` 是否稳定
   - ``train/loss``、``train/action_loss``、``train/dynamics_loss`` 是否可解释

2. TensorBoard：

.. code:: bash

   tensorboard --logdir ./logs

3. 首轮建议做数据对齐检查：

   - ``images/state/action`` 的 shape、dtype、范围
   - ``state_mask/action_mask/text_attention_mask`` 有效位比例
   - WAN2.2 路径重点关注输入分辨率与 ``frame_seqlen`` 一致性


常见问题
----------

1. **找不到模型权重（No safetensors weights）**

- 检查 ``model_path`` 下是否存在 ``model.safetensors`` 或分片索引
- 若走 WAN 基模冷启动，确认加载逻辑支持无全量 safetensors 模式

2. **WAN2.2 维度不匹配**

- 检查最终生效配置（``model_path/config.json`` 或 Hydra ``actor.model``）里 ``diffusion_model_cfg`` 是否为 WAN2.2 参数
- 检查 ``vae_cfg`` 是否使用 ``WanVideoVAE38``

3. **action_loss 异常偏高**

- 检查 action/state 归一化与 relative_action 配置是否重复或冲突
- 检查数据时序（horizon/chunk）是否与模型配置一致

4. **DROID 视频尺寸不一致**

- 避免代码写死尺寸
- 使用配置推断或显式覆盖，保持 WAN2.1 与 WAN2.2 兼容


实践建议
----------

- 想快速稳定收敛：优先从 DreamZero 已有权重继续 SFT
- 想做完整 WAN2.2 适配：可从基模冷启动，但需更多数据与更长训练
- 每次改配置后，先跑小步数（如 50~200 step）确认 shape / loss / 吞吐正常，再长训

