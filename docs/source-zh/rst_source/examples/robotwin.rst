基于RoboTwin模拟器的强化学习训练
=============================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档给出在 RLinf 框架内启动与管理 **Vision-Language-Action Models (VLAs)** 训练任务的完整指南，
在 RoboTwin 环境中微调 VLA 模型以完成机器人操作。

主要目标是让模型具备以下能力：

1. **视觉理解**：处理来自机器人相机的 RGB 图像。  
2. **语言理解**：理解自然语言的任务描述。  
3. **动作生成**：产生精确的机器人动作（位置、旋转、夹爪控制）。  
4. **强化学习**：结合环境反馈，使用 PPO 优化策略。

RLinf RoboTwinEnv 环境介绍
-----------------------

**RLinf RoboTwinEnv 环境**

- **Environment**：RLinf 框架基于 RoboTwin 2.0 仿真环境提供了用于强化学习训练的 RoboTwinEnv 环境。  
- **Task**：控制机械臂完成多种操作任务。RLinf RoboTwinEnv 目前支持以下 **20 个任务** （均已实现 dense reward 奖励函数），用户可以根据需要选择任务进行训练。

  **放置类任务（Place Tasks）**
  
  - ``place_empty_cup``：放置空杯到杯垫上
  - ``place_shoe``：放置单只鞋子
  - ``place_dual_shoes``：放置两只鞋子
  - ``place_fan``：放置风扇
  - ``place_bread_skillet``：将面包放置到平底锅上
  - ``place_a2b_left``：从左到右放置物体
  - ``place_a2b_right``：从右到左放置物体

  **抓取类任务（Pick Tasks）**
  
  - ``pick_dual_bottles``：抓取两个瓶子
  - ``pick_diverse_bottles``：抓取多样化的瓶子

  **堆叠类任务（Stack Tasks）**
  
  - ``stack_blocks_two``：堆叠两个积木
  - ``stack_blocks_three``：堆叠三个积木
  - ``stack_bowls_two``：堆叠两个碗
  - ``stack_bowls_three``：堆叠三个碗

  **排序类任务（Ranking Tasks）**
  
  - ``blocks_ranking_rgb``：按 RGB 颜色排序积木
  - ``blocks_ranking_size``：按大小排序积木

  **交互类任务（Interaction Tasks）**
  
  - ``click_alarmclock``：点击闹钟
  - ``click_bell``：点击铃铛
  - ``beat_block_hammer``：用锤子敲击积木
  - ``adjust_bottle``：调整瓶子位置

  .. note::
     更多任务正在持续开发中。RoboTwin 平台计划支持超过 50 个任务，dense reward 奖励函数的实现将逐步扩展到所有任务。

- **Observation**：RLinf RoboTwinEnv 环境返回的观测信息是一个字典（dict），包含以下字段：

  - ``images``：头部相机 RGB 图像

    - **类型**：``torch.Tensor``
    - **形状**：``[batch_size, 224, 224, 3]``
    - **数据类型**：``uint8`` （0-255）
    - **说明**：经过中心裁剪（center crop）处理的头部相机图像，每个环境返回一张图像

  - ``wrist_images``：腕部相机 RGB 图像（可选）
  
    - **类型**：``torch.Tensor`` 或 ``None``
    - **形状**：``[batch_size, num_wrist_images, 224, 224, 3]`` （如果存在）
    - **数据类型**：``uint8`` （0-255）
    - **说明**：可能包含左腕相机（``left_wrist_image``）和/或右腕相机（``right_wrist_image``）的图像，如果任务不需要腕部图像则为 ``None``

  - ``states``：本体感觉信息（proprioception）

    - **类型**：``torch.Tensor``
    - **形状**：``[batch_size, 14]``
    - **数据类型**：``float32``
    - **说明**：包含末端执行器的位姿信息（位置和姿态），共 14 维，对应 ``proprio_dim=14``

  - ``task_descriptions``：任务描述文本

    - **类型**：``List[str]``
    - **长度**：``batch_size``
    - **说明**：每个环境对应的自然语言任务描述，例如 "What action should the robot take to place the empty cup on the coaster?"

- **Action Space**：14 维连续动作空间

  - **类型**：``torch.Tensor`` 或 ``numpy.ndarray``
  - **形状**：``[batch_size, action_dim]`` 或 ``[batch_size, horizon, action_dim]``，其中 ``action_dim=14``
  - **数据类型**：``float32``
  - **动作组成**：

    - 末端执行器三维位置控制（x, y, z）：3 维
    - 三维旋转控制（roll, pitch, yaw）：3 维
    - 夹爪控制（开/合）：1 维
    - 关节位置控制：7 维
    - **总计**：14 维

依赖安装
-----------------------

RLinf 提供了两种安装方式：**Docker 镜像** （推荐，最简单）和**手动安装** （使用安装脚本）。

方式 1：使用 Docker 镜像（推荐）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 提供了预配置的 RoboTwin 环境 Docker 镜像，镜像中已包含所有必需的依赖，可以直接使用，**无需进行后续安装步骤**。

.. code-block:: bash

   # 拉取 RoboTwin 环境镜像
   docker pull rlinf/robotwin:latest
   
   # 运行容器（基础用法）
   docker run -it --gpus all rlinf/robotwin:latest
   
   # 运行容器（挂载数据目录，推荐）
   docker run -it --gpus all \
     -v /path/to/robotwin_assets:/opt/robotwin_assets \
     -v /path/to/models:/opt/models \
     -v /path/to/results:/opt/results \
     rlinf/robotwin:latest

.. note::
   Docker 镜像已包含：
   
   - RLinf RoboTwinEnv 环境
   - ``embodied`` 和 ``robotwin`` extra 依赖
   - RoboTwin 平台相关依赖
   - 兼容性补丁已应用
   - 支持 OpenVLA、OpenVLA-OFT、OpenPI 模型

   **使用 Docker 镜像后，可以直接跳转到** `模型下载`_ **和** `运行脚本`_ **章节，无需进行后续安装步骤。**

方式 2：手动安装
~~~~~~~~~~~~~~~~

如果您需要在本地环境安装，可以使用以下两种方法：

**方法 2.1：使用安装脚本（推荐）**

使用 ``requirements/install.sh`` 脚本，通过 ``--robotwin`` 参数安装 RoboTwin 环境。根据您要训练的模型，将第一个参数替换为对应的模型名称（``openvla``、``openvla-oft`` 或 ``openpi``）：

.. code-block:: bash

   # 创建一个虚拟环境
   uv venv --python=3.11 --name openvla-oft-robotwin
   source ./venv/openvla-oft-robotwin/bin/activate

   # 以 OpenVLA-OFT 为例（用于 RoboTwin）
   bash requirements/install.sh openvla-oft --robotwin

该脚本会自动完成：

- 安装 ``embodied`` 和 ``robotwin`` extra 依赖
- 安装 RoboTwin 平台相关依赖
- 应用 RoboTwin 兼容性补丁（修复 sapien 和 mplib 的兼容性问题）
- 安装对应 VLA 模型的依赖包

**方法 2.2：完全手动安装**

如果您想完全手动控制安装过程，可以按以下步骤操作：

.. code-block:: bash

   # 步骤 1: 创建一个虚拟环境
   uv venv --python=3.11 --name openvla-oft-robotwin
   source ./venv/openvla-oft-robotwin/bin/activate

   # 步骤 2: 安装 RLinf 基础依赖和 RoboTwin extra
   uv pip install -e ".[embodied,robotwin]"
   
   # 步骤 3: 安装 embodied 环境的系统依赖
   bash requirements/install_embodied_deps.sh
   
   # 步骤 4: 应用 RoboTwin 兼容性补丁
   bash requirements/patch_sapien_mplib_for_robotwin.sh
   
   # 步骤 5: 根据使用的模型安装对应依赖（以 OpenVLA-OFT 为例）
   # OpenVLA-OFT:
   uv pip install -r requirements/openvla_oft.txt

.. note::
   **依赖冲突说明**：``mplib==0.2.1`` 是 RoboTwin 必需的，但与 ManiSkill 存在冲突。
   如果您同时需要 ManiSkill 和 RoboTwin，建议：
   
   - 使用不同的虚拟环境分别运行
   - 或者先安装 ``embodied``，然后根据需要使用 ``robotwin`` extra

.. note::
   **兼容性补丁说明**：补丁脚本会修复以下问题：
   
   - ``sapien/wrapper/urdf_loader.py`` 中的编码问题
   - ``mplib/planner.py`` 中的碰撞检测逻辑
   
   如果使用 ``install.sh`` 脚本安装，补丁会自动应用，无需手动运行。

Assets 下载
-----------------------

RoboTwin Assets 是 RoboTwin 环境所需的资产文件，需要从 HuggingFace 下载。

.. code-block:: bash

   # 1. 克隆 RoboTwin 仓库
   git clone https://github.com/RoboTwin-Platform/RoboTwin.git
   cd RoboTwin
   
   # 2. 下载并解压 Assets 文件
   bash script/_download_assets.sh
   
   # 3. 更新配置文件路径
   python script/update_embodiment_config_path.py


模型下载
-----------------------

在开始训练之前，您需要下载相应的SFT模型：

.. code-block:: bash

   # 下载 OpenVLA-OFT 模型
   pip install huggingface-hub
   huggingface-cli download RLinf/RLinf-OpenVLAOFT-RoboTwin

下载后，请确保在配置 yaml 文件中正确指定模型路径（``actor.model.model_dir``）。

运行脚本
-------------------

**1. 关键参数配置**

以 OpenVLA-OFT 模型为例，在 ``actor.model`` 中需要配置以下关键参数：

.. code-block:: yaml

   actor:
     model:
       model_dir: "/path/to/RLinf-OpenVLAOFT-RoboTwin"  # SFT 模型路径
       model_name: "openvla_oft"
       action_dim: 14                                    # RoboTwin 动作维度（14维）
       use_proprio: True                                 # 是否使用本体感觉信息
       proprio_dim: 14                                   # 本体感觉维度（需与 action_dim 一致）
       use_film: False                                   # 是否使用 FiLM 层
       num_images_in_input: 1                            # 输入图像数量
       num_action_chunks: 25                             # 动作块数量
       unnorm_key: "place_empty_cup"                     # 动作归一化键（需与SFT训练时使用的unnorm_key一致）


**2. 环境配置**

在环境配置文件中，需要设置以下关键参数：

.. code-block:: yaml

   env/train: robotwin_single_task
   env/eval: robotwin_single_task
   
   # 在 env/train/robotwin_single_task.yaml 中：
   simulator_type: robotwin
   assets_path: "/path/to/robotwin_assets"
   
   task_config:
     task_name: place_empty_cup  # 或其他任务名称
     step_lim: 200
     embodiment: [piper, piper, 0.6]
     camera:
       head_camera_type: D435
       wrist_camera_type: D435
       collect_head_camera: true
       collect_wrist_camera: false

**3. 配置文件**

支持 **OpenVLA-OFT** 模型，算法为 **PPO**。  
对应配置文件：

- **OpenVLA-OFT + PPO**：``examples/embodiment/config/robotwin_ppo_openvlaoft.yaml``

**4. 启动命令**

选择配置后，运行以下命令开始训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，在 RoboTwin 环境中使用 PPO 训练 OpenVLA-OFT 模型：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh robotwin_ppo_openvlaoft

可视化与结果
-------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 视频生成**

训练和评估过程中的视频会自动保存。配置如下：

.. code-block:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train  # 训练视频
     # 或
     video_base_dir: ${runner.logger.log_path}/video/eval   # 评估视频


配置说明
-----------------------

**关键配置参数**

1. **模型配置**：

   - ``actor.model.model_name: "openvla_oft"``：使用 OpenVLA-OFT 模型
   - ``actor.model.action_dim: 14``：14 维动作空间（包含本体感觉）
   - ``actor.model.use_proprio: True``：启用本体感觉输入
   - ``actor.model.proprio_dim: 14``：本体感觉维度
   - ``actor.model.num_action_chunks: 25``：动作块数量

2. **算法配置**：

   - ``algorithm.reward_type: chunk_level``：chunk 级别的奖励
   - ``algorithm.logprob_type: token_level``：token 级别的对数概率
   - ``algorithm.n_chunk_steps: 8``：每个 chunk 的步数

3. **环境配置**：

   - ``env.train.task_config.task_name``：任务名称（如 ``place_empty_cup``）
   - ``env.train.task_config.embodiment``：机器人配置
   - ``env.train.task_config.camera``：相机配置

更多关于 RoboTwin 配置的详细信息，请参考 `RoboTwin 配置文档 <https://robotwin-platform.github.io/doc/usage/configurations.html>`_。

注意事项
-----------------------

1. **资源路径**：确保 ``assets_path`` 路径正确
2. **环境变量**：确保将 RoboTwin repo路径加在 PYTHONPATH 中，如 ``export PYTHONPATH=/opt/robotwin:$PYTHONPATH``
3. **GPU 内存**：RoboTwin 环境可能需要较多 GPU 内存，建议使用 ``enable_offload: True``
4. **任务配置**：根据具体任务修改 ``task_config`` 中的参数

