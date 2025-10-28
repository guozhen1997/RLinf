基于 Behavior 模拟器的强化学习
==============================

Behavior 1K 是一个包含 1000 个真实世界操作任务的数据集。

Behavior 使用 IsaacSim 作为仿真平台。

环境
-----------------------

**Behavior 环境**

- **环境**: 基于 *IsaacSim* 构建的 Behavior 仿真基准测试。
- **任务**: 控制双臂 R1 Pro 机器人执行各种家庭操作技能（抓取放置、堆叠、打开抽屉、空间重排）。
- **观察**: 由机器人搭载的传感器捕获的多相机 RGB 图像：
  - **头部相机**: 提供 224×224 RGB 图像用于全局场景理解
  - **手腕相机**: 左右 RealSense 相机提供 224×224 RGB 图像用于精确操作
- **动作空间**: 23 维连续动作（3-DOF (x,y,rz) 关节组、4-DOF 躯干、x2 7-DOF 手臂和 x2 1-DOF 平行夹爪）

**数据结构**

- **任务描述**: 从 `behavior-1k` 任务中选择
- **图像**: 多相机 RGB 张量
  - 头部图像: ``[batch_size, 3, 224, 224]``
  - 手腕图像: ``[batch_size, 2, 3, 224, 224]`` (左右相机)


算法
-----------------------------------------

**核心算法组件**

1. **PPO (近端策略优化)**

   - 使用 GAE (广义优势估计) 进行优势估计

   - 带比例限制的策略裁剪

   - 价值函数裁剪

   - 熵正则化

2. **GRPO (组相对策略优化)**

   - 对于每个状态/提示，策略生成 *G* 个独立动作

   - 通过减去组平均奖励来计算每个动作的优势

3. **视觉-语言-动作模型**

   - 具有多模态融合的 OpenVLA 架构

   - 动作标记化和去标记化

   - 用于批评函数的价值头

运行脚本
-------------------
**安装步骤**

1. **克隆所需仓库**

   .. code-block:: bash
      git clone https://github.com/StanfordVL/BEHAVIOR-1K.git third_party/BEHAVIOR-1K

2. **下载资源**

   .. code-block:: bash
      cd third_party/BEHAVIOR-1K
      ./setup.sh --omnigibson --bddl --joylo --dataset

3. **设置环境变量和资源路径**

   .. code-block:: bash
      export OMNIGIBSON_DATASET_PATH=/path/to/third_party/BEHAVIOR-1K/datasets/behavior-1k-assets/
      export OMNIGIBSON_KEY_PATH=/path/to/third_party/BEHAVIOR-1K/datasets/omnigibson.key
      export OMNIGIBSON_ASSET_PATH=/path/to/third_party/BEHAVIOR-1K/datasets/omnigibson-robot-assets/
      export OMNIGIBSON_DATA_PATH=/path/to/third_party/BEHAVIOR-1K/datasets/
      export OMNIGIBSON_HEADLESS=1

4. **评估环境**
   .. code-block:: bash

      bash examples/embodiment/eval_embodiment.sh behavior_eval

5. **训练环境**
   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh behavior_ppo_openvlaoft


可视化和结果
-------------------------

**1. TensorBoard 日志记录**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 关键指标跟踪**

- **训练指标**:

  - ``actor/loss``: PPO 策略损失
  - ``actor/value_loss``: 价值函数损失
  - ``actor/entropy``: 策略熵
  - ``actor/grad_norm``: 梯度范数
  - ``actor/lr``: 学习率

- **Rollout 指标**:

  - ``rollout/reward_mean``: 平均回合奖励
  - ``rollout/reward_std``: 奖励标准差
  - ``rollout/episode_length``: 平均回合长度
  - ``rollout/success_rate``: 任务完成率

- **环境指标**:

  - ``env/success_rate``: 跨环境成功率
  - ``env/step_reward``: 逐步奖励
  - ``env/termination_rate``: 回合终止率

**3. 视频生成**

Behavior 环境支持多相机视图的综合视频录制：

.. code-block:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ./logs/video/train

**视频功能**:
- **多相机布局**: 在单个视频中组合头部相机 (448×448) 和手腕相机 (224×224 每个)
- **布局**: 头部相机在右侧，手腕相机堆叠在左侧
- **分辨率**: 最终视频分辨率为 448×672 像素
- **格式**: MP4 格式，RGB 编码
- **信息覆盖**: 任务描述和成功指标覆盖在视频帧上

**4. WandB 集成**

.. code-block:: yaml

   trainer:
     logger:
       wandb:
         enable: True
         project_name: "RLinf"
         experiment_name: "openvlaoft-Behavior"

**5. 环境指标跟踪**

Behavior 环境提供全面的指标跟踪：

- **成功指标**: 每回合成功率和累积成功跟踪
- **回合信息**: 回合长度、回报和奖励统计
- **多环境支持**: 跨多个并行环境的指标跟踪
- **实时监控**: 成功率、失败率和性能指标
- **视频集成**: 指标覆盖在生成的视频上用于视觉分析


**技术实现细节**

Behavior 环境实现包含几个关键技术特性：

- **多相机处理**: 从多个相机传感器自动提取和处理图像
- **任务描述加载**: 从 JSONL 文件动态加载任务描述，支持任务名称映射
- **动作处理**: 支持单步和分块动作执行
- **指标收集**: 全面跟踪成功率、回合长度和性能指标
- **视频录制**: 实时视频生成，具有多相机布局和指标覆盖
- **环境管理**: 支持并行环境，具有单独的指标跟踪

对于 Behavior 实验，我们受到了 
`https://github.com/StanfordVL/b1k-baselines.git` 的启发， 
仅进行了少量修改。我们感谢作者发布开源代码。
