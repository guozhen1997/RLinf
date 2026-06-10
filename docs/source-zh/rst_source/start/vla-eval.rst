具身智能评估
===============

简介
---------

RLinf 提供统一的具身智能评估入口，支持在仿真或真机环境中并行 rollout，并输出任务级成功率等指标。

所有评估相关代码与配置位于仓库根目录的 ``evaluations/`` 下：

.. code-block:: text

   evaluations/
   ├── eval_embodied_agent.py   # 评估主程序
   ├── run_eval.sh              # 一键启动脚本
   ├── libero/                  # LIBERO 评测配置
   ├── robotwin/                # RoboTwin 评测配置
   ├── behavior/                # BEHAVIOR-1K 评测配置
   ├── maniskill/               # ManiSkill OOD 评测配置
   ├── realworld/               # 真机评测配置
   └── polaris/                 # PolaRiS 评测配置

评估流程由 ``EmbodiedEvalRunner`` 驱动：Env Worker 与 Rollout Worker 通过 Channel 交互，在 ``env.eval`` 配置下完成并行评测。终端与日志中会输出 ``eval/success_once``、``eval/return`` 等指标。

支持的 Benchmark
---------------------

下表列出 ``evaluations/`` 目录中已提供示例配置、且可通过 ``run_eval.sh`` 直接启动的 benchmark。

.. list-table::
   :header-rows: 1
   :widths: 18 28 34

   * - Benchmark
     - 任务 / 环境配置
     - 示例配置文件
   * - LIBERO
     - ``libero_spatial``、``libero_object``、``libero_goal``、``libero_10``、``libero_90``
     - ``libero/libero_spatial_openpi_pi05_eval.yaml`` 等
   * - RoboTwin
     - ``robotwin_place_empty_cup``、``robotwin_adjust_bottle``、``robotwin_place_shoe``、``robotwin_click_bell``
     - ``robotwin/robotwin_place_empty_cup_openvlaoft_eval.yaml`` 等
   * - BEHAVIOR-1K
     - ``behavior_r1pro``
     - ``behavior/behavior_openpi_pi05_eval.yaml``
   * - ManiSkill OOD
     - ``maniskill_ood_template`` （分布外泛化评测）
     - ``maniskill/maniskill_ood_openvlaoft_eval.yaml``
   * - RealWorld
     - ``realworld_franka_sft_env``、``realworld_bin_relocation``
     - ``realworld/realworld_eval.yaml``、``realworld/realworld_pnp_eval.yaml``、``realworld/realworld_pnp_eval_dreamzero.yaml``
   * - PolaRiS
     - ``polaris_droid_tapeintocontainer``、``polaris_droid_movelattecup`` 等
     - ``polaris/polaris_tapeintocontainer_openpi_pi05_eval.yaml``、``polaris/polaris_movelattecup_openpi_eval.yaml``

**LIBERO 变体：** 标准 LIBERO、LIBERO-PRO、LIBERO-PLUS 均支持，通过环境变量切换（见下文「一键启动」）。

**配置回退：** 若 ``evaluations/<benchmark>/<config>.yaml`` 不存在，``run_eval.sh`` 会自动回退到 ``examples/embodiment/config/`` 下同名配置，便于复用训练配置做评测。

支持的模型
---------------

评估配置通过 ``defaults`` 引用 ``examples/embodiment/config/model/`` 下的模型 preset，并在 ``rollout.model`` 中覆盖 ``model_path`` 等字段。当前 ``evaluations/`` 中已有示例的模型如下：

.. list-table::
   :header-rows: 1
   :widths: 20 18 42

   * - 模型
     - ``model_type``
     - 示例配置
   * - π₀ / π₀.₅（OpenPI）
     - ``openpi``
     - ``libero_spatial_openpi_pi05_eval``、``libero_goal_openpi_eval``、``robotwin_adjust_bottle_openpi_eval`` 等
   * - OpenVLA-OFT
     - ``openvla_oft``
     - ``libero_10_openvlaoft_eval``、``robotwin_place_empty_cup_openvlaoft_eval``、``maniskill_ood_openvlaoft_eval`` 等
   * - StarVLA
     - ``starvla``
     - ``libero_spatial_starvla_eval``
   * - DreamZero
     - ``dreamzero``
     - ``libero_spatial_dreamzero_eval``
   * - LingBotVLA
     - ``lingbotvla``
     - ``robotwin_click_bell_lingbotvla_eval``、``robotwin_place_shoe_lingbotvla_eval``

安装环境
-------------

评估与训练共用同一套具身环境安装流程。在仓库根目录执行：

.. code-block:: bash

   bash requirements/install.sh embodied --model <model> --env <env>
   source .venv/bin/activate

其中 ``<model>`` 与 ``<env>`` 需与目标 benchmark 匹配。常用组合如下：

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Benchmark
     - 推荐 ``--model``
     - 推荐 ``--env``
   * - LIBERO
     - ``openpi`` / ``openvla-oft`` / ``starvla`` / ``dreamzero``
     - ``maniskill_libero`` 或 ``libero``
   * - RoboTwin
     - ``openvla-oft`` / ``openpi`` / ``lingbotvla``
     - ``robotwin``
   * - BEHAVIOR-1K
     - ``openpi``
     - ``behavior``
   * - ManiSkill OOD
     - ``openvla-oft``
     - ``maniskill_libero``
   * - RealWorld
     - ``openpi``
     - ``franka``

**额外环境变量（按 benchmark 设置）：**

- **LIBERO：** ``export LIBERO_PATH=/path/to/LIBERO``
- **RoboTwin：** ``export ROBOTWIN_PATH=/path/to/RoboTwin``，``export ROBOT_PLATFORM=ALOHA``
- **BEHAVIOR-1K：** 设置 ``OMNIGIBSON_DATA_PATH`` 及相关 OmniGibson 路径（见 :doc:`../examples/embodied/behavior`）
- **DreamZero：** ``export DREAMZERO_PATH=/path/to/DreamZero``

配置 YAML
--------------

评估配置为 Hydra YAML，默认放在 ``evaluations/<benchmark>/`` 下。以 ``libero_spatial_openpi_pi05_eval.yaml`` 为例，核心结构如下：

.. code-block:: yaml

   defaults:
     - env/libero_spatial@env.eval      # 环境 preset
     - model/pi0_5@rollout.model        # 模型 preset
     - override hydra/job_logging: stdout

   hydra:
     searchpath:
       - file://${oc.env:EMBODIED_PATH}/config/

   runner:
     task_type: embodied_eval   # 必须为 embodied_eval
     only_eval: True            # 仅评测，不训练
     ckpt_path: null            # 可选：加载 .pt 权重
     logger:
       log_path: "../results"

   cluster:
     component_placement:
       env,rollout: all          # env 与 rollout 的 GPU 分配

   env:
     eval:
       total_num_envs: 500       # 并行环境数
       rollout_epoch: 1          # 评测轮次
       max_episode_steps: 240
       auto_reset: True
       is_eval: True
       video_cfg:
         save_video: True

   rollout:
     generation_backend: "huggingface"
     model:
       model_path: "/path/to/model"   # 必填：模型权重路径
       model_type: "openpi"

**必须修改的字段：**

1. ``rollout.model.model_path``：指向本地模型目录或 HuggingFace 缓存路径。
2. ``env.eval`` 中与资源相关的字段：``total_num_envs``、``max_episode_steps``、``assets_path`` （RoboTwin）等。
3. ``cluster.component_placement``：按可用 GPU 数量调整 ``env`` 与 ``rollout`` 的 placement。
4. **真机评测：** 在 ``cluster.node_groups`` 中配置 Franka IP 与节点拓扑（参考 ``realworld/realworld_eval.yaml``）。

**从训练配置派生：** 可复制 ``examples/embodiment/config/`` 或 ``tests/e2e_tests/embodied/`` 中对应训练 YAML，删除 ``algorithm``、``actor`` 等训练段，保留 ``env.eval`` 与 ``rollout``，并设置 ``runner.task_type: embodied_eval``、``runner.only_eval: True``。

一键启动
--------------

进入仓库根目录并激活虚拟环境后，使用 ``evaluations/run_eval.sh`` 启动评测。

**方式一：显式指定 benchmark**

.. code-block:: bash

   source .venv/bin/activate
   bash evaluations/run_eval.sh <benchmark> <config_name> [hydra_overrides...]

示例：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval
   bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval
   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval

**方式二：自动推断 benchmark**

配置名以 ``libero_``、``robotwin_``、``behavior_`` 等前缀开头时，可省略 benchmark：

.. code-block:: bash

   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval

**方式三：命令行覆盖 Hydra 参数**

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model/RLinf-Pi05-SFT \
     env.eval.total_num_envs=64 \
     runner.ckpt_path=/path/to/checkpoint.pt

**LIBERO-PRO / LIBERO-PLUS：**

.. code-block:: bash

   export LIBERO_TYPE=pro
   export LIBERO_PERTURBATION=all
   bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval

   export LIBERO_TYPE=plus
   export LIBERO_SUFFIX=all
   bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval

**RoboTwin：**

.. code-block:: bash

   export ROBOTWIN_PATH=/path/to/repo/RoboTwin
   export ROBOT_PLATFORM=ALOHA
   bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval

**ManiSkill OOD 批量评测：**

``mani-ood`` 模式会依次在多个 OOD 场景上运行评测，需预先设置环境变量：

.. code-block:: bash

   export EVAL_NAME=my_ood_eval
   export CKPT_PATH=/path/to/checkpoint.pt
   export TOTAL_NUM_ENVS=16
   export EVAL_ROLLOUT_EPOCH=1
   bash evaluations/run_eval.sh mani-ood maniskill_ood_openvlaoft_eval

日志与结果
----------

- 默认日志目录：``logs/<时间戳>-<config_name>/eval_embodiment.log``
- ManiSkill OOD：``logs/eval/<EVAL_NAME>/<时间戳>-<env_id>-<obj_set>/run_ppo.log``
- 终端输出示例：``eval/success_once``、``eval/return`` 等
- 若 ``env.eval.video_cfg.save_video: True``，视频保存在 ``<log_path>/video/eval/``

相关文档
--------

- 各 benchmark 的环境搭建与训练示例：:doc:`../examples/embodied/index`
- 环境安装详情：:doc:`installation`
- 模型专属 standalone 评测脚本（非统一入口）：``toolkits/standalone_eval_scripts/``
