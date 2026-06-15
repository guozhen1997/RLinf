快速体验
========

本教程以 **LIBERO Spatial + OpenPI π₀.₅** 为例，在约 5 分钟内完成第一次具身评测。示例配置为 ``evaluations/libero/libero_spatial_openpi_pi05_eval.yaml``。

Step 1：安装环境
----------------

在仓库根目录执行：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env libero
   source .venv/bin/activate

Step 2：设置环境变量
--------------------

.. code-block:: bash

   export LIBERO_PATH=/path/to/LIBERO

Step 3：准备模型
----------------

将预训练或 SFT 后的模型下载到本地，记下目录路径。也可在启动时通过命令行覆盖 ``rollout.model.model_path``。

Step 4：启动评测
----------------

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model

若配置名以 ``libero_`` 开头，可省略 benchmark 参数：

.. code-block:: bash

   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model

Step 5：查看结果
----------------

- 终端会输出 ``eval/success_once``、``eval/return`` 等指标
- 日志目录：``logs/<时间戳>-libero_spatial_openpi_pi05_eval/eval_embodiment.log``
- 若配置中 ``env.eval.video_cfg.save_video: True``，视频保存在 ``<log_path>/video/eval/``

更多细节见 :doc:`../reference/results`。

下一步
------

- 了解 YAML 配置：:doc:`../reference/configuration`
- 探索其他 benchmark：:doc:`../guides/libero`
- 查看更多 CLI 选项：:doc:`../reference/cli`
