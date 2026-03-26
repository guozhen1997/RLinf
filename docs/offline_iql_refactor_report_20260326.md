# Offline IQL 链路改造汇报（2026-03-26）

## 目标

本次改造围绕 `fsdp_iql` 离线训练链路，目标是：

1. 训练阶段参考 SFT pipeline，由 actor worker 直接加载离线数据，去除 dataset worker 依赖。
2. 评估阶段保留 env 交互能力，支持 `runner.only_eval` 仅初始化 eval env。
3. 清理 offline 配置结构，避免 `env` 节点承载与环境无关的 dataset 字段。
4. 补充可复用的 `d4rl` 默认 env 配置，便于其它模型复用。

---

## 主要改动

### 1) 训练数据链路：DatasetWorker -> Actor 本地 DataLoader

- 文件：`rlinf/workers/actor/fsdp_iql_policy_worker.py`
- 关键改动：
  - 新增 `OfflineTransitionDataset`，将离线 transition 数据包装为 torch dataset。
  - 新增 actor 侧 `_build_offline_dataloader()`，从 `dataset.*` 配置加载离线数据。
  - 使用 `torch.utils.data.DistributedSampler` 完成分布式采样。
  - `run_training()` 改为从本地 `data_iter` 连续取 batch，不再依赖 runner 转发。
  - 从离线数据自动推断 `obs_dim/action_dim`，无需入口通过 dataset worker 回填。
  - checkpoint 新增 data state（`data_epoch`, `data_iter_offset`）并支持恢复。

### 2) OfflineRunner 简化为 actor 主驱动

- 文件：`rlinf/runners/offline_runner.py`
- 关键改动：
  - 构造参数移除 `dataset`。
  - 删除 dataset->actor 的 batch 转发流程。
  - 训练仅调用 `actor.run_training()`。
  - `env/rollout` 初始化改为按需（`val_check_interval > 0` 或 `runner.only_eval`）。

### 3) offline 入口脚本去掉 DatasetWorker

- 文件：`examples/embodiment/train_offlinerl.py`
- 关键改动：
  - 移除 `DatasetWorker` 的创建、初始化和维度探测逻辑。
  - actor 组直接启动。
  - `env/rollout` 仅在需要 eval 时创建。

### 4) only_eval 与 env.eval 兼容改造

- 文件：`rlinf/config.py`
  - 增强 `validate_offline_cfg()`：
    - 校验 `algorithm.batch_size` 必填且 > 0。
    - 对 `actor.global_batch_size / micro_batch_size` 执行“校验优先、缺失推导”的默认策略，不覆盖用户显式值。
    - 增加数值合法性约束（>0、`global >= micro`）。
    - 设置 `runner.only_eval` 默认值。
    - 若缺失 `env.train`，从 `env.eval` 生成兼容副本，避免旧路径访问报错。
    - offline 场景仅对 `env.eval` 做并行环境配置校验。

- 文件：`rlinf/workers/env/env_worker.py`
  - `only_eval=true` 时不再初始化 train rank 映射和 train env。
  - 对 `env.train` 的读取加保护，避免仅 eval 配置时报错。

- 文件：`rlinf/workers/rollout/hf/huggingface_worker.py`
  - `only_eval=true` 时不再构建 train 路径相关 batch/rank 映射。

### 5) DatasetWorker 发送逻辑优化（避免同步发送时重复采样）

- 文件：`rlinf/workers/dataset/dataset_worker.py`
- 关键改动：
  - `send_batches_to_actor()` 改为每个 dataset rank 每步先做一次联合采样：
    - 采样量：`num_receivers * batch_size`
    - 再按 receiver 切块发送给各个 actor rank
  - 新增 `_sample_joint_batch()`：
    - 当数据集暴露数组和 `size` 时，优先走 index 采样；
    - 容量足够时使用不放回采样，减少同一轮跨 dp 重叠；
    - 否则回退到原 `dataset.sample()`。

### 6) 配置结构清理 + D4RL env 默认配置

- 新增文件：`examples/embodiment/config/env/d4rl.yaml`
  - 提供可复用的 D4RL 环境默认项（`env_type`、`max_steps_per_rollout_epoch`、`auto_reset`、`video_cfg` 等）。

- 调整 offline IQL 配置（以及 e2e）：
  - `examples/embodiment/config/d4rl_iql_mujoco.yaml`
  - `examples/embodiment/config/d4rl_iql_antmaze.yaml`
  - `examples/embodiment/config/d4rl_iql_kitchen_adroit.yaml`
  - `tests/e2e_tests/embodied/d4rl_iql_mujoco.yaml`
  - 变化：
    - 移除 `component_placement.dataset`
    - 增加 `runner.only_eval: true`
    - `dataset_type/dataset_path/env_name` 迁移到 `dataset` 节点
    - `env` 仅保留 `env.eval`
    - 通过 `defaults: - env/d4rl@env.eval` 复用通用 D4RL env 配置

---

## 对“env 下两个 env_name + dataset 字段”的检查结论

1. **env 下两个 `env_name` 的问题**  
   在当前 `d4rl_iql_*` 配置中，`env` 节点仅保留了 `env.eval.env_name`，不再存在 `env.train.env_name` 与 `env.eval.env_name` 同时出现的重复结构。

2. **`dataset_type/dataset_path` 放在 env 下的问题**  
   这两个字段已经迁移到 `dataset` 节点，不再由 `env` 承载（`env` 仅表达环境交互配置）。

---

## 验证情况

- 已对改动的 Python 文件执行 `compileall`，语法通过。
- IDE lints 检查通过（无新增 lint 报错）。
- 受当前运行环境依赖限制（缺少部分 Python 包），未在此环境完成完整训练/评估端到端启动验证。

---

## 备注

`rlinf/workers/dataset/` 已删除；离线数据采样在 actor 内完成，入口为 `rlinf.data.datasets.d4rl.D4RLDataset.build_offline_actor_batch_provider`。
