# RLT 在 RLinf-dev 中的完整设计与实现说明

这份文档面向当前 `RLinf-dev` 主线中的 RLT joint 路线，目标是把论文方法、RLinf 中的实现对齐方式、工程性优化、实际运行步骤、测试方式和排查思路统一说明清楚。这里写的是一份内部设计与交付说明，重点是当前仓库真实存在、真实可运行、真实会影响结果解释的内容。

本文覆盖的主线是：

1. 基于 OpenPI `pi0.5` 的 ManiSkill joint-control SFT 基座训练。
2. Stage 1 RL token 训练。
3. Stage 2 基于 frozen VLA 和 frozen RL token 的在线 TD3 训练。
4. 与这条主线相关的数据契约、配置、评测入口、日志指标和常见失败模式。

本文不再把已经移出主线的 debug/helper 脚本当成正式工作流的一部分，旧 ee 路线也不作为当前推荐路径。

## 1. 这条 RLT 路线在 RLinf 里到底是什么

RLT 在 RLinf 里更适合被理解成一套建立在已有 VLA 之上的在线 RL 训练范式。它没有被实现成新的基础 VLA 模型族，也没有作为平行于 `openpi`、`openvla` 的另一套大模型体系接入。

当前主线里这个 VLA 是 OpenPI `pi0.5`。RLT 所做的事情是：

1. 先把 OpenPI 适配到当前 ManiSkill 任务数据分布，得到一个能工作的 SFT 基座。
2. 在冻结或近冻结的 VLA 表征上训练一个 RL token encoder-decoder，把高维 VLA prefix embedding 压缩成可供在线 RL 使用的状态表示。
3. 再在冻结的 VLA 和冻结的 RL token 之上，只训练很小的 actor 和 critic，让在线 RL 去修正最关键、最精细、最容易失败的任务阶段。

这和论文主线是一致的。论文强调两个核心点：

1. VLA 保留广义感知和动作先验。
2. 在线 RL 只训练轻量 actor-critic，通过 RL token 和 reference action 提高样本效率。

当前 `RLinf-dev` 的 joint 路线，就是把这个方法落到了 OpenPI + ManiSkill + RLinf runner/worker 架构上。

## 2. 为什么当前主线改成了 joint 路线

旧路线主要是 ee 动作语义，当前主线已经转成了 joint-control 路线。核心考虑是让数据、SFT、eval、online RL 之间的动作语义真正闭合。

当前主线使用的契约是：

```text
task    = insert the peg in the hole
image   = 3rd_view_camera
wrist   = wide_hand_camera
state   = Panda qpos[:9]
action  = 8D pd_joint_delta_pos
chunk   = 10-step action chunk
env     = PegInsertionSideWideClearance-v1
```

这条路线的关键收益是：训练数据中的动作和评测时环境控制器解释的动作处于同一语义空间，回放与评测更容易严格对齐。对 peg insertion 这种接触精度很高的任务，这一点非常重要。

当前正式公开和内部继续维护的主线都应理解为 `rlt_maniskill_joint` 路线，旧 ee 路线不应再作为默认基线去解释当前结果。

## 3. 论文方法的核心结构

结合论文正文，RLT 的核心设计可以压缩成三件事。

### 3.1 RL token

论文先从 VLA 的内部 token 表征中构造一个紧凑状态表示。做法是：

1. 取 VLA 的最终层 token embeddings。
2. 在序列尾部追加一个可学习的特殊 token。
3. 通过轻量 encoder 得到这个特殊 token 的输出向量，作为 `z_rl`。
4. 再用 decoder 以 `z_rl` 为瓶颈，去自回归重建原始 prefix embeddings。

这个 reconstruction objective 让 `z_rl` 保留任务相关信息，同时足够紧凑，便于后续在线 RL 用一个很小的 actor-critic 来学习。

### 3.2 参考动作条件化 actor

论文里的在线 RL 不会让 actor 直接从零在高维动作空间里盲搜，而是让 actor 读取：

```text
x = [z_rl, proprio]
reference action chunk = a_tilde
```

然后输出 refined action chunk。这里的 `a_tilde` 来自冻结 VLA 的 reference action。

这样做的本质，是把 online RL 收敛成对已有动作提案的 local refinement。

### 3.3 Chunk-level off-policy RL

论文按 action chunk 做 TD 学习。这样能明显缩短高频控制任务里的有效 credit assignment horizon，也更适合 VLA 原本就用 chunk 输出动作的设定。

同时，论文使用 replay buffer、off-policy actor-critic、warmup 和 actor regularization，让整个系统在很少的在线数据下也能开始稳定优化。

## 4. 论文结构在 RLinf-dev 中的对应关系

当前 `RLinf-dev` 的 joint 实现基本保留了论文骨架，但做了面向 OpenPI 和 RLinf 基础设施的适配。

### 4.1 基础模型

论文描述的是 Physical Intelligence 自家的 VLA 体系。当前 RLinf 落地时，基础模型换成了 OpenPI `pi0.5`，但方法结构没有变：

1. 先训练 task-adapted SFT base。
2. 再训练 RL token。
3. 最后冻结大模型，只训练轻量在线 RL 头。

### 4.2 状态与动作接口

当前 joint 路线的 RL 状态定义是：

```text
x = [z_rl, proprio]
```

其中：

1. `z_rl` 来自冻结 RL token encoder。
2. `proprio` 目前是 `Panda qpos[:9]`。

动作接口是 `10 x 8` 的 joint delta-pos chunk，和当前 OpenPI joint dataconfig、环境控制器、eval wrapper 使用同一套语义。

当前公开 joint 配置还做了一个更直接的工程收敛：VLA reference chunk、Stage 2 actor 输出 chunk、环境默认开环执行 chunk 都使用 `10`。论文里会把 VLA 输出 horizon `H` 和 RL chunk length `C` 分开讨论，当前主线为了先把链路跑稳，公开配置没有强调这层拆分。

### 4.3 在线 RL 算法

当前实现用的是 chunk-level TD3 风格更新。对应配置中：

1. `algorithm.loss_type: rlt_td3`
2. `algorithm.adv_type: rlt_td3`

这条路径由专门的 actor worker 承担，不走 RLinf 默认 PPO actor 路径。

## 5. 当前主线的文件与模块结构

要看懂这条链路，最重要的文件可以分成五层。

### 5.1 数据与 OpenPI dataconfig

- `rlinf/models/embodiment/openpi/dataconfig/rlt_maniskill_joint_dataconfig.py`
- `rlinf/models/embodiment/openpi/policies/maniskill_policy.py`

这里负责把 joint LeRobot 数据映射成 OpenPI 能读的 schema，并且把 `norm_stats` 以显式路径方式接入。

### 5.2 SFT 基座

- `examples/sft/config/rlt_maniskill_joint_pi05_sft.yaml`
- `examples/sft/run_vla_sft.sh`

这一步训练一个适配当前 joint 数据分布的 OpenPI 基座。

### 5.3 Stage 1

- `examples/sft/train_rlt_stage1.py`
- `examples/sft/train_rlt_stage1.sh`
- `examples/sft/config/rlt_stage1_maniskill_joint.yaml`
- `rlinf/models/embodiment/rlt_stage1/rlt_stage1_policy.py`
- `rlinf/workers/sft/fsdp_rlt_stage1_sft_worker.py`

这一步训练 RL token，并导出 `rl_token_model.pt`。

### 5.4 Stage 2

- `examples/embodiment/config/rlt_stage2_maniskill_joint.yaml`
- `examples/embodiment/config/model/rlt_stage2_joint.yaml`
- `examples/embodiment/run_embodiment.sh`
- `rlinf/models/embodiment/rlt_stage2/rlt_stage2_policy.py`
- `rlinf/models/embodiment/rlt_stage2/components.py`
- `rlinf/models/embodiment/rlt_stage2/replay_buffer.py`
- `rlinf/models/embodiment/rlt_stage2/vla_wrapper.py`
- `rlinf/workers/actor/fsdp_rlt_stage2_policy_worker.py`

这一步实现 online TD3 训练、replay、weight sync 和缓存特征训练。

### 5.5 Eval 与环境侧接口

- `examples/embodiment/config/rlt_maniskill_joint_pi05_sft_eval.yaml`
- `examples/embodiment/eval_embodiment.sh`
- `examples/embodiment/config/env/maniskill_rlt_joint.yaml`
- `rlinf/workers/rollout/hf/huggingface_worker.py`
- `rlinf/workers/env/env_worker.py`

这里负责 rollout、eval、online gate、expert intervention、环境信息回传等行为。

## 6. 数据契约是怎么定义的

### 6.1 数据字段

`LeRobotRLTManiSkillJointDataConfig` 会把 joint 数据整理成中间字段：

```text
observation/image
observation/wrist_image
observation/state
actions
prompt 或 task
```

然后 `ManiSkillInputs` 会进一步映射成 OpenPI 输入：

1. `base_0_rgb` 对应主视角。
2. `left_wrist_0_rgb` 对应腕部视角。
3. `right_wrist_0_rgb` 目前补零。
4. `state` 直接来自 joint 数据中的 state。
5. `prompt` 来自数据字段或默认 prompt。

### 6.2 Prompt 约定

当前 joint 主线默认 prompt 是：

```text
insert the peg in the hole
```

如果数据里没有显式 `prompt`，会回退到 `task` 或配置里的默认 prompt。

### 6.3 Norm stats

这条链路强依赖显式 `norm_stats_path`。当前做法是：

1. 数据集路径和 `norm_stats.json` 路径明确写进配置。
2. SFT、Stage 1、Stage 2 共用同一份 joint 数据集对应的 `norm_stats.json`。

这样做的目的是避免数据归一化和模型实际输入在不同阶段发生漂移。

## 7. SFT 基座阶段的设计与实现

### 7.1 目标

SFT 是整个系统的地基，目标是得到一个在当前 ManiSkill joint 数据上已经具备基本任务能力的 OpenPI policy。

如果这一步不成立，后面的 Stage 1 和 Stage 2 都没有可靠 reference。

### 7.2 当前实现

配置入口是 `examples/sft/config/rlt_maniskill_joint_pi05_sft.yaml`。这一步复用了 RLinf 里现成的 OpenPI SFT 路径，只在数据配置和 joint 动作语义上做了收敛。

当前配置的重要点：

1. `actor.model.openpi.config_name = pi05_rlt_maniskill_joint`
2. `actor.model.num_action_chunks = 10`
3. `actor.model.action_dim = 8`
4. `actor.openpi_data.repo_id` 指向 joint LeRobot 数据
5. `actor.openpi_data.norm_stats_path` 指向 joint 数据的 `norm_stats.json`

### 7.3 为什么先做 SFT

论文里虽然重点在 RL token 和 online RL，但它默认前提就是已有一个能给出合理 reference action 的 VLA。当前实现里这一步显式拆成 SFT 阶段，是为了把“基础 policy 是否成立”和“后续在线 RL 是否成立”分开验证。

## 8. Stage 1 的设计与实现

### 8.1 Stage 1 学什么

Stage 1 训练的是 RL token encoder-decoder，重点在表征压缩与重建。

`RLTStage1Policy` 的核心流程是：

1. 用 `Stage1VLAWrapper` 跑冻结或近冻结的 OpenPI。
2. 从 VLA 中抽取 prefix embeddings 和 pad mask。
3. 把这些 embeddings 输入 `RLTokenModel`。
4. 用 reconstruction loss `l_ro` 训练 RL token。
5. 如果 `vla_finetune_alpha > 0`，可以再混入一部分 VLA action loss。

### 8.2 当前实现细节

`rlinf/models/embodiment/rlt_stage1/rlt_stage1_policy.py` 里，Stage 1 有两种模式：

1. 纯 RL token 训练，VLA 完全冻结。
2. RL token 训练加少量 VLA 联合微调。

当前 joint 主线默认是前者，也就是 `vla_finetune_alpha: 0.0`。

### 8.3 导出物

`FSDPRLTStage1SftWorker.save_checkpoint()` 会在普通 checkpoint 之外，额外导出：

```text
actor/rl_token/rl_token_model.pt
```

Stage 2 不需要理解整个 Stage 1 checkpoint，只需要这个精简的 RL token 权重文件。

这就是当前工程实现里一个很重要的优化点：Stage 2 加载依赖被压到最小。

## 9. Stage 2 的设计与实现

### 9.1 整体结构

`RLTStage2Policy` 保留了论文的基本分工：

1. frozen OpenPI VLA
2. frozen RL token encoder
3. trainable direct Gaussian actor
4. trainable twin-Q critic

和早期一些内部表述相比，当前主线使用的是 direct Gaussian actor。它仍然读取 reference action，但会直接输出最终 action chunk。

### 9.2 特征抽取

`_prepare_features()` 的流程是：

1. `Stage2VLAWrapper.prepare_obs()` 把 env obs 转成 OpenPI observation。
2. `extract_embeddings()` 抽 prefix embeddings。
3. `RLTokenModel.encode()` 得到 `z_rl`。
4. `get_rl_chunk_reference()` 得到 reference chunk `a_tilde`。
5. `extract_proprio()` 取前 `proprio_dim` 维 state。
6. 拼出 `x = [z_rl, proprio]`。

这一步是 Stage 2 所有 actor/critic 更新的状态入口。

### 9.3 Actor 设计

当前 `DirectGaussianActor` 在 `components.py` 中实现。输入是：

```text
[x, a_tilde]
```

输出是 action chunk 的均值，再按固定 `sigma` 采样，最后 clamp 到 `[-1, 1]`。

实现里还有两个关键点：

1. `ref_action_dropout`
2. `actor_noise_sigma`

前者在训练时随机屏蔽 reference action，避免 actor 只学复制 VLA。后者给训练期探索加轻量高斯噪声。

### 9.4 Critic 设计

当前 critic 是 twin-Q。输入是 `(x, a)`，输出两个 Q 值。

TD target 计算方式是 chunk-level：

1. 先对 chunk 内 reward 做折扣求和。
2. 再拼上 `(gamma ** chunk_length) * next_q` bootstrap。
3. target actor 用当前 actor 的 target copy，并带 target noise。

这和论文的 chunk-level off-policy RL 思路是一致的。

### 9.5 Actor loss

当前 actor loss 由三部分组成：

1. BC loss
2. `-Q` 项
3. chunk delta regularization

其中 BC 部分还会根据 `source_chunk` 区分 RL reference 和 intervention/human source。发生 intervention 时，BC target 可以切到 human/expert action。

这点对解释训练曲线很关键，因为当前 actor 同时受到 reference、Q 值和 chunk 内平滑约束的共同作用。

## 10. Replay buffer 与缓存特征训练

### 10.1 为什么 replay 不直接存原始图像

如果每次 actor/critic 更新都重新跑一次 VLA 视觉主干和 RL token encoder，训练会非常慢，而且 learner 端显存和吞吐都会很差。

所以当前 Stage 2 replay 存的是已经抽好的中间特征：

1. `x`
2. `a`
3. `a_tilde`
4. `rewards`
5. `next_x`
6. `next_a_tilde`
7. source/intervention/success 等附加字段

这就是 `RLTStage2ReplayBuffer` 的核心设计。

### 10.2 当前 replay 默认模式

当前公开默认配置：

```text
actor.model.rlt_stage2.replay_subsample_stride: 0
```

这表示默认使用 boundary-only replay，也就是只在 chunk boundary 处构 transition。

代码里确实还保留了 stride replay 的更细粒度路径，但当前主线默认值没有走它。

### 10.3 这带来的工程收益

这套缓存特征方案的收益非常直接：

1. learner 侧不需要加载大 VLA 主干去做在线更新。
2. actor-only 训练路径成为可能。
3. replay 样本更轻，更新步数可以明显提高。

这也是当前 RLinf 集成和论文朴素实现之间一个很重要的工程优化差异。

## 11. RLinf 中最关键的工程优化

### 11.1 actor-only learner path

当前配置里：

```text
algorithm.actor_only_train_model: True
```

这表示 learner 端只训练轻量 actor/critic，不把完整 VLA/RL token 主干留在更新路径中。rollout 侧负责在线抽特征并把中间结果缓存下来，actor worker 只拿缓存特征训练。

这大幅降低了 learner 负担，也让 RLinf 原有 actor-rollout-env 三段式框架能更自然地承载 Stage 2。

### 11.2 weight sync 只同步 actor

`RLTStage2Policy.ROLLOUT_SYNC_PREFIXES = ("actor.",)`。

这意味着 rollout 侧只需要同步轻量 actor 的参数。

这样做的前提是 rollout 本地自己持有 frozen VLA 和 frozen RL token，因此只要 actor 权重同步正确，rollout 就能构成当前 student policy。

### 11.3 version gate 绑定 learner update

这是当前实现里最重要的一个稳定性设计。

在 `fsdp_rlt_stage2_policy_worker.py` 中：

```text
await self.weight_syncer.sync(state_dict, send_func, version=self.update_step)
```

这里同步出去的版本号是 `self.update_step`，也就是已经完成的 learner update 数。

这样 rollout 侧的 online gate 实际比较的是：

1. 学习器到底已经更新了多少次。
2. 是否达到 `warmup_post_collect_updates`。

这样 rollout 看到的是 learner 真实完成的更新进度。

这个设计能防止一种非常隐蔽的问题：runner loop 在推进，但 learner 其实还没做够更新，结果 rollout 过早用 student actor 接管。

### 11.4 warmup gate 的真实含义

当前配置里：

```text
algorithm.warmup_post_collect_updates: 30000
```

它表达的是：learner 已经累计完成多少次更新之后，student policy 才允许真正接管 rollout。

这点一定要和训练日志一起看，否则很容易误判是 env 跑得不够，实际问题却是 learner 还没达到 gate。

## 12. intervention 是怎么接入的

### 12.1 env 侧策略信息

`env_worker.py` 会基于 peg/hole 几何关系、progress、yz 误差、grasp 状态等信息判断：

1. 是否进入 intervention region。
2. 是否需要触发 expert takeover。
3. 当前 takeover 还剩多少 chunk。

### 12.2 rollout 侧动作替换

在 `huggingface_worker.py` 中，如果：

1. 当前是 RLT Stage 2
2. online gate 已经打开
3. intervention/expert takeover 被触发

那么 rollout 会把 student 的动作替换成 expert action。

### 12.3 replay 侧 reference 同步更新

当前实现除了替换环境执行动作，还会把 intervention 信息一起写入 replay，包括：

1. `intervention_flags`
2. `source_chunk`
3. `source`
4. `action_chunk`
5. `ref_chunk`

这意味着 replay 中的 reference 语义会随着 intervention 发生局部替换，不再固定等于 base VLA。

这对 actor 的 BC target 和训练解释非常关键。

## 13. 当前 joint 路线的重要超参语义

这里列当前最关键的几个值，目的是方便理解训练行为。

### 13.1 SFT

当前 joint SFT 默认：

1. `num_action_chunks: 10`
2. `action_dim: 8`
3. `config_name: pi05_rlt_maniskill_joint`

### 13.2 Stage 1

当前 Stage 1 默认：

1. `embedding_dim: 2048`
2. `encoder_layers: 2`
3. `decoder_layers: 2`
4. `vla_finetune_alpha: 0.0`

### 13.3 Stage 2

当前 Stage 2 默认：

1. `warmup_min_size: 1000`
2. `train_every_transitions: 5`
3. `max_updates_per_train_step: 1600`
4. `warmup_post_collect_updates: 30000`
5. `replay_subsample_stride: 0`
6. `buffer_capacity: 50000`
7. `actor_noise_sigma: 0.002`
8. `ref_action_dropout: 0.5`
9. `proprio_dim: 9`

其中：

1. `replay_subsample_stride: 0` 表示默认 boundary-only replay。
2. `warmup_post_collect_updates` 是 learner update gate。
3. `train_every_transitions: 5` 表示每积累一定新 transition 就会安排一轮训练预算。

## 14. 从头跑通这条链路的标准步骤

这部分只写当前主线推荐步骤，路径全部用占位写法。

### 14.1 准备 joint 数据集

需要准备一份符合 `pi05_rlt_maniskill_joint` 契约的 LeRobot 数据集，至少包含：

1. 主视角图像
2. wrist 图像
3. `state`
4. `actions`
5. `task` 或 `prompt`

建议准备的目录形态：

```text
<joint_dataset_root>/
<joint_dataset_root>/norm_stats.json
```

### 14.2 生成或确认 `norm_stats.json`

如果这份数据是新整理出来的，需要先生成 `norm_stats.json`。命令入口可用仓内工具：

```bash
python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_rlt_maniskill_joint \
  --repo-id <joint_dataset_root>
```

跑完后先确认 `<joint_dataset_root>/norm_stats.json` 存在。

### 14.3 训练 SFT 基座

最基础的启动命令：

```bash
bash examples/sft/run_vla_sft.sh rlt_maniskill_joint_pi05_sft
```

如果需要显式覆盖数据和模型路径，建议直接调用 Python 入口，因为当前 `run_vla_sft.sh` 不透传额外 Hydra override：

```bash
python examples/sft/train_vla_sft.py \
  --config-path examples/sft/config \
  --config-name rlt_maniskill_joint_pi05_sft \
  runner.logger.log_path=<sft_log_dir> \
  data.train_data_paths[0].dataset_path=<joint_dataset_root> \
  actor.openpi_data.repo_id=<joint_dataset_root> \
  actor.openpi_data.norm_stats_path=<joint_dataset_root>/norm_stats.json \
  actor.model.model_path=<pi05_base_ckpt>
```

训练结束后，记录输出的 actor checkpoint：

```text
<sft_actor_ckpt> = logs/<time>/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_xxx/actor
```

### 14.4 先评测 SFT 基座

建议在进入 Stage 1/Stage 2 之前，先确认 SFT 基座本身没有偏掉。

```bash
python examples/embodiment/eval_embodied_agent.py \
  --config-path examples/embodiment/config \
  --config-name rlt_maniskill_joint_pi05_sft_eval \
  runner.logger.log_path=<sft_eval_log_dir> \
  actor.model.model_path=<sft_actor_ckpt> \
  rollout.model.model_path=<sft_actor_ckpt> \
  actor.model.openpi.config_name=pi05_rlt_maniskill_joint \
  actor.model.openpi_data.repo_id=<joint_dataset_root> \
  actor.model.openpi_data.norm_stats_path=<joint_dataset_root>/norm_stats.json
```

如果想看更短的开环执行影响，可以顺手试：

```bash
python examples/embodiment/eval_embodied_agent.py \
  --config-path examples/embodiment/config \
  --config-name rlt_maniskill_joint_pi05_sft_eval \
  runner.logger.log_path=<sft_eval_log_dir_small_chunk> \
  env.eval.action_exec_chunks=1 \
  actor.model.model_path=<sft_actor_ckpt> \
  rollout.model.model_path=<sft_actor_ckpt> \
  actor.model.openpi.config_name=pi05_rlt_maniskill_joint \
  actor.model.openpi_data.repo_id=<joint_dataset_root> \
  actor.model.openpi_data.norm_stats_path=<joint_dataset_root>/norm_stats.json
```

### 14.5 训练 Stage 1 RL token

默认入口：

```bash
bash examples/sft/train_rlt_stage1.sh
```

如果需要显式覆盖依赖，`train_rlt_stage1.sh` 是可以透传额外 Hydra override 的：

```bash
bash examples/sft/train_rlt_stage1.sh rlt_stage1_maniskill_joint \
  data.train_data_paths[0].dataset_path=<joint_dataset_root> \
  actor.openpi_data.repo_id=<joint_dataset_root> \
  actor.openpi_data.norm_stats_path=<joint_dataset_root>/norm_stats.json \
  actor.model.model_path=<sft_actor_ckpt>
```

Stage 1 结束后，最关键的产物是：

```text
<rl_token_ckpt> = logs/<time>/rlt_stage1_maniskill_joint/checkpoints/global_step_xxx/actor/rl_token/rl_token_model.pt
```

### 14.6 训练 Stage 2

默认入口：

```bash
bash examples/embodiment/run_embodiment.sh rlt_stage2_maniskill_joint LIBERO
```

如果需要显式覆盖关键路径，建议直接调用 Python 入口，因为当前 `run_embodiment.sh` 不透传额外 Hydra override：

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config \
  --config-name rlt_stage2_maniskill_joint \
  runner.logger.log_path=<stage2_log_dir> \
  actor.model.model_path=<student_sft_actor_ckpt> \
  rollout.model.model_path=<student_sft_actor_ckpt> \
  rollout.expert_model.model_path=<expert_sft_actor_ckpt> \
  actor.model.rlt_stage2.rl_token_path=<rl_token_ckpt> \
  actor.model.rlt_stage2.norm_stats_path=<joint_dataset_root>/norm_stats.json
```

这里四个路径的语义分别是：

1. `actor.model.model_path`：student 侧冻结 VLA 基座。
2. `rollout.model.model_path`：rollout 侧 student VLA 基座。
3. `rollout.expert_model.model_path`：需要 intervention 时使用的 expert/reference VLA。
4. `actor.model.rlt_stage2.rl_token_path`：Stage 1 导出的 RL token 权重。

### 14.7 评测 Stage 2 checkpoint

训练中或训练后都可以单独拉起 eval：

```bash
python examples/embodiment/eval_embodied_agent.py \
  --config-path examples/embodiment/config \
  --config-name rlt_stage2_maniskill_joint \
  runner.logger.log_path=<stage2_eval_log_dir> \
  actor.model.model_path=<student_sft_actor_ckpt> \
  rollout.model.model_path=<student_sft_actor_ckpt> \
  actor.model.rlt_stage2.rl_token_path=<rl_token_ckpt> \
  actor.model.rlt_stage2.norm_stats_path=<joint_dataset_root>/norm_stats.json \
  runner.ckpt_path=<stage2_ckpt_if_needed>
```

如果要复现实验时的 deterministic eval，确认：

1. `env.eval.policy_mode = eval`
2. `algorithm.sampling_params.temperature_eval = 0.0`

## 15. 每一步应该怎么判断“跑通了”

### 15.1 数据层

数据层过关，不能只看 dataloader 是否报错。至少要确认：

1. 主图和 wrist 图字段都能被 dataconfig 正确读到。
2. `prompt` 或 `task` 能稳定进入模型。
3. `norm_stats.json` 路径准确。
4. `state` 和 `actions` 维度与 joint 契约一致。

### 15.2 SFT 层

SFT 层过关，至少要看到：

1. checkpoint 能被重新加载。
2. eval 时有非平凡行为，不会只表现为随机抖动。
3. 在 joint 任务上能学到接近、抓取、靠近 peg 等基础行为。

### 15.3 Stage 1 层

Stage 1 层过关，至少要看到：

1. `rl_token_model.pt` 成功导出。
2. `l_ro` 稳定下降或稳定在合理量级。
3. Stage 2 加载这个 checkpoint 时没有 shape/key 错误。

### 15.4 Stage 2 层

Stage 2 层必须分成三层看：

1. 数值层：`x`、`a_tilde`、actor 输出都有限。
2. 结构层：weight sync 正常、online gate 正常、replay 正常积累。
3. 行为层：reference policy 或 warmup policy 在环境中至少有任务推进迹象，student 接管后再看 RL 是否带来提升。

如果第三层不成立，先不要急着调 TD3 超参。

## 16. 训练时最值得盯的指标

当前 joint Stage 2 下，建议优先盯这些指标：

1. `train/rlt_stage2/replay_buffer_size`
2. `train/rlt_stage2/min_replay_buffer_size`
3. `train/rlt_stage2/pending_update_budget`
4. `train/rlt_stage2/update_step`
5. `env/expert_intervention_actual_rate`
6. `env/expert_intervention_requested_rate`
7. `eval/success`
8. `env/reward_nonzero_frac`

解释方式：

1. `replay_buffer_size` 看数据是否真的进入 buffer。
2. `pending_update_budget` 看 learner 是否落后于 incoming data。
3. `update_step` 关系到 rollout 是否已经过了 online gate。
4. `intervention` 两个指标看 expert takeover 是否频繁触发。
5. `eval/success` 和 `reward_nonzero_frac` 看行为层是否活着。

## 17. 当前这条实现和论文相比，最重要的工程差异

### 17.1 基础模型不同

论文是其原始 VLA 体系，当前实现是 OpenPI `pi0.5`。这会影响：

1. dataconfig
2. 动作 horizon
3. 图像输入布局
4. checkpoint 加载方式

### 17.2 训练基础设施不同

论文按方法讲述，当前实现必须嵌进 RLinf 的 runner、worker、FSDP、weight syncer、rollout backend 体系。

因此当前实现中会出现很多论文里不会写但工程上非常重要的部分：

1. actor-only training path
2. rollout local feature extraction
3. actor-only weight sync
4. learner-update-based version gate
5. chunk replay 缓存与恢复

### 17.3 reward 来源不同

论文侧主要讨论真实机器人 sparse success label。当前 ManiSkill 路线的 eval/训练依赖的是仿真环境奖励和相关几何信息。虽然 joint 公开配置仍然用 `only_success` 作为主任务信号，但 intervention 逻辑、环境辅助信息和训练调试解释依然会依赖一系列环境内部字段。

## 18. 当前主线已经解决了哪些问题

到当前 `RLinf-dev` 版本，至少以下问题已经被系统性梳理过：

1. RLT 在 RLinf 中的定位已经明确，不再把它当成新基础模型。
2. joint 数据、SFT、Stage 1、Stage 2 已经形成完整链路。
3. Stage 2 已经支持 actor-only learner path。
4. rollout 侧只同步 actor 权重，减少了不必要同步负担。
5. online gate 已经绑定 learner update version，而不是 runner step。
6. intervention 逻辑已经与 replay source/reference 写入对齐。
7. joint 路线的文档、配置和当前实现语义已经基本收敛。

## 19. 当前还需要继续谨慎对待的问题

即使这条链路已经完整，下面这些点依然值得继续严格检查。

### 19.1 数据语义是否真的稳定

尤其是 `state` 和 `norm_stats.json`。这部分一旦有问题，会连锁影响：

1. OpenPI 输入归一化
2. Stage 1 表征质量
3. Stage 2 的 proprio 语义

### 19.2 SFT 基座是否真的是可靠 reference

Stage 2 的 RL 上限受 reference 质量影响很大。如果 SFT 基座在当前 env/controller 下本身没有有效行为，后面在线 RL 很难靠少量样本救回来。

### 19.3 eval 配置要和训练解释分开看

尤其是：

1. `action_exec_chunks`
2. deterministic eval
3. fixed eval seed

这些设置会明显影响你看到的 success rate 和 rollout 风格。

## 20. 一个更实用的排查顺序

如果后面 joint RLT 效果又变差了，建议按下面顺序查，不要一上来只看 reward 曲线。

### 20.1 先查基础路径

1. SFT checkpoint 是否对。
2. Stage 1 `rl_token_model.pt` 是否和这套 SFT 基座配套。
3. `norm_stats.json` 是否来自同一份 joint 数据。
4. `actor.model.model_path`、`rollout.model.model_path`、`rollout.expert_model.model_path` 是否按预期指向。

### 20.2 再查在线 gate

1. `update_step` 是否在增长。
2. `warmup_post_collect_updates` 是否远大于实际 learner update。
3. rollout 是否一直没真正切到 student。

### 20.3 再查 replay

1. buffer 是否真的在积累。
2. `source_chunk` 是否符合预期。
3. intervention 是否把 reference 和 action 同步写进去了。

### 20.4 最后再看超参和随机性

只有在前面都确认没问题之后，再去讨论：

1. `buffer_capacity`
2. `actor_noise_sigma`
3. `ref_action_dropout`
4. seed
5. eval seed 和固定 reset state

## 21. 总结

当前 `RLinf-dev` 的 RLT joint 路线，已经不是一个零散的实验拼接，而是一条比较完整的两阶段方法落地：

1. 先用 OpenPI joint SFT 得到 base VLA。
2. 再训练 RL token，获得压缩表征。
3. 最后在 RLinf embodied runner 中用 actor-only 的 chunk-level TD3 做在线优化。

这条实现保留了论文最核心的思路：

1. RL token 作为在线 RL 的紧凑状态接口。
2. frozen VLA 作为感知和 reference action 先验。
3. 轻量 actor-critic 在少量在线数据上快速优化。
4. chunk-level replay 和 off-policy 更新提高样本效率。

同时，它也做了几项对 RLinf 很关键的工程优化：

1. actor-only learner path
2. rollout 本地特征抽取
3. actor-only weight sync
4. learner-update-based warmup gate
5. intervention 与 replay source 对齐

后续如果要继续补正式文档，这份说明最适合拆成三块：

1. 给正式 `rst` 的运行说明。
2. 给内部技术报告的设计细节。
3. 给排障说明的 checklist。
