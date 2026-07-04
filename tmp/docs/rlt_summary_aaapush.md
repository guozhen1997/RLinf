# aaapush/RLinf 中的 RLT 汇总

这份说明汇总 `aaapush/RLinf` 中与 RLT 相关的几份内部 markdown 设计稿。它们的信息量比公开文档大得多，适合作为内部设计档案，但在并入正式用户文档前需要做筛选。

## 已检查的源文件

- `aaapush/RLinf/examples/embodiment/rlt_maniskill_joint.md`
- `aaapush/RLinf/examples/embodiment/rlt_stage2_algorithm_alignment.md`
- `aaapush/RLinf/examples/embodiment/rlt_realworld_franka_adaptation.md`
- `aaapush/RLinf/examples/embodiment/rlt_meeting_talk.md`

## 这些内部稿件的共同主线

这些内部稿件对 RLT 的描述高度一致，都把它定义成：

```text
OpenPI SFT base
-> Stage1 RL token
-> Stage2 frozen VLA + frozen RL token + lightweight actor-critic
```

并且它们反复强调以下几个设计点：

- Stage2 当前不应该更新大 VLA。
- Stage2 是基于 chunk action 和 chunk TD target 的 actor-critic。
- intervention 属于 replay 语义的一部分，不是单独的 DAgger 监督目标。
- 最终应优先用固定 reset ids 的 `eval/success_once` 来解释效果，而不是只看 train success。

## ManiSkill joint-control 设计稿总结

### 核心状态 / 动作语义

内部设计稿给出的当前任务语义是：

- task：`PegInsertionSideWideClearance-v1`
- prompt：`insert the peg in the hole`
- observation：第三视角 RGB、wrist RGB、Panda qpos 子集
- action：10-step chunk，每步 8 维 joint action
- control mode：`pd_joint_delta_pos`
- reward：sparse success

### Stage2 数据流

内部稿里对 rollout 路径的描述最完整，大致是：

```text
env obs
-> frozen VLA embeddings + reference chunk
-> frozen RL-token encoding
-> actor action
-> optional expert takeover
-> env chunk step
-> cached forward inputs
-> replay conversion in actor worker
```

这一部分非常适合用来解释：

- `x` 里到底有什么
- `a_tilde` 的具体语义
- 为什么 rollout 要缓存特征，而不是 learner 端重复编码
- intervention 发生后 replay reference 为什么也要一起替换

### Warmup 与 online gate

内部的 ManiSkill 设计稿已经抓住了一个最关键的调度事实：

- gate 取决于 learner 已完成的 update 数
- 不是看 runner global step

这一点很稳定，值得保留到正式文档中。

## Stage2 论文对齐稿总结

`rlt_stage2_algorithm_alignment.md` 最适合用来解释“当前实现和预期 RLT 语义的关系”。

### 适合流入正式文档的稳定部分

- frozen VLA 和 frozen RL token 是当前设计本意
- actor 输入是 `[x, a_tilde]`
- critic 是 chunk-level twin-Q TD3
- actor loss 同时包含 Q 提升和 BC regularizer
- reference action dropout 的目的，是防止 actor 只复制 `a_tilde`
- intervention 会替换 replay 侧 reference action

### 更适合继续留在 tmp 的部分

- 很长的论文符号逐项映射
- 对旧 residual actor 的历史说明
- 已经过时的超参快照
- 某些对齐差异的草稿表述，它们可能已经不适用于当前主线

## Franka 真机适配稿总结

`rlt_realworld_franka_adaptation.md` 不是 ManiSkill 公开使用文档，而是一份未来 realworld RLT 集成的设计备忘。

它的核心主张是：

- 不要把 ManiSkill joint schema 强行套到 Franka 真机上
- 应单独定义 Franka RLT schema
- action / state 语义必须显式、规范化
- 要保留 raw trace，方便未来表示方式调整
- replay 中要支持 `source`、`source_chunk`、intervention 等字段

这部分内容很重要，但更适合未来单独写一页 realworld RLT 文档，或者继续保存在内部设计稿里，不适合直接塞进当前 ManiSkill 示例页。

## 会议汇报稿总结

`rlt_meeting_talk.md` 更像是一份阶段性诊断叙事，价值主要在于：

- 说明 RLT 想解决 base VLA 的什么问题
- 梳理 RLinf 里已经接通了哪些模块
- 解释为什么 train success 可能高、而 eval 不涨
- 记录当时怀疑的实现差异或调参问题

它尤其适合抽取下面这些内容：

- 指标解释方式
- train success 与 eval success 的区别
- 为什么当时重点盯 replay 对齐、warmup 行为和 intervention 吸收

这份稿子不适合直接复制到正式文档，但其中的一些诊断 heuristics 很适合改写成简短的 troubleshooting section。

## 最适合进入正式 RST 的部分

从这些内部 markdown 里，最值得带入正式 EN / ZH 示例页的内容是：

1. Stage2 当前的状态 / 动作语义
2. actor-only learner 设计和 rollout 侧特征缓存
3. 基于 update-step 的 online gate
4. 默认 boundary replay 与可选 stride replay 的关系
5. intervention 替换 replay reference 的语义
6. train / eval 指标解释
7. checkpoint 与 norm-stats 对齐检查清单

## 更适合继续留在内部稿的内容

以下内容更适合继续留在 tmp 或未来工程设计稿中：

- 过于细的符号表
- 强实验阶段色彩的时间线与背景
- 尚未在主线实现的 realworld 适配规划
- 会议讲稿式叙述
- 过期超参和旧代码结构描述

## tmp 与正式文档的建议分工

### tmp

`tmp` 更适合保留丰富的内部材料，例如：

- 算法解释
- 真机迁移规划
- 更细的 replay / intervention 语义
- 过去实现与当前实现之间的迁移说明

### 正式 EN / ZH RST

正式文档则更适合只保留用户真正需要的内容：

- 当前支持的两阶段工作流
- 要用哪些配置和脚本
- 当前默认值到底意味着什么
- 应该如何解释评测结果与常见失败模式

