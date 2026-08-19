# Skill SFT 10k / 10M Campaign v1

## 这次先定什么

这轮先冻结数据生成口径，不先生成再挑一个好看的数字。

- 最终至少保留 **10,000 条 train 样本**。
- 这些 train 样本用本地 `Qwen3.5-4B` tokenizer 计算后，合计至少
  **10,000,000 tokens**。
- 另保留至少 **400 条 synthetic dev 样本**，只用于筛选 skill
  候选和检查回归，不计入 10,000 条或 10M tokens。
- 两个 train 下限必须同时满足。10,000 条和 10M tokens 不是二选一。
- 这是第一轮工程量级，不是经过实验验证的“4B 最优训练规模”。生成完只证明
  数据管线完成，模型是否变好必须看后续 matched ablation。

权威配置是
`manifests/skill_sft_campaign_v1.json`。测试会拒绝低于目标、tokenizer
身份漂移、漏禁 benchmark、去重放宽或提前解锁训练的修改。

## 生成哪些数据

| 数据族 | Train 样本 | Train tokens | Synthetic dev | 主要训练内容 |
| --- | ---: | ---: | ---: | --- |
| tool-use-and-recovery | 2,500 | 2.8M | 100 | 工具选择、参数、观察和错误恢复 |
| planning-and-state | 2,000 | 2.0M | 80 | 约束、证据、待办和停止条件 |
| verified-reasoning | 2,500 | 2.3M | 100 | 可由本地程序精确复算的推理 |
| coding-and-validation | 1,500 | 1.8M | 60 | 小型合成仓库中的修改和测试闭环 |
| skill-routing-and-reflection | 1,500 | 1.1M | 60 | 选择 skill、执行 contract、从失败提出改版 |
| **合计** | **10,000** | **10.0M** | **400** | |

family token 配额同样是下限。某一族样本数够了但 tokens 不够，仍要补该族数据。

## 怎么并行生成

初始计划生成 32 个 shard，每个 shard 最多 512 条候选，最多同时运行 8 个
generator subagent。初始候选量按最终目标的 1.3 倍准备：

- 候选样本容量 16,384 条；
- 候选 token 预算至少 13M；
- 给 critic 拒绝、verifier 失败、精确去重和语义近重复留出空间。

每个 generator 只拿到数据族说明、合成 seed、冻结的 skill 版本和输出 schema。
另一个 critic subagent 独立审查，最后由本地确定性脚本验算。generator 看不到
同一条数据的 critic 判断，critic 也看不到 generator 的隐藏推理。

初始 shard 合并后运行一次全局检查。如果某个 family 或全局样本/token 数不足，
调度器按缺口生成 4 个 refill shard，再检查一次，直到同时达到所有下限。过程按
`campaign_id/family_id/shard_id/attempt` 续跑，已经验收的 shard 不重算。

## 哪些内容绝不能进训练

以下 benchmark 的 prompt、reference、模型输出、canary row 和 holdout row
都不能作为 SFT、DPO、RL、reward 或 verifier 训练数据：

- SkillBench
- SWE-bench
- ClawBench
- WildClawBench
- Terminal-Bench
- GSM8K
- MMLU
- GPQA

当前独立 holdout 继续保持未读。skill 自进化只能使用合成 dev 的失败聚类、
critic 拒绝聚类和 verifier 失败聚类，不能根据 benchmark row 修改 skill。

## 怎样才算一条数据合格

一条数据只有同时满足下面条件才进入 accepted ledger：

1. 字段齐全，来源和 generator/critic/verifier receipt 可追踪；
2. 独立 critic 分数至少 0.8；
3. 对应数据族的本地确定性 verifier 通过；
4. 与所有已接受样本没有 exact duplicate；
5. 语义相似度不超过冻结阈值 0.92；
6. train 和 synthetic dev 没有样本、exact hash 或 semantic hash 交叉；
7. 用冻结的 Qwen3.5 tokenizer 重新计算 token 数，不能相信 subagent 自报。

全量训练只有在 family 配额、全局去重、来源策略、tokenizer 身份、10,000 train
样本和 10M train tokens 六项都通过后才解锁。

## Skill 自进化边界

最多做 3 个合成开发周期。每轮根据失败聚类提出 skill instruction、routing
或 verifier 选择的候选修改。候选只有在冻结 synthetic dev 上有提升，并且所有
safety family 都不退步时才替换父版本；否则保留父版本。

选定 skill 后先冻结，再生成 train 数据。train rows 不能反过来继续改 skill，
避免一边生成一边改规则导致数据版本无法复现。
