# Real Subagent Pilot v1

## 做了什么

这次验证真实模型是否能接入已经冻结的数据管线，不是全量数据生成。

- `Qwen3.5-4B` 作为 generator；
- `Qwen3.5-9B` 作为独立 critic；
- 两个模型分别运行在 GPU 0 / GPU 1；
- generator 和 critic 通过不同端口、不同请求 ID 调用；
- 每个数据族生成 1 条候选，共 5 个 shard；
- 每条目标约 900 个 Qwen3.5 chat-template tokens；
- 模型只生成答案和 critic 意见，`task_spec`、verifier、token count、
  hash、去重和最终 acceptance 都由本地程序控制。

使用的代码版本：

- `nano-data-pipeline`: `ac70d18`
- campaign manifest: `skill-sft-10k-10m-v1`
- generator model: `Qwen3.5-4B`
- critic model: `Qwen3.5-9B`
- vLLM: `0.19.1`
- max model length: `2048`
- dtype: `float16`
- temperature: `0`

## 结果

| 数据族 | 候选 | Accepted | Accepted tokens | 结果 |
| --- | ---: | ---: | ---: | --- |
| tool-use-and-recovery | 1 | 1 | 963 | 通过 |
| planning-and-state | 1 | 1 | 953 | 通过 |
| verified-reasoning | 1 | 0 | 0 | 正确拒绝 |
| coding-and-validation | 1 | 1 | 1,006 | 通过 |
| skill-routing-and-reflection | 1 | 1 | 949 | 通过 |
| **合计** | **5** | **4** | **3,871** | **4/5 accepted** |

5 个 shard 都完成，没有 transport error。第二次运行直接跳过 5/5 已完成
shard，没有重复调用模型。

`verified-reasoning` 的 4B 输出算错。9B critic 拒绝该样本，本地
`safe_execution_receipt_v1` 也独立复算为失败，所以这不是 critic 假阴性。该样本
没有进入 accepted ledger。

## 本地重审

合并后重新执行完整 audit：

- critic independent receipt：通过；
- family verifier revalidation：通过；
- source policy：通过；
- skill identity：通过；
- exact / semantic dedup：通过；
- Qwen3.5 tokenizer identity：通过；
- token recomputation：通过；
- train sample target：未通过；
- train token target：未通过；
- `training_unblocked=false`。

因此当前不能启动 SFT。真实 pilot 只证明 adapter、角色分离和本地验收闭环可用。

## Artifact 身份

Raw 模型输出仅保留在本地临时目录，不提交 GitHub。公开报告只记录聚合指标：

- plan SHA256:
  `6d0a179f06d35ceaa2195aefdf75536dc9d76eef32f50b47059bfe741ead409b`
- audit SHA256:
  `66e39085a3a9b298bd8e7f6f821a37b6d4e5d5ea520883ff5e04da31609e6ca9`
- merge SHA256:
  `5a6196211d39d5d48e7f20d078b3589d0c0cd24248195232b3f849be2e3ddbcb`

## 证明了什么

已证明：

1. 4B generator 和 9B critic 可以通过 OpenAI-compatible API 分角色运行；
2. 5 个数据族都能进入真实模型调用路径；
3. critic 与本地 verifier 可以一致拦截错误样本；
4. 每条 accepted row 的 token、hash、source、skill 和 verifier receipt 可以重算；
5. 任务可续跑，已完成 shard 不重复生成。

尚未证明：

1. 10,000 train 样本或 10M train tokens 已生成；
2. 当前数据足以提升 4B；
3. SFT、RL 或 benchmark 有任何提升；
4. 4B 已超过 9B 或 27B；
5. production 量级的时间、吞吐、拒绝率和成本已稳定。

## 下一步

先扩成一个 pre-production batch，并根据真实 accept rate 和 tokens/sample 重新估算
shard 数；只有 family quota、10,000 train 样本、10M train tokens、global dedup
和 tokenizer identity 全部通过后，才允许创建 SFT 数据版本。
