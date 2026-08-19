# Skill SFT 10k / 10M Release v2

## 结果

- Train 样本：**15,888**
- Train tokens：**11,425,166**
- Synthetic dev：**400**
- 最终 accepted rows：**16,288**
- 完成 shard：**55**
- 首轮 / refill shard：**32 / 23**
- Generator recipe 调用：**55**
- Critic recipe 调用：**55**
- `training_unblocked`：**true**

## Family 分布

| Family | Train rows | Train tokens | Dev rows |
| --- | ---: | ---: | ---: |
| coding-and-validation | 2,500 | 2,602,572 | 60 |
| planning-and-state | 2,960 | 2,202,181 | 80 |
| skill-routing-and-reflection | 2,500 | 1,204,842 | 60 |
| tool-use-and-recovery | 3,964 | 3,035,225 | 100 |
| verified-reasoning | 3,964 | 2,380,346 | 100 |

## Gate

- `critic_revalidation_pass`: 通过
- `dev_sample_target_pass`: 通过
- `family_dev_quotas_pass`: 通过
- `family_quotas_pass`: 通过
- `global_dedup_pass`: 通过
- `recipe_call_budget_pass`: 通过
- `skill_identity_pass`: 通过
- `source_policy_pass`: 通过
- `tokenizer_identity_pass`: 通过
- `train_sample_target_pass`: 通过
- `train_token_target_pass`: 通过
- `verifier_revalidation_pass`: 通过

## 去重

- Shard-local accepted：28,160
- Global accepted：16,288
- Global rejected：11,872
- 语义口径：`family + task_spec`；长 padding 只计 token，不决定语义唯一性。

## 失败与修复

首轮 compiler 的数值取模空间太小，导致不同 shard 生成了相同 `task_spec`。Shard-local verifier 无法发现跨 shard 重复；global audit 拒绝了 11,872 行。随后将任务 identity 改为 hash 派生的大空间，从原始 shard 重算 semantic hash，并按真实 family/sample/token/dev 缺口增加 23 个 refill shard。旧失败证据保留，未删除或改写。

## Artifact

- `accepted_jsonl_sha256`: `b5503761900cfa290dba03aff306ff511630d01544a4dab93de3a0be1e74abc1`
- `audit_json_sha256`: `084d39ba07f3bfbd8ed4375812a4cf35bda7d699cd158d97fa43cd3c435836cb`
- `merge_json_sha256`: `00e92e74bcbf8ec453bae95a074c41dc41d618a5553c2a1321ebd37c9d029baa`
- `plan_json_sha256`: `b82a736a6b04c9b3110ce86ad4993924b88ab5dacadc46650496efc10fec5ef9`
- `refill_plan_json_sha256`: `de9528ed6bcedb4167b4131f31c007e335924f6974d4537e2f45ff25ec091fce`

Raw accepted JSONL 只保存在本地 ignored dataset 目录，不提交 GitHub。

## 结论边界

This release proves the data campaign met its frozen row, token, development, provenance, verifier, critic, deduplication, and call-budget gates. It does not prove model or benchmark uplift.
