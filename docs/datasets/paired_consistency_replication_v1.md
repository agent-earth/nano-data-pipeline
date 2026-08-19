# Paired Consistency Replication Data v1

## 样本量依据

- consistency v1 在 24 个 fresh final-only case 上修复 1 个、回归 0 个；
- 观测修复率：0.041667；
- 双侧 exact McNemar 在 0 loss 时至少需要 6 wins 才能 `p<0.05`；
- 在真实修复率 1/24 下，至少 189 pairs 才有
  80% 概率观察到 ≥6 wins；
- 冻结为 192 fresh dev pairs，组成完整
  16×12 grid；train 也冻结为 192 pairs。

这不是“多跑一点看看”，而是训练前写死的 replication power contract。

## 数据规模

- 总行数：1,152；
- Train：640 rows，
  405,007 Qwen3.5 tokens；
- Dev：512 rows；
- Train pairs：192；
- Dev pairs：192；
- Train JSON：每 family 64 条；
- Dev JSON：每 family 32 条。

每个 pair 仍包含 verified process view 与 final-only view。objective 权重、teacher
detach、temperature、模型、LR、seed、LoRA scope、prompt/parser 和 decision
threshold 都不允许改变。

## Freshness

- train/dev semantic overlap：0；
- train/dev semantic-task overlap：0；
- 与 prior paired dataset sample-ID overlap：
  0；
- 与 prior paired dataset semantic overlap：
  0；
- 与 prior paired dataset expression overlap：
  0；
- 与 10k/10M ledger sample-ID overlap：
  0；
- 与 10k/10M ledger semantic overlap：
  0；
- 与 10k/10M ledger expression overlap：
  0。

## Significance Gate

- aggregate bootstrap CI lower > 0；
- aggregate exact McNemar p < 0.05；
- final bootstrap CI lower > 0；
- final exact McNemar p < 0.05；
- both-pair bootstrap CI lower > 0；
- both-pair exact McNemar p < 0.05；
- final-only wins ≥ 6；
- final-only losses = 0；
- every JSON family non-regression。

## Gate

- `answer_leakage_pass`：通过
- `dev_json_family_pass`：通过
- `dev_pairs_pass`：通过
- `dev_rows_pass`：通过
- `exact_hash_unique_pass`：通过
- `final_verifier_pass`：通过
- `forbidden_content_pass`：通过
- `hash_recomputation_pass`：通过
- `json_verifier_pass`：通过
- `paired_consistency_pass`：通过
- `prior_expression_overlap_pass`：通过
- `prior_ledger_expression_overlap_pass`：通过
- `prior_ledger_sample_id_overlap_pass`：通过
- `prior_ledger_semantic_overlap_pass`：通过
- `prior_sample_id_overlap_pass`：通过
- `prior_semantic_overlap_pass`：通过
- `prior_task_overlap_pass`：通过
- `process_verifier_pass`：通过
- `sample_id_unique_pass`：通过
- `sample_size_pass`：通过
- `semantic_hash_unique_pass`：通过
- `semantic_task_hash_pairing_pass`：通过
- `source_result_identity_pass`：通过
- `token_accounting_pass`：通过
- `tokenizer_identity_pass`：通过
- `train_dev_semantic_overlap_pass`：通过
- `train_dev_task_overlap_pass`：通过
- `train_json_family_pass`：通过
- `train_pairs_pass`：通过
- `train_rows_pass`：通过

`training_unblocked`：
**true**。

## Evidence

- dataset canonical SHA256:
  `40514440b95b94981a756a8f325d5a1111db1ec7bc94125875e40e8bc3898797`;
- source result SHA256:
  `76968fc034c0bf6543d3e140129280140274d7296d93834e569de53cd1bfb856`;
- prior accepted JSONL SHA256:
  `b5503761900cfa290dba03aff306ff511630d01544a4dab93de3a0be1e74abc1`.

## 结论边界

这份 release 只证明更大的 fresh replication data 和 significance contract 通过全部检查。它不证明模型能力提升，也不允许访问 benchmark/holdout 或启动 RL。
