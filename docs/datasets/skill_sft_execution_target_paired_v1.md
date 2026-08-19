# Skill SFT Execution-Target Paired Data v1

## 生成了什么

- 总行数：592；
- Train：512 行，324,545 tokens；
- Dev：80 行；
- Train relation paired views：256；
- Dev relation paired views：48；
- Train JSON preservation：coding-and-validation 64 条、planning-and-state 64 条、skill-routing-and-reflection 64 条、tool-use-and-recovery 64 条；
- Dev JSON non-regression：coding-and-validation 8 条、planning-and-state 8 条、skill-routing-and-reflection 8 条、tool-use-and-recovery 8 条。

每个 relation semantic task 都有两个 view：

1. `process`：逐步执行并验证每个中间值；
2. `final`：只输出最终答案，防止模型只会套 STEP 模板。

## 为什么这版和旧数据不同

旧 release 覆盖目标公式关系，但只有 final-only 监督；v6 有 process
监督，却没有覆盖“同一个操作数先相加、最后再减掉”的关系。这版数据把两者放在
同一 semantic task 的 paired views 中。

## Token 与分布

- Qwen3.5 tokenizer 真实计数：324,545；
- 最低要求：300,000；
- Relation train：128 semantic tasks × 2 views；
- JSON train：256 行，每个 family 64 行；
- Relation dev：24 semantic tasks × 2 views；
- JSON dev：32 行，每个 family 8 行。

## 重叠与泄露

- train/dev semantic overlap：0；
- train/dev semantic-task overlap：0；
- 与旧 release semantic overlap：0；
- 与旧 release expression overlap：0；
- 与旧 release sample-ID overlap：0；
- answer value leakage：0；
- forbidden content rows：0。

## Gate

- `answer_value_leakage_pass`：通过
- `deterministic_verifier_pass`：通过
- `dev_json_family_quotas_pass`：通过
- `dev_json_rows_pass`：通过
- `dev_relation_views_pass`：通过
- `dev_rows_pass`：通过
- `exact_hash_unique_pass`：通过
- `forbidden_content_pass`：通过
- `hash_recomputation_pass`：通过
- `paired_view_consistency_pass`：通过
- `prior_release_expression_overlap_pass`：通过
- `prior_release_sample_id_overlap_pass`：通过
- `prior_release_semantic_overlap_pass`：通过
- `sample_id_unique_pass`：通过
- `semantic_hash_unique_pass`：通过
- `token_accounting_pass`：通过
- `tokenizer_identity_pass`：通过
- `train_dev_semantic_overlap_pass`：通过
- `train_dev_task_overlap_pass`：通过
- `train_json_family_quotas_pass`：通过
- `train_json_rows_pass`：通过
- `train_relation_views_pass`：通过
- `train_rows_pass`：通过
- `train_tokens_pass`：通过

`training_unblocked`：
**true**。

## Evidence

- dataset canonical SHA256:
  `77728a0531f18e55989c172a21fb267284aa1001c17fd62de6bcd13b9d300659`;
- coverage audit SHA256:
  `aca3f5995eda6ba4d4f7015b11ce4062b15653e404677145eec2b4292d280817`;
- prior release manifest SHA256:
  `26ddb15f5c2e043d20527103a5a59216e54290aabeea2a6d228ebce7b7bb35e3`;
- prior accepted JSONL SHA256:
  `b5503761900cfa290dba03aff306ff511630d01544a4dab93de3a0be1e74abc1`.

## 结论边界

这份 release 只证明冻结的 synthetic data contract 通过了行数、token、paired view、verifier、provenance、overlap 和 leakage 检查。它不证明模型能力或 benchmark 指标提升。
