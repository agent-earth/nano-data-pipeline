# Qwen3.5 Router Negative-Diversity Release v2

## Release

- Train：6,144，A/B/C 各
  2,048；
- Dev：1,536，A/B/C 各
  512；
- Total：7,680；
- C subtypes：8；
- 每个 subtype train/dev：
  256/
  64；
- 每个 subtype train/dev templates：
  16/
  4；
- answer-task fraction train/dev：
  1.000/
  1.000；
- Train tokens：766,519。

## Overlap

```json
{
  "benchmark_prompts": {
    "gpqa_diamond": 0,
    "gsm8k": 0,
    "mmlu": 0
  },
  "benchmark_rows_hashed": {
    "gpqa_diamond": 198,
    "gsm8k": 1319,
    "mmlu": 14042
  },
  "integration_prompts": {
    "integration_v1": 0,
    "integration_v2": 0
  },
  "source_v1_sample_ids": 0,
  "source_v1_semantic": 0,
  "train_dev_semantic": 0
}
```

## Gates

- `all_exact_hashes_unique`：通过
- `all_overlap_counts_zero`：通过
- `all_sample_ids_unique`：通过
- `all_semantic_hashes_unique`：通过
- `answer_task_fraction_pass`：通过
- `dev_answer_task_only_pass`：通过
- `exact_negative_subtype_balance`：通过
- `exact_row_and_class_balance`：通过
- `forbidden_content_zero`：通过
- `minimum_train_tokens_pass`：通过
- `template_diversity_pass`：通过
- `tokenizer_identity_pinned`：通过

`training_unblocked`：**true**。

## Evidence

- dataset canonical SHA：
  `f63c58b54ef4747f274599784bad9ffe4143117482c22b33005a2dbf725b1f2f`；
- audit SHA：`9aaa69de746dbdc5cefbb52fb271c8f9ec86716d10ada70704c7e346dc2f7c17`；
- contract SHA：`c195a7373ea283546dde1866f70593f0912833d987ff5f1a8cb424c2bc340335`；
- tokenizer SHA：
  `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`。

## Boundary

This release proves only deterministic synthetic data quality, balance, diversity, token, overlap, provenance, and leakage gates. It does not establish model quality and unlocks only one separately pre-registered router SFT run.
