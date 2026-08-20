# Qwen3.5 Router Classification Data v1

## Release

- Train：768 rows，A/B/C 各256；
- Dev：192 rows，A/B/C 各64；
- Total：960；
- Train tokens：84,160；
- NONE train/dev：4种 subtype 各64/16；
- train/dev semantic overlap：0；
- train/dev template overlap：0；
- forbidden terms：0。

## Gate

- `benchmark_content_absence_pass`：通过
- `dev_label_balance_pass`：通过
- `dev_none_subtype_balance_pass`：通过
- `forbidden_content_pass`：通过
- `model_output_absence_pass`：通过
- `row_count_pass`：通过
- `teacher_output_absence_pass`：通过
- `token_accounting_pass`：通过
- `tokenizer_identity_pass`：通过
- `train_dev_semantic_overlap_pass`：通过
- `train_dev_template_overlap_pass`：通过
- `train_label_balance_pass`：通过
- `train_none_subtype_balance_pass`：通过

`training_unblocked`：**true**。

## Evidence

- dataset canonical SHA：
  `b9f4ef24f16c680f6c5d5999e3ca86cd7c044b83e093d18b39f7e220da70bfad`；
- multiclass negative report SHA：
  `c8e4034a27e925025589bc1a8a52abc6720ee0d7fc97e03983ff192cd44c3742`；
- binary detector negative report SHA：
  `0f50860efd48378be11314f83a38025bca533180d63f44b5bcaf902045ef2ae4`。

## 边界

This release proves only that a deterministic synthetic router classification dataset passed frozen balance, deduplication, split, token, provenance, and leakage gates. It is not model quality or benchmark evidence and unlocks only one separately pre-registered bounded SFT smoke.
