# Qwen3.5 Router Classification Data v1

## 目标

Inference prompt 路线已经负向关闭。本合同冻结 synthetic router SFT 数据，
不使用 benchmark/canary/holdout 或模型输出。

## 规模

- Train：768 rows，A/B/C 各
  256；
- Dev：192 rows，A/B/C 各
  64；
- Total：960；
- Train tokens：至少 50,000 Qwen3.5 tokens；
- NONE 类覆盖4种 unsupported subtype。

## Label

- `FINAL: A` → implicit_scale_total；
- `FINAL: B` → first_strict_profit_period；
- `FINAL: C` → NONE。

Train/dev 模板与数值区间都分离，semantic overlap 必须为0。

## Gate

- `row_count_pass`
- `train_label_balance_pass`
- `dev_label_balance_pass`
- `train_none_subtype_balance_pass`
- `dev_none_subtype_balance_pass`
- `train_dev_semantic_overlap_pass`
- `train_dev_template_overlap_pass`
- `forbidden_content_pass`
- `model_output_absence_pass`
- `teacher_output_absence_pass`
- `benchmark_content_absence_pass`
- `token_accounting_pass`
- `tokenizer_identity_pass`

只有全部通过才允许另行预注册一次 bounded router SFT smoke。

## Boundary

- config SHA：`f6a1a87b2ce77422f18225c7b29fb4d157cc10efa6d4200d337ca32b193df662`；
- generator SHA：`db2e3aa3818f0b30b7df26caa2035bba7a7e3236811ac3ee275ce7760a225d95`；
- data generation started：false；
- dataset/release file exists：false；
- training started：false；
- benchmark/canary/holdout accessed：false。
