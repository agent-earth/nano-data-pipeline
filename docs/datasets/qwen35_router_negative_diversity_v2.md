# Qwen3.5 Router Negative-Diversity Audit v2

## 结论

当前缺口不是 C 类样本总量，而是 subtype 和词法覆盖：

- 旧 release 有 320 条 C；
- train C：256，dev C：64；
- 每个 subtype 在每个 split 只有 1 个模板 / 1 个 generation rule；
- train/dev 共 320 条 C 全部显式要求
  `route/classify`；
- 自然 answer-task C：0 条。

在 namespace 修复后，fresh V2 上：

- A/B：64/64；
- remaining-stock C：32/32；
- box-total C：0/32，全部误判为 A；
- false routes / fallbacks：32/32。

审计只读取 public aggregate subtype 结果和 prereg prompt hashes，没有读取 V1/V2
rows 或 outputs。

## Source Coverage

```json
{
  "by_split": {
    "train": {
      "answer_task_rows": 0,
      "by_subtype": {
        "box_total": {
          "answer_task_rows": 0,
          "explicit_classification_rows": 64,
          "rows": 64,
          "unique_generation_rules": 1,
          "unique_template_ids": 1
        },
        "paired_average": {
          "answer_task_rows": 0,
          "explicit_classification_rows": 64,
          "rows": 64,
          "unique_generation_rules": 1,
          "unique_template_ids": 1
        },
        "remaining_stock": {
          "answer_task_rows": 0,
          "explicit_classification_rows": 64,
          "rows": 64,
          "unique_generation_rules": 1,
          "unique_template_ids": 1
        },
        "single_operation": {
          "answer_task_rows": 0,
          "explicit_classification_rows": 64,
          "rows": 64,
          "unique_generation_rules": 1,
          "unique_template_ids": 1
        }
      },
      "explicit_classification_rows": 256,
      "rows": 256
    },
    "validation": {
      "answer_task_rows": 0,
      "by_subtype": {
        "box_total": {
          "answer_task_rows": 0,
          "explicit_classification_rows": 16,
          "rows": 16,
          "unique_generation_rules": 1,
          "unique_template_ids": 1
        },
        "paired_average": {
          "answer_task_rows": 0,
          "explicit_classification_rows": 16,
          "rows": 16,
          "unique_generation_rules": 1,
          "unique_template_ids": 1
        },
        "remaining_stock": {
          "answer_task_rows": 0,
          "explicit_classification_rows": 16,
          "rows": 16,
          "unique_generation_rules": 1,
          "unique_template_ids": 1
        },
        "single_operation": {
          "answer_task_rows": 0,
          "explicit_classification_rows": 16,
          "rows": 16,
          "unique_generation_rules": 1,
          "unique_template_ids": 1
        }
      },
      "explicit_classification_rows": 64,
      "rows": 64
    }
  },
  "label_counts": {
    "A": 320,
    "B": 320,
    "C": 320
  },
  "negative_rows": 320,
  "negative_subtypes": [
    "box_total",
    "paired_average",
    "remaining_stock",
    "single_operation"
  ],
  "total_rows": 960
}
```

## Fresh Data Contract

- train：6,144，A/B/C 各
  2,048；
- dev：1,536，A/B/C 各
  512；
- C 扩展到 8 个 subtype；
- 每个 subtype train/dev：
  256/
  64；
- 每个 subtype 至少 16 个 train 模板、4 个 dev 模板；
- train 至少 75% natural answer-task，dev 100% natural answer-task；
- 至少 600,000 tokenizer-counted train tokens；
- 与 v1、V1/V2 integration prompt hashes、完整 benchmark prompt hashes 的
  overlap 都必须为0；
- integration V1/V2 rows/outputs、benchmark/canary/holdout 内容、模型/teacher
  outputs 全部禁止。

合同全部通过后，只允许另行预注册一次 SFT；当前不允许训练。

## Decision

```json
{
  "benchmark_allowed": false,
  "canary_allowed": false,
  "generate_negative_diversity_v2_next": true,
  "holdout_allowed": false,
  "integration_v1_or_v2_training_use_allowed": false,
  "reuse_v1_data_unchanged": false,
  "rl_allowed": false,
  "training_allowed_now": false
}
```

## Boundary

This audit compares the released synthetic router data with public aggregate subtype outcomes and preregistered prompt hashes. It loads no integration rows or outputs, performs no model generation, and does not establish model or benchmark quality.
