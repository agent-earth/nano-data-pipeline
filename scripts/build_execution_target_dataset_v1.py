#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from nano_data_pipeline.execution_target_dataset import (
    build_execution_target_dataset,
    validate_execution_target_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "../../../models/Qwen3.5-4B"
ACCEPTED = (
    ROOT
    / "../../../datasets/ultimate-distill/"
    "skill-sft-10k-10m-v2/accepted.jsonl"
)
PRIOR_RELEASE = (
    ROOT
    / "../../../datasets/ultimate-distill/"
    "skill-sft-10k-10m-v2/release.json"
)
AUDIT = (
    ROOT
    / "docs/datasets/skill_release_execution_target_coverage_v1.public.json"
)
DATASET = ROOT / "datasets/skill_sft_execution_target_paired_v1.json"
RELEASE = ROOT / "manifests/skill_sft_execution_target_paired_v1.release.json"
REPORT = ROOT / "docs/datasets/skill_sft_execution_target_paired_v1.md"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER,
        local_files_only=True,
    )
    dataset, generated_release = build_execution_target_dataset(
        tokenizer=tokenizer,
        accepted_jsonl_path=ACCEPTED,
        release_manifest_path=PRIOR_RELEASE,
        audit_path=AUDIT,
        tokenizer_path=TOKENIZER,
    )
    DATASET.write_text(
        json.dumps(dataset, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reloaded = json.loads(DATASET.read_text(encoding="utf-8"))
    verified_release = validate_execution_target_dataset(
        reloaded,
        accepted_jsonl_path=ACCEPTED,
        release_manifest_path=PRIOR_RELEASE,
        audit_path=AUDIT,
        tokenizer=tokenizer,
        tokenizer_path=TOKENIZER,
    )
    if verified_release != generated_release:
        raise ValueError("reloaded execution-target release differs")
    if verified_release["training_unblocked"] is not True:
        raise ValueError("execution-target dataset did not pass every gate")
    RELEASE.write_text(
        json.dumps(verified_release, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    REPORT.write_text(
        render_markdown(verified_release),
        encoding="utf-8",
    )
    print(json.dumps(verified_release, indent=2, sort_keys=True))


def render_markdown(release: dict) -> str:
    accepted = release["accepted"]
    overlap = release["overlap"]
    checks = "\n".join(
        f"- `{name}`：{'通过' if passed else '失败'}"
        for name, passed in sorted(release["checks"].items())
    )
    train_family = "、".join(
        f"{family} {count} 条"
        for family, count in accepted["train_json_by_family"].items()
    )
    dev_family = "、".join(
        f"{family} {count} 条"
        for family, count in accepted["dev_json_by_family"].items()
    )
    return f"""# Skill SFT Execution-Target Paired Data v1

## 生成了什么

- 总行数：{accepted['rows']}；
- Train：{accepted['train_rows']} 行，{accepted['train_tokens']:,} tokens；
- Dev：{accepted['dev_rows']} 行；
- Train relation paired views：{accepted['train_relation_views']}；
- Dev relation paired views：{accepted['dev_relation_views']}；
- Train JSON preservation：{train_family}；
- Dev JSON non-regression：{dev_family}。

每个 relation semantic task 都有两个 view：

1. `process`：逐步执行并验证每个中间值；
2. `final`：只输出最终答案，防止模型只会套 STEP 模板。

## 为什么这版和旧数据不同

旧 release 覆盖目标公式关系，但只有 final-only 监督；v6 有 process
监督，却没有覆盖“同一个操作数先相加、最后再减掉”的关系。这版数据把两者放在
同一 semantic task 的 paired views 中。

## Token 与分布

- Qwen3.5 tokenizer 真实计数：{accepted['train_tokens']:,}；
- 最低要求：300,000；
- Relation train：128 semantic tasks × 2 views；
- JSON train：256 行，每个 family 64 行；
- Relation dev：24 semantic tasks × 2 views；
- JSON dev：32 行，每个 family 8 行。

## 重叠与泄露

- train/dev semantic overlap：{overlap['train_dev_semantic']}；
- train/dev semantic-task overlap：{overlap['train_dev_task_semantic']}；
- 与旧 release semantic overlap：{overlap['prior_release_semantic']}；
- 与旧 release expression overlap：{overlap['prior_release_expression']}；
- 与旧 release sample-ID overlap：{overlap['prior_release_sample_id']}；
- answer value leakage：{len(release['leakage']['answer_value_rows'])}；
- forbidden content rows：{len(release['leakage']['forbidden_content_rows'])}。

## Gate

{checks}

`training_unblocked`：
**{str(release['training_unblocked']).lower()}**。

## Evidence

- dataset canonical SHA256:
  `{release['source']['dataset_canonical_sha256']}`;
- coverage audit SHA256:
  `{release['source']['coverage_audit_sha256']}`;
- prior release manifest SHA256:
  `{release['source']['prior_release_manifest_sha256']}`;
- prior accepted JSONL SHA256:
  `{release['source']['prior_accepted_jsonl_sha256']}`.

## 结论边界

{release['claim_boundary']}
"""


if __name__ == "__main__":
    main()
