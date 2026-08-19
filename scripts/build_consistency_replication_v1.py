#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from nano_data_pipeline.consistency_replication import (
    build_consistency_replication_dataset,
    validate_consistency_replication_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "../../../models/Qwen3.5-4B"
PRIOR_DATASETS = [
    ROOT / "datasets/skill_sft_execution_target_paired_v1.json",
]
PRIOR_LEDGER = (
    ROOT
    / "../../../datasets/ultimate-distill/"
    "skill-sft-10k-10m-v2/accepted.jsonl"
)
SOURCE_RESULT = (
    ROOT.parent
    / "nano-train-skillgen-traex-02/docs/results/"
    "execution_target_paired_consistency_v1.public.json"
)
DATASET = ROOT / "datasets/paired_consistency_replication_v1.json"
RELEASE = ROOT / "manifests/paired_consistency_replication_v1.release.json"
REPORT = ROOT / "docs/datasets/paired_consistency_replication_v1.md"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER,
        local_files_only=True,
    )
    dataset, release = build_consistency_replication_dataset(
        tokenizer=tokenizer,
        tokenizer_path=TOKENIZER,
        prior_dataset_paths=PRIOR_DATASETS,
        prior_accepted_jsonl_path=PRIOR_LEDGER,
        source_result_path=SOURCE_RESULT,
    )
    DATASET.write_text(
        json.dumps(dataset, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reloaded = json.loads(DATASET.read_text(encoding="utf-8"))
    verified = validate_consistency_replication_dataset(
        reloaded,
        tokenizer=tokenizer,
        tokenizer_path=TOKENIZER,
        prior_dataset_paths=PRIOR_DATASETS,
        prior_accepted_jsonl_path=PRIOR_LEDGER,
        source_result_path=SOURCE_RESULT,
    )
    if verified != release:
        raise ValueError("reloaded replication release differs")
    if verified["training_unblocked"] is not True:
        raise ValueError("replication data did not pass every gate")
    RELEASE.write_text(
        json.dumps(verified, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    REPORT.write_text(render_markdown(verified), encoding="utf-8")
    print(json.dumps(verified, indent=2, sort_keys=True))


def render_markdown(release: dict) -> str:
    accepted = release["accepted"]
    sample_size = release["sample_size"]
    overlap = release["overlap"]
    checks = "\n".join(
        f"- `{name}`：{'通过' if passed else '失败'}"
        for name, passed in sorted(release["checks"].items())
    )
    return f"""# Paired Consistency Replication Data v1

## 样本量依据

- consistency v1 在 24 个 fresh final-only case 上修复 1 个、回归 0 个；
- 观测修复率：{sample_size['observed_fix_rate']:.6f}；
- 双侧 exact McNemar 在 0 loss 时至少需要 6 wins 才能 `p<0.05`；
- 在真实修复率 1/24 下，至少 {sample_size['minimum_pairs']} pairs 才有
  {sample_size['target_probability']:.0%} 概率观察到 ≥6 wins；
- 冻结为 {sample_size['frozen_dev_pairs']} fresh dev pairs，组成完整
  16×12 grid；train 也冻结为 {sample_size['frozen_train_pairs']} pairs。

这不是“多跑一点看看”，而是训练前写死的 replication power contract。

## 数据规模

- 总行数：{accepted['rows']:,}；
- Train：{accepted['train_rows']:,} rows，
  {accepted['train_tokens']:,} Qwen3.5 tokens；
- Dev：{accepted['dev_rows']:,} rows；
- Train pairs：{accepted['train_pairs']}；
- Dev pairs：{accepted['dev_pairs']}；
- Train JSON：每 family 64 条；
- Dev JSON：每 family 32 条。

每个 pair 仍包含 verified process view 与 final-only view。objective 权重、teacher
detach、temperature、模型、LR、seed、LoRA scope、prompt/parser 和 decision
threshold 都不允许改变。

## Freshness

- train/dev semantic overlap：{overlap['train_dev_semantic']}；
- train/dev semantic-task overlap：{overlap['train_dev_task']}；
- 与 prior paired dataset sample-ID overlap：
  {overlap['prior_sample_id']}；
- 与 prior paired dataset semantic overlap：
  {overlap['prior_semantic']}；
- 与 prior paired dataset expression overlap：
  {overlap['prior_expression']}；
- 与 10k/10M ledger sample-ID overlap：
  {overlap['prior_ledger_sample_id']}；
- 与 10k/10M ledger semantic overlap：
  {overlap['prior_ledger_semantic']}；
- 与 10k/10M ledger expression overlap：
  {overlap['prior_ledger_expression']}。

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

{checks}

`training_unblocked`：
**{str(release['training_unblocked']).lower()}**。

## Evidence

- dataset canonical SHA256:
  `{release['source']['dataset_canonical_sha256']}`;
- source result SHA256:
  `{release['source']['source_result_sha256']}`;
- prior accepted JSONL SHA256:
  `{release['source']['prior_accepted_jsonl_sha256']}`.

## 结论边界

{release['claim_boundary']}
"""


if __name__ == "__main__":
    main()
