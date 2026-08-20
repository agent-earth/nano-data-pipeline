#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_data_pipeline.router_negative_diversity import (
    build_audit,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/router_classification/"
    "qwen35_router_negative_diversity_v2.json"
)
PUBLIC_JSON = (
    ROOT / "docs/datasets/qwen35_router_negative_diversity_v2.public.json"
)
REPORT = ROOT / "docs/datasets/qwen35_router_negative_diversity_v2.md"
CONTRACT = (
    ROOT
    / "configs/router_classification/"
    "qwen35_router_negative_diversity_release_v2.json"
)


def render_markdown(audit: dict) -> str:
    source = audit["source_dataset"]
    train = source["by_split"]["train"]
    dev = source["by_split"]["validation"]
    contract = audit["minimum_fresh_data_contract"]
    return f"""# Qwen3.5 Router Negative-Diversity Audit v2

## 结论

当前缺口不是 C 类样本总量，而是 subtype 和词法覆盖：

- 旧 release 有 {source['negative_rows']} 条 C；
- train C：{train['rows']}，dev C：{dev['rows']}；
- 每个 subtype 在每个 split 只有 1 个模板 / 1 个 generation rule；
- train/dev 共 {train['rows'] + dev['rows']} 条 C 全部显式要求
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
{json.dumps(source, indent=2, sort_keys=True)}
```

## Fresh Data Contract

- train：{contract['rows']['train']:,}，A/B/C 各
  {contract['train_by_label']['A']:,}；
- dev：{contract['rows']['dev']:,}，A/B/C 各
  {contract['dev_by_label']['A']:,}；
- C 扩展到 {len(contract['negative_subtypes'])} 个 subtype；
- 每个 subtype train/dev：
  {next(iter(contract['negative_subtypes'].values()))['train_rows']}/
  {next(iter(contract['negative_subtypes'].values()))['dev_rows']}；
- 每个 subtype 至少 16 个 train 模板、4 个 dev 模板；
- train 至少 75% natural answer-task，dev 100% natural answer-task；
- 至少 {contract['minimum_train_tokens']:,} tokenizer-counted train tokens；
- 与 v1、V1/V2 integration prompt hashes、完整 benchmark prompt hashes 的
  overlap 都必须为0；
- integration V1/V2 rows/outputs、benchmark/canary/holdout 内容、模型/teacher
  outputs 全部禁止。

合同全部通过后，只允许另行预注册一次 SFT；当前不允许训练。

## Decision

```json
{json.dumps(audit['decision'], indent=2, sort_keys=True)}
```

## Boundary

{audit['claim_boundary']}
"""


def main() -> None:
    config = load_config(CONFIG)
    audit = build_audit(config)
    PUBLIC_JSON.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_JSON.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    REPORT.write_text(render_markdown(audit), encoding="utf-8")
    CONTRACT.write_text(
        json.dumps(
            audit["minimum_fresh_data_contract"],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audit_id": audit["audit_id"],
                "findings": audit["findings"],
                "decision": audit["decision"],
                "contract": audit["minimum_fresh_data_contract"],
                "public_json": str(PUBLIC_JSON),
                "markdown": str(REPORT),
                "contract_path": str(CONTRACT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
