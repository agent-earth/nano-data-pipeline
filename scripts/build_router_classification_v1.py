#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from nano_data_pipeline.router_classification import (
    build_dataset,
    load_config,
    validate_router_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/router_classification/qwen35_router_classification_v1.json"
)
REPORT = ROOT / "docs/datasets/qwen35_router_classification_v1.md"


def main() -> None:
    config = load_config(CONFIG)
    tokenizer_path = (ROOT / config.tokenizer_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    dataset = build_dataset(config)
    release = validate_router_dataset(
        dataset,
        config=config,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
    )
    if not release["training_unblocked"]:
        raise ValueError("router release did not pass all checks")
    dataset_path = ROOT / config.output_dataset_path
    release_path = ROOT / config.output_release_path
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    release_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps(dataset, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reloaded = json.loads(dataset_path.read_text(encoding="utf-8"))
    revalidated = validate_router_dataset(
        reloaded,
        config=config,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
    )
    if revalidated != release:
        raise ValueError("router release differs after reload")
    release_path.write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_markdown(release), encoding="utf-8")
    print(json.dumps(release, indent=2, sort_keys=True))


def render_markdown(release: dict) -> str:
    accepted = release["accepted"]
    checks = "\n".join(
        f"- `{name}`：{'通过' if passed else '失败'}"
        for name, passed in sorted(release["checks"].items())
    )
    return f"""# Qwen3.5 Router Classification Data v1

## Release

- Train：{accepted['train_rows']} rows，A/B/C 各256；
- Dev：{accepted['dev_rows']} rows，A/B/C 各64；
- Total：{accepted['rows']}；
- Train tokens：{accepted['train_tokens']:,}；
- NONE train/dev：4种 subtype 各64/16；
- train/dev semantic overlap：{release['overlap']['train_dev_semantic']}；
- train/dev template overlap：{release['overlap']['train_dev_template']}；
- forbidden terms：{len(release['leakage']['forbidden_terms'])}。

## Gate

{checks}

`training_unblocked`：**{str(release['training_unblocked']).lower()}**。

## Evidence

- dataset canonical SHA：
  `{release['source']['dataset_canonical_sha256']}`；
- multiclass negative report SHA：
  `{release['source']['multiclass_report_sha256']}`；
- binary detector negative report SHA：
  `{release['source']['binary_detector_report_sha256']}`。

## 边界

{release['claim_boundary']}
"""


if __name__ == "__main__":
    main()
