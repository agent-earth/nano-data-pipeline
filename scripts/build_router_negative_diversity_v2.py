#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from nano_data_pipeline.router_negative_diversity_release import (
    build_dataset,
    load_build_config,
    validate_release,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/router_classification/"
    "qwen35_router_negative_diversity_build_v2.json"
)
REPORT = ROOT / "docs/datasets/qwen35_router_negative_diversity_release_v2.md"


def render_markdown(release: dict) -> str:
    accepted = release["accepted"]
    checks = "\n".join(
        f"- `{name}`：{'通过' if passed else '失败'}"
        for name, passed in sorted(release["checks"].items())
    )
    return f"""# Qwen3.5 Router Negative-Diversity Release v2

## Release

- Train：{accepted['train_rows']:,}，A/B/C 各
  {accepted['train_by_label']['A']:,}；
- Dev：{accepted['dev_rows']:,}，A/B/C 各
  {accepted['dev_by_label']['A']:,}；
- Total：{accepted['rows']:,}；
- C subtypes：{len(accepted['train_none_by_subtype'])}；
- 每个 subtype train/dev：
  {next(iter(accepted['train_none_by_subtype'].values()))}/
  {next(iter(accepted['dev_none_by_subtype'].values()))}；
- 每个 subtype train/dev templates：
  {next(iter(accepted['train_templates_by_subtype'].values()))}/
  {next(iter(accepted['dev_templates_by_subtype'].values()))}；
- answer-task fraction train/dev：
  {accepted['answer_task_fraction']['train']:.3f}/
  {accepted['answer_task_fraction']['validation']:.3f}；
- Train tokens：{accepted['train_tokens']:,}。

## Overlap

```json
{json.dumps(release['overlap'], indent=2, sort_keys=True)}
```

## Gates

{checks}

`training_unblocked`：**{str(release['training_unblocked']).lower()}**。

## Evidence

- dataset canonical SHA：
  `{release['source']['dataset_canonical_sha256']}`；
- audit SHA：`{release['source']['audit_sha256']}`；
- contract SHA：`{release['source']['contract_sha256']}`；
- tokenizer SHA：
  `{release['source']['tokenizer_file_sha256']['tokenizer.json']}`。

## Boundary

{release['claim_boundary']}
"""


def main() -> None:
    config = load_build_config(CONFIG)
    tokenizer_path = (ROOT / config.tokenizer_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    dataset = build_dataset(config)
    release = validate_release(
        dataset,
        config=config,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
    )
    if not release["training_unblocked"]:
        raise ValueError("negative-diversity release did not pass all gates")
    dataset_path = ROOT / config.output_dataset_path
    release_path = ROOT / config.output_release_path
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    release_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps(dataset, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reloaded = json.loads(dataset_path.read_text(encoding="utf-8"))
    revalidated = validate_release(
        reloaded,
        config=config,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
    )
    if revalidated != release:
        raise ValueError("negative-diversity release differs after reload")
    release_path.write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_markdown(release), encoding="utf-8")
    print(json.dumps(release, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
