#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from nano_data_pipeline.orca_math import (
    build_dataset,
    load_config,
    validate_dataset,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/orca_math_sft_v1.json"


def render_markdown(release: dict) -> str:
    accepted = release["accepted"]
    filtering = release["filtering"]
    checks = "\n".join(
        f"- `{name}`: {'pass' if value else 'fail'}"
        for name, value in sorted(release["checks"].items())
    )
    return f"""# Orca Math SFT v1 Data Release

## Result

- Train: {accepted['train_rows']:,} rows,
  {accepted['train_tokens']:,} Qwen3.5 tokens;
- development: {accepted['dev_rows']:,} rows,
  {accepted['dev_tokens']:,} tokens;
- token quantiles:
  `{json.dumps(accepted['token_quantiles'], sort_keys=True)}`;
- split/stratum counts:
  `{json.dumps(accepted['by_split_stratum'], sort_keys=True)}`;
- dataset SHA256:
  `{release['source']['dataset_file_sha256']}`.

The local dataset retains external teacher reasoning and appends a normalized
numeric `FINAL:` line. It is stored outside GitHub. The public release contains
only aggregate counts, identities, and gate receipts.

## Filtering

- Source rejections:
  `{json.dumps(filtering['source_rejected'], sort_keys=True)}`;
- selection rejections:
  `{json.dumps(filtering['selection_rejected'], sort_keys=True)}`;
- overlap counts:
  `{json.dumps(filtering['overlap_counts'], sort_keys=True)}`.

## Gates

{checks}

`training_unblocked`: **{str(release['training_unblocked']).lower()}**.

## Boundary

{release['claim_boundary']}
"""


def main() -> None:
    config = load_config(CONFIG)
    tokenizer_path = config.resolve(
        config.raw["token_accounting"]["tokenizer_path"]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    rows, build_audit = build_dataset(config, tokenizer=tokenizer)
    dataset_path = config.resolve(config.raw["output"]["dataset_path"])
    local_release_path = config.resolve(
        config.raw["output"]["local_release_path"]
    )
    public_release_path = config.resolve(
        config.raw["output"]["public_release_path"]
    )
    write_jsonl(dataset_path, rows)
    release = validate_dataset(
        config,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        build_audit=build_audit,
    )
    if release["training_unblocked"] is not True:
        raise ValueError("Orca Math SFT release did not pass every gate")
    local_release_path.parent.mkdir(parents=True, exist_ok=True)
    local_release_path.write_text(
        json.dumps(
            {
                "release": release,
                "build_audit": build_audit,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    public_release_path.parent.mkdir(parents=True, exist_ok=True)
    public_release_path.write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = (
        ROOT / "docs/datasets/orca_math_sft_v1.md"
    )
    report_path.write_text(render_markdown(release), encoding="utf-8")
    print(json.dumps(release, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
