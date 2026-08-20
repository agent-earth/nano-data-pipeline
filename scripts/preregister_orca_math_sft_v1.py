#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_data_pipeline.orca_math import build_preregister, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/orca_math_sft_v1.json"
OUTPUT = ROOT / "docs/experiments/orca_math_sft_v1.preregister.json"
MARKDOWN = ROOT / "docs/experiments/orca_math_sft_v1.md"


def render_markdown(receipt: dict) -> str:
    selection = receipt["selection"]
    tokens = receipt["token_accounting"]
    return f"""# Orca Math SFT v1 Data Pre-Registration

## Frozen Release

- Source: `microsoft/orca-math-word-problems-200k` at revision
  `{receipt['source']['revision']}`;
- license: `{receipt['source']['license']}`;
- source rows: {receipt['source']['rows']:,};
- selected train rows: {selection['train_rows']:,};
- selected development rows: {selection['dev_rows']:,};
- Qwen3.5 train-token minimum: {tokens['minimum_train_tokens']:,};
- maximum full sequence: {tokens['max_sequence_tokens']:,} tokens;
- split seed: `{selection['seed']}`.

The release preserves the teacher reasoning, normalizes a final numeric answer,
and requires exact plus near-duplicate exclusion against GSM8K, MMLU, and
GPQA source questions. Benchmark rows and model outputs are never training
eligible.

## Boundary

This commit only pre-registers source identities, deterministic selection,
quality filters, token accounting, contamination checks, and output paths.
It does not select rows, generate model outputs, train a model, or report model
quality. Passing the data release unlocks only one separately pre-registered
SFT smoke; RL and OPD remain closed.
"""


def main() -> None:
    receipt = build_preregister(load_config(CONFIG))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(receipt), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
