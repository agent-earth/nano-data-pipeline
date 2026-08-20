# Orca Math SFT v1 Data Pre-Registration

## Frozen Release

- Source: `microsoft/orca-math-word-problems-200k` at revision
  `29255d1770cc4eac66e5e7fa378cba542c026350`;
- license: `mit`;
- source rows: 200,035;
- selected train rows: 32,768;
- selected development rows: 1,024;
- Qwen3.5 train-token minimum: 10,000,000;
- maximum full sequence: 1,024 tokens;
- split seed: `orca-math-sft-v1:20260821`.

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
