# Format Contract Analog v1

## Purpose

`format-contract-analog-v1` is a deterministic synthetic dataset for a minimal
SFT smoke. It tests whether training can improve exact `FINAL:` contract
compliance without consuming sealed benchmark content.

It is not evidence of semantic benchmark improvement.

## Composition

- 128 training-eligible samples;
- 64 choice-contract and 64 numeric-contract samples;
- 96 single-step and 32 two-step arithmetic samples;
- 102 train and 26 validation samples.

Each sample contains `system`, `user`, and `assistant` chat messages. Targets
are exactly one of:

- `FINAL: <A-D>`;
- `FINAL: <number>`.

## Source And Leakage Boundary

Samples are generated from fixed arithmetic templates and index values. The
generator does not read benchmark prompts, references, predictions, source
indices, or raw model outputs.

The sealed feedback manifest is used only as a versioned requirement and
provenance reference. The builder scans the rendered analog samples for every
sealed case ID and fails on a match.

All sample IDs start with `synthetic-`; benchmark-style IDs are forbidden.

## Deduplication

- exact deduplication hashes canonical chat messages;
- semantic deduplication hashes lowercased, whitespace-normalized role/content
  sequences;
- all 128 exact and all 128 semantic hashes are unique.

## Split Policy

For each 64-sample task family, indices divisible by five enter validation and
all others enter train. This deterministic policy yields 102 train and 26
validation samples.

The validation split checks format-contract learning only. It is not a
benchmark, holdout, or claim of downstream quality.

## SFT Smoke Boundary

The first training run must remain a smoke:

- use only the 102 analog train samples;
- evaluate only the 26 analog validation samples plus the unchanged matched
  benchmark harness;
- compare base model versus SFT checkpoint;
- require no GSM8K/MMLU/GPQA regression before any scale-up;
- do not start RL until SFT artifact, loss, format accuracy, and matched
  benchmark evidence are reproducible.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli build-format-analog \
  --feedback-manifest manifests/qwen35_large_confirmation_feedback_v1.json \
  --output datasets/format_contract_analog_v1.json
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli validate-analog \
  datasets/format_contract_analog_v1.json
PYTHONPATH=. ../.venv/bin/python scripts/validate_release.py
```
