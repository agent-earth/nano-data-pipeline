# Format Contract Curriculum Analog v2

## Purpose

`format-contract-curriculum-analog-v2` is a fresh, two-step-heavy synthetic
dataset for FP32 SFT v3. It follows SFT v2's stable 25/26 result, where the sole
remaining validation failure was a valid-format but semantically wrong
two-step precedence example.

It does not reuse the observed v1 validation split.

## Composition

- 160 samples;
- 80 choice-contract and 80 numeric-contract samples;
- 32 single-step and 128 two-step samples;
- 128 train and 32 validation samples.

Difficulty by split:

- train: 24 single-step and 104 two-step;
- validation: 8 single-step and 24 two-step.

Validation positions rotate within each five-sample generation block. This
prevents validation membership from being identical to one expression mode.

## Freshness And Deduplication

The generator checks all v2 samples against `format-contract-analog-v1`:

- sample ID overlap: 0;
- exact message hash overlap: 0;
- normalized semantic hash overlap: 0.

Within v2, all 160 exact and all 160 semantic hashes are unique.

## Leakage Boundary

Like v1, v2 is deterministic synthetic arithmetic:

- no benchmark prompt, answer, prediction, source index, or raw model output;
- no teacher output;
- no sealed case ID;
- all sample IDs use the `synthetic-` namespace.

The sealed feedback manifest supplies only a requirement/provenance identity.

## SFT v3 Boundary

SFT v3 must retain v2's FP32 numerical fix, fail-fast checks, model identity,
seed, LoRA scope, effective batch, learning rate, and 20-step cap. Only dataset
identity changes.

The validation split is fresh. Do not use the observed v1 validation split for
v3 acceptance. Benchmark evaluation remains blocked until v3 passes its
synthetic smoke and reload checks.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli build-curriculum-analog \
  --feedback-manifest manifests/qwen35_large_confirmation_feedback_v1.json \
  --prior-dataset datasets/format_contract_analog_v1.json \
  --output datasets/format_contract_curriculum_analog_v2.json
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli validate-analog \
  datasets/format_contract_curriculum_analog_v2.json
PYTHONPATH=. ../.venv/bin/python scripts/validate_release.py
```
