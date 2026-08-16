# Packing Isolation Preservation Mix v9

## Purpose

Broad v7 mixes three failure-targeted numerical families. V13 shows that the
percentage family alone is harmful at its frozen exposure. V9 isolates packing
efficiency as the next independent mechanism test.

## Intervention

Starting from `targeted-preservation-mix-v6`, v9 changes exactly 8 numeric
training rows. The replacements are byte-identical to the verified packing
subset from broad v7.

The other 184 rows are byte-identical to v6, including all development,
targeted-host, choice, process, percentage, and recurring-schedule strata.
The frozen 32-step schedule exposes 5 packing replacements.

## Boundary

The builder consumes only an irreversible abstract-family receipt and fresh
synthetic rows. It uses no benchmark/canary payload, model output, teacher
output, or independent-holdout row. All targets are restricted-AST verified.

## Verification

- samples: 192;
- train / development: 160 / 32;
- replacement / unchanged rows: 8 / 184;
- ID, exact, semantic, and source-signature overlap with v1-v6: 0;
- dataset SHA256:
  `9f79b1cf5af9fa4b36c7507318b32991692f253d2210b5b6ed70a44bee940f2d`.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli \
  build-packing-isolation-preservation-mix \
  --feedback-manifest manifests/qwen35_large_confirmation_feedback_v1.json \
  --failure-family-receipt \
    ../nano-harness/configs/feedback/v11_base_only_failure_families_v1.json \
  --base-dataset datasets/targeted_preservation_mix_v6.json \
  --broad-dataset datasets/failure_targeted_preservation_mix_v7.json \
  --prior-dataset datasets/format_contract_analog_v1.json \
  --prior-dataset datasets/format_contract_curriculum_analog_v2.json \
  --prior-dataset datasets/verified_semantic_arithmetic_traces_v3.json \
  --prior-dataset datasets/verified_arithmetic_process_traces_v4.json \
  --prior-dataset datasets/hard_preservation_mix_v5.json \
  --output datasets/packing_isolation_preservation_mix_v9.json
PYTHONPATH=. ../.venv/bin/python scripts/validate_release.py
```
