# Percentage Isolation Preservation Mix v8

## Purpose

The broad v7 data intervention replaces 24 numeric training rows across three
families. SFT v12 receives 19 of them in its 32-step schedule but gains no
numeric semantic cases and regresses strict exact and choice behavior.

V8 isolates one family so the next smoke can test a narrower mechanism:
percentage increase versus resulting total composition.

## Single-Family Intervention

Starting from `targeted-preservation-mix-v6`, v8 changes exactly 8 numeric
training rows. The replacement rows are byte-identical to the verified
percentage-family subset already present in broad v7.

Everything else is byte-identical to v6:

- all 32 development rows;
- all 16 targeted host-count rows;
- all choice and process rows;
- all average, sequential-fraction, packing, and schedule strata;
- sample order, split/family counts, and training eligibility.

The frozen 32-step schedule exposes 7 of the 8 replacement rows. Packing
efficiency, recurring schedule, and the choice-domain abstraction remain
deferred.

## Leakage Boundary

V8 inherits only fresh deterministic synthetic rows from v7. Its source
failure-family receipt contains abstract labels and irreversible hashes, not
case IDs, prompts, references, predictions, outputs, or reversible payloads.

- no benchmark or canary row enters training;
- no independent-holdout row is loaded or used;
- no model/teacher output enters training;
- all replacement targets are restricted-AST verified;
- the unchanged 32-row split remains development evidence only.

## Verification

- samples: 192;
- train / development: 160 / 32;
- replacement rows: 8;
- unchanged rows: 184;
- replacement overlap with analog v1-v6:
  - sample ID: 0;
  - exact message hash: 0;
  - normalized semantic hash: 0;
  - source signature: 0;
- all 192 exact and semantic hashes are unique;
- dataset SHA256:
  `0ae81bb4c385703592946b5c75971b39cbb388b02a76fafa477e53e55756bc9c`.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli \
  build-percentage-isolation-preservation-mix \
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
  --output datasets/percentage_isolation_preservation_mix_v8.json
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli validate-analog \
  datasets/percentage_isolation_preservation_mix_v8.json
PYTHONPATH=. ../.venv/bin/python scripts/validate_release.py
```
