# Failure-Targeted Preservation Mix v7

## Purpose

SFT v11 improves the local development gate and passes the sealed regression
canary, but its 211-case matched result is 162/211 versus base 4B at 163/211.
Four base-only discordances were reduced to an irreversible public-safe
failure-family receipt:

- percentage increase versus resulting total composition;
- packing efficiency as effective occupied volume;
- weighted recurring schedules across weeks;
- a developmental-perception choice concept.

V7 builds fresh synthetic supervision from the three numerical abstractions.
The single choice-domain abstraction is deferred to avoid narrow memorization
from one benchmark concept.

## Data-Only Intervention

Starting from `targeted-preservation-mix-v6`, v7 replaces 24 existing numeric
training slots in place:

- 8 `percentage_increase_total_composition`;
- 8 `packing_efficiency_effective_volume`;
- 8 `weighted_recurring_schedule_total`.

All 32 development rows and the other 136 training rows are byte-identical to
v6. In particular, the 16 targeted host-count rows, all 40 choice train rows,
all 40 process train rows, and their development strata are unchanged.

In-place replacement preserves the 160-row ordering and lets the next SFT
smoke freeze seed, optimizer, max steps, and all non-data fields.

## Leakage Boundary

The builder consumes only
`v11-base-only-failure-families-v1`, which contains four abstract labels,
counts, and irreversible source-set hashes. It contains no case IDs, prompts,
references, predictions, raw outputs, or reversible payloads.

- no benchmark or canary row enters training;
- no independent-holdout row is loaded or used;
- no model or teacher output enters training;
- all replacements are deterministic synthetic data;
- every numeric target is executed by the restricted AST verifier;
- the prior observed 32-row split remains a development gate only.

## Verification

- samples: 192;
- train / development: 160 / 32;
- replacement rows: 24;
- unchanged rows: 168;
- replacement overlap with analog v1-v6:
  - sample ID: 0;
  - exact message hash: 0;
  - normalized semantic hash: 0;
  - source signature: 0;
- all 192 exact and semantic hashes are unique;
- dataset SHA256:
  `b9dcbec512831a3f2c96e7db5abf4a0750420f26a28cc0f2a27699661f79aa23`.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli \
  build-failure-targeted-preservation-mix \
  --feedback-manifest manifests/qwen35_large_confirmation_feedback_v1.json \
  --failure-family-receipt \
    ../nano-harness/configs/feedback/v11_base_only_failure_families_v1.json \
  --base-dataset datasets/targeted_preservation_mix_v6.json \
  --prior-dataset datasets/format_contract_analog_v1.json \
  --prior-dataset datasets/format_contract_curriculum_analog_v2.json \
  --prior-dataset datasets/verified_semantic_arithmetic_traces_v3.json \
  --prior-dataset datasets/verified_arithmetic_process_traces_v4.json \
  --prior-dataset datasets/hard_preservation_mix_v5.json \
  --output datasets/failure_targeted_preservation_mix_v7.json
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli validate-analog \
  datasets/failure_targeted_preservation_mix_v7.json
PYTHONPATH=. ../.venv/bin/python scripts/validate_release.py
```
