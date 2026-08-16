# Targeted Preservation Mix v6

## Purpose

Hard-preservation SFT v9 and v10 both reach numeric 9/16. V10 exactly
reproduces every v9 validation metric and failure ID, so additional max-step
interpolation is exhausted.

The seven persistent numeric failures all belong to the v5
`host_and_companion_count` family. A machine audit finds a covariate support
gap:

- all 16 host-count train rows use 3 or 4 companions;
- all 8 host-count development rows use 2 companions;
- 7/8 of those development rows fail in v10.

V6 repairs that support gap without copying any development prompt, target, or
model output.

## Single Data Intervention

Starting from `hard-preservation-mix-v5`, v6 replaces exactly the 16
host-count training rows in place with fresh deterministic examples whose
participant companion count is 2.

Everything else is byte-identical to v5:

- all 32 development rows;
- all 40 choice train rows and 8 choice development rows;
- all 40 process train rows and 8 process development rows;
- the other 64 numeric train rows and 8 average-total development rows;
- sample order, split sizes, task-family counts, and training eligibility.

In-place replacement preserves the deterministic training slot order. A
future data-only SFT ablation can therefore freeze seed, model, LoRA,
optimizer, max steps, and family gates.

## Evidence Boundary

The builder reads only the public-safe v10 report identity and numeric failure
sample IDs. It verifies that all seven IDs resolve to host-count development
rows and audits the multiplier support from verifier expressions.

No development prompt, target, raw generation, benchmark content, canary
content, or teacher output is copied into a replacement. The development split
has already informed this intervention and is therefore labeled
`development_gate_only`; it cannot provide independent quality evidence.

The sealed 40-case canary remains unread and excluded from training.

## Verification

- samples: 192;
- train / development: 160 / 32;
- replacement rows: 16;
- replacement overlap with analog v1-v5:
  - sample ID: 0;
  - exact message hash: 0;
  - normalized semantic hash: 0;
  - source signature: 0;
- all 192 exact and semantic hashes are unique;
- all numeric targets execute under the restricted AST verifier;
- dataset SHA256:
  `ab51a1be5f45d7f71796fbf98ef6cce83ff9cb0f0a756fed01cb1e7aea55651d`.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli \
  build-targeted-preservation-mix \
  --feedback-manifest manifests/qwen35_large_confirmation_feedback_v1.json \
  --base-dataset datasets/hard_preservation_mix_v5.json \
  --development-report \
    ../nano-train/docs/results/hard_preservation_sft_smoke_v10.public.json \
  --prior-dataset datasets/format_contract_analog_v1.json \
  --prior-dataset datasets/format_contract_curriculum_analog_v2.json \
  --prior-dataset datasets/verified_semantic_arithmetic_traces_v3.json \
  --prior-dataset datasets/verified_arithmetic_process_traces_v4.json \
  --output datasets/targeted_preservation_mix_v6.json
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli validate-analog \
  datasets/targeted_preservation_mix_v6.json
PYTHONPATH=. ../.venv/bin/python scripts/validate_release.py
```
