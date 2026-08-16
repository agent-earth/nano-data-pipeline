# Generic Choice Verifier Matrix v2

## Purpose

This fresh evaluation-only matrix tests target-blind verified execution for
host-count and verbal-average choice tasks. It expands evidence coverage
without using prior matrix rows or benchmark prompts.

## Composition

The matrix contains 48 cases across six balanced families:

- host count with one exact option: 8;
- host count with no exact option: 8;
- host count with duplicate options: 8;
- verbal average with one exact option: 8;
- verbal average with no exact option: 8;
- verbal average with duplicate options: 8.

There are 16 scored exact cases and 32 ambiguity cases. Expected routing is
frozen at 16 verified overrides and 32 conservative fallbacks.

## Evidence Boundary

All cases are deterministic synthetic and explicitly
`training_eligible=false`. The matrix forbids SFT, preference training, RL,
reward-model training, verifier training, and case-level feedback training.

No benchmark, canary, independent-holdout, model-output, or teacher-output
payload is used.

History checks cover all v1-v11 datasets plus generic choice matrix v1 and
require zero overlap in case ID, normalized user prompt, exact message,
semantic message, and source signature.

## Verification

- cases: 48;
- scored / ambiguity: 16 / 32;
- training eligible: 0;
- unique prompt / exact / semantic hashes: 48 / 48 / 48;
- all prior overlap counts: 0;
- matrix SHA256:
  `70330f730f144c1fb05d50a27e566321561451a962d0bdf736f27b8faa2f79b0`.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli \
  build-choice-verifier-matrix-v2 \
  --prior-dataset datasets/format_contract_analog_v1.json \
  --prior-dataset datasets/format_contract_curriculum_analog_v2.json \
  --prior-dataset datasets/verified_semantic_arithmetic_traces_v3.json \
  --prior-dataset datasets/verified_arithmetic_process_traces_v4.json \
  --prior-dataset datasets/hard_preservation_mix_v5.json \
  --prior-dataset datasets/targeted_preservation_mix_v6.json \
  --prior-dataset datasets/failure_targeted_preservation_mix_v7.json \
  --prior-dataset datasets/percentage_isolation_preservation_mix_v8.json \
  --prior-dataset datasets/packing_isolation_preservation_mix_v9.json \
  --prior-dataset datasets/schedule_isolation_preservation_mix_v10.json \
  --prior-dataset datasets/generic_choice_replay_v11.json \
  --prior-matrix datasets/generic_choice_capability_matrix_v1.json \
  --output datasets/generic_choice_verifier_matrix_v2.json
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli \
  validate-choice-verifier-matrix-v2 \
  datasets/generic_choice_verifier_matrix_v2.json
PYTHONPATH=. ../.venv/bin/python scripts/validate_release.py
```
