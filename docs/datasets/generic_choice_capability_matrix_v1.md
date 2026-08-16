# Generic Choice Capability Matrix v1

## Purpose

This matrix is a fresh, deterministic evaluation surface for separating
choice reasoning, parser coverage, option mapping, ambiguity, and conservative
fallback. It is not training data.

## Composition

The matrix contains 48 cases across six balanced families:

- explicit two-expression average with one exact option: 8;
- explicit two-expression average with no exact option: 8;
- verbal average with no explicit arithmetic expression: 8;
- host and guest count: 8;
- sequential remaining fraction: 8;
- duplicate-option ambiguity: 8.

There are 32 scored cases and 16 ambiguity cases with no reference answer.
Expected routing is frozen at 8 verified overrides, 16 ambiguous fallbacks,
and 24 unsupported fallbacks.

## Evidence Boundary

All cases are deterministic synthetic and explicitly
`training_eligible=false`. Policy forbids use for:

- SFT or distillation;
- preference training;
- RL or RLVR;
- reward-model or verifier training;
- case-level feedback training.

No benchmark, canary, independent-holdout, model-output, or teacher-output
payload is used.

The builder compares against all committed v1-v11 datasets and requires zero
overlap in:

- case/sample identity;
- normalized user-prompt hash;
- exact message hash;
- semantic message hash;
- source signature.

## Verification

- cases: 48;
- scored / ambiguity: 32 / 16;
- training eligible: 0;
- unique prompt / exact / semantic hashes: 48 / 48 / 48;
- all prior overlap counts: 0;
- dataset SHA256:
  `5db7561b95f6b951ef7fb45293e24a39276b69b5b43e04c63712f8450e37b933`.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli \
  build-choice-capability-matrix \
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
  --output datasets/generic_choice_capability_matrix_v1.json
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli \
  validate-choice-capability-matrix \
  datasets/generic_choice_capability_matrix_v1.json
PYTHONPATH=. ../.venv/bin/python scripts/validate_release.py
```
