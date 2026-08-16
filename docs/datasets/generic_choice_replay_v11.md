# Generic Choice Replay v11

## Purpose

Anchored-v1 is the first candidate to improve the 4B aggregate over the base
model, but it remains one MMLU item below base on the frozen development suite.
V11 provides a bounded, generic answer-choice preservation intervention before
the independent holdout is opened.

## Composition

The builder selects only rows already present in
`targeted-preservation-mix-v6`:

- 40 deterministic synthetic choice rows from the train split;
- 32 unchanged rows from the development split for local gating only.

The train rows cover three generic arithmetic choice rules:

- `preservation_host_count_choice_v5`: 16;
- `preservation_sequential_fraction_choice_v5`: 16;
- `preservation_participant_average_choice_v5`: 8.

All 40 training targets use the standalone `FINAL: <letter>` contract. A
trainer must select `split=train`; the 32 development rows must never be
optimized.

## Evidence Boundary

This dataset contains no benchmark prompt, reference, source index, model
output, teacher output, sealed canary row, or independent-holdout payload. It
does not use observed benchmark failure identities or feedback. The retained
development split has already been observed and can provide only a local
regression gate, not independent quality evidence.

## Verification

- samples: 72;
- train / development: 40 / 32;
- train choice / numeric / process: 40 / 0 / 0;
- development choice / numeric / process: 8 / 16 / 8;
- unique exact / semantic hashes: 72 / 72;
- base dataset SHA256:
  `ab51a1be5f45d7f71796fbf98ef6cce83ff9cb0f0a756fed01cb1e7aea55651d`;
- dataset SHA256:
  `4657e96af9f9d1b81bfdb5fac6a29c31baf24b23d7753508aafdea5603ffd80d`.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli \
  build-choice-replay \
  --base-dataset datasets/targeted_preservation_mix_v6.json \
  --output datasets/generic_choice_replay_v11.json
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli validate-analog \
  datasets/generic_choice_replay_v11.json
PYTHONPATH=. ../.venv/bin/python scripts/validate_release.py
```
