# nano-data-pipeline

`nano-data-pipeline` turns benchmark and harness feedback into versioned
distillation and preference data.

## Required Properties

- immutable dataset versions and source lineage;
- exact and semantic deduplication reports;
- quality filters with rejection reasons;
- difficulty and failure-mode strata;
- leakage-aware train/eval splits;
- teacher, verifier, and reward provenance;
- feedback ingestion from failed and borderline benchmark cases;
- compact smoke datasets for pipeline and training validation.

Pipeline implementation expands only after the harness establishes a stable,
matched 4B/9B comparison. Data volume is not evidence of quality; each
filtering or generation strategy must be tested through downstream ablation.

## First Feedback Asset

`manifests/qwen35_large_confirmation_feedback_v1.json` is a public-safe,
sealed-evaluation feedback index. It separates format failures from semantic
discordances without publishing prompts, references, predictions, source
indices, or raw model outputs.

Every row is excluded from direct training. Only leak-checked analog data from
non-evaluation sources may become training eligible.

Validate the committed asset:

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_release.py
PYTHONPATH=. ../.venv/bin/python -m unittest discover -s tests -v
```

Regenerate it from the local nano-harness checkout:

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli build-feedback \
  --case-manifest ../nano-harness/configs/generated/qwen35_large_confirmation_v1_cases.json \
  --four-b-results ../nano-harness/results/harness/qwen35-large-confirmation-v1/4b/cases.jsonl \
  --nine-b-results ../nano-harness/results/harness/qwen35-large-confirmation-v1/9b/cases.jsonl \
  --public-report ../nano-harness/docs/results/large_confirmation_v1.public.json \
  --source-revision 3545480 \
  --output manifests/qwen35_large_confirmation_feedback_v1.json
```

Build the leak-free format analog dataset for SFT smoke:

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli build-format-analog \
  --feedback-manifest manifests/qwen35_large_confirmation_feedback_v1.json \
  --output datasets/format_contract_analog_v1.json
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli validate-analog \
  datasets/format_contract_analog_v1.json
```

The analog dataset is deterministic synthetic arithmetic, not benchmark
content. Its validation split checks only exact `FINAL:` contract learning.

Build the fresh two-step curriculum successor:

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli build-curriculum-analog \
  --feedback-manifest manifests/qwen35_large_confirmation_feedback_v1.json \
  --prior-dataset datasets/format_contract_analog_v1.json \
  --output datasets/format_contract_curriculum_analog_v2.json
```

Curriculum v2 has zero sample-ID, exact-hash, or semantic-hash overlap with v1
and uses an independent validation split.

Build verifier-backed semantic arithmetic traces:

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli build-semantic-traces \
  --feedback-manifest manifests/qwen35_large_confirmation_feedback_v1.json \
  --prior-dataset datasets/format_contract_analog_v1.json \
  --prior-dataset datasets/format_contract_curriculum_analog_v2.json \
  --output datasets/verified_semantic_arithmetic_traces_v3.json
```

Every trace is executed by a restricted AST arithmetic verifier before release.

Build fresh verifier-backed intermediate process traces:

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli build-process-traces \
  --feedback-manifest manifests/qwen35_large_confirmation_feedback_v1.json \
  --prior-dataset datasets/format_contract_analog_v1.json \
  --prior-dataset datasets/format_contract_curriculum_analog_v2.json \
  --prior-dataset datasets/verified_semantic_arithmetic_traces_v3.json \
  --output datasets/verified_arithmetic_process_traces_v4.json
```

Each target executes two or three individually verified `STEP` lines. Every
later step consumes the preceding result, and the last step, source expression,
and `FINAL` must agree.

Build the hard capability-preservation mix:

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli build-preservation-mix \
  --feedback-manifest manifests/qwen35_large_confirmation_feedback_v1.json \
  --prior-dataset datasets/format_contract_analog_v1.json \
  --prior-dataset datasets/format_contract_curriculum_analog_v2.json \
  --prior-dataset datasets/verified_semantic_arithmetic_traces_v3.json \
  --prior-dataset datasets/verified_arithmetic_process_traces_v4.json \
  --output datasets/hard_preservation_mix_v5.json
```

The mix combines hard numeric boundary problems, answer-only choice contract
examples, and a smaller process-trace stratum. It contains no benchmark or
sealed canary content.

Build the targeted numeric covariate repair:

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
```

V6 replaces only the 16 host-count training slots with fresh companion-count
2 examples. The observed 32-row development split and every other v5 row stay
byte-identical. Because development failures informed this intervention, that
split is a development gate only, not independent quality evidence.

Build fresh data from the irreversible v11 failure-family receipt:

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
```

V7 replaces 24 numeric training slots with three fresh verifier-backed
families while preserving all development, targeted-host, choice, and process
strata. It consumes no benchmark/canary payload and does not read the frozen
independent holdout.

Build the conservative percentage-family isolation:

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
```

V8 changes only eight numeric train slots and keeps the remaining 184 rows
byte-identical to v6. It isolates percentage-total composition while deferring
the other failure families and the independent holdout.

Packing-efficiency isolation is available through
`build-packing-isolation-preservation-mix`. It applies the same eight-row,
train-only contract to the packing family while restoring the percentage and
schedule rows to v6.

Build the generic choice-preservation replay:

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli \
  build-choice-replay \
  --base-dataset datasets/targeted_preservation_mix_v6.json \
  --output datasets/generic_choice_replay_v11.json
```

V11 selects the 40 deterministic synthetic choice train rows from v6 and
retains the unchanged 32-row development split for local gating. Training
must select `split=train`; no benchmark, model-output, canary, or independent
holdout payload is used.

Build the fresh evaluation-only choice capability matrix:

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
```

The 48-row matrix is history-disjoint and explicitly ineligible for training,
preference optimization, RL, reward-model training, or verifier training.

Build the fresh host-count and verbal-average verifier matrix:

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
```

Matrix v2 has 16 scored exact cases and 32 ambiguity cases. It is
evaluation-only and carries the same hard prohibition on every training use.

Build the larger exact-only replication matrix:

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli \
  build-choice-exact-replication-matrix-v3 \
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
  --prior-matrix datasets/generic_choice_verifier_matrix_v2.json \
  --output datasets/generic_choice_exact_replication_matrix_v3.json
```

Matrix v3 has 32 scored exact cases split evenly between host-count and
verbal-average. It is history-disjoint, evaluation-only, and forbidden for
SFT, preference optimization, RL, reward-model training, verifier training,
or case-level feedback training.
