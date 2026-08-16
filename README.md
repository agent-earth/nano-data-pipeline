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
