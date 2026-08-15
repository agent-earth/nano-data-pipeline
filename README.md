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
