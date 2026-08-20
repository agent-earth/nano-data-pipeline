# Orca Math SFT v1 Data Release

## Result

- Train: 32,768 rows,
  12,820,576 Qwen3.5 tokens;
- development: 1,024 rows,
  398,945 tokens;
- token quantiles:
  `{"max": 1024, "min": 68, "p25": 289, "p50": 373, "p75": 470, "p90": 598, "p95": 687, "p99": 850}`;
- split/stratum counts:
  `{"dev:long": 256, "dev:medium": 512, "dev:short": 256, "train:long": 8192, "train:medium": 16384, "train:short": 8192}`;
- dataset SHA256:
  `85a09db0743d94fc313574eaf15661556426195112da8ba59ca9abd7710fa0ff`.

The local dataset retains external teacher reasoning and appends a normalized
numeric `FINAL:` line. It is stored outside GitHub. The public release contains
only aggregate counts, identities, and gate receipts.

## Filtering

- Source rejections:
  `{"answer_char_bounds": 43, "forbidden_exact_overlap": 7447, "forbidden_near_overlap": 6771, "numeric_answer_extraction": 573, "source_exact_duplicate": 13636}`;
- selection rejections:
  `{"selected_near_duplicate": 1605, "sequence_too_long": 67}`;
- overlap counts:
  `{}`.

## Gates

- `build_preregister_identity_pass`: pass
- `exact_hash_unique`: pass
- `forbidden_exact_overlap_zero`: pass
- `forbidden_identity_pass`: pass
- `forbidden_near_overlap_zero`: pass
- `row_count_pass`: pass
- `sample_id_unique`: pass
- `schema_pass`: pass
- `selected_near_overlap_zero`: pass
- `semantic_hash_unique`: pass
- `source_identity_pass`: pass
- `source_index_unique`: pass
- `strata_count_pass`: pass
- `token_recomputation_pass`: pass
- `tokenizer_identity_pass`: pass
- `train_token_target_pass`: pass
- `training_boundary_pass`: pass
- `verifier_pass`: pass

`training_unblocked`: **true**.

## Boundary

This release proves deterministic source selection, row and token scale, provenance, split isolation, exact/near dedup, numeric suffix verification, and zero overlap with the pinned GSM8K/MMLU/GPQA corpora. It is not model-quality evidence and unlocks only one separately pre-registered SFT smoke.
