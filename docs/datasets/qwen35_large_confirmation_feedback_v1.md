# Qwen3.5 Large Confirmation Feedback v1

## Purpose

This public-safe manifest indexes failure signals from the sealed 512-case
Qwen3.5-4B/9B confirmation. It supports failure accounting, analog-data
generation, verifier design, and SFT-smoke planning.

It is not training data. Benchmark prompts, references, predictions, raw model
outputs, and source row indices are intentionally absent.

## Source Identity

- Experiment: `qwen35-large-confirmation-v1`
- Harness source revision: `47ee48f`
- Committed case manifest SHA256:
  `7742995c2b006228f6c6f60e937cd3d54d6fb743fdc84be0d101695f028af5cc`
- 4B raw result SHA256:
  `977a99e978936fbcfc99d45d859a8324746215b3ba518ed3bc5c4a2b5990b33e`
- 9B raw result SHA256:
  `07b442f620245379444f02d633c76502032df84dfe5ad0b74d88996d3c0103bd`

The generator verifies these hashes against the harness public report before
emitting a manifest.

## Selection And Taxonomy

A case is included when:

- official paired correctness is discordant; or
- either arm has an official format failure.

Cases where both models are parseable and have the same correctness state are
excluded. Rows are classified as:

- `format`: either arm is unparseable, including length truncation or a strict
  output-contract failure;
- `semantic_discordance`: both arms are parseable and exactly one is correct.

The v1 manifest has:

- 96 unique rows;
- 18 GSM8K and 78 MMLU rows;
- 66 format and 30 semantic-discordance rows;
- 35 4B-only wins, 23 9B-only wins, and 38 both-wrong format cases.

## Deduplication

Exact deduplication uses stable benchmark case ID. Duplicate IDs are rejected.

Semantic deduplication is intentionally not run on this public view because it
contains no question or answer content. A future analog-data builder must run
semantic deduplication on its non-evaluation source material before any train
split is marked eligible.

## Split And Leakage Policy

Every row is:

- `source_split: sealed_eval_feedback`;
- `training_eligible: false`.

Direct SFT, DPO, RL, reward-model, or verifier training on these rows is
forbidden. The only allowed training path is:

1. use taxonomy and aggregate statistics to define a generation requirement;
2. generate or retrieve analog examples from non-evaluation sources;
3. preserve source lineage;
4. deduplicate against all sealed case IDs and source content;
5. pass quality filters and split validation;
6. mark only the derived `non_eval_analog_only` artifact training eligible.

## Reproduction

From the repository root:

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli validate-feedback \
  manifests/qwen35_large_confirmation_feedback_v1.json
PYTHONPATH=. ../.venv/bin/python scripts/validate_release.py
PYTHONPATH=. ../.venv/bin/python -m unittest discover -s tests -v
```
