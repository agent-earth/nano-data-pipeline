# Verified Arithmetic Process Traces v4

## Purpose

The 48-token audit of semantic SFT v5 produces complete, format-valid traces
but leaves 18/32 arithmetic execution failures. V4 targets that mechanism by
supervising each operation separately instead of copying the full expression
into one `CALC` line.

Targets use:

```text
STEP 1: <expression> = <result>
STEP 2: <expression> = <result>
[STEP 3: <expression> = <result>]
FINAL: <result>
```

## Composition

- 192 deterministic synthetic samples;
- 160 train and 32 validation;
- 96 two-step and 96 three-step expressions;
- numeric targets only;
- 192 unique exact and normalized semantic hashes.

The validation split is deterministic and independent of v3 membership.

With the frozen Qwen3.5-4B tokenizer:

- target content is 42-71 tokens;
- target plus EOS is at most 72 tokens;
- full prompt plus target is 140-187 tokens.

Process SFT must therefore use at least 192 sequence tokens and a generation
budget greater than 72. These conditions must be machine-checked before
training.

## Process Verifier

Every sample stores:

- the original source expression;
- an ordered list of two or three step expressions;
- the verified expected result of every step;
- the final expected result.

The release validator:

1. requires contiguous `STEP 1`, `STEP 2`, and optional `STEP 3` lines;
2. executes every step with the restricted arithmetic AST evaluator;
3. requires rendered expressions and results to match stored verifier fields;
4. requires each later step to consume the preceding step result;
5. executes the source expression independently;
6. requires the last step, source expression, and `FINAL` to agree.

Calls, names, attributes, indexing, powers, comprehensions, and unsafe AST
nodes remain forbidden. A test proves a tampered intermediate result is
rejected.

## Freshness

Against format analog v1, curriculum analog v2, and semantic traces v3:

- sample-ID overlap: 0;
- exact message hash overlap: 0;
- normalized semantic hash overlap: 0;
- source-expression overlap: 0.

## Leakage Boundary

- no benchmark content;
- no sealed case ID;
- no raw model output;
- no teacher output;
- deterministic synthetic source only.

The sealed feedback manifest supplies only requirement lineage. Its cases are
not copied or transformed into training rows.

## SFT Boundary

Before process SFT:

- preflight must prove the generation budget exceeds maximum target content
  plus EOS;
- validation must parse and execute every generated process step;
- official v5 remains unchanged;
- benchmark, merge, scale-up, and RL remain forbidden until a separately
  pre-registered SFT smoke passes its own gates.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli build-process-traces \
  --feedback-manifest manifests/qwen35_large_confirmation_feedback_v1.json \
  --prior-dataset datasets/format_contract_analog_v1.json \
  --prior-dataset datasets/format_contract_curriculum_analog_v2.json \
  --prior-dataset datasets/verified_semantic_arithmetic_traces_v3.json \
  --output datasets/verified_arithmetic_process_traces_v4.json
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli validate-analog \
  datasets/verified_arithmetic_process_traces_v4.json
PYTHONPATH=. ../.venv/bin/python scripts/validate_release.py
```
