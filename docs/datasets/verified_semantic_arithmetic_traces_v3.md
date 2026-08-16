# Verified Semantic Arithmetic Traces v3

## Purpose

`verified-semantic-arithmetic-traces-v3` targets the mechanism left after SFT
v3 reaches perfect output-format compliance: multi-step arithmetic semantics.

Each assistant target contains:

```text
CALC: <expression> = <result>
FINAL: <result>
```

## Composition

- 192 deterministic synthetic samples;
- 160 train and 32 validation;
- 96 two-step and 96 three-step expressions;
- numeric targets only.

Validation membership rotates across six-sample generation blocks. The split
is independent of the observed format-only validation sets.

## Deterministic Verifier

The release pipeline parses each expression with Python AST and permits only:

- finite integer/float constants;
- parentheses;
- unary `+` and `-`;
- binary `+`, `-`, `*`, `/`.

Names, calls, attributes, indexing, powers, comprehensions, and all other AST
nodes are rejected. The verifier executes the expression, canonicalizes the
number, and requires the computed value, `CALC` result, `FINAL` result, and
stored expected result to match.

All 192 targets pass this verifier before release. A test proves tampered
`FINAL` values are rejected.

## Freshness

Against format analog v1 and curriculum analog v2:

- sample-ID overlap: 0;
- exact message hash overlap: 0;
- normalized semantic hash overlap: 0.

Within v3, all 192 exact and all 192 semantic hashes are unique.

## Leakage Boundary

- no benchmark content;
- no sealed case ID;
- no raw model or teacher output;
- deterministic synthetic source only.

The sealed feedback manifest is a requirement/provenance reference, not a
content source.

## SFT Boundary

SFT v4 may supervise the complete trace and final. It must retain FP32,
fail-fast, model/LoRA scope, seed, effective batch, LR, and 20-step cap from
SFT v3. Evaluation must report both:

- exact two-line target match;
- verifier-valid semantic correctness.

Passing synthetic validation still requires matched benchmark non-regression
before merge, scale, or RL.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli build-semantic-traces \
  --feedback-manifest manifests/qwen35_large_confirmation_feedback_v1.json \
  --prior-dataset datasets/format_contract_analog_v1.json \
  --prior-dataset datasets/format_contract_curriculum_analog_v2.json \
  --output datasets/verified_semantic_arithmetic_traces_v3.json
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli validate-analog \
  datasets/verified_semantic_arithmetic_traces_v3.json
PYTHONPATH=. ../.venv/bin/python scripts/validate_release.py
```
