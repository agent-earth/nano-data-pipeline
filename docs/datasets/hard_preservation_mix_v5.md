# Hard Preservation Mix v5

## Purpose

Process SFT v6 reaches 32/32 synthetic process compliance but regresses the
frozen three-task suite from base 4B 163/211 to 145/211. V5 replaces the
process-only objective with a mixed non-evaluation curriculum that targets the
observed failure families without copying sealed cases.

## Composition

- 192 deterministic synthetic samples;
- 160 train and 32 validation;
- 96 hard numeric word problems;
- 48 answer-only choice problems;
- 48 verified arithmetic process traces.

The train split contains 80 numeric, 40 choice, and 40 process samples. The
validation split contains 16 numeric, 8 choice, and 8 process samples.

## Preservation Families

Hard numeric and choice generators cover four abstract boundaries:

- include the host or primary entity in multi-group totals;
- distinguish a category total from a percentage increase;
- average participant totals rather than individual objects;
- apply fractions sequentially to the remaining quantity.

The templates, wording, entities, and numeric domains are synthetic and were
not derived from benchmark prompts or raw model outputs.

Numeric targets use:

```text
WORK: <restricted arithmetic expression> = <result>
FINAL: <result>
```

The validator executes `WORK` with the restricted AST evaluator and requires
the computed value, rendered work result, standalone `FINAL`, and stored
expected result to agree.

Choice targets contain only a standalone `FINAL: <letter>` line. Process
targets retain the fully verified two- or three-step contract from v4.

## Freshness

Against analog v1, curriculum v2, semantic traces v3, and process traces v4:

- sample-ID overlap: 0;
- exact message hash overlap: 0;
- normalized semantic hash overlap: 0;
- source-signature/expression overlap: 0.

Within v5, all 192 exact and semantic hashes are unique.

## Leakage Boundary

- no benchmark content;
- no sealed case ID;
- no benchmark or canary output;
- no teacher output;
- `sealed_canary_used_for_training: false`;
- deterministic synthetic source only.

The 40-case regression canary remains sealed evaluation data and is never
read by this builder.

## Token Contract

With the frozen Qwen3.5-4B tokenizer:

- target content is at most 75 tokens;
- target plus EOS is at most 76 tokens;
- full prompt plus target is at most 191 tokens.

An SFT smoke may retain `max_length=192` and
`generation_max_new_tokens=80`, but must machine-check both bounds.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli build-preservation-mix \
  --feedback-manifest manifests/qwen35_large_confirmation_feedback_v1.json \
  --prior-dataset datasets/format_contract_analog_v1.json \
  --prior-dataset datasets/format_contract_curriculum_analog_v2.json \
  --prior-dataset datasets/verified_semantic_arithmetic_traces_v3.json \
  --prior-dataset datasets/verified_arithmetic_process_traces_v4.json \
  --output datasets/hard_preservation_mix_v5.json
PYTHONPATH=. ../.venv/bin/python -m nano_data_pipeline.cli validate-analog \
  datasets/hard_preservation_mix_v5.json
PYTHONPATH=. ../.venv/bin/python scripts/validate_release.py
```
