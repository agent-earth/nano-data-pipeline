from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import re
from collections import Counter
from pathlib import Path
from typing import Any

from nano_data_pipeline.feedback import (
    sha256_file,
    validate_feedback_manifest,
)


SCHEMA_VERSION = "nano_analog_dataset_v1"
SYSTEM_PROMPT = (
    "Follow the requested answer contract exactly. Return one FINAL line and "
    "no additional text."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalized_text(messages: list[dict[str, str]]) -> str:
    normalized = []
    for message in messages:
        content = re.sub(r"\s+", " ", message["content"]).strip().lower()
        normalized.append(f"{message['role']}:{content}")
    return "\n".join(normalized)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


ARITHMETIC_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def evaluate_arithmetic(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if (
            isinstance(node, ast.Constant)
            and type(node.value) in {int, float}
        ):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ARITHMETIC_OPERATORS:
            return ARITHMETIC_OPERATORS[type(node.op)](
                evaluate(node.left),
                evaluate(node.right),
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in ARITHMETIC_OPERATORS:
            return ARITHMETIC_OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError(f"unsafe arithmetic node: {type(node).__name__}")

    result = evaluate(tree)
    if not isinstance(result, (int, float)) or not math.isfinite(float(result)):
        raise ValueError("arithmetic result is not finite")
    return result


def format_number(value: int | float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def _choice_sample(index: int) -> dict[str, Any]:
    left = 7 + index
    right = 2 + (index * 5 % 17)
    if index % 2:
        expression = f"{left} + {right}"
        result = left + right
        operation = "addition"
    else:
        expression = f"{left} - {right}"
        result = left - right
        operation = "subtraction"
    distractors = [result - 2, result - 1, result + 1, result + 2]
    correct_index = index % 4
    options = distractors[:]
    options[correct_index] = result
    letters = "ABCD"
    user = (
        f"Compute {expression}.\n"
        + "\n".join(
            f"{letter}. {value}" for letter, value in zip(letters, options)
        )
        + "\nReturn exactly one line in the form FINAL: <letter>."
    )
    return {
        "task_family": "choice_contract",
        "format_family": "final_choice",
        "difficulty": "single_step",
        "generation_rule": f"arithmetic_{operation}_choice_v1",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": f"FINAL: {letters[correct_index]}",
            },
        ],
    }


def _numeric_sample(index: int) -> dict[str, Any]:
    left = 11 + index
    middle = 2 + (index * 7 % 13)
    right = 1 + (index * 3 % 5)
    mode = index % 4
    if mode == 0:
        expression = f"{left} + {middle}"
        result = left + middle
        difficulty = "single_step"
        rule = "addition"
    elif mode == 1:
        expression = f"{left} * {middle}"
        result = left * middle
        difficulty = "single_step"
        rule = "multiplication"
    elif mode == 2:
        expression = f"{left} + {middle} * {right}"
        result = left + middle * right
        difficulty = "two_step"
        rule = "precedence"
    else:
        expression = f"({left} + {middle}) * {right}"
        result = (left + middle) * right
        difficulty = "two_step"
        rule = "parenthesized"
    user = (
        f"Compute {expression}. Return exactly one line in the form "
        "FINAL: <number>."
    )
    return {
        "task_family": "numeric_contract",
        "format_family": "final_numeric",
        "difficulty": difficulty,
        "generation_rule": f"arithmetic_{rule}_numeric_v1",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": f"FINAL: {result}"},
        ],
    }


def build_format_analog_dataset(feedback_manifest_path: Path) -> dict[str, Any]:
    feedback = json.loads(feedback_manifest_path.read_text(encoding="utf-8"))
    validate_feedback_manifest(feedback)
    samples = []
    for family, builder in (
        ("choice_contract", _choice_sample),
        ("numeric_contract", _numeric_sample),
    ):
        for index in range(64):
            sample = builder(index)
            if sample["task_family"] != family:
                raise ValueError("analog builder returned the wrong task family")
            identity = {
                "task_family": sample["task_family"],
                "generation_rule": sample["generation_rule"],
                "messages": sample["messages"],
            }
            sample["sample_id"] = f"synthetic-{_hash(_canonical_json(identity))[:20]}"
            sample["split"] = "validation" if index % 5 == 0 else "train"
            sample["source_kind"] = "deterministic_synthetic"
            sample["training_eligible"] = True
            sample["exact_sha256"] = _hash(_canonical_json(sample["messages"]))
            sample["semantic_sha256"] = _hash(
                _normalized_text(sample["messages"])
            )
            samples.append(sample)
    samples.sort(key=lambda sample: sample["sample_id"])

    dataset = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "format-contract-analog-v1",
        "version": "v1",
        "source": {
            "source_kind": "deterministic_synthetic",
            "generator": "nano_data_pipeline.analog",
            "feedback_requirement_id": feedback["dataset_id"],
            "feedback_manifest_sha256": sha256_file(feedback_manifest_path),
            "benchmark_content_used": False,
            "sealed_case_ids_used": False,
        },
        "policy": {
            "source_split": "non_eval_analog_only",
            "training_allowed": True,
            "contains_benchmark_content": False,
            "contains_model_outputs": False,
            "contains_teacher_outputs": False,
            "purpose": "format_contract_sft_smoke",
        },
        "samples": samples,
    }
    rendered_samples = _canonical_json(samples)
    leaked_case_ids = [
        row["case_id"]
        for row in feedback["rows"]
        if str(row["case_id"]) in rendered_samples
    ]
    if leaked_case_ids:
        raise ValueError(f"sealed case IDs leaked into analog data: {leaked_case_ids[:5]}")
    dataset["summary"] = summarize_analog_dataset(dataset)
    validate_analog_dataset(dataset)
    return dataset


def _curriculum_choice_sample(index: int) -> dict[str, Any]:
    left = 211 + index * 3
    middle = 5 + (index * 11 % 23)
    right = 2 + (index * 7 % 9)
    mode = index % 5
    if mode == 0:
        expression = f"{left} + {middle}"
        result = left + middle
        difficulty = "single_step"
        rule = "addition"
    elif mode in {1, 2}:
        expression = f"{left} + {middle} * {right}"
        result = left + middle * right
        difficulty = "two_step"
        rule = "precedence"
    else:
        expression = f"({left} + {middle}) * {right}"
        result = (left + middle) * right
        difficulty = "two_step"
        rule = "parenthesized"
    correct_index = (index * 3 + 1) % 4
    offsets = [-3, -1, 2, 4]
    options = [result + offset for offset in offsets]
    options[correct_index] = result
    letters = "ABCD"
    user = (
        f"Compute {expression}.\n"
        + "\n".join(
            f"{letter}. {value}" for letter, value in zip(letters, options)
        )
        + "\nReturn exactly one line in the form FINAL: <letter>."
    )
    return {
        "task_family": "choice_contract",
        "format_family": "final_choice",
        "difficulty": difficulty,
        "generation_rule": f"curriculum_{rule}_choice_v2",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": f"FINAL: {letters[correct_index]}",
            },
        ],
    }


def _curriculum_numeric_sample(index: int) -> dict[str, Any]:
    left = 307 + index * 4
    middle = 7 + (index * 13 % 29)
    right = 2 + (index * 5 % 11)
    mode = index % 5
    if mode == 0:
        expression = f"{left} - {middle}"
        result = left - middle
        difficulty = "single_step"
        rule = "subtraction"
    elif mode in {1, 2}:
        expression = f"{left} + {middle} * {right}"
        result = left + middle * right
        difficulty = "two_step"
        rule = "precedence"
    else:
        expression = f"({left} - {middle}) * {right}"
        result = (left - middle) * right
        difficulty = "two_step"
        rule = "parenthesized"
    user = (
        f"Compute {expression}. Return exactly one line in the form "
        "FINAL: <number>."
    )
    return {
        "task_family": "numeric_contract",
        "format_family": "final_numeric",
        "difficulty": difficulty,
        "generation_rule": f"curriculum_{rule}_numeric_v2",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": f"FINAL: {result}"},
        ],
    }


def build_curriculum_analog_dataset(
    feedback_manifest_path: Path,
    prior_dataset_path: Path,
) -> dict[str, Any]:
    feedback = json.loads(feedback_manifest_path.read_text(encoding="utf-8"))
    validate_feedback_manifest(feedback)
    prior = json.loads(prior_dataset_path.read_text(encoding="utf-8"))
    validate_analog_dataset(prior)
    prior_exact = {sample["exact_sha256"] for sample in prior["samples"]}
    prior_semantic = {sample["semantic_sha256"] for sample in prior["samples"]}
    prior_ids = {sample["sample_id"] for sample in prior["samples"]}

    samples = []
    for family, builder in (
        ("choice_contract", _curriculum_choice_sample),
        ("numeric_contract", _curriculum_numeric_sample),
    ):
        for index in range(80):
            sample = builder(index)
            if sample["task_family"] != family:
                raise ValueError("curriculum builder returned the wrong family")
            identity = {
                "dataset_version": "v2",
                "task_family": sample["task_family"],
                "generation_rule": sample["generation_rule"],
                "messages": sample["messages"],
            }
            sample["sample_id"] = f"synthetic-{_hash(_canonical_json(identity))[:20]}"
            validation_offset = (index // 5) % 5
            sample["split"] = (
                "validation" if index % 5 == validation_offset else "train"
            )
            sample["source_kind"] = "deterministic_synthetic"
            sample["training_eligible"] = True
            sample["exact_sha256"] = _hash(_canonical_json(sample["messages"]))
            sample["semantic_sha256"] = _hash(_normalized_text(sample["messages"]))
            samples.append(sample)
    samples.sort(key=lambda sample: sample["sample_id"])

    exact_overlap = sorted(
        sample["exact_sha256"]
        for sample in samples
        if sample["exact_sha256"] in prior_exact
    )
    semantic_overlap = sorted(
        sample["semantic_sha256"]
        for sample in samples
        if sample["semantic_sha256"] in prior_semantic
    )
    id_overlap = sorted(
        sample["sample_id"] for sample in samples if sample["sample_id"] in prior_ids
    )
    if exact_overlap or semantic_overlap or id_overlap:
        raise ValueError("curriculum analog overlaps prior analog dataset")

    rendered_samples = _canonical_json(samples)
    leaked_case_ids = [
        row["case_id"]
        for row in feedback["rows"]
        if str(row["case_id"]) in rendered_samples
    ]
    if leaked_case_ids:
        raise ValueError(f"sealed case IDs leaked into analog data: {leaked_case_ids[:5]}")

    dataset = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "format-contract-curriculum-analog-v2",
        "version": "v2",
        "source": {
            "source_kind": "deterministic_synthetic",
            "generator": "nano_data_pipeline.analog",
            "feedback_requirement_id": feedback["dataset_id"],
            "feedback_manifest_sha256": sha256_file(feedback_manifest_path),
            "prior_dataset_id": prior["dataset_id"],
            "prior_dataset_sha256": sha256_file(prior_dataset_path),
            "benchmark_content_used": False,
            "sealed_case_ids_used": False,
            "prior_exact_overlap": 0,
            "prior_semantic_overlap": 0,
            "prior_sample_id_overlap": 0,
        },
        "policy": {
            "source_split": "non_eval_analog_only",
            "training_allowed": True,
            "contains_benchmark_content": False,
            "contains_model_outputs": False,
            "contains_teacher_outputs": False,
            "purpose": "two_step_curriculum_sft_smoke",
            "observed_validation_reused": False,
        },
        "samples": samples,
    }
    dataset["summary"] = summarize_analog_dataset(dataset)
    validate_analog_dataset(dataset)
    return dataset


def _semantic_trace_sample(index: int) -> dict[str, Any]:
    left = 701 + index * 5
    middle = 11 + (index * 17 % 37)
    right = 2 + (index * 7 % 13)
    extra = 3 + (index * 11 % 17)
    mode = index % 4
    if mode == 0:
        expression = f"{left} + {middle} * {right}"
        rule = "precedence_add_multiply"
        difficulty = "two_step"
    elif mode == 1:
        expression = f"({left} - {middle}) * {right}"
        rule = "parenthesized_subtract_multiply"
        difficulty = "two_step"
    elif mode == 2:
        expression = f"{left} + {middle} * {right} - {extra}"
        rule = "precedence_add_multiply_subtract"
        difficulty = "three_step"
    else:
        expression = f"({left} + {middle}) * {right} - {extra}"
        rule = "parenthesized_add_multiply_subtract"
        difficulty = "three_step"
    result = format_number(evaluate_arithmetic(expression))
    user = (
        f"Compute {expression}. Show one executable calculation line, then the "
        "numeric final. Use exactly:\nCALC: <expression> = <result>\n"
        "FINAL: <number>"
    )
    return {
        "task_family": "semantic_arithmetic",
        "format_family": "trace_numeric",
        "difficulty": difficulty,
        "generation_rule": f"verified_{rule}_v3",
        "verifier": {
            "kind": "safe_ast_arithmetic_v1",
            "expression": expression,
            "expected_result": result,
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "Follow the arithmetic trace contract exactly. The CALC "
                    "expression and FINAL value must agree."
                ),
            },
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": f"CALC: {expression} = {result}\nFINAL: {result}",
            },
        ],
    }


def _process_trace_sample(index: int) -> dict[str, Any]:
    left = 2003 + index * 7
    middle = 13 + (index * 19 % 41)
    right = 2 + (index * 11 % 13)
    extra = 5 + (index * 13 % 19)
    mode = index % 4
    if mode == 0:
        expression = f"{left} + {middle} * {right}"
        first = middle * right
        final = left + first
        steps = [
            (f"{middle} * {right}", first),
            (f"{left} + {first}", final),
        ]
        rule = "precedence_add_multiply"
        difficulty = "two_step"
    elif mode == 1:
        expression = f"({left} - {middle}) * {right}"
        first = left - middle
        final = first * right
        steps = [
            (f"{left} - {middle}", first),
            (f"{first} * {right}", final),
        ]
        rule = "parenthesized_subtract_multiply"
        difficulty = "two_step"
    elif mode == 2:
        expression = f"{left} + {middle} * {right} - {extra}"
        first = middle * right
        second = left + first
        final = second - extra
        steps = [
            (f"{middle} * {right}", first),
            (f"{left} + {first}", second),
            (f"{second} - {extra}", final),
        ]
        rule = "precedence_add_multiply_subtract"
        difficulty = "three_step"
    else:
        expression = f"({left} + {middle}) * {right} - {extra}"
        first = left + middle
        second = first * right
        final = second - extra
        steps = [
            (f"{left} + {middle}", first),
            (f"{first} * {right}", second),
            (f"{second} - {extra}", final),
        ]
        rule = "parenthesized_add_multiply_subtract"
        difficulty = "three_step"
    expected_result = format_number(final)
    verifier_steps = [
        {
            "expression": step_expression,
            "expected_result": format_number(step_result),
        }
        for step_expression, step_result in steps
    ]
    rendered_steps = "\n".join(
        f"STEP {number}: {step['expression']} = {step['expected_result']}"
        for number, step in enumerate(verifier_steps, start=1)
    )
    user = (
        f"Compute {expression}. Execute one operation per line in evaluation "
        f"order using exactly {len(steps)} STEP lines, then FINAL. Use:\n"
        "STEP 1: <expression> = <result>\n"
        "STEP 2: <expression> = <result>\n"
    )
    if len(steps) == 3:
        user += "STEP 3: <expression> = <result>\n"
    user += "FINAL: <number>"
    return {
        "task_family": "semantic_arithmetic_process",
        "format_family": "process_trace_numeric",
        "difficulty": difficulty,
        "generation_rule": f"verified_process_{rule}_v4",
        "verifier": {
            "kind": "safe_ast_arithmetic_process_v2",
            "source_expression": expression,
            "steps": verifier_steps,
            "expected_result": expected_result,
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "Follow the arithmetic process contract exactly. Execute "
                    "and verify every STEP before returning FINAL."
                ),
            },
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": f"{rendered_steps}\nFINAL: {expected_result}",
            },
        ],
    }


def build_semantic_trace_dataset(
    feedback_manifest_path: Path,
    prior_dataset_paths: list[Path],
) -> dict[str, Any]:
    feedback = json.loads(feedback_manifest_path.read_text(encoding="utf-8"))
    validate_feedback_manifest(feedback)
    priors = []
    prior_ids: set[str] = set()
    prior_exact: set[str] = set()
    prior_semantic: set[str] = set()
    for path in prior_dataset_paths:
        prior = json.loads(path.read_text(encoding="utf-8"))
        validate_analog_dataset(prior)
        priors.append(
            {
                "dataset_id": prior["dataset_id"],
                "sha256": sha256_file(path),
            }
        )
        prior_ids.update(sample["sample_id"] for sample in prior["samples"])
        prior_exact.update(sample["exact_sha256"] for sample in prior["samples"])
        prior_semantic.update(
            sample["semantic_sha256"] for sample in prior["samples"]
        )

    samples = []
    for index in range(192):
        sample = _semantic_trace_sample(index)
        identity = {
            "dataset_version": "v3",
            "generation_rule": sample["generation_rule"],
            "messages": sample["messages"],
        }
        sample["sample_id"] = f"synthetic-{_hash(_canonical_json(identity))[:20]}"
        validation_offset = (index // 6) % 6
        sample["split"] = (
            "validation" if index % 6 == validation_offset else "train"
        )
        sample["source_kind"] = "deterministic_synthetic"
        sample["training_eligible"] = True
        sample["exact_sha256"] = _hash(_canonical_json(sample["messages"]))
        sample["semantic_sha256"] = _hash(_normalized_text(sample["messages"]))
        samples.append(sample)
    samples.sort(key=lambda sample: sample["sample_id"])

    overlaps = {
        "prior_sample_id_overlap": sum(
            sample["sample_id"] in prior_ids for sample in samples
        ),
        "prior_exact_overlap": sum(
            sample["exact_sha256"] in prior_exact for sample in samples
        ),
        "prior_semantic_overlap": sum(
            sample["semantic_sha256"] in prior_semantic for sample in samples
        ),
    }
    if any(overlaps.values()):
        raise ValueError(f"semantic trace data overlaps prior analogs: {overlaps}")
    rendered_samples = _canonical_json(samples)
    leaked_case_ids = [
        row["case_id"]
        for row in feedback["rows"]
        if str(row["case_id"]) in rendered_samples
    ]
    if leaked_case_ids:
        raise ValueError(f"sealed case IDs leaked into analog data: {leaked_case_ids[:5]}")

    dataset = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "verified-semantic-arithmetic-traces-v3",
        "version": "v3",
        "source": {
            "source_kind": "deterministic_synthetic",
            "generator": "nano_data_pipeline.analog",
            "feedback_requirement_id": feedback["dataset_id"],
            "feedback_manifest_sha256": sha256_file(feedback_manifest_path),
            "prior_datasets": priors,
            "benchmark_content_used": False,
            "sealed_case_ids_used": False,
            **overlaps,
        },
        "policy": {
            "source_split": "non_eval_analog_only",
            "training_allowed": True,
            "contains_benchmark_content": False,
            "contains_model_outputs": False,
            "contains_teacher_outputs": False,
            "purpose": "verified_semantic_arithmetic_sft_smoke",
            "observed_validation_reused": False,
            "all_targets_deterministically_verified": True,
        },
        "samples": samples,
    }
    dataset["summary"] = summarize_analog_dataset(dataset)
    validate_analog_dataset(dataset)
    return dataset


def build_process_trace_dataset(
    feedback_manifest_path: Path,
    prior_dataset_paths: list[Path],
) -> dict[str, Any]:
    feedback = json.loads(feedback_manifest_path.read_text(encoding="utf-8"))
    validate_feedback_manifest(feedback)
    priors = []
    prior_ids: set[str] = set()
    prior_exact: set[str] = set()
    prior_semantic: set[str] = set()
    prior_expressions: set[str] = set()
    for path in prior_dataset_paths:
        prior = json.loads(path.read_text(encoding="utf-8"))
        validate_analog_dataset(prior)
        priors.append(
            {
                "dataset_id": prior["dataset_id"],
                "sha256": sha256_file(path),
            }
        )
        prior_ids.update(sample["sample_id"] for sample in prior["samples"])
        prior_exact.update(sample["exact_sha256"] for sample in prior["samples"])
        prior_semantic.update(
            sample["semantic_sha256"] for sample in prior["samples"]
        )
        prior_expressions.update(
            str(sample.get("verifier", {}).get("expression"))
            for sample in prior["samples"]
            if sample.get("verifier", {}).get("expression") is not None
        )
        prior_expressions.update(
            str(sample.get("verifier", {}).get("source_expression"))
            for sample in prior["samples"]
            if sample.get("verifier", {}).get("source_expression") is not None
        )

    samples = []
    for index in range(192):
        sample = _process_trace_sample(index)
        identity = {
            "dataset_version": "v4",
            "generation_rule": sample["generation_rule"],
            "messages": sample["messages"],
        }
        sample["sample_id"] = f"synthetic-{_hash(_canonical_json(identity))[:20]}"
        validation_offset = (index // 6 + 1) % 6
        sample["split"] = (
            "validation" if index % 6 == validation_offset else "train"
        )
        sample["source_kind"] = "deterministic_synthetic"
        sample["training_eligible"] = True
        sample["exact_sha256"] = _hash(_canonical_json(sample["messages"]))
        sample["semantic_sha256"] = _hash(_normalized_text(sample["messages"]))
        samples.append(sample)
    samples.sort(key=lambda sample: sample["sample_id"])

    overlaps = {
        "prior_sample_id_overlap": sum(
            sample["sample_id"] in prior_ids for sample in samples
        ),
        "prior_exact_overlap": sum(
            sample["exact_sha256"] in prior_exact for sample in samples
        ),
        "prior_semantic_overlap": sum(
            sample["semantic_sha256"] in prior_semantic for sample in samples
        ),
        "prior_source_expression_overlap": sum(
            sample["verifier"]["source_expression"] in prior_expressions
            for sample in samples
        ),
    }
    if any(overlaps.values()):
        raise ValueError(f"process trace data overlaps prior analogs: {overlaps}")
    rendered_samples = _canonical_json(samples)
    leaked_case_ids = [
        row["case_id"]
        for row in feedback["rows"]
        if str(row["case_id"]) in rendered_samples
    ]
    if leaked_case_ids:
        raise ValueError(f"sealed case IDs leaked into analog data: {leaked_case_ids[:5]}")

    dataset = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "verified-arithmetic-process-traces-v4",
        "version": "v4",
        "source": {
            "source_kind": "deterministic_synthetic",
            "generator": "nano_data_pipeline.analog",
            "feedback_requirement_id": feedback["dataset_id"],
            "feedback_manifest_sha256": sha256_file(feedback_manifest_path),
            "prior_datasets": priors,
            "benchmark_content_used": False,
            "sealed_case_ids_used": False,
            **overlaps,
        },
        "policy": {
            "source_split": "non_eval_analog_only",
            "training_allowed": True,
            "contains_benchmark_content": False,
            "contains_model_outputs": False,
            "contains_teacher_outputs": False,
            "purpose": "verified_arithmetic_process_sft_smoke",
            "observed_validation_reused": False,
            "all_targets_deterministically_verified": True,
            "all_intermediate_steps_verified": True,
        },
        "samples": samples,
    }
    dataset["summary"] = summarize_analog_dataset(dataset)
    validate_analog_dataset(dataset)
    return dataset


def summarize_analog_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    samples = dataset["samples"]
    return {
        "samples": len(samples),
        "by_split": dict(sorted(Counter(row["split"] for row in samples).items())),
        "by_task_family": dict(
            sorted(Counter(row["task_family"] for row in samples).items())
        ),
        "by_difficulty": dict(
            sorted(Counter(row["difficulty"] for row in samples).items())
        ),
        "training_eligible_samples": sum(
            row["training_eligible"] for row in samples
        ),
        "unique_exact_hashes": len({row["exact_sha256"] for row in samples}),
        "unique_semantic_hashes": len(
            {row["semantic_sha256"] for row in samples}
        ),
    }


def validate_analog_dataset(dataset: dict[str, Any]) -> None:
    if dataset.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported analog dataset schema")
    samples = dataset.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("analog samples must be a non-empty list")
    ids = [sample.get("sample_id") for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("analog sample IDs are not unique")
    exact = [sample.get("exact_sha256") for sample in samples]
    semantic = [sample.get("semantic_sha256") for sample in samples]
    if len(exact) != len(set(exact)):
        raise ValueError("analog samples are not exact-deduplicated")
    if len(semantic) != len(set(semantic)):
        raise ValueError("analog samples are not semantic-deduplicated")
    for sample in samples:
        if sample.get("source_kind") != "deterministic_synthetic":
            raise ValueError("analog source must be deterministic synthetic")
        if sample.get("training_eligible") is not True:
            raise ValueError("analog sample must be explicitly training eligible")
        if sample.get("split") not in {"train", "validation"}:
            raise ValueError("unknown analog split")
        if not str(sample.get("sample_id", "")).startswith("synthetic-"):
            raise ValueError("analog sample ID must not reuse benchmark case IDs")
        messages = sample.get("messages")
        if (
            not isinstance(messages, list)
            or [message.get("role") for message in messages]
            != ["system", "user", "assistant"]
        ):
            raise ValueError("analog sample must use system/user/assistant messages")
        assistant = str(messages[-1].get("content", ""))
        if sample["format_family"] == "final_choice":
            if re.fullmatch(r"FINAL: [A-D]", assistant) is None:
                raise ValueError("invalid choice target")
        elif sample["format_family"] == "final_numeric":
            if re.fullmatch(
                r"FINAL: [-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)",
                assistant,
            ) is None:
                raise ValueError("invalid numeric target")
            if "verifier" in sample:
                raise ValueError("format-only sample must not include a verifier")
        elif sample["format_family"] == "trace_numeric":
            verifier = sample.get("verifier", {})
            if set(verifier) != {"kind", "expression", "expected_result"}:
                raise ValueError("trace sample verifier fields are invalid")
            if verifier["kind"] != "safe_ast_arithmetic_v1":
                raise ValueError("unknown trace verifier")
            match = re.fullmatch(
                (
                    r"CALC: (.+) = "
                    r"([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))\n"
                    r"FINAL: ([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))"
                ),
                assistant,
            )
            if match is None:
                raise ValueError("invalid trace target")
            expression, calc_result, final_result = match.groups()
            verified = format_number(evaluate_arithmetic(expression))
            if (
                expression != verifier["expression"]
                or calc_result != verifier["expected_result"]
                or final_result != verifier["expected_result"]
                or verified != verifier["expected_result"]
            ):
                raise ValueError("trace verifier mismatch")
        elif sample["format_family"] == "process_trace_numeric":
            verifier = sample.get("verifier", {})
            if set(verifier) != {
                "kind",
                "source_expression",
                "steps",
                "expected_result",
            }:
                raise ValueError("process trace verifier fields are invalid")
            if verifier["kind"] != "safe_ast_arithmetic_process_v2":
                raise ValueError("unknown process trace verifier")
            steps = verifier["steps"]
            if not isinstance(steps, list) or len(steps) not in {2, 3}:
                raise ValueError("process trace must contain two or three steps")
            lines = assistant.splitlines()
            if len(lines) != len(steps) + 1:
                raise ValueError("process trace line count mismatch")
            previous_result = None
            for index, (line, step) in enumerate(
                zip(lines[:-1], steps),
                start=1,
            ):
                if set(step) != {"expression", "expected_result"}:
                    raise ValueError("process step verifier fields are invalid")
                match = re.fullmatch(
                    (
                        rf"STEP {index}: (.+) = "
                        r"([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))"
                    ),
                    line,
                )
                if match is None:
                    raise ValueError("invalid process trace step")
                expression, result = match.groups()
                verified = format_number(evaluate_arithmetic(expression))
                if (
                    expression != step["expression"]
                    or result != step["expected_result"]
                    or verified != step["expected_result"]
                ):
                    raise ValueError("process step verifier mismatch")
                if (
                    previous_result is not None
                    and previous_result not in expression.split()
                ):
                    raise ValueError(
                        "process step does not consume the prior result"
                    )
                previous_result = step["expected_result"]
            final_match = re.fullmatch(
                (
                    r"FINAL: "
                    r"([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))"
                ),
                lines[-1],
            )
            source_result = format_number(
                evaluate_arithmetic(verifier["source_expression"])
            )
            if (
                final_match is None
                or previous_result != verifier["expected_result"]
                or final_match.group(1) != verifier["expected_result"]
                or source_result != verifier["expected_result"]
            ):
                raise ValueError("process final verifier mismatch")
        else:
            raise ValueError("unknown analog format family")
        if sample["exact_sha256"] != _hash(_canonical_json(messages)):
            raise ValueError("analog exact hash mismatch")
        if sample["semantic_sha256"] != _hash(_normalized_text(messages)):
            raise ValueError("analog semantic hash mismatch")
    if dataset.get("summary") != summarize_analog_dataset(dataset):
        raise ValueError("analog summary does not match samples")
    source = dataset.get("source", {})
    if source.get("benchmark_content_used") is not False:
        raise ValueError("analog data must not use benchmark content")
    if source.get("sealed_case_ids_used") is not False:
        raise ValueError("analog data must not use sealed case IDs")
    policy = dataset.get("policy", {})
    if (
        policy.get("source_split") != "non_eval_analog_only"
        or policy.get("training_allowed") is not True
        or policy.get("contains_benchmark_content") is not False
    ):
        raise ValueError("analog training boundary is invalid")
