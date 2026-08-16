from __future__ import annotations

import ast
import copy
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


def _preservation_word_sample(index: int) -> dict[str, Any]:
    mode = index % 4
    if mode == 0:
        invited = 40 + index * 2
        companions = 2 + (index % 3)
        direct = 5 + (index * 3 % 17)
        partners = 2 + (index * 5 % max(3, direct - 1))
        expression = (
            f"1 + {invited} + {invited} * {companions} + "
            f"{direct} + {partners}"
        )
        result = 1 + invited + invited * companions + direct + partners
        user = (
            f"Nora hosts a community dinner and attends it herself. She "
            f"invites {invited} neighbors, and each neighbor brings "
            f"{companions} additional guests. Nora separately invites "
            f"{direct} coworkers, and {partners} of those coworkers bring one "
            "partner each. How many seats are needed in total, including "
            "Nora? Show one WORK line, then put FINAL on its own line."
        )
        rule = "host_and_companion_count"
    elif mode == 1:
        red = 24 + (index // 4) * 4
        percent = (25, 50, 75, 100)[(index // 4) % 4]
        green = red + red * percent // 100
        yellow = red + green
        expression = f"{red} + {green} + {yellow}"
        result = red + green + yellow
        user = (
            f"A craft box has {red} red tiles. It has {percent}% more green "
            "tiles than red tiles. The number of yellow tiles equals the sum "
            "of the red and green tiles. How many tiles are in the three "
            "colors altogether? Show one WORK line, then put FINAL on its own "
            "line."
        )
        rule = "percentage_category_total"
    elif mode == 2:
        first_items = 18 + index
        first_distance = 6 + (index * 5 % 13)
        second_items = 15 + index
        second_distance = 7 + (index * 7 % 11)
        first_total = first_items * first_distance
        second_total = second_items * second_distance
        if (first_total + second_total) % 2:
            second_items += 1
            second_total = second_items * second_distance
        expression = f"({first_total} + {second_total}) / 2"
        result = (first_total + second_total) / 2
        user = (
            f"In a two-person throwing contest, Imani throws {first_items} "
            f"markers {first_distance} meters each. Pavel throws "
            f"{second_items} markers {second_distance} meters each. What is "
            "the average of the two contestants' total distances? Average the "
            "two contestant totals, not the individual markers. Show one WORK "
            "line, then put FINAL on its own line."
        )
        rule = "average_participant_totals"
    else:
        initial = 1200 + (index // 4) * 120
        first = initial // 4
        remaining = initial - first
        second = remaining // 3
        result = remaining - second
        expression = f"{initial} - {first} - {second}"
        user = (
            f"A mosaic kit begins with {initial} pieces unplaced. Tariq places "
            "one quarter of all the pieces. Then Mei places one third of the "
            "pieces that remain after Tariq. How many pieces are still "
            "unplaced? Show one WORK line, then put FINAL on its own line."
        )
        rule = "sequential_remaining_fraction"
    expected = format_number(result)
    return {
        "task_family": "capability_preservation_numeric",
        "format_family": "reasoning_numeric",
        "difficulty": "hard_multi_step",
        "generation_rule": f"preservation_{rule}_v5",
        "source_signature": f"{rule}:{index}",
        "verifier": {
            "kind": "safe_ast_reasoning_numeric_v1",
            "expression": expression,
            "expected_result": expected,
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "Solve from the stated facts. Return one executable WORK "
                    "line and then a standalone FINAL line."
                ),
            },
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": f"WORK: {expression} = {expected}\nFINAL: {expected}",
            },
        ],
    }


def _preservation_choice_sample(index: int) -> dict[str, Any]:
    mode = index % 3
    if mode == 0:
        invited = 12 + index
        guests = 2 + (index % 3)
        result = 1 + invited + invited * guests
        prompt = (
            f"A host attends an event, invites {invited} people, and each "
            f"invitee brings {guests} additional guests. Including the host, "
            "how many people attend?"
        )
        rule = "host_count"
    elif mode == 1:
        first = 20 + index
        first_rate = 4 + (index % 7)
        second = 18 + index
        second_rate = 5 + (index % 5)
        total = first * first_rate + second * second_rate
        if total % 2:
            second += 1
            total = first * first_rate + second * second_rate
        result = total // 2
        prompt = (
            f"Two players have total scores {first} * {first_rate} and "
            f"{second} * {second_rate}. What is the average of the two player "
            "totals?"
        )
        rule = "participant_average"
    else:
        initial = 600 + index * 12
        after_first = initial * 3 // 4
        result = after_first * 2 // 3
        prompt = (
            f"A collection has {initial} items. One quarter are removed, then "
            "one third of the remaining items are removed. How many remain?"
        )
        rule = "sequential_fraction"
    correct_index = (index * 5 + 2) % 4
    offsets = (-12, -4, 6, 14)
    options = [result + offset for offset in offsets]
    options[correct_index] = result
    letters = "ABCD"
    user = (
        f"{prompt}\n"
        + "\n".join(
            f"{letter}. {value}" for letter, value in zip(letters, options)
        )
        + "\nReturn only one standalone line: FINAL: <letter>."
    )
    return {
        "task_family": "capability_preservation_choice",
        "format_family": "final_choice",
        "difficulty": "hard_multi_step",
        "generation_rule": f"preservation_{rule}_choice_v5",
        "source_signature": f"{rule}:{index}",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Solve internally and return only the required standalone "
                    "FINAL line."
                ),
            },
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": f"FINAL: {letters[correct_index]}",
            },
        ],
    }


def _preservation_process_sample(index: int) -> dict[str, Any]:
    sample = _process_trace_sample(700 + index)
    sample["generation_rule"] = sample["generation_rule"].replace(
        "verified_process_",
        "preservation_process_",
    ).replace("_v4", "_v5")
    sample["source_signature"] = f"process:{index}"
    return sample


def _targeted_host_two_sample(index: int) -> dict[str, Any]:
    participants = 260 + index * 8
    staff = 23 + (index * 5 % 19)
    assistants = 3 + (index * 7 % max(4, staff - 2))
    expression = (
        f"1 + {participants} + {participants} * 2 + "
        f"{staff} + {assistants}"
    )
    result = 1 + participants + participants * 2 + staff + assistants
    expected = format_number(result)
    return {
        "task_family": "capability_preservation_numeric",
        "format_family": "reasoning_numeric",
        "difficulty": "hard_multi_step",
        "generation_rule": "targeted_host_two_count_v6",
        "source_signature": f"targeted_host_two_count:{index}",
        "verifier": {
            "kind": "safe_ast_reasoning_numeric_v1",
            "expression": expression,
            "expected_result": expected,
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "Solve from the stated facts. Return one executable WORK "
                    "line and then a standalone FINAL line."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Elena coordinates a field day and counts herself. She "
                    f"registers {participants} participants, and every "
                    "participant arrives with exactly 2 helpers. A separate "
                    f"logistics team has {staff} members, and {assistants} "
                    "of those members each bring one assistant. How many "
                    "people are present in total, including Elena? Show one "
                    "WORK line, then put FINAL on its own line."
                ),
            },
            {
                "role": "assistant",
                "content": f"WORK: {expression} = {expected}\nFINAL: {expected}",
            },
        ],
    }


def _host_multiplier(sample: dict[str, Any]) -> int:
    expression = str(sample.get("verifier", {}).get("expression", ""))
    match = re.fullmatch(
        r"1 \+ ([0-9]+) \+ \1 \* ([0-9]+) \+ [0-9]+ \+ [0-9]+",
        expression,
    )
    if match is None:
        raise ValueError("host-count expression does not match its contract")
    return int(match.group(2))


def _failure_targeted_numeric_sample(
    family: str,
    index: int,
) -> dict[str, Any]:
    if family == "percentage_increase_total_composition":
        baseline = 120 + index * 20
        percent = (20, 25, 50, 75)[index % 4]
        increased = baseline + baseline * percent / 100
        expression = (
            f"{baseline} + ({baseline} + {baseline} * {percent} / 100)"
        )
        result = baseline + increased
        user = (
            f"A records archive stores {baseline} paper files. It stores "
            f"{percent}% more digital files than paper files. How many files "
            "are stored altogether across both types? Show one WORK line, "
            "then put FINAL on its own line."
        )
    elif family == "packing_efficiency_effective_volume":
        packed_items = 120 + index * 12
        item_volume = 2 + index % 3
        efficiency = (50, 60, 75, 80)[index % 4]
        category_percent = (25, 50, 75, 25)[index % 4]
        volume = packed_items * item_volume * 100 // efficiency
        expression = (
            f"{volume} * {efficiency} / 100 / {item_volume} * "
            f"{category_percent} / 100"
        )
        result = (
            volume
            * efficiency
            / 100
            / item_volume
            * category_percent
            / 100
        )
        user = (
            f"A storage chamber has volume {volume} cubic units. Packing uses "
            f"{efficiency}% of that volume, and each component occupies "
            f"{item_volume} cubic units. If {category_percent}% of the packed "
            "components are blue, how many blue components are packed? Show "
            "one WORK line, then put FINAL on its own line."
        )
    elif family == "weighted_recurring_schedule_total":
        first_days = 2 + index % 3
        first_sessions = 1 + index % 2
        first_hours = 1 + index % 3
        second_days = 1 + index % 2
        second_sessions = 2
        second_hours = 2 + index % 2
        weeks = 10 + index
        expression = (
            f"({first_days} * {first_sessions} * {first_hours} + "
            f"{second_days} * {second_sessions} * {second_hours}) * {weeks}"
        )
        result = (
            first_days * first_sessions * first_hours
            + second_days * second_sessions * second_hours
        ) * weeks
        user = (
            f"Each week, a technician runs {first_sessions} sessions lasting "
            f"{first_hours} hours on each of {first_days} days, and "
            f"{second_sessions} sessions lasting {second_hours} hours on each "
            f"of {second_days} other days. Over {weeks} weeks, how many total "
            "session-hours does the technician run? Show one WORK line, then "
            "put FINAL on its own line."
        )
    else:
        raise ValueError(f"unknown failure-targeted family: {family}")

    expected = format_number(result)
    return {
        "task_family": "capability_preservation_numeric",
        "format_family": "reasoning_numeric",
        "difficulty": "hard_multi_step",
        "generation_rule": f"failure_targeted_{family}_v7",
        "source_signature": f"failure_targeted:{family}:{index}",
        "verifier": {
            "kind": "safe_ast_reasoning_numeric_v1",
            "expression": expression,
            "expected_result": expected,
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "Solve from the stated facts. Return one executable WORK "
                    "line and then a standalone FINAL line."
                ),
            },
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": f"WORK: {expression} = {expected}\nFINAL: {expected}",
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


def build_preservation_mix_dataset(
    feedback_manifest_path: Path,
    prior_dataset_paths: list[Path],
) -> dict[str, Any]:
    feedback = json.loads(feedback_manifest_path.read_text(encoding="utf-8"))
    validate_feedback_manifest(feedback)
    priors = []
    prior_ids: set[str] = set()
    prior_exact: set[str] = set()
    prior_semantic: set[str] = set()
    prior_signatures: set[str] = set()
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
        prior_signatures.update(
            str(sample["source_signature"])
            for sample in prior["samples"]
            if sample.get("source_signature") is not None
        )
        prior_signatures.update(
            str(sample["verifier"]["expression"])
            for sample in prior["samples"]
            if sample.get("verifier", {}).get("expression") is not None
        )
        prior_signatures.update(
            str(sample["verifier"]["source_expression"])
            for sample in prior["samples"]
            if sample.get("verifier", {}).get("source_expression") is not None
        )

    samples = []
    families = (
        ("preservation_numeric", _preservation_word_sample, 96, 0),
        ("preservation_choice", _preservation_choice_sample, 48, 1),
        ("preservation_process", _preservation_process_sample, 48, 2),
    )
    for family, builder, count, validation_offset in families:
        for index in range(count):
            sample = builder(index)
            identity = {
                "dataset_version": "v5",
                "family": family,
                "generation_rule": sample["generation_rule"],
                "messages": sample["messages"],
            }
            sample["sample_id"] = (
                f"synthetic-{_hash(_canonical_json(identity))[:20]}"
            )
            sample["split"] = (
                "validation"
                if index % 6 == validation_offset
                else "train"
            )
            sample["source_kind"] = "deterministic_synthetic"
            sample["training_eligible"] = True
            sample["exact_sha256"] = _hash(
                _canonical_json(sample["messages"])
            )
            sample["semantic_sha256"] = _hash(
                _normalized_text(sample["messages"])
            )
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
        "prior_source_signature_overlap": sum(
            sample["source_signature"] in prior_signatures
            or sample.get("verifier", {}).get("expression")
            in prior_signatures
            or sample.get("verifier", {}).get("source_expression")
            in prior_signatures
            for sample in samples
        ),
    }
    if any(overlaps.values()):
        raise ValueError(
            f"preservation mix overlaps prior analogs: {overlaps}"
        )
    rendered_samples = _canonical_json(samples)
    leaked_case_ids = [
        row["case_id"]
        for row in feedback["rows"]
        if str(row["case_id"]) in rendered_samples
    ]
    if leaked_case_ids:
        raise ValueError(
            f"sealed case IDs leaked into analog data: {leaked_case_ids[:5]}"
        )

    dataset = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "hard-preservation-mix-v5",
        "version": "v5",
        "source": {
            "source_kind": "deterministic_synthetic",
            "generator": "nano_data_pipeline.analog",
            "feedback_requirement_id": feedback["dataset_id"],
            "feedback_manifest_sha256": sha256_file(
                feedback_manifest_path
            ),
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
            "purpose": "hard_capability_preservation_sft_smoke",
            "observed_validation_reused": False,
            "all_numeric_targets_deterministically_verified": True,
            "all_intermediate_steps_verified": True,
            "sealed_canary_used_for_training": False,
        },
        "samples": samples,
    }
    dataset["summary"] = summarize_analog_dataset(dataset)
    validate_analog_dataset(dataset)
    return dataset


def build_targeted_preservation_mix_dataset(
    feedback_manifest_path: Path,
    base_dataset_path: Path,
    development_report_path: Path,
    prior_dataset_paths: list[Path],
) -> dict[str, Any]:
    feedback = json.loads(feedback_manifest_path.read_text(encoding="utf-8"))
    validate_feedback_manifest(feedback)
    base = json.loads(base_dataset_path.read_text(encoding="utf-8"))
    validate_analog_dataset(base)
    if base.get("dataset_id") != "hard-preservation-mix-v5":
        raise ValueError("targeted preservation base must be v5")
    development = json.loads(
        development_report_path.read_text(encoding="utf-8")
    )
    if development.get("experiment_id") != "hard-preservation-sft-smoke-v10":
        raise ValueError("targeted preservation requires the frozen v10 report")
    numeric_failures = set(
        development["post_sft_validation"]["by_family"][
            "capability_preservation_numeric"
        ]["semantic_failure_sample_ids"]
    )
    failure_rows = [
        sample
        for sample in base["samples"]
        if sample["sample_id"] in numeric_failures
    ]
    if (
        len(numeric_failures) != 7
        or len(failure_rows) != 7
        or any(sample["split"] != "validation" for sample in failure_rows)
        or {
            sample["generation_rule"] for sample in failure_rows
        }
        != {"preservation_host_and_companion_count_v5"}
    ):
        raise ValueError(
            "v10 numeric failures do not match the frozen host-count diagnosis"
        )
    host_support = {
        split: dict(
            sorted(
                Counter(
                    str(_host_multiplier(sample))
                    for sample in base["samples"]
                    if sample["split"] == split
                    and sample["generation_rule"]
                    == "preservation_host_and_companion_count_v5"
                ).items()
            )
        )
        for split in ("train", "validation")
    }
    if host_support != {
        "train": {"3": 8, "4": 8},
        "validation": {"2": 8},
    }:
        raise ValueError(f"unexpected v5 host-count support: {host_support}")

    prior_ids: set[str] = set()
    prior_exact: set[str] = set()
    prior_semantic: set[str] = set()
    prior_signatures: set[str] = set()
    priors = []
    for path in [*prior_dataset_paths, base_dataset_path]:
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
        prior_signatures.update(
            str(sample["source_signature"])
            for sample in prior["samples"]
            if sample.get("source_signature") is not None
        )

    samples = copy.deepcopy(base["samples"])
    replaced_ids = []
    replacement_index = 0
    for position, existing in enumerate(samples):
        if (
            existing["split"] != "train"
            or existing["generation_rule"]
            != "preservation_host_and_companion_count_v5"
        ):
            continue
        sample = _targeted_host_two_sample(replacement_index)
        identity = {
            "dataset_version": "v6",
            "position": position,
            "generation_rule": sample["generation_rule"],
            "messages": sample["messages"],
        }
        sample["sample_id"] = (
            f"synthetic-{_hash(_canonical_json(identity))[:20]}"
        )
        sample["split"] = "train"
        sample["source_kind"] = "deterministic_synthetic"
        sample["training_eligible"] = True
        sample["exact_sha256"] = _hash(_canonical_json(sample["messages"]))
        sample["semantic_sha256"] = _hash(
            _normalized_text(sample["messages"])
        )
        replaced_ids.append(existing["sample_id"])
        samples[position] = sample
        replacement_index += 1

    if replacement_index != 16:
        raise ValueError(
            f"targeted preservation expected 16 replacements, got "
            f"{replacement_index}"
        )
    replacements = [
        sample
        for sample in samples
        if sample["generation_rule"] == "targeted_host_two_count_v6"
    ]
    overlaps = {
        "prior_sample_id_overlap": sum(
            sample["sample_id"] in prior_ids for sample in replacements
        ),
        "prior_exact_overlap": sum(
            sample["exact_sha256"] in prior_exact for sample in replacements
        ),
        "prior_semantic_overlap": sum(
            sample["semantic_sha256"] in prior_semantic
            for sample in replacements
        ),
        "prior_source_signature_overlap": sum(
            sample["source_signature"] in prior_signatures
            for sample in replacements
        ),
    }
    if any(overlaps.values()):
        raise ValueError(
            f"targeted preservation replacements overlap priors: {overlaps}"
        )
    rendered_replacements = _canonical_json(replacements)
    leaked_case_ids = [
        row["case_id"]
        for row in feedback["rows"]
        if str(row["case_id"]) in rendered_replacements
    ]
    if leaked_case_ids:
        raise ValueError(
            "sealed case IDs leaked into targeted data: "
            f"{leaked_case_ids[:5]}"
        )

    dataset = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "targeted-preservation-mix-v6",
        "version": "v6",
        "source": {
            "source_kind": "deterministic_synthetic",
            "generator": "nano_data_pipeline.analog",
            "feedback_requirement_id": feedback["dataset_id"],
            "feedback_manifest_sha256": sha256_file(
                feedback_manifest_path
            ),
            "base_dataset": {
                "dataset_id": base["dataset_id"],
                "sha256": sha256_file(base_dataset_path),
            },
            "development_evidence": {
                "experiment_id": development["experiment_id"],
                "report_sha256": sha256_file(development_report_path),
                "numeric_failure_count": len(numeric_failures),
                "numeric_failure_ids_sha256": _hash(
                    _canonical_json(sorted(numeric_failures))
                ),
                "failure_generation_rules": {
                    "preservation_host_and_companion_count_v5": 7
                },
                "base_host_multiplier_support": host_support,
            },
            "prior_datasets": priors,
            "benchmark_content_used": False,
            "sealed_case_ids_used": False,
            "replacement_count": replacement_index,
            "replaced_sample_ids_sha256": _hash(
                _canonical_json(sorted(replaced_ids))
            ),
            **overlaps,
        },
        "policy": {
            "source_split": "non_eval_analog_only",
            "training_allowed": True,
            "contains_benchmark_content": False,
            "contains_model_outputs": False,
            "contains_teacher_outputs": False,
            "purpose": "targeted_numeric_covariate_sft_smoke",
            "observed_validation_reused": True,
            "validation_role": "development_gate_only",
            "all_numeric_targets_deterministically_verified": True,
            "all_intermediate_steps_verified": True,
            "sealed_canary_used_for_training": False,
        },
        "samples": samples,
    }
    dataset["summary"] = summarize_analog_dataset(dataset)
    validate_analog_dataset(dataset)
    return dataset


def build_failure_targeted_preservation_mix_dataset(
    feedback_manifest_path: Path,
    failure_family_receipt_path: Path,
    base_dataset_path: Path,
    prior_dataset_paths: list[Path],
) -> dict[str, Any]:
    feedback = json.loads(feedback_manifest_path.read_text(encoding="utf-8"))
    validate_feedback_manifest(feedback)
    receipt = json.loads(
        failure_family_receipt_path.read_text(encoding="utf-8")
    )
    required_families = {
        "percentage_increase_total_composition",
        "packing_efficiency_effective_volume",
        "weighted_recurring_schedule_total",
        "developmental_perception_experience_choice",
    }
    if (
        receipt.get("schema_version")
        != "nano_harness_failure_family_receipt_v1"
        or {row["family"] for row in receipt.get("families", [])}
        != required_families
        or receipt.get("policy", {}).get("contains_case_ids") is not False
        or receipt.get("policy", {}).get("contains_prompts") is not False
        or receipt.get("policy", {}).get("contains_references") is not False
        or receipt.get("policy", {}).get("contains_predictions") is not False
        or receipt.get("policy", {}).get("contains_raw_outputs") is not False
        or receipt.get("policy", {}).get("direct_training_allowed") is not False
        or receipt.get("policy", {}).get("fresh_analog_generation_allowed")
        is not True
    ):
        raise ValueError("failure-family receipt violates the v7 boundary")

    base = json.loads(base_dataset_path.read_text(encoding="utf-8"))
    validate_analog_dataset(base)
    if base.get("dataset_id") != "targeted-preservation-mix-v6":
        raise ValueError("failure-targeted preservation base must be v6")

    prior_ids: set[str] = set()
    prior_exact: set[str] = set()
    prior_semantic: set[str] = set()
    prior_signatures: set[str] = set()
    priors = []
    for path in [*prior_dataset_paths, base_dataset_path]:
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
        prior_signatures.update(
            str(sample["source_signature"])
            for sample in prior["samples"]
            if sample.get("source_signature") is not None
        )

    samples = copy.deepcopy(base["samples"])
    replaced_ids = []
    family_counts: Counter[str] = Counter()
    percentage_positions = [
        position
        for position, sample in enumerate(samples)
        if sample["split"] == "train"
        and sample["generation_rule"]
        == "preservation_percentage_category_total_v5"
    ]
    sequential_positions = [
        position
        for position, sample in enumerate(samples)
        if sample["split"] == "train"
        and sample["generation_rule"]
        == "preservation_sequential_remaining_fraction_v5"
    ]
    position_families = [
        *[
            (position, "percentage_increase_total_composition")
            for position in percentage_positions[:8]
        ],
        *[
            (position, "packing_efficiency_effective_volume")
            for position in percentage_positions[8:16]
        ],
        *[
            (position, "weighted_recurring_schedule_total")
            for position in sequential_positions[:8]
        ],
    ]
    if len(percentage_positions) != 24 or len(sequential_positions) != 24:
        raise ValueError("v6 numeric support differs from the frozen contract")
    if len(position_families) != 24:
        raise ValueError("failure-targeted replacement plan is incomplete")

    for position, family in position_families:
        existing = samples[position]
        family_index = family_counts[family]
        sample = _failure_targeted_numeric_sample(family, family_index)
        identity = {
            "dataset_version": "v7",
            "position": position,
            "generation_rule": sample["generation_rule"],
            "messages": sample["messages"],
        }
        sample["sample_id"] = (
            f"synthetic-{_hash(_canonical_json(identity))[:20]}"
        )
        sample["split"] = "train"
        sample["source_kind"] = "deterministic_synthetic"
        sample["training_eligible"] = True
        sample["exact_sha256"] = _hash(_canonical_json(sample["messages"]))
        sample["semantic_sha256"] = _hash(
            _normalized_text(sample["messages"])
        )
        replaced_ids.append(existing["sample_id"])
        samples[position] = sample
        family_counts[family] += 1

    expected_family_counts = {
        "percentage_increase_total_composition": 8,
        "packing_efficiency_effective_volume": 8,
        "weighted_recurring_schedule_total": 8,
    }
    if dict(sorted(family_counts.items())) != expected_family_counts:
        raise ValueError(
            f"failure-targeted family counts differ: {family_counts}"
        )
    replacements = [
        sample
        for sample in samples
        if sample["generation_rule"].startswith("failure_targeted_")
    ]
    overlaps = {
        "prior_sample_id_overlap": sum(
            sample["sample_id"] in prior_ids for sample in replacements
        ),
        "prior_exact_overlap": sum(
            sample["exact_sha256"] in prior_exact for sample in replacements
        ),
        "prior_semantic_overlap": sum(
            sample["semantic_sha256"] in prior_semantic
            for sample in replacements
        ),
        "prior_source_signature_overlap": sum(
            sample["source_signature"] in prior_signatures
            for sample in replacements
        ),
    }
    if any(overlaps.values()):
        raise ValueError(
            f"failure-targeted replacements overlap priors: {overlaps}"
        )
    rendered_replacements = _canonical_json(replacements)
    leaked_case_ids = [
        row["case_id"]
        for row in feedback["rows"]
        if str(row["case_id"]) in rendered_replacements
    ]
    if leaked_case_ids:
        raise ValueError(
            "sealed case IDs leaked into failure-targeted data: "
            f"{leaked_case_ids[:5]}"
        )

    base_validation = [
        sample for sample in base["samples"] if sample["split"] == "validation"
    ]
    new_validation = [
        sample for sample in samples if sample["split"] == "validation"
    ]
    if new_validation != base_validation:
        raise ValueError("v7 must preserve all v6 development rows")

    dataset = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "failure-targeted-preservation-mix-v7",
        "version": "v7",
        "source": {
            "source_kind": "deterministic_synthetic",
            "generator": "nano_data_pipeline.analog",
            "feedback_requirement_id": feedback["dataset_id"],
            "feedback_manifest_sha256": sha256_file(
                feedback_manifest_path
            ),
            "failure_family_receipt": {
                "receipt_id": receipt["receipt_id"],
                "sha256": sha256_file(failure_family_receipt_path),
                "source_case_id_set_sha256": receipt["source"][
                    "source_case_id_set_sha256"
                ],
            },
            "base_dataset": {
                "dataset_id": base["dataset_id"],
                "sha256": sha256_file(base_dataset_path),
            },
            "prior_datasets": priors,
            "benchmark_content_used": False,
            "sealed_case_ids_used": False,
            "replacement_count": len(replacements),
            "replacement_family_counts": dict(sorted(family_counts.items())),
            "deferred_feedback_families": [
                "developmental_perception_experience_choice"
            ],
            "replaced_sample_ids_sha256": _hash(
                _canonical_json(sorted(replaced_ids))
            ),
            **overlaps,
        },
        "policy": {
            "source_split": "non_eval_analog_only",
            "training_allowed": True,
            "contains_benchmark_content": False,
            "contains_model_outputs": False,
            "contains_teacher_outputs": False,
            "purpose": "failure_targeted_preservation_sft_smoke",
            "observed_validation_reused": True,
            "validation_role": "development_gate_only",
            "all_numeric_targets_deterministically_verified": True,
            "all_intermediate_steps_verified": True,
            "sealed_canary_used_for_training": False,
            "independent_holdout_used_for_training": False,
        },
        "samples": samples,
    }
    dataset["summary"] = summarize_analog_dataset(dataset)
    validate_analog_dataset(dataset)
    return dataset


def build_percentage_isolation_preservation_mix_dataset(
    feedback_manifest_path: Path,
    failure_family_receipt_path: Path,
    base_dataset_path: Path,
    broad_dataset_path: Path,
    prior_dataset_paths: list[Path],
) -> dict[str, Any]:
    feedback = json.loads(feedback_manifest_path.read_text(encoding="utf-8"))
    validate_feedback_manifest(feedback)
    receipt = json.loads(
        failure_family_receipt_path.read_text(encoding="utf-8")
    )
    if (
        receipt.get("schema_version")
        != "nano_harness_failure_family_receipt_v1"
        or "percentage_increase_total_composition"
        not in {row["family"] for row in receipt.get("families", [])}
        or receipt.get("policy", {}).get("contains_case_ids") is not False
        or receipt.get("policy", {}).get("contains_prompts") is not False
        or receipt.get("policy", {}).get("contains_references") is not False
        or receipt.get("policy", {}).get("contains_predictions") is not False
        or receipt.get("policy", {}).get("contains_raw_outputs") is not False
        or receipt.get("policy", {}).get("fresh_analog_generation_allowed")
        is not True
    ):
        raise ValueError("failure-family receipt violates the v8 boundary")

    base = json.loads(base_dataset_path.read_text(encoding="utf-8"))
    broad = json.loads(broad_dataset_path.read_text(encoding="utf-8"))
    validate_analog_dataset(base)
    validate_analog_dataset(broad)
    if base.get("dataset_id") != "targeted-preservation-mix-v6":
        raise ValueError("percentage-isolation base must be v6")
    if broad.get("dataset_id") != "failure-targeted-preservation-mix-v7":
        raise ValueError("percentage-isolation source must be broad v7")
    if len(base["samples"]) != len(broad["samples"]):
        raise ValueError("v6 and v7 sample counts differ")

    prior_ids: set[str] = set()
    prior_exact: set[str] = set()
    prior_semantic: set[str] = set()
    prior_signatures: set[str] = set()
    priors = []
    for path in [*prior_dataset_paths, base_dataset_path]:
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
        prior_signatures.update(
            str(sample["source_signature"])
            for sample in prior["samples"]
            if sample.get("source_signature") is not None
        )

    samples = copy.deepcopy(base["samples"])
    selected_positions = []
    selected_rows = []
    for position, (base_row, broad_row) in enumerate(
        zip(base["samples"], broad["samples"])
    ):
        if broad_row["generation_rule"] != (
            "failure_targeted_"
            "percentage_increase_total_composition_v7"
        ):
            continue
        if (
            base_row["split"] != broad_row["split"]
            or base_row["split"] != "train"
            or base_row["task_family"] != broad_row["task_family"]
            or base_row["format_family"] != broad_row["format_family"]
        ):
            raise ValueError(
                "percentage-isolation row changes split or family contract"
            )
        samples[position] = copy.deepcopy(broad_row)
        selected_positions.append(position)
        selected_rows.append(broad_row)
    if len(selected_rows) != 8:
        raise ValueError(
            f"percentage-isolation expected 8 rows, got {len(selected_rows)}"
        )

    overlaps = {
        "prior_sample_id_overlap": sum(
            sample["sample_id"] in prior_ids for sample in selected_rows
        ),
        "prior_exact_overlap": sum(
            sample["exact_sha256"] in prior_exact for sample in selected_rows
        ),
        "prior_semantic_overlap": sum(
            sample["semantic_sha256"] in prior_semantic
            for sample in selected_rows
        ),
        "prior_source_signature_overlap": sum(
            sample["source_signature"] in prior_signatures
            for sample in selected_rows
        ),
    }
    if any(overlaps.values()):
        raise ValueError(
            f"percentage-isolation rows overlap v1-v6: {overlaps}"
        )
    rendered_selected = _canonical_json(selected_rows)
    leaked_case_ids = [
        row["case_id"]
        for row in feedback["rows"]
        if str(row["case_id"]) in rendered_selected
    ]
    if leaked_case_ids:
        raise ValueError(
            "sealed case IDs leaked into percentage-isolation data: "
            f"{leaked_case_ids[:5]}"
        )
    if (
        [row for row in samples if row["split"] == "validation"]
        != [row for row in base["samples"] if row["split"] == "validation"]
    ):
        raise ValueError("v8 must preserve all v6 development rows")

    dataset = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "percentage-isolation-preservation-mix-v8",
        "version": "v8",
        "source": {
            "source_kind": "deterministic_synthetic",
            "generator": "nano_data_pipeline.analog",
            "feedback_requirement_id": feedback["dataset_id"],
            "feedback_manifest_sha256": sha256_file(
                feedback_manifest_path
            ),
            "failure_family_receipt": {
                "receipt_id": receipt["receipt_id"],
                "sha256": sha256_file(failure_family_receipt_path),
                "source_case_id_set_sha256": receipt["source"][
                    "source_case_id_set_sha256"
                ],
            },
            "base_dataset": {
                "dataset_id": base["dataset_id"],
                "sha256": sha256_file(base_dataset_path),
            },
            "broad_ablation_source": {
                "dataset_id": broad["dataset_id"],
                "sha256": sha256_file(broad_dataset_path),
            },
            "prior_datasets": priors,
            "benchmark_content_used": False,
            "sealed_case_ids_used": False,
            "replacement_count": len(selected_rows),
            "replacement_family_counts": {
                "percentage_increase_total_composition": len(selected_rows)
            },
            "selected_positions_sha256": _hash(
                _canonical_json(selected_positions)
            ),
            "deferred_feedback_families": [
                "packing_efficiency_effective_volume",
                "weighted_recurring_schedule_total",
                "developmental_perception_experience_choice",
            ],
            **overlaps,
        },
        "policy": {
            "source_split": "non_eval_analog_only",
            "training_allowed": True,
            "contains_benchmark_content": False,
            "contains_model_outputs": False,
            "contains_teacher_outputs": False,
            "purpose": "percentage_family_isolation_sft_smoke",
            "observed_validation_reused": True,
            "validation_role": "development_gate_only",
            "all_numeric_targets_deterministically_verified": True,
            "all_intermediate_steps_verified": True,
            "sealed_canary_used_for_training": False,
            "independent_holdout_used_for_training": False,
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
        elif sample["format_family"] == "reasoning_numeric":
            verifier = sample.get("verifier", {})
            if set(verifier) != {"kind", "expression", "expected_result"}:
                raise ValueError("reasoning verifier fields are invalid")
            if verifier["kind"] != "safe_ast_reasoning_numeric_v1":
                raise ValueError("unknown reasoning verifier")
            match = re.fullmatch(
                (
                    r"WORK: (.+) = "
                    r"([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))\n"
                    r"FINAL: ([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))"
                ),
                assistant,
            )
            if match is None:
                raise ValueError("invalid reasoning target")
            expression, work_result, final_result = match.groups()
            verified = format_number(evaluate_arithmetic(expression))
            if (
                expression != verifier["expression"]
                or work_result != verifier["expected_result"]
                or final_result != verifier["expected_result"]
                or verified != verifier["expected_result"]
            ):
                raise ValueError("reasoning verifier mismatch")
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
