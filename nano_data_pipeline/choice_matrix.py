from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from nano_data_pipeline.analog import (
    _canonical_json,
    _hash,
    _normalized_text,
    validate_analog_dataset,
)
from nano_data_pipeline.feedback import sha256_file


SCHEMA_VERSION = "nano_choice_capability_matrix_v1"
LETTERS = "ABCD"
SYSTEM_PROMPT = (
    "Solve the choice task from the stated facts and return exactly one "
    "standalone FINAL: <letter> line."
)


def _normalized_prompt(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _options(
    result: int,
    index: int,
    *,
    exact: bool = True,
    duplicate: bool = False,
) -> tuple[list[int], str | None]:
    correct = (index * 3 + 1) % 4
    values = [result - 17, result - 5, result + 7, result + 19]
    if exact:
        values[correct] = result
    if duplicate:
        values[(correct + 1) % 4] = values[correct]
        return values, None
    return values, LETTERS[correct] if exact else None


def _render(
    family: str,
    index: int,
    prompt: str,
    values: list[int],
    reference: str | None,
    *,
    expected_route: str,
    difficulty: str,
) -> dict[str, Any]:
    user = (
        f"{prompt}\n"
        + "\n".join(
            f"{letter}. {value}" for letter, value in zip(LETTERS, values)
        )
        + "\nReturn exactly one standalone line: FINAL: <letter>."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    identity = {
        "matrix_version": "v1",
        "family": family,
        "index": index,
        "messages": messages,
        "reference": reference,
        "expected_route": expected_route,
    }
    return {
        "case_id": f"matrix-{_hash(_canonical_json(identity))[:20]}",
        "family": family,
        "difficulty": difficulty,
        "source_kind": "deterministic_synthetic",
        "source_signature": f"matrix_v1:{family}:{index}",
        "prompt": user,
        "reference": reference,
        "expected_route": expected_route,
        "training_eligible": False,
        "prompt_sha256": _hash(_normalized_prompt(user)),
        "exact_sha256": _hash(_canonical_json(messages)),
        "semantic_sha256": _hash(_normalized_text(messages)),
    }


def _explicit_average(index: int) -> dict[str, Any]:
    left = 310 + index * 11
    left_rate = 7 + index
    right = 270 + index * 13
    right_rate = 9 + (index % 3)
    first = left * left_rate
    second = right * right_rate
    if (first + second) % 2:
        right += 1
        second = right * right_rate
    result = (first + second) // 2
    values, reference = _options(result, index)
    return _render(
        "explicit_average_exact",
        index,
        (
            f"Two teams have totals {left} * {left_rate} and "
            f"{right} * {right_rate}. What is the average of the two totals?"
        ),
        values,
        reference,
        expected_route="verified_override",
        difficulty="two_expression",
    )


def _fractional_average(index: int) -> dict[str, Any]:
    left = 421 + index * 10
    left_rate = 5 + (index % 4)
    right = 337 + index * 8
    right_rate = 6 + (index % 3)
    first = left * left_rate
    second = right * right_rate
    if (first + second) % 2 == 0:
        right += 1
        second = right * right_rate
    result = Fraction(first + second, 2)
    nearest = result.numerator // result.denominator
    values, _ = _options(nearest, index, exact=False)
    return _render(
        "explicit_average_no_exact_option",
        index,
        (
            f"Two teams have totals {left} * {left_rate} and "
            f"{right} * {right_rate}. What is the average of the two totals?"
        ),
        values,
        None,
        expected_route="ambiguous_fallback",
        difficulty="fractional_no_match",
    )


def _verbal_average(index: int) -> dict[str, Any]:
    first = 840 + index * 31
    second = 1260 + index * 29
    result = (first + second) // 2
    if (first + second) % 2:
        second += 1
        result = (first + second) // 2
    values, reference = _options(result, index)
    return _render(
        "verbal_average_exact",
        index,
        (
            f"One warehouse shipped {first} units and another shipped "
            f"{second} units. What is their average shipment count?"
        ),
        values,
        reference,
        expected_route="unsupported_fallback",
        difficulty="implicit_expression",
    )


def _host_count(index: int) -> dict[str, Any]:
    invited = 73 + index * 7
    guests = 3 + (index % 3)
    result = 1 + invited + invited * guests
    values, reference = _options(result, index)
    return _render(
        "host_count_exact",
        index,
        (
            f"A host attends a reception, invites {invited} people, and each "
            f"invitee brings {guests} guests. How many people attend?"
        ),
        values,
        reference,
        expected_route="unsupported_fallback",
        difficulty="multi_step_word",
    )


def _sequential_fraction(index: int) -> dict[str, Any]:
    initial = 1800 + index * 120
    result = initial * 3 // 4 * 2 // 3
    values, reference = _options(result, index)
    return _render(
        "sequential_fraction_exact",
        index,
        (
            f"A collection begins with {initial} items. One quarter are "
            "removed, then one third of the remainder are removed. "
            "How many items remain?"
        ),
        values,
        reference,
        expected_route="unsupported_fallback",
        difficulty="multi_step_fraction",
    )


def _duplicate_options(index: int) -> dict[str, Any]:
    left = 510 + index * 9
    right = 390 + index * 7
    result = (left + right) // 2
    if (left + right) % 2:
        right += 1
        result = (left + right) // 2
    values, _ = _options(result, index, duplicate=True)
    return _render(
        "duplicate_option_ambiguity",
        index,
        (
            f"Two ledgers have totals {left} * 1 and {right} * 1. "
            "What is the average of the two totals?"
        ),
        values,
        None,
        expected_route="ambiguous_fallback",
        difficulty="duplicate_options",
    )


BUILDERS = (
    _explicit_average,
    _fractional_average,
    _verbal_average,
    _host_count,
    _sequential_fraction,
    _duplicate_options,
)


def build_choice_capability_matrix(
    prior_dataset_paths: list[Path],
) -> dict[str, Any]:
    if not prior_dataset_paths:
        raise ValueError("choice matrix requires prior dataset history")
    prior_ids: set[str] = set()
    prior_exact: set[str] = set()
    prior_semantic: set[str] = set()
    prior_prompts: set[str] = set()
    prior_signatures: set[str] = set()
    priors = []
    for path in prior_dataset_paths:
        prior = json.loads(path.read_text(encoding="utf-8"))
        validate_analog_dataset(prior)
        priors.append({"dataset_id": prior["dataset_id"], "sha256": sha256_file(path)})
        prior_ids.update(str(row["sample_id"]) for row in prior["samples"])
        prior_exact.update(str(row["exact_sha256"]) for row in prior["samples"])
        prior_semantic.update(str(row["semantic_sha256"]) for row in prior["samples"])
        prior_prompts.update(
            _hash(_normalized_prompt(str(row["messages"][1]["content"])))
            for row in prior["samples"]
        )
        prior_signatures.update(
            str(row["source_signature"])
            for row in prior["samples"]
            if row.get("source_signature") is not None
        )

    cases = [
        builder(index)
        for builder in BUILDERS
        for index in range(8)
    ]
    cases.sort(key=lambda row: row["case_id"])
    overlaps = {
        "prior_case_id_overlap": sum(row["case_id"] in prior_ids for row in cases),
        "prior_exact_overlap": sum(
            row["exact_sha256"] in prior_exact for row in cases
        ),
        "prior_semantic_overlap": sum(
            row["semantic_sha256"] in prior_semantic for row in cases
        ),
        "prior_prompt_overlap": sum(
            row["prompt_sha256"] in prior_prompts for row in cases
        ),
        "prior_source_signature_overlap": sum(
            row["source_signature"] in prior_signatures for row in cases
        ),
    }
    if any(overlaps.values()):
        raise ValueError(f"choice matrix overlaps prior history: {overlaps}")
    dataset = {
        "schema_version": SCHEMA_VERSION,
        "matrix_id": "generic-choice-capability-matrix-v1",
        "version": "v1",
        "source": {
            "source_kind": "deterministic_synthetic",
            "generator": "nano_data_pipeline.choice_matrix",
            "prior_datasets": priors,
            "benchmark_content_used": False,
            "canary_content_used": False,
            "independent_holdout_content_used": False,
            "model_outputs_used": False,
            "teacher_outputs_used": False,
            **overlaps,
        },
        "policy": {
            "purpose": "history_disjoint_choice_capability_evaluation",
            "training_allowed": False,
            "sft_allowed": False,
            "preference_training_allowed": False,
            "rl_allowed": False,
            "reward_model_training_allowed": False,
            "verifier_training_allowed": False,
            "case_level_feedback_training_allowed": False,
        },
        "cases": cases,
    }
    dataset["summary"] = summarize_choice_capability_matrix(dataset)
    validate_choice_capability_matrix(dataset)
    return dataset


def summarize_choice_capability_matrix(dataset: dict[str, Any]) -> dict[str, Any]:
    cases = dataset["cases"]
    return {
        "cases": len(cases),
        "by_family": dict(sorted(Counter(row["family"] for row in cases).items())),
        "by_expected_route": dict(
            sorted(Counter(row["expected_route"] for row in cases).items())
        ),
        "scored_cases": sum(row["reference"] is not None for row in cases),
        "ambiguity_cases": sum(row["reference"] is None for row in cases),
        "training_eligible_cases": sum(row["training_eligible"] for row in cases),
        "unique_exact_hashes": len({row["exact_sha256"] for row in cases}),
        "unique_prompt_hashes": len({row["prompt_sha256"] for row in cases}),
        "unique_semantic_hashes": len(
            {row["semantic_sha256"] for row in cases}
        ),
    }


def validate_choice_capability_matrix(dataset: dict[str, Any]) -> None:
    if dataset.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported choice matrix schema")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or len(cases) != 48:
        raise ValueError("choice matrix must contain 48 cases")
    if len({row.get("case_id") for row in cases}) != len(cases):
        raise ValueError("choice matrix case IDs are not unique")
    if len({row.get("exact_sha256") for row in cases}) != len(cases):
        raise ValueError("choice matrix exact hashes are not unique")
    if len({row.get("prompt_sha256") for row in cases}) != len(cases):
        raise ValueError("choice matrix prompt hashes are not unique")
    if len({row.get("semantic_sha256") for row in cases}) != len(cases):
        raise ValueError("choice matrix semantic hashes are not unique")
    for row in cases:
        if (
            row.get("source_kind") != "deterministic_synthetic"
            or row.get("training_eligible") is not False
            or not str(row.get("case_id", "")).startswith("matrix-")
            or row.get("expected_route")
            not in {
                "verified_override",
                "ambiguous_fallback",
                "unsupported_fallback",
            }
        ):
            raise ValueError("choice matrix case boundary differs")
        if row["reference"] is not None and re.fullmatch(
            r"[A-D]", str(row["reference"])
        ) is None:
            raise ValueError("choice matrix reference is invalid")
        expected_reference = (
            row["expected_route"] not in {"ambiguous_fallback"}
        )
        if (row["reference"] is not None) != expected_reference:
            raise ValueError("choice matrix reference boundary differs")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
        ]
        if row["exact_sha256"] != _hash(_canonical_json(messages)):
            raise ValueError("choice matrix exact hash mismatch")
        if row["prompt_sha256"] != _hash(
            _normalized_prompt(row["prompt"])
        ):
            raise ValueError("choice matrix prompt hash mismatch")
        if row["semantic_sha256"] != _hash(_normalized_text(messages)):
            raise ValueError("choice matrix semantic hash mismatch")
    if dataset.get("summary") != summarize_choice_capability_matrix(dataset):
        raise ValueError("choice matrix summary mismatch")
    source = dataset.get("source", {})
    if any(
        source.get(key) is not False
        for key in (
            "benchmark_content_used",
            "canary_content_used",
            "independent_holdout_content_used",
            "model_outputs_used",
            "teacher_outputs_used",
        )
    ):
        raise ValueError("choice matrix source boundary differs")
    if any(
        source.get(key) != 0
        for key in (
            "prior_case_id_overlap",
            "prior_exact_overlap",
            "prior_semantic_overlap",
            "prior_prompt_overlap",
            "prior_source_signature_overlap",
        )
    ):
        raise ValueError("choice matrix prior overlap differs")
    policy = dataset.get("policy", {})
    if policy.get("training_allowed") is not False or any(
        policy.get(key) is not False
        for key in (
            "sft_allowed",
            "preference_training_allowed",
            "rl_allowed",
            "reward_model_training_allowed",
            "verifier_training_allowed",
            "case_level_feedback_training_allowed",
        )
    ):
        raise ValueError("choice matrix training policy differs")
