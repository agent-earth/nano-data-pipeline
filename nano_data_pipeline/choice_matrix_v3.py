from __future__ import annotations

import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from nano_data_pipeline.analog import (
    _canonical_json,
    _hash,
    _normalized_text,
    validate_analog_dataset,
)
from nano_data_pipeline.choice_matrix import (
    SYSTEM_PROMPT,
    _normalized_prompt,
    validate_choice_capability_matrix,
)
from nano_data_pipeline.choice_matrix_v2 import (
    validate_choice_verifier_matrix_v2,
)
from nano_data_pipeline.feedback import sha256_file


SCHEMA_VERSION = "nano_choice_exact_replication_matrix_v3"
LETTERS = "ABCD"


def _option_values(result: int, index: int) -> tuple[list[int], str]:
    correct_index = (index * 7 + 3) % 4
    values = [result - 31, result - 13, result + 17, result + 43]
    values[correct_index] = result
    return values, LETTERS[correct_index]


def _render(
    family: str,
    index: int,
    prompt: str,
    values: list[int],
    reference: str,
    *,
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
        "matrix_version": "v3",
        "family": family,
        "index": index,
        "messages": messages,
        "reference": reference,
        "expected_route": "verified_override",
    }
    return {
        "case_id": f"matrix3-{_hash(_canonical_json(identity))[:20]}",
        "family": family,
        "difficulty": difficulty,
        "source_kind": "deterministic_synthetic",
        "source_signature": f"matrix_v3:{family}:{index}",
        "prompt": user,
        "reference": reference,
        "expected_route": "verified_override",
        "training_eligible": False,
        "prompt_sha256": _hash(_normalized_prompt(user)),
        "exact_sha256": _hash(_canonical_json(messages)),
        "semantic_sha256": _hash(_normalized_text(messages)),
    }


def _host_case(index: int) -> dict[str, Any]:
    delegates = 2003 + index * 97
    guests = 7 + (index % 6)
    result = 1 + delegates + delegates * guests
    values, reference = _option_values(result, index)
    return _render(
        "host_count_exact_replication",
        index,
        (
            f"A coordinator attends a summit, registers {delegates} delegates, "
            f"and every delegate brings {guests} guests. Including the "
            "coordinator, how many people attend?"
        ),
        values,
        reference,
        difficulty="host_count_exact_replication",
    )


def _verbal_average_case(index: int) -> dict[str, Any]:
    north = 12013 + index * 109
    south = 15017 + index * 83
    if (north + south) % 2:
        south += 1
    result = Fraction(north + south, 2)
    if result.denominator != 1:
        raise ValueError("verbal-average replication result must be integral")
    values, reference = _option_values(result.numerator, index + 16)
    return _render(
        "verbal_average_exact_replication",
        index,
        (
            f"A north depot processed {north} parcels and a south depot "
            f"processed {south} parcels. What is the average number of "
            "parcels processed by the two depots?"
        ),
        values,
        reference,
        difficulty="verbal_average_exact_replication",
    )


BUILDERS: tuple[Callable[[int], dict[str, Any]], ...] = (
    _host_case,
    _verbal_average_case,
)


def _validate_prior_matrix(matrix: dict[str, Any]) -> None:
    schema = matrix.get("schema_version")
    if schema == "nano_choice_capability_matrix_v1":
        validate_choice_capability_matrix(matrix)
    elif schema == "nano_choice_verifier_matrix_v2":
        validate_choice_verifier_matrix_v2(matrix)
    else:
        raise ValueError(f"unsupported prior matrix schema: {schema}")


def build_choice_exact_replication_matrix_v3(
    prior_dataset_paths: list[Path],
    prior_matrix_paths: list[Path],
) -> dict[str, Any]:
    if not prior_dataset_paths or len(prior_matrix_paths) < 2:
        raise ValueError("matrix v3 requires dataset history and matrices v1-v2")
    prior_ids: set[str] = set()
    prior_exact: set[str] = set()
    prior_semantic: set[str] = set()
    prior_prompts: set[str] = set()
    prior_signatures: set[str] = set()
    priors = []
    for path in prior_dataset_paths:
        prior = json.loads(path.read_text(encoding="utf-8"))
        validate_analog_dataset(prior)
        priors.append(
            {
                "kind": "dataset",
                "id": prior["dataset_id"],
                "sha256": sha256_file(path),
            }
        )
        for row in prior["samples"]:
            prior_ids.add(str(row["sample_id"]))
            prior_exact.add(str(row["exact_sha256"]))
            prior_semantic.add(str(row["semantic_sha256"]))
            prior_prompts.add(
                _hash(_normalized_prompt(str(row["messages"][1]["content"])))
            )
            if row.get("source_signature") is not None:
                prior_signatures.add(str(row["source_signature"]))
    for path in prior_matrix_paths:
        prior = json.loads(path.read_text(encoding="utf-8"))
        _validate_prior_matrix(prior)
        priors.append(
            {
                "kind": "matrix",
                "id": prior["matrix_id"],
                "sha256": sha256_file(path),
            }
        )
        for row in prior["cases"]:
            prior_ids.add(str(row["case_id"]))
            prior_exact.add(str(row["exact_sha256"]))
            prior_semantic.add(str(row["semantic_sha256"]))
            prior_prompts.add(str(row["prompt_sha256"]))
            prior_signatures.add(str(row["source_signature"]))

    cases = [builder(index) for builder in BUILDERS for index in range(16)]
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
        raise ValueError(f"matrix v3 overlaps prior history: {overlaps}")
    matrix = {
        "schema_version": SCHEMA_VERSION,
        "matrix_id": "generic-choice-exact-replication-matrix-v3",
        "version": "v3",
        "source": {
            "source_kind": "deterministic_synthetic",
            "generator": "nano_data_pipeline.choice_matrix_v3",
            "prior_assets": priors,
            "benchmark_content_used": False,
            "canary_content_used": False,
            "independent_holdout_content_used": False,
            "model_outputs_used": False,
            "teacher_outputs_used": False,
            **overlaps,
        },
        "policy": {
            "purpose": "history_disjoint_exact_verifier_replication",
            "evaluation_only": True,
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
    matrix["summary"] = summarize_choice_exact_replication_matrix_v3(matrix)
    validate_choice_exact_replication_matrix_v3(matrix)
    return matrix


def summarize_choice_exact_replication_matrix_v3(
    matrix: dict[str, Any],
) -> dict[str, Any]:
    cases = matrix["cases"]
    return {
        "cases": len(cases),
        "by_family": dict(sorted(Counter(row["family"] for row in cases).items())),
        "by_expected_route": dict(
            sorted(Counter(row["expected_route"] for row in cases).items())
        ),
        "scored_cases": sum(row["reference"] is not None for row in cases),
        "ambiguity_cases": sum(row["reference"] is None for row in cases),
        "training_eligible_cases": sum(row["training_eligible"] for row in cases),
        "unique_prompt_hashes": len({row["prompt_sha256"] for row in cases}),
        "unique_exact_hashes": len({row["exact_sha256"] for row in cases}),
        "unique_semantic_hashes": len(
            {row["semantic_sha256"] for row in cases}
        ),
    }


def validate_choice_exact_replication_matrix_v3(
    matrix: dict[str, Any],
) -> None:
    if matrix.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported choice exact replication schema")
    cases = matrix.get("cases")
    if not isinstance(cases, list) or len(cases) != 32:
        raise ValueError("choice exact replication must contain 32 cases")
    for field in ("case_id", "prompt_sha256", "exact_sha256", "semantic_sha256"):
        if len({row.get(field) for row in cases}) != len(cases):
            raise ValueError(f"choice exact replication {field} is not unique")
    for row in cases:
        if (
            row.get("source_kind") != "deterministic_synthetic"
            or row.get("training_eligible") is not False
            or not str(row.get("case_id", "")).startswith("matrix3-")
            or row.get("expected_route") != "verified_override"
            or row.get("family")
            not in {
                "host_count_exact_replication",
                "verbal_average_exact_replication",
            }
            or re.fullmatch(r"[A-D]", str(row.get("reference"))) is None
        ):
            raise ValueError("choice exact replication case boundary differs")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
        ]
        if row["prompt_sha256"] != _hash(
            _normalized_prompt(row["prompt"])
        ):
            raise ValueError("choice exact replication prompt hash mismatch")
        if row["exact_sha256"] != _hash(_canonical_json(messages)):
            raise ValueError("choice exact replication exact hash mismatch")
        if row["semantic_sha256"] != _hash(_normalized_text(messages)):
            raise ValueError("choice exact replication semantic hash mismatch")
    if matrix.get("summary") != summarize_choice_exact_replication_matrix_v3(
        matrix
    ):
        raise ValueError("choice exact replication summary mismatch")
    source = matrix.get("source", {})
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
        raise ValueError("choice exact replication source boundary differs")
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
        raise ValueError("choice exact replication overlap differs")
    policy = matrix.get("policy", {})
    if policy.get("evaluation_only") is not True:
        raise ValueError("choice exact replication must be evaluation-only")
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
        raise ValueError("choice exact replication training policy differs")
