from __future__ import annotations

import hashlib
import json
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
