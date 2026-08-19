from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from nano_data_pipeline.analog import (
    evaluate_arithmetic,
    format_number,
    sha256_file,
    validate_analog_dataset,
)


FAILURE_SCHEMA = "nano_execution_failure_manifest_v1"
AUDIT_SCHEMA = "nano_execution_target_coverage_audit_v1"


def _constants(expression: str) -> list[int | float]:
    tree = ast.parse(expression, mode="eval")
    evaluate_arithmetic(expression)
    result = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.BinOp):
            visit(node.left)
            visit(node.right)
        elif (
            isinstance(node, ast.Constant)
            and type(node.value) in {int, float}
        ):
            result.append(node.value)

    visit(tree.body)
    return result


def _shape(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    evaluate_arithmetic(expression)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and type(node.value) in {int, float}
        ):
            node.value = 0
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def _equality_pattern(values: list[int | float]) -> tuple[int, ...]:
    identities: dict[int | float, int] = {}
    return tuple(
        identities.setdefault(value, len(identities)) for value in values
    )


def expression_features(expression: str) -> dict[str, Any]:
    constants = _constants(expression)
    return {
        "expression": expression,
        "result": format_number(evaluate_arithmetic(expression)),
        "shape": _shape(expression),
        "constant_values": constants,
        "equality_pattern": list(_equality_pattern(constants)),
    }


def _record(
    *,
    sample_id: str,
    expression: str,
    assistant: str,
    user: str,
    verifier: dict[str, Any],
) -> dict[str, Any]:
    features = expression_features(expression)
    steps = verifier.get("steps")
    explicit_steps = isinstance(steps, list) and bool(steps)
    expected = features["result"]
    return {
        **features,
        "sample_id": sample_id,
        "explicit_steps": explicit_steps,
        "step_count": len(steps) if explicit_steps else 0,
        "final_only_target": bool(
            re.fullmatch(r"FINAL: [-+]?[0-9]+(?:\.[0-9]+)?", assistant.strip())
        ),
        "assistant_characters": len(assistant),
        "user_characters": len(user),
        "synthetic_distractor_count": user.count("Synthetic evidence"),
        "target_value_in_user": bool(
            re.search(
                rf"(?<![0-9]){re.escape(expected)}(?![0-9])",
                user,
            )
        ),
    }


def _release_records(
    accepted_jsonl_path: Path,
    *,
    split: str,
) -> list[dict[str, Any]]:
    result = []
    with accepted_jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("family_id") != "verified-reasoning":
                continue
            if row.get("split") != split:
                continue
            messages = row["messages"]
            result.append(
                _record(
                    sample_id=row["sample_id"],
                    expression=row["task_spec"]["expression"],
                    assistant=messages[-1]["content"],
                    user=messages[-2]["content"],
                    verifier=row["verifier"],
                )
            )
    return result


def _process_records(path: Path) -> list[dict[str, Any]]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    validate_analog_dataset(dataset)
    result = []
    for row in dataset["samples"]:
        if row["split"] != "train":
            continue
        messages = row["messages"]
        verifier = row["verifier"]
        result.append(
            _record(
                sample_id=row["sample_id"],
                expression=verifier["source_expression"],
                assistant=messages[-1]["content"],
                user=messages[-2]["content"],
                verifier=verifier,
            )
        )
    return result


def _bounds(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    width = max(len(row["constant_values"]) for row in records)
    return [
        {
            "position": position,
            "minimum": min(
                row["constant_values"][position]
                for row in records
                if len(row["constant_values"]) > position
            ),
            "maximum": max(
                row["constant_values"][position]
                for row in records
                if len(row["constant_values"]) > position
            ),
        }
        for position in range(width)
    ]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("execution coverage requires non-empty records")
    distractors = sorted(row["synthetic_distractor_count"] for row in records)
    return {
        "rows": len(records),
        "unique_expressions": len({row["expression"] for row in records}),
        "unique_results": len({row["result"] for row in records}),
        "shape_counts": dict(sorted(Counter(row["shape"] for row in records).items())),
        "equality_pattern_counts": dict(
            sorted(
                Counter(
                    ",".join(map(str, row["equality_pattern"]))
                    for row in records
                ).items()
            )
        ),
        "explicit_process_rows": sum(row["explicit_steps"] for row in records),
        "final_only_rows": sum(row["final_only_target"] for row in records),
        "target_value_in_user_rows": sum(
            row["target_value_in_user"] for row in records
        ),
        "constant_bounds": _bounds(records),
        "result_bounds": {
            "minimum": min(float(row["result"]) for row in records),
            "maximum": max(float(row["result"]) for row in records),
        },
        "median_synthetic_distractors": distractors[len(distractors) // 2],
    }


def _coverage(
    records: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    expressions = Counter(row["expression"] for row in records)
    results = Counter(row["result"] for row in records)
    shapes = Counter(row["shape"] for row in records)
    relations = Counter(
        (row["shape"], tuple(row["equality_pattern"])) for row in records
    )
    process_shapes = Counter(
        row["shape"] for row in records if row["explicit_steps"]
    )
    process_relations = Counter(
        (row["shape"], tuple(row["equality_pattern"]))
        for row in records
        if row["explicit_steps"]
    )
    bounds = _bounds(records)
    result_values = [float(row["result"]) for row in records]
    rows = []
    for target in targets:
        constants = target["constant_values"]
        operand_bounds_pass = (
            len(constants) <= len(bounds)
            and all(
                bounds[index]["minimum"]
                <= value
                <= bounds[index]["maximum"]
                for index, value in enumerate(constants)
            )
        )
        target_result = float(target["result"])
        rows.append(
            {
                "sample_id": target["sample_id"],
                "expression": target["expression"],
                "expected_result": target["result"],
                "exact_expression_matches": expressions[target["expression"]],
                "exact_result_matches": results[target["result"]],
                "shape_matches": shapes[target["shape"]],
                "relation_matches": relations[
                    (target["shape"], tuple(target["equality_pattern"]))
                ],
                "process_shape_matches": process_shapes[target["shape"]],
                "process_relation_matches": process_relations[
                    (target["shape"], tuple(target["equality_pattern"]))
                ],
                "operand_bounds_pass": operand_bounds_pass,
                "result_bounds_pass": min(result_values)
                <= target_result
                <= max(result_values),
                "nearest_result_distance": min(
                    abs(value - target_result) for value in result_values
                ),
            }
        )
    aggregate_fields = (
        "exact_expression_matches",
        "exact_result_matches",
        "shape_matches",
        "relation_matches",
        "process_shape_matches",
        "process_relation_matches",
        "operand_bounds_pass",
        "result_bounds_pass",
    )
    return {
        "targets": len(targets),
        "covered_targets": {
            field: sum(bool(row[field]) for row in rows)
            for field in aggregate_fields
        },
        "rows": rows,
    }


def validate_failure_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != FAILURE_SCHEMA:
        raise ValueError("unsupported execution failure manifest")
    policy = manifest.get("policy", {})
    if (
        policy.get("training_eligible") is not False
        or policy.get("contains_benchmark_content") is not False
        or policy.get("source_result_public") is not True
    ):
        raise ValueError("execution failure manifest policy is unsafe")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("execution failure manifest contains no rows")
    seen = set()
    for row in rows:
        if row.get("sample_id") in seen:
            raise ValueError("execution failure manifest repeats a sample")
        seen.add(row.get("sample_id"))
        features = expression_features(row["expression"])
        if row.get("expected") != f"FINAL: {features['result']}":
            raise ValueError("execution failure expected result is invalid")
        if row.get("baseline_verified") or row.get("post_sft_verified"):
            raise ValueError("failure manifest contains a passing row")


def build_execution_coverage_audit(
    *,
    accepted_jsonl_path: Path,
    release_manifest_path: Path,
    process_dataset_path: Path,
    failure_manifest_path: Path,
    selected_train_rows: int,
) -> dict[str, Any]:
    release = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    expected_sha = release.get("artifacts", {}).get("accepted_jsonl_sha256")
    if sha256_file(accepted_jsonl_path) != expected_sha:
        raise ValueError("skill release accepted ledger identity mismatch")
    failures = json.loads(failure_manifest_path.read_text(encoding="utf-8"))
    validate_failure_manifest(failures)

    full_release = _release_records(accepted_jsonl_path, split="train")
    selected_release = full_release[:selected_train_rows]
    process = _process_records(process_dataset_path)
    targets = [
        {
            **expression_features(row["expression"]),
            "sample_id": row["sample_id"],
        }
        for row in failures["rows"]
    ]
    selected_coverage = _coverage(selected_release, targets)
    full_coverage = _coverage(full_release, targets)
    process_coverage = _coverage(process, targets)
    union = selected_release + process
    union_coverage = _coverage(union, targets)

    joint_process_relation_rows = sum(
        row["explicit_steps"]
        and any(
            row["shape"] == target["shape"]
            and row["equality_pattern"] == target["equality_pattern"]
            for target in targets
        )
        for row in union
    )
    minimum_contract = {
        "schema_version": "nano_execution_target_data_contract_v1",
        "purpose": "target_wrong_final_execution_without_format_shortcuts",
        "train_rows": 512,
        "dev_rows": 80,
        "minimum_train_tokens": 300000,
        "train_composition": {
            "relation_grid_semantic_tasks": 128,
            "paired_process_views": 128,
            "paired_final_only_views": 128,
            "json_preservation_rows": 256,
            "json_preservation_rows_per_family": 64,
        },
        "dev_composition": {
            "fresh_relation_semantic_tasks": 24,
            "paired_execution_views": 48,
            "json_non_regression_rows": 32,
            "json_non_regression_rows_per_family": 8,
        },
        "relation_grid": {
            "left_bins": 8,
            "repeated_operand_bins": 8,
            "multipliers": [2, 3],
            "required_relation": "(left + repeated) * multiplier - repeated",
            "paired_views_per_semantic_task": [
                "verified_process_trace",
                "final_only_contract",
            ],
        },
        "gates": {
            "every_execution_row_relation_match": True,
            "every_process_view_intermediate_verified": True,
            "paired_view_final_consistency": True,
            "train_dev_semantic_overlap": 0,
            "prior_release_semantic_overlap": 0,
            "benchmark_or_holdout_content": 0,
            "answer_value_leakage": 0,
            "family_dev_non_regression_required": True,
        },
        "training_blocked_until_contract_passes": True,
        "benchmark_holdout_rl_blocked": True,
    }
    return {
        "schema_version": AUDIT_SCHEMA,
        "audit_id": "skill-release-execution-target-coverage-v1",
        "sources": {
            "release_id": release["release_id"],
            "release_manifest_sha256": sha256_file(release_manifest_path),
            "accepted_jsonl_sha256": expected_sha,
            "process_dataset_id": "verified-arithmetic-process-traces-v4",
            "process_dataset_sha256": sha256_file(process_dataset_path),
            "failure_manifest_sha256": sha256_file(failure_manifest_path),
            "failure_source_commit": failures["source"]["result_commit"],
        },
        "target_failures": {
            "rows": len(targets),
            "sample_ids": sorted(row["sample_id"] for row in targets),
            "shape": targets[0]["shape"],
            "equality_pattern": targets[0]["equality_pattern"],
            "expected_results": sorted(
                int(row["result"]) for row in targets
            ),
        },
        "datasets": {
            "selected_release_reasoning_train": summarize_records(
                selected_release
            ),
            "full_release_reasoning_train": summarize_records(full_release),
            "process_v6_train": summarize_records(process),
        },
        "coverage": {
            "selected_release_reasoning_train": selected_coverage,
            "full_release_reasoning_train": full_coverage,
            "process_v6_train": process_coverage,
            "selected_release_plus_process_v6": union_coverage,
            "joint_process_relation_rows": joint_process_relation_rows,
        },
        "findings": {
            "release_relation_without_process": (
                selected_coverage["covered_targets"]["relation_matches"]
                == len(targets)
                and selected_coverage["covered_targets"][
                    "process_relation_matches"
                ]
                == 0
            ),
            "process_shape_without_relation": (
                process_coverage["covered_targets"]["process_shape_matches"]
                == len(targets)
                and process_coverage["covered_targets"][
                    "process_relation_matches"
                ]
                == 0
            ),
            "joint_mechanism_coverage_missing": (
                joint_process_relation_rows == 0
            ),
            "selected_release_is_final_only": (
                all(row["final_only_target"] for row in selected_release)
                and not any(row["explicit_steps"] for row in selected_release)
            ),
            "target_value_leakage_detected": any(
                row["target_value_in_user"] for row in selected_release
            ),
        },
        "minimum_fresh_data_contract": minimum_contract,
        "decision": {
            "more_sft_allowed_now": False,
            "generate_contract_dataset_next": True,
            "reuse_answer_only_oversampling": False,
            "reuse_process_v6_unchanged": False,
            "benchmark_allowed": False,
            "independent_holdout_allowed": False,
            "rl_allowed": False,
        },
        "claim_boundary": (
            "这次审计只比较 public-safe synthetic 训练机制，没有运行或"
            "评估模型。它不能证明模型能力提升，也不允许启动训练、访问 "
            "benchmark 或 independent holdout，或启动 RL。"
        ),
    }
