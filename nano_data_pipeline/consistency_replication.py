from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from nano_data_pipeline.execution_target_dataset import (
    JSON_FAMILIES,
    _json_row,
    _relation_row,
    _verify_final,
    _verify_process,
)
from nano_data_pipeline.feedback import sha256_file
from nano_data_pipeline.subagent_campaign import (
    canonical_json,
    count_tokens,
    semantic_basis,
    sha256_text,
    verify_candidate,
)


DATASET_SCHEMA = "nano_consistency_replication_dataset_v1"
RELEASE_SCHEMA = "nano_consistency_replication_release_v1"
DATASET_ID = "paired-consistency-replication-v1"


def minimum_pairs_for_exact_mcnemar_power(
    *,
    observed_fix_rate: float,
    minimum_wins: int,
    target_probability: float,
) -> dict[str, Any]:
    if not 0 < observed_fix_rate < 1:
        raise ValueError("observed fix rate must be in (0, 1)")
    if minimum_wins < 1:
        raise ValueError("minimum wins must be positive")
    if not 0 < target_probability < 1:
        raise ValueError("target probability must be in (0, 1)")
    for pairs in range(minimum_wins, 10_000):
        probability = sum(
            math.comb(pairs, wins)
            * observed_fix_rate**wins
            * (1 - observed_fix_rate) ** (pairs - wins)
            for wins in range(minimum_wins, pairs + 1)
        )
        if probability >= target_probability:
            return {
                "observed_fix_rate": observed_fix_rate,
                "minimum_wins": minimum_wins,
                "target_probability": target_probability,
                "minimum_pairs": pairs,
                "achieved_probability": probability,
            }
    raise ValueError("sample-size search exceeded bounded range")


def _relation_grid(split: str) -> list[tuple[int, int, int]]:
    left_values = [450 + 40 * index for index in range(16)]
    repeated_values = [83 + 8 * index for index in range(12)]
    values = [
        (left, repeated, multiplier)
        for left_index, left in enumerate(left_values)
        for repeated_index, repeated in enumerate(repeated_values)
        for multiplier in (2, 3)
        if (
            (left_index + repeated_index) % 2
            == (0 if split == "train" else 1)
        )
    ]
    if split not in {"train", "dev"}:
        raise ValueError("unknown replication split")
    return values


def _canonical_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )


def _sanitize_forbidden_padding(
    row: dict[str, Any],
    *,
    tokenizer: Any,
) -> dict[str, Any]:
    user = row["messages"][-2]["content"]
    for marker in (
        "benchmark",
        "canary",
        "gpqa",
        "gsm8k",
        "holdout",
        "mmlu",
        "skillbench",
        "swe-bench",
        "terminal-bench",
        "wildclawbench",
    ):
        user = re.sub(
            marker,
            "x" * len(marker),
            user,
            flags=re.IGNORECASE,
        )
    row["messages"][-2]["content"] = user
    row["token_count"] = count_tokens(tokenizer, row["messages"])
    row["exact_hash"] = sha256_text(canonical_json(row["messages"]))
    return row


def _collect_prior_identity(paths: list[Path]) -> dict[str, set[str]]:
    identity = {
        "sample_ids": set(),
        "semantic_hashes": set(),
        "semantic_task_hashes": set(),
        "expressions": set(),
    }
    for path in paths:
        dataset = json.loads(path.read_text(encoding="utf-8"))
        for row in dataset["samples"]:
            identity["sample_ids"].add(str(row["sample_id"]))
            identity["semantic_hashes"].add(str(row["semantic_hash"]))
            identity["semantic_task_hashes"].add(
                str(row.get("semantic_task_hash", ""))
            )
            expression = row.get("task_spec", {}).get("expression")
            if isinstance(expression, str):
                identity["expressions"].add(expression)
    identity["semantic_task_hashes"].discard("")
    return identity


def _collect_prior_ledger_identity(path: Path) -> dict[str, set[str]]:
    identity = {
        "sample_ids": set(),
        "semantic_hashes": set(),
        "expressions": set(),
    }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity["sample_ids"].add(str(row["sample_id"]))
            identity["semantic_hashes"].add(str(row["semantic_hash"]))
            expression = row.get("task_spec", {}).get("expression")
            if isinstance(expression, str):
                identity["expressions"].add(expression)
    return identity


def build_consistency_replication_dataset(
    *,
    tokenizer: Any,
    tokenizer_path: Path,
    prior_dataset_paths: list[Path],
    prior_accepted_jsonl_path: Path,
    source_result_path: Path,
    target_tokens_per_row: int = 620,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = json.loads(source_result_path.read_text(encoding="utf-8"))
    final_comparison = result["evaluation"]["comparisons"]["final"]
    if (
        final_comparison["samples"] != 24
        or final_comparison["wins"] != 1
        or final_comparison["losses"] != 0
    ):
        raise ValueError("consistency v1 source result differs")
    fix_rate = final_comparison["wins"] / final_comparison["samples"]
    sample_size = minimum_pairs_for_exact_mcnemar_power(
        observed_fix_rate=fix_rate,
        minimum_wins=6,
        target_probability=0.8,
    )
    if sample_size["minimum_pairs"] != 189:
        raise ValueError("replication sample-size result differs")
    frozen_pairs = 192

    rows = []
    for split in ("train", "dev"):
        relation_values = _relation_grid(split)
        if len(relation_values) != frozen_pairs:
            raise ValueError("replication relation grid count differs")
        for semantic_index, (left, repeated, multiplier) in enumerate(
            relation_values
        ):
            for view in ("process", "final"):
                row = _relation_row(
                    tokenizer,
                    split=split,
                    semantic_index=semantic_index + (1_000 if split == "train" else 2_000),
                    left=left,
                    repeated=repeated,
                    multiplier=multiplier,
                    view=view,
                    target_tokens=target_tokens_per_row,
                )
                row["schema_version"] = (
                    "nano_consistency_replication_sample_v1"
                )
                row["replication_id"] = DATASET_ID
                row["sample_id"] = "replication-" + sha256_text(
                    canonical_json(
                        {
                            "replication_id": DATASET_ID,
                            "split": split,
                            "view": view,
                            "task_spec": row["task_spec"],
                        }
                    )
                )[:24]
                row["pair_id"] = (
                    f"{split}-replication-{semantic_index:03d}"
                )
                rows.append(
                    _sanitize_forbidden_padding(row, tokenizer=tokenizer)
                )
        json_per_family = 64 if split == "train" else 32
        index_offset = 10_000 if split == "train" else 20_000
        for family_id in JSON_FAMILIES:
            for family_index in range(json_per_family):
                row = _json_row(
                    tokenizer,
                    split=split,
                    family_id=family_id,
                    family_index=index_offset + family_index,
                    target_tokens=target_tokens_per_row,
                )
                row["schema_version"] = (
                    "nano_consistency_replication_sample_v1"
                )
                row["replication_id"] = DATASET_ID
                row["sample_id"] = "replication-" + sha256_text(
                    canonical_json(
                        {
                            "replication_id": DATASET_ID,
                            "split": split,
                            "family_id": family_id,
                            "task_spec": row["task_spec"],
                        }
                    )
                )[:24]
                rows.append(
                    _sanitize_forbidden_padding(row, tokenizer=tokenizer)
                )
    rows.sort(key=lambda row: (row["split"] != "train", row["sample_id"]))

    tokenizer_files = {
        filename: sha256_file(tokenizer_path / filename)
        for filename in (
            "chat_template.jinja",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    }
    dataset = {
        "schema_version": DATASET_SCHEMA,
        "dataset_id": DATASET_ID,
        "source": {
            "kind": "deterministic_synthetic",
            "generator": "nano_data_pipeline.consistency_replication",
            "source_result_sha256": sha256_file(source_result_path),
            "prior_datasets": [
                {
                    "path": path.name,
                    "file_sha256": sha256_file(path),
                }
                for path in prior_dataset_paths
            ],
            "prior_accepted_jsonl_sha256": sha256_file(
                prior_accepted_jsonl_path
            ),
            "benchmark_content_used": False,
            "independent_holdout_used": False,
            "model_outputs_used": False,
            "teacher_outputs_used": False,
        },
        "policy": {
            "training_allowed_after_release_gate": True,
            "dev_training_allowed": False,
            "contains_benchmark_content": False,
            "contains_independent_holdout": False,
            "contains_model_outputs": False,
            "contains_teacher_outputs": False,
            "all_targets_deterministically_verified": True,
            "all_process_intermediates_verified": True,
            "paired_view_final_consistency_required": True,
        },
        "sample_size": {
            **sample_size,
            "frozen_train_pairs": frozen_pairs,
            "frozen_dev_pairs": frozen_pairs,
            "exact_mcnemar_alpha": 0.05,
            "minimum_wins_for_zero_losses": 6,
            "rounding": "189 rounded up to 192 for a 16x12 grid per split",
        },
        "significance_gates": {
            "aggregate_bootstrap_ci_lower_gt_zero": True,
            "aggregate_mcnemar_p_lt": 0.05,
            "final_bootstrap_ci_lower_gt_zero": True,
            "final_mcnemar_p_lt": 0.05,
            "pair_bootstrap_ci_lower_gt_zero": True,
            "pair_mcnemar_p_lt": 0.05,
            "minimum_final_only_wins": 6,
            "maximum_final_only_losses": 0,
            "json_family_non_regression": True,
        },
        "objective_frozen": {
            "process_ce_weight": 0.5,
            "final_ce_weight": 0.5,
            "consistency_weight": 1.0,
            "temperature": 1.0,
            "teacher_detach": True,
        },
        "token_accounting": {
            "unit": "qwen3.5_tokenizer_input_id",
            "enable_thinking": False,
            "add_generation_prompt": False,
            "counted_split": "train",
            "file_sha256": tokenizer_files,
        },
        "samples": rows,
    }
    release = validate_consistency_replication_dataset(
        dataset,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        prior_dataset_paths=prior_dataset_paths,
        prior_accepted_jsonl_path=prior_accepted_jsonl_path,
        source_result_path=source_result_path,
    )
    return dataset, release


def validate_consistency_replication_dataset(
    dataset: dict[str, Any],
    *,
    tokenizer: Any,
    tokenizer_path: Path,
    prior_dataset_paths: list[Path],
    prior_accepted_jsonl_path: Path,
    source_result_path: Path,
) -> dict[str, Any]:
    if dataset.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("unsupported consistency replication schema")
    rows = dataset.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("consistency replication contains no rows")
    prior = _collect_prior_identity(prior_dataset_paths)
    prior_ledger = _collect_prior_ledger_identity(
        prior_accepted_jsonl_path
    )
    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    ids = [row["sample_id"] for row in rows]
    exact_hashes = [row["exact_hash"] for row in rows]
    semantic_hashes = [row["semantic_hash"] for row in rows]
    task_hashes = [row["semantic_task_hash"] for row in rows]
    train_semantic = {row["semantic_hash"] for row in train}
    dev_semantic = {row["semantic_hash"] for row in dev}
    train_task = {row["semantic_task_hash"] for row in train}
    dev_task = {row["semantic_task_hash"] for row in dev}
    expressions = {
        row["task_spec"]["expression"]
        for row in rows
        if row["view"] in {"process", "final"}
    }
    process_verified = True
    final_verified = True
    json_verified = True
    paired_consistency = True
    token_pass = True
    hash_pass = True
    answer_leakage = []
    forbidden = []
    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        messages = row["messages"]
        token_pass = token_pass and row["token_count"] == count_tokens(
            tokenizer,
            messages,
        )
        hash_pass = (
            hash_pass
            and row["exact_hash"] == sha256_text(canonical_json(messages))
            and row["semantic_hash"]
            == sha256_text(
                semantic_basis(row["task_family"], row["task_spec"])
            )
        )
        if row["view"] == "process":
            process_verified = process_verified and _verify_process(row)
            pairs.setdefault(row["pair_id"], []).append(row)
        elif row["view"] == "final":
            final_verified = final_verified and _verify_final(row)
            pairs.setdefault(row["pair_id"], []).append(row)
        elif row["view"] == "json_preservation":
            passed, _ = verify_candidate(
                {
                    "messages": messages,
                    "task_spec": row["task_spec"],
                    "verifier": row["verifier"],
                }
            )
            json_verified = json_verified and passed
        else:
            raise ValueError("unknown consistency replication view")
        if row["view"] in {"process", "final"}:
            expected = row["messages"][-1]["content"].splitlines()[-1].split(
                ": ",
                1,
            )[1]
            if re.search(
                rf"(?<![0-9]){re.escape(expected)}(?![0-9])",
                row["messages"][-2]["content"],
            ):
                answer_leakage.append(row["sample_id"])
        serialized = canonical_json(
            {"messages": messages, "task_spec": row["task_spec"]}
        ).lower()
        if any(
            marker in serialized
            for marker in (
                "benchmark",
                "canary",
                "gpqa",
                "gsm8k",
                "holdout",
                "mmlu",
                "skillbench",
                "swe-bench",
                "terminal-bench",
                "wildclawbench",
            )
        ):
            forbidden.append(row["sample_id"])
    for pair_rows in pairs.values():
        if {row["view"] for row in pair_rows} != {"process", "final"}:
            paired_consistency = False
            continue
        outputs = {
            row["view"]: row["messages"][-1]["content"] for row in pair_rows
        }
        paired_consistency = (
            paired_consistency
            and outputs["process"].splitlines()[-1] == outputs["final"]
        )
    train_json = Counter(
        row["task_family"]
        for row in train
        if row["view"] == "json_preservation"
    )
    dev_json = Counter(
        row["task_family"]
        for row in dev
        if row["view"] == "json_preservation"
    )
    train_pairs = {
        row["pair_id"] for row in train if row["pair_id"] is not None
    }
    dev_pairs = {
        row["pair_id"] for row in dev if row["pair_id"] is not None
    }
    tokenizer_expected = dataset["token_accounting"]["file_sha256"]
    tokenizer_identity = all(
        sha256_file(tokenizer_path / filename) == digest
        for filename, digest in tokenizer_expected.items()
    )
    checks = {
        "train_rows_pass": len(train) == 640,
        "dev_rows_pass": len(dev) == 512,
        "train_pairs_pass": len(train_pairs) == 192,
        "dev_pairs_pass": len(dev_pairs) == 192,
        "train_json_family_pass": all(
            train_json[family] == 64 for family in JSON_FAMILIES
        ),
        "dev_json_family_pass": all(
            dev_json[family] == 32 for family in JSON_FAMILIES
        ),
        "sample_id_unique_pass": len(ids) == len(set(ids)),
        "exact_hash_unique_pass": len(exact_hashes)
        == len(set(exact_hashes)),
        "semantic_hash_unique_pass": len(semantic_hashes)
        == len(set(semantic_hashes)),
        "semantic_task_hash_pairing_pass": (
            len(task_hashes) - len(set(task_hashes)) == 384
        ),
        "train_dev_semantic_overlap_pass": not (
            train_semantic & dev_semantic
        ),
        "train_dev_task_overlap_pass": not (train_task & dev_task),
        "prior_sample_id_overlap_pass": not (
            set(ids) & prior["sample_ids"]
        ),
        "prior_semantic_overlap_pass": not (
            set(semantic_hashes) & prior["semantic_hashes"]
        ),
        "prior_task_overlap_pass": not (
            set(task_hashes) & prior["semantic_task_hashes"]
        ),
        "prior_expression_overlap_pass": not (
            expressions & prior["expressions"]
        ),
        "prior_ledger_sample_id_overlap_pass": not (
            set(ids) & prior_ledger["sample_ids"]
        ),
        "prior_ledger_semantic_overlap_pass": not (
            set(semantic_hashes) & prior_ledger["semantic_hashes"]
        ),
        "prior_ledger_expression_overlap_pass": not (
            expressions & prior_ledger["expressions"]
        ),
        "process_verifier_pass": process_verified,
        "final_verifier_pass": final_verified,
        "json_verifier_pass": json_verified,
        "paired_consistency_pass": paired_consistency,
        "token_accounting_pass": token_pass,
        "hash_recomputation_pass": hash_pass,
        "tokenizer_identity_pass": tokenizer_identity,
        "answer_leakage_pass": not answer_leakage,
        "forbidden_content_pass": not forbidden,
        "sample_size_pass": (
            dataset["sample_size"]["minimum_pairs"] == 189
            and dataset["sample_size"]["frozen_dev_pairs"] == 192
        ),
        "source_result_identity_pass": (
            dataset["source"]["source_result_sha256"]
            == sha256_file(source_result_path)
        ),
    }
    train_tokens = sum(row["token_count"] for row in train)
    release = {
        "schema_version": RELEASE_SCHEMA,
        "release_id": DATASET_ID,
        "dataset_schema": DATASET_SCHEMA,
        "source": {
            "source_result_sha256": sha256_file(source_result_path),
            "prior_dataset_file_sha256": [
                sha256_file(path) for path in prior_dataset_paths
            ],
            "prior_accepted_jsonl_sha256": sha256_file(
                prior_accepted_jsonl_path
            ),
            "dataset_canonical_sha256": _canonical_sha256(dataset),
            "tokenizer_file_sha256": tokenizer_expected,
        },
        "sample_size": dataset["sample_size"],
        "significance_gates": dataset["significance_gates"],
        "objective_frozen": dataset["objective_frozen"],
        "accepted": {
            "rows": len(rows),
            "train_rows": len(train),
            "dev_rows": len(dev),
            "train_pairs": len(train_pairs),
            "dev_pairs": len(dev_pairs),
            "train_json_by_family": dict(sorted(train_json.items())),
            "dev_json_by_family": dict(sorted(dev_json.items())),
            "train_tokens": train_tokens,
        },
        "overlap": {
            "train_dev_semantic": len(train_semantic & dev_semantic),
            "train_dev_task": len(train_task & dev_task),
            "prior_sample_id": len(set(ids) & prior["sample_ids"]),
            "prior_semantic": len(
                set(semantic_hashes) & prior["semantic_hashes"]
            ),
            "prior_task": len(
                set(task_hashes) & prior["semantic_task_hashes"]
            ),
            "prior_expression": len(expressions & prior["expressions"]),
            "prior_ledger_sample_id": len(
                set(ids) & prior_ledger["sample_ids"]
            ),
            "prior_ledger_semantic": len(
                set(semantic_hashes) & prior_ledger["semantic_hashes"]
            ),
            "prior_ledger_expression": len(
                expressions & prior_ledger["expressions"]
            ),
        },
        "leakage": {
            "answer_value_rows": answer_leakage,
            "forbidden_content_rows": forbidden,
        },
        "checks": checks,
        "training_unblocked": all(checks.values()),
        "claim_boundary": (
            "这份 release 只证明更大的 fresh replication data 和 significance "
            "contract 通过全部检查。它不证明模型能力提升，也不允许访问 "
            "benchmark/holdout 或启动 RL。"
        ),
    }
    return release
