from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from nano_data_pipeline.analog import evaluate_arithmetic, format_number
from nano_data_pipeline.feedback import sha256_file
from nano_data_pipeline.openai_subagent import (
    FAMILY_VERIFIERS,
    _base_task,
    solve_compiled_task,
)
from nano_data_pipeline.subagent_campaign import (
    canonical_json,
    count_tokens,
    semantic_basis,
    sha256_text,
    verify_candidate,
)


DATASET_SCHEMA = "nano_execution_target_dataset_v1"
RELEASE_SCHEMA = "nano_execution_target_release_v1"
DATASET_ID = "skill-sft-execution-target-paired-v1"
JSON_FAMILIES = (
    "coding-and-validation",
    "planning-and-state",
    "skill-routing-and-reflection",
    "tool-use-and-recovery",
)
FORBIDDEN_MARKERS = (
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


def _letters(value: str, length: int = 20) -> str:
    number = int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)
    result = []
    for _ in range(length):
        number, remainder = divmod(number, 26)
        result.append(chr(ord("a") + remainder))
    return "".join(result)


def _pad_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    identity: str,
    target_tokens: int,
) -> list[dict[str, str]]:
    if [message["role"] for message in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError("execution-target messages must have three roles")
    system, user, assistant = messages

    def build(lines: int) -> list[dict[str, str]]:
        context = [
            (
                "Synthetic context "
                f"{_letters(f'{identity}:{line}')} is unrelated unless the "
                "task contract explicitly references it."
            )
            for line in range(lines)
        ]
        content = user["content"]
        if context:
            content += "\n" + "\n".join(context)
        return [
            dict(system),
            {"role": "user", "content": content},
            dict(assistant),
        ]

    base = build(0)
    if count_tokens(tokenizer, base) >= target_tokens:
        return base
    lower = 0
    upper = 8
    while count_tokens(tokenizer, build(upper)) < target_tokens:
        lower = upper
        upper *= 2
        if upper > 4_096:
            raise ValueError("unable to reach execution-target token floor")
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if count_tokens(tokenizer, build(middle)) >= target_tokens:
            upper = middle
        else:
            lower = middle
    return build(upper)


def _relation_values(split: str) -> list[tuple[int, int, int]]:
    if split == "train":
        left_values = (130, 170, 210, 250, 290, 330, 370, 410)
        repeated_values = (23, 31, 39, 47, 55, 63, 71, 79)
    elif split == "dev":
        left_values = (145, 225, 305, 385)
        repeated_values = (27, 43, 59)
    else:
        raise ValueError("unknown execution-target split")
    return [
        (left, repeated, multiplier)
        for left in left_values
        for repeated in repeated_values
        for multiplier in (2, 3)
    ]


def _relation_messages(
    expression: str,
    *,
    view: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    left, repeated, multiplier, repeated_again = [
        int(value)
        for value in re.fullmatch(
            r"\(([0-9]+) \+ ([0-9]+)\) \* ([0-9]+) - ([0-9]+)",
            expression,
        ).groups()
    ]
    if repeated != repeated_again:
        raise ValueError("execution-target relation operand is not repeated")
    first = left + repeated
    second = first * multiplier
    final = second - repeated
    if view == "process":
        assistant = (
            f"STEP 1: {left} + {repeated} = {first}\n"
            f"STEP 2: {first} * {multiplier} = {second}\n"
            f"STEP 3: {second} - {repeated} = {final}\n"
            f"FINAL: {final}"
        )
        user = (
            f"Compute {expression}. Execute exactly one operation per STEP, "
            "reuse each verified intermediate result in the next STEP, and "
            "return three STEP lines followed by FINAL."
        )
        verifier = {
            "kind": "safe_ast_arithmetic_process_v2",
            "source_expression": expression,
            "steps": [
                {
                    "expression": f"{left} + {repeated}",
                    "expected_result": str(first),
                },
                {
                    "expression": f"{first} * {multiplier}",
                    "expected_result": str(second),
                },
                {
                    "expression": f"{second} - {repeated}",
                    "expected_result": str(final),
                },
            ],
            "expected_result": str(final),
        }
    elif view == "final":
        assistant = f"FINAL: {final}"
        user = (
            f"Compute {expression}. Return exactly one line: FINAL: <number>."
        )
        verifier = {"kind": "safe_execution_receipt_v1"}
    else:
        raise ValueError("unknown execution-target paired view")
    messages = [
        {
            "role": "system",
            "content": (
                "Use only the synthetic arithmetic contract. Calculate the "
                "final value rather than copying a nearby number."
            ),
        },
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return messages, verifier


def _relation_row(
    tokenizer: Any,
    *,
    split: str,
    semantic_index: int,
    left: int,
    repeated: int,
    multiplier: int,
    view: str,
    target_tokens: int,
) -> dict[str, Any]:
    expression = f"({left} + {repeated}) * {multiplier} - {repeated}"
    expected = format_number(evaluate_arithmetic(expression))
    semantic_task = {
        "expression": expression,
        "left": left,
        "multiplier": multiplier,
        "relation": "repeat_subtrahend",
        "repeated_operand": repeated,
    }
    task_spec = {**semantic_task, "view": view}
    messages, verifier = _relation_messages(expression, view=view)
    messages = _pad_messages(
        tokenizer,
        messages,
        identity=f"relation:{split}:{semantic_index}:{view}",
        target_tokens=target_tokens,
    )
    user = messages[-2]["content"]
    if re.search(
        rf"(?<![0-9]){re.escape(expected)}(?![0-9])",
        user,
    ):
        raise ValueError("execution-target final answer leaked into prompt")
    family_id = f"execution-target-{view}"
    semantic_hash = sha256_text(semantic_basis(family_id, task_spec))
    semantic_task_hash = sha256_text(canonical_json(semantic_task))
    sample_id = "execution-" + sha256_text(
        canonical_json(
            {
                "dataset": DATASET_ID,
                "semantic_task": semantic_task,
                "split": split,
                "view": view,
            }
        )
    )[:24]
    return {
        "schema_version": "nano_execution_target_sample_v1",
        "sample_id": sample_id,
        "split": split,
        "training_eligible": split == "train",
        "task_family": family_id,
        "format_family": (
            "process_trace_numeric" if view == "process" else "final_numeric"
        ),
        "view": view,
        "pair_id": f"{split}-relation-{semantic_index:03d}",
        "source_kind": "deterministic_synthetic",
        "messages": messages,
        "task_spec": task_spec,
        "verifier": verifier,
        "token_count": count_tokens(tokenizer, messages),
        "exact_hash": sha256_text(canonical_json(messages)),
        "semantic_hash": semantic_hash,
        "semantic_task_hash": semantic_task_hash,
        "semantic_basis_version": "family_task_spec_v1",
    }


def _json_row(
    tokenizer: Any,
    *,
    split: str,
    family_id: str,
    family_index: int,
    target_tokens: int,
) -> dict[str, Any]:
    seed = 2_026_081_901 if split == "train" else 2_026_081_902
    base = _base_task(family_id, seed, family_index)
    assistant = solve_compiled_task(family_id, base["task_spec"])
    messages = [
        {
            "role": "system",
            "content": (
                "Follow the synthetic JSON contract exactly and preserve all "
                "required fields. Return one JSON object only."
            ),
        },
        {"role": "user", "content": base["user"]},
        {"role": "assistant", "content": assistant},
    ]
    messages = _pad_messages(
        tokenizer,
        messages,
        identity=f"json:{split}:{family_id}:{family_index}",
        target_tokens=target_tokens,
    )
    task_spec = base["task_spec"]
    verifier = {"kind": FAMILY_VERIFIERS[family_id]}
    verified, _ = verify_candidate(
        {
            "messages": messages,
            "task_spec": task_spec,
            "verifier": verifier,
        }
    )
    if not verified:
        raise ValueError(f"generated JSON preservation row failed: {family_id}")
    sample_id = "execution-" + sha256_text(
        canonical_json(
            {
                "dataset": DATASET_ID,
                "family_id": family_id,
                "family_index": family_index,
                "split": split,
                "task_spec": task_spec,
            }
        )
    )[:24]
    return {
        "schema_version": "nano_execution_target_sample_v1",
        "sample_id": sample_id,
        "split": split,
        "training_eligible": split == "train",
        "task_family": family_id,
        "format_family": "skill_release_json",
        "view": "json_preservation",
        "pair_id": None,
        "source_kind": "deterministic_synthetic",
        "messages": messages,
        "task_spec": task_spec,
        "verifier": verifier,
        "token_count": count_tokens(tokenizer, messages),
        "exact_hash": sha256_text(canonical_json(messages)),
        "semantic_hash": sha256_text(semantic_basis(family_id, task_spec)),
        "semantic_task_hash": sha256_text(
            canonical_json({"family_id": family_id, "task_spec": task_spec})
        ),
        "semantic_basis_version": "family_task_spec_v1",
    }


def _verify_process(row: dict[str, Any]) -> bool:
    verifier = row["verifier"]
    lines = row["messages"][-1]["content"].splitlines()
    steps = verifier.get("steps")
    if (
        verifier.get("kind") != "safe_ast_arithmetic_process_v2"
        or not isinstance(steps, list)
        or len(lines) != len(steps) + 1
    ):
        return False
    previous = None
    for index, (line, step) in enumerate(zip(lines[:-1], steps), start=1):
        match = re.fullmatch(
            rf"STEP {index}: (.+) = ([-+]?[0-9]+(?:\.[0-9]+)?)",
            line,
        )
        if match is None:
            return False
        expression, result = match.groups()
        if (
            expression != step["expression"]
            or result != step["expected_result"]
            or format_number(evaluate_arithmetic(expression)) != result
            or (
                previous is not None
                and previous not in expression.split()
            )
        ):
            return False
        previous = result
    final = re.fullmatch(r"FINAL: ([-+]?[0-9]+(?:\.[0-9]+)?)", lines[-1])
    return (
        final is not None
        and previous == verifier.get("expected_result")
        and final.group(1) == verifier.get("expected_result")
        and format_number(evaluate_arithmetic(verifier["source_expression"]))
        == verifier.get("expected_result")
    )


def _verify_final(row: dict[str, Any]) -> bool:
    assistant = row["messages"][-1]["content"]
    expression = row["task_spec"]["expression"]
    return assistant == f"FINAL: {format_number(evaluate_arithmetic(expression))}"


def _prior_release_identity(
    accepted_jsonl_path: Path,
) -> tuple[set[str], set[str], set[str]]:
    semantic_hashes = set()
    expressions = set()
    sample_ids = set()
    with accepted_jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            semantic_hashes.add(str(row["semantic_hash"]))
            sample_ids.add(str(row["sample_id"]))
            expression = row.get("task_spec", {}).get("expression")
            if isinstance(expression, str):
                expressions.add(expression)
    return semantic_hashes, expressions, sample_ids


def build_execution_target_dataset(
    *,
    tokenizer: Any,
    accepted_jsonl_path: Path,
    release_manifest_path: Path,
    audit_path: Path,
    tokenizer_path: Path,
    target_tokens_per_row: int = 620,
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    contract = audit["minimum_fresh_data_contract"]
    prior_release = json.loads(
        release_manifest_path.read_text(encoding="utf-8")
    )
    expected_accepted_sha = prior_release["artifacts"][
        "accepted_jsonl_sha256"
    ]
    if sha256_file(accepted_jsonl_path) != expected_accepted_sha:
        raise ValueError("prior release accepted ledger identity mismatch")

    rows = []
    for split in ("train", "dev"):
        relation_values = _relation_values(split)
        for semantic_index, (left, repeated, multiplier) in enumerate(
            relation_values
        ):
            for view in ("process", "final"):
                rows.append(
                    _relation_row(
                        tokenizer,
                        split=split,
                        semantic_index=semantic_index,
                        left=left,
                        repeated=repeated,
                        multiplier=multiplier,
                        view=view,
                        target_tokens=target_tokens_per_row,
                    )
                )
        family_rows = 64 if split == "train" else 8
        for family_id in JSON_FAMILIES:
            for family_index in range(family_rows):
                rows.append(
                    _json_row(
                        tokenizer,
                        split=split,
                        family_id=family_id,
                        family_index=family_index,
                        target_tokens=target_tokens_per_row,
                    )
                )
    rows.sort(key=lambda row: (row["split"] != "train", row["sample_id"]))

    tokenizer_files = {}
    for filename in (
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        tokenizer_files[filename] = sha256_file(tokenizer_path / filename)
    dataset = {
        "schema_version": DATASET_SCHEMA,
        "dataset_id": DATASET_ID,
        "source": {
            "kind": "deterministic_synthetic",
            "generator": "nano_data_pipeline.execution_target_dataset",
            "coverage_audit_sha256": sha256_file(audit_path),
            "prior_release_id": prior_release["release_id"],
            "prior_release_manifest_sha256": sha256_file(
                release_manifest_path
            ),
            "prior_accepted_jsonl_sha256": expected_accepted_sha,
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
        "token_accounting": {
            "unit": "qwen3.5_tokenizer_input_id",
            "enable_thinking": False,
            "add_generation_prompt": False,
            "counted_split": "train",
            "tokenizer_path_reference": "../../../models/Qwen3.5-4B",
            "file_sha256": tokenizer_files,
        },
        "contract": contract,
        "samples": rows,
    }
    release = validate_execution_target_dataset(
        dataset,
        accepted_jsonl_path=accepted_jsonl_path,
        release_manifest_path=release_manifest_path,
        audit_path=audit_path,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
    )
    return dataset, release


def validate_execution_target_dataset(
    dataset: dict[str, Any],
    *,
    accepted_jsonl_path: Path,
    release_manifest_path: Path,
    audit_path: Path,
    tokenizer: Any | None = None,
    tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    if dataset.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("unsupported execution-target dataset schema")
    rows = dataset.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("execution-target dataset contains no rows")
    contract = dataset.get("contract", {})
    prior_semantic, prior_expressions, prior_ids = _prior_release_identity(
        accepted_jsonl_path
    )
    ids = [row["sample_id"] for row in rows]
    exact_hashes = [row["exact_hash"] for row in rows]
    semantic_hashes = [row["semantic_hash"] for row in rows]
    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    train_semantic = {row["semantic_hash"] for row in train}
    dev_semantic = {row["semantic_hash"] for row in dev}
    train_task_semantic = {row["semantic_task_hash"] for row in train}
    dev_task_semantic = {row["semantic_task_hash"] for row in dev}

    verifier_pass = True
    hash_pass = True
    token_pass = True
    answer_leakage = []
    forbidden_rows = []
    for row in rows:
        messages = row["messages"]
        hash_pass = hash_pass and row["exact_hash"] == sha256_text(
            canonical_json(messages)
        )
        hash_pass = hash_pass and row["semantic_hash"] == sha256_text(
            semantic_basis(row["task_family"], row["task_spec"])
        )
        token_pass = token_pass and row["token_count"] > 0
        if tokenizer is not None:
            token_pass = token_pass and row["token_count"] == count_tokens(
                tokenizer,
                messages,
            )
        if row["view"] == "process":
            verifier_pass = verifier_pass and _verify_process(row)
        elif row["view"] == "final":
            verifier_pass = verifier_pass and _verify_final(row)
        elif row["view"] == "json_preservation":
            passed, _ = verify_candidate(
                {
                    "messages": messages,
                    "task_spec": row["task_spec"],
                    "verifier": row["verifier"],
                }
            )
            verifier_pass = verifier_pass and passed
        else:
            verifier_pass = False
        if row["view"] in {"process", "final"}:
            expected = format_number(
                evaluate_arithmetic(row["task_spec"]["expression"])
            )
            if re.search(
                rf"(?<![0-9]){re.escape(expected)}(?![0-9])",
                messages[-2]["content"],
            ):
                answer_leakage.append(row["sample_id"])
        serialized = canonical_json(
            {
                "messages": messages,
                "task_spec": row["task_spec"],
            }
        ).lower()
        if any(marker in serialized for marker in FORBIDDEN_MARKERS):
            forbidden_rows.append(row["sample_id"])

    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["pair_id"]:
            pairs.setdefault(row["pair_id"], []).append(row)
    paired_consistency = True
    for pair_rows in pairs.values():
        if {row["view"] for row in pair_rows} != {"process", "final"}:
            paired_consistency = False
            continue
        outputs = {
            row["view"]: row["messages"][-1]["content"] for row in pair_rows
        }
        process_final = outputs["process"].splitlines()[-1]
        paired_consistency = (
            paired_consistency and process_final == outputs["final"]
        )

    train_family = Counter(
        row["task_family"]
        for row in train
        if row["view"] == "json_preservation"
    )
    dev_family = Counter(
        row["task_family"]
        for row in dev
        if row["view"] == "json_preservation"
    )
    train_relation = [
        row for row in train if row["view"] in {"process", "final"}
    ]
    dev_relation = [
        row for row in dev if row["view"] in {"process", "final"}
    ]
    train_tokens = sum(row["token_count"] for row in train)
    prior_expression_overlap = sum(
        row["task_spec"].get("expression") in prior_expressions
        for row in rows
        if row["view"] in {"process", "final"}
    )
    checks = {
        "train_rows_pass": len(train) == contract["train_rows"],
        "dev_rows_pass": len(dev) == contract["dev_rows"],
        "train_tokens_pass": (
            train_tokens >= contract["minimum_train_tokens"]
        ),
        "train_relation_views_pass": len(train_relation) == 256,
        "dev_relation_views_pass": len(dev_relation) == 48,
        "train_json_rows_pass": sum(train_family.values()) == 256,
        "dev_json_rows_pass": sum(dev_family.values()) == 32,
        "train_json_family_quotas_pass": all(
            train_family[family] == 64 for family in JSON_FAMILIES
        ),
        "dev_json_family_quotas_pass": all(
            dev_family[family] == 8 for family in JSON_FAMILIES
        ),
        "sample_id_unique_pass": len(ids) == len(set(ids)),
        "exact_hash_unique_pass": len(exact_hashes)
        == len(set(exact_hashes)),
        "semantic_hash_unique_pass": len(semantic_hashes)
        == len(set(semantic_hashes)),
        "train_dev_semantic_overlap_pass": not (
            train_semantic & dev_semantic
        ),
        "train_dev_task_overlap_pass": not (
            train_task_semantic & dev_task_semantic
        ),
        "prior_release_semantic_overlap_pass": not (
            set(semantic_hashes) & prior_semantic
        ),
        "prior_release_expression_overlap_pass": (
            prior_expression_overlap == 0
        ),
        "prior_release_sample_id_overlap_pass": not (
            set(ids) & prior_ids
        ),
        "deterministic_verifier_pass": verifier_pass,
        "paired_view_consistency_pass": paired_consistency,
        "answer_value_leakage_pass": not answer_leakage,
        "forbidden_content_pass": not forbidden_rows,
        "hash_recomputation_pass": hash_pass,
        "token_accounting_pass": token_pass,
    }
    tokenizer_identity = True
    if tokenizer_path is not None:
        expected_files = dataset.get("token_accounting", {}).get(
            "file_sha256",
            {},
        )
        tokenizer_identity = (
            isinstance(expected_files, dict)
            and len(expected_files) == 3
            and all(
                (tokenizer_path / filename).is_file()
                and sha256_file(tokenizer_path / filename) == digest
                for filename, digest in expected_files.items()
            )
        )
    checks["tokenizer_identity_pass"] = tokenizer_identity
    training_unblocked = all(checks.values())
    dataset_sha256 = sha256_text(
        json.dumps(
            dataset,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )
    release = {
        "schema_version": RELEASE_SCHEMA,
        "release_id": DATASET_ID,
        "dataset_schema": DATASET_SCHEMA,
        "source": {
            "coverage_audit_sha256": sha256_file(audit_path),
            "prior_release_manifest_sha256": sha256_file(
                release_manifest_path
            ),
            "prior_accepted_jsonl_sha256": sha256_file(
                accepted_jsonl_path
            ),
            "dataset_canonical_sha256": dataset_sha256,
            "tokenizer_file_sha256": dataset["token_accounting"][
                "file_sha256"
            ],
        },
        "accepted": {
            "rows": len(rows),
            "train_rows": len(train),
            "dev_rows": len(dev),
            "train_tokens": train_tokens,
            "train_relation_views": len(train_relation),
            "dev_relation_views": len(dev_relation),
            "train_json_rows": sum(train_family.values()),
            "dev_json_rows": sum(dev_family.values()),
            "train_json_by_family": dict(sorted(train_family.items())),
            "dev_json_by_family": dict(sorted(dev_family.items())),
        },
        "overlap": {
            "train_dev_semantic": len(train_semantic & dev_semantic),
            "train_dev_task_semantic": len(
                train_task_semantic & dev_task_semantic
            ),
            "prior_release_semantic": len(
                set(semantic_hashes) & prior_semantic
            ),
            "prior_release_expression": prior_expression_overlap,
            "prior_release_sample_id": len(set(ids) & prior_ids),
        },
        "leakage": {
            "answer_value_rows": answer_leakage,
            "forbidden_content_rows": forbidden_rows,
        },
        "checks": checks,
        "training_unblocked": training_unblocked,
        "claim_boundary": (
            "这份 release 只证明冻结的 synthetic data contract 通过了行数、"
            "token、paired view、verifier、provenance、overlap 和 leakage "
            "检查。它不证明模型能力或 benchmark 指标提升。"
        ),
    }
    return release
