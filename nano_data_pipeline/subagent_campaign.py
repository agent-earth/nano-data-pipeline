from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from nano_data_pipeline.analog import evaluate_arithmetic, format_number
from nano_data_pipeline.campaign import load_skill_sft_campaign
from nano_data_pipeline.feedback import sha256_file


PLAN_SCHEMA = "nano_subagent_campaign_plan_v1"
CANDIDATE_SCHEMA = "nano_subagent_candidate_v1"
CRITIC_SCHEMA = "nano_subagent_critic_v1"
ACCEPTED_SCHEMA = "nano_skill_sft_sample_v1"
ALLOWED_SOURCE_KINDS = {
    "license_compatible_non_evaluation",
    "procedurally_generated_synthetic",
    "synthetic_development_failure_summary",
}
SUPPORTED_VERIFIERS = {
    "patch_test_receipt_v1",
    "safe_execution_receipt_v1",
    "skill_route_receipt_v1",
    "state_plan_consistency_v1",
    "tool_trace_contract_v1",
}
PLACEHOLDERS = {
    "attempt",
    "family_id",
    "input",
    "output",
    "shard_id",
    "skill_path",
}
FORBIDDEN_PAYLOAD_MARKERS = (
    "clawbench",
    "gpqa",
    "gsm8k",
    "mmlu",
    "skillbench",
    "swe-bench",
    "terminal-bench",
    "wildclawbench",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_messages(messages: list[dict[str, str]]) -> str:
    normalized = []
    for message in messages:
        content = re.sub(r"\s+", " ", message["content"]).strip().lower()
        normalized.append(f"{message['role']}:{content}")
    return "\n".join(normalized)


def load_command(path: str | Path) -> list[str]:
    command = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(argument, str) or not argument for argument in command)
    ):
        raise ValueError("subagent command JSON must be a non-empty string array")
    unknown = set()
    for argument in command:
        unknown.update(
            field
            for _, field, _, _ in _FORMATTER.parse(argument)
            if field and field not in PLACEHOLDERS
        )
    if unknown:
        raise ValueError(f"unknown subagent command placeholders: {sorted(unknown)}")
    return command


class _Formatter:
    def parse(self, value: str):
        import string

        return string.Formatter().parse(value)


_FORMATTER = _Formatter()


def render_command(command: list[str], values: dict[str, Any]) -> list[str]:
    return [argument.format_map(values) for argument in command]


def plan_campaign(
    campaign_path: str | Path,
    *,
    skill_path: str | Path,
    max_shards: int | None = None,
    candidate_samples_override: int | None = None,
) -> dict[str, Any]:
    campaign = load_skill_sft_campaign(campaign_path)
    skill = Path(skill_path).resolve()
    if not skill.is_file():
        raise ValueError("skill path does not exist")
    initial_shards = campaign["sharding"]["initial_shards"]
    mode = "production"
    if max_shards is not None:
        if max_shards <= 0 or max_shards > initial_shards:
            raise ValueError("max_shards must be in the initial shard range")
        initial_shards = max_shards
        mode = "smoke"
    samples_per_shard = campaign["sharding"]["candidate_samples_per_shard"]
    if candidate_samples_override is not None:
        if candidate_samples_override <= 0:
            raise ValueError("candidate sample override must be positive")
        samples_per_shard = candidate_samples_override
        mode = "smoke"

    families = campaign["data_families"]
    family_weights = {
        family["family_id"]: family["accepted_train_samples_min"]
        for family in families
    }
    shard_counts = allocate_integer_total(initial_shards, family_weights)
    token_weights = {
        family["family_id"]: family["accepted_train_tokens_min"]
        for family in families
    }
    family_token_budget = allocate_integer_total(
        campaign["sharding"]["initial_candidate_tokens_min"],
        token_weights,
    )
    shards = []
    shard_id = 0
    for family in families:
        family_id = family["family_id"]
        count = shard_counts[family_id]
        if count == 0:
            continue
        token_splits = split_integer(family_token_budget[family_id], count)
        for family_shard_id in range(count):
            shards.append(
                {
                    "shard_id": shard_id,
                    "family_shard_id": family_shard_id,
                    "family_id": family_id,
                    "attempt": 0,
                    "candidate_samples": samples_per_shard,
                    "candidate_tokens_min": token_splits[family_shard_id],
                    "seed": 202608190000 + shard_id,
                }
            )
            shard_id += 1
    plan = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": sha256_file(Path(campaign_path)),
        "mode": mode,
        "skill_path": str(skill),
        "skill_sha256": sha256_file(skill),
        "max_parallel_subagents": campaign["sharding"]["max_parallel_subagents"],
        "shards": shards,
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported subagent campaign plan")
    if plan.get("mode") not in {"production", "refill", "smoke"}:
        raise ValueError("invalid campaign plan mode")
    if len(str(plan.get("campaign_sha256", ""))) != 64:
        raise ValueError("plan must pin the campaign SHA256")
    if len(str(plan.get("skill_sha256", ""))) != 64:
        raise ValueError("plan must pin the skill SHA256")
    shards = plan.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("campaign plan must contain shards")
    shard_ids = [shard.get("shard_id") for shard in shards]
    if len(shard_ids) != len(set(shard_ids)):
        raise ValueError("campaign shard IDs must be unique")
    for shard in shards:
        for key in (
            "attempt",
            "candidate_samples",
            "candidate_tokens_min",
            "family_id",
            "seed",
            "shard_id",
        ):
            if key not in shard:
                raise ValueError(f"campaign shard is missing {key}")


def allocate_integer_total(
    total: int,
    weights: dict[str, int],
) -> dict[str, int]:
    if total < 0 or not weights or any(weight <= 0 for weight in weights.values()):
        raise ValueError("integer allocation needs non-negative total and weights")
    weight_sum = sum(weights.values())
    raw = {key: total * weight / weight_sum for key, weight in weights.items()}
    allocated = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(allocated.values())
    order = sorted(weights, key=lambda key: (-(raw[key] - allocated[key]), key))
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def split_integer(total: int, parts: int) -> list[int]:
    if total < 0 or parts <= 0:
        raise ValueError("invalid integer split")
    quotient, remainder = divmod(total, parts)
    return [quotient + (index < remainder) for index in range(parts)]


def write_plan(plan: dict[str, Any], run_dir: str | Path) -> Path:
    validate_plan(plan)
    output = Path(run_dir) / "plan.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        current = json.loads(output.read_text(encoding="utf-8"))
        if current != plan:
            raise ValueError("existing campaign plan differs from requested plan")
        return output
    output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def run_plan(
    campaign_path: str | Path,
    run_dir: str | Path,
    *,
    generator_command: list[str],
    critic_command: list[str],
    tokenizer: Any,
    plan_path: str | Path | None = None,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    campaign = load_skill_sft_campaign(campaign_path)
    root = Path(run_dir)
    selected_plan_path = Path(plan_path) if plan_path else root / "plan.json"
    plan = json.loads(selected_plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    if plan["campaign_sha256"] != sha256_file(Path(campaign_path)):
        raise ValueError("campaign changed after shard planning")
    if plan["skill_sha256"] != sha256_file(Path(plan["skill_path"])):
        raise ValueError("skill changed after shard planning")
    family_ids = {family["family_id"] for family in campaign["data_families"]}
    pending = [
        shard
        for shard in plan["shards"]
        if not _shard_completed(root, shard)
    ]
    results = []
    max_workers = min(
        campaign["sharding"]["max_parallel_subagents"],
        max(1, len(pending)),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_shard,
                campaign,
                root,
                shard,
                plan,
                generator_command,
                critic_command,
                tokenizer,
                family_ids,
                timeout_seconds,
            )
            for shard in pending
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    rebuild_accepted_ledger(
        root,
        semantic_threshold=float(
            campaign["acceptance_gates"]["semantic_similarity_max"]
        ),
    )
    return {
        "schema_version": "nano_subagent_campaign_run_receipt_v1",
        "campaign_id": campaign["campaign_id"],
        "planned_shards": len(plan["shards"]),
        "executed_shards": len(results),
        "skipped_completed_shards": len(plan["shards"]) - len(results),
        "completed_shards": sum(result["status"] == "completed" for result in results),
        "failed_shards": sum(result["status"] == "failed" for result in results),
    }


def _run_shard(
    campaign: dict[str, Any],
    root: Path,
    shard: dict[str, Any],
    plan: dict[str, Any],
    generator_command: list[str],
    critic_command: list[str],
    tokenizer: Any,
    family_ids: set[str],
    timeout_seconds: int,
) -> dict[str, Any]:
    shard_dir = _shard_dir(root, shard)
    shard_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "schema_version": "nano_subagent_generator_request_v1",
        "campaign_id": campaign["campaign_id"],
        "family_id": shard["family_id"],
        "shard_id": shard["shard_id"],
        "attempt": shard["attempt"],
        "candidate_samples": shard["candidate_samples"],
        "candidate_tokens_min": shard["candidate_tokens_min"],
        "seed": shard["seed"],
        "skill_path": plan["skill_path"],
        "skill_sha256": plan["skill_sha256"],
        "output_schema": CANDIDATE_SCHEMA,
    }
    generator_input = shard_dir / "generator-input.json"
    candidate_output = shard_dir / "candidates.jsonl"
    critic_input = shard_dir / "critic-input.json"
    critic_output = shard_dir / "critic.jsonl"
    accepted_output = shard_dir / "accepted.jsonl"
    status_path = shard_dir / "status.json"
    generator_input.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    values = {
        "attempt": shard["attempt"],
        "family_id": shard["family_id"],
        "input": str(generator_input),
        "output": str(candidate_output),
        "shard_id": shard["shard_id"],
        "skill_path": plan["skill_path"],
    }
    try:
        generator_receipt = _run_command(
            render_command(generator_command, values),
            shard_dir / "generator.log",
            timeout_seconds,
        )
        candidates = read_jsonl(candidate_output)
        if len(candidates) != shard["candidate_samples"]:
            raise ValueError(
                f"generator returned {len(candidates)} candidates; "
                f"expected {shard['candidate_samples']}"
            )
        critic_request = {
            "schema_version": "nano_subagent_critic_request_v1",
            "campaign_id": campaign["campaign_id"],
            "family_id": shard["family_id"],
            "shard_id": shard["shard_id"],
            "attempt": shard["attempt"],
            "candidates": candidates,
        }
        critic_input.write_text(
            json.dumps(critic_request, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        critic_values = dict(values)
        critic_values["input"] = str(critic_input)
        critic_values["output"] = str(critic_output)
        critic_receipt = _run_command(
            render_command(critic_command, critic_values),
            shard_dir / "critic.log",
            timeout_seconds,
        )
        decisions = read_jsonl(critic_output)
        accepted, rejected = accept_candidates(
            campaign,
            candidates,
            decisions,
            tokenizer=tokenizer,
            skill_sha256=plan["skill_sha256"],
            family_ids=family_ids,
            shard=shard,
        )
        write_jsonl(accepted_output, accepted)
        status = {
            "schema_version": "nano_subagent_shard_status_v1",
            "status": "completed",
            "campaign_id": campaign["campaign_id"],
            "family_id": shard["family_id"],
            "shard_id": shard["shard_id"],
            "attempt": shard["attempt"],
            "candidate_rows": len(candidates),
            "accepted_rows": len(accepted),
            "rejected_rows": len(rejected),
            "accepted_tokens": sum(row["token_count"] for row in accepted),
            "rejection_reasons": dict(Counter(rejected)),
            "generator_process": generator_receipt,
            "critic_process": critic_receipt,
            "accepted_sha256": sha256_file(accepted_output),
        }
    except Exception as exc:
        status = {
            "schema_version": "nano_subagent_shard_status_v1",
            "status": "failed",
            "campaign_id": campaign["campaign_id"],
            "family_id": shard["family_id"],
            "shard_id": shard["shard_id"],
            "attempt": shard["attempt"],
            "failure_type": type(exc).__name__,
            "failure": str(exc),
        }
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return status


def _run_command(
    command: list[str],
    log_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout_seconds,
    )
    log_path.write_text(
        result.stdout + ("\n[stderr]\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"subagent exited {result.returncode}; see {log_path.name}"
        )
    return {
        "argv_sha256": sha256_text(canonical_json(command)),
        "returncode": result.returncode,
        "log_sha256": sha256_file(log_path),
    }


def accept_candidates(
    campaign: dict[str, Any],
    candidates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    *,
    tokenizer: Any,
    skill_sha256: str,
    family_ids: set[str],
    shard: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    decision_map = {}
    for decision in decisions:
        if decision.get("schema_version") != CRITIC_SCHEMA:
            raise ValueError("critic returned an unsupported schema")
        candidate_id = str(decision.get("candidate_id", ""))
        if not candidate_id or candidate_id in decision_map:
            raise ValueError("critic candidate IDs must be non-empty and unique")
        decision_map[candidate_id] = decision
    candidate_ids = [str(candidate.get("candidate_id", "")) for candidate in candidates]
    if set(candidate_ids) != set(decision_map):
        raise ValueError("critic decisions do not exactly cover generator candidates")

    accepted = []
    rejected = []
    seen_candidate_ids = set()
    minimum_score = float(campaign["acceptance_gates"]["minimum_critic_score"])
    expected_verifiers = {
        family["family_id"]: family["deterministic_verifier"]
        for family in campaign["data_families"]
    }
    for candidate in candidates:
        try:
            validate_candidate(
                candidate,
                skill_sha256=skill_sha256,
                family_ids=family_ids,
                expected_family=shard["family_id"],
                expected_verifier=expected_verifiers[shard["family_id"]],
            )
        except ValueError as exc:
            rejected.append(f"candidate_invalid:{exc}")
            continue
        candidate_id = candidate["candidate_id"]
        if candidate_id in seen_candidate_ids:
            rejected.append("candidate_duplicate_id")
            continue
        seen_candidate_ids.add(candidate_id)
        decision = decision_map[candidate_id]
        score = decision.get("score")
        critic_receipt = decision.get("critic_receipt")
        generator_request_id = candidate["generator_receipt"]["request_id"]
        critic_request_id = (
            critic_receipt.get("request_id")
            if isinstance(critic_receipt, dict)
            else None
        )
        independent_critic = (
            isinstance(critic_receipt, dict)
            and bool(critic_receipt.get("critic"))
            and bool(critic_request_id)
            and critic_request_id != generator_request_id
        )
        if (
            decision.get("accept") is not True
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or float(score) < minimum_score
            or not independent_critic
        ):
            rejected.append(
                "critic_not_independent"
                if not independent_critic
                else "critic_rejected"
            )
            continue
        passed, verifier_receipt = verify_candidate(candidate)
        if not passed:
            rejected.append("verifier_rejected")
            continue
        messages = candidate["messages"]
        exact_hash = sha256_text(canonical_json(messages))
        semantic_hash = sha256_text(normalize_messages(messages))
        sample_id = sha256_text(
            (
                f"{campaign['campaign_id']}:{shard['family_id']}:"
                f"{shard['shard_id']}:{shard['attempt']}:{candidate_id}"
            )
        )[:24]
        token_count = count_tokens(tokenizer, messages)
        accepted.append(
            {
                "schema_version": ACCEPTED_SCHEMA,
                "sample_id": sample_id,
                "candidate_id": candidate_id,
                "campaign_id": campaign["campaign_id"],
                "family_id": candidate["family_id"],
                "task_family": str(
                    candidate.get("task_family", candidate["family_id"])
                ),
                "split": candidate["split"],
                "skill_id": candidate["skill_id"],
                "skill_sha256": skill_sha256,
                "messages": messages,
                "source": candidate["source"],
                "task_spec": candidate["task_spec"],
                "generator_receipt": candidate["generator_receipt"],
                "critic_receipt": critic_receipt,
                "critic_accept": decision["accept"],
                "critic_score": float(score),
                "critic_reasons": list(decision.get("reasons", [])),
                "verifier": candidate["verifier"],
                "verifier_receipt": verifier_receipt,
                "shard_receipt": {
                    "shard_id": shard["shard_id"],
                    "attempt": shard["attempt"],
                    "seed": shard["seed"],
                },
                "token_count": token_count,
                "exact_hash": exact_hash,
                "semantic_hash": semantic_hash,
            }
        )
    return accepted, rejected


def validate_candidate(
    candidate: dict[str, Any],
    *,
    skill_sha256: str,
    family_ids: set[str],
    expected_family: str,
    expected_verifier: str,
) -> None:
    if candidate.get("schema_version") != CANDIDATE_SCHEMA:
        raise ValueError("schema")
    candidate_id = str(candidate.get("candidate_id", ""))
    if not candidate_id:
        raise ValueError("candidate_id")
    family_id = str(candidate.get("family_id", ""))
    if family_id not in family_ids or family_id != expected_family:
        raise ValueError("family_id")
    if candidate.get("split") not in {"train", "dev"}:
        raise ValueError("split")
    if not candidate.get("skill_id"):
        raise ValueError("skill_id")
    source = candidate.get("source")
    if not isinstance(source, dict) or source.get("kind") not in ALLOWED_SOURCE_KINDS:
        raise ValueError("source")
    if not payload_source_allowed(
        source,
        candidate.get("messages", []),
        candidate.get("task_spec", {}),
    ):
        raise ValueError("forbidden_source")
    messages = candidate.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("messages")
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"assistant", "system", "user"}
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError("message_shape")
    if messages[-1]["role"] != "assistant":
        raise ValueError("assistant_target")
    if not isinstance(candidate.get("task_spec"), dict):
        raise ValueError("task_spec")
    receipt = candidate.get("generator_receipt")
    if (
        not isinstance(receipt, dict)
        or receipt.get("skill_sha256") != skill_sha256
        or not receipt.get("request_id")
    ):
        raise ValueError("generator_receipt")
    verifier = candidate.get("verifier")
    if (
        not isinstance(verifier, dict)
        or verifier.get("kind") not in SUPPORTED_VERIFIERS
        or verifier.get("kind") != expected_verifier
    ):
        raise ValueError("verifier")


def verify_candidate(candidate: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    verifier = candidate["verifier"]
    assistant = candidate["messages"][-1]["content"]
    task_spec = candidate["task_spec"]
    kind = verifier["kind"]
    passed = False
    if kind == "safe_execution_receipt_v1":
        passed = _verify_safe_execution(task_spec, assistant)
    elif kind == "tool_trace_contract_v1":
        passed = _verify_tool_trace(task_spec, assistant)
    elif kind == "state_plan_consistency_v1":
        passed = _verify_state_plan(task_spec, assistant)
    elif kind == "patch_test_receipt_v1":
        passed = _verify_patch_test_receipt(task_spec, assistant)
    elif kind == "skill_route_receipt_v1":
        passed = _verify_skill_route(task_spec, assistant)
    return passed, {
        "kind": kind,
        "passed": passed,
        "verifier_input_sha256": sha256_text(
            canonical_json({"task_spec": task_spec, "verifier": verifier})
        ),
        "assistant_sha256": sha256_text(assistant),
    }


def _json_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _verify_safe_execution(task_spec: dict[str, Any], assistant: str) -> bool:
    expression = task_spec.get("expression")
    if not isinstance(expression, str):
        return False
    try:
        expected = format_number(evaluate_arithmetic(expression))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return False
    match = re.fullmatch(
        r"FINAL: ([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))",
        assistant.strip(),
    )
    return match is not None and match.group(1) == expected


def _verify_tool_trace(task_spec: dict[str, Any], assistant: str) -> bool:
    value = _json_object(assistant)
    if value is None or value.get("final_status") != "verified":
        return False
    calls = value.get("tool_calls")
    required = task_spec.get("required_calls")
    if not isinstance(calls, list) or not isinstance(required, list):
        return False
    if len(calls) != len(required):
        return False
    for actual, expected in zip(calls, required):
        if (
            not isinstance(actual, dict)
            or not isinstance(expected, dict)
            or actual.get("name") != expected.get("name")
            or actual.get("arguments") != expected.get("arguments")
            or actual.get("status") != expected.get("status")
        ):
            return False
    return True


def _verify_state_plan(task_spec: dict[str, Any], assistant: str) -> bool:
    value = _json_object(assistant)
    if value is None:
        return False
    for key in ("constraints", "evidence", "pending"):
        actual = value.get(key)
        expected = task_spec.get(key)
        if (
            not isinstance(actual, list)
            or not isinstance(expected, list)
            or sorted(actual) != sorted(expected)
        ):
            return False
    return value.get("stop") is task_spec.get("stop")


def _verify_patch_test_receipt(
    task_spec: dict[str, Any],
    assistant: str,
) -> bool:
    value = _json_object(assistant)
    if value is None:
        return False
    original = task_spec.get("original_content")
    if not isinstance(original, str):
        return False
    return (
        value.get("file") == task_spec.get("file")
        and value.get("before_sha256") == sha256_text(original)
        and value.get("after_content") == task_spec.get("expected_content")
        and value.get("test_command") == task_spec.get("test_command")
        and value.get("test_status") == "passed"
    )


def _verify_skill_route(task_spec: dict[str, Any], assistant: str) -> bool:
    value = _json_object(assistant)
    if value is None:
        return False
    request_tags = task_spec.get("request_tags")
    skills = task_spec.get("skills")
    if not isinstance(request_tags, list) or not isinstance(skills, list):
        return False
    required = set(request_tags)
    eligible = []
    for skill in skills:
        if not isinstance(skill, dict) or not isinstance(skill.get("tags"), list):
            return False
        tags = set(skill["tags"])
        if required <= tags and skill.get("skill_id"):
            eligible.append((len(tags), str(skill["skill_id"])))
    if not eligible:
        return False
    selected = sorted(eligible)[0][1]
    steps = value.get("steps")
    return (
        value.get("selected_skill") == selected
        and isinstance(steps, list)
        and bool(steps)
        and all(isinstance(step, str) and step.strip() for step in steps)
    )


def payload_source_allowed(
    source: dict[str, Any],
    messages: list[dict[str, str]],
    task_spec: dict[str, Any],
) -> bool:
    if source.get("kind") not in ALLOWED_SOURCE_KINDS:
        return False
    serialized_source = canonical_json(source).lower()
    serialized_messages = canonical_json(messages).lower()
    serialized_task_spec = canonical_json(task_spec).lower()
    return not any(
        marker in serialized_source
        or marker in serialized_messages
        or marker in serialized_task_spec
        for marker in FORBIDDEN_PAYLOAD_MARKERS
    )


def count_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if hasattr(token_ids, "get"):
        token_ids = token_ids.get("input_ids")
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("tokenizer returned no input IDs")
    return len(token_ids)


def audit_campaign(
    campaign_path: str | Path,
    run_dir: str | Path,
    *,
    tokenizer: Any,
    tokenizer_path: str | Path | None = None,
) -> dict[str, Any]:
    campaign = load_skill_sft_campaign(campaign_path)
    root = Path(run_dir)
    rows = read_jsonl(root / "accepted.jsonl", missing_ok=True)
    source_policy_pass = True
    hash_pass = True
    token_pass = True
    verifier_pass = True
    critic_pass = True
    skill_identity_pass = True
    sample_ids = []
    exact_hashes = []
    semantic_hashes = []
    recomputed_rows = []
    for row in rows:
        if row.get("schema_version") != ACCEPTED_SCHEMA:
            hash_pass = False
            continue
        try:
            messages = row["messages"]
            exact_hash = sha256_text(canonical_json(messages))
            semantic_hash = sha256_text(normalize_messages(messages))
            tokens = count_tokens(tokenizer, messages)
        except (KeyError, TypeError, ValueError):
            hash_pass = False
            token_pass = False
            continue
        hash_pass = hash_pass and row.get("exact_hash") == exact_hash
        hash_pass = hash_pass and row.get("semantic_hash") == semantic_hash
        token_pass = token_pass and row.get("token_count") == tokens
        source = row.get("source")
        source_policy_pass = source_policy_pass and (
            isinstance(source, dict)
            and isinstance(row.get("task_spec"), dict)
            and payload_source_allowed(source, messages, row["task_spec"])
        )
        candidate = {
            "messages": messages,
            "task_spec": row.get("task_spec"),
            "verifier": row.get("verifier"),
        }
        try:
            locally_verified, expected_verifier_receipt = verify_candidate(candidate)
        except (KeyError, TypeError, ValueError):
            locally_verified = False
            expected_verifier_receipt = {}
        verifier_pass = (
            verifier_pass
            and locally_verified
            and row.get("verifier_receipt") == expected_verifier_receipt
        )
        critic_receipt = row.get("critic_receipt")
        generator_receipt = row.get("generator_receipt")
        generator_request_id = (
            generator_receipt.get("request_id")
            if isinstance(generator_receipt, dict)
            else None
        )
        critic_pass = critic_pass and (
            isinstance(critic_receipt, dict)
            and isinstance(generator_receipt, dict)
            and bool(critic_receipt.get("critic"))
            and bool(critic_receipt.get("request_id"))
            and critic_receipt.get("request_id")
            != generator_request_id
            and row.get("critic_accept") is True
            and isinstance(row.get("critic_score"), (int, float))
            and not isinstance(row.get("critic_score"), bool)
            and float(row["critic_score"])
            >= float(campaign["acceptance_gates"]["minimum_critic_score"])
        )
        skill_identity_pass = skill_identity_pass and (
            isinstance(generator_receipt, dict)
            and row.get("skill_sha256")
            == generator_receipt.get("skill_sha256")
        )
        sample_ids.append(row.get("sample_id"))
        exact_hashes.append(exact_hash)
        semantic_hashes.append(semantic_hash)
        recomputed_rows.append((row, tokens))
    exact_unique = len(exact_hashes) == len(set(exact_hashes))
    semantic_unique = len(semantic_hashes) == len(set(semantic_hashes))
    id_unique = len(sample_ids) == len(set(sample_ids))
    semantic_near_duplicates = find_semantic_near_duplicates(
        [row["messages"] for row, _ in recomputed_rows],
        threshold=float(campaign["acceptance_gates"]["semantic_similarity_max"]),
    )
    global_dedup_pass = (
        exact_unique
        and semantic_unique
        and id_unique
        and not semantic_near_duplicates
    )
    split_hashes: dict[str, set[str]] = defaultdict(set)
    for row, _ in recomputed_rows:
        split_hashes[row["split"]].add(row["semantic_hash"])
    cross_split_overlap = sorted(
        split_hashes.get("train", set()) & split_hashes.get("dev", set())
    )
    global_dedup_pass = global_dedup_pass and not cross_split_overlap

    train_rows = [(row, tokens) for row, tokens in recomputed_rows if row["split"] == "train"]
    train_samples = len(train_rows)
    train_tokens = sum(tokens for _, tokens in train_rows)
    family_summary = {}
    family_quotas_pass = True
    for family in campaign["data_families"]:
        family_id = family["family_id"]
        matched = [
            (row, tokens)
            for row, tokens in train_rows
            if row["family_id"] == family_id
        ]
        samples = len(matched)
        tokens = sum(value for _, value in matched)
        sample_pass = samples >= family["accepted_train_samples_min"]
        token_quota_pass = tokens >= family["accepted_train_tokens_min"]
        family_quotas_pass = family_quotas_pass and sample_pass and token_quota_pass
        family_summary[family_id] = {
            "train_samples": samples,
            "train_tokens": tokens,
            "sample_target": family["accepted_train_samples_min"],
            "token_target": family["accepted_train_tokens_min"],
            "sample_deficit": max(
                0,
                family["accepted_train_samples_min"] - samples,
            ),
            "token_deficit": max(
                0,
                family["accepted_train_tokens_min"] - tokens,
            ),
            "sample_target_pass": sample_pass,
            "token_target_pass": token_quota_pass,
        }
    tokenizer_identity_pass = True
    if tokenizer_path is not None:
        tokenizer_identity_pass = verify_tokenizer_identity(
            campaign,
            Path(tokenizer_path),
        )
    checks = {
        "critic_revalidation_pass": critic_pass,
        "family_quotas_pass": family_quotas_pass,
        "global_dedup_pass": global_dedup_pass,
        "source_policy_pass": source_policy_pass,
        "skill_identity_pass": skill_identity_pass,
        "tokenizer_identity_pass": tokenizer_identity_pass,
        "train_sample_target_pass": (
            train_samples >= campaign["targets"]["accepted_train_samples_min"]
        ),
        "train_token_target_pass": (
            train_tokens >= campaign["targets"]["accepted_train_tokens_min"]
        ),
        "verifier_revalidation_pass": verifier_pass,
    }
    training_unblocked = all(checks.values()) and hash_pass and token_pass
    report = {
        "schema_version": "nano_skill_sft_campaign_audit_v1",
        "campaign_id": campaign["campaign_id"],
        "accepted_rows": len(recomputed_rows),
        "accepted_train_samples": train_samples,
        "accepted_train_tokens": train_tokens,
        "checks": checks,
        "hash_recomputation_pass": hash_pass,
        "token_recomputation_pass": token_pass,
        "family_summary": family_summary,
        "near_duplicate_pairs": semantic_near_duplicates[:100],
        "cross_split_semantic_overlap": cross_split_overlap[:100],
        "training_unblocked": training_unblocked,
    }
    (root / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def verify_tokenizer_identity(
    campaign: dict[str, Any],
    tokenizer_path: Path,
) -> bool:
    expected = campaign["token_accounting"]["file_sha256"]
    return all(
        (tokenizer_path / filename).is_file()
        and sha256_file(tokenizer_path / filename) == digest
        for filename, digest in expected.items()
    )


def find_semantic_near_duplicates(
    message_sets: list[list[dict[str, str]]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    index = SemanticDuplicateIndex(threshold)
    duplicates = []
    for row_index, messages in enumerate(message_sets):
        normalized = normalize_messages(messages)
        for match in index.matches(normalized):
            duplicates.append(
                {
                    "left_index": match["index"],
                    "right_index": row_index,
                    "jaccard": match["jaccard"],
                    "sequence_ratio": match["sequence_ratio"],
                }
            )
        index.add(normalized, row_index)
    return duplicates


class SemanticDuplicateIndex:
    def __init__(self, threshold: float):
        if not 0 < threshold < 1:
            raise ValueError("semantic duplicate threshold must be in (0, 1)")
        self.threshold = threshold
        self.normalized: list[str] = []
        self.token_sets: list[set[str]] = []
        self.prefix_index: dict[str, list[int]] = defaultdict(list)

    def matches(self, normalized: str) -> list[dict[str, Any]]:
        tokens = set(re.findall(r"\w+", normalized))
        prefix = self._prefix(tokens)
        candidates = {
            prior
            for token in prefix
            for prior in self.prefix_index.get(token, [])
        }
        matches = []
        for prior in sorted(candidates):
            left = self.token_sets[prior]
            if not self._lengths_can_match(len(tokens), len(left)):
                continue
            union = tokens | left
            jaccard = len(tokens & left) / len(union) if union else 1.0
            if jaccard < self.threshold:
                continue
            sequence = SequenceMatcher(
                None,
                self.normalized[prior],
                normalized,
                autojunk=False,
            ).ratio()
            if sequence >= self.threshold:
                matches.append(
                    {
                        "index": prior,
                        "jaccard": jaccard,
                        "sequence_ratio": sequence,
                    }
                )
        return matches

    def add(self, normalized: str, index: int | None = None) -> int:
        if index is None:
            index = len(self.normalized)
        if index != len(self.normalized):
            raise ValueError("semantic duplicate index rows must be append-only")
        tokens = set(re.findall(r"\w+", normalized))
        self.normalized.append(normalized)
        self.token_sets.append(tokens)
        for token in self._prefix(tokens):
            self.prefix_index[token].append(index)
        return index

    def _prefix(self, tokens: set[str]) -> list[str]:
        ordered = sorted(tokens, key=lambda value: (sha256_text(value), value))
        prefix_length = max(
            1,
            len(ordered) - math.ceil(self.threshold * len(ordered)) + 1,
        )
        return ordered[:prefix_length]

    def _lengths_can_match(self, left: int, right: int) -> bool:
        shorter = min(left, right)
        longer = max(left, right)
        return shorter >= self.threshold * longer


def plan_refill(
    campaign_path: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    campaign = load_skill_sft_campaign(campaign_path)
    root = Path(run_dir)
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    if audit.get("training_unblocked") is True:
        return {
            "schema_version": PLAN_SCHEMA,
            "campaign_id": campaign["campaign_id"],
            "mode": "refill",
            "shards": [],
            "reason": "targets_already_met",
        }
    family_needs = {}
    for family_id, summary in audit["family_summary"].items():
        samples = summary["train_samples"]
        tokens = summary["train_tokens"]
        average_tokens = (
            tokens / samples
            if samples
            else summary["token_target"] / summary["sample_target"]
        )
        sample_need = summary["sample_deficit"]
        token_need_as_samples = math.ceil(
            summary["token_deficit"] / max(1, average_tokens)
        )
        accepted_need = max(sample_need, token_need_as_samples)
        if accepted_need:
            family_needs[family_id] = accepted_need
    refill_batch = campaign["sharding"]["refill_batch_shards"]
    candidates_per_shard = campaign["sharding"]["candidate_samples_per_shard"]
    oversubscription = campaign["sharding"]["initial_oversubscription_ratio"]
    required_shards = sum(
        math.ceil(need * oversubscription / candidates_per_shard)
        for need in family_needs.values()
    )
    shard_count = max(refill_batch, required_shards)
    allocation = allocate_integer_total(shard_count, family_needs)
    existing_plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    prior_plans = [existing_plan]
    for prior_path in sorted(root.glob("refill-plan-*.json")):
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        validate_plan(prior)
        prior_plans.append(prior)
    next_shard_id = (
        max(
            shard["shard_id"]
            for prior in prior_plans
            for shard in prior["shards"]
        )
        + 1
    )
    shards = []
    for family_id in sorted(allocation):
        for _ in range(allocation[family_id]):
            shards.append(
                {
                    "shard_id": next_shard_id,
                    "family_shard_id": None,
                    "family_id": family_id,
                    "attempt": 0,
                    "candidate_samples": candidates_per_shard,
                    "candidate_tokens_min": math.ceil(
                        audit["family_summary"][family_id]["token_deficit"]
                        / max(1, allocation[family_id])
                    ),
                    "seed": 202608199000 + next_shard_id,
                }
            )
            next_shard_id += 1
    refill = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": existing_plan["campaign_sha256"],
        "mode": "refill",
        "skill_path": existing_plan["skill_path"],
        "skill_sha256": existing_plan["skill_sha256"],
        "max_parallel_subagents": campaign["sharding"]["max_parallel_subagents"],
        "shards": shards,
        "deficits": family_needs,
    }
    validate_plan(refill)
    refill_index = len(list(root.glob("refill-plan-*.json"))) + 1
    output = root / f"refill-plan-{refill_index:03d}.json"
    output.write_text(
        json.dumps(refill, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return refill


def rebuild_accepted_ledger(
    run_dir: str | Path,
    *,
    semantic_threshold: float = 0.92,
) -> Path:
    root = Path(run_dir)
    rows = []
    rejected = Counter()
    sample_ids = set()
    exact_hashes = set()
    semantic_hashes = set()
    semantic_index = SemanticDuplicateIndex(semantic_threshold)
    for accepted_path in sorted(root.glob("shards/*/attempt-*/accepted.jsonl")):
        for row in read_jsonl(accepted_path):
            if row["sample_id"] in sample_ids:
                rejected["duplicate_sample_id"] += 1
                continue
            if row["exact_hash"] in exact_hashes:
                rejected["duplicate_exact_hash"] += 1
                continue
            if row["semantic_hash"] in semantic_hashes:
                rejected["duplicate_semantic_hash"] += 1
                continue
            normalized = normalize_messages(row["messages"])
            if semantic_index.matches(normalized):
                rejected["semantic_near_duplicate"] += 1
                continue
            sample_ids.add(row["sample_id"])
            exact_hashes.add(row["exact_hash"])
            semantic_hashes.add(row["semantic_hash"])
            semantic_index.add(normalized)
            rows.append(row)
    output = root / "accepted.jsonl"
    write_jsonl(output, rows)
    receipt = {
        "schema_version": "nano_skill_sft_ledger_merge_v1",
        "accepted_rows": len(rows),
        "globally_rejected_rows": sum(rejected.values()),
        "global_rejection_reasons": dict(rejected),
        "accepted_sha256": sha256_file(output),
        "semantic_similarity_metric": (
            "token_set_jaccard_and_normalized_sequence_ratio"
        ),
        "semantic_similarity_max": semantic_threshold,
    }
    (root / "merge.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def read_jsonl(
    path: str | Path,
    *,
    missing_ok: bool = False,
) -> list[dict[str, Any]]:
    source = Path(path)
    if missing_ok and not source.exists():
        return []
    rows = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{source}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _shard_completed(root: Path, shard: dict[str, Any]) -> bool:
    status_path = _shard_dir(root, shard) / "status.json"
    if not status_path.exists():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return status.get("status") == "completed"


def _shard_dir(root: Path, shard: dict[str, Any]) -> Path:
    return (
        root
        / "shards"
        / f"shard-{shard['shard_id']:05d}"
        / f"attempt-{shard['attempt']:03d}"
    )
