#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "critic"):
        child = subparsers.add_parser(command)
        child.add_argument("--input", required=True)
        child.add_argument("--output", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.command == "generate":
        rows = generate(request)
    else:
        rows = criticize(request)
    Path(args.output).write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def generate(request: dict) -> list[dict]:
    rows = []
    for index in range(request["candidate_samples"]):
        left = request["seed"] % 97 + index + 3
        right = request["shard_id"] * 11 + index + 5
        result = left + right
        candidate_id = (
            f"{request['family_id']}-{request['shard_id']}-"
            f"{request['attempt']}-{index}"
        )
        task_spec, target, verifier_kind = build_task(
            request["family_id"],
            left,
            right,
            result,
            index,
        )
        if index % 5 == 0:
            target = tamper_target(request["family_id"], target)
        rows.append(
            {
                "schema_version": "nano_subagent_candidate_v1",
                "candidate_id": candidate_id,
                "family_id": request["family_id"],
                "task_family": f"{request['family_id']}-smoke",
                "split": "dev" if index == 1 else "train",
                "skill_id": "skill-sft-campaign",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Solve this synthetic contract and return one "
                            "standalone final line."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Synthetic shard {request['shard_id']} item {index}: "
                            f"compute {left} plus {right}. Use FINAL: <number>."
                        ),
                    },
                    {"role": "assistant", "content": target},
                ],
                "source": {
                    "kind": "procedurally_generated_synthetic",
                    "generator": "fake-subagent-smoke",
                    "seed": request["seed"],
                },
                "task_spec": task_spec,
                "verifier": {"kind": verifier_kind},
                "generator_receipt": {
                    "request_id": f"fake-generator-{candidate_id}",
                    "skill_sha256": request["skill_sha256"],
                },
            }
        )
    return rows


def build_task(family_id: str, left: int, right: int, result: int, index: int):
    if family_id == "verified-reasoning":
        return (
            {"expression": f"{left} + {right}"},
            f"FINAL: {result}",
            "safe_execution_receipt_v1",
        )
    if family_id == "tool-use-and-recovery":
        task_spec = {
            "required_calls": [
                {
                    "name": "lookup",
                    "arguments": {"key": f"synthetic-{left}"},
                    "status": "error",
                },
                {
                    "name": "calculator",
                    "arguments": {"expression": f"{left}+{right}"},
                    "status": "ok",
                },
            ]
        }
        target = json.dumps(
            {
                "tool_calls": task_spec["required_calls"],
                "final_status": "verified",
            },
            sort_keys=True,
        )
        return task_spec, target, "tool_trace_contract_v1"
    if family_id == "planning-and-state":
        task_spec = {
            "constraints": [f"budget-{left}", "no-benchmark-payload"],
            "evidence": [f"synthetic-observation-{right}"],
            "pending": [f"verify-{result}"],
            "stop": False,
        }
        return (
            task_spec,
            json.dumps(task_spec, sort_keys=True),
            "state_plan_consistency_v1",
        )
    if family_id == "coding-and-validation":
        original = f"value = {left}\n"
        expected = f"value = {result}\n"
        task_spec = {
            "file": f"synthetic_{index}.py",
            "original_content": original,
            "expected_content": expected,
            "test_command": f"python -m unittest synthetic_{index}",
        }
        target = json.dumps(
            {
                "file": task_spec["file"],
                "before_sha256": sha256_text(original),
                "after_content": expected,
                "test_command": task_spec["test_command"],
                "test_status": "passed",
            },
            sort_keys=True,
        )
        return task_spec, target, "patch_test_receipt_v1"
    if family_id == "skill-routing-and-reflection":
        task_spec = {
            "request_tags": ["data", "validation"],
            "skills": [
                {
                    "skill_id": f"broad-{index}",
                    "tags": ["data", "validation", "training"],
                },
                {
                    "skill_id": f"minimal-{index}",
                    "tags": ["data", "validation"],
                },
            ],
        }
        target = json.dumps(
            {
                "selected_skill": f"minimal-{index}",
                "steps": ["validate manifest", "run local audit"],
            },
            sort_keys=True,
        )
        return task_spec, target, "skill_route_receipt_v1"
    raise ValueError(f"unsupported fake family: {family_id}")


def tamper_target(family_id: str, target: str) -> str:
    if family_id == "verified-reasoning":
        return "FINAL: -999"
    value = json.loads(target)
    if family_id == "tool-use-and-recovery":
        value["final_status"] = "unverified"
    elif family_id == "planning-and-state":
        value["pending"] = []
    elif family_id == "coding-and-validation":
        value["test_status"] = "failed"
    elif family_id == "skill-routing-and-reflection":
        value["selected_skill"] = "missing-skill"
    return json.dumps(value, sort_keys=True)


def sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def criticize(request: dict) -> list[dict]:
    rows = []
    for index, candidate in enumerate(request["candidates"]):
        accept = index % 4 != 3
        rows.append(
            {
                "schema_version": "nano_subagent_critic_v1",
                "candidate_id": candidate["candidate_id"],
                "score": 0.95 if accept else 0.4,
                "accept": accept,
                "reasons": [] if accept else ["synthetic_quality_rejection"],
                "critic_receipt": {
                    "request_id": f"fake-critic-{candidate['candidate_id']}",
                    "critic": "fake-independent-critic-smoke",
                },
            }
        )
    return rows


if __name__ == "__main__":
    main()
