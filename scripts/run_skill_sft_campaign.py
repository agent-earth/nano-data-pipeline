#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_data_pipeline.skill_evolution import (
    load_skill_scorecard,
    select_skill_candidate,
)
from nano_data_pipeline.subagent_campaign import (
    audit_campaign,
    load_command,
    plan_campaign,
    plan_refill,
    run_plan,
    write_plan,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILL = ROOT / "skills/skill-sft-campaign/SKILL.md"


def main() -> None:
    parser = argparse.ArgumentParser(prog="run_skill_sft_campaign.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--campaign", required=True)
    plan_parser.add_argument("--run-dir", required=True)
    plan_parser.add_argument("--skill", default=str(DEFAULT_SKILL))
    plan_parser.add_argument("--max-shards", type=int)
    plan_parser.add_argument("--candidate-samples", type=int)
    plan_parser.add_argument("--candidate-tokens-per-sample", type=int)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--campaign", required=True)
    run_parser.add_argument("--run-dir", required=True)
    run_parser.add_argument("--plan")
    run_parser.add_argument("--generator-command-json", required=True)
    run_parser.add_argument("--critic-command-json", required=True)
    run_parser.add_argument("--tokenizer", required=True)
    run_parser.add_argument("--timeout-seconds", type=int, default=1800)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--campaign", required=True)
    audit_parser.add_argument("--run-dir", required=True)
    audit_parser.add_argument("--tokenizer", required=True)

    refill_parser = subparsers.add_parser("refill")
    refill_parser.add_argument("--campaign", required=True)
    refill_parser.add_argument("--run-dir", required=True)

    promote_parser = subparsers.add_parser("promote-skill")
    promote_parser.add_argument("--parent-scorecard", required=True)
    promote_parser.add_argument(
        "--candidate-scorecard",
        action="append",
        default=[],
    )
    promote_parser.add_argument(
        "--protected-family",
        action="append",
        required=True,
    )
    promote_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "plan":
        plan = plan_campaign(
            args.campaign,
            skill_path=args.skill,
            max_shards=args.max_shards,
            candidate_samples_override=args.candidate_samples,
            candidate_tokens_per_sample_override=(
                args.candidate_tokens_per_sample
            ),
        )
        write_plan(plan, args.run_dir)
        print(json.dumps(_plan_summary(plan), indent=2, sort_keys=True))
    elif args.command == "run":
        tokenizer_path = _local_path(args.tokenizer, "tokenizer")
        tokenizer = _load_tokenizer(tokenizer_path)
        receipt = run_plan(
            args.campaign,
            args.run_dir,
            generator_command=load_command(args.generator_command_json),
            critic_command=load_command(args.critic_command_json),
            tokenizer=tokenizer,
            plan_path=args.plan,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        if receipt["failed_shards"]:
            raise SystemExit(1)
    elif args.command == "audit":
        tokenizer_path = _local_path(args.tokenizer, "tokenizer")
        tokenizer = _load_tokenizer(tokenizer_path)
        report = audit_campaign(
            args.campaign,
            args.run_dir,
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "refill":
        plan = plan_refill(args.campaign, args.run_dir)
        print(json.dumps(_plan_summary(plan), indent=2, sort_keys=True))
    elif args.command == "promote-skill":
        parent = load_skill_scorecard(args.parent_scorecard)
        candidates = [
            load_skill_scorecard(path) for path in args.candidate_scorecard
        ]
        receipt = select_skill_candidate(
            parent,
            candidates,
            protected_families=args.protected_family,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))


def _load_tokenizer(path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def _local_path(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"{label} path does not exist: {resolved}")
    return resolved


def _plan_summary(plan: dict) -> dict:
    family_shards: dict[str, int] = {}
    for shard in plan.get("shards", []):
        family_id = shard["family_id"]
        family_shards[family_id] = family_shards.get(family_id, 0) + 1
    return {
        "schema_version": plan["schema_version"],
        "campaign_id": plan["campaign_id"],
        "mode": plan["mode"],
        "shards": len(plan.get("shards", [])),
        "family_shards": dict(sorted(family_shards.items())),
        "candidate_samples": sum(
            shard["candidate_samples"] for shard in plan.get("shards", [])
        ),
        "candidate_tokens_min": sum(
            shard["candidate_tokens_min"] for shard in plan.get("shards", [])
        ),
    }


if __name__ == "__main__":
    main()
