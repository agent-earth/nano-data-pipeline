#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from transformers import AutoTokenizer

from nano_data_pipeline.subagent_campaign import (
    audit_campaign,
    plan_campaign,
    plan_refill,
    run_plan,
    write_plan,
)


ROOT = Path(__file__).resolve().parents[1]
FAKE_SUBAGENT = ROOT / "tests/fixtures/fake_subagent.py"
SKILL = ROOT / "skills/skill-sft-campaign/SKILL.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign",
        default=str(ROOT / "manifests/skill_sft_campaign_v1.json"),
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--samples-per-shard", type=int, default=8)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if args.reset and run_dir.exists():
        shutil.rmtree(run_dir)
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    if not tokenizer_path.is_dir():
        raise SystemExit(f"tokenizer path does not exist: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
    )
    plan = plan_campaign(
        args.campaign,
        skill_path=SKILL,
        max_shards=args.shards,
        candidate_samples_override=args.samples_per_shard,
    )
    write_plan(plan, run_dir)
    generator = [
        sys.executable,
        str(FAKE_SUBAGENT),
        "generate",
        "--input",
        "{input}",
        "--output",
        "{output}",
    ]
    critic = [
        sys.executable,
        str(FAKE_SUBAGENT),
        "critic",
        "--input",
        "{input}",
        "--output",
        "{output}",
    ]
    first = run_plan(
        args.campaign,
        run_dir,
        generator_command=generator,
        critic_command=critic,
        tokenizer=tokenizer,
        timeout_seconds=60,
    )
    resumed = run_plan(
        args.campaign,
        run_dir,
        generator_command=generator,
        critic_command=critic,
        tokenizer=tokenizer,
        timeout_seconds=60,
    )
    audit = audit_campaign(
        args.campaign,
        run_dir,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
    )
    refill = plan_refill(args.campaign, run_dir)
    receipt = {
        "schema_version": "nano_skill_sft_campaign_smoke_v1",
        "first_run": first,
        "resume_run": resumed,
        "audit": audit,
        "refill_shards": len(refill["shards"]),
        "assertions": {
            "parallel_shards_completed": first["completed_shards"] == args.shards,
            "resume_skipped_completed": (
                resumed["skipped_completed_shards"] == args.shards
            ),
            "accepted_rows_present": audit["accepted_rows"] > 0,
            "token_recomputation_pass": audit["token_recomputation_pass"],
            "training_remains_blocked_below_scale": (
                audit["training_unblocked"] is False
            ),
            "refill_planned": len(refill["shards"]) > 0,
        },
    }
    if not all(receipt["assertions"].values()):
        raise SystemExit(json.dumps(receipt, indent=2, sort_keys=True))
    (run_dir / "smoke.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
