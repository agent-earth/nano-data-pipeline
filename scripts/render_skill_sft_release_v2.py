#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_data_pipeline.feedback import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--initial-source-revision", required=True)
    parser.add_argument("--refill-source-revision", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()

    root = Path(args.run_dir)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    merge = json.loads((root / "merge.json").read_text(encoding="utf-8"))
    statuses = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("shards/*/attempt-*/status.json"))
    ]
    release = {
        "schema_version": "nano_skill_sft_release_v1",
        "release_id": "skill-sft-10k-10m-v2",
        "source_revisions": {
            "initial_shards": args.initial_source_revision,
            "refill_shards": args.refill_source_revision,
        },
        "contains_raw_rows": False,
        "raw_ledger_local_only": True,
        "models": {
            "generator": "Qwen3.5-4B",
            "critic": "Qwen3.5-9B",
        },
        "campaign": {
            "campaign_id": audit["campaign_id"],
            "planned_shards": len(plan["shards"]),
            "total_completed_shards": len(statuses),
            "initial_completed_shards": sum(
                status.get("shard_id", -1) < len(plan["shards"])
                for status in statuses
            ),
            "refill_completed_shards": sum(
                status.get("shard_id", -1) >= len(plan["shards"])
                for status in statuses
            ),
            "failed_shards": sum(
                status.get("status") != "completed" for status in statuses
            ),
            "generator_model_calls": sum(
                status.get("generator_model_calls", 0) for status in statuses
            ),
            "critic_model_calls": sum(
                status.get("critic_model_calls", 0) for status in statuses
            ),
            "shard_accepted_rows_before_global_dedup": sum(
                status.get("accepted_rows", 0) for status in statuses
            ),
        },
        "accepted": {
            "rows": audit["accepted_rows"],
            "train_samples": audit["accepted_train_samples"],
            "train_tokens": audit["accepted_train_tokens"],
            "dev_samples": audit["accepted_dev_samples"],
            "family_summary": audit["family_summary"],
        },
        "deduplication": {
            "globally_rejected_rows": merge["globally_rejected_rows"],
            "global_rejection_reasons": merge["global_rejection_reasons"],
            "semantic_similarity_metric": merge["semantic_similarity_metric"],
            "semantic_similarity_max": merge["semantic_similarity_max"],
        },
        "checks": audit["checks"],
        "training_unblocked": audit["training_unblocked"],
        "artifacts": {
            "accepted_jsonl_sha256": sha256_file(root / "accepted.jsonl"),
            "audit_json_sha256": sha256_file(root / "audit.json"),
            "merge_json_sha256": sha256_file(root / "merge.json"),
            "plan_json_sha256": sha256_file(root / "plan.json"),
            "refill_plan_json_sha256": sha256_file(
                root / "refill-plan-001.json"
            ),
        },
        "claim_boundary": (
            "This release proves the data campaign met its frozen row, token, "
            "development, provenance, verifier, critic, deduplication, and "
            "call-budget gates. It does not prove model or benchmark uplift."
        ),
    }
    if release["training_unblocked"] is not True:
        raise SystemExit("release cannot be rendered while training is blocked")
    manifest_output = Path(args.manifest_output)
    report_output = Path(args.report_output)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_output.write_text(render_report(release), encoding="utf-8")


def render_report(release: dict) -> str:
    accepted = release["accepted"]
    campaign = release["campaign"]
    lines = [
        "# Skill SFT 10k / 10M Release v2",
        "",
        "## 结果",
        "",
        f"- Train 样本：**{accepted['train_samples']:,}**",
        f"- Train tokens：**{accepted['train_tokens']:,}**",
        f"- Synthetic dev：**{accepted['dev_samples']:,}**",
        f"- 最终 accepted rows：**{accepted['rows']:,}**",
        f"- 完成 shard：**{campaign['total_completed_shards']}**",
            f"- 首轮 / refill shard：**{campaign['initial_completed_shards']} / "
            f"{campaign['refill_completed_shards']}**",
        f"- Generator recipe 调用：**{campaign['generator_model_calls']}**",
        f"- Critic recipe 调用：**{campaign['critic_model_calls']}**",
        f"- `training_unblocked`：**{str(release['training_unblocked']).lower()}**",
        "",
        "## Family 分布",
        "",
        "| Family | Train rows | Train tokens | Dev rows |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family, summary in sorted(accepted["family_summary"].items()):
        lines.append(
            f"| {family} | {summary['train_samples']:,} | "
            f"{summary['train_tokens']:,} | {summary['dev_samples']:,} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
        ]
    )
    for name, passed in sorted(release["checks"].items()):
        lines.append(f"- `{name}`: {'通过' if passed else '未通过'}")
    lines.extend(
        [
            "",
            "## 去重",
            "",
            f"- Shard-local accepted："
            f"{campaign['shard_accepted_rows_before_global_dedup']:,}",
            f"- Global accepted：{accepted['rows']:,}",
            f"- Global rejected："
            f"{release['deduplication']['globally_rejected_rows']:,}",
            "- 语义口径：`family + task_spec`；长 padding 只计 token，"
            "不决定语义唯一性。",
            "",
            "## 失败与修复",
            "",
            "首轮 compiler 的数值取模空间太小，导致不同 shard 生成了相同 "
            "`task_spec`。Shard-local verifier 无法发现跨 shard 重复；global audit "
            "拒绝了 11,872 行。随后将任务 identity 改为 hash 派生的大空间，"
            "从原始 shard 重算 semantic hash，并按真实 family/sample/token/dev "
            "缺口增加 23 个 refill shard。旧失败证据保留，未删除或改写。",
            "",
            "## Artifact",
            "",
        ]
    )
    for name, digest in sorted(release["artifacts"].items()):
        lines.append(f"- `{name}`: `{digest}`")
    lines.extend(
        [
            "",
            "Raw accepted JSONL 只保存在本地 ignored dataset 目录，不提交 GitHub。",
            "",
            "## 结论边界",
            "",
            release["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
