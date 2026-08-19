from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_data_pipeline.campaign import load_skill_sft_campaign
from nano_data_pipeline.subagent_campaign import (
    SemanticDuplicateIndex,
    accept_candidates,
    allocate_integer_total,
    audit_campaign,
    find_semantic_near_duplicates,
    plan_campaign,
    plan_refill,
    rebuild_accepted_ledger,
    split_integer,
    validate_plan,
    write_jsonl,
    write_plan,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "manifests/skill_sft_campaign_v1.json"
CAMPAIGN_V2 = ROOT / "manifests/skill_sft_campaign_v2.json"
SKILL = ROOT / "skills/skill-sft-campaign/SKILL.md"


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return list(range(sum(len(row["content"].split()) + 1 for row in messages)))


class MappingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": super().apply_chat_template(messages, **kwargs)}


class SubagentCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.campaign = load_skill_sft_campaign(CAMPAIGN)
        self.skill_sha256 = _sha256(SKILL)

    def test_production_plan_meets_frozen_capacity(self):
        plan = plan_campaign(CAMPAIGN, skill_path=SKILL)

        self.assertEqual(plan["mode"], "production")
        self.assertEqual(len(plan["shards"]), 32)
        self.assertGreaterEqual(
            sum(row["candidate_samples"] for row in plan["shards"]),
            13_000,
        )
        self.assertEqual(
            sum(row["candidate_tokens_min"] for row in plan["shards"]),
            13_000_000,
        )
        self.assertEqual(
            {row["family_id"] for row in plan["shards"]},
            {row["family_id"] for row in self.campaign["data_families"]},
        )

    def test_recipe_production_plan_has_exact_dev_quota(self):
        plan = plan_campaign(CAMPAIGN_V2, skill_path=SKILL)

        self.assertEqual(len(plan["shards"]), 32)
        self.assertEqual(
            sum(row["dev_samples"] for row in plan["shards"]),
            400,
        )
        self.assertEqual(
            sum(row["candidate_samples"] for row in plan["shards"]),
            16_384,
        )
        self.assertEqual(
            sum(row["candidate_tokens_min"] for row in plan["shards"]),
            13_000_000,
        )

    def test_smoke_plan_is_labeled_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = plan_campaign(
                CAMPAIGN,
                skill_path=SKILL,
                max_shards=5,
                candidate_samples_override=8,
                candidate_tokens_per_sample_override=900,
            )
            first = write_plan(plan, directory)
            second = write_plan(plan, directory)

            self.assertEqual(plan["mode"], "smoke")
            self.assertEqual(first, second)
            self.assertEqual(len(plan["shards"]), 5)
            self.assertEqual(
                sum(row["candidate_samples"] for row in plan["shards"]),
                40,
            )
            self.assertEqual(
                sum(row["candidate_tokens_min"] for row in plan["shards"]),
                36_000,
            )

    def test_accepts_only_critic_and_verifier_passes(self):
        shard = {
            "shard_id": 0,
            "attempt": 0,
            "family_id": "verified-reasoning",
            "seed": 1,
        }
        passing = self._candidate("pass", "FINAL: 7", "3 + 4")
        verifier_fail = self._candidate("verify-fail", "FINAL: 7", "3 + 5")
        critic_fail = self._candidate("critic-fail", "FINAL: 7", "3 + 4")
        decisions = [
            self._decision("pass", True, 0.95),
            self._decision("verify-fail", True, 0.95),
            self._decision("critic-fail", False, 0.2),
        ]

        accepted, rejected = accept_candidates(
            self.campaign,
            [passing, verifier_fail, critic_fail],
            decisions,
            tokenizer=FakeTokenizer(),
            skill_sha256=self.skill_sha256,
            family_ids={
                family["family_id"] for family in self.campaign["data_families"]
            },
            shard=shard,
        )

        self.assertEqual([row["candidate_id"] for row in accepted], ["pass"])
        self.assertEqual(
            sorted(rejected),
            ["critic_rejected", "verifier_rejected"],
        )
        self.assertGreater(accepted[0]["token_count"], 0)

    def test_rejects_incomplete_critic_coverage(self):
        candidate = self._candidate("pass", "FINAL: 7", "3 + 4")

        with self.assertRaisesRegex(ValueError, "exactly cover"):
            accept_candidates(
                self.campaign,
                [candidate],
                [],
                tokenizer=FakeTokenizer(),
                skill_sha256=self.skill_sha256,
                family_ids={
                    family["family_id"]
                    for family in self.campaign["data_families"]
                },
                shard={
                    "shard_id": 0,
                    "attempt": 0,
                    "family_id": "verified-reasoning",
                    "seed": 1,
                },
            )

    def test_rejects_generator_self_critique(self):
        candidate = self._candidate("same-agent", "FINAL: 7", "3 + 4")
        decision = self._decision("same-agent", True, 0.95)
        decision["critic_receipt"]["request_id"] = candidate[
            "generator_receipt"
        ]["request_id"]

        accepted, rejected = accept_candidates(
            self.campaign,
            [candidate],
            [decision],
            tokenizer=FakeTokenizer(),
            skill_sha256=self.skill_sha256,
            family_ids={
                family["family_id"] for family in self.campaign["data_families"]
            },
            shard={
                "shard_id": 0,
                "attempt": 0,
                "family_id": "verified-reasoning",
                "seed": 1,
            },
        )

        self.assertEqual(accepted, [])
        self.assertEqual(rejected, ["critic_not_independent"])

    def test_rejects_benchmark_marker_in_messages(self):
        candidate = self._candidate("leak", "FINAL: 7", "3 + 4")
        candidate["messages"][1]["content"] = "Copy this MMLU answer."

        accepted, rejected = accept_candidates(
            self.campaign,
            [candidate],
            [self._decision("leak", True, 0.95)],
            tokenizer=FakeTokenizer(),
            skill_sha256=self.skill_sha256,
            family_ids={
                family["family_id"] for family in self.campaign["data_families"]
            },
            shard={
                "shard_id": 0,
                "attempt": 0,
                "family_id": "verified-reasoning",
                "seed": 1,
            },
        )

        self.assertEqual(accepted, [])
        self.assertEqual(rejected, ["candidate_invalid:forbidden_source"])

    def test_rejects_benchmark_marker_in_task_spec(self):
        candidate = self._candidate("task-leak", "FINAL: 7", "3 + 4")
        candidate["task_spec"]["provenance"] = "SWE-bench fixture"

        accepted, rejected = accept_candidates(
            self.campaign,
            [candidate],
            [self._decision("task-leak", True, 0.95)],
            tokenizer=FakeTokenizer(),
            skill_sha256=self.skill_sha256,
            family_ids={
                family["family_id"] for family in self.campaign["data_families"]
            },
            shard={
                "shard_id": 0,
                "attempt": 0,
                "family_id": "verified-reasoning",
                "seed": 1,
            },
        )

        self.assertEqual(accepted, [])
        self.assertEqual(rejected, ["candidate_invalid:forbidden_source"])

    def test_global_ledger_deduplicates_across_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._accepted_row("one")
            duplicate = copy.deepcopy(row)
            duplicate["sample_id"] = "different-id"
            write_jsonl(
                root / "shards/shard-00000/attempt-000/accepted.jsonl",
                [row],
            )
            write_jsonl(
                root / "shards/shard-00001/attempt-000/accepted.jsonl",
                [duplicate],
            )

            rebuild_accepted_ledger(root)
            merge = json.loads((root / "merge.json").read_text(encoding="utf-8"))

            self.assertEqual(merge["accepted_rows"], 1)
            self.assertEqual(
                merge["global_rejection_reasons"],
                {"duplicate_exact_hash": 1},
            )

    def test_ledger_recomputes_stale_semantic_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._accepted_row("stale-semantic")
            row["semantic_hash"] = "0" * 64
            write_jsonl(
                root / "shards/shard-00000/attempt-000/accepted.jsonl",
                [row],
            )

            output = rebuild_accepted_ledger(root)
            rebuilt = json.loads(output.read_text(encoding="utf-8"))

            self.assertNotEqual(rebuilt["semantic_hash"], "0" * 64)
            self.assertEqual(
                rebuilt["semantic_basis_version"],
                "family_task_spec_v1",
            )

    def test_audit_stays_blocked_and_creates_refill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = plan_campaign(
                CAMPAIGN,
                skill_path=SKILL,
                max_shards=5,
                candidate_samples_override=2,
            )
            write_plan(plan, root)
            write_jsonl(root / "accepted.jsonl", [self._accepted_row("one")])

            report = audit_campaign(
                CAMPAIGN,
                root,
                tokenizer=FakeTokenizer(),
            )
            refill = plan_refill(CAMPAIGN, root)

            self.assertFalse(report["training_unblocked"])
            self.assertGreater(len(refill["shards"]), 0)
            self.assertEqual(refill["mode"], "refill")

    def test_audit_rejects_tampered_verifier_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._accepted_row("tampered")
            row["verifier_receipt"]["passed"] = False
            write_jsonl(root / "accepted.jsonl", [row])

            report = audit_campaign(
                CAMPAIGN,
                root,
                tokenizer=FakeTokenizer(),
            )

            self.assertFalse(report["checks"]["verifier_revalidation_pass"])
            self.assertFalse(report["training_unblocked"])

    def test_audit_rejects_tampered_recipe_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._accepted_row("recipe-tamper")
            recipe = {
                "narrative_style": "technical",
                "evidence_label": "record",
                "instruction_order": "contract-first",
                "response_tone": "neutral",
            }
            row["generator_recipe"] = recipe
            row["source"]["compiler"] = "deterministic_recipe_v2"
            row["source"]["recipe_sha256"] = "0" * 64
            row["generator_receipt"]["recipe_sha256"] = "0" * 64
            row["critic_receipt"]["recipe_sha256"] = "0" * 64
            write_jsonl(root / "accepted.jsonl", [row])

            report = audit_campaign(
                CAMPAIGN,
                root,
                tokenizer=FakeTokenizer(),
            )

            self.assertFalse(report["checks"]["source_policy_pass"])
            self.assertFalse(report["training_unblocked"])

    def test_audit_fails_closed_on_malformed_generator_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._accepted_row("malformed")
            row["generator_receipt"] = "not-a-receipt"
            write_jsonl(root / "accepted.jsonl", [row])

            report = audit_campaign(
                CAMPAIGN,
                root,
                tokenizer=FakeTokenizer(),
            )

            self.assertFalse(report["checks"]["critic_revalidation_pass"])
            self.assertFalse(report["checks"]["skill_identity_pass"])
            self.assertFalse(report["training_unblocked"])

    def test_recipe_audit_fails_closed_when_call_budget_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = plan_campaign(
                CAMPAIGN_V2,
                skill_path=SKILL,
                max_shards=5,
                candidate_samples_override=2,
                candidate_tokens_per_sample_override=300,
            )
            write_plan(plan, root)
            write_jsonl(root / "accepted.jsonl", [])

            report = audit_campaign(
                CAMPAIGN_V2,
                root,
                tokenizer=FakeTokenizer(),
            )

            self.assertFalse(report["checks"]["recipe_call_budget_pass"])
            self.assertFalse(report["checks"]["dev_sample_target_pass"])
            self.assertFalse(report["training_unblocked"])

    def test_refill_shard_ids_advance_across_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = plan_campaign(
                CAMPAIGN,
                skill_path=SKILL,
                max_shards=5,
                candidate_samples_override=2,
            )
            write_plan(plan, root)
            write_jsonl(root / "accepted.jsonl", [])
            audit_campaign(CAMPAIGN, root, tokenizer=FakeTokenizer())

            first = plan_refill(CAMPAIGN, root)
            second = plan_refill(CAMPAIGN, root)

            first_ids = {row["shard_id"] for row in first["shards"]}
            second_ids = {row["shard_id"] for row in second["shards"]}
            self.assertFalse(first_ids & second_ids)

    def test_semantic_index_finds_close_rows(self):
        index = SemanticDuplicateIndex(0.8)
        left = "alpha beta gamma delta epsilon"
        index.add(left)

        matches = index.matches("alpha beta gamma delta epsilon extra")

        self.assertEqual(len(matches), 1)

    def test_semantic_dedup_uses_task_spec_not_padding(self):
        duplicate_rows = [
            {
                "family_id": "verified-reasoning",
                "task_spec": {"expression": "3 + 4"},
                "messages": [
                    {"role": "user", "content": "short padding"},
                ],
            },
            {
                "family_id": "verified-reasoning",
                "task_spec": {"expression": "3 + 4"},
                "messages": [
                    {"role": "user", "content": "completely different padding"},
                ],
            },
        ]
        distinct_rows = [
            duplicate_rows[0],
            {
                "family_id": "verified-reasoning",
                "task_spec": {"expression": "3 + 5"},
                "messages": duplicate_rows[0]["messages"],
            },
        ]

        duplicates = find_semantic_near_duplicates(
            duplicate_rows,
            threshold=0.92,
        )
        distinct = find_semantic_near_duplicates(
            distinct_rows,
            threshold=0.92,
        )

        self.assertEqual(len(duplicates), 1)
        self.assertEqual(distinct, [])

    def test_integer_allocations_are_exact(self):
        allocation = allocate_integer_total(7, {"a": 3, "b": 2})

        self.assertEqual(sum(allocation.values()), 7)
        self.assertEqual(sum(split_integer(13, 4)), 13)

    def test_accepts_mapping_tokenizer_output(self):
        accepted, rejected = accept_candidates(
            self.campaign,
            [self._candidate("mapping", "FINAL: 7", "3 + 4")],
            [self._decision("mapping", True, 0.95)],
            tokenizer=MappingTokenizer(),
            skill_sha256=self.skill_sha256,
            family_ids={
                family["family_id"] for family in self.campaign["data_families"]
            },
            shard={
                "shard_id": 0,
                "attempt": 0,
                "family_id": "verified-reasoning",
                "seed": 1,
            },
        )

        self.assertEqual(rejected, [])
        self.assertGreater(accepted[0]["token_count"], 0)

    def _candidate(self, candidate_id: str, output: str, expression: str):
        return {
            "schema_version": "nano_subagent_candidate_v1",
            "candidate_id": candidate_id,
            "family_id": "verified-reasoning",
            "task_family": "smoke",
            "split": "train",
            "skill_id": "skill-sft-campaign",
            "messages": [
                {"role": "system", "content": "Follow the contract."},
                {"role": "user", "content": f"Synthetic request {candidate_id}."},
                {"role": "assistant", "content": output},
            ],
            "source": {
                "kind": "procedurally_generated_synthetic",
                "generator": "unit-test",
                "seed": 1,
            },
            "task_spec": {"expression": expression},
            "verifier": {
                "kind": "safe_execution_receipt_v1",
            },
            "generator_receipt": {
                "request_id": f"generator-{candidate_id}",
                "skill_sha256": self.skill_sha256,
            },
        }

    def _decision(self, candidate_id: str, accept: bool, score: float):
        return {
            "schema_version": "nano_subagent_critic_v1",
            "candidate_id": candidate_id,
            "accept": accept,
            "score": score,
            "reasons": [],
            "critic_receipt": {
                "request_id": f"critic-{candidate_id}",
                "critic": "unit-test",
            },
        }

    def _accepted_row(self, candidate_id: str):
        shard = {
            "shard_id": 0,
            "attempt": 0,
            "family_id": "verified-reasoning",
            "seed": 1,
        }
        accepted, _ = accept_candidates(
            self.campaign,
            [self._candidate(candidate_id, "FINAL: 7", "3 + 4")],
            [self._decision(candidate_id, True, 0.95)],
            tokenizer=FakeTokenizer(),
            skill_sha256=self.skill_sha256,
            family_ids={
                family["family_id"] for family in self.campaign["data_families"]
            },
            shard=shard,
        )
        return accepted[0]


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
