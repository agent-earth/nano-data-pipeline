from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from nano_data_pipeline.campaign import (
    load_skill_sft_campaign,
    validate_skill_sft_campaign,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "manifests/skill_sft_campaign_v1.json"
CAMPAIGN_V2 = ROOT / "manifests/skill_sft_campaign_v2.json"


class SkillSFTCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))

    def test_committed_campaign_is_valid(self):
        loaded = load_skill_sft_campaign(CAMPAIGN)

        self.assertEqual(
            loaded["targets"]["accepted_train_samples_min"],
            10_000,
        )
        self.assertEqual(
            loaded["targets"]["accepted_train_tokens_min"],
            10_000_000,
        )
        self.assertEqual(len(loaded["data_families"]), 5)

    def test_rejects_optimality_claim(self):
        invalid = copy.deepcopy(self.campaign)
        invalid["claim_boundary"]["target_is_empirical_optimum"] = True

        with self.assertRaisesRegex(ValueError, "optimum"):
            validate_skill_sft_campaign(invalid)

    def test_rejects_missing_token_quota(self):
        invalid = copy.deepcopy(self.campaign)
        invalid["data_families"][0]["accepted_train_tokens_min"] -= 1

        with self.assertRaisesRegex(ValueError, "token quotas"):
            validate_skill_sft_campaign(invalid)

    def test_rejects_missing_forbidden_benchmark(self):
        invalid = copy.deepcopy(self.campaign)
        invalid["source_policy"]["forbidden_payload_sources"].remove("MMLU")

        with self.assertRaisesRegex(ValueError, "mmlu"):
            validate_skill_sft_campaign(invalid)

    def test_rejects_training_before_global_checks(self):
        invalid = copy.deepcopy(self.campaign)
        invalid["completion_contract"][
            "training_unblocks_only_after_all_checks"
        ] = False

        with self.assertRaisesRegex(ValueError, "training must remain blocked"):
            validate_skill_sft_campaign(invalid)

    def test_rejects_underprovisioned_shards(self):
        invalid = copy.deepcopy(self.campaign)
        invalid["sharding"]["initial_shards"] = 10

        with self.assertRaisesRegex(ValueError, "sample oversubscription"):
            validate_skill_sft_campaign(invalid)

    def test_recipe_overlay_is_pinned_and_requires_scale_gates(self):
        campaign = load_skill_sft_campaign(CAMPAIGN_V2)

        self.assertEqual(campaign["campaign_id"], "skill-sft-10k-10m-v2")
        self.assertEqual(
            campaign["generation_protocol"]["mode"],
            "recipe_per_shard_v1",
        )
        self.assertEqual(
            campaign["generation_protocol"]["generator_calls_per_shard_max"],
            1,
        )
        self.assertIn(
            "dev_sample_target_pass",
            campaign["completion_contract"]["required_checks"],
        )
        self.assertIn(
            "recipe_call_budget_pass",
            campaign["completion_contract"]["required_checks"],
        )
        self.assertEqual(
            campaign["overlay_receipt"]["base_sha256"],
            "e24a16404f4462d7b2cd09312489c05327ed9e7ec08dc608e7a01e7a14ef9b0d",
        )


if __name__ == "__main__":
    unittest.main()
