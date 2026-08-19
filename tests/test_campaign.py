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


if __name__ == "__main__":
    unittest.main()
