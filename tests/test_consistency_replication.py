from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from transformers import AutoTokenizer

from nano_data_pipeline.consistency_replication import (
    _relation_grid,
    minimum_pairs_for_exact_mcnemar_power,
    validate_consistency_replication_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/paired_consistency_replication_v1.json"
TOKENIZER = ROOT / "../../../models/Qwen3.5-4B"
PRIOR = [ROOT / "datasets/skill_sft_execution_target_paired_v1.json"]
PRIOR_LEDGER = (
    ROOT
    / "../../../datasets/ultimate-distill/"
    "skill-sft-10k-10m-v2/accepted.jsonl"
)
SOURCE_RESULT = (
    ROOT.parent
    / "nano-train-skillgen-traex-02/docs/results/"
    "execution_target_paired_consistency_v1.public.json"
)


class ConsistencyReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER,
            local_files_only=True,
        )
        cls.dataset = json.loads(DATASET.read_text(encoding="utf-8"))

    def _validate(self, dataset: dict):
        return validate_consistency_replication_dataset(
            dataset,
            tokenizer=self.tokenizer,
            tokenizer_path=TOKENIZER,
            prior_dataset_paths=PRIOR,
            prior_accepted_jsonl_path=PRIOR_LEDGER,
            source_result_path=SOURCE_RESULT,
        )

    def test_sample_size_minimum_is_189_pairs(self):
        result = minimum_pairs_for_exact_mcnemar_power(
            observed_fix_rate=1 / 24,
            minimum_wins=6,
            target_probability=0.8,
        )
        self.assertEqual(result["minimum_pairs"], 189)
        self.assertGreaterEqual(result["achieved_probability"], 0.8)

    def test_relation_grid_is_balanced_and_disjoint(self):
        train = _relation_grid("train")
        dev = _relation_grid("dev")
        self.assertEqual(len(train), 192)
        self.assertEqual(len(dev), 192)
        self.assertFalse(set(train) & set(dev))
        self.assertEqual({value[2] for value in train}, {2, 3})
        self.assertEqual({value[2] for value in dev}, {2, 3})

    def test_committed_replication_release_passes(self):
        release = self._validate(self.dataset)
        self.assertTrue(release["training_unblocked"])
        self.assertTrue(all(release["checks"].values()))
        self.assertEqual(release["accepted"]["train_pairs"], 192)
        self.assertEqual(release["accepted"]["dev_pairs"], 192)

    def test_tampered_process_step_fails_closed(self):
        tampered = copy.deepcopy(self.dataset)
        process = next(
            row for row in tampered["samples"] if row["view"] == "process"
        )
        process["messages"][-1]["content"] = process["messages"][-1][
            "content"
        ].replace(" = ", " = 999", 1)
        release = self._validate(tampered)
        self.assertFalse(release["checks"]["process_verifier_pass"])
        self.assertFalse(release["training_unblocked"])

    def test_tampered_source_result_identity_fails_closed(self):
        tampered = copy.deepcopy(self.dataset)
        tampered["source"]["source_result_sha256"] = "0" * 64
        release = self._validate(tampered)
        self.assertFalse(release["checks"]["source_result_identity_pass"])
        self.assertFalse(release["training_unblocked"])


if __name__ == "__main__":
    unittest.main()
