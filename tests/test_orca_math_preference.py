from __future__ import annotations

import json
import unittest
from pathlib import Path

from transformers import AutoTokenizer

from nano_data_pipeline.orca_math_preference import (
    build_dataset,
    build_preregister,
    load_config,
    rejected_value,
    replace_final,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/orca_math_preference_v1.json"


class OrcaMathPreferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)

    def test_contract_is_frozen_and_closed(self):
        self.assertEqual(self.config.raw["selection"]["train_rows"], 512)
        self.assertEqual(self.config.raw["selection"]["dev_rows"], 192)
        self.assertFalse(
            self.config.raw["training_boundary"]["prior_sft_rows_reused"]
        )
        self.assertFalse(
            self.config.raw["training_boundary"]["rl_or_opd_unlocked"]
        )

    def test_rejected_value_is_deterministic_and_wrong(self):
        first = rejected_value("3/4", "sample-a")
        second = rejected_value("3/4", "sample-a")
        self.assertEqual(first, second)
        self.assertNotEqual(first, "3/4")
        self.assertEqual(
            replace_final("reasoning\nFINAL: 3/4", "3/4", first),
            f"reasoning\nFINAL: {first}",
        )

    def test_preregister_is_deterministic_and_contains_no_rows(self):
        first = build_preregister(self.config)
        second = build_preregister(self.config)
        self.assertEqual(first, second)
        serialized = json.dumps(first).lower()
        self.assertNotIn("prompt_messages", serialized)
        self.assertTrue(
            first["training_boundary"]["this_receipt_only_preregisters"]
        )

    def test_builder_excludes_prior_sft_and_changes_only_final(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.resolve(self.config.raw["tokenizer"]["path"]),
            local_files_only=True,
        )
        rows, _ = build_dataset(self.config, tokenizer=tokenizer)
        self.assertEqual(len(rows), 704)
        prior = json.loads(
            self.config.resolve(
                self.config.raw["prior_sft"]["preregister_path"]
            ).read_text(encoding="utf-8")
        )
        prior_ids = set(prior["selection"]["train_sample_ids"]) | set(
            prior["selection"]["dev_sample_ids"]
        )
        self.assertFalse(
            prior_ids & {row["source_sample_id"] for row in rows}
        )
        for row in rows[:20]:
            self.assertEqual(
                row["rejected"],
                replace_final(
                    row["chosen"],
                    row["expected"],
                    row["rejected_value"],
                ),
            )


if __name__ == "__main__":
    unittest.main()
