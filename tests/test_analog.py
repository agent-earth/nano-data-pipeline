from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nano_data_pipeline.analog import (
    build_format_analog_dataset,
    validate_analog_dataset,
)


class AnalogTests(unittest.TestCase):
    def _feedback(self, path: Path) -> None:
        manifest = {
            "schema_version": "nano_feedback_manifest_v1",
            "dataset_id": "feedback-test-v1",
            "policy": {
                "contains_raw_outputs": False,
                "contains_prompts": False,
                "contains_references": False,
                "contains_predictions": False,
                "direct_training_allowed": False,
            },
            "rows": [
                {
                    "case_id": "mmlu-sealed",
                    "benchmark": "mmlu",
                    "paired_outcome": "four_b_only",
                    "failure_family": "format",
                    "four_b": {
                        "correct": True,
                        "format_class": "parseable",
                        "format_letter_matches_reference": None,
                    },
                    "nine_b": {
                        "correct": False,
                        "format_class": "final_missing_colon",
                        "format_letter_matches_reference": True,
                    },
                    "source_split": "sealed_eval_feedback",
                    "training_eligible": False,
                }
            ],
            "summary": {
                "rows": 1,
                "by_benchmark": {"mmlu": 1},
                "by_failure_family": {"format": 1},
                "by_paired_outcome": {"four_b_only": 1},
                "training_eligible_rows": 0,
            },
        }
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_builds_balanced_deduplicated_analog_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            feedback = Path(directory) / "feedback.json"
            self._feedback(feedback)
            dataset = build_format_analog_dataset(feedback)

        self.assertEqual(dataset["summary"]["samples"], 128)
        self.assertEqual(
            dataset["summary"]["by_task_family"],
            {"choice_contract": 64, "numeric_contract": 64},
        )
        self.assertEqual(
            dataset["summary"]["by_split"],
            {"train": 102, "validation": 26},
        )
        self.assertEqual(dataset["summary"]["unique_exact_hashes"], 128)
        self.assertEqual(dataset["summary"]["unique_semantic_hashes"], 128)
        self.assertTrue(
            all(sample["training_eligible"] for sample in dataset["samples"])
        )
        self.assertTrue(
            all(
                sample["sample_id"].startswith("synthetic-")
                for sample in dataset["samples"]
            )
        )
        self.assertNotIn(
            "mmlu-sealed",
            json.dumps(dataset, sort_keys=True),
        )

    def test_build_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            feedback = Path(directory) / "feedback.json"
            self._feedback(feedback)
            first = build_format_analog_dataset(feedback)
            second = build_format_analog_dataset(feedback)
        self.assertEqual(first, second)

    def test_validator_rejects_semantic_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            feedback = Path(directory) / "feedback.json"
            self._feedback(feedback)
            dataset = build_format_analog_dataset(feedback)
        dataset["samples"][1]["semantic_sha256"] = dataset["samples"][0][
            "semantic_sha256"
        ]
        dataset["summary"]["unique_semantic_hashes"] -= 1
        with self.assertRaisesRegex(ValueError, "semantic-deduplicated"):
            validate_analog_dataset(dataset)

    def test_builder_rejects_sealed_case_id_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            feedback = Path(directory) / "feedback.json"
            self._feedback(feedback)
            manifest = json.loads(feedback.read_text(encoding="utf-8"))
            manifest["rows"][0]["case_id"] = "choice_contract"
            feedback.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sealed case IDs leaked"):
                build_format_analog_dataset(feedback)


if __name__ == "__main__":
    unittest.main()
