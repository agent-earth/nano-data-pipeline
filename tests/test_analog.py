from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from nano_data_pipeline.analog import (
    build_curriculum_analog_dataset,
    build_format_analog_dataset,
    build_preservation_mix_dataset,
    build_process_trace_dataset,
    build_semantic_trace_dataset,
    evaluate_arithmetic,
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

    def test_builds_fresh_two_step_curriculum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            prior_path = root / "prior.json"
            self._feedback(feedback)
            prior = build_format_analog_dataset(feedback)
            prior_path.write_text(json.dumps(prior), encoding="utf-8")
            curriculum = build_curriculum_analog_dataset(
                feedback,
                prior_path,
            )

        self.assertEqual(curriculum["summary"]["samples"], 160)
        self.assertEqual(
            curriculum["summary"]["by_split"],
            {"train": 128, "validation": 32},
        )
        self.assertEqual(
            curriculum["summary"]["by_task_family"],
            {"choice_contract": 80, "numeric_contract": 80},
        )
        train = [
            sample for sample in curriculum["samples"] if sample["split"] == "train"
        ]
        validation = [
            sample
            for sample in curriculum["samples"]
            if sample["split"] == "validation"
        ]
        self.assertEqual(
            dict(Counter(sample["difficulty"] for sample in train)),
            {"single_step": 24, "two_step": 104},
        )
        self.assertEqual(
            dict(Counter(sample["difficulty"] for sample in validation)),
            {"single_step": 8, "two_step": 24},
        )
        self.assertEqual(curriculum["source"]["prior_exact_overlap"], 0)
        self.assertEqual(curriculum["source"]["prior_semantic_overlap"], 0)
        self.assertEqual(curriculum["source"]["prior_sample_id_overlap"], 0)
        self.assertFalse(curriculum["policy"]["observed_validation_reused"])

    def test_safe_arithmetic_rejects_code(self):
        self.assertEqual(evaluate_arithmetic("(7 + 5) * 3"), 36)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            evaluate_arithmetic("__import__('os').system('id')")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            evaluate_arithmetic("2 ** 10")

    def test_builds_verified_semantic_trace_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            v1_path = root / "v1.json"
            v2_path = root / "v2.json"
            self._feedback(feedback)
            v1 = build_format_analog_dataset(feedback)
            v1_path.write_text(json.dumps(v1), encoding="utf-8")
            v2 = build_curriculum_analog_dataset(feedback, v1_path)
            v2_path.write_text(json.dumps(v2), encoding="utf-8")
            traces = build_semantic_trace_dataset(
                feedback,
                [v1_path, v2_path],
            )

        self.assertEqual(traces["summary"]["samples"], 192)
        self.assertEqual(
            traces["summary"]["by_split"],
            {"train": 160, "validation": 32},
        )
        self.assertEqual(
            traces["summary"]["by_task_family"],
            {"semantic_arithmetic": 192},
        )
        self.assertEqual(traces["source"]["prior_sample_id_overlap"], 0)
        self.assertEqual(traces["source"]["prior_exact_overlap"], 0)
        self.assertEqual(traces["source"]["prior_semantic_overlap"], 0)
        self.assertTrue(
            traces["policy"]["all_targets_deterministically_verified"]
        )
        self.assertTrue(
            all(
                sample["format_family"] == "trace_numeric"
                for sample in traces["samples"]
            )
        )

    def test_trace_validator_rejects_tampered_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            v1_path = root / "v1.json"
            v2_path = root / "v2.json"
            self._feedback(feedback)
            v1 = build_format_analog_dataset(feedback)
            v1_path.write_text(json.dumps(v1), encoding="utf-8")
            v2 = build_curriculum_analog_dataset(feedback, v1_path)
            v2_path.write_text(json.dumps(v2), encoding="utf-8")
            traces = build_semantic_trace_dataset(feedback, [v1_path, v2_path])
        sample = traces["samples"][0]
        sample["messages"][-1]["content"] = sample["messages"][-1][
            "content"
        ].replace("FINAL: ", "FINAL: 999")
        with self.assertRaisesRegex(ValueError, "trace verifier mismatch|invalid trace"):
            validate_analog_dataset(traces)

    def test_builds_fresh_verified_process_trace_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            prior_paths = []
            self._feedback(feedback)
            v1 = build_format_analog_dataset(feedback)
            v1_path = root / "v1.json"
            v1_path.write_text(json.dumps(v1), encoding="utf-8")
            prior_paths.append(v1_path)
            v2 = build_curriculum_analog_dataset(feedback, v1_path)
            v2_path = root / "v2.json"
            v2_path.write_text(json.dumps(v2), encoding="utf-8")
            prior_paths.append(v2_path)
            v3 = build_semantic_trace_dataset(
                feedback,
                [v1_path, v2_path],
            )
            v3_path = root / "v3.json"
            v3_path.write_text(json.dumps(v3), encoding="utf-8")
            prior_paths.append(v3_path)
            process = build_process_trace_dataset(feedback, prior_paths)

        self.assertEqual(process["summary"]["samples"], 192)
        self.assertEqual(
            process["summary"]["by_split"],
            {"train": 160, "validation": 32},
        )
        self.assertEqual(
            process["summary"]["by_task_family"],
            {"semantic_arithmetic_process": 192},
        )
        self.assertEqual(
            process["summary"]["by_difficulty"],
            {"three_step": 96, "two_step": 96},
        )
        self.assertTrue(
            process["policy"]["all_intermediate_steps_verified"]
        )
        self.assertEqual(process["source"]["prior_sample_id_overlap"], 0)
        self.assertEqual(process["source"]["prior_exact_overlap"], 0)
        self.assertEqual(process["source"]["prior_semantic_overlap"], 0)
        self.assertEqual(
            process["source"]["prior_source_expression_overlap"],
            0,
        )
        self.assertTrue(
            all(
                sample["format_family"] == "process_trace_numeric"
                for sample in process["samples"]
            )
        )

    def test_process_validator_rejects_tampered_intermediate_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            v1_path = root / "v1.json"
            v2_path = root / "v2.json"
            v3_path = root / "v3.json"
            self._feedback(feedback)
            v1 = build_format_analog_dataset(feedback)
            v1_path.write_text(json.dumps(v1), encoding="utf-8")
            v2 = build_curriculum_analog_dataset(feedback, v1_path)
            v2_path.write_text(json.dumps(v2), encoding="utf-8")
            v3 = build_semantic_trace_dataset(
                feedback,
                [v1_path, v2_path],
            )
            v3_path.write_text(json.dumps(v3), encoding="utf-8")
            process = build_process_trace_dataset(
                feedback,
                [v1_path, v2_path, v3_path],
            )
        sample = process["samples"][0]
        target = sample["messages"][-1]["content"]
        first_result = sample["verifier"]["steps"][0]["expected_result"]
        sample["messages"][-1]["content"] = target.replace(
            f"= {first_result}",
            f"= {int(first_result) + 1}",
            1,
        )
        with self.assertRaisesRegex(
            ValueError,
            "process step verifier mismatch",
        ):
            validate_analog_dataset(process)

    def _prior_datasets(self, root: Path, feedback: Path) -> list[Path]:
        paths = []
        v1 = build_format_analog_dataset(feedback)
        v1_path = root / "v1.json"
        v1_path.write_text(json.dumps(v1), encoding="utf-8")
        paths.append(v1_path)
        v2 = build_curriculum_analog_dataset(feedback, v1_path)
        v2_path = root / "v2.json"
        v2_path.write_text(json.dumps(v2), encoding="utf-8")
        paths.append(v2_path)
        v3 = build_semantic_trace_dataset(feedback, paths)
        v3_path = root / "v3.json"
        v3_path.write_text(json.dumps(v3), encoding="utf-8")
        paths.append(v3_path)
        v4 = build_process_trace_dataset(feedback, paths)
        v4_path = root / "v4.json"
        v4_path.write_text(json.dumps(v4), encoding="utf-8")
        paths.append(v4_path)
        return paths

    def test_builds_hard_preservation_mix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            self._feedback(feedback)
            priors = self._prior_datasets(root, feedback)
            mix = build_preservation_mix_dataset(feedback, priors)

        self.assertEqual(mix["summary"]["samples"], 192)
        self.assertEqual(
            mix["summary"]["by_split"],
            {"train": 160, "validation": 32},
        )
        self.assertEqual(
            mix["summary"]["by_task_family"],
            {
                "capability_preservation_choice": 48,
                "capability_preservation_numeric": 96,
                "semantic_arithmetic_process": 48,
            },
        )
        self.assertEqual(mix["source"]["prior_sample_id_overlap"], 0)
        self.assertEqual(mix["source"]["prior_exact_overlap"], 0)
        self.assertEqual(mix["source"]["prior_semantic_overlap"], 0)
        self.assertEqual(mix["source"]["prior_source_signature_overlap"], 0)
        self.assertFalse(mix["policy"]["sealed_canary_used_for_training"])
        self.assertTrue(
            mix["policy"]["all_numeric_targets_deterministically_verified"]
        )
        self.assertTrue(
            all(sample["training_eligible"] for sample in mix["samples"])
        )

    def test_preservation_validator_rejects_tampered_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            self._feedback(feedback)
            mix = build_preservation_mix_dataset(
                feedback,
                self._prior_datasets(root, feedback),
            )
        sample = next(
            row
            for row in mix["samples"]
            if row["format_family"] == "reasoning_numeric"
        )
        expected = sample["verifier"]["expected_result"]
        sample["messages"][-1]["content"] = sample["messages"][-1][
            "content"
        ].replace(f"= {expected}", f"= {int(expected) + 1}", 1)
        with self.assertRaisesRegex(ValueError, "reasoning verifier mismatch"):
            validate_analog_dataset(mix)

    def test_preservation_builder_rejects_sealed_case_id_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            self._feedback(feedback)
            priors = self._prior_datasets(root, feedback)
            manifest = json.loads(feedback.read_text(encoding="utf-8"))
            manifest["rows"][0]["case_id"] = "capability_preservation_numeric"
            feedback.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sealed case IDs leaked"):
                build_preservation_mix_dataset(feedback, priors)


if __name__ == "__main__":
    unittest.main()
