from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from nano_data_pipeline.analog import (
    build_curriculum_analog_dataset,
    build_choice_replay_dataset,
    build_failure_targeted_preservation_mix_dataset,
    build_format_analog_dataset,
    build_packing_isolation_preservation_mix_dataset,
    build_percentage_isolation_preservation_mix_dataset,
    build_schedule_isolation_preservation_mix_dataset,
    build_preservation_mix_dataset,
    build_process_trace_dataset,
    build_semantic_trace_dataset,
    build_targeted_preservation_mix_dataset,
    evaluate_arithmetic,
    validate_analog_dataset,
)
from nano_data_pipeline.choice_matrix import build_choice_capability_matrix
from nano_data_pipeline.choice_matrix_v2 import build_choice_verifier_matrix_v2
from nano_data_pipeline.choice_matrix_v3 import (
    build_choice_exact_replication_matrix_v3,
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

    def _development_report(self, path: Path, base: dict) -> None:
        host_validation_ids = [
            sample["sample_id"]
            for sample in base["samples"]
            if sample["split"] == "validation"
            and sample["generation_rule"]
            == "preservation_host_and_companion_count_v5"
        ]
        report = {
            "experiment_id": "hard-preservation-sft-smoke-v10",
            "post_sft_validation": {
                "by_family": {
                    "capability_preservation_numeric": {
                        "semantic_failure_sample_ids": host_validation_ids[:7],
                    }
                }
            },
        }
        path.write_text(json.dumps(report), encoding="utf-8")

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

    def test_builds_targeted_preservation_mix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            base_path = root / "v5.json"
            report_path = root / "v10.public.json"
            self._feedback(feedback)
            priors = self._prior_datasets(root, feedback)
            base = build_preservation_mix_dataset(feedback, priors)
            base_path.write_text(json.dumps(base), encoding="utf-8")
            self._development_report(report_path, base)
            targeted = build_targeted_preservation_mix_dataset(
                feedback,
                base_path,
                report_path,
                priors,
            )

        self.assertEqual(targeted["summary"], base["summary"])
        self.assertEqual(targeted["source"]["replacement_count"], 16)
        self.assertEqual(targeted["source"]["prior_sample_id_overlap"], 0)
        self.assertEqual(targeted["source"]["prior_exact_overlap"], 0)
        self.assertEqual(targeted["source"]["prior_semantic_overlap"], 0)
        self.assertEqual(
            targeted["source"]["prior_source_signature_overlap"],
            0,
        )
        self.assertEqual(
            targeted["source"]["development_evidence"][
                "base_host_multiplier_support"
            ],
            {"train": {"3": 8, "4": 8}, "validation": {"2": 8}},
        )
        self.assertEqual(
            targeted["source"]["development_evidence"][
                "failure_generation_rules"
            ],
            {"preservation_host_and_companion_count_v5": 7},
        )
        self.assertTrue(targeted["policy"]["observed_validation_reused"])
        self.assertEqual(
            targeted["policy"]["validation_role"],
            "development_gate_only",
        )
        base_validation = [
            sample for sample in base["samples"] if sample["split"] == "validation"
        ]
        targeted_validation = [
            sample
            for sample in targeted["samples"]
            if sample["split"] == "validation"
        ]
        self.assertEqual(targeted_validation, base_validation)
        changed_positions = [
            index
            for index, (before, after) in enumerate(
                zip(base["samples"], targeted["samples"])
            )
            if before != after
        ]
        self.assertEqual(len(changed_positions), 16)
        for index in changed_positions:
            before = base["samples"][index]
            after = targeted["samples"][index]
            self.assertEqual(before["split"], "train")
            self.assertEqual(
                before["generation_rule"],
                "preservation_host_and_companion_count_v5",
            )
            self.assertEqual(
                after["generation_rule"],
                "targeted_host_two_count_v6",
            )
            self.assertIn(
                "every participant arrives with exactly 2 helpers",
                after["messages"][1]["content"],
            )

    def test_targeted_preservation_build_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            base_path = root / "v5.json"
            report_path = root / "v10.public.json"
            self._feedback(feedback)
            priors = self._prior_datasets(root, feedback)
            base = build_preservation_mix_dataset(feedback, priors)
            base_path.write_text(json.dumps(base), encoding="utf-8")
            self._development_report(report_path, base)
            first = build_targeted_preservation_mix_dataset(
                feedback,
                base_path,
                report_path,
                priors,
            )
            second = build_targeted_preservation_mix_dataset(
                feedback,
                base_path,
                report_path,
                priors,
            )
        self.assertEqual(first, second)

    def _failure_family_receipt(self, path: Path) -> None:
        receipt = {
            "schema_version": "nano_harness_failure_family_receipt_v1",
            "receipt_id": "test-failure-families-v1",
            "source": {
                "source_case_id_set_sha256": "0" * 64,
            },
            "families": [
                {
                    "family": family,
                    "count": 1,
                    "task_kind": task_kind,
                }
                for family, task_kind in (
                    (
                        "percentage_increase_total_composition",
                        "numeric_reasoning",
                    ),
                    (
                        "packing_efficiency_effective_volume",
                        "numeric_reasoning",
                    ),
                    (
                        "weighted_recurring_schedule_total",
                        "numeric_reasoning",
                    ),
                    (
                        "developmental_perception_experience_choice",
                        "choice_reasoning",
                    ),
                )
            ],
            "policy": {
                "contains_case_ids": False,
                "contains_prompts": False,
                "contains_references": False,
                "contains_predictions": False,
                "contains_raw_outputs": False,
                "direct_training_allowed": False,
                "fresh_analog_generation_allowed": True,
            },
        }
        path.write_text(json.dumps(receipt), encoding="utf-8")

    def test_builds_failure_targeted_preservation_mix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            report_path = root / "v10.public.json"
            targeted_path = root / "v6.json"
            receipt_path = root / "families.json"
            self._feedback(feedback)
            priors = self._prior_datasets(root, feedback)
            v5 = build_preservation_mix_dataset(feedback, priors)
            v5_path = root / "v5.json"
            v5_path.write_text(json.dumps(v5), encoding="utf-8")
            self._development_report(report_path, v5)
            v6 = build_targeted_preservation_mix_dataset(
                feedback,
                v5_path,
                report_path,
                priors,
            )
            targeted_path.write_text(json.dumps(v6), encoding="utf-8")
            self._failure_family_receipt(receipt_path)
            v7 = build_failure_targeted_preservation_mix_dataset(
                feedback,
                receipt_path,
                targeted_path,
                [*priors, v5_path],
            )

        self.assertEqual(v7["summary"], v6["summary"])
        self.assertEqual(v7["source"]["replacement_count"], 24)
        self.assertEqual(
            v7["source"]["replacement_family_counts"],
            {
                "packing_efficiency_effective_volume": 8,
                "percentage_increase_total_composition": 8,
                "weighted_recurring_schedule_total": 8,
            },
        )
        self.assertEqual(
            v7["source"]["deferred_feedback_families"],
            ["developmental_perception_experience_choice"],
        )
        self.assertEqual(v7["source"]["prior_sample_id_overlap"], 0)
        self.assertEqual(v7["source"]["prior_exact_overlap"], 0)
        self.assertEqual(v7["source"]["prior_semantic_overlap"], 0)
        self.assertEqual(v7["source"]["prior_source_signature_overlap"], 0)
        self.assertFalse(
            v7["policy"]["independent_holdout_used_for_training"]
        )
        self.assertEqual(
            [row for row in v7["samples"] if row["split"] == "validation"],
            [row for row in v6["samples"] if row["split"] == "validation"],
        )
        changed = [
            (before, after)
            for before, after in zip(v6["samples"], v7["samples"])
            if before != after
        ]
        self.assertEqual(len(changed), 24)
        self.assertTrue(
            all(before["split"] == after["split"] == "train" for before, after in changed)
        )
        self.assertTrue(
            all(
                after["generation_rule"].startswith("failure_targeted_")
                for _, after in changed
            )
        )

    def test_failure_targeted_builder_rejects_payload_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            report_path = root / "v10.public.json"
            targeted_path = root / "v6.json"
            receipt_path = root / "families.json"
            self._feedback(feedback)
            priors = self._prior_datasets(root, feedback)
            v5 = build_preservation_mix_dataset(feedback, priors)
            v5_path = root / "v5.json"
            v5_path.write_text(json.dumps(v5), encoding="utf-8")
            self._development_report(report_path, v5)
            v6 = build_targeted_preservation_mix_dataset(
                feedback,
                v5_path,
                report_path,
                priors,
            )
            targeted_path.write_text(json.dumps(v6), encoding="utf-8")
            self._failure_family_receipt(receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["policy"]["contains_prompts"] = True
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "boundary"):
                build_failure_targeted_preservation_mix_dataset(
                    feedback,
                    receipt_path,
                    targeted_path,
                    [*priors, v5_path],
                )

    def test_builds_percentage_isolation_preservation_mix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            report_path = root / "v10.public.json"
            receipt_path = root / "families.json"
            self._feedback(feedback)
            priors = self._prior_datasets(root, feedback)
            v5 = build_preservation_mix_dataset(feedback, priors)
            v5_path = root / "v5.json"
            v5_path.write_text(json.dumps(v5), encoding="utf-8")
            self._development_report(report_path, v5)
            v6 = build_targeted_preservation_mix_dataset(
                feedback,
                v5_path,
                report_path,
                priors,
            )
            v6_path = root / "v6.json"
            v6_path.write_text(json.dumps(v6), encoding="utf-8")
            self._failure_family_receipt(receipt_path)
            v7 = build_failure_targeted_preservation_mix_dataset(
                feedback,
                receipt_path,
                v6_path,
                [*priors, v5_path],
            )
            v7_path = root / "v7.json"
            v7_path.write_text(json.dumps(v7), encoding="utf-8")
            v8 = build_percentage_isolation_preservation_mix_dataset(
                feedback,
                receipt_path,
                v6_path,
                v7_path,
                [*priors, v5_path],
            )

        self.assertEqual(v8["summary"], v6["summary"])
        self.assertEqual(v8["source"]["replacement_count"], 8)
        self.assertEqual(
            v8["source"]["replacement_family_counts"],
            {"percentage_increase_total_composition": 8},
        )
        self.assertEqual(
            v8["source"]["deferred_feedback_families"],
            [
                "packing_efficiency_effective_volume",
                "weighted_recurring_schedule_total",
                "developmental_perception_experience_choice",
            ],
        )
        self.assertEqual(v8["source"]["prior_sample_id_overlap"], 0)
        self.assertEqual(v8["source"]["prior_exact_overlap"], 0)
        self.assertEqual(v8["source"]["prior_semantic_overlap"], 0)
        self.assertEqual(v8["source"]["prior_source_signature_overlap"], 0)
        self.assertFalse(
            v8["policy"]["independent_holdout_used_for_training"]
        )
        changed = [
            (before, after)
            for before, after in zip(v6["samples"], v8["samples"])
            if before != after
        ]
        self.assertEqual(len(changed), 8)
        self.assertTrue(
            all(
                before["split"] == after["split"] == "train"
                for before, after in changed
            )
        )
        self.assertTrue(
            all(
                after["generation_rule"]
                == "failure_targeted_percentage_increase_total_composition_v7"
                for _, after in changed
            )
        )
        self.assertEqual(
            [row for row in v8["samples"] if row["split"] == "validation"],
            [row for row in v6["samples"] if row["split"] == "validation"],
        )

    def test_builds_packing_isolation_preservation_mix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            report_path = root / "v10.public.json"
            receipt_path = root / "families.json"
            self._feedback(feedback)
            priors = self._prior_datasets(root, feedback)
            v5 = build_preservation_mix_dataset(feedback, priors)
            v5_path = root / "v5.json"
            v5_path.write_text(json.dumps(v5), encoding="utf-8")
            self._development_report(report_path, v5)
            v6 = build_targeted_preservation_mix_dataset(
                feedback,
                v5_path,
                report_path,
                priors,
            )
            v6_path = root / "v6.json"
            v6_path.write_text(json.dumps(v6), encoding="utf-8")
            self._failure_family_receipt(receipt_path)
            v7 = build_failure_targeted_preservation_mix_dataset(
                feedback,
                receipt_path,
                v6_path,
                [*priors, v5_path],
            )
            v7_path = root / "v7.json"
            v7_path.write_text(json.dumps(v7), encoding="utf-8")
            v9 = build_packing_isolation_preservation_mix_dataset(
                feedback,
                receipt_path,
                v6_path,
                v7_path,
                [*priors, v5_path],
            )

        self.assertEqual(v9["summary"], v6["summary"])
        self.assertEqual(v9["source"]["replacement_count"], 8)
        self.assertEqual(
            v9["source"]["replacement_family_counts"],
            {"packing_efficiency_effective_volume": 8},
        )
        self.assertEqual(v9["source"]["prior_sample_id_overlap"], 0)
        self.assertEqual(v9["source"]["prior_exact_overlap"], 0)
        self.assertEqual(v9["source"]["prior_semantic_overlap"], 0)
        self.assertEqual(v9["source"]["prior_source_signature_overlap"], 0)
        changed = [
            (before, after)
            for before, after in zip(v6["samples"], v9["samples"])
            if before != after
        ]
        self.assertEqual(len(changed), 8)
        self.assertTrue(
            all(
                before["split"] == after["split"] == "train"
                for before, after in changed
            )
        )
        self.assertTrue(
            all(
                after["generation_rule"]
                == "failure_targeted_packing_efficiency_effective_volume_v7"
                for _, after in changed
            )
        )
        self.assertEqual(
            [row for row in v9["samples"] if row["split"] == "validation"],
            [row for row in v6["samples"] if row["split"] == "validation"],
        )

    def test_builds_schedule_isolation_preservation_mix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            report_path = root / "v10.public.json"
            receipt_path = root / "families.json"
            self._feedback(feedback)
            priors = self._prior_datasets(root, feedback)
            v5 = build_preservation_mix_dataset(feedback, priors)
            v5_path = root / "v5.json"
            v5_path.write_text(json.dumps(v5), encoding="utf-8")
            self._development_report(report_path, v5)
            v6 = build_targeted_preservation_mix_dataset(
                feedback, v5_path, report_path, priors
            )
            v6_path = root / "v6.json"
            v6_path.write_text(json.dumps(v6), encoding="utf-8")
            self._failure_family_receipt(receipt_path)
            v7 = build_failure_targeted_preservation_mix_dataset(
                feedback, receipt_path, v6_path, [*priors, v5_path]
            )
            v7_path = root / "v7.json"
            v7_path.write_text(json.dumps(v7), encoding="utf-8")
            v10 = build_schedule_isolation_preservation_mix_dataset(
                feedback,
                receipt_path,
                v6_path,
                v7_path,
                [*priors, v5_path],
            )
        self.assertEqual(v10["summary"], v6["summary"])
        self.assertEqual(v10["source"]["replacement_count"], 8)
        self.assertEqual(
            v10["source"]["replacement_family_counts"],
            {"weighted_recurring_schedule_total": 8},
        )
        changed = [
            (before, after)
            for before, after in zip(v6["samples"], v10["samples"])
            if before != after
        ]
        self.assertEqual(len(changed), 8)
        self.assertTrue(
            all(
                after["generation_rule"]
                == "failure_targeted_weighted_recurring_schedule_total_v7"
                for _, after in changed
            )
        )
        self.assertEqual(
            [row for row in v10["samples"] if row["split"] == "validation"],
            [row for row in v6["samples"] if row["split"] == "validation"],
        )

    def test_builds_generic_choice_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            report_path = root / "v10.public.json"
            self._feedback(feedback)
            priors = self._prior_datasets(root, feedback)
            v5 = build_preservation_mix_dataset(feedback, priors)
            v5_path = root / "v5.json"
            v5_path.write_text(json.dumps(v5), encoding="utf-8")
            self._development_report(report_path, v5)
            v6 = build_targeted_preservation_mix_dataset(
                feedback, v5_path, report_path, priors
            )
            v6_path = root / "v6.json"
            v6_path.write_text(json.dumps(v6), encoding="utf-8")
            replay = build_choice_replay_dataset(v6_path)
        self.assertEqual(replay["summary"]["samples"], 72)
        self.assertEqual(
            replay["summary"]["by_split"],
            {"train": 40, "validation": 32},
        )
        self.assertTrue(
            all(
                row["task_family"] == "capability_preservation_choice"
                for row in replay["samples"]
                if row["split"] == "train"
            )
        )
        self.assertEqual(
            [row for row in replay["samples"] if row["split"] == "validation"],
            [row for row in v6["samples"] if row["split"] == "validation"],
        )
        self.assertFalse(
            replay["policy"]["independent_holdout_used_for_training"]
        )

    def test_builds_history_disjoint_choice_capability_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            report_path = root / "v10.public.json"
            self._feedback(feedback)
            priors = self._prior_datasets(root, feedback)
            v5 = build_preservation_mix_dataset(feedback, priors)
            v5_path = root / "v5.json"
            v5_path.write_text(json.dumps(v5), encoding="utf-8")
            self._development_report(report_path, v5)
            v6 = build_targeted_preservation_mix_dataset(
                feedback, v5_path, report_path, priors
            )
            v6_path = root / "v6.json"
            v6_path.write_text(json.dumps(v6), encoding="utf-8")
            matrix = build_choice_capability_matrix(
                [*priors, v5_path, v6_path]
            )
        self.assertEqual(matrix["summary"]["cases"], 48)
        self.assertEqual(
            set(matrix["summary"]["by_family"].values()),
            {8},
        )
        self.assertEqual(matrix["summary"]["scored_cases"], 32)
        self.assertEqual(matrix["summary"]["ambiguity_cases"], 16)
        self.assertEqual(matrix["summary"]["training_eligible_cases"], 0)
        self.assertEqual(
            matrix["summary"]["by_expected_route"],
            {
                "ambiguous_fallback": 16,
                "unsupported_fallback": 24,
                "verified_override": 8,
            },
        )
        self.assertTrue(
            all(
                matrix["source"][key] == 0
                for key in (
                    "prior_case_id_overlap",
                    "prior_exact_overlap",
                    "prior_semantic_overlap",
                    "prior_prompt_overlap",
                    "prior_source_signature_overlap",
                )
            )
        )
        self.assertTrue(
            all(
                row["reference"] is None
                for row in matrix["cases"]
                if row["family"]
                in {
                    "explicit_average_no_exact_option",
                    "duplicate_option_ambiguity",
                }
            )
        )
        self.assertFalse(matrix["policy"]["training_allowed"])

    def test_builds_history_disjoint_choice_verifier_matrix_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            report_path = root / "v10.public.json"
            self._feedback(feedback)
            priors = self._prior_datasets(root, feedback)
            v5 = build_preservation_mix_dataset(feedback, priors)
            v5_path = root / "v5.json"
            v5_path.write_text(json.dumps(v5), encoding="utf-8")
            self._development_report(report_path, v5)
            v6 = build_targeted_preservation_mix_dataset(
                feedback, v5_path, report_path, priors
            )
            v6_path = root / "v6.json"
            v6_path.write_text(json.dumps(v6), encoding="utf-8")
            matrix_v1 = build_choice_capability_matrix(
                [*priors, v5_path, v6_path]
            )
            matrix_v1_path = root / "matrix-v1.json"
            matrix_v1_path.write_text(json.dumps(matrix_v1), encoding="utf-8")
            matrix_v2 = build_choice_verifier_matrix_v2(
                [*priors, v5_path, v6_path],
                [matrix_v1_path],
            )
        self.assertEqual(matrix_v2["summary"]["cases"], 48)
        self.assertEqual(
            set(matrix_v2["summary"]["by_family"].values()),
            {8},
        )
        self.assertEqual(matrix_v2["summary"]["scored_cases"], 16)
        self.assertEqual(matrix_v2["summary"]["ambiguity_cases"], 32)
        self.assertEqual(matrix_v2["summary"]["training_eligible_cases"], 0)
        self.assertEqual(
            matrix_v2["summary"]["by_expected_route"],
            {"ambiguous_fallback": 32, "verified_override": 16},
        )
        self.assertTrue(
            all(
                matrix_v2["source"][key] == 0
                for key in (
                    "prior_case_id_overlap",
                    "prior_exact_overlap",
                    "prior_semantic_overlap",
                    "prior_prompt_overlap",
                    "prior_source_signature_overlap",
                )
            )
        )
        self.assertTrue(
            all(
                row["reference"] is None
                for row in matrix_v2["cases"]
                if row["expected_route"] == "ambiguous_fallback"
            )
        )
        self.assertFalse(matrix_v2["policy"]["training_allowed"])

    def test_builds_history_disjoint_choice_exact_replication_matrix_v3(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.json"
            report_path = root / "v10.public.json"
            self._feedback(feedback)
            priors = self._prior_datasets(root, feedback)
            v5 = build_preservation_mix_dataset(feedback, priors)
            v5_path = root / "v5.json"
            v5_path.write_text(json.dumps(v5), encoding="utf-8")
            self._development_report(report_path, v5)
            v6 = build_targeted_preservation_mix_dataset(
                feedback, v5_path, report_path, priors
            )
            v6_path = root / "v6.json"
            v6_path.write_text(json.dumps(v6), encoding="utf-8")
            matrix_v1 = build_choice_capability_matrix(
                [*priors, v5_path, v6_path]
            )
            matrix_v1_path = root / "matrix-v1.json"
            matrix_v1_path.write_text(json.dumps(matrix_v1), encoding="utf-8")
            matrix_v2 = build_choice_verifier_matrix_v2(
                [*priors, v5_path, v6_path],
                [matrix_v1_path],
            )
            matrix_v2_path = root / "matrix-v2.json"
            matrix_v2_path.write_text(json.dumps(matrix_v2), encoding="utf-8")
            matrix_v3 = build_choice_exact_replication_matrix_v3(
                [*priors, v5_path, v6_path],
                [matrix_v1_path, matrix_v2_path],
            )
        self.assertEqual(matrix_v3["summary"]["cases"], 32)
        self.assertEqual(
            matrix_v3["summary"]["by_family"],
            {
                "host_count_exact_replication": 16,
                "verbal_average_exact_replication": 16,
            },
        )
        self.assertEqual(matrix_v3["summary"]["scored_cases"], 32)
        self.assertEqual(matrix_v3["summary"]["ambiguity_cases"], 0)
        self.assertEqual(matrix_v3["summary"]["training_eligible_cases"], 0)
        self.assertEqual(
            matrix_v3["summary"]["by_expected_route"],
            {"verified_override": 32},
        )
        self.assertTrue(
            all(
                matrix_v3["source"][key] == 0
                for key in (
                    "prior_case_id_overlap",
                    "prior_exact_overlap",
                    "prior_semantic_overlap",
                    "prior_prompt_overlap",
                    "prior_source_signature_overlap",
                )
            )
        )
        self.assertTrue(matrix_v3["policy"]["evaluation_only"])
        self.assertFalse(matrix_v3["policy"]["training_allowed"])


if __name__ == "__main__":
    unittest.main()
