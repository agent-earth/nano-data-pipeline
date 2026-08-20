from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_data_pipeline.router_negative_diversity import (
    build_audit,
    is_explicit_classification_prompt,
    load_config,
    next_contract,
    summarize_source,
)
from scripts.render_router_negative_diversity_audit_v2 import render_markdown


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/router_classification/"
    "qwen35_router_negative_diversity_v2.json"
)
DATASET = ROOT / "datasets/qwen35_router_classification_v1.json"


class RouterNegativeDiversityTests(unittest.TestCase):
    def test_config_freezes_balanced_expanded_contract(self):
        config = load_config(CONFIG)
        self.assertEqual(config.seed, 20260827)
        self.assertEqual(len(config.negative_subtypes), 8)
        self.assertEqual(config.train_rows_per_positive_label, 2048)
        self.assertEqual(config.train_rows_per_negative_subtype, 256)
        self.assertEqual(config.dev_rows_per_positive_label, 512)
        self.assertEqual(config.dev_rows_per_negative_subtype, 64)
        self.assertEqual(
            config.minimum_train_templates_per_negative_subtype, 16
        )
        self.assertEqual(config.minimum_train_tokens, 600_000)

    def test_source_has_one_template_per_subtype_and_no_answer_tasks(self):
        dataset = json.loads(DATASET.read_text(encoding="utf-8"))
        source = summarize_source(dataset)
        self.assertEqual(source["negative_rows"], 320)
        self.assertEqual(
            source["negative_subtypes"],
            [
                "box_total",
                "paired_average",
                "remaining_stock",
                "single_operation",
            ],
        )
        for split, rows in (
            ("train", 256),
            ("validation", 64),
        ):
            summary = source["by_split"][split]
            self.assertEqual(summary["rows"], rows)
            self.assertEqual(summary["explicit_classification_rows"], rows)
            self.assertEqual(summary["answer_task_rows"], 0)
            for subtype in summary["by_subtype"].values():
                self.assertEqual(subtype["unique_template_ids"], 1)
                self.assertEqual(subtype["unique_generation_rules"], 1)
                self.assertEqual(
                    subtype["explicit_classification_rows"],
                    subtype["rows"],
                )
                self.assertEqual(subtype["answer_task_rows"], 0)

    def test_explicit_classification_detector_separates_answer_tasks(self):
        self.assertTrue(
            is_explicit_classification_prompt(
                "Which router class applies to this exact total?"
            )
        )
        self.assertTrue(
            is_explicit_classification_prompt(
                "Classify the task asking for remaining stock."
            )
        )
        self.assertFalse(
            is_explicit_classification_prompt(
                "Compute the exact inventory total."
            )
        )
        self.assertFalse(
            is_explicit_classification_prompt(
                "How many units remain after the batches are used?"
            )
        )

    def test_next_contract_preserves_equal_class_prior(self):
        contract = next_contract(load_config(CONFIG))
        self.assertEqual(
            contract["rows"],
            {"train": 6144, "dev": 1536, "total": 7680},
        )
        self.assertEqual(
            contract["train_by_label"],
            {"A": 2048, "B": 2048, "C": 2048},
        )
        self.assertEqual(
            contract["dev_by_label"],
            {"A": 512, "B": 512, "C": 512},
        )
        self.assertEqual(len(contract["negative_subtypes"]), 8)
        self.assertTrue(
            all(
                row["train_rows"] == 256
                and row["dev_rows"] == 64
                and row["minimum_train_templates"] == 16
                and row["minimum_dev_templates"] == 4
                for row in contract["negative_subtypes"].values()
            )
        )
        self.assertEqual(
            contract["lexical_contract"][
                "minimum_answer_task_fraction_train"
            ],
            0.75,
        )
        self.assertEqual(
            contract["lexical_contract"]["minimum_answer_task_fraction_dev"],
            1.0,
        )
        self.assertTrue(
            contract["provenance"]["integration_v1_v2_rows_used"] is False
        )
        self.assertTrue(
            contract["provenance"]["integration_v1_v2_outputs_used"] is False
        )
        self.assertTrue(contract["training_unblocked_only_after_release"])

    def test_audit_attributes_public_box_total_gap_without_eval_rows(self):
        audit = build_audit(load_config(CONFIG))
        self.assertTrue(
            audit["findings"]["negative_row_count_is_not_the_primary_gap"]
        )
        self.assertTrue(
            audit["findings"]["one_template_per_subtype_per_split"]
        )
        self.assertTrue(
            audit["findings"][
                "all_negative_rows_explicitly_ask_for_classification"
            ]
        )
        self.assertTrue(
            audit["findings"]["negative_answer_task_rows_zero"]
        )
        self.assertTrue(
            audit["findings"]["serving_namespace_issue_excluded"]
        )
        self.assertTrue(
            audit["findings"]["box_total_specific_generalization_gap"]
        )
        observed = audit["observed_public_failure"]
        self.assertEqual(observed["box_total_route_correct"], 0)
        self.assertEqual(observed["remaining_stock_route_correct"], 32)
        self.assertFalse(observed["integration_rows_or_outputs_loaded"])
        self.assertEqual(observed["evidence_kind"], "public_aggregate_only")
        decision = audit["decision"]
        self.assertFalse(decision["reuse_v1_data_unchanged"])
        self.assertTrue(decision["generate_negative_diversity_v2_next"])
        self.assertFalse(decision["training_allowed_now"])
        self.assertFalse(
            decision["integration_v1_or_v2_training_use_allowed"]
        )
        self.assertFalse(decision["benchmark_allowed"])
        markdown = render_markdown(audit)
        self.assertIn("320 条 C", markdown)
        self.assertIn("box-total C：0/32", markdown)
        self.assertIn("6,144", markdown)

    def test_config_rejects_source_or_class_prior_changes(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("source_dataset_sha256", "0" * 64, "source_dataset_sha256"),
            (
                "integration_v2_report_sha256",
                "0" * 64,
                "integration_v2_report_sha256",
            ),
            ("train_rows_per_positive_label", 1024, "train_rows_per_positive_label"),
            ("train_rows_per_negative_subtype", 128, "train_rows_per_negative_subtype"),
            ("minimum_answer_task_fraction_dev", 0.5, "minimum_answer_task_fraction_dev"),
        )
        for key, value, error in mutations:
            with self.subTest(key=key):
                altered = copy.deepcopy(raw)
                altered[key] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_config(path)


if __name__ == "__main__":
    unittest.main()
