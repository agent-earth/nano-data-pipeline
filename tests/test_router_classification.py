from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_data_pipeline.router_classification import (
    build_dataset,
    load_config,
    validate_router_dataset,
)
from nano_data_pipeline.analog import (
    _canonical_json,
    _hash,
    _normalized_text,
    summarize_analog_dataset,
)
from scripts.preregister_router_classification_v1 import (
    build_receipt,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/router_classification/qwen35_router_classification_v1.json"
)


class RouterClassificationTests(unittest.TestCase):
    def test_config_freezes_balanced_contract_and_sources(self):
        config = load_config(CONFIG)
        self.assertEqual(config.train_rows_per_label, 256)
        self.assertEqual(config.dev_rows_per_label, 64)
        self.assertEqual(config.minimum_train_tokens, 50_000)
        self.assertEqual(
            config.label_contract,
            {
                "A": "implicit_scale_total",
                "B": "first_strict_profit_period",
                "C": "NONE",
            },
        )
        self.assertEqual(len(config.negative_subtypes), 4)

    def test_dataset_is_balanced_disjoint_and_benchmark_free(self):
        config = load_config(CONFIG)
        dataset = build_dataset(config)
        release = validate_router_dataset(dataset, config=config)
        self.assertEqual(release["accepted"]["rows"], 960)
        self.assertEqual(
            release["accepted"]["train_by_label"],
            {"A": 256, "B": 256, "C": 256},
        )
        self.assertEqual(
            release["accepted"]["dev_by_label"],
            {"A": 64, "B": 64, "C": 64},
        )
        self.assertEqual(release["overlap"]["train_dev_semantic"], 0)
        self.assertEqual(release["overlap"]["train_dev_template"], 0)
        self.assertEqual(release["leakage"]["forbidden_terms"], [])
        self.assertTrue(
            all(
                value
                for key, value in release["checks"].items()
                if key not in {"token_accounting_pass"}
            )
        )

    def test_tamper_breaks_balance_split_or_forbidden_gate(self):
        config = load_config(CONFIG)
        dataset = build_dataset(config)

        missing = copy.deepcopy(dataset)
        missing["samples"].pop()
        missing["summary"] = summarize_analog_dataset(missing)
        release = validate_router_dataset(missing, config=config)
        self.assertFalse(release["checks"]["row_count_pass"])
        self.assertFalse(release["training_unblocked"])

        overlap = copy.deepcopy(dataset)
        train = next(
            row for row in overlap["samples"] if row["split"] == "train"
        )
        dev = next(
            row for row in overlap["samples"] if row["split"] == "validation"
        )
        dev["semantic_sha256"] = train["semantic_sha256"]
        with self.assertRaisesRegex(
            ValueError,
            "semantic-deduplicated|semantic hash mismatch",
        ):
            validate_router_dataset(overlap, config=config)

        forbidden = copy.deepcopy(dataset)
        sample = forbidden["samples"][0]
        sample["messages"][1]["content"] += " gsm8k"
        sample["exact_sha256"] = _hash(
            _canonical_json(sample["messages"])
        )
        sample["semantic_sha256"] = _hash(
            _normalized_text(sample["messages"])
        )
        forbidden["summary"] = summarize_analog_dataset(forbidden)
        release = validate_router_dataset(forbidden, config=config)
        self.assertFalse(release["checks"]["forbidden_content_pass"])

    def test_config_rejects_quota_label_or_evidence_changes(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("train_rows_per_label", 128, "train_rows_per_label"),
            ("minimum_train_tokens", 1, "minimum_train_tokens"),
            (
                "label_contract",
                {"A": "NONE", "B": "first_strict_profit_period", "C": "NONE"},
                "label_contract",
            ),
            ("binary_detector_report_sha256", "0" * 64, "binary_detector_report_sha256"),
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

    def test_preregister_is_deterministic_and_generation_closed(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["frozen_contract"]["rows"], 960)
        self.assertFalse(
            first["execution_boundary"]["data_generation_started"]
        )
        self.assertFalse(first["execution_boundary"]["dataset_file_exists"])
        self.assertFalse(first["execution_boundary"]["release_file_exists"])
        self.assertFalse(first["execution_boundary"]["training_started"])
        markdown = render_markdown(first)
        self.assertIn("768 rows", markdown)
        self.assertIn("192 rows", markdown)
        self.assertIn("data generation started：false", markdown)


if __name__ == "__main__":
    unittest.main()
