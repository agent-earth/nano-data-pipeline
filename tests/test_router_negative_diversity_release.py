from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from transformers import AutoTokenizer

from nano_data_pipeline.analog import (
    _canonical_json,
    _hash,
    _normalized_text,
    summarize_analog_dataset,
)
from nano_data_pipeline.router_negative_diversity_release import (
    build_dataset,
    load_build_config,
    validate_release,
)
from scripts.build_router_negative_diversity_v2 import render_markdown


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/router_classification/"
    "qwen35_router_negative_diversity_build_v2.json"
)
DATASET = ROOT / "datasets/qwen35_router_negative_diversity_v2.json"
RELEASE = (
    ROOT / "manifests/qwen35_router_negative_diversity_v2.release.json"
)


class RouterNegativeDiversityReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_build_config(CONFIG)
        cls.tokenizer_path = (ROOT / cls.config.tokenizer_path).resolve()
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.tokenizer_path,
            local_files_only=True,
        )

    def test_build_config_freezes_audit_contract_and_outputs(self):
        self.assertEqual(
            self.config.audit_sha256,
            "9aaa69de746dbdc5cefbb52fb271c8f9ec86716d10ada70704c7e346dc2f7c17",
        )
        self.assertEqual(
            self.config.contract_sha256,
            "c195a7373ea283546dde1866f70593f0912833d987ff5f1a8cb424c2bc340335",
        )
        self.assertEqual(len(self.config.integration_preregistrations), 2)
        self.assertEqual(len(self.config.benchmark_sources), 3)

    def test_deterministic_dataset_has_balanced_natural_tasks(self):
        first = build_dataset(self.config)
        second = build_dataset(self.config)
        self.assertEqual(first, second)
        rows = first["samples"]
        self.assertEqual(len(rows), 7680)
        self.assertEqual(
            {
                split: {
                    label: sum(
                        row["split"] == split
                        and row["route_label"] == label
                        for row in rows
                    )
                    for label in ("A", "B", "C")
                }
                for split in ("train", "validation")
            },
            {
                "train": {"A": 2048, "B": 2048, "C": 2048},
                "validation": {"A": 512, "B": 512, "C": 512},
            },
        )
        self.assertTrue(all(row["prompt_mode"] == "answer_task" for row in rows))
        self.assertTrue(
            all(
                not any(
                    term in row["messages"][1]["content"].casefold()
                    for term in (
                        "route",
                        "router",
                        "classify",
                        "classification",
                        "semantic class",
                    )
                )
                for row in rows
            )
        )
        for split, templates in (("train", 16), ("validation", 4)):
            for subtype in (
                "box_total",
                "remaining_stock",
                "paired_average",
                "single_operation",
                "weighted_total",
                "quotient_remainder",
                "time_conversion",
                "percentage_change",
            ):
                selected = [
                    row
                    for row in rows
                    if row["split"] == split
                    and row["negative_subtype"] == subtype
                ]
                self.assertEqual(
                    len({row["template_id"] for row in selected}),
                    templates,
                )

    def test_tamper_breaks_class_balance_or_answer_task_gate(self):
        dataset = build_dataset(self.config)
        altered = copy.deepcopy(dataset)
        row = next(
            row
            for row in altered["samples"]
            if row["split"] == "train" and row["route_label"] == "C"
        )
        row["route_label"] = "A"
        row["route_name"] = "implicit_scale_total"
        row["messages"][-1]["content"] = "FINAL: A"
        row["exact_sha256"] = _hash(_canonical_json(row["messages"]))
        row["semantic_sha256"] = _hash(_normalized_text(row["messages"]))
        altered["summary"] = summarize_analog_dataset(altered)
        release = validate_release(
            altered,
            config=self.config,
            tokenizer=self.tokenizer,
            tokenizer_path=self.tokenizer_path,
        )
        self.assertFalse(release["checks"]["exact_row_and_class_balance"])
        self.assertFalse(release["training_unblocked"])

        altered = copy.deepcopy(dataset)
        dev = next(
            row
            for row in altered["samples"]
            if row["split"] == "validation"
        )
        dev["messages"][1]["content"] += " Choose the router class."
        dev["exact_sha256"] = _hash(_canonical_json(dev["messages"]))
        dev["semantic_sha256"] = _hash(_normalized_text(dev["messages"]))
        altered["summary"] = summarize_analog_dataset(altered)
        release = validate_release(
            altered,
            config=self.config,
            tokenizer=self.tokenizer,
            tokenizer_path=self.tokenizer_path,
        )
        self.assertFalse(release["checks"]["dev_answer_task_only_pass"])
        self.assertFalse(release["training_unblocked"])

    def test_build_config_rejects_contract_or_output_changes(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("audit_sha256", "0" * 64, "audit_sha256"),
            ("contract_sha256", "0" * 64, "contract_sha256"),
            (
                "output_dataset_path",
                "datasets/other.json",
                "output_dataset_path",
            ),
        )
        for key, value, error in mutations:
            with self.subTest(key=key):
                altered = copy.deepcopy(raw)
                altered[key] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_build_config(path)

    def test_committed_release_revalidates_exactly(self):
        if not DATASET.exists() or not RELEASE.exists():
            self.skipTest("release artifacts are generated after prereg tests")
        dataset = json.loads(DATASET.read_text(encoding="utf-8"))
        release = json.loads(RELEASE.read_text(encoding="utf-8"))
        recomputed = validate_release(
            dataset,
            config=self.config,
            tokenizer=self.tokenizer,
            tokenizer_path=self.tokenizer_path,
        )
        self.assertEqual(recomputed, release)
        self.assertTrue(release["training_unblocked"])
        self.assertTrue(all(release["checks"].values()))
        self.assertGreaterEqual(release["accepted"]["train_tokens"], 600_000)
        self.assertIn("6,144", render_markdown(release))


if __name__ == "__main__":
    unittest.main()
