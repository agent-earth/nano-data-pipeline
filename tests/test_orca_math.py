from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nano_data_pipeline.orca_math import (
    OrcaMathConfig,
    build_preregister,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/orca_math_sft_v1.json"


class OrcaMathTests(unittest.TestCase):
    def test_config_freezes_scale_and_closed_boundary(self):
        config = load_config(CONFIG)
        self.assertEqual(config.raw["selection"]["train_rows"], 32_768)
        self.assertEqual(config.raw["selection"]["dev_rows"], 1_024)
        self.assertEqual(
            config.raw["token_accounting"]["minimum_train_tokens"],
            10_000_000,
        )
        self.assertFalse(
            config.raw["training_boundary"][
                "benchmark_rows_training_eligible"
            ]
        )
        self.assertFalse(
            config.raw["training_boundary"]["rl_or_opd_unlocked"]
        )

    def test_config_rejects_scale_mutation(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        raw["selection"]["train_rows"] = 10_000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split contract"):
                load_config(path)

    def test_preregister_is_deterministic_and_contains_no_selected_rows(self):
        first = build_preregister(load_config(CONFIG))
        second = build_preregister(load_config(CONFIG))
        self.assertEqual(first, second)
        self.assertEqual(first["source"]["rows"], 200_035)
        self.assertEqual(len(first["forbidden_corpora"]), 6)
        self.assertTrue(
            first["training_boundary"]["this_receipt_only_preregisters"]
        )
        serialized = json.dumps(first).lower()
        self.assertNotIn("selected_sample", serialized)
        self.assertNotIn("source_question", serialized)

    def test_preregister_rejects_source_identity_drift(self):
        config = load_config(CONFIG)
        raw = copy.deepcopy(config.raw)
        raw["source"]["parquet_sha256"] = "0" * 64
        altered = OrcaMathConfig(path=config.path, raw=raw)
        with self.assertRaisesRegex(ValueError, "source identity"):
            build_preregister(altered)


if __name__ == "__main__":
    unittest.main()
