from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from difflib import SequenceMatcher
from pathlib import Path
from unittest import mock

from nano_data_pipeline.orca_math import (
    OrcaMathConfig,
    QuestionNearDuplicateIndex,
    build_preregister,
    extract_numeric_answer,
    load_config,
    normalize_question,
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

    def test_numeric_answer_extraction_uses_teacher_tail(self):
        self.assertEqual(
            extract_numeric_answer(
                "We first compute 7 + 8 = 15. Therefore the answer is 15."
            ),
            "15",
        )
        self.assertEqual(
            extract_numeric_answer("The requested fraction is 18/24 = 3/4."),
            "3/4",
        )
        self.assertEqual(
            extract_numeric_answer(
                r"The requested fraction is \frac{18}{24}=\frac{3}{4}."
            ),
            "3/4",
        )
        self.assertEqual(
            extract_numeric_answer("So the approximate rate is 20.03%."),
            "20.03",
        )
        self.assertIsNone(extract_numeric_answer("No numeric answer."))

    def test_question_near_duplicate_index(self):
        normalized_rows = [
            normalize_question(
                "A shop has 24 red apples and sells 6. How many remain?"
            ),
            normalize_question(
                "A train travels 90 miles in two hours. Find its speed."
            ),
        ]
        frequencies = {}
        for row in normalized_rows:
            for token in set(row.split()):
                frequencies[token] = frequencies.get(token, 0) + 1
        index = QuestionNearDuplicateIndex(
            jaccard_threshold=0.8,
            sequence_threshold=0.85,
            token_frequencies=frequencies,
        )
        original = normalized_rows[0]
        near = normalize_question(
            "A shop has 24 red apples and sells 6; how many remain?"
        )
        different = normalize_question(
            "A train travels 90 miles in two hours. Find its speed."
        )
        index.add(original)
        self.assertEqual(len(index.matches(near)), 1)
        self.assertEqual(index.matches(different), [])

    def test_rare_prefix_filter_matches_brute_force(self):
        randomizer = random.Random(20260821)
        vocabulary = [f"token{index}" for index in range(80)]
        rows = []
        for _ in range(200):
            size = randomizer.randint(8, 24)
            rows.append(
                " ".join(sorted(randomizer.sample(vocabulary, size)))
            )
        frequencies = {}
        for row in rows:
            for token in set(row.split()):
                frequencies[token] = frequencies.get(token, 0) + 1
        index = QuestionNearDuplicateIndex(
            jaccard_threshold=0.8,
            sequence_threshold=0.7,
            token_frequencies=frequencies,
        )
        indexed = []
        for row in rows:
            actual = {match["index"] for match in index.matches(row)}
            expected = set()
            right = set(row.split())
            for prior_index, prior in enumerate(indexed):
                left = set(prior.split())
                union = left | right
                jaccard = len(left & right) / len(union)
                sequence = max(
                    SequenceMatcher(
                        None,
                        prior,
                        row,
                        autojunk=False,
                    ).ratio(),
                    SequenceMatcher(
                        None,
                        row,
                        prior,
                        autojunk=False,
                    ).ratio(),
                )
                if jaccard >= 0.8 and sequence >= 0.7:
                    expected.add(prior_index)
            self.assertEqual(actual, expected)
            index.add(row)
            indexed.append(row)

    def test_same_frozen_frequency_map_is_stable_across_rebuilds(self):
        frequencies = {
            "a": 100,
            "shop": 8,
            "has": 80,
            "24": 2,
            "red": 3,
            "apples": 4,
            "sells": 5,
            "6": 2,
            "how": 60,
            "many": 50,
            "remain": 6,
        }
        rows = [
            normalize_question(
                "A shop has 24 red apples and sells 6. How many remain?"
            ),
            normalize_question(
                "A shop has 24 red apples and sells 6; how many remain?"
            ),
        ]
        build_index = QuestionNearDuplicateIndex(
            jaccard_threshold=0.92,
            sequence_threshold=0.92,
            token_frequencies=frequencies,
        )
        validation_index = QuestionNearDuplicateIndex(
            jaccard_threshold=0.92,
            sequence_threshold=0.92,
            token_frequencies=frequencies,
        )
        build_index.add(rows[0])
        validation_index.add(rows[1])
        self.assertEqual(len(build_index.matches(rows[1])), 1)
        self.assertEqual(len(validation_index.matches(rows[0])), 1)

    def test_sequence_similarity_is_symmetric(self):
        frequencies = {
            "number": 10,
            "students": 4,
            "each": 3,
            "receive": 2,
            "pencils": 2,
            "total": 8,
            "how": 9,
            "many": 9,
        }
        left = normalize_question(
            "A number of students each receive pencils. How many total?"
        )
        right = normalize_question(
            "How many pencils total if each student receives the same number?"
        )
        index = QuestionNearDuplicateIndex(
            jaccard_threshold=0.2,
            sequence_threshold=0.1,
            token_frequencies=frequencies,
        )
        index.add(left)
        forward = index.matches(right)[0]["sequence_ratio"]
        reverse_index = QuestionNearDuplicateIndex(
            jaccard_threshold=0.2,
            sequence_threshold=0.1,
            token_frequencies=frequencies,
        )
        reverse_index.add(right)
        reverse = reverse_index.matches(left)[0]["sequence_ratio"]
        self.assertEqual(forward, reverse)


if __name__ == "__main__":
    unittest.main()
