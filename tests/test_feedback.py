from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nano_data_pipeline.feedback import (
    build_feedback_manifest,
    sha256_file,
    validate_feedback_manifest,
)


def _record(
    case_id: str,
    *,
    model: str,
    score: float,
    prediction: str | None,
    output: str,
    expected: str,
    benchmark: str = "mmlu",
    finish_reason: str = "stop",
):
    return {
        "case_id": case_id,
        "benchmark": benchmark,
        "model": model,
        "score": score,
        "prediction": prediction,
        "output": output,
        "expected": expected,
        "finish_reason": finish_reason,
    }


class FeedbackTests(unittest.TestCase):
    def test_builds_sealed_public_safe_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases.json"
            four = root / "4b.jsonl"
            nine = root / "9b.jsonl"
            report = root / "report.json"
            cases.write_text(
                json.dumps(
                    [
                        {"case_id": "mmlu-a"},
                        {"case_id": "mmlu-b"},
                        {"case_id": "mmlu-c"},
                    ]
                ),
                encoding="utf-8",
            )
            four.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        _record(
                            "mmlu-a",
                            model="4b",
                            score=1,
                            prediction="C",
                            output="FINAL: C",
                            expected="C",
                        ),
                        _record(
                            "mmlu-b",
                            model="4b",
                            score=1,
                            prediction="A",
                            output="FINAL: A",
                            expected="A",
                        ),
                        _record(
                            "mmlu-c",
                            model="4b",
                            score=0,
                            prediction="B",
                            output="FINAL: B",
                            expected="D",
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            nine.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        _record(
                            "mmlu-a",
                            model="9b",
                            score=0,
                            prediction=None,
                            output="FINAL C",
                            expected="C",
                        ),
                        _record(
                            "mmlu-b",
                            model="9b",
                            score=1,
                            prediction="A",
                            output="FINAL: A",
                            expected="A",
                        ),
                        _record(
                            "mmlu-c",
                            model="9b",
                            score=1,
                            prediction="D",
                            output="FINAL: D",
                            expected="D",
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            report.write_text(
                json.dumps(
                    {
                        "experiment_id": "test",
                        "artifacts": {
                            "four_b_raw_sha256": sha256_file(four),
                            "nine_b_raw_sha256": sha256_file(nine),
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest = build_feedback_manifest(
                case_manifest_path=cases,
                four_b_path=four,
                nine_b_path=nine,
                public_report_path=report,
                source_revision="abc123",
            )

            self.assertEqual(manifest["summary"]["rows"], 2)
            self.assertEqual(
                manifest["summary"]["by_failure_family"],
                {"format": 1, "semantic_discordance": 1},
            )
            format_row = next(
                row for row in manifest["rows"] if row["case_id"] == "mmlu-a"
            )
            self.assertEqual(
                format_row["nine_b"]["format_class"],
                "final_missing_colon",
            )
            self.assertTrue(
                format_row["nine_b"]["format_letter_matches_reference"]
            )
            self.assertTrue(
                all(not row["training_eligible"] for row in manifest["rows"])
            )
            rendered = json.dumps(manifest)
            self.assertNotIn("FINAL", rendered)
            self.assertNotIn('"expected"', rendered)

    def test_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases.json"
            four = root / "4b.jsonl"
            nine = root / "9b.jsonl"
            report = root / "report.json"
            cases.write_text('[{"case_id":"a"}]', encoding="utf-8")
            row = _record(
                "a",
                model="4b",
                score=1,
                prediction="A",
                output="FINAL: A",
                expected="A",
            )
            four.write_text(json.dumps(row) + "\n", encoding="utf-8")
            row["model"] = "9b"
            nine.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report.write_text(
                json.dumps(
                    {
                        "experiment_id": "test",
                        "artifacts": {
                            "four_b_raw_sha256": "0" * 64,
                            "nine_b_raw_sha256": sha256_file(nine),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "hashes"):
                build_feedback_manifest(
                    case_manifest_path=cases,
                    four_b_path=four,
                    nine_b_path=nine,
                    public_report_path=report,
                    source_revision="abc",
                )

    def test_validator_rejects_training_leakage(self):
        manifest = {
            "schema_version": "nano_feedback_manifest_v1",
            "policy": {
                "contains_raw_outputs": False,
                "contains_prompts": False,
                "contains_references": False,
                "contains_predictions": False,
                "direct_training_allowed": False,
            },
            "rows": [
                {
                    "case_id": "x",
                    "benchmark": "mmlu",
                    "paired_outcome": "four_b_only",
                    "failure_family": "semantic_discordance",
                    "four_b": {
                        "correct": True,
                        "format_class": "parseable",
                        "format_letter_matches_reference": None,
                    },
                    "nine_b": {
                        "correct": False,
                        "format_class": "parseable",
                        "format_letter_matches_reference": None,
                    },
                    "source_split": "sealed_eval_feedback",
                    "training_eligible": True,
                }
            ],
        }
        manifest["summary"] = {
            "rows": 1,
            "by_benchmark": {"mmlu": 1},
            "by_failure_family": {"semantic_discordance": 1},
            "by_paired_outcome": {"four_b_only": 1},
            "training_eligible_rows": 1,
        }
        with self.assertRaisesRegex(ValueError, "training eligible"):
            validate_feedback_manifest(manifest)

    def test_validator_rejects_benchmark_payload(self):
        manifest = {
            "schema_version": "nano_feedback_manifest_v1",
            "policy": {
                "contains_raw_outputs": False,
                "contains_prompts": False,
                "contains_references": False,
                "contains_predictions": False,
                "direct_training_allowed": False,
            },
            "rows": [
                {
                    "case_id": "x",
                    "benchmark": "mmlu",
                    "paired_outcome": "four_b_only",
                    "failure_family": "semantic_discordance",
                    "four_b": {
                        "correct": True,
                        "format_class": "parseable",
                        "format_letter_matches_reference": None,
                    },
                    "nine_b": {
                        "correct": False,
                        "format_class": "parseable",
                        "format_letter_matches_reference": None,
                    },
                    "source_split": "sealed_eval_feedback",
                    "training_eligible": False,
                    "prompt": "sealed benchmark prompt",
                }
            ],
        }
        manifest["summary"] = {
            "rows": 1,
            "by_benchmark": {"mmlu": 1},
            "by_failure_family": {"semantic_discordance": 1},
            "by_paired_outcome": {"four_b_only": 1},
            "training_eligible_rows": 0,
        }
        with self.assertRaisesRegex(ValueError, "forbidden row fields"):
            validate_feedback_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
