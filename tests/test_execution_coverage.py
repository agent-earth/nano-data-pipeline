from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nano_data_pipeline.execution_coverage import (
    build_execution_coverage_audit,
    expression_features,
    validate_failure_manifest,
)
from nano_data_pipeline.analog import (
    _canonical_json,
    _hash,
    _normalized_text,
    summarize_analog_dataset,
)


class ExecutionCoverageTests(unittest.TestCase):
    def test_expression_features_capture_repeated_operand_relation(self):
        features = expression_features("(82 + 46) * 3 - 46")
        self.assertEqual(features["result"], "338")
        self.assertEqual(features["equality_pattern"], [0, 1, 2, 1])

    def test_failure_manifest_rejects_wrong_expected_result(self):
        manifest = {
            "schema_version": "nano_execution_failure_manifest_v1",
            "policy": {
                "training_eligible": False,
                "contains_benchmark_content": False,
                "source_result_public": True,
            },
            "rows": [
                {
                    "sample_id": "failure-1",
                    "expression": "(2 + 3) * 3 - 3",
                    "expected": "FINAL: 99",
                    "baseline_verified": False,
                    "post_sft_verified": False,
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "expected result"):
            validate_failure_manifest(manifest)

    def test_audit_detects_split_mechanism_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accepted = root / "accepted.jsonl"
            release_row = {
                "family_id": "verified-reasoning",
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": "Compute (2 + 3) * 3 - 3.",
                    },
                    {"role": "assistant", "content": "FINAL: 12"},
                ],
                "sample_id": "release-1",
                "split": "train",
                "task_spec": {"expression": "(2 + 3) * 3 - 3"},
                "verifier": {"kind": "safe_execution_receipt_v1"},
            }
            accepted.write_text(
                json.dumps(release_row) + "\n",
                encoding="utf-8",
            )
            release = root / "release.json"
            release.write_text(
                json.dumps(
                    {
                        "release_id": "release-test",
                        "artifacts": {
                            "accepted_jsonl_sha256": hashlib.sha256(
                                accepted.read_bytes()
                            ).hexdigest()
                        },
                    }
                ),
                encoding="utf-8",
            )
            process = root / "process.json"
            process_sample = {
                "sample_id": "synthetic-process-1",
                "split": "train",
                "task_family": "process",
                "format_family": "process_trace_numeric",
                "difficulty": "three_step",
                "generation_rule": "process",
                "source_kind": "deterministic_synthetic",
                "training_eligible": True,
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": "Compute (20 + 4) * 2 - 1.",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "STEP 1: 20 + 4 = 24\n"
                            "STEP 2: 24 * 2 = 48\n"
                            "STEP 3: 48 - 1 = 47\n"
                            "FINAL: 47"
                        ),
                    },
                ],
                "verifier": {
                    "kind": "safe_ast_arithmetic_process_v2",
                    "source_expression": "(20 + 4) * 2 - 1",
                    "expected_result": "47",
                    "steps": [
                        {
                            "expression": "20 + 4",
                            "expected_result": "24",
                        },
                        {
                            "expression": "24 * 2",
                            "expected_result": "48",
                        },
                        {
                            "expression": "48 - 1",
                            "expected_result": "47",
                        },
                    ],
                },
            }
            process_sample["exact_sha256"] = _hash(
                _canonical_json(process_sample["messages"])
            )
            process_sample["semantic_sha256"] = _hash(
                _normalized_text(process_sample["messages"])
            )
            process_dataset = {
                "schema_version": "nano_analog_dataset_v1",
                "dataset_id": "process-test",
                "source": {
                    "benchmark_content_used": False,
                    "sealed_case_ids_used": False,
                },
                "policy": {
                    "source_split": "non_eval_analog_only",
                    "training_allowed": True,
                    "contains_benchmark_content": False,
                },
                "samples": [process_sample],
            }
            process_dataset["summary"] = summarize_analog_dataset(
                process_dataset
            )
            process.write_text(
                json.dumps(process_dataset),
                encoding="utf-8",
            )
            failure = root / "failure.json"
            failure.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "nano_execution_failure_manifest_v1"
                        ),
                        "policy": {
                            "training_eligible": False,
                            "contains_benchmark_content": False,
                            "source_result_public": True,
                        },
                        "source": {"result_commit": "test"},
                        "rows": [
                            {
                                "sample_id": "failure-1",
                                "expression": "(2 + 3) * 3 - 3",
                                "expected": "FINAL: 12",
                                "baseline_verified": False,
                                "post_sft_verified": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            audit = build_execution_coverage_audit(
                accepted_jsonl_path=accepted,
                release_manifest_path=release,
                process_dataset_path=process,
                failure_manifest_path=failure,
                selected_train_rows=1,
            )

        self.assertTrue(
            audit["findings"]["release_relation_without_process"]
        )
        self.assertTrue(audit["findings"]["process_shape_without_relation"])
        self.assertTrue(audit["findings"]["joint_mechanism_coverage_missing"])
        self.assertFalse(audit["decision"]["more_sft_allowed_now"])


if __name__ == "__main__":
    unittest.main()
