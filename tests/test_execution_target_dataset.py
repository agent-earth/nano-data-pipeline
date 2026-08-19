from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nano_data_pipeline.execution_target_dataset import (
    validate_execution_target_dataset,
)
from nano_data_pipeline.subagent_campaign import (
    canonical_json,
    semantic_basis,
    sha256_text,
)


class MappingTokenizer:
    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return list(range(self.counts[canonical_json(messages)]))


class ExecutionTargetDatasetTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict, dict[str, int], dict]:
        expression = "(20 + 4) * 2 - 4"
        process_messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"Compute {expression}."},
            {
                "role": "assistant",
                "content": (
                    "STEP 1: 20 + 4 = 24\n"
                    "STEP 2: 24 * 2 = 48\n"
                    "STEP 3: 48 - 4 = 44\n"
                    "FINAL: 44"
                ),
            },
        ]
        final_messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"Compute {expression}."},
            {"role": "assistant", "content": "FINAL: 44"},
        ]
        semantic_task_hash = sha256_text(expression)
        process_spec = {"expression": expression, "view": "process"}
        final_spec = {"expression": expression, "view": "final"}
        process = {
            "sample_id": "execution-process",
            "split": "train",
            "task_family": "execution-target-process",
            "view": "process",
            "pair_id": "pair-1",
            "messages": process_messages,
            "task_spec": process_spec,
            "verifier": {
                "kind": "safe_ast_arithmetic_process_v2",
                "source_expression": expression,
                "steps": [
                    {"expression": "20 + 4", "expected_result": "24"},
                    {"expression": "24 * 2", "expected_result": "48"},
                    {"expression": "48 - 4", "expected_result": "44"},
                ],
                "expected_result": "44",
            },
            "token_count": 20,
            "exact_hash": sha256_text(canonical_json(process_messages)),
            "semantic_hash": sha256_text(
                semantic_basis("execution-target-process", process_spec)
            ),
            "semantic_task_hash": semantic_task_hash,
        }
        final = {
            "sample_id": "execution-final",
            "split": "train",
            "task_family": "execution-target-final",
            "view": "final",
            "pair_id": "pair-1",
            "messages": final_messages,
            "task_spec": final_spec,
            "verifier": {"kind": "safe_execution_receipt_v1"},
            "token_count": 12,
            "exact_hash": sha256_text(canonical_json(final_messages)),
            "semantic_hash": sha256_text(
                semantic_basis("execution-target-final", final_spec)
            ),
            "semantic_task_hash": semantic_task_hash,
        }
        dataset = {
            "schema_version": "nano_execution_target_dataset_v1",
            "contract": {
                "train_rows": 2,
                "dev_rows": 0,
                "minimum_train_tokens": 1,
            },
            "token_accounting": {"file_sha256": {}},
            "samples": [process, final],
        }
        counts = {
            canonical_json(process_messages): 20,
            canonical_json(final_messages): 12,
        }
        prior = root / "accepted.jsonl"
        prior.write_text(
            json.dumps(
                {
                    "sample_id": "prior",
                    "semantic_hash": "prior-semantic",
                    "task_spec": {"expression": "(1 + 1) * 2 - 1"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        prior_release = root / "release.json"
        prior_release.write_text("{}", encoding="utf-8")
        audit = root / "audit.json"
        audit.write_text("{}", encoding="utf-8")
        paths = {
            "accepted_jsonl_path": prior,
            "release_manifest_path": prior_release,
            "audit_path": audit,
        }
        return dataset, counts, paths

    def _validate(self, dataset: dict, counts: dict[str, int], paths: dict):
        return validate_execution_target_dataset(
            dataset,
            tokenizer=MappingTokenizer(counts),
            tokenizer_path=None,
            **paths,
        )

    def test_tampered_intermediate_step_fails_verifier_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, counts, paths = self._fixture(Path(directory))
            tampered = copy.deepcopy(dataset)
            tampered["samples"][0]["messages"][-1]["content"] = (
                tampered["samples"][0]["messages"][-1]["content"].replace(
                    "20 + 4 = 24",
                    "20 + 4 = 25",
                )
            )
            counts[canonical_json(tampered["samples"][0]["messages"])] = 20
            tampered["samples"][0]["exact_hash"] = sha256_text(
                canonical_json(tampered["samples"][0]["messages"])
            )
            release = self._validate(tampered, counts, paths)
        self.assertFalse(release["checks"]["deterministic_verifier_pass"])
        self.assertFalse(release["training_unblocked"])

    def test_tampered_paired_final_fails_consistency_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, counts, paths = self._fixture(Path(directory))
            tampered = copy.deepcopy(dataset)
            tampered["samples"][1]["messages"][-1]["content"] = "FINAL: 45"
            counts[canonical_json(tampered["samples"][1]["messages"])] = 12
            tampered["samples"][1]["exact_hash"] = sha256_text(
                canonical_json(tampered["samples"][1]["messages"])
            )
            release = self._validate(tampered, counts, paths)
        self.assertFalse(release["checks"]["paired_view_consistency_pass"])
        self.assertFalse(release["training_unblocked"])

    def test_tampered_token_count_fails_accounting_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, counts, paths = self._fixture(Path(directory))
            tampered = copy.deepcopy(dataset)
            tampered["samples"][0]["token_count"] += 1
            release = self._validate(tampered, counts, paths)
        self.assertFalse(release["checks"]["token_accounting_pass"])
        self.assertFalse(release["training_unblocked"])


if __name__ == "__main__":
    unittest.main()
