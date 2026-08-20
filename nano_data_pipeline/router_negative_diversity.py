from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nano_data_pipeline.feedback import sha256_file


AUDIT_SCHEMA = "nano_router_negative_diversity_audit_v2"
CONTRACT_SCHEMA = "nano_router_negative_diversity_contract_v2"
EXPLICIT_CLASSIFICATION_TERMS = (
    "route",
    "router",
    "classify",
    "classification",
    "semantic class",
)


@dataclass(frozen=True)
class RouterNegativeDiversityConfig:
    schema_version: str
    audit_id: str
    seed: int
    source_dataset_path: str
    source_dataset_sha256: str
    source_release_path: str
    source_release_sha256: str
    parity_report_path: str
    parity_report_sha256: str
    integration_v2_preregister_path: str
    integration_v2_preregister_sha256: str
    integration_v2_report_path: str
    integration_v2_report_sha256: str
    negative_subtypes: list[str]
    train_rows_per_positive_label: int
    dev_rows_per_positive_label: int
    train_rows_per_negative_subtype: int
    dev_rows_per_negative_subtype: int
    minimum_train_templates_per_negative_subtype: int
    minimum_dev_templates_per_negative_subtype: int
    minimum_answer_task_fraction_train: float
    minimum_answer_task_fraction_dev: float
    minimum_train_tokens: int
    output_audit_path: str
    output_contract_path: str


def load_config(path: str | Path) -> RouterNegativeDiversityConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(RouterNegativeDiversityConfig.__dataclass_fields__):
        raise ValueError("router negative diversity config fields differ")
    config = RouterNegativeDiversityConfig(**raw)
    expected = {
        "schema_version": AUDIT_SCHEMA,
        "audit_id": "qwen35-router-negative-diversity-v2",
        "seed": 20260827,
        "source_dataset_path": "datasets/qwen35_router_classification_v1.json",
        "source_dataset_sha256": (
            "dacd3663639fe9ddc054865b87afdd0c918f0fddb12c8c9355819d4bbce95d65"
        ),
        "source_release_path": (
            "manifests/qwen35_router_classification_v1.release.json"
        ),
        "source_release_sha256": (
            "fb265e125e181056856a196322cf5da3b1d7d890d60ad653839d2707ebe3781d"
        ),
        "parity_report_path": (
            "../nano-harness-fullstack-traex-03/docs/results/"
            "qwen35_router_serving_parity_v1.public.json"
        ),
        "parity_report_sha256": (
            "539517c890e53f2a0e4034c724d1324df6cc828186d9621f77c106c08d4a1c01"
        ),
        "integration_v2_preregister_path": (
            "../nano-harness-fullstack-traex-03/docs/experiments/"
            "qwen35_router_adapter_integration_v2.preregister.json"
        ),
        "integration_v2_preregister_sha256": (
            "1a23a1bb391a3ebac7e70aecd5e2d855ef624e825c26e6bfe9ed942d07cc9e2e"
        ),
        "integration_v2_report_path": (
            "../nano-harness-fullstack-traex-03/docs/results/"
            "qwen35_router_adapter_integration_v2.public.json"
        ),
        "integration_v2_report_sha256": (
            "251e6b45cbcd487079a4c52815eae5c24e89765b5b4f6521f4b7fb0a34a8f5b9"
        ),
        "negative_subtypes": [
            "box_total",
            "remaining_stock",
            "paired_average",
            "single_operation",
            "weighted_total",
            "quotient_remainder",
            "time_conversion",
            "percentage_change",
        ],
        "train_rows_per_positive_label": 2048,
        "dev_rows_per_positive_label": 512,
        "train_rows_per_negative_subtype": 256,
        "dev_rows_per_negative_subtype": 64,
        "minimum_train_templates_per_negative_subtype": 16,
        "minimum_dev_templates_per_negative_subtype": 4,
        "minimum_answer_task_fraction_train": 0.75,
        "minimum_answer_task_fraction_dev": 1.0,
        "minimum_train_tokens": 600_000,
        "output_audit_path": (
            "docs/datasets/qwen35_router_negative_diversity_v2.public.json"
        ),
        "output_contract_path": (
            "configs/router_classification/"
            "qwen35_router_negative_diversity_release_v2.json"
        ),
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"router negative diversity freezes {field}={expected_value}"
            )
    for source, digest in (
        (config.source_dataset_path, config.source_dataset_sha256),
        (config.source_release_path, config.source_release_sha256),
        (config.parity_report_path, config.parity_report_sha256),
        (
            config.integration_v2_preregister_path,
            config.integration_v2_preregister_sha256,
        ),
        (config.integration_v2_report_path, config.integration_v2_report_sha256),
    ):
        if sha256_file(Path(source)) != digest:
            raise ValueError("router negative diversity evidence identity differs")
    return config


def is_explicit_classification_prompt(prompt: str) -> bool:
    normalized = " ".join(prompt.casefold().split())
    return any(term in normalized for term in EXPLICIT_CLASSIFICATION_TERMS)


def _prompt(messages: list[dict[str, str]]) -> str:
    users = [
        message["content"]
        for message in messages
        if message.get("role") == "user"
    ]
    if len(users) != 1:
        raise ValueError("router negative diversity user-message count differs")
    return users[0]


def summarize_source(dataset: dict[str, Any]) -> dict[str, Any]:
    rows = dataset["samples"]
    negative = [row for row in rows if row["route_label"] == "C"]
    by_split = {}
    for split in ("train", "validation"):
        selected = [row for row in negative if row["split"] == split]
        by_subtype = {}
        for subtype in sorted(
            {str(row["negative_subtype"]) for row in selected}
        ):
            subtype_rows = [
                row for row in selected if row["negative_subtype"] == subtype
            ]
            explicit = sum(
                is_explicit_classification_prompt(_prompt(row["messages"]))
                for row in subtype_rows
            )
            by_subtype[subtype] = {
                "rows": len(subtype_rows),
                "unique_template_ids": len(
                    {row["template_id"] for row in subtype_rows}
                ),
                "unique_generation_rules": len(
                    {row["generation_rule"] for row in subtype_rows}
                ),
                "explicit_classification_rows": explicit,
                "answer_task_rows": len(subtype_rows) - explicit,
            }
        explicit = sum(
            is_explicit_classification_prompt(_prompt(row["messages"]))
            for row in selected
        )
        by_split[split] = {
            "rows": len(selected),
            "explicit_classification_rows": explicit,
            "answer_task_rows": len(selected) - explicit,
            "by_subtype": by_subtype,
        }
    return {
        "total_rows": len(rows),
        "label_counts": dict(
            sorted(Counter(row["route_label"] for row in rows).items())
        ),
        "negative_rows": len(negative),
        "negative_subtypes": sorted(
            {str(row["negative_subtype"]) for row in negative}
        ),
        "by_split": by_split,
    }


def next_contract(config: RouterNegativeDiversityConfig) -> dict[str, Any]:
    negative_train = (
        len(config.negative_subtypes) * config.train_rows_per_negative_subtype
    )
    negative_dev = (
        len(config.negative_subtypes) * config.dev_rows_per_negative_subtype
    )
    positive_train = 2 * config.train_rows_per_positive_label
    positive_dev = 2 * config.dev_rows_per_positive_label
    if (
        negative_train != config.train_rows_per_positive_label
        or negative_dev != config.dev_rows_per_positive_label
    ):
        raise ValueError("router negative diversity class balance differs")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "dataset_id": "qwen35-router-negative-diversity-v2",
        "seed": config.seed,
        "label_contract": {
            "A": "implicit_scale_total",
            "B": "first_strict_profit_period",
            "C": "NONE",
        },
        "rows": {
            "train": positive_train + negative_train,
            "dev": positive_dev + negative_dev,
            "total": positive_train
            + negative_train
            + positive_dev
            + negative_dev,
        },
        "train_by_label": {
            "A": config.train_rows_per_positive_label,
            "B": config.train_rows_per_positive_label,
            "C": negative_train,
        },
        "dev_by_label": {
            "A": config.dev_rows_per_positive_label,
            "B": config.dev_rows_per_positive_label,
            "C": negative_dev,
        },
        "negative_subtypes": {
            subtype: {
                "train_rows": config.train_rows_per_negative_subtype,
                "dev_rows": config.dev_rows_per_negative_subtype,
                "minimum_train_templates": (
                    config.minimum_train_templates_per_negative_subtype
                ),
                "minimum_dev_templates": (
                    config.minimum_dev_templates_per_negative_subtype
                ),
            }
            for subtype in config.negative_subtypes
        },
        "lexical_contract": {
            "minimum_answer_task_fraction_train": (
                config.minimum_answer_task_fraction_train
            ),
            "minimum_answer_task_fraction_dev": (
                config.minimum_answer_task_fraction_dev
            ),
            "dev_explicit_classification_rows": 0,
            "answer_task_prompts_must_not_contain": list(
                EXPLICIT_CLASSIFICATION_TERMS
            ),
            "positive_and_negative_answer_task_templates_required": True,
        },
        "minimum_train_tokens": config.minimum_train_tokens,
        "split_contract": {
            "train_dev_template_overlap": 0,
            "train_dev_semantic_overlap": 0,
            "source_v1_sample_overlap": 0,
            "source_v1_semantic_overlap": 0,
            "integration_v1_v2_prompt_overlap": 0,
            "benchmark_prompt_overlap": 0,
        },
        "provenance": {
            "deterministic_synthetic_only": True,
            "model_outputs_used": False,
            "teacher_outputs_used": False,
            "integration_v1_v2_rows_used": False,
            "integration_v1_v2_outputs_used": False,
            "benchmark_canary_holdout_content_used": False,
            "allowed_feedback": (
                "public aggregate subtype counts and prompt hashes only"
            ),
        },
        "release_gates": {
            "exact_row_and_class_balance": True,
            "exact_negative_subtype_balance": True,
            "template_diversity_pass": True,
            "answer_task_fraction_pass": True,
            "dev_answer_task_only_pass": True,
            "all_sample_ids_unique": True,
            "all_exact_hashes_unique": True,
            "all_semantic_hashes_unique": True,
            "all_overlap_counts_zero": True,
            "minimum_train_tokens_pass": True,
            "tokenizer_identity_pinned": True,
            "forbidden_content_zero": True,
        },
        "training_unblocked_only_after_release": True,
        "allowed_after_release": (
            "One separately pre-registered Qwen3.5-4B router SFT run with "
            "fresh validation and serving-namespace parity."
        ),
        "forbidden": [
            "reuse_integration_v1_or_v2_rows",
            "reuse_integration_v1_or_v2_outputs",
            "train_on_benchmark_canary_or_holdout",
            "post_release_template_or_class_prior_change",
            "second_training_run_without_new_preregistration",
        ],
    }


def build_audit(config: RouterNegativeDiversityConfig) -> dict[str, Any]:
    dataset = json.loads(
        Path(config.source_dataset_path).read_text(encoding="utf-8")
    )
    release = json.loads(
        Path(config.source_release_path).read_text(encoding="utf-8")
    )
    parity = json.loads(
        Path(config.parity_report_path).read_text(encoding="utf-8")
    )
    v2_preregister = json.loads(
        Path(config.integration_v2_preregister_path).read_text(encoding="utf-8")
    )
    v2 = json.loads(
        Path(config.integration_v2_report_path).read_text(encoding="utf-8")
    )
    if (
        release.get("training_unblocked") is not True
        or not all(release.get("checks", {}).values())
        or parity.get("decision", {}).get(
            "serving_namespace_root_cause_supported"
        )
        is not True
        or v2.get("decision", {}).get("adapter_integration_v2_admitted")
        is not False
        or v2.get("decision", {}).get("integration_v2_rerun_allowed")
        is not False
        or v2.get("routing_by_family", {}).get("box_total", {}).get(
            "route_correct"
        )
        != 0
        or v2.get("routing_by_family", {}).get("remaining_stock", {}).get(
            "route_correct"
        )
        != 32
        or v2_preregister.get("freshness", {}).get(
            "integration_v1_outputs_loaded"
        )
        is not False
    ):
        raise ValueError("router negative diversity predecessor evidence differs")
    source = summarize_source(dataset)
    train = source["by_split"]["train"]
    dev = source["by_split"]["validation"]
    subtype_template_singletons = all(
        row["unique_template_ids"] == 1
        and row["unique_generation_rules"] == 1
        for split in (train, dev)
        for row in split["by_subtype"].values()
    )
    all_negative_explicit = (
        train["explicit_classification_rows"] == train["rows"]
        and dev["explicit_classification_rows"] == dev["rows"]
    )
    return {
        "schema_version": AUDIT_SCHEMA,
        "audit_id": config.audit_id,
        "sources": {
            "source_dataset_sha256": config.source_dataset_sha256,
            "source_release_sha256": config.source_release_sha256,
            "parity_report_sha256": config.parity_report_sha256,
            "integration_v2_preregister_sha256": (
                config.integration_v2_preregister_sha256
            ),
            "integration_v2_report_sha256": (
                config.integration_v2_report_sha256
            ),
        },
        "source_dataset": source,
        "observed_public_failure": {
            "routing_by_family": v2["routing_by_family"],
            "box_total_route_correct": 0,
            "remaining_stock_route_correct": 32,
            "negative_false_positive_routes": v2["routing"][
                "negative_false_positive_routes"
            ],
            "integration_rows_or_outputs_loaded": False,
            "evidence_kind": "public_aggregate_only",
        },
        "findings": {
            "negative_row_count_is_not_the_primary_gap": (
                source["negative_rows"] == 320
            ),
            "one_template_per_subtype_per_split": subtype_template_singletons,
            "all_negative_rows_explicitly_ask_for_classification": (
                all_negative_explicit
            ),
            "negative_answer_task_rows_zero": (
                train["answer_task_rows"] == 0
                and dev["answer_task_rows"] == 0
            ),
            "serving_namespace_issue_excluded": parity["decision"][
                "serving_namespace_root_cause_supported"
            ],
            "box_total_specific_generalization_gap": (
                v2["routing_by_family"]["box_total"]["route_correct"] == 0
                and v2["routing_by_family"]["remaining_stock"][
                    "route_correct"
                ]
                == 32
            ),
        },
        "minimum_fresh_data_contract": next_contract(config),
        "decision": {
            "reuse_v1_data_unchanged": False,
            "generate_negative_diversity_v2_next": True,
            "training_allowed_now": False,
            "integration_v1_or_v2_training_use_allowed": False,
            "benchmark_allowed": False,
            "canary_allowed": False,
            "holdout_allowed": False,
            "rl_allowed": False,
        },
        "claim_boundary": (
            "This audit compares the released synthetic router data with "
            "public aggregate subtype outcomes and preregistered prompt hashes. "
            "It loads no integration rows or outputs, performs no model "
            "generation, and does not establish model or benchmark quality."
        ),
    }
