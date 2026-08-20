from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nano_data_pipeline.analog import (
    _canonical_json,
    _hash,
    _normalized_text,
    summarize_analog_dataset,
    validate_analog_dataset,
)
from nano_data_pipeline.feedback import sha256_file
from nano_data_pipeline.subagent_campaign import count_tokens


CONTRACT_SCHEMA = "nano_router_classification_contract_v1"
RELEASE_SCHEMA = "nano_router_classification_release_v1"
SYSTEM_PROMPT = (
    "Classify the task for a semantic tool router. Return exactly one line: "
    "FINAL: A for implicit rectangular scale totals, FINAL: B for first "
    "strictly profitable whole periods, or FINAL: C for every unsupported task."
)


@dataclass(frozen=True)
class RouterClassificationConfig:
    schema_version: str
    dataset_id: str
    seed: int
    train_rows_per_label: int
    dev_rows_per_label: int
    minimum_train_tokens: int
    label_contract: dict[str, str]
    negative_subtypes: list[str]
    forbidden_terms: list[str]
    tokenizer_path: str
    multiclass_report_path: str
    multiclass_report_sha256: str
    binary_detector_report_path: str
    binary_detector_report_sha256: str
    output_dataset_path: str
    output_release_path: str


def load_config(path: str | Path) -> RouterClassificationConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(RouterClassificationConfig.__dataclass_fields__):
        raise ValueError("router classification config fields differ")
    config = RouterClassificationConfig(**raw)
    expected = {
        "schema_version": CONTRACT_SCHEMA,
        "dataset_id": "qwen35-router-classification-v1",
        "seed": 20260824,
        "train_rows_per_label": 256,
        "dev_rows_per_label": 64,
        "minimum_train_tokens": 50_000,
        "label_contract": {
            "A": "implicit_scale_total",
            "B": "first_strict_profit_period",
            "C": "NONE",
        },
        "negative_subtypes": [
            "box_total",
            "remaining_stock",
            "paired_average",
            "single_operation",
        ],
        "forbidden_terms": [
            "gsm8k",
            "mmlu",
            "gpqa",
            "canary",
            "holdout",
            "semantic-router-",
            "binary-detector-",
        ],
        "tokenizer_path": "../../../models/Qwen3.5-4B",
        "multiclass_report_path": (
            "../nano-harness-fullstack-traex-03/docs/results/"
            "qwen35_semantic_model_router_v1.public.json"
        ),
        "multiclass_report_sha256": (
            "c8e4034a27e925025589bc1a8a52abc6720ee0d7fc97e03983ff192cd44c3742"
        ),
        "binary_detector_report_path": (
            "../nano-harness-fullstack-traex-03/docs/results/"
            "qwen35_semantic_binary_detectors_v1.public.json"
        ),
        "binary_detector_report_sha256": (
            "0f50860efd48378be11314f83a38025bca533180d63f44b5bcaf902045ef2ae4"
        ),
        "output_dataset_path": "datasets/qwen35_router_classification_v1.json",
        "output_release_path": (
            "manifests/qwen35_router_classification_v1.release.json"
        ),
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"router classification freezes {field}={expected_value}"
            )
    for path_value, digest in (
        (config.multiclass_report_path, config.multiclass_report_sha256),
        (config.binary_detector_report_path, config.binary_detector_report_sha256),
    ):
        if sha256_file(Path(path_value)) != digest:
            raise ValueError("router classification source evidence differs")
    multiclass = json.loads(
        Path(config.multiclass_report_path).read_text(encoding="utf-8")
    )
    binary = json.loads(
        Path(config.binary_detector_report_path).read_text(encoding="utf-8")
    )
    if (
        multiclass.get("decision", {}).get("router_admitted") is not False
        or binary.get("decision", {}).get("detectors_admitted") is not False
        or binary.get("decision", {}).get("training_allowed") is not False
    ):
        raise ValueError("router classification source decisions differ")
    return config


def _messages(prompt: str, label: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"FINAL: {label}"},
    ]


def _sample(
    *,
    split: str,
    label: str,
    subtype: str,
    template_id: int,
    index: int,
    prompt: str,
) -> dict[str, Any]:
    messages = _messages(prompt, label)
    identity = {
        "split": split,
        "label": label,
        "subtype": subtype,
        "template_id": template_id,
        "index": index,
        "messages": messages,
    }
    return {
        "sample_id": f"synthetic-{_hash(_canonical_json(identity))[:20]}",
        "split": split,
        "task_family": f"router_{label.lower()}",
        "format_family": "final_choice",
        "difficulty": f"router_{subtype}",
        "generation_rule": (
            f"router_classification_{subtype}_template_{template_id}_v1"
        ),
        "messages": messages,
        "source_kind": "deterministic_synthetic",
        "training_eligible": True,
        "exact_sha256": _hash(_canonical_json(messages)),
        "semantic_sha256": _hash(_normalized_text(messages)),
        "route_label": label,
        "route_name": {
            "A": "implicit_scale_total",
            "B": "first_strict_profit_period",
            "C": "NONE",
        }[label],
        "negative_subtype": subtype if label == "C" else None,
        "template_id": template_id,
    }


def _positive_prompt(
    label: str,
    template_id: int,
    value: int,
) -> tuple[str, str]:
    if label == "A":
        rows = value * 3 + 11
        columns = value * 2 + 7
        extra = value * 5 + 13
        scale = ("twice", "three times", "twofold", "triple")[template_id % 4]
        templates = (
            (
                "A venue has {rows} rows and {columns} seats in each row. "
                "The order is {scale} that capacity plus {extra} spare seats. "
                "Which semantic route applies?"
            ),
            (
                "A rectangular rack is {rows} by {columns}. Procurement wants "
                "{extra} additional units beyond {scale} the rack capacity. "
                "Select the router class."
            ),
            (
                "Capacity comes from {rows} tiers with {columns} positions per "
                "tier. A request asks for {scale} capacity and then {extra} "
                "extras. Classify the task."
            ),
            (
                "A layout contains {rows} lines of {columns} slots. The target "
                "quantity equals {extra} more than {scale} all slots. Choose "
                "the semantic tool route."
            ),
        )
        return (
            templates[template_id % 4].format(
                rows=rows,
                columns=columns,
                extra=extra,
                scale=scale,
            ),
            "implicit_scale_total",
        )
    units = value * 2 + 5
    price = value * 3 + 7
    net = value * 4 + 29
    recurring = units * price - net
    setup = net * (value * 2 + 17)
    templates = (
        (
            "Opening costs {setup}. Each month sells {units} subscriptions at "
            "{price} each and pays {recurring} monthly. Which route handles "
            "the first whole month with cumulative profit above zero?"
        ),
        (
            "A project starts with expense {setup}; every cycle moves {units} "
            "units for {price} each with cycle cost {recurring}. Classify the "
            "task asking when cumulative profit first becomes positive."
        ),
        (
            "Initial investment is {setup}. Per period revenue is {units} times "
            "{price}, while recurring expense is {recurring}. Select the route "
            "for the earliest integer period after break-even."
        ),
        (
            "A service pays {setup} upfront, earns {units}*{price} each period, "
            "and spends {recurring} each period. Which semantic class answers "
            "the first strictly profitable whole period?"
        ),
    )
    return (
        templates[template_id % 4].format(
            setup=setup,
            units=units,
            price=price,
            recurring=recurring,
        ),
        "first_strict_profit_period",
    )


def _negative_prompt(
    subtype: str,
    template_id: int,
    value: int,
) -> str:
    if subtype == "box_total":
        boxes = value * 2 + 3
        per_box = value * 3 + 5
        loose = value * 7 + 11
        return (
            f"A shipment has {boxes} cartons with {per_box} items each and "
            f"{loose} loose items. Which router class applies to the exact total?"
        )
    if subtype == "remaining_stock":
        batches = value * 2 + 7
        units = value * 3 + 13
        remaining = value * 5 + 17
        starting = batches * units + remaining
        return (
            f"Inventory starts at {starting}; {batches} batches of {units} are "
            "used. Classify the task asking for remaining stock."
        )
    if subtype == "paired_average":
        first = value * 11 + 101
        second = value * 13 + 103
        return (
            f"Two audited totals are {first} and {second}. Which route applies "
            "to finding their arithmetic average?"
        )
    left = value * 5 + 19
    right = value * 7 + 23
    operation = ("sum", "difference", "product", "ratio")[template_id % 4]
    return (
        f"Given {left} and {right}, classify a request for their exact "
        f"{operation}."
    )


def build_dataset(config: RouterClassificationConfig) -> dict[str, Any]:
    samples = []
    for split, count, offset, template_offset in (
        ("train", config.train_rows_per_label, 10_000, 0),
        ("validation", config.dev_rows_per_label, 50_000, 4),
    ):
        for label in ("A", "B"):
            for index in range(count):
                template_id = template_offset + index % 4
                prompt, subtype = _positive_prompt(
                    label,
                    template_id,
                    offset + index,
                )
                samples.append(
                    _sample(
                        split=split,
                        label=label,
                        subtype=subtype,
                        template_id=template_id,
                        index=index,
                        prompt=prompt,
                    )
                )
        per_subtype = count // len(config.negative_subtypes)
        if per_subtype * len(config.negative_subtypes) != count:
            raise ValueError("NONE quota must divide across negative subtypes")
        for subtype_index, subtype in enumerate(config.negative_subtypes):
            for index in range(per_subtype):
                template_id = template_offset + subtype_index
                prompt = _negative_prompt(
                    subtype,
                    template_id,
                    offset + subtype_index * 1000 + index,
                )
                samples.append(
                    _sample(
                        split=split,
                        label="C",
                        subtype=subtype,
                        template_id=template_id,
                        index=index,
                        prompt=prompt,
                    )
                )
    samples.sort(key=lambda row: row["sample_id"])
    dataset = {
        "schema_version": "nano_analog_dataset_v1",
        "dataset_id": config.dataset_id,
        "version": "v1",
        "source": {
            "source_kind": "deterministic_synthetic",
            "generator": "nano_data_pipeline.router_classification",
            "multiclass_report_sha256": config.multiclass_report_sha256,
            "binary_detector_report_sha256": config.binary_detector_report_sha256,
            "benchmark_content_used": False,
            "sealed_case_ids_used": False,
        },
        "policy": {
            "source_split": "non_eval_analog_only",
            "training_allowed": True,
            "contains_benchmark_content": False,
            "contains_model_outputs": False,
            "contains_teacher_outputs": False,
            "purpose": "semantic_router_classification_sft_smoke",
            "benchmark_feedback_used_for_training": False,
            "canary_used_for_training": False,
            "holdout_used_for_training": False,
        },
        "label_contract": config.label_contract,
        "samples": samples,
    }
    dataset["summary"] = summarize_analog_dataset(dataset)
    validate_router_dataset(dataset, config=config)
    return dataset


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_router_dataset(
    dataset: dict[str, Any],
    *,
    config: RouterClassificationConfig,
    tokenizer: Any | None = None,
    tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    validate_analog_dataset(dataset)
    samples = dataset["samples"]
    by_split_label = Counter(
        (row["split"], row["route_label"]) for row in samples
    )
    train_label = {
        label: by_split_label[("train", label)] for label in ("A", "B", "C")
    }
    dev_label = {
        label: by_split_label[("validation", label)]
        for label in ("A", "B", "C")
    }
    none_subtypes = {
        split: dict(
            sorted(
                Counter(
                    row["negative_subtype"]
                    for row in samples
                    if row["split"] == split and row["route_label"] == "C"
                ).items()
            )
        )
        for split in ("train", "validation")
    }
    train_semantic = {
        row["semantic_sha256"] for row in samples if row["split"] == "train"
    }
    dev_semantic = {
        row["semantic_sha256"]
        for row in samples
        if row["split"] == "validation"
    }
    train_templates = {
        row["template_id"] for row in samples if row["split"] == "train"
    }
    dev_templates = {
        row["template_id"]
        for row in samples
        if row["split"] == "validation"
    }
    rendered = json.dumps(samples, ensure_ascii=False).lower()
    forbidden_hits = [
        term for term in config.forbidden_terms if term.lower() in rendered
    ]
    train_tokens = None
    tokenizer_files = {}
    if tokenizer is not None:
        train_tokens = sum(
            count_tokens(tokenizer, row["messages"])
            for row in samples
            if row["split"] == "train"
        )
    if tokenizer_path is not None:
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ):
            tokenizer_files[filename] = sha256_file(tokenizer_path / filename)
    expected_train_subtype = config.train_rows_per_label // 4
    expected_dev_subtype = config.dev_rows_per_label // 4
    checks = {
        "row_count_pass": len(samples)
        == 3
        * (config.train_rows_per_label + config.dev_rows_per_label),
        "train_label_balance_pass": train_label
        == dict.fromkeys(("A", "B", "C"), config.train_rows_per_label),
        "dev_label_balance_pass": dev_label
        == dict.fromkeys(("A", "B", "C"), config.dev_rows_per_label),
        "train_none_subtype_balance_pass": none_subtypes["train"]
        == dict.fromkeys(config.negative_subtypes, expected_train_subtype),
        "dev_none_subtype_balance_pass": none_subtypes["validation"]
        == dict.fromkeys(config.negative_subtypes, expected_dev_subtype),
        "train_dev_semantic_overlap_pass": not (train_semantic & dev_semantic),
        "train_dev_template_overlap_pass": not (train_templates & dev_templates),
        "forbidden_content_pass": not forbidden_hits,
        "model_output_absence_pass": dataset["policy"]["contains_model_outputs"]
        is False,
        "teacher_output_absence_pass": dataset["policy"]["contains_teacher_outputs"]
        is False,
        "benchmark_content_absence_pass": dataset["source"][
            "benchmark_content_used"
        ]
        is False,
        "token_accounting_pass": (
            train_tokens is None or train_tokens >= config.minimum_train_tokens
        ),
        "tokenizer_identity_pass": (
            tokenizer_path is None or len(tokenizer_files) == 3
        ),
    }
    release = {
        "schema_version": RELEASE_SCHEMA,
        "release_id": config.dataset_id,
        "dataset_schema": dataset["schema_version"],
        "source": {
            "dataset_canonical_sha256": canonical_sha256(dataset),
            "multiclass_report_sha256": config.multiclass_report_sha256,
            "binary_detector_report_sha256": (
                config.binary_detector_report_sha256
            ),
            "tokenizer_file_sha256": tokenizer_files,
        },
        "accepted": {
            "rows": len(samples),
            "train_rows": sum(row["split"] == "train" for row in samples),
            "dev_rows": sum(
                row["split"] == "validation" for row in samples
            ),
            "train_by_label": train_label,
            "dev_by_label": dev_label,
            "train_none_by_subtype": none_subtypes["train"],
            "dev_none_by_subtype": none_subtypes["validation"],
            "train_tokens": train_tokens,
        },
        "overlap": {
            "train_dev_semantic": len(train_semantic & dev_semantic),
            "train_dev_template": len(train_templates & dev_templates),
        },
        "leakage": {"forbidden_terms": forbidden_hits},
        "checks": checks,
        "training_unblocked": all(checks.values()),
        "claim_boundary": (
            "This release proves only that a deterministic synthetic router "
            "classification dataset passed frozen balance, deduplication, "
            "split, token, provenance, and leakage gates. It is not model "
            "quality or benchmark evidence and unlocks only one separately "
            "pre-registered bounded SFT smoke."
        ),
    }
    return release
