from __future__ import annotations

import hashlib
import json
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
from nano_data_pipeline.router_classification import SYSTEM_PROMPT
from nano_data_pipeline.router_negative_diversity import (
    CONTRACT_SCHEMA,
    is_explicit_classification_prompt,
)
from nano_data_pipeline.subagent_campaign import count_tokens


BUILD_SCHEMA = "nano_router_negative_diversity_build_v2"
RELEASE_SCHEMA = "nano_router_negative_diversity_release_v2"
SCENARIOS = (
    "auditorium",
    "warehouse",
    "research_lab",
    "community_center",
    "manufacturing_site",
)
QUESTION_FORMS = (
    "Compute the exact requested value.",
    "What is the exact result?",
    "Find the requested integer.",
    "Determine the final quantity exactly.",
)


@dataclass(frozen=True)
class RouterNegativeDiversityBuildConfig:
    schema_version: str
    audit_path: str
    audit_sha256: str
    contract_path: str
    contract_sha256: str
    source_dataset_path: str
    source_dataset_sha256: str
    source_release_path: str
    source_release_sha256: str
    integration_preregistrations: list[dict[str, str]]
    benchmark_sources: list[dict[str, str]]
    tokenizer_path: str
    output_dataset_path: str
    output_release_path: str


def load_build_config(
    path: str | Path,
) -> RouterNegativeDiversityBuildConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(
        RouterNegativeDiversityBuildConfig.__dataclass_fields__
    ):
        raise ValueError("router negative diversity build config fields differ")
    config = RouterNegativeDiversityBuildConfig(**raw)
    expected = {
        "schema_version": BUILD_SCHEMA,
        "audit_path": (
            "docs/datasets/qwen35_router_negative_diversity_v2.public.json"
        ),
        "audit_sha256": (
            "9aaa69de746dbdc5cefbb52fb271c8f9ec86716d10ada70704c7e346dc2f7c17"
        ),
        "contract_path": (
            "configs/router_classification/"
            "qwen35_router_negative_diversity_release_v2.json"
        ),
        "contract_sha256": (
            "c195a7373ea283546dde1866f70593f0912833d987ff5f1a8cb424c2bc340335"
        ),
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
        "integration_preregistrations": [
            {
                "name": "integration_v1",
                "path": (
                    "../nano-harness-fullstack-traex-03/docs/experiments/"
                    "qwen35_router_adapter_integration_v1.preregister.json"
                ),
                "sha256": (
                    "ed5c4e6800385e7a4bfce0aed027bd1f81a6854bb1ed5b3f6aa0cc6e808491f3"
                ),
            },
            {
                "name": "integration_v2",
                "path": (
                    "../nano-harness-fullstack-traex-03/docs/experiments/"
                    "qwen35_router_adapter_integration_v2.preregister.json"
                ),
                "sha256": (
                    "1a23a1bb391a3ebac7e70aecd5e2d855ef624e825c26e6bfe9ed942d07cc9e2e"
                ),
            },
        ],
        "benchmark_sources": [
            {
                "name": "gsm8k",
                "path": (
                    "../../../datasets/gsm8k/gsm8k/main/"
                    "test-00000-of-00001.parquet"
                ),
                "prompt_column": "question",
                "sha256": (
                    "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
                ),
            },
            {
                "name": "mmlu",
                "path": (
                    "../../../datasets/mmlu_no_train/mmlu_no_train/all/"
                    "test-00000-of-00001.parquet"
                ),
                "prompt_column": "question",
                "sha256": (
                    "02033371a64dbe5a0d8b6fb9d612900afcd0fea5140e53490993a4540b3a58fd"
                ),
            },
            {
                "name": "gpqa_diamond",
                "path": (
                    "../../../datasets/GPQA-Diamond/GPQA-Diamond/test/"
                    "gpqa_diamond.parquet"
                ),
                "prompt_column": "question",
                "sha256": (
                    "fdd6e95117cdf87075f56bf673a5bae4680b143bc2d29b486470810122c33f39"
                ),
            },
        ],
        "tokenizer_path": "../../../models/Qwen3.5-4B",
        "output_dataset_path": (
            "datasets/qwen35_router_negative_diversity_v2.json"
        ),
        "output_release_path": (
            "manifests/qwen35_router_negative_diversity_v2.release.json"
        ),
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"router negative diversity build freezes {field}={expected_value}"
            )
    for source, digest in (
        (config.audit_path, config.audit_sha256),
        (config.contract_path, config.contract_sha256),
        (config.source_dataset_path, config.source_dataset_sha256),
        (config.source_release_path, config.source_release_sha256),
    ):
        if sha256_file(Path(source)) != digest:
            raise ValueError("router negative diversity source identity differs")
    for source in (
        *config.integration_preregistrations,
        *config.benchmark_sources,
    ):
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError(
                f"router negative diversity external identity differs: "
                f"{source['name']}"
            )
    audit = json.loads(Path(config.audit_path).read_text(encoding="utf-8"))
    contract = json.loads(
        Path(config.contract_path).read_text(encoding="utf-8")
    )
    if (
        audit.get("decision", {}).get("generate_negative_diversity_v2_next")
        is not True
        or audit.get("decision", {}).get("training_allowed_now") is not False
        or contract.get("schema_version") != CONTRACT_SCHEMA
        or contract.get("training_unblocked_only_after_release") is not True
    ):
        raise ValueError("router negative diversity audit decision differs")
    return config


def _template(template_id: int) -> tuple[str, str]:
    return SCENARIOS[template_id // 4], QUESTION_FORMS[template_id % 4]


def _positive_prompt(
    label: str,
    template_id: int,
    value: int,
) -> tuple[str, str]:
    scenario, question = _template(template_id)
    if label == "A":
        rows = value * 3 + 101
        columns = value * 2 + 103
        extra = value * 5 + 107
        scale = ("twice", "three times")[template_id % 2]
        prompt = (
            f"For a {scenario}, a rectangular layout has {rows} rows with "
            f"{columns} positions per row. The planned capacity is {scale} "
            f"the rectangular count, plus {extra} additional positions. "
            f"{question}"
        )
        return prompt, "implicit_scale_total"
    units = value * 2 + 109
    price = value * 3 + 113
    net = value * 4 + 127
    recurring = units * price - net
    setup = net * (value * 2 + 131)
    prompt = (
        f"A {scenario} project costs {setup} before operation. Each whole "
        f"period it sells {units} units at {price} each and pays {recurring} "
        "in recurring expenses. Find the first whole period when cumulative "
        f"profit is strictly greater than zero. {question}"
    )
    return prompt, "first_strict_profit_period"


def _negative_prompt(
    subtype: str,
    template_id: int,
    value: int,
) -> str:
    scenario, question = _template(template_id)
    if subtype == "box_total":
        boxes = value * 2 + 137
        per_box = value * 3 + 139
        loose = value * 7 + 149
        return (
            f"A {scenario} received {boxes} containers holding {per_box} "
            f"components each, plus {loose} loose components. Find the exact "
            f"total number of components. {question}"
        )
    if subtype == "remaining_stock":
        batches = value * 2 + 151
        units = value * 3 + 157
        remaining = value * 5 + 163
        starting = batches * units + remaining
        return (
            f"A {scenario} starts with {starting} units and consumes {batches} "
            f"batches of {units} units. How many units remain? {question}"
        )
    if subtype == "paired_average":
        first = value * 10 + 170
        second = value * 14 + 174
        return (
            f"Two audited {scenario} totals are {first} and {second}. Find "
            f"their exact arithmetic mean. {question}"
        )
    if subtype == "single_operation":
        mode = template_id % 4
        left = value * 5 + 167
        right = value * 2 + 173
        operation = ("sum", "difference", "product", "integer quotient")[mode]
        if mode == 3:
            left = right * (value % 97 + 11)
        return (
            f"A {scenario} ledger gives the integers {left} and {right}. Find "
            f"their exact {operation}. {question}"
        )
    if subtype == "weighted_total":
        first_count = value * 2 + 179
        first_weight = value % 19 + 7
        second_count = value * 3 + 181
        second_weight = value % 23 + 11
        return (
            f"A {scenario} has {first_count} items weighing {first_weight} "
            f"units each and {second_count} items weighing {second_weight} "
            f"units each. Find the combined weight. {question}"
        )
    if subtype == "quotient_remainder":
        divisor = value % 83 + 17
        quotient = value * 2 + 191
        remainder = value % divisor
        total = divisor * quotient + remainder
        return (
            f"A {scenario} distributes {total} records into groups of "
            f"{divisor}. Find the whole-number quotient and remainder. "
            f"{question}"
        )
    if subtype == "time_conversion":
        days = value % 31 + 3
        hours = value % 23
        minutes = value % 59
        return (
            f"A {scenario} process lasts {days} days, {hours} hours, and "
            f"{minutes} minutes. Convert the duration to total minutes. "
            f"{question}"
        )
    original = value * 4 + 197
    percent = (5, 10, 20, 25)[template_id % 4]
    direction = "increase" if template_id % 2 == 0 else "decrease"
    return (
        f"A {scenario} metric begins at {original} and then has a {percent}% "
        f"{direction}. Find the exact updated value. {question}"
    )


def _sample(
    *,
    split: str,
    label: str,
    subtype: str,
    template_id: int,
    index: int,
    prompt: str,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"FINAL: {label}"},
    ]
    identity = {
        "dataset_version": "v2",
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
        "difficulty": f"router_{subtype}_answer_task",
        "generation_rule": (
            f"router_negative_diversity_{subtype}_template_{template_id}_v2"
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
        "prompt_mode": "answer_task",
    }


def build_dataset(
    config: RouterNegativeDiversityBuildConfig,
) -> dict[str, Any]:
    contract = json.loads(
        Path(config.contract_path).read_text(encoding="utf-8")
    )
    samples = []
    split_specs = (
        ("train", range(16), 200_000),
        ("validation", range(16, 20), 800_000),
    )
    for split, templates, offset in split_specs:
        positive_count = contract[f"{'train' if split == 'train' else 'dev'}_by_label"]["A"]
        per_positive_template = positive_count // len(templates)
        for label in ("A", "B"):
            for template_id in templates:
                for index in range(per_positive_template):
                    value = offset + template_id * 10_000 + index
                    prompt, subtype = _positive_prompt(
                        label, template_id, value
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
        for subtype_index, subtype in enumerate(
            contract["negative_subtypes"]
        ):
            count = contract["negative_subtypes"][subtype][
                f"{'train' if split == 'train' else 'dev'}_rows"
            ]
            per_template = count // len(templates)
            for template_id in templates:
                for index in range(per_template):
                    value = (
                        offset
                        + 2_000_000
                        + subtype_index * 200_000
                        + template_id * 5_000
                        + index
                    )
                    samples.append(
                        _sample(
                            split=split,
                            label="C",
                            subtype=subtype,
                            template_id=template_id,
                            index=index,
                            prompt=_negative_prompt(
                                subtype, template_id, value
                            ),
                        )
                    )
    samples.sort(key=lambda row: row["sample_id"])
    dataset = {
        "schema_version": "nano_analog_dataset_v1",
        "dataset_id": contract["dataset_id"],
        "version": "v2",
        "source": {
            "source_kind": "deterministic_synthetic",
            "generator": (
                "nano_data_pipeline.router_negative_diversity_release"
            ),
            "audit_sha256": config.audit_sha256,
            "contract_sha256": config.contract_sha256,
            "benchmark_content_used": False,
            "sealed_case_ids_used": False,
            "integration_rows_or_outputs_used": False,
        },
        "policy": {
            "source_split": "non_eval_analog_only",
            "training_allowed": True,
            "contains_benchmark_content": False,
            "contains_model_outputs": False,
            "contains_teacher_outputs": False,
            "contains_integration_rows_or_outputs": False,
            "purpose": "router_negative_diversity_sft_v2",
        },
        "label_contract": contract["label_contract"],
        "samples": samples,
    }
    dataset["summary"] = summarize_analog_dataset(dataset)
    validate_analog_dataset(dataset)
    return dataset


def _prompt_hashes_from_preregister(path: Path) -> set[str]:
    prereg = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["prompt_sha256"] for row in prereg["case_contract"]["cases"]
    }


def validate_release(
    dataset: dict[str, Any],
    *,
    config: RouterNegativeDiversityBuildConfig,
    tokenizer: Any,
    tokenizer_path: Path,
) -> dict[str, Any]:
    import pyarrow.parquet as parquet

    validate_analog_dataset(dataset)
    contract = json.loads(
        Path(config.contract_path).read_text(encoding="utf-8")
    )
    source_dataset = json.loads(
        Path(config.source_dataset_path).read_text(encoding="utf-8")
    )
    rows = dataset["samples"]
    by_split_label = Counter(
        (row["split"], row["route_label"]) for row in rows
    )
    by_split_subtype = Counter(
        (row["split"], row["negative_subtype"])
        for row in rows
        if row["route_label"] == "C"
    )
    templates = {
        split: {
            subtype: {
                row["template_id"]
                for row in rows
                if row["split"] == split
                and row["negative_subtype"] == subtype
            }
            for subtype in contract["negative_subtypes"]
        }
        for split in ("train", "validation")
    }
    answer_task = {
        split: [
            row
            for row in rows
            if row["split"] == split
            and row["prompt_mode"] == "answer_task"
            and not is_explicit_classification_prompt(
                row["messages"][1]["content"]
            )
        ]
        for split in ("train", "validation")
    }
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation")
    }
    source_ids = {row["sample_id"] for row in source_dataset["samples"]}
    source_semantic = {
        row["semantic_sha256"] for row in source_dataset["samples"]
    }
    train_semantic = {
        row["semantic_sha256"] for row in split_rows["train"]
    }
    dev_semantic = {
        row["semantic_sha256"] for row in split_rows["validation"]
    }
    prompt_hashes = {
        hashlib.sha256(row["messages"][1]["content"].encode()).hexdigest()
        for row in rows
    }
    integration_overlap = {
        source["name"]: len(
            prompt_hashes
            & _prompt_hashes_from_preregister(Path(source["path"]))
        )
        for source in config.integration_preregistrations
    }
    benchmark_overlap = {}
    benchmark_rows = {}
    for source in config.benchmark_sources:
        values = parquet.read_table(
            source["path"], columns=[source["prompt_column"]]
        )[source["prompt_column"]].to_pylist()
        hashes = {
            hashlib.sha256(str(value).encode()).hexdigest()
            for value in values
        }
        benchmark_overlap[source["name"]] = len(prompt_hashes & hashes)
        benchmark_rows[source["name"]] = len(values)
    train_tokens = sum(
        count_tokens(tokenizer, row["messages"])
        for row in split_rows["train"]
    )
    tokenizer_files = {
        filename: sha256_file(tokenizer_path / filename)
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        )
    }
    train_label = {
        label: by_split_label[("train", label)] for label in ("A", "B", "C")
    }
    dev_label = {
        label: by_split_label[("validation", label)]
        for label in ("A", "B", "C")
    }
    train_subtypes = {
        subtype: by_split_subtype[("train", subtype)]
        for subtype in contract["negative_subtypes"]
    }
    dev_subtypes = {
        subtype: by_split_subtype[("validation", subtype)]
        for subtype in contract["negative_subtypes"]
    }
    answer_fractions = {
        split: len(answer_task[split]) / len(split_rows[split])
        for split in split_rows
    }
    forbidden_terms = (
        "gsm8k",
        "mmlu",
        "gpqa",
        "canary",
        "holdout",
        "router-adapter-v1",
        "router-adapter-v2",
    )
    rendered = json.dumps(rows, ensure_ascii=False).casefold()
    forbidden_hits = [term for term in forbidden_terms if term in rendered]
    checks = {
        "exact_row_and_class_balance": (
            len(split_rows["train"]) == contract["rows"]["train"]
            and len(split_rows["validation"]) == contract["rows"]["dev"]
            and train_label == contract["train_by_label"]
            and dev_label == contract["dev_by_label"]
        ),
        "exact_negative_subtype_balance": (
            train_subtypes
            == {
                subtype: row["train_rows"]
                for subtype, row in contract["negative_subtypes"].items()
            }
            and dev_subtypes
            == {
                subtype: row["dev_rows"]
                for subtype, row in contract["negative_subtypes"].items()
            }
        ),
        "template_diversity_pass": all(
            len(templates["train"][subtype])
            >= contract["negative_subtypes"][subtype][
                "minimum_train_templates"
            ]
            and len(templates["validation"][subtype])
            >= contract["negative_subtypes"][subtype][
                "minimum_dev_templates"
            ]
            for subtype in contract["negative_subtypes"]
        ),
        "answer_task_fraction_pass": (
            answer_fractions["train"]
            >= contract["lexical_contract"][
                "minimum_answer_task_fraction_train"
            ]
        ),
        "dev_answer_task_only_pass": (
            answer_fractions["validation"]
            == contract["lexical_contract"][
                "minimum_answer_task_fraction_dev"
            ]
        ),
        "all_sample_ids_unique": (
            len({row["sample_id"] for row in rows}) == len(rows)
        ),
        "all_exact_hashes_unique": (
            len({row["exact_sha256"] for row in rows}) == len(rows)
        ),
        "all_semantic_hashes_unique": (
            len({row["semantic_sha256"] for row in rows}) == len(rows)
        ),
        "all_overlap_counts_zero": (
            not (
                {row["sample_id"] for row in rows} & source_ids
                or {row["semantic_sha256"] for row in rows}
                & source_semantic
                or train_semantic & dev_semantic
                or any(integration_overlap.values())
                or any(benchmark_overlap.values())
            )
        ),
        "minimum_train_tokens_pass": (
            train_tokens >= contract["minimum_train_tokens"]
        ),
        "tokenizer_identity_pinned": len(tokenizer_files) == 3,
        "forbidden_content_zero": not forbidden_hits,
    }
    release = {
        "schema_version": RELEASE_SCHEMA,
        "release_id": contract["dataset_id"],
        "source": {
            "dataset_canonical_sha256": _hash(_canonical_json(dataset)),
            "audit_sha256": config.audit_sha256,
            "contract_sha256": config.contract_sha256,
            "source_dataset_sha256": config.source_dataset_sha256,
            "source_release_sha256": config.source_release_sha256,
            "tokenizer_file_sha256": tokenizer_files,
        },
        "accepted": {
            "rows": len(rows),
            "train_rows": len(split_rows["train"]),
            "dev_rows": len(split_rows["validation"]),
            "train_by_label": train_label,
            "dev_by_label": dev_label,
            "train_none_by_subtype": train_subtypes,
            "dev_none_by_subtype": dev_subtypes,
            "train_templates_by_subtype": {
                subtype: len(value)
                for subtype, value in templates["train"].items()
            },
            "dev_templates_by_subtype": {
                subtype: len(value)
                for subtype, value in templates["validation"].items()
            },
            "answer_task_fraction": answer_fractions,
            "train_tokens": train_tokens,
        },
        "overlap": {
            "source_v1_sample_ids": len(
                {row["sample_id"] for row in rows} & source_ids
            ),
            "source_v1_semantic": len(
                {row["semantic_sha256"] for row in rows} & source_semantic
            ),
            "train_dev_semantic": len(train_semantic & dev_semantic),
            "integration_prompts": integration_overlap,
            "benchmark_prompts": benchmark_overlap,
            "benchmark_rows_hashed": benchmark_rows,
        },
        "leakage": {"forbidden_terms": forbidden_hits},
        "policy": dataset["policy"],
        "checks": checks,
        "training_unblocked": all(checks.values()),
        "claim_boundary": (
            "This release proves only deterministic synthetic data quality, "
            "balance, diversity, token, overlap, provenance, and leakage "
            "gates. It does not establish model quality and unlocks only one "
            "separately pre-registered router SFT run."
        ),
    }
    return release
