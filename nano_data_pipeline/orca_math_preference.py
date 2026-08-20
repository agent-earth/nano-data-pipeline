from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from nano_data_pipeline.feedback import sha256_file
from nano_data_pipeline.subagent_campaign import (
    canonical_json,
    count_tokens,
    sha256_text,
)


CONFIG_SCHEMA = "nano_orca_math_preference_config_v1"
SAMPLE_SCHEMA = "nano_orca_math_preference_sample_v1"
RELEASE_SCHEMA = "nano_orca_math_preference_release_v1"


@dataclass(frozen=True)
class PreferenceConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parent.parent

    def resolve(self, value: str) -> Path:
        return (self.root / value).resolve()


def load_config(path: str | Path) -> PreferenceConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported preference config schema")
    selection = raw["selection"]
    boundary = raw["training_boundary"]
    if (
        raw["dataset_id"] != "orca-math-preference-v1"
        or selection["train_rows"] != 512
        or selection["dev_rows"] != 192
        or selection["train_rows_by_stratum"]
        != {"short": 128, "medium": 256, "long": 128}
        or selection["dev_rows_by_stratum"]
        != {"short": 48, "medium": 96, "long": 48}
        or selection["max_sequence_tokens"] != 512
        or selection["seed"] != "orca-math-preference-v1:20260821"
        or boundary
        != {
            "benchmark_rows_training_eligible": False,
            "benchmark_text_published": False,
            "prior_sft_rows_reused": False,
            "rl_or_opd_unlocked": False,
            "source_model_outputs_generated_locally": False,
        }
    ):
        raise ValueError("preference release contract differs")
    return PreferenceConfig(path=config_path, raw=raw)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _rank(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}\n{sample_id}".encode()).hexdigest()


def _numeric(value: str) -> Fraction:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
    return Fraction(Decimal(value))


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def rejected_value(expected: str, sample_id: str) -> str:
    digest = hashlib.sha256(sample_id.encode()).digest()
    magnitude = 1 + digest[0] % 7
    sign = -1 if digest[1] % 2 else 1
    rejected = _numeric(expected) + sign * magnitude
    if rejected == _numeric(expected):
        raise ValueError("preference rejected value equals expected")
    return _format_fraction(rejected)


def replace_final(target: str, expected: str, rejected: str) -> str:
    pattern = re.compile(rf"(?m)^FINAL: {re.escape(expected)}\s*$")
    matches = list(pattern.finditer(target))
    if len(matches) != 1:
        raise ValueError("preference target needs one exact FINAL suffix")
    start, end = matches[0].span()
    return target[:start] + f"FINAL: {rejected}" + target[end:]


def _load_prior_ids(config: PreferenceConfig) -> set[str]:
    prior = config.raw["prior_sft"]
    preregister_path = config.resolve(prior["preregister_path"])
    public_path = config.resolve(prior["public_result_path"])
    if (
        sha256_file(preregister_path) != prior["preregister_sha256"]
        or sha256_file(public_path) != prior["public_result_sha256"]
    ):
        raise ValueError("preference prior SFT identity differs")
    preregister = json.loads(preregister_path.read_text(encoding="utf-8"))
    result = json.loads(public_path.read_text(encoding="utf-8"))
    if (
        preregister.get("schema_version")
        != "nano_train_orca_math_sft_preregister_v1"
        or result.get("schema_version")
        != "nano_train_orca_math_sft_public_v1"
        or result.get("decision", {}).get("candidate_admitted") is not False
    ):
        raise ValueError("preference prior SFT boundary differs")
    return set(preregister["selection"]["train_sample_ids"]) | set(
        preregister["selection"]["dev_sample_ids"]
    )


def build_preregister(config: PreferenceConfig) -> dict[str, Any]:
    raw = config.raw
    source = raw["source"]
    dataset_path = config.resolve(source["dataset_path"])
    release_path = config.resolve(source["release_path"])
    if (
        sha256_file(dataset_path) != source["dataset_sha256"]
        or sha256_file(release_path) != source["release_sha256"]
    ):
        raise ValueError("preference source identity differs")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if (
        release.get("schema_version") != "nano_orca_math_sft_release_v1"
        or release.get("training_unblocked") is not True
        or release.get("source", {}).get("dataset_file_sha256")
        != source["dataset_sha256"]
    ):
        raise ValueError("preference source release is not admitted")
    prior_ids = _load_prior_ids(config)
    tokenizer = raw["tokenizer"]
    tokenizer_path = config.resolve(tokenizer["path"])
    tokenizer_files = {
        filename: sha256_file(tokenizer_path / filename)
        for filename in tokenizer["tokenizer_files"]
    }
    if tokenizer_files != tokenizer["tokenizer_files"]:
        raise ValueError("preference tokenizer identity differs")
    return {
        "schema_version": "nano_orca_math_preference_preregister_v1",
        "dataset_id": raw["dataset_id"],
        "identity": {
            "config_sha256": sha256_file(config.path),
            "source_dataset_sha256": source["dataset_sha256"],
            "source_release_sha256": source["release_sha256"],
            "prior_sft_preregister_sha256": raw["prior_sft"][
                "preregister_sha256"
            ],
            "prior_sft_public_result_sha256": raw["prior_sft"][
                "public_result_sha256"
            ],
            "tokenizer_files": tokenizer_files,
            "prior_sft_ids_sha256": sha256_text(
                "\n".join(sorted(prior_ids))
            ),
        },
        "selection": raw["selection"],
        "transformation": raw["transformation"],
        "training_boundary": {
            **raw["training_boundary"],
            "data_generation_started": False,
            "training_started": False,
            "this_receipt_only_preregisters": True,
        },
        "claim_boundary": (
            "This receipt freezes a fresh non-benchmark preference release "
            "before selecting rows. It contains no prompts, answers, chosen "
            "or rejected targets, model outputs, or model-quality result."
        ),
    }


def build_dataset(
    config: PreferenceConfig,
    *,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = config.raw
    source_rows = read_jsonl(config.resolve(raw["source"]["dataset_path"]))
    prior_ids = _load_prior_ids(config)
    available = [
        row
        for row in source_rows
        if row["sample_id"] not in prior_ids
        and len(row["messages"]) == 3
        and row["split"] in {"train", "dev"}
    ]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in available:
        buckets.setdefault(row["stratum"], []).append(row)
    selected = []
    for stratum in ("short", "medium", "long"):
        contract_train = raw["selection"]["train_rows_by_stratum"][stratum]
        contract_dev = raw["selection"]["dev_rows_by_stratum"][stratum]
        ranked = sorted(
            buckets[stratum],
            key=lambda row: (
                _rank(raw["selection"]["seed"], row["sample_id"]),
                row["sample_id"],
            ),
        )
        eligible = []
        for row in ranked:
            expected = str(row["numeric_answer"])
            rejected = rejected_value(expected, row["sample_id"])
            chosen = row["messages"][-1]["content"]
            rejected_target = replace_final(chosen, expected, rejected)
            prompt_messages = row["messages"][:-1]
            chosen_messages = prompt_messages + [
                {"role": "assistant", "content": chosen}
            ]
            rejected_messages = prompt_messages + [
                {"role": "assistant", "content": rejected_target}
            ]
            chosen_tokens = count_tokens(tokenizer, chosen_messages)
            rejected_tokens = count_tokens(tokenizer, rejected_messages)
            if max(chosen_tokens, rejected_tokens) > raw["selection"][
                "max_sequence_tokens"
            ]:
                continue
            eligible.append(
                {
                    "source": row,
                    "expected": expected,
                    "rejected_value": rejected,
                    "chosen": chosen,
                    "rejected": rejected_target,
                    "chosen_tokens": chosen_tokens,
                    "rejected_tokens": rejected_tokens,
                }
            )
            if len(eligible) == contract_train + contract_dev:
                break
        if len(eligible) != contract_train + contract_dev:
            raise ValueError("insufficient preference rows")
        for index, item in enumerate(eligible):
            split = "dev" if index < contract_dev else "train"
            source_row = item["source"]
            sample_id = "orca-pref-" + sha256_text(
                canonical_json(
                    {
                        "dataset_id": raw["dataset_id"],
                        "source_sample_id": source_row["sample_id"],
                        "split": split,
                    }
                )
            )[:24]
            selected.append(
                {
                    "schema_version": SAMPLE_SCHEMA,
                    "sample_id": sample_id,
                    "source_sample_id": source_row["sample_id"],
                    "split": split,
                    "stratum": stratum,
                    "training_eligible": split == "train",
                    "prompt_messages": source_row["messages"][:-1],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                    "expected": item["expected"],
                    "rejected_value": item["rejected_value"],
                    "chosen_tokens": item["chosen_tokens"],
                    "rejected_tokens": item["rejected_tokens"],
                    "pair_hash": sha256_text(
                        canonical_json(
                            {
                                "prompt": source_row["messages"][:-1],
                                "chosen": item["chosen"],
                                "rejected": item["rejected"],
                            }
                        )
                    ),
                    "verifier": {
                        "kind": "numeric_preference_v1",
                        "chosen_correct": True,
                        "rejected_correct": False,
                    },
                }
            )
    selected.sort(
        key=lambda row: (
            row["split"] != "train",
            row["stratum"],
            row["sample_id"],
        )
    )
    return selected, {
        "source_rows": len(source_rows),
        "prior_sft_ids_excluded": len(prior_ids),
        "eligible_source_rows": len(available),
    }


def validate_dataset(
    config: PreferenceConfig,
    *,
    tokenizer: Any,
    path: Path,
) -> dict[str, Any]:
    rows = read_jsonl(path)
    raw = config.raw
    prior_ids = _load_prior_ids(config)
    sample_ids = set()
    source_ids = set()
    pair_hashes = set()
    checks = {
        "row_schema_pass": True,
        "chosen_rejected_prompt_identity_pass": True,
        "chosen_only_final_diff_pass": True,
        "numeric_verifier_pass": True,
        "token_recomputation_pass": True,
        "max_sequence_pass": True,
        "prior_sft_overlap_zero": True,
    }
    counts = {}
    for row in rows:
        split = row.get("split")
        stratum = row.get("stratum")
        checks["row_schema_pass"] = checks["row_schema_pass"] and (
            row.get("schema_version") == SAMPLE_SCHEMA
            and split in {"train", "dev"}
            and stratum in {"short", "medium", "long"}
            and row.get("training_eligible") is (split == "train")
        )
        source_id = row.get("source_sample_id")
        checks["prior_sft_overlap_zero"] = (
            checks["prior_sft_overlap_zero"] and source_id not in prior_ids
        )
        expected = str(row.get("expected"))
        rejected = str(row.get("rejected_value"))
        chosen = str(row.get("chosen"))
        rejected_target = str(row.get("rejected"))
        try:
            reconstructed = replace_final(chosen, expected, rejected)
        except ValueError:
            reconstructed = ""
        checks["chosen_only_final_diff_pass"] = (
            checks["chosen_only_final_diff_pass"]
            and reconstructed == rejected_target
        )
        checks["numeric_verifier_pass"] = (
            checks["numeric_verifier_pass"]
            and _numeric(expected) != _numeric(rejected)
            and row.get("verifier")
            == {
                "kind": "numeric_preference_v1",
                "chosen_correct": True,
                "rejected_correct": False,
            }
        )
        prompt = row.get("prompt_messages")
        chosen_tokens = count_tokens(
            tokenizer,
            prompt + [{"role": "assistant", "content": chosen}],
        )
        rejected_tokens = count_tokens(
            tokenizer,
            prompt
            + [{"role": "assistant", "content": rejected_target}],
        )
        checks["token_recomputation_pass"] = (
            checks["token_recomputation_pass"]
            and chosen_tokens == row.get("chosen_tokens")
            and rejected_tokens == row.get("rejected_tokens")
        )
        checks["max_sequence_pass"] = checks["max_sequence_pass"] and (
            max(chosen_tokens, rejected_tokens)
            <= raw["selection"]["max_sequence_tokens"]
        )
        checks["chosen_rejected_prompt_identity_pass"] = (
            checks["chosen_rejected_prompt_identity_pass"]
            and isinstance(prompt, list)
            and len(prompt) == 2
        )
        sample_ids.add(str(row.get("sample_id")))
        source_ids.add(str(source_id))
        pair_hashes.add(str(row.get("pair_hash")))
        key = f"{split}:{stratum}"
        counts[key] = counts.get(key, 0) + 1
    expected_counts = {
        f"train:{key}": value
        for key, value in raw["selection"]["train_rows_by_stratum"].items()
    } | {
        f"dev:{key}": value
        for key, value in raw["selection"]["dev_rows_by_stratum"].items()
    }
    checks.update(
        {
            "row_count_pass": len(rows)
            == raw["selection"]["train_rows"]
            + raw["selection"]["dev_rows"],
            "strata_count_pass": counts == expected_counts,
            "sample_id_unique": len(sample_ids) == len(rows),
            "source_id_unique": len(source_ids) == len(rows),
            "pair_hash_unique": len(pair_hashes) == len(rows),
        }
    )
    release = {
        "schema_version": RELEASE_SCHEMA,
        "release_id": raw["dataset_id"],
        "source": {
            "config_sha256": sha256_file(config.path),
            "source_dataset_sha256": raw["source"]["dataset_sha256"],
            "source_release_sha256": raw["source"]["release_sha256"],
            "prior_sft_preregister_sha256": raw["prior_sft"][
                "preregister_sha256"
            ],
            "prior_sft_public_result_sha256": raw["prior_sft"][
                "public_result_sha256"
            ],
            "dataset_file_sha256": sha256_file(path),
        },
        "accepted": {
            "rows": len(rows),
            "train_rows": sum(row["split"] == "train" for row in rows),
            "dev_rows": sum(row["split"] == "dev" for row in rows),
            "by_split_stratum": dict(sorted(counts.items())),
            "max_chosen_tokens": max(row["chosen_tokens"] for row in rows),
            "max_rejected_tokens": max(row["rejected_tokens"] for row in rows),
        },
        "checks": checks,
        "training_unblocked": all(checks.values()),
        "benchmark_unlocked": False,
        "claim_boundary": (
            "This release proves fresh, verifier-labeled preference pairs "
            "with identical prompt/reasoning and only the FINAL value changed. "
            "It is not model-quality evidence and unlocks one pre-registered "
            "reference-free preference optimization smoke only."
        ),
    }
    return release
