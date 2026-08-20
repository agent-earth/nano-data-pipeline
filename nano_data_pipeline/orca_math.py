from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from nano_data_pipeline.feedback import sha256_file


CONFIG_SCHEMA = "nano_orca_math_sft_config_v1"
PREREGISTER_SCHEMA = "nano_orca_math_sft_preregister_v1"


@dataclass(frozen=True)
class OrcaMathConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parent.parent

    def resolve(self, value: str) -> Path:
        return (self.root / value).resolve()


def load_config(path: str | Path) -> OrcaMathConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported Orca Math config schema")
    _validate_config(raw)
    return OrcaMathConfig(path=config_path, raw=raw)


def _validate_config(raw: dict[str, Any]) -> None:
    source = raw["source"]
    selection = raw["selection"]
    token_accounting = raw["token_accounting"]
    quality = raw["quality"]
    boundary = raw["training_boundary"]
    strata = selection["strata"]
    if (
        raw["dataset_id"] != "orca-math-sft-v1"
        or source["repository"]
        != "microsoft/orca-math-word-problems-200k"
        or source["dataset_revision"]
        != "29255d1770cc4eac66e5e7fa378cba542c026350"
        or source["license"] != "mit"
        or source["parquet_rows"] != 200_035
        or source["question_column"] != "question"
        or source["answer_column"] != "answer"
    ):
        raise ValueError("Orca Math source contract differs")
    if (
        selection["train_rows"] != 32_768
        or selection["dev_rows"] != 1_024
        or selection["smoke_dev_rows"] != 192
        or selection["seed"] != "orca-math-sft-v1:20260821"
        or selection["difficulty_proxy"] != "teacher_answer_char_count"
        or sum(row["train_rows"] for row in strata.values()) != 32_768
        or sum(row["dev_rows"] for row in strata.values()) != 1_024
        or strata["short"] != {
            "answer_chars_max": 600,
            "train_rows": 8_192,
            "dev_rows": 256,
        }
        or strata["medium"] != {
            "answer_chars_min": 601,
            "answer_chars_max": 1_000,
            "train_rows": 16_384,
            "dev_rows": 512,
        }
        or strata["long"] != {
            "answer_chars_min": 1_001,
            "train_rows": 8_192,
            "dev_rows": 256,
        }
    ):
        raise ValueError("Orca Math split contract differs")
    if (
        token_accounting["unit"] != "qwen3.5_tokenizer_input_id"
        or token_accounting["enable_thinking"] is not False
        or token_accounting["minimum_train_tokens"] != 10_000_000
        or token_accounting["max_sequence_tokens"] != 1_024
        or set(token_accounting["tokenizer_files"])
        != {
            "chat_template.jinja",
            "tokenizer.json",
            "tokenizer_config.json",
        }
    ):
        raise ValueError("Orca Math token contract differs")
    if (
        quality["answer_min_chars"] != 24
        or quality["answer_max_chars"] != 6_000
        or quality["exact_question_duplicates_allowed"] != 0
        or quality["forbidden_exact_overlap_allowed"] != 0
        or quality["forbidden_near_overlap_allowed"] != 0
        or quality["selected_near_duplicate_threshold"] != 0.92
    ):
        raise ValueError("Orca Math quality contract differs")
    if (
        boundary["benchmark_rows_training_eligible"] is not False
        or boundary["benchmark_text_published"] is not False
        or boundary["source_model_outputs_generated_locally"] is not False
        or boundary["rl_or_opd_unlocked"] is not False
        or boundary["full_release_unlocks_only_preregistered_sft_smoke"]
        is not True
    ):
        raise ValueError("Orca Math training boundary differs")
    if len(raw["forbidden_corpora"]) != 6 or any(
        not row.get("sha256") or not row.get("question_column")
        for row in raw["forbidden_corpora"]
    ):
        raise ValueError("Orca Math forbidden corpus contract differs")


def _parquet_identity(
    path: Path,
    *,
    required_columns: set[str],
) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema_arrow.names)
    if not required_columns.issubset(columns):
        raise ValueError(
            f"required columns missing from {path}: "
            f"{sorted(required_columns - columns)}"
        )
    return {
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.num_row_groups,
        "columns": sorted(columns),
        "sha256": sha256_file(path),
    }


def build_preregister(config: OrcaMathConfig) -> dict[str, Any]:
    raw = config.raw
    source = raw["source"]
    source_path = config.resolve(source["parquet_path"])
    metadata_path = config.resolve(source["metadata_path"])
    source_identity = _parquet_identity(
        source_path,
        required_columns={
            source["question_column"],
            source["answer_column"],
        },
    )
    if (
        source_identity["rows"] != source["parquet_rows"]
        or source_identity["sha256"] != source["parquet_sha256"]
        or sha256_file(metadata_path) != source["metadata_sha256"]
    ):
        raise ValueError("Orca Math source identity differs")

    tokenizer = raw["token_accounting"]
    tokenizer_path = config.resolve(tokenizer["tokenizer_path"])
    tokenizer_files = {
        filename: sha256_file(tokenizer_path / filename)
        for filename in tokenizer["tokenizer_files"]
    }
    if tokenizer_files != tokenizer["tokenizer_files"]:
        raise ValueError("Orca Math tokenizer identity differs")

    forbidden = []
    for row in raw["forbidden_corpora"]:
        path = config.resolve(row["path"])
        identity = _parquet_identity(
            path,
            required_columns={row["question_column"]},
        )
        if identity["sha256"] != row["sha256"]:
            raise ValueError("Orca Math forbidden corpus identity differs")
        forbidden.append(
            {
                "path_role": path.name,
                "question_column": row["question_column"],
                **identity,
            }
        )
    return {
        "schema_version": PREREGISTER_SCHEMA,
        "dataset_id": raw["dataset_id"],
        "identity": {
            "config_sha256": sha256_file(config.path),
            "source_parquet_sha256": source_identity["sha256"],
            "source_metadata_sha256": sha256_file(metadata_path),
            "tokenizer_file_sha256": tokenizer_files,
            "forbidden_corpora_sha256": [
                row["sha256"] for row in forbidden
            ],
        },
        "source": {
            "repository": source["repository"],
            "revision": source["dataset_revision"],
            "license": source["license"],
            "rows": source_identity["rows"],
            "row_groups": source_identity["row_groups"],
            "columns": source_identity["columns"],
        },
        "selection": raw["selection"],
        "quality": raw["quality"],
        "token_accounting": {
            "unit": tokenizer["unit"],
            "enable_thinking": tokenizer["enable_thinking"],
            "minimum_train_tokens": tokenizer["minimum_train_tokens"],
            "max_sequence_tokens": tokenizer["max_sequence_tokens"],
        },
        "forbidden_corpora": forbidden,
        "transformation": raw["transformation"],
        "training_boundary": {
            **raw["training_boundary"],
            "data_generation_started": False,
            "training_started": False,
            "this_receipt_only_preregisters": True,
        },
        "forbidden_after_observation": [
            "source_change",
            "split_seed_change",
            "strata_change",
            "row_target_change",
            "token_target_change",
            "sequence_budget_change",
            "answer_extraction_change",
            "dedup_threshold_change",
            "forbidden_corpus_change",
            "quality_threshold_change",
        ],
        "claim_boundary": (
            "This receipt freezes an external non-benchmark SFT data release "
            "before sample selection. It contains no selected source rows, "
            "benchmark text, model generations, or model-quality result."
        ),
    }
