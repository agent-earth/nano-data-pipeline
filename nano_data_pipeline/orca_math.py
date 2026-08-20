from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from nano_data_pipeline.feedback import sha256_file
from nano_data_pipeline.subagent_campaign import (
    canonical_json,
    count_tokens,
    sha256_text,
)


CONFIG_SCHEMA = "nano_orca_math_sft_config_v1"
PREREGISTER_SCHEMA = "nano_orca_math_sft_preregister_v1"
SAMPLE_SCHEMA = "nano_orca_math_sft_sample_v1"
RELEASE_SCHEMA = "nano_orca_math_sft_release_v1"


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


def normalize_question(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", normalized))


def extract_numeric_answer(value: str) -> str | None:
    tail = value[-1_500:]
    candidates = []
    latex_spans = []
    for match in re.finditer(
        (
            r"\\frac\s*\{\s*([-+]?[0-9]+(?:\.[0-9]+)?)\s*\}"
            r"\s*\{\s*([0-9]+(?:\.[0-9]+)?)\s*\}"
        ),
        tail,
    ):
        latex_spans.append(match.span())
        candidates.append(
            (match.start(), f"{match.group(1)}/{match.group(2)}")
        )
    for match in re.finditer(
        (
            r"(?<![a-zA-Z0-9_])"
            r"[-+]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
            r"(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)?%?"
        ),
        tail,
    ):
        if any(start <= match.start() < end for start, end in latex_spans):
            continue
        candidates.append((match.start(), match.group(0)))
    if not candidates:
        return None
    result = max(candidates, key=lambda row: row[0])[1]
    result = result.replace(",", "").removesuffix("%")
    if "/" in result:
        numerator, denominator = result.split("/", 1)
        if float(denominator) == 0:
            return None
        return f"{numerator}/{denominator}"
    try:
        number = float(result)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


class QuestionNearDuplicateIndex:
    def __init__(
        self,
        *,
        jaccard_threshold: float,
        sequence_threshold: float,
        token_frequencies: dict[str, int] | None = None,
    ):
        if not 0 < jaccard_threshold < 1:
            raise ValueError("jaccard threshold must be in (0, 1)")
        if not 0 < sequence_threshold < 1:
            raise ValueError("sequence threshold must be in (0, 1)")
        self.jaccard_threshold = jaccard_threshold
        self.sequence_threshold = sequence_threshold
        self.token_frequencies = token_frequencies or {}
        self.normalized: list[str] = []
        self.token_sets: list[set[str]] = []
        self.prefix_index: dict[str, list[int]] = defaultdict(list)

    def _tokens(self, normalized: str) -> set[str]:
        return set(normalized.split())

    def _prefix(self, tokens: set[str]) -> list[str]:
        ordered = sorted(
            tokens,
            key=lambda value: (
                self.token_frequencies.get(value, 0),
                sha256_text(value),
                value,
            ),
        )
        length = max(
            1,
            len(ordered)
            - math.ceil(self.jaccard_threshold * len(ordered))
            + 1,
        )
        return ordered[:length]

    def add(self, normalized: str) -> None:
        index = len(self.normalized)
        tokens = self._tokens(normalized)
        self.normalized.append(normalized)
        self.token_sets.append(tokens)
        for token in self._prefix(tokens):
            self.prefix_index[token].append(index)

    def matches(self, normalized: str) -> list[dict[str, Any]]:
        tokens = self._tokens(normalized)
        candidates = {
            index
            for token in tokens
            for index in self.prefix_index.get(token, [])
        }
        matches = []
        for index in sorted(candidates):
            prior_tokens = self.token_sets[index]
            shorter = min(len(tokens), len(prior_tokens))
            longer = max(len(tokens), len(prior_tokens))
            if shorter < self.jaccard_threshold * longer:
                continue
            union = tokens | prior_tokens
            jaccard = len(tokens & prior_tokens) / len(union) if union else 1.0
            if jaccard < self.jaccard_threshold:
                continue
            forward_sequence = SequenceMatcher(
                None,
                self.normalized[index],
                normalized,
                autojunk=False,
            ).ratio()
            reverse_sequence = SequenceMatcher(
                None,
                normalized,
                self.normalized[index],
                autojunk=False,
            ).ratio()
            sequence = max(forward_sequence, reverse_sequence)
            if sequence >= self.sequence_threshold:
                matches.append(
                    {
                        "index": index,
                        "jaccard": jaccard,
                        "sequence_ratio": sequence,
                        "forward_sequence_ratio": forward_sequence,
                        "reverse_sequence_ratio": reverse_sequence,
                    }
                )
        return matches


def _forbidden_question_index(
    config: OrcaMathConfig,
) -> tuple[set[str], QuestionNearDuplicateIndex, dict[str, int]]:
    quality = config.raw["quality"]
    exact = set()
    counts = {}
    for corpus in config.raw["forbidden_corpora"]:
        path = config.resolve(corpus["path"])
        if sha256_file(path) != corpus["sha256"]:
            raise ValueError("forbidden corpus identity differs")
        rows = 0
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=8_192,
            columns=[corpus["question_column"]],
        ):
            for question in batch.column(0).to_pylist():
                normalized = normalize_question(str(question))
                if normalized and normalized not in exact:
                    exact.add(normalized)
                rows += 1
        counts[path.name + ":" + corpus["sha256"][:12]] = rows
    token_frequencies = Counter(
        token
        for normalized in exact
        for token in set(normalized.split())
    )
    index = QuestionNearDuplicateIndex(
        jaccard_threshold=quality["near_duplicate_jaccard"],
        sequence_threshold=quality["near_duplicate_sequence_ratio"],
        token_frequencies=dict(token_frequencies),
    )
    for normalized in sorted(exact):
        index.add(normalized)
    return exact, index, counts


def _stratum(answer_chars: int, strata: dict[str, Any]) -> str | None:
    for name in ("short", "medium", "long"):
        contract = strata[name]
        minimum = contract.get("answer_chars_min", 0)
        maximum = contract.get("answer_chars_max", math.inf)
        if minimum <= answer_chars <= maximum:
            return name
    return None


def _messages(
    config: OrcaMathConfig,
    *,
    question: str,
    answer: str,
    numeric_answer: str,
) -> list[dict[str, str]]:
    target = answer.strip()
    if not re.search(
        rf"(?m)^FINAL: {re.escape(numeric_answer)}\s*$",
        target,
    ):
        target += f"\nFINAL: {numeric_answer}"
    return [
        {
            "role": "system",
            "content": config.raw["transformation"]["system_prompt"],
        },
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": target},
    ]


def _selection_rank(
    seed: str,
    *,
    source_index: int,
    normalized_question: str,
) -> str:
    return sha256_text(
        f"{seed}\n{source_index}\n{sha256_text(normalized_question)}"
    )


def _quantiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {}
    result = {}
    for label, fraction in (
        ("min", 0.0),
        ("p25", 0.25),
        ("p50", 0.5),
        ("p75", 0.75),
        ("p90", 0.9),
        ("p95", 0.95),
        ("p99", 0.99),
        ("max", 1.0),
    ):
        index = round(fraction * (len(ordered) - 1))
        result[label] = ordered[index]
    return result


def build_dataset(
    config: OrcaMathConfig,
    *,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    preregister = build_preregister(config)
    raw = config.raw
    source = raw["source"]
    selection = raw["selection"]
    quality = raw["quality"]
    source_path = config.resolve(source["parquet_path"])
    forbidden_exact, forbidden_index, forbidden_counts = (
        _forbidden_question_index(config)
    )
    source_seen = set()
    candidates: dict[str, list[dict[str, Any]]] = {
        name: [] for name in selection["strata"]
    }
    rejected = Counter()
    source_index = 0
    for batch in pq.ParquetFile(source_path).iter_batches(
        batch_size=8_192,
        columns=[source["question_column"], source["answer_column"]],
    ):
        questions = batch.column(0).to_pylist()
        answers = batch.column(1).to_pylist()
        for question_value, answer_value in zip(questions, answers):
            question = str(question_value).strip()
            answer = str(answer_value).strip()
            normalized = normalize_question(question)
            if not normalized:
                rejected["empty_question"] += 1
            elif normalized in source_seen:
                rejected["source_exact_duplicate"] += 1
            elif normalized in forbidden_exact:
                rejected["forbidden_exact_overlap"] += 1
            elif forbidden_index.matches(normalized):
                rejected["forbidden_near_overlap"] += 1
            elif not (
                quality["answer_min_chars"]
                <= len(answer)
                <= quality["answer_max_chars"]
            ):
                rejected["answer_char_bounds"] += 1
            else:
                numeric_answer = extract_numeric_answer(answer)
                stratum = _stratum(len(answer), selection["strata"])
                if numeric_answer is None:
                    rejected["numeric_answer_extraction"] += 1
                elif stratum is None:
                    rejected["difficulty_stratum"] += 1
                else:
                    source_seen.add(normalized)
                    candidates[stratum].append(
                        {
                            "source_index": source_index,
                            "question": question,
                            "answer": answer,
                            "numeric_answer": numeric_answer,
                            "normalized_question": normalized,
                            "rank": _selection_rank(
                                selection["seed"],
                                source_index=source_index,
                                normalized_question=normalized,
                            ),
                        }
                    )
                    source_index += 1
                    continue
            source_seen.add(normalized)
            source_index += 1

    selected_index = QuestionNearDuplicateIndex(
        jaccard_threshold=quality["selected_near_duplicate_threshold"],
        sequence_threshold=quality["selected_near_duplicate_threshold"],
        token_frequencies=forbidden_index.token_frequencies,
    )
    selected: list[dict[str, Any]] = []
    selected_rejected = Counter()
    for stratum in ("short", "medium", "long"):
        contract = selection["strata"][stratum]
        target = contract["train_rows"] + contract["dev_rows"]
        chosen = []
        for candidate in sorted(
            candidates[stratum],
            key=lambda row: (row["rank"], row["source_index"]),
        ):
            normalized = candidate["normalized_question"]
            if selected_index.matches(normalized):
                selected_rejected["selected_near_duplicate"] += 1
                continue
            messages = _messages(
                config,
                question=candidate["question"],
                answer=candidate["answer"],
                numeric_answer=candidate["numeric_answer"],
            )
            tokens = count_tokens(tokenizer, messages)
            if tokens > raw["token_accounting"]["max_sequence_tokens"]:
                selected_rejected["sequence_too_long"] += 1
                continue
            selected_index.add(normalized)
            candidate = {**candidate, "messages": messages, "token_count": tokens}
            chosen.append(candidate)
            if len(chosen) == target:
                break
        if len(chosen) != target:
            raise ValueError(
                f"insufficient eligible rows for {stratum}: "
                f"{len(chosen)} != {target}"
            )
        chosen.sort(
            key=lambda row: sha256_text(
                f"{selection['seed']}:split:{row['source_index']}:"
                f"{row['rank']}"
            )
        )
        for index, candidate in enumerate(chosen):
            split = "dev" if index < contract["dev_rows"] else "train"
            question_hash = sha256_text(candidate["normalized_question"])
            source_index_hash = sha256_text(
                f"{source['dataset_revision']}:{candidate['source_index']}"
            )
            sample_id = "orca-math-" + sha256_text(
                canonical_json(
                    {
                        "dataset_id": raw["dataset_id"],
                        "split": split,
                        "stratum": stratum,
                        "question_hash": question_hash,
                        "source_index_hash": source_index_hash,
                    }
                )
            )[:24]
            selected.append(
                {
                    "schema_version": SAMPLE_SCHEMA,
                    "sample_id": sample_id,
                    "split": split,
                    "stratum": stratum,
                    "training_eligible": split == "train",
                    "messages": candidate["messages"],
                    "numeric_answer": candidate["numeric_answer"],
                    "token_count": candidate["token_count"],
                    "question_hash": question_hash,
                    "exact_hash": sha256_text(
                        canonical_json(candidate["messages"])
                    ),
                    "semantic_hash": question_hash,
                    "source": {
                        "kind": "external_open_dataset",
                        "repository": source["repository"],
                        "revision": source["dataset_revision"],
                        "source_index_hash": source_index_hash,
                        "license": source["license"],
                    },
                    "verifier": {
                        "kind": "numeric_final_v1",
                        "expected": candidate["numeric_answer"],
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
    train = [row for row in selected if row["split"] == "train"]
    dev = [row for row in selected if row["split"] == "dev"]
    if sum(row["token_count"] for row in train) < raw["token_accounting"][
        "minimum_train_tokens"
    ]:
        raise ValueError("selected Orca Math train token target not met")
    audit = {
        "schema_version": "nano_orca_math_sft_build_audit_v1",
        "dataset_id": raw["dataset_id"],
        "preregister_sha256": sha256_text(
            json.dumps(
                preregister,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "source_rows": source["parquet_rows"],
        "candidate_rows": {
            name: len(rows) for name, rows in candidates.items()
        },
        "rejected": dict(sorted(rejected.items())),
        "selected_rejected": dict(sorted(selected_rejected.items())),
        "forbidden_corpus_rows": forbidden_counts,
        "selected": {
            "rows": len(selected),
            "train_rows": len(train),
            "dev_rows": len(dev),
            "train_tokens": sum(row["token_count"] for row in train),
            "dev_tokens": sum(row["token_count"] for row in dev),
            "token_quantiles": _quantiles(
                [row["token_count"] for row in selected]
            ),
            "by_split_stratum": dict(
                sorted(
                    Counter(
                        f"{row['split']}:{row['stratum']}"
                        for row in selected
                    ).items()
                )
            ),
        },
    }
    return selected, audit


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_dataset(
    config: OrcaMathConfig,
    *,
    tokenizer: Any,
    dataset_path: Path,
    build_audit: dict[str, Any],
) -> dict[str, Any]:
    raw = config.raw
    preregister = build_preregister(config)
    rows = read_jsonl(dataset_path)
    forbidden_exact, forbidden_index, forbidden_counts = (
        _forbidden_question_index(config)
    )
    selected_index = QuestionNearDuplicateIndex(
        jaccard_threshold=raw["quality"][
            "selected_near_duplicate_threshold"
        ],
        sequence_threshold=raw["quality"][
            "selected_near_duplicate_threshold"
        ],
        token_frequencies=forbidden_index.token_frequencies,
    )
    sample_ids = set()
    exact_hashes = set()
    semantic_hashes = set()
    source_hashes = set()
    overlap_counts = Counter()
    token_recomputation = True
    schema_pass = True
    verifier_pass = True
    training_boundary_pass = True
    for row in rows:
        messages = row.get("messages")
        if (
            row.get("schema_version") != SAMPLE_SCHEMA
            or not isinstance(messages, list)
            or [message.get("role") for message in messages]
            != ["system", "user", "assistant"]
            or row.get("split") not in {"train", "dev"}
            or row.get("stratum") not in {"short", "medium", "long"}
        ):
            schema_pass = False
            continue
        normalized = normalize_question(messages[1]["content"])
        question_hash = sha256_text(normalized)
        exact_hash = sha256_text(canonical_json(messages))
        token_count = count_tokens(tokenizer, messages)
        token_recomputation = token_recomputation and (
            token_count == row.get("token_count")
            and token_count <= raw["token_accounting"]["max_sequence_tokens"]
        )
        if normalized in forbidden_exact:
            overlap_counts["forbidden_exact"] += 1
        if forbidden_index.matches(normalized):
            overlap_counts["forbidden_near"] += 1
        if selected_index.matches(normalized):
            overlap_counts["selected_near"] += 1
        selected_index.add(normalized)
        expected = row.get("verifier", {}).get("expected")
        verifier_pass = verifier_pass and (
            row.get("verifier", {}).get("kind") == "numeric_final_v1"
            and row.get("numeric_answer") == expected
            and re.search(
                rf"(?m)^FINAL: {re.escape(str(expected))}\s*$",
                messages[2]["content"],
            )
            is not None
        )
        training_boundary_pass = training_boundary_pass and (
            row.get("training_eligible") is (row["split"] == "train")
            and row.get("source", {}).get("kind")
            == "external_open_dataset"
            and row.get("source", {}).get("repository")
            == raw["source"]["repository"]
        )
        schema_pass = schema_pass and (
            row.get("question_hash") == question_hash
            and row.get("semantic_hash") == question_hash
            and row.get("exact_hash") == exact_hash
        )
        sample_ids.add(str(row.get("sample_id")))
        exact_hashes.add(str(row.get("exact_hash")))
        semantic_hashes.add(str(row.get("semantic_hash")))
        source_hashes.add(str(row.get("source", {}).get("source_index_hash")))

    train = [row for row in rows if row.get("split") == "train"]
    dev = [row for row in rows if row.get("split") == "dev"]
    counts = Counter(
        f"{row.get('split')}:{row.get('stratum')}" for row in rows
    )
    expected_counts = {
        f"train:{name}": contract["train_rows"]
        for name, contract in raw["selection"]["strata"].items()
    } | {
        f"dev:{name}": contract["dev_rows"]
        for name, contract in raw["selection"]["strata"].items()
    }
    train_tokens = sum(int(row.get("token_count", 0)) for row in train)
    checks = {
        "schema_pass": schema_pass,
        "row_count_pass": (
            len(rows)
            == raw["selection"]["train_rows"]
            + raw["selection"]["dev_rows"]
            and len(train) == raw["selection"]["train_rows"]
            and len(dev) == raw["selection"]["dev_rows"]
        ),
        "strata_count_pass": dict(counts) == expected_counts,
        "sample_id_unique": len(sample_ids) == len(rows),
        "exact_hash_unique": len(exact_hashes) == len(rows),
        "semantic_hash_unique": len(semantic_hashes) == len(rows),
        "source_index_unique": len(source_hashes) == len(rows),
        "token_recomputation_pass": token_recomputation,
        "train_token_target_pass": (
            train_tokens
            >= raw["token_accounting"]["minimum_train_tokens"]
        ),
        "verifier_pass": verifier_pass,
        "training_boundary_pass": training_boundary_pass,
        "forbidden_exact_overlap_zero": (
            overlap_counts["forbidden_exact"]
            == raw["quality"]["forbidden_exact_overlap_allowed"]
        ),
        "forbidden_near_overlap_zero": (
            overlap_counts["forbidden_near"]
            == raw["quality"]["forbidden_near_overlap_allowed"]
        ),
        "selected_near_overlap_zero": (
            overlap_counts["selected_near"] == 0
        ),
        "tokenizer_identity_pass": (
            preregister["identity"]["tokenizer_file_sha256"]
            == raw["token_accounting"]["tokenizer_files"]
        ),
        "source_identity_pass": (
            preregister["identity"]["source_parquet_sha256"]
            == raw["source"]["parquet_sha256"]
        ),
        "forbidden_identity_pass": (
            forbidden_counts == build_audit["forbidden_corpus_rows"]
        ),
        "build_preregister_identity_pass": (
            build_audit["preregister_sha256"]
            == sha256_text(
                json.dumps(
                    preregister,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        ),
    }
    release = {
        "schema_version": RELEASE_SCHEMA,
        "release_id": raw["dataset_id"],
        "source": {
            "repository": raw["source"]["repository"],
            "revision": raw["source"]["dataset_revision"],
            "license": raw["source"]["license"],
            "source_rows": raw["source"]["parquet_rows"],
            "source_parquet_sha256": raw["source"]["parquet_sha256"],
            "config_sha256": sha256_file(config.path),
            "preregister_sha256": sha256_text(
                json.dumps(
                    preregister,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "dataset_file_sha256": sha256_file(dataset_path),
        },
        "accepted": {
            "rows": len(rows),
            "train_rows": len(train),
            "dev_rows": len(dev),
            "train_tokens": train_tokens,
            "dev_tokens": sum(
                int(row.get("token_count", 0)) for row in dev
            ),
            "token_quantiles": _quantiles(
                [int(row.get("token_count", 0)) for row in rows]
            ),
            "by_split_stratum": dict(sorted(counts.items())),
        },
        "filtering": {
            "source_rejected": build_audit["rejected"],
            "selection_rejected": build_audit["selected_rejected"],
            "overlap_counts": dict(overlap_counts),
        },
        "checks": checks,
        "training_unblocked": all(checks.values()),
        "rl_or_opd_unlocked": False,
        "claim_boundary": (
            "This release proves deterministic source selection, row and "
            "token scale, provenance, split isolation, exact/near dedup, "
            "numeric suffix verification, and zero overlap with the pinned "
            "GSM8K/MMLU/GPQA corpora. It is not model-quality evidence and "
            "unlocks only one separately pre-registered SFT smoke."
        ),
    }
    return release
