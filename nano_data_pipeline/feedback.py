from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "nano_feedback_manifest_v1"
FORBIDDEN_ROW_FIELDS = {
    "answer",
    "choices",
    "expected",
    "output",
    "prediction",
    "prompt",
    "question",
    "reference",
    "source_index",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row["case_id"])
        if case_id in records:
            raise ValueError(f"duplicate case_id in {path}: {case_id}")
        records[case_id] = row
    return records


def classify_format(row: dict[str, Any]) -> tuple[str, bool | None]:
    if row.get("prediction") is not None:
        return "parseable", None
    output = str(row.get("output", "")).strip()
    expected = str(row.get("expected", "")).upper()
    missing_colon = re.fullmatch(r"FINAL\s+([A-D])", output, re.IGNORECASE)
    if missing_colon:
        return (
            "final_missing_colon",
            missing_colon.group(1).upper() == expected,
        )
    if row.get("finish_reason") == "length":
        return "length_truncation", None
    if not output:
        return "empty_output", None
    return "unparseable_other", None


def paired_outcome(
    four_b: dict[str, Any],
    nine_b: dict[str, Any],
) -> str:
    four_correct = float(four_b["score"]) == 1.0
    nine_correct = float(nine_b["score"]) == 1.0
    if four_correct and nine_correct:
        return "both_correct"
    if four_correct:
        return "four_b_only"
    if nine_correct:
        return "nine_b_only"
    return "both_wrong"


def build_feedback_manifest(
    *,
    case_manifest_path: Path,
    four_b_path: Path,
    nine_b_path: Path,
    public_report_path: Path,
    source_revision: str,
) -> dict[str, Any]:
    case_rows = json.loads(case_manifest_path.read_text(encoding="utf-8"))
    case_ids = [str(row["case_id"]) for row in case_rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("committed case manifest contains duplicate case IDs")

    four_b = load_jsonl(four_b_path)
    nine_b = load_jsonl(nine_b_path)
    expected_ids = set(case_ids)
    if set(four_b) != expected_ids or set(nine_b) != expected_ids:
        raise ValueError("raw result case IDs do not match committed manifest")

    report = json.loads(public_report_path.read_text(encoding="utf-8"))
    raw_hashes = {
        "four_b": sha256_file(four_b_path),
        "nine_b": sha256_file(nine_b_path),
    }
    report_hashes = report["artifacts"]
    if (
        raw_hashes["four_b"] != report_hashes["four_b_raw_sha256"]
        or raw_hashes["nine_b"] != report_hashes["nine_b_raw_sha256"]
    ):
        raise ValueError("raw result hashes do not match public report")

    rows = []
    for case_id in sorted(expected_ids):
        four = four_b[case_id]
        nine = nine_b[case_id]
        if (
            four["benchmark"] != nine["benchmark"]
            or four["expected"] != nine["expected"]
        ):
            raise ValueError(f"paired contract mismatch: {case_id}")
        outcome = paired_outcome(four, nine)
        four_format, four_letter_match = classify_format(four)
        nine_format, nine_letter_match = classify_format(nine)
        if (
            outcome in {"both_correct", "both_wrong"}
            and four_format == "parseable"
            and nine_format == "parseable"
        ):
            continue

        if four_format != "parseable" or nine_format != "parseable":
            failure_family = "format"
        else:
            failure_family = "semantic_discordance"
        row = {
            "case_id": case_id,
            "benchmark": str(four["benchmark"]),
            "paired_outcome": outcome,
            "failure_family": failure_family,
            "four_b": {
                "correct": float(four["score"]) == 1.0,
                "format_class": four_format,
                "format_letter_matches_reference": four_letter_match,
            },
            "nine_b": {
                "correct": float(nine["score"]) == 1.0,
                "format_class": nine_format,
                "format_letter_matches_reference": nine_letter_match,
            },
            "source_split": "sealed_eval_feedback",
            "training_eligible": False,
        }
        rows.append(row)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "qwen35-large-confirmation-feedback-v1",
        "version": "v1",
        "source": {
            "experiment_id": report["experiment_id"],
            "source_revision": source_revision,
            "case_manifest_sha256": sha256_file(case_manifest_path),
            "four_b_raw_sha256": raw_hashes["four_b"],
            "nine_b_raw_sha256": raw_hashes["nine_b"],
            "public_report_sha256": sha256_file(public_report_path),
        },
        "policy": {
            "contains_raw_outputs": False,
            "contains_prompts": False,
            "contains_references": False,
            "contains_predictions": False,
            "direct_training_allowed": False,
            "required_derived_split": "non_eval_analog_only",
        },
        "rows": rows,
    }
    manifest["summary"] = summarize_manifest(manifest)
    validate_feedback_manifest(manifest)
    return manifest


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = manifest["rows"]
    return {
        "rows": len(rows),
        "by_benchmark": dict(sorted(Counter(row["benchmark"] for row in rows).items())),
        "by_failure_family": dict(
            sorted(Counter(row["failure_family"] for row in rows).items())
        ),
        "by_paired_outcome": dict(
            sorted(Counter(row["paired_outcome"] for row in rows).items())
        ),
        "training_eligible_rows": sum(row["training_eligible"] for row in rows),
    }


def validate_feedback_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported feedback manifest schema")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    ids = [row.get("case_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("feedback manifest case IDs are not unique")
    for row in rows:
        forbidden = FORBIDDEN_ROW_FIELDS & set(row)
        if forbidden:
            raise ValueError(f"forbidden row fields: {sorted(forbidden)}")
        if row.get("source_split") != "sealed_eval_feedback":
            raise ValueError("feedback row split must remain sealed")
        if row.get("training_eligible") is not False:
            raise ValueError("sealed feedback cannot be training eligible")
        if row.get("failure_family") not in {"format", "semantic_discordance"}:
            raise ValueError("unknown failure family")
        for model_key in ("four_b", "nine_b"):
            model = row.get(model_key, {})
            if set(model) != {
                "correct",
                "format_class",
                "format_letter_matches_reference",
            }:
                raise ValueError(f"unexpected {model_key} fields")
    if manifest.get("summary") != summarize_manifest(manifest):
        raise ValueError("feedback summary does not match rows")
    policy = manifest.get("policy", {})
    if any(
        policy.get(field) is not False
        for field in (
            "contains_raw_outputs",
            "contains_prompts",
            "contains_references",
            "contains_predictions",
            "direct_training_allowed",
        )
    ):
        raise ValueError("public feedback policy boundary is not closed")
