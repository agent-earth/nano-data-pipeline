from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nano_data_pipeline.feedback import sha256_file


SCORECARD_SCHEMA = "nano_skill_scorecard_v1"


def load_skill_scorecard(path: str | Path) -> dict[str, Any]:
    scorecard = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_skill_scorecard(scorecard)
    return scorecard


def validate_skill_scorecard(scorecard: dict[str, Any]) -> None:
    if scorecard.get("schema_version") != SCORECARD_SCHEMA:
        raise ValueError("unsupported skill scorecard schema")
    if not scorecard.get("skill_id"):
        raise ValueError("skill scorecard needs a skill_id")
    digest = str(scorecard.get("skill_sha256", ""))
    if len(digest) != 64:
        raise ValueError("skill scorecard needs a SHA256 skill identity")
    case_digest = str(scorecard.get("case_ids_sha256", ""))
    if len(case_digest) != 64:
        raise ValueError("skill scorecard needs a case identity")
    aggregate = scorecard.get("aggregate_score")
    if not isinstance(aggregate, (int, float)) or isinstance(aggregate, bool):
        raise ValueError("aggregate_score must be numeric")
    if not 0 <= float(aggregate) <= 1:
        raise ValueError("aggregate_score must be in [0, 1]")
    family_scores = scorecard.get("family_scores")
    if not isinstance(family_scores, dict) or not family_scores:
        raise ValueError("skill scorecard needs family_scores")
    for family, score in family_scores.items():
        if not family:
            raise ValueError("family score names must not be empty")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("family scores must be numeric")
        if not 0 <= float(score) <= 1:
            raise ValueError("family scores must be in [0, 1]")


def score_skill_file(
    skill_path: str | Path,
    *,
    skill_id: str,
    case_ids_sha256: str,
    aggregate_score: float,
    family_scores: dict[str, float],
) -> dict[str, Any]:
    scorecard = {
        "schema_version": SCORECARD_SCHEMA,
        "skill_id": skill_id,
        "skill_sha256": sha256_file(Path(skill_path)),
        "case_ids_sha256": case_ids_sha256,
        "aggregate_score": aggregate_score,
        "family_scores": dict(sorted(family_scores.items())),
    }
    validate_skill_scorecard(scorecard)
    return scorecard


def select_skill_candidate(
    parent: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    protected_families: list[str],
) -> dict[str, Any]:
    validate_skill_scorecard(parent)
    eligible: list[dict[str, Any]] = []
    rejection_reasons: dict[str, list[str]] = {}
    for candidate in candidates:
        validate_skill_scorecard(candidate)
        reasons = []
        if candidate["case_ids_sha256"] != parent["case_ids_sha256"]:
            reasons.append("case_identity_mismatch")
        if candidate["aggregate_score"] <= parent["aggregate_score"]:
            reasons.append("aggregate_not_improved")
        for family in protected_families:
            if family not in parent["family_scores"]:
                raise ValueError(f"parent scorecard lacks protected family {family}")
            if candidate["family_scores"].get(family, -1) < parent[
                "family_scores"
            ][family]:
                reasons.append(f"protected_family_regression:{family}")
        if reasons:
            rejection_reasons[candidate["skill_id"]] = reasons
        else:
            eligible.append(candidate)

    selected = parent
    promoted = False
    if eligible:
        selected = sorted(
            eligible,
            key=lambda row: (-row["aggregate_score"], row["skill_id"]),
        )[0]
        promoted = True
    return {
        "schema_version": "nano_skill_promotion_receipt_v1",
        "parent_skill_id": parent["skill_id"],
        "parent_skill_sha256": parent["skill_sha256"],
        "selected_skill_id": selected["skill_id"],
        "selected_skill_sha256": selected["skill_sha256"],
        "promoted": promoted,
        "protected_families": sorted(protected_families),
        "eligible_candidate_ids": sorted(
            candidate["skill_id"] for candidate in eligible
        ),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }
