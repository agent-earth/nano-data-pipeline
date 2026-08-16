#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_data_pipeline.feedback import (
    build_feedback_manifest,
    validate_feedback_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT.parent / "nano-harness"
COMMITTED = ROOT / "manifests/qwen35_large_confirmation_feedback_v1.json"


def main() -> None:
    actual = json.loads(COMMITTED.read_text(encoding="utf-8"))
    validate_feedback_manifest(actual)
    expected = build_feedback_manifest(
        case_manifest_path=(
            HARNESS
            / "configs/generated/qwen35_large_confirmation_v1_cases.json"
        ),
        four_b_path=(
            HARNESS
            / "results/harness/qwen35-large-confirmation-v1/4b/cases.jsonl"
        ),
        nine_b_path=(
            HARNESS
            / "results/harness/qwen35-large-confirmation-v1/9b/cases.jsonl"
        ),
        public_report_path=(
            HARNESS / "docs/results/large_confirmation_v1.public.json"
        ),
        source_revision="47ee48f",
    )
    if actual != expected:
        raise SystemExit("committed feedback manifest is not reproducible")
    print(
        json.dumps(
            {
                "ok": True,
                "dataset_id": actual["dataset_id"],
                "rows": actual["summary"]["rows"],
                "training_eligible_rows": actual["summary"][
                    "training_eligible_rows"
                ],
                "byte_reproducible": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
