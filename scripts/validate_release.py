#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_data_pipeline.analog import (
    build_curriculum_analog_dataset,
    build_format_analog_dataset,
    build_preservation_mix_dataset,
    build_process_trace_dataset,
    build_semantic_trace_dataset,
    validate_analog_dataset,
)
from nano_data_pipeline.feedback import (
    build_feedback_manifest,
    validate_feedback_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT.parent / "nano-harness"
COMMITTED = ROOT / "manifests/qwen35_large_confirmation_feedback_v1.json"
ANALOG = ROOT / "datasets/format_contract_analog_v1.json"
CURRICULUM = ROOT / "datasets/format_contract_curriculum_analog_v2.json"
SEMANTIC = ROOT / "datasets/verified_semantic_arithmetic_traces_v3.json"
PROCESS = ROOT / "datasets/verified_arithmetic_process_traces_v4.json"
PRESERVATION = ROOT / "datasets/hard_preservation_mix_v5.json"


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
        source_revision="3545480",
    )
    if actual != expected:
        raise SystemExit("committed feedback manifest is not reproducible")
    analog = json.loads(ANALOG.read_text(encoding="utf-8"))
    validate_analog_dataset(analog)
    expected_analog = build_format_analog_dataset(COMMITTED)
    if analog != expected_analog:
        raise SystemExit("committed analog dataset is not reproducible")
    curriculum = json.loads(CURRICULUM.read_text(encoding="utf-8"))
    validate_analog_dataset(curriculum)
    expected_curriculum = build_curriculum_analog_dataset(
        COMMITTED,
        ANALOG,
    )
    if curriculum != expected_curriculum:
        raise SystemExit("committed curriculum dataset is not reproducible")
    semantic = json.loads(SEMANTIC.read_text(encoding="utf-8"))
    validate_analog_dataset(semantic)
    expected_semantic = build_semantic_trace_dataset(
        COMMITTED,
        [ANALOG, CURRICULUM],
    )
    if semantic != expected_semantic:
        raise SystemExit("committed semantic trace dataset is not reproducible")
    process = json.loads(PROCESS.read_text(encoding="utf-8"))
    validate_analog_dataset(process)
    expected_process = build_process_trace_dataset(
        COMMITTED,
        [ANALOG, CURRICULUM, SEMANTIC],
    )
    if process != expected_process:
        raise SystemExit("committed process trace dataset is not reproducible")
    preservation = json.loads(PRESERVATION.read_text(encoding="utf-8"))
    validate_analog_dataset(preservation)
    expected_preservation = build_preservation_mix_dataset(
        COMMITTED,
        [ANALOG, CURRICULUM, SEMANTIC, PROCESS],
    )
    if preservation != expected_preservation:
        raise SystemExit("committed preservation mix is not reproducible")
    print(
        json.dumps(
            {
                "ok": True,
                "dataset_id": actual["dataset_id"],
                "rows": actual["summary"]["rows"],
                "training_eligible_rows": actual["summary"][
                    "training_eligible_rows"
                ],
                "analog_dataset_id": analog["dataset_id"],
                "analog_samples": analog["summary"]["samples"],
                "analog_train_samples": analog["summary"]["by_split"]["train"],
                "analog_validation_samples": analog["summary"]["by_split"][
                    "validation"
                ],
                "curriculum_dataset_id": curriculum["dataset_id"],
                "curriculum_samples": curriculum["summary"]["samples"],
                "curriculum_train_samples": curriculum["summary"]["by_split"][
                    "train"
                ],
                "curriculum_validation_samples": curriculum["summary"][
                    "by_split"
                ]["validation"],
                "semantic_dataset_id": semantic["dataset_id"],
                "semantic_samples": semantic["summary"]["samples"],
                "semantic_train_samples": semantic["summary"]["by_split"]["train"],
                "semantic_validation_samples": semantic["summary"]["by_split"][
                    "validation"
                ],
                "process_dataset_id": process["dataset_id"],
                "process_samples": process["summary"]["samples"],
                "process_train_samples": process["summary"]["by_split"]["train"],
                "process_validation_samples": process["summary"]["by_split"][
                    "validation"
                ],
                "preservation_dataset_id": preservation["dataset_id"],
                "preservation_samples": preservation["summary"]["samples"],
                "preservation_train_samples": preservation["summary"][
                    "by_split"
                ]["train"],
                "preservation_validation_samples": preservation["summary"][
                    "by_split"
                ]["validation"],
                "feedback_byte_reproducible": True,
                "analog_byte_reproducible": True,
                "curriculum_byte_reproducible": True,
                "semantic_byte_reproducible": True,
                "process_byte_reproducible": True,
                "preservation_byte_reproducible": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
