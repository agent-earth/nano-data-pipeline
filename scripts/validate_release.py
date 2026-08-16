#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_data_pipeline.analog import (
    build_curriculum_analog_dataset,
    build_choice_replay_dataset,
    build_failure_targeted_preservation_mix_dataset,
    build_format_analog_dataset,
    build_packing_isolation_preservation_mix_dataset,
    build_percentage_isolation_preservation_mix_dataset,
    build_schedule_isolation_preservation_mix_dataset,
    build_preservation_mix_dataset,
    build_process_trace_dataset,
    build_semantic_trace_dataset,
    build_targeted_preservation_mix_dataset,
    validate_analog_dataset,
)
from nano_data_pipeline.feedback import (
    build_feedback_manifest,
    validate_feedback_manifest,
)
from nano_data_pipeline.choice_matrix import (
    build_choice_capability_matrix,
    validate_choice_capability_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT.parent / "nano-harness"
COMMITTED = ROOT / "manifests/qwen35_large_confirmation_feedback_v1.json"
ANALOG = ROOT / "datasets/format_contract_analog_v1.json"
CURRICULUM = ROOT / "datasets/format_contract_curriculum_analog_v2.json"
SEMANTIC = ROOT / "datasets/verified_semantic_arithmetic_traces_v3.json"
PROCESS = ROOT / "datasets/verified_arithmetic_process_traces_v4.json"
PRESERVATION = ROOT / "datasets/hard_preservation_mix_v5.json"
TARGETED = ROOT / "datasets/targeted_preservation_mix_v6.json"
FAILURE_TARGETED = ROOT / "datasets/failure_targeted_preservation_mix_v7.json"
PERCENTAGE_ISOLATION = (
    ROOT / "datasets/percentage_isolation_preservation_mix_v8.json"
)
PACKING_ISOLATION = (
    ROOT / "datasets/packing_isolation_preservation_mix_v9.json"
)
SCHEDULE_ISOLATION = (
    ROOT / "datasets/schedule_isolation_preservation_mix_v10.json"
)
CHOICE_REPLAY = ROOT / "datasets/generic_choice_replay_v11.json"
CHOICE_MATRIX = ROOT / "datasets/generic_choice_capability_matrix_v1.json"
FAILURE_FAMILIES = (
    ROOT.parent
    / "nano-harness/configs/feedback/v11_base_only_failure_families_v1.json"
)
V10_REPORT = (
    ROOT.parent
    / "nano-train/docs/results/hard_preservation_sft_smoke_v10.public.json"
)


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
    targeted = json.loads(TARGETED.read_text(encoding="utf-8"))
    validate_analog_dataset(targeted)
    expected_targeted = build_targeted_preservation_mix_dataset(
        COMMITTED,
        PRESERVATION,
        V10_REPORT,
        [ANALOG, CURRICULUM, SEMANTIC, PROCESS],
    )
    if targeted != expected_targeted:
        raise SystemExit(
            "committed targeted preservation mix is not reproducible"
        )
    failure_targeted = json.loads(
        FAILURE_TARGETED.read_text(encoding="utf-8")
    )
    validate_analog_dataset(failure_targeted)
    expected_failure_targeted = (
        build_failure_targeted_preservation_mix_dataset(
            COMMITTED,
            FAILURE_FAMILIES,
            TARGETED,
            [ANALOG, CURRICULUM, SEMANTIC, PROCESS, PRESERVATION],
        )
    )
    if failure_targeted != expected_failure_targeted:
        raise SystemExit(
            "committed failure-targeted preservation mix is not reproducible"
        )
    percentage_isolation = json.loads(
        PERCENTAGE_ISOLATION.read_text(encoding="utf-8")
    )
    validate_analog_dataset(percentage_isolation)
    expected_percentage_isolation = (
        build_percentage_isolation_preservation_mix_dataset(
            COMMITTED,
            FAILURE_FAMILIES,
            TARGETED,
            FAILURE_TARGETED,
            [ANALOG, CURRICULUM, SEMANTIC, PROCESS, PRESERVATION],
        )
    )
    if percentage_isolation != expected_percentage_isolation:
        raise SystemExit(
            "committed percentage-isolation mix is not reproducible"
        )
    packing_isolation = json.loads(
        PACKING_ISOLATION.read_text(encoding="utf-8")
    )
    validate_analog_dataset(packing_isolation)
    expected_packing_isolation = (
        build_packing_isolation_preservation_mix_dataset(
            COMMITTED,
            FAILURE_FAMILIES,
            TARGETED,
            FAILURE_TARGETED,
            [ANALOG, CURRICULUM, SEMANTIC, PROCESS, PRESERVATION],
        )
    )
    if packing_isolation != expected_packing_isolation:
        raise SystemExit("committed packing-isolation mix is not reproducible")
    schedule_isolation = json.loads(
        SCHEDULE_ISOLATION.read_text(encoding="utf-8")
    )
    validate_analog_dataset(schedule_isolation)
    expected_schedule_isolation = (
        build_schedule_isolation_preservation_mix_dataset(
            COMMITTED,
            FAILURE_FAMILIES,
            TARGETED,
            FAILURE_TARGETED,
            [ANALOG, CURRICULUM, SEMANTIC, PROCESS, PRESERVATION],
        )
    )
    if schedule_isolation != expected_schedule_isolation:
        raise SystemExit("committed schedule-isolation mix is not reproducible")
    choice_replay = json.loads(CHOICE_REPLAY.read_text(encoding="utf-8"))
    validate_analog_dataset(choice_replay)
    expected_choice_replay = build_choice_replay_dataset(TARGETED)
    if choice_replay != expected_choice_replay:
        raise SystemExit("committed choice replay is not reproducible")
    choice_matrix = json.loads(CHOICE_MATRIX.read_text(encoding="utf-8"))
    validate_choice_capability_matrix(choice_matrix)
    expected_choice_matrix = build_choice_capability_matrix(
        [
            ANALOG,
            CURRICULUM,
            SEMANTIC,
            PROCESS,
            PRESERVATION,
            TARGETED,
            FAILURE_TARGETED,
            PERCENTAGE_ISOLATION,
            PACKING_ISOLATION,
            SCHEDULE_ISOLATION,
            CHOICE_REPLAY,
        ]
    )
    if choice_matrix != expected_choice_matrix:
        raise SystemExit("committed choice matrix is not reproducible")
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
                "targeted_dataset_id": targeted["dataset_id"],
                "targeted_samples": targeted["summary"]["samples"],
                "targeted_train_samples": targeted["summary"]["by_split"][
                    "train"
                ],
                "targeted_validation_samples": targeted["summary"]["by_split"][
                    "validation"
                ],
                "targeted_replacement_count": targeted["source"][
                    "replacement_count"
                ],
                "failure_targeted_dataset_id": failure_targeted["dataset_id"],
                "failure_targeted_samples": failure_targeted["summary"][
                    "samples"
                ],
                "failure_targeted_train_samples": failure_targeted["summary"][
                    "by_split"
                ]["train"],
                "failure_targeted_validation_samples": failure_targeted[
                    "summary"
                ]["by_split"]["validation"],
                "failure_targeted_replacement_count": failure_targeted[
                    "source"
                ]["replacement_count"],
                "percentage_isolation_dataset_id": percentage_isolation[
                    "dataset_id"
                ],
                "percentage_isolation_samples": percentage_isolation[
                    "summary"
                ]["samples"],
                "percentage_isolation_train_samples": percentage_isolation[
                    "summary"
                ]["by_split"]["train"],
                "percentage_isolation_validation_samples": percentage_isolation[
                    "summary"
                ]["by_split"]["validation"],
                "percentage_isolation_replacement_count": percentage_isolation[
                    "source"
                ]["replacement_count"],
                "packing_isolation_dataset_id": packing_isolation["dataset_id"],
                "packing_isolation_samples": packing_isolation["summary"][
                    "samples"
                ],
                "packing_isolation_train_samples": packing_isolation["summary"][
                    "by_split"
                ]["train"],
                "packing_isolation_validation_samples": packing_isolation[
                    "summary"
                ]["by_split"]["validation"],
                "packing_isolation_replacement_count": packing_isolation[
                    "source"
                ]["replacement_count"],
                "schedule_isolation_dataset_id": schedule_isolation["dataset_id"],
                "schedule_isolation_samples": schedule_isolation["summary"][
                    "samples"
                ],
                "schedule_isolation_replacement_count": schedule_isolation[
                    "source"
                ]["replacement_count"],
                "choice_replay_dataset_id": choice_replay["dataset_id"],
                "choice_replay_samples": choice_replay["summary"]["samples"],
                "choice_replay_train_samples": choice_replay["summary"][
                    "by_split"
                ]["train"],
                "choice_replay_validation_samples": choice_replay["summary"][
                    "by_split"
                ]["validation"],
                "choice_matrix_id": choice_matrix["matrix_id"],
                "choice_matrix_cases": choice_matrix["summary"]["cases"],
                "choice_matrix_scored_cases": choice_matrix["summary"][
                    "scored_cases"
                ],
                "choice_matrix_ambiguity_cases": choice_matrix["summary"][
                    "ambiguity_cases"
                ],
                "feedback_byte_reproducible": True,
                "analog_byte_reproducible": True,
                "curriculum_byte_reproducible": True,
                "semantic_byte_reproducible": True,
                "process_byte_reproducible": True,
                "preservation_byte_reproducible": True,
                "targeted_byte_reproducible": True,
                "failure_targeted_byte_reproducible": True,
                "percentage_isolation_byte_reproducible": True,
                "packing_isolation_byte_reproducible": True,
                "schedule_isolation_byte_reproducible": True,
                "choice_replay_byte_reproducible": True,
                "choice_matrix_byte_reproducible": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
