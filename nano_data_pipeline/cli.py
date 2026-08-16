from __future__ import annotations

import argparse
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
from nano_data_pipeline.choice_matrix_v2 import (
    build_choice_verifier_matrix_v2,
    validate_choice_verifier_matrix_v2,
)
from nano_data_pipeline.choice_matrix_v3 import (
    build_choice_exact_replication_matrix_v3,
    validate_choice_exact_replication_matrix_v3,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="nano-data-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-feedback")
    build.add_argument("--case-manifest", required=True)
    build.add_argument("--four-b-results", required=True)
    build.add_argument("--nine-b-results", required=True)
    build.add_argument("--public-report", required=True)
    build.add_argument("--source-revision", required=True)
    build.add_argument("--output", required=True)

    validate = subparsers.add_parser("validate-feedback")
    validate.add_argument("path")

    analog = subparsers.add_parser("build-format-analog")
    analog.add_argument("--feedback-manifest", required=True)
    analog.add_argument("--output", required=True)

    validate_analog = subparsers.add_parser("validate-analog")
    validate_analog.add_argument("path")

    curriculum = subparsers.add_parser("build-curriculum-analog")
    curriculum.add_argument("--feedback-manifest", required=True)
    curriculum.add_argument("--prior-dataset", required=True)
    curriculum.add_argument("--output", required=True)

    semantic = subparsers.add_parser("build-semantic-traces")
    semantic.add_argument("--feedback-manifest", required=True)
    semantic.add_argument("--prior-dataset", action="append", required=True)
    semantic.add_argument("--output", required=True)

    process = subparsers.add_parser("build-process-traces")
    process.add_argument("--feedback-manifest", required=True)
    process.add_argument("--prior-dataset", action="append", required=True)
    process.add_argument("--output", required=True)

    preservation = subparsers.add_parser("build-preservation-mix")
    preservation.add_argument("--feedback-manifest", required=True)
    preservation.add_argument(
        "--prior-dataset",
        action="append",
        required=True,
    )
    preservation.add_argument("--output", required=True)

    targeted = subparsers.add_parser("build-targeted-preservation-mix")
    targeted.add_argument("--feedback-manifest", required=True)
    targeted.add_argument("--base-dataset", required=True)
    targeted.add_argument("--development-report", required=True)
    targeted.add_argument(
        "--prior-dataset",
        action="append",
        required=True,
    )
    targeted.add_argument("--output", required=True)

    failure_targeted = subparsers.add_parser(
        "build-failure-targeted-preservation-mix"
    )
    failure_targeted.add_argument("--feedback-manifest", required=True)
    failure_targeted.add_argument("--failure-family-receipt", required=True)
    failure_targeted.add_argument("--base-dataset", required=True)
    failure_targeted.add_argument(
        "--prior-dataset",
        action="append",
        required=True,
    )
    failure_targeted.add_argument("--output", required=True)

    percentage_isolation = subparsers.add_parser(
        "build-percentage-isolation-preservation-mix"
    )
    percentage_isolation.add_argument("--feedback-manifest", required=True)
    percentage_isolation.add_argument(
        "--failure-family-receipt",
        required=True,
    )
    percentage_isolation.add_argument("--base-dataset", required=True)
    percentage_isolation.add_argument("--broad-dataset", required=True)
    percentage_isolation.add_argument(
        "--prior-dataset",
        action="append",
        required=True,
    )
    percentage_isolation.add_argument("--output", required=True)

    packing_isolation = subparsers.add_parser(
        "build-packing-isolation-preservation-mix"
    )
    packing_isolation.add_argument("--feedback-manifest", required=True)
    packing_isolation.add_argument("--failure-family-receipt", required=True)
    packing_isolation.add_argument("--base-dataset", required=True)
    packing_isolation.add_argument("--broad-dataset", required=True)
    packing_isolation.add_argument(
        "--prior-dataset",
        action="append",
        required=True,
    )
    packing_isolation.add_argument("--output", required=True)

    schedule_isolation = subparsers.add_parser(
        "build-schedule-isolation-preservation-mix"
    )
    schedule_isolation.add_argument("--feedback-manifest", required=True)
    schedule_isolation.add_argument("--failure-family-receipt", required=True)
    schedule_isolation.add_argument("--base-dataset", required=True)
    schedule_isolation.add_argument("--broad-dataset", required=True)
    schedule_isolation.add_argument(
        "--prior-dataset",
        action="append",
        required=True,
    )
    schedule_isolation.add_argument("--output", required=True)

    choice_replay = subparsers.add_parser("build-choice-replay")
    choice_replay.add_argument("--base-dataset", required=True)
    choice_replay.add_argument("--output", required=True)

    choice_matrix = subparsers.add_parser("build-choice-capability-matrix")
    choice_matrix.add_argument("--prior-dataset", action="append", required=True)
    choice_matrix.add_argument("--output", required=True)

    validate_choice_matrix = subparsers.add_parser(
        "validate-choice-capability-matrix"
    )
    validate_choice_matrix.add_argument("path")

    choice_matrix_v2 = subparsers.add_parser("build-choice-verifier-matrix-v2")
    choice_matrix_v2.add_argument("--prior-dataset", action="append", required=True)
    choice_matrix_v2.add_argument("--prior-matrix", action="append", required=True)
    choice_matrix_v2.add_argument("--output", required=True)

    validate_choice_matrix_v2 = subparsers.add_parser(
        "validate-choice-verifier-matrix-v2"
    )
    validate_choice_matrix_v2.add_argument("path")

    choice_matrix_v3 = subparsers.add_parser(
        "build-choice-exact-replication-matrix-v3"
    )
    choice_matrix_v3.add_argument("--prior-dataset", action="append", required=True)
    choice_matrix_v3.add_argument("--prior-matrix", action="append", required=True)
    choice_matrix_v3.add_argument("--output", required=True)

    validate_choice_matrix_v3 = subparsers.add_parser(
        "validate-choice-exact-replication-matrix-v3"
    )
    validate_choice_matrix_v3.add_argument("path")

    args = parser.parse_args()
    if args.command == "build-feedback":
        manifest = build_feedback_manifest(
            case_manifest_path=Path(args.case_manifest),
            four_b_path=Path(args.four_b_results),
            nine_b_path=Path(args.nine_b_results),
            public_report_path=Path(args.public_report),
            source_revision=args.source_revision,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "validate-feedback":
        manifest = json.loads(Path(args.path).read_text(encoding="utf-8"))
        validate_feedback_manifest(manifest)
    elif args.command == "build-format-analog":
        manifest = build_format_analog_dataset(Path(args.feedback_manifest))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "build-curriculum-analog":
        manifest = build_curriculum_analog_dataset(
            Path(args.feedback_manifest),
            Path(args.prior_dataset),
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "build-semantic-traces":
        manifest = build_semantic_trace_dataset(
            Path(args.feedback_manifest),
            [Path(path) for path in args.prior_dataset],
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "build-process-traces":
        manifest = build_process_trace_dataset(
            Path(args.feedback_manifest),
            [Path(path) for path in args.prior_dataset],
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "build-preservation-mix":
        manifest = build_preservation_mix_dataset(
            Path(args.feedback_manifest),
            [Path(path) for path in args.prior_dataset],
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "build-targeted-preservation-mix":
        manifest = build_targeted_preservation_mix_dataset(
            Path(args.feedback_manifest),
            Path(args.base_dataset),
            Path(args.development_report),
            [Path(path) for path in args.prior_dataset],
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "build-failure-targeted-preservation-mix":
        manifest = build_failure_targeted_preservation_mix_dataset(
            Path(args.feedback_manifest),
            Path(args.failure_family_receipt),
            Path(args.base_dataset),
            [Path(path) for path in args.prior_dataset],
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "build-percentage-isolation-preservation-mix":
        manifest = build_percentage_isolation_preservation_mix_dataset(
            Path(args.feedback_manifest),
            Path(args.failure_family_receipt),
            Path(args.base_dataset),
            Path(args.broad_dataset),
            [Path(path) for path in args.prior_dataset],
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "build-packing-isolation-preservation-mix":
        manifest = build_packing_isolation_preservation_mix_dataset(
            Path(args.feedback_manifest),
            Path(args.failure_family_receipt),
            Path(args.base_dataset),
            Path(args.broad_dataset),
            [Path(path) for path in args.prior_dataset],
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "build-schedule-isolation-preservation-mix":
        manifest = build_schedule_isolation_preservation_mix_dataset(
            Path(args.feedback_manifest),
            Path(args.failure_family_receipt),
            Path(args.base_dataset),
            Path(args.broad_dataset),
            [Path(path) for path in args.prior_dataset],
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "build-choice-replay":
        manifest = build_choice_replay_dataset(Path(args.base_dataset))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "build-choice-capability-matrix":
        manifest = build_choice_capability_matrix(
            [Path(path) for path in args.prior_dataset]
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "validate-choice-capability-matrix":
        manifest = json.loads(Path(args.path).read_text(encoding="utf-8"))
        validate_choice_capability_matrix(manifest)
    elif args.command == "build-choice-verifier-matrix-v2":
        manifest = build_choice_verifier_matrix_v2(
            [Path(path) for path in args.prior_dataset],
            [Path(path) for path in args.prior_matrix],
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "validate-choice-verifier-matrix-v2":
        manifest = json.loads(Path(args.path).read_text(encoding="utf-8"))
        validate_choice_verifier_matrix_v2(manifest)
    elif args.command == "build-choice-exact-replication-matrix-v3":
        manifest = build_choice_exact_replication_matrix_v3(
            [Path(path) for path in args.prior_dataset],
            [Path(path) for path in args.prior_matrix],
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "validate-choice-exact-replication-matrix-v3":
        manifest = json.loads(Path(args.path).read_text(encoding="utf-8"))
        validate_choice_exact_replication_matrix_v3(manifest)
    else:
        manifest = json.loads(Path(args.path).read_text(encoding="utf-8"))
        validate_analog_dataset(manifest)
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
