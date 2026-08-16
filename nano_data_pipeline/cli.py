from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_data_pipeline.analog import (
    build_curriculum_analog_dataset,
    build_format_analog_dataset,
    validate_analog_dataset,
)
from nano_data_pipeline.feedback import (
    build_feedback_manifest,
    validate_feedback_manifest,
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
    else:
        manifest = json.loads(Path(args.path).read_text(encoding="utf-8"))
        validate_analog_dataset(manifest)
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
