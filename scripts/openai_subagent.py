#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_data_pipeline.openai_subagent import (
    FamilyCompiler,
    OpenAICompatibleSubagent,
    SubagentConfig,
    criticize_candidates,
    generate_candidates,
)
from nano_data_pipeline.subagent_campaign import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "critic"):
        child = subparsers.add_parser(command)
        child.add_argument("--input", required=True)
        child.add_argument("--output", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.input).read_text(encoding="utf-8"))
    config = SubagentConfig.from_env(args.command)
    subagent = OpenAICompatibleSubagent(config)
    if args.command == "generate":
        if config.tokenizer_path is None:
            raise SystemExit("generator tokenizer path is missing")
        compiler = FamilyCompiler(config.tokenizer_path)
        rows = generate_candidates(request, subagent, compiler)
    else:
        rows = criticize_candidates(request, subagent)
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
