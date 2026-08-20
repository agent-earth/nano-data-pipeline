#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from nano_data_pipeline.orca_math_preference import (
    build_dataset,
    load_config,
    validate_dataset,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/orca_math_preference_v1.json"


def main() -> None:
    config = load_config(CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(
        config.resolve(config.raw["tokenizer"]["path"]),
        local_files_only=True,
    )
    rows, audit = build_dataset(config, tokenizer=tokenizer)
    dataset_path = config.resolve(config.raw["output"]["dataset_path"])
    write_jsonl(dataset_path, rows)
    release = validate_dataset(
        config,
        tokenizer=tokenizer,
        path=dataset_path,
    )
    if release["training_unblocked"] is not True:
        raise ValueError("Orca preference release failed")
    local_release = config.resolve(
        config.raw["output"]["local_release_path"]
    )
    local_release.parent.mkdir(parents=True, exist_ok=True)
    local_release.write_text(
        json.dumps(
            {"release": release, "build_audit": audit},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    public_release = config.resolve(
        config.raw["output"]["public_release_path"]
    )
    public_release.write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(release, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
