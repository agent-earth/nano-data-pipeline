#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from nano_data_pipeline.router_classification import (
    build_dataset,
    load_config,
    validate_router_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/router_classification/qwen35_router_classification_v1.json"
)


def main() -> None:
    config = load_config(CONFIG)
    tokenizer_path = (ROOT / config.tokenizer_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    dataset = build_dataset(config)
    release = validate_router_dataset(
        dataset,
        config=config,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
    )
    if not release["training_unblocked"]:
        raise ValueError("router release did not pass all checks")
    dataset_path = ROOT / config.output_dataset_path
    release_path = ROOT / config.output_release_path
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    release_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps(dataset, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reloaded = json.loads(dataset_path.read_text(encoding="utf-8"))
    revalidated = validate_router_dataset(
        reloaded,
        config=config,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
    )
    if revalidated != release:
        raise ValueError("router release differs after reload")
    release_path.write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(release, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
