#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_data_pipeline.orca_math_preference import build_preregister, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/orca_math_preference_v1.json"
OUTPUT = (
    ROOT / "docs/experiments/orca_math_preference_v1.preregister.json"
)


def main() -> None:
    receipt = build_preregister(load_config(CONFIG))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
