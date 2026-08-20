#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from nano_data_pipeline.feedback import sha256_file
from nano_data_pipeline.router_classification import (
    SYSTEM_PROMPT,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/router_classification/qwen35_router_classification_v1.json"
)
MODULE = ROOT / "nano_data_pipeline/router_classification.py"
PREREG = (
    ROOT
    / "docs/experiments/qwen35_router_classification_v1.preregister.json"
)
MARKDOWN = ROOT / "docs/experiments/qwen35_router_classification_v1.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    tokenizer = (ROOT / config.tokenizer_path).resolve()
    tokenizer_files = {
        filename: sha256_file(tokenizer / filename)
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        )
    }
    train_rows = config.train_rows_per_label * 3
    dev_rows = config.dev_rows_per_label * 3
    return {
        "schema_version": "nano_router_classification_preregister_v1",
        "dataset_id": config.dataset_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "generator_source_sha256": sha256_file(MODULE),
            "multiclass_report_sha256": config.multiclass_report_sha256,
            "binary_detector_report_sha256": (
                config.binary_detector_report_sha256
            ),
            "system_prompt_sha256": hashlib.sha256(
                SYSTEM_PROMPT.encode()
            ).hexdigest(),
            "tokenizer_file_sha256": tokenizer_files,
        },
        "frozen_contract": {
            "train_rows": train_rows,
            "dev_rows": dev_rows,
            "rows": train_rows + dev_rows,
            "train_rows_per_label": config.train_rows_per_label,
            "dev_rows_per_label": config.dev_rows_per_label,
            "label_contract": config.label_contract,
            "negative_subtypes": config.negative_subtypes,
            "minimum_train_tokens": config.minimum_train_tokens,
            "train_template_ids": [0, 1, 2, 3],
            "dev_template_ids": [4, 5, 6, 7],
            "train_dev_template_overlap": 0,
            "train_dev_value_regime_overlap": 0,
        },
        "required_checks": [
            "row_count_pass",
            "train_label_balance_pass",
            "dev_label_balance_pass",
            "train_none_subtype_balance_pass",
            "dev_none_subtype_balance_pass",
            "train_dev_semantic_overlap_pass",
            "train_dev_template_overlap_pass",
            "forbidden_content_pass",
            "model_output_absence_pass",
            "teacher_output_absence_pass",
            "benchmark_content_absence_pass",
            "token_accounting_pass",
            "tokenizer_identity_pass",
        ],
        "decision_policy": {
            "training_unblocked_only_if_all_checks_pass": True,
            "allowed_after_release": (
                "One separately pre-registered bounded Qwen3.5-4B router SFT "
                "smoke using the exact release identity."
            ),
            "forbidden_after_generation": [
                "label_contract_change",
                "class_quota_change",
                "negative_subtype_change",
                "template_change",
                "split_change",
                "minimum_token_change",
                "forbidden_term_change",
                "sample_regeneration",
                "benchmark_content_use",
                "canary_content_use",
                "model_output_use",
            ],
        },
        "execution_boundary": {
            "data_generation_started": False,
            "dataset_file_exists": False,
            "release_file_exists": False,
            "training_started": False,
            "benchmark_accessed": False,
            "canary_accessed": False,
            "holdout_accessed": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "This pre-registration freezes a deterministic synthetic router "
            "classification data contract. It does not generate data, unblock "
            "training, or establish model or benchmark quality."
        ),
    }


def render_markdown(receipt: dict) -> str:
    frozen = receipt["frozen_contract"]
    return f"""# Qwen3.5 Router Classification Data v1

## 目标

Inference prompt 路线已经负向关闭。本合同冻结 synthetic router SFT 数据，
不使用 benchmark/canary/holdout 或模型输出。

## 规模

- Train：{frozen['train_rows']} rows，A/B/C 各
  {frozen['train_rows_per_label']}；
- Dev：{frozen['dev_rows']} rows，A/B/C 各
  {frozen['dev_rows_per_label']}；
- Total：{frozen['rows']}；
- Train tokens：至少 {frozen['minimum_train_tokens']:,} Qwen3.5 tokens；
- NONE 类覆盖4种 unsupported subtype。

## Label

- `FINAL: A` → implicit_scale_total；
- `FINAL: B` → first_strict_profit_period；
- `FINAL: C` → NONE。

Train/dev 模板与数值区间都分离，semantic overlap 必须为0。

## Gate

{chr(10).join(f"- `{name}`" for name in receipt['required_checks'])}

只有全部通过才允许另行预注册一次 bounded router SFT smoke。

## Boundary

- config SHA：`{receipt['identity']['config_sha256']}`；
- generator SHA：`{receipt['identity']['generator_source_sha256']}`；
- data generation started：false；
- dataset/release file exists：false；
- training started：false；
- benchmark/canary/holdout accessed：false。
"""


def main() -> None:
    receipt = build_receipt()
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(receipt), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
