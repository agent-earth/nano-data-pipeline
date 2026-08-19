#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_data_pipeline.execution_coverage import (
    build_execution_coverage_audit,
)


ROOT = Path(__file__).resolve().parents[1]
ACCEPTED = (
    ROOT
    / "../../../datasets/ultimate-distill/"
    "skill-sft-10k-10m-v2/accepted.jsonl"
)
RELEASE = (
    ROOT
    / "../../../datasets/ultimate-distill/"
    "skill-sft-10k-10m-v2/release.json"
)
PROCESS = ROOT / "datasets/verified_arithmetic_process_traces_v4.json"
FAILURES = ROOT / "manifests/skill_release_reasoning_failures_v1.json"
PUBLIC_JSON = (
    ROOT / "docs/datasets/skill_release_execution_target_coverage_v1.public.json"
)
REPORT = ROOT / "docs/datasets/skill_release_execution_target_coverage_v1.md"


def main() -> None:
    audit = build_execution_coverage_audit(
        accepted_jsonl_path=ACCEPTED,
        release_manifest_path=RELEASE,
        process_dataset_path=PROCESS,
        failure_manifest_path=FAILURES,
        selected_train_rows=16,
    )
    PUBLIC_JSON.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    REPORT.write_text(render_markdown(audit), encoding="utf-8")


def render_markdown(audit: dict) -> str:
    datasets = audit["datasets"]
    coverage = audit["coverage"]
    findings = audit["findings"]
    contract = audit["minimum_fresh_data_contract"]
    selected = coverage["selected_release_reasoning_train"]["covered_targets"]
    process = coverage["process_v6_train"]["covered_targets"]
    union = coverage["selected_release_plus_process_v6"]["covered_targets"]
    release_summary = datasets["selected_release_reasoning_train"]
    process_summary = datasets["process_v6_train"]
    return f"""# Skill Release Execution-Target Coverage Audit v1

## 为什么做这个审计

Reasoning-preservation SFT v4 在一组 fresh dev 上是 16/20 → 16/20，
`verified-reasoning` 是 0/4 → 0/4。此前 process-trace v6 把步骤格式从
28/32 提到 32/32，但最终答案本来就是 32/32，随后在 211-case matched
benchmark 上相对 base 4B 显著退化。

所以现在不能直接再训。先回答一个更具体的问题：现有训练数据有没有在同一条样本
里同时覆盖目标公式关系、正确中间执行和最终答案？

## 审计对象

- skill release 中当前实际选中的前 16 条 reasoning train；
- skill release 全部 {datasets['full_release_reasoning_train']['rows']:,}
  条 reasoning train；
- v6 的 {process_summary['rows']} 条 process-trace train；
- v4 fresh dev 中 4 条错误的 `verified-reasoning` case。

## 关键结果

### Release reasoning 数据

- 16/16 都是 `FINAL: <number>`，显式过程监督 0/16；
- 目标关系覆盖：{selected['relation_matches']}/4；
- 同时具备过程监督的目标关系覆盖：
  {selected['process_relation_matches']}/4；
- 每条 prompt 中位数包含
  {release_summary['median_synthetic_distractors']} 条无关 synthetic evidence；
- 目标答案泄露到 user prompt：
  {release_summary['target_value_in_user_rows']} 条。

它覆盖了公式关系，但只监督最终 token，不监督怎么得到最终值。

### v6 process-trace 数据

- 显式过程监督：{process_summary['explicit_process_rows']}/
  {process_summary['rows']}；
- AST 形状覆盖：{process['process_shape_matches']}/4；
- 相同“加数在最后再次被减掉”关系覆盖：
  {process['process_relation_matches']}/4；
- v4 四个目标结果的 exact-result 覆盖：
  {process['exact_result_matches']}/4。

它监督了步骤，但没有覆盖当前错误所需的重复操作数关系。

### 合并后

- shape 覆盖：{union['shape_matches']}/4；
- relation 覆盖：{union['relation_matches']}/4；
- process relation 覆盖：{union['process_relation_matches']}/4；
- 同一条样本同时满足“目标 relation + process”的行数：
  {coverage['joint_process_relation_rows']}。

两套数据各覆盖一半，合起来仍然没有一条训练样本同时提供完整机制监督。

## 最小 Fresh Data Contract

下一步只允许生成数据，暂不允许训练：

- {contract['train_rows']} train / {contract['dev_rows']} dev；
- 至少 {contract['minimum_train_tokens']:,} train tokens；
- 128 个 fresh semantic task，覆盖 8×8 operand grid 和 multiplier 2/3；
- 每个 semantic task 同时生成 verified process view 和 final-only view；
- 256 条 JSON preservation replay，每个 JSON family 64 条；
- dev 中 24 个 fresh relation task、48 个 paired execution view、32 条
  JSON non-regression；
- train/dev 与旧 release semantic overlap 必须是 0；
- 所有 process view 必须逐步执行验证；
- 禁止 benchmark、independent holdout、模型输出和答案泄露进入数据。

数据合同全部通过之前，SFT、benchmark、holdout 和 RL 都保持关闭。

## 决策

- more_sft_allowed_now：false；
- generate_contract_dataset_next：true；
- reuse_answer_only_oversampling：false；
- reuse_process_v6_unchanged：false；
- benchmark / independent holdout / RL：全部关闭。

## Evidence

- accepted JSONL SHA256:
  `{audit['sources']['accepted_jsonl_sha256']}`;
- process v6 SHA256:
  `{audit['sources']['process_dataset_sha256']}`;
- failure manifest SHA256:
  `{audit['sources']['failure_manifest_sha256']}`;
- failure source commit: `{audit['sources']['failure_source_commit']}`.

## 结论边界

{audit['claim_boundary']}
"""


if __name__ == "__main__":
    main()
