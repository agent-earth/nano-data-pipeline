from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from nano_data_pipeline.analog import evaluate_arithmetic, format_number
from nano_data_pipeline.campaign import load_skill_sft_campaign
from nano_data_pipeline.openai_subagent import (
    FamilyCompiler,
    SubagentConfig,
    criticize_candidates,
    generate_candidates,
    parse_json_object,
    sha256_text,
)
from nano_data_pipeline.subagent_campaign import accept_candidates


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "manifests/skill_sft_campaign_v1.json"
SKILL = ROOT / "skills/skill-sft-campaign/SKILL.md"


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        words = sum(len(message["content"].split()) + 4 for message in messages)
        return {"input_ids": list(range(words))}


class SolvingSubagent:
    def __init__(self, role: str):
        self.role = role
        self.config = type("Config", (), {"max_tokens": 512})()
        self.calls = 0

    def complete_json(self, messages, *, max_tokens=None):
        del max_tokens
        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        if self.role == "generator":
            response = {"assistant": solve(payload["family_id"], payload["task_spec"])}
        else:
            response = {
                "accept": True,
                "score": 0.95,
                "reasons": [],
            }
        return response, {
            "request_id": f"{self.role}-{self.calls}",
            "model": f"fake-{self.role}",
            "usage": {"total_tokens": 10},
            "finish_reason": "stop",
        }


class OpenAISubagentTests(unittest.TestCase):
    def test_parses_fenced_json(self):
        value = parse_json_object("```json\n{\"accept\": true}\n```")

        self.assertEqual(value, {"accept": True})

    def test_role_config_prefers_role_specific_environment(self):
        environment = {
            "NANO_SUBAGENT_BASE_URL": "http://shared/v1",
            "NANO_SUBAGENT_MODEL": "shared",
            "NANO_GENERATOR_BASE_URL": "http://generator/v1",
            "NANO_GENERATOR_MODEL": "generator",
            "NANO_GENERATOR_TOKENIZER": str(ROOT),
        }
        with patch.dict(os.environ, environment, clear=True):
            config = SubagentConfig.from_env("generator")

        self.assertEqual(config.base_url, "http://generator/v1")
        self.assertEqual(config.model, "generator")
        self.assertEqual(config.api_key, "local-not-required")

    def test_generate_cli_alias_uses_generator_environment(self):
        environment = {
            "NANO_GENERATOR_BASE_URL": "http://generator/v1",
            "NANO_GENERATOR_MODEL": "generator",
            "NANO_GENERATOR_TOKENIZER": str(ROOT),
        }
        with patch.dict(os.environ, environment, clear=True):
            config = SubagentConfig.from_env("generate")

        self.assertEqual(config.role, "generator")
        self.assertEqual(config.base_url, "http://generator/v1")

    def test_compiler_reaches_target_without_forbidden_markers(self):
        compiler = FamilyCompiler.__new__(FamilyCompiler)
        compiler.tokenizer = FakeTokenizer()

        compiled = compiler.compile(
            family_id="verified-reasoning",
            seed=7,
            index=0,
            target_tokens=300,
        )

        token_count = compiler._token_count(compiled["messages"])
        self.assertGreaterEqual(token_count, 300)
        self.assertNotIn("benchmark", json.dumps(compiled).lower())
        self.assertEqual(
            compiled["verifier"]["kind"],
            "safe_execution_receipt_v1",
        )

    def test_real_adapter_shapes_pass_all_local_family_verifiers(self):
        campaign = load_skill_sft_campaign(CAMPAIGN)
        skill_sha256 = sha256_text(SKILL.read_text(encoding="utf-8"))
        compiler = FamilyCompiler.__new__(FamilyCompiler)
        compiler.tokenizer = FakeTokenizer()
        family_ids = {
            family["family_id"] for family in campaign["data_families"]
        }
        expected_verifiers = {
            family["family_id"]: family["deterministic_verifier"]
            for family in campaign["data_families"]
        }

        for shard_id, family_id in enumerate(sorted(family_ids)):
            request = {
                "campaign_id": campaign["campaign_id"],
                "family_id": family_id,
                "shard_id": shard_id,
                "attempt": 0,
                "candidate_samples": 1,
                "candidate_tokens_min": 300,
                "seed": 100 + shard_id,
                "skill_sha256": skill_sha256,
            }
            generator = SolvingSubagent("generator")
            critic = SolvingSubagent("critic")
            candidates = generate_candidates(request, generator, compiler)
            decisions = criticize_candidates(
                {"candidates": candidates},
                critic,
            )
            accepted, rejected = accept_candidates(
                campaign,
                candidates,
                decisions,
                tokenizer=FakeTokenizer(),
                skill_sha256=skill_sha256,
                family_ids=family_ids,
                shard={
                    "shard_id": shard_id,
                    "attempt": 0,
                    "family_id": family_id,
                    "seed": request["seed"],
                },
            )

            self.assertEqual(rejected, [], family_id)
            self.assertEqual(len(accepted), 1, family_id)
            self.assertEqual(
                accepted[0]["verifier"]["kind"],
                expected_verifiers[family_id],
            )
            self.assertNotEqual(
                accepted[0]["generator_receipt"]["request_id"],
                accepted[0]["critic_receipt"]["request_id"],
            )
            self.assertGreaterEqual(accepted[0]["token_count"], 300)


def solve(family_id: str, task_spec: dict) -> str:
    if family_id == "verified-reasoning":
        result = format_number(evaluate_arithmetic(task_spec["expression"]))
        return f"FINAL: {result}"
    if family_id == "tool-use-and-recovery":
        return json.dumps(
            {
                "tool_calls": task_spec["required_calls"],
                "final_status": "verified",
            },
            sort_keys=True,
        )
    if family_id == "planning-and-state":
        return json.dumps(task_spec, sort_keys=True)
    if family_id == "coding-and-validation":
        return json.dumps(
            {
                "file": task_spec["file"],
                "before_sha256": sha256_text(task_spec["original_content"]),
                "after_content": task_spec["expected_content"],
                "test_command": task_spec["test_command"],
                "test_status": "passed",
            },
            sort_keys=True,
        )
    if family_id == "skill-routing-and-reflection":
        required = set(task_spec["request_tags"])
        eligible = [
            (len(skill["tags"]), skill["skill_id"])
            for skill in task_spec["skills"]
            if required <= set(skill["tags"])
        ]
        selected = sorted(eligible)[0][1]
        return json.dumps(
            {
                "selected_skill": selected,
                "steps": ["validate manifest", "run local audit"],
            },
            sort_keys=True,
        )
    raise ValueError(f"unsupported family: {family_id}")


if __name__ == "__main__":
    unittest.main()
