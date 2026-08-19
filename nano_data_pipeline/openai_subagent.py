from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nano_data_pipeline.subagent_campaign import (
    CANDIDATE_SCHEMA,
    CRITIC_SCHEMA,
    canonical_json,
)


FAMILY_VERIFIERS = {
    "coding-and-validation": "patch_test_receipt_v1",
    "planning-and-state": "state_plan_consistency_v1",
    "skill-routing-and-reflection": "skill_route_receipt_v1",
    "tool-use-and-recovery": "tool_trace_contract_v1",
    "verified-reasoning": "safe_execution_receipt_v1",
}
FORBIDDEN_CONTEXT_MARKERS = {
    "benchmark",
    "canary",
    "gpqa",
    "gsm8k",
    "holdout",
    "mmlu",
    "swe-bench",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"\A```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```\Z", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response contains no JSON object")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


@dataclass(frozen=True)
class SubagentConfig:
    role: str
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float
    temperature: float
    max_tokens: int
    tokenizer_path: Path | None

    @classmethod
    def from_env(cls, role: str) -> "SubagentConfig":
        role = {"generate": "generator", "criticize": "critic"}.get(
            role,
            role,
        )
        if role not in {"generator", "critic"}:
            raise ValueError("subagent role must be generator or critic")
        prefix = f"NANO_{role.upper()}_"

        def get(name: str, default: str | None = None) -> str:
            return os.getenv(prefix + name, os.getenv("NANO_SUBAGENT_" + name, default or ""))

        base_url = get("BASE_URL")
        model = get("MODEL")
        if not base_url or not model:
            raise ValueError(
                f"{prefix}BASE_URL and {prefix}MODEL are required"
            )
        tokenizer_raw = get("TOKENIZER")
        tokenizer_path = (
            Path(tokenizer_raw).expanduser().resolve() if tokenizer_raw else None
        )
        if role == "generator" and (
            tokenizer_path is None or not tokenizer_path.is_dir()
        ):
            raise ValueError("generator requires a local tokenizer directory")
        return cls(
            role=role,
            base_url=base_url,
            model=model,
            api_key=get("API_KEY", "local-not-required"),
            timeout_seconds=float(get("TIMEOUT_SECONDS", "300")),
            temperature=float(get("TEMPERATURE", "0")),
            max_tokens=int(get("MAX_TOKENS", "512")),
            tokenizer_path=tokenizer_path,
        )


class OpenAICompatibleSubagent:
    def __init__(self, config: SubagentConfig):
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=2,
        )

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content or ""
        value = parse_json_object(content)
        usage = (
            response.usage.model_dump(exclude_none=True)
            if response.usage
            else {}
        )
        return value, {
            "request_id": response.id,
            "model": self.config.model,
            "usage": usage,
            "finish_reason": response.choices[0].finish_reason,
        }


class FamilyCompiler:
    def __init__(self, tokenizer_path: Path):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
        )

    def compile(
        self,
        *,
        family_id: str,
        seed: int,
        index: int,
        target_tokens: int,
    ) -> dict[str, Any]:
        if family_id not in FAMILY_VERIFIERS:
            raise ValueError(f"unsupported data family: {family_id}")
        base = _base_task(family_id, seed, index)
        system = (
            "Follow the synthetic task contract exactly. Use only the supplied "
            "facts. Output one JSON object or one FINAL line as requested."
        )
        messages = self._pad_messages(
            system=system,
            user=base["user"],
            family_id=family_id,
            seed=seed,
            index=index,
            target_tokens=max(64, target_tokens),
        )
        compiled = {
            "messages": messages,
            "task_spec": base["task_spec"],
            "verifier": {"kind": FAMILY_VERIFIERS[family_id]},
            "compiled_prompt_tokens": self._token_count(messages),
        }
        serialized = canonical_json(compiled).lower()
        if any(marker in serialized for marker in FORBIDDEN_CONTEXT_MARKERS):
            raise ValueError("compiled task contains a forbidden marker")
        return compiled

    def _pad_messages(
        self,
        *,
        system: str,
        user: str,
        family_id: str,
        seed: int,
        index: int,
        target_tokens: int,
    ) -> list[dict[str, str]]:
        def build(fact_count: int) -> list[dict[str, str]]:
            evidence = [
                self._evidence_line(family_id, seed, index, fact_index)
                for fact_index in range(1, fact_count + 1)
            ]
            content = user
            if evidence:
                content += "\n" + "\n".join(evidence)
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ]

        base = build(0)
        if self._token_count(base) >= target_tokens:
            return base
        lower = 0
        upper = 8
        while self._token_count(build(upper)) < target_tokens:
            lower = upper
            upper *= 2
            if upper > 4_096:
                raise ValueError("unable to reach target prompt token count")
        while lower + 1 < upper:
            middle = (lower + upper) // 2
            if self._token_count(build(middle)) >= target_tokens:
                upper = middle
            else:
                lower = middle
        return build(upper)

    def _evidence_line(
        self,
        family_id: str,
        seed: int,
        index: int,
        fact_index: int,
    ) -> str:
        key = sha256_text(
            f"{family_id}:{seed}:{index}:{fact_index}"
        )[:10]
        value = (seed * 17 + index * 31 + fact_index * 43) % 100_003
        return (
            f"Synthetic evidence {fact_index:04d}: record-{key} has value "
            f"{value} and is unrelated unless explicitly referenced by the "
            "task contract."
        )

    def _token_count(self, messages: list[dict[str, str]]) -> int:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(encoded, "get"):
            encoded = encoded.get("input_ids")
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if not isinstance(encoded, list):
            raise ValueError("tokenizer did not return input IDs")
        return len(encoded)


def generate_candidates(
    request: dict[str, Any],
    subagent: OpenAICompatibleSubagent,
    compiler: FamilyCompiler,
) -> list[dict[str, Any]]:
    sample_count = request["candidate_samples"]
    target_tokens = math.ceil(
        request["candidate_tokens_min"] / sample_count
    )
    rows = []
    for index in range(sample_count):
        compiled = compiler.compile(
            family_id=request["family_id"],
            seed=request["seed"],
            index=index,
            target_tokens=target_tokens,
        )
        response, receipt = subagent.complete_json(
            _generator_messages(request, compiled),
        )
        assistant = response.get("assistant")
        if not isinstance(assistant, str) or not assistant.strip():
            assistant = canonical_json(response)
        candidate_id = (
            f"{request['family_id']}-{request['shard_id']}-"
            f"{request['attempt']}-{index}"
        )
        rows.append(
            {
                "schema_version": CANDIDATE_SCHEMA,
                "candidate_id": candidate_id,
                "family_id": request["family_id"],
                "task_family": f"{request['family_id']}-real-pilot",
                "split": "train",
                "skill_id": "skill-sft-campaign",
                "messages": [
                    *compiled["messages"],
                    {"role": "assistant", "content": assistant.strip()},
                ],
                "task_spec": compiled["task_spec"],
                "source": {
                    "kind": "procedurally_generated_synthetic",
                    "generator": receipt["model"],
                    "seed": request["seed"],
                    "compiler": "deterministic_long_form_v1",
                },
                "verifier": compiled["verifier"],
                "generator_receipt": {
                    **receipt,
                    "skill_sha256": request["skill_sha256"],
                    "compiled_prompt_tokens": compiled[
                        "compiled_prompt_tokens"
                    ],
                },
            }
        )
    return rows


def criticize_candidates(
    request: dict[str, Any],
    subagent: OpenAICompatibleSubagent,
) -> list[dict[str, Any]]:
    decisions = []
    for candidate in request["candidates"]:
        response, receipt = subagent.complete_json(
            _critic_messages(candidate),
            max_tokens=min(256, subagent.config.max_tokens),
        )
        score = response.get("score", 0)
        try:
            score_value = min(1.0, max(0.0, float(score)))
        except (TypeError, ValueError):
            score_value = 0.0
        accept = response.get("accept") is True and score_value >= 0.8
        reasons = response.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = ["critic_reasons_not_list"]
        decisions.append(
            {
                "schema_version": CRITIC_SCHEMA,
                "candidate_id": candidate["candidate_id"],
                "score": score_value,
                "accept": accept,
                "reasons": [str(reason) for reason in reasons],
                "critic_receipt": {
                    **receipt,
                    "critic": receipt["model"],
                },
            }
        )
    return decisions


def _generator_messages(
    request: dict[str, Any],
    compiled: dict[str, Any],
) -> list[dict[str, str]]:
    payload = {
        "family_id": request["family_id"],
        "messages": compiled["messages"],
        "task_spec": compiled["task_spec"],
        "verifier": compiled["verifier"],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a generator subagent. Solve the synthetic task. "
                "Return one JSON object with exactly one key named assistant. "
                "The assistant value must be the complete target response. "
                "Do not change task_spec and do not mention evaluation data."
            ),
        },
        {
            "role": "user",
            "content": canonical_json(payload),
        },
    ]


def _critic_messages(candidate: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "family_id": candidate["family_id"],
        "messages": candidate["messages"],
        "task_spec": candidate["task_spec"],
        "verifier": candidate["verifier"],
        "source": candidate["source"],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an independent critic subagent. Judge whether the "
                "assistant response follows the stated synthetic contract, is "
                "internally consistent, and contains no unsupported claims. "
                "Return one JSON object with accept (boolean), score (0 to 1), "
                "and reasons (array of short strings). Do not optimize for a "
                "quota and do not infer hidden reference answers."
            ),
        },
        {"role": "user", "content": canonical_json(payload)},
    ]


def _base_task(family_id: str, seed: int, index: int) -> dict[str, Any]:
    left = 40 + (seed + index * 7) % 80
    right = 20 + (seed * 3 + index * 11) % 60
    if family_id == "verified-reasoning":
        expression = f"({left} + {right}) * 3 - {right}"
        return {
            "user": (
                f"Compute {expression}. Return exactly one line: "
                "FINAL: <number>."
            ),
            "task_spec": {"expression": expression},
        }
    if family_id == "tool-use-and-recovery":
        required_calls = [
            {
                "name": "lookup",
                "arguments": {"key": f"synthetic-{left}"},
                "status": "error",
            },
            {
                "name": "calculator",
                "arguments": {"expression": f"{left}+{right}"},
                "status": "ok",
            },
        ]
        return {
            "user": (
                "Return one JSON object with tool_calls equal to the required "
                "calls and final_status set to verified. Required calls: "
                + canonical_json(required_calls)
            ),
            "task_spec": {"required_calls": required_calls},
        }
    if family_id == "planning-and-state":
        task_spec = {
            "constraints": [
                f"limit-{left}",
                "use-only-synthetic-evidence",
            ],
            "evidence": [f"observation-{right}", f"receipt-{left + right}"],
            "pending": [f"verify-{left * right}"],
            "stop": False,
        }
        return {
            "user": (
                "Return one JSON object with constraints, evidence, pending, "
                "and stop exactly matching this state: "
                + canonical_json(task_spec)
            ),
            "task_spec": task_spec,
        }
    if family_id == "coding-and-validation":
        original = f"total = {left}\n"
        expected = f"total = {left + right}\n"
        task_spec = {
            "file": f"synthetic_{index}.py",
            "original_content": original,
            "expected_content": expected,
            "test_command": f"python -m unittest synthetic_{index}",
        }
        before_hash = sha256_text(original)
        return {
            "user": (
                "Return one JSON object with file, before_sha256, "
                "after_content, test_command, and test_status=passed. "
                f"before_sha256 is {before_hash}. Contract: "
                + canonical_json(task_spec)
            ),
            "task_spec": task_spec,
        }
    if family_id == "skill-routing-and-reflection":
        task_spec = {
            "request_tags": ["data", "validation"],
            "skills": [
                {
                    "skill_id": f"broad-{index}",
                    "tags": ["data", "validation", "training"],
                },
                {
                    "skill_id": f"minimal-{index}",
                    "tags": ["data", "validation"],
                },
            ],
        }
        return {
            "user": (
                "Select the eligible skill with the fewest tags. Return one "
                "JSON object with selected_skill and a non-empty steps array. "
                "Contract: "
                + canonical_json(task_spec)
            ),
            "task_spec": task_spec,
        }
    raise ValueError(f"unsupported data family: {family_id}")
