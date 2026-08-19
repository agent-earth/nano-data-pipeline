from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "nano_skill_sft_campaign_v1"
OVERLAY_SCHEMA_VERSION = "nano_skill_sft_campaign_overlay_v1"
REQUIRED_FORBIDDEN_SOURCES = {
    "clawbench",
    "gpqa",
    "gsm8k",
    "mmlu",
    "skillbench",
    "swe-bench",
    "terminal-bench",
    "wildclawbench",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def load_skill_sft_campaign(path: str | Path) -> dict[str, Any]:
    campaign_path = Path(path).resolve()
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("schema_version") == OVERLAY_SCHEMA_VERSION:
        campaign = _resolve_campaign_overlay(campaign_path, campaign)
    validate_skill_sft_campaign(campaign)
    return campaign


def validate_skill_sft_campaign(campaign: dict[str, Any]) -> None:
    if campaign.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported skill SFT campaign schema")
    if campaign.get("status") != "pre_registered":
        raise ValueError("campaign must remain pre_registered before generation")

    claim_boundary = campaign.get("claim_boundary", {})
    if claim_boundary.get("target_is_empirical_optimum") is not False:
        raise ValueError("campaign target must not be presented as an optimum")
    if claim_boundary.get("generation_completion_proves_quality") is not False:
        raise ValueError("data volume must not be presented as quality evidence")

    targets = campaign.get("targets", {})
    train_samples = _positive_int(
        targets,
        "accepted_train_samples_min",
    )
    train_tokens = _positive_int(
        targets,
        "accepted_train_tokens_min",
    )
    dev_samples = _positive_int(targets, "accepted_dev_samples_min")
    if train_samples < 10_000:
        raise ValueError("campaign must target at least 10,000 train samples")
    if train_tokens < 10_000_000:
        raise ValueError("campaign must target at least 10,000,000 train tokens")
    if dev_samples < 1:
        raise ValueError("campaign needs a held-out synthetic development split")

    token_accounting = campaign.get("token_accounting", {})
    if token_accounting.get("unit") != "qwen3.5_tokenizer_input_id":
        raise ValueError("token target must use Qwen3.5 tokenizer input IDs")
    if token_accounting.get("enable_thinking") is not False:
        raise ValueError("token accounting must disable thinking")
    if token_accounting.get("counted_split") != "train":
        raise ValueError("only accepted train rows may count toward the target")
    for filename, digest in token_accounting.get("file_sha256", {}).items():
        if not filename or SHA256_PATTERN.fullmatch(str(digest)) is None:
            raise ValueError("tokenizer file identities must be SHA256 digests")
    if len(token_accounting.get("file_sha256", {})) < 3:
        raise ValueError("tokenizer identity must pin config, vocabulary, and template")

    families = campaign.get("data_families")
    if not isinstance(families, list) or not families:
        raise ValueError("campaign must define data-family quotas")
    family_ids: set[str] = set()
    family_train_samples = 0
    family_train_tokens = 0
    family_dev_samples = 0
    for family in families:
        family_id = str(family.get("family_id", ""))
        if not family_id or family_id in family_ids:
            raise ValueError("data-family IDs must be non-empty and unique")
        family_ids.add(family_id)
        family_train_samples += _positive_int(
            family,
            "accepted_train_samples_min",
        )
        family_train_tokens += _positive_int(
            family,
            "accepted_train_tokens_min",
        )
        family_dev_samples += _positive_int(
            family,
            "accepted_dev_samples_min",
        )
        if not family.get("deterministic_verifier"):
            raise ValueError(f"{family_id} needs a deterministic verifier")
    if family_train_samples < train_samples:
        raise ValueError("family sample quotas do not cover the campaign target")
    if family_train_tokens < train_tokens:
        raise ValueError("family token quotas do not cover the campaign target")
    if family_dev_samples < dev_samples:
        raise ValueError("family development quotas do not cover the target")

    sharding = campaign.get("sharding", {})
    initial_shards = _positive_int(sharding, "initial_shards")
    parallel_subagents = _positive_int(
        sharding,
        "max_parallel_subagents",
    )
    samples_per_shard = _positive_int(
        sharding,
        "candidate_samples_per_shard",
    )
    candidate_tokens = _positive_int(
        sharding,
        "initial_candidate_tokens_min",
    )
    oversubscription = float(sharding.get("initial_oversubscription_ratio", 0))
    if parallel_subagents < 2:
        raise ValueError("generation must support multiple parallel subagents")
    if initial_shards < parallel_subagents:
        raise ValueError("initial shards must cover every parallel subagent")
    if oversubscription < 1.2:
        raise ValueError("initial generation must reserve at least 20% rejection room")
    if initial_shards * samples_per_shard < math.ceil(
        train_samples * oversubscription
    ):
        raise ValueError("initial shard plan cannot meet the sample oversubscription")
    if candidate_tokens < math.ceil(train_tokens * oversubscription):
        raise ValueError("initial token plan cannot meet the token oversubscription")
    if sharding.get("refill_until_global_targets_pass") is not True:
        raise ValueError("campaign must refill after global validation")

    source_policy = campaign.get("source_policy", {})
    if source_policy.get("benchmark_payload_training_allowed") is not False:
        raise ValueError("benchmark payloads must be forbidden from training")
    forbidden = {
        str(value).lower()
        for value in source_policy.get("forbidden_payload_sources", [])
    }
    missing_forbidden = REQUIRED_FORBIDDEN_SOURCES - forbidden
    if missing_forbidden:
        raise ValueError(
            f"missing forbidden benchmark sources: {sorted(missing_forbidden)}"
        )
    if source_policy.get("independent_holdout_may_be_read") is not False:
        raise ValueError("independent holdout must remain unread")

    gates = campaign.get("acceptance_gates", {})
    if gates.get("deterministic_verifier_pass_required") is not True:
        raise ValueError("every accepted row must pass a deterministic verifier")
    if gates.get("independent_critic_required") is not True:
        raise ValueError("every accepted row must have an independent critic")
    minimum_critic_score = gates.get("minimum_critic_score")
    if (
        not isinstance(minimum_critic_score, (int, float))
        or isinstance(minimum_critic_score, bool)
        or not 0 < float(minimum_critic_score) <= 1
    ):
        raise ValueError("minimum critic score must be in (0, 1]")
    if gates.get("global_exact_duplicates_allowed") != 0:
        raise ValueError("exact duplicate rows must be rejected")
    semantic_threshold = float(gates.get("semantic_similarity_max", 1.0))
    if not 0 < semantic_threshold < 1:
        raise ValueError("semantic similarity threshold must be in (0, 1)")
    if gates.get("cross_split_overlap_allowed") != 0:
        raise ValueError("train/development overlap must be zero")

    self_evolution = campaign.get("self_evolution", {})
    if self_evolution.get("benchmark_feedback_may_mutate_skills") is not False:
        raise ValueError("benchmark feedback must not mutate skills")
    if _positive_int(self_evolution, "max_cycles_before_freeze") > 5:
        raise ValueError("skill evolution must freeze after a bounded cycle count")

    completion = campaign.get("completion_contract", {})
    required = {
        "family_quotas_pass",
        "global_dedup_pass",
        "source_policy_pass",
        "tokenizer_identity_pass",
        "train_sample_target_pass",
        "train_token_target_pass",
    }
    if required - set(completion.get("required_checks", [])):
        raise ValueError("completion contract is missing mandatory checks")
    if completion.get("training_unblocks_only_after_all_checks") is not True:
        raise ValueError("training must remain blocked until all checks pass")

    generation_mode = campaign.get("generation_protocol", {}).get(
        "mode",
        "per_row_subagent_v1",
    )
    if generation_mode not in {
        "per_row_subagent_v1",
        "recipe_per_shard_v1",
    }:
        raise ValueError("unsupported generation protocol mode")
    if generation_mode == "recipe_per_shard_v1":
        protocol = campaign["generation_protocol"]
        if protocol.get("generator_calls_per_shard_max") != 1:
            raise ValueError("recipe mode requires one generator call per shard")
        if protocol.get("critic_calls_per_shard_max") != 1:
            raise ValueError("recipe mode requires one critic call per shard")
        if protocol.get("row_expander") != "deterministic_local_compiler_v2":
            raise ValueError("recipe mode requires the frozen local row expander")


def _resolve_campaign_overlay(
    overlay_path: Path,
    overlay: dict[str, Any],
) -> dict[str, Any]:
    allowed = {
        "base_manifest",
        "base_sha256",
        "campaign_id",
        "overrides",
        "schema_version",
        "status",
    }
    unknown = set(overlay) - allowed
    if unknown:
        raise ValueError(f"unknown campaign overlay fields: {sorted(unknown)}")
    relative = Path(str(overlay.get("base_manifest", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("campaign overlay base must stay beside the overlay")
    base_path = (overlay_path.parent / relative).resolve()
    if overlay_path.parent.resolve() not in base_path.parents:
        raise ValueError("campaign overlay base escapes its directory")
    expected_sha256 = str(overlay.get("base_sha256", ""))
    from nano_data_pipeline.feedback import sha256_file

    if sha256_file(base_path) != expected_sha256:
        raise ValueError("campaign overlay base SHA256 mismatch")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("campaign overlay base must use the v1 schema")
    overrides = overlay.get("overrides")
    if not isinstance(overrides, dict):
        raise ValueError("campaign overlay overrides must be an object")
    resolved = _deep_merge(base, overrides)
    resolved["campaign_id"] = str(overlay.get("campaign_id", ""))
    resolved["status"] = str(overlay.get("status", ""))
    resolved["overlay_receipt"] = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "overlay_sha256": sha256_file(overlay_path),
        "base_manifest": relative.as_posix(),
        "base_sha256": expected_sha256,
    }
    return resolved


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _positive_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value
