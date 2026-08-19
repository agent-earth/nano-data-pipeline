# Campaign Contracts

## Command JSON

Generator and critic command files are JSON arrays. Do not invoke a shell.
The runner replaces placeholders in each argument:

```json
[
  "agent-cli",
  "generate",
  "--input",
  "{input}",
  "--output",
  "{output}"
]
```

Supported placeholders are `{input}`, `{output}`, `{family_id}`, `{shard_id}`,
`{attempt}`, and `{skill_path}`. The process must write its declared output
path and exit zero. Environment credentials remain process-local.

## Generator Input

The runner writes one JSON request per shard:

```json
{
  "schema_version": "nano_subagent_generator_request_v1",
  "campaign_id": "skill-sft-10k-10m-v1",
  "family_id": "verified-reasoning",
  "shard_id": 0,
  "attempt": 0,
  "candidate_samples": 512,
  "seed": 202608190000,
  "skill_path": "skills/skill-sft-campaign/SKILL.md",
  "skill_sha256": "...",
  "output_schema": "nano_subagent_candidate_v1"
}
```

The generator writes JSONL. Each row must contain:

```json
{
  "schema_version": "nano_subagent_candidate_v1",
  "candidate_id": "stable-id",
  "family_id": "verified-reasoning",
  "split": "train",
  "skill_id": "skill-sft-campaign",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "task_spec": {
    "expression": "17 + 29"
  },
  "source": {
    "kind": "procedurally_generated_synthetic",
    "generator": "provider/model",
    "seed": 202608190000
  },
  "verifier": {
    "kind": "safe_execution_receipt_v1"
  },
  "generator_receipt": {
    "request_id": "provider-request-id",
    "skill_sha256": "..."
  }
}
```

The local runner rejects unknown families, benchmark-like payloads, missing
assistant targets, a verifier that does not match the family's frozen
verifier, wrong skill hashes, and duplicate IDs. The five production verifier
kinds are `tool_trace_contract_v1`, `state_plan_consistency_v1`,
`safe_execution_receipt_v1`, `patch_test_receipt_v1`, and
`skill_route_receipt_v1`.

## Critic Input And Output

The critic receives candidates but no generator hidden reasoning or quota
deficit. It returns one JSONL decision per candidate:

```json
{
  "schema_version": "nano_subagent_critic_v1",
  "candidate_id": "stable-id",
  "score": 0.9,
  "accept": true,
  "reasons": [],
  "critic_receipt": {
    "request_id": "independent-request-id",
    "critic": "provider/model"
  }
}
```

The runner requires exactly one decision for every candidate. A row passes
only when `accept=true`, `score` meets the campaign threshold, and the local
verifier passes.

## Local Acceptance

For accepted rows, the runner executes the family's verifier from `task_spec`
and computes:

- `sample_id` from campaign, family, shard, attempt, and candidate ID;
- canonical `exact_hash` over the full messages;
- normalized `semantic_hash`;
- Qwen3.5 chat-template `token_count`;
- generator, critic, verifier, source, skill, shard, and attempt receipts.

Exact hashes, semantic hashes, and sample IDs must be globally unique.
Train/dev sets must have zero overlap. The audit recomputes every hash and
token count from source rows.

## Ledger And Resume

Each shard has an immutable plan and one status record. Completed accepted
shards are not rerun. Failed or interrupted shards keep their attempt receipt
and may be retried as a new attempt. `accepted.jsonl` is derived from shard
artifacts; it is not edited by subagents.

## Skill Candidate Scorecard

Use the same frozen development case IDs for parent and candidate:

```json
{
  "schema_version": "nano_skill_scorecard_v1",
  "skill_id": "candidate-v2",
  "skill_sha256": "...",
  "case_ids_sha256": "...",
  "aggregate_score": 0.84,
  "family_scores": {
    "safety": 1.0,
    "tool-use-and-recovery": 0.8
  }
}
```

Promotion requires candidate aggregate above parent and every protected
family score greater than or equal to the parent. Tie means retain parent.
