---
name: skill-sft-campaign
description: Plan, generate, validate, resume, and audit skill-driven SFT data campaigns for small language models. Use when work involves parallel generator and critic subagents, self-evolving skill candidates, deterministic verifier gates, global deduplication, tokenizer-based token targets, shard refill, or proving that a requested sample and token scale has actually been reached.
---

# Skill SFT Campaign

Treat data volume as an engineering target, not proof of model quality. Keep
benchmark payloads out of training and require downstream matched evaluation.

## Workflow

1. Read and validate the campaign manifest before launching generators.
2. Freeze the parent skill and synthetic development suite.
3. Generate skill candidates only from synthetic development, critic, and
   deterministic-verifier failure clusters.
4. Promote one candidate only when it improves the frozen development suite
   with zero safety-family regressions. Otherwise retain the parent.
5. Freeze the selected skill version before generating train rows.
6. Plan deterministic shards. Run generator and critic as separate subagents.
7. Recompute source policy, verifier result, hashes, token count, and duplicate
   decisions locally. Never trust subagent-reported totals.
8. Merge accepted rows into an append-only ledger, audit family and global
   targets, and create refill shards for deficits.
9. Unlock SFT only after every completion check passes.
10. Run matched baseline, skill-only, SFT-only, and combined evaluations.

## Commands

Use the repository environment and frozen manifest:

```bash
PYTHON=${NANO_WORKSPACE_PYTHON:-../.venv/bin/python}
MANIFEST=manifests/skill_sft_campaign_v1.json
RUN_DIR=runs/skill-sft-10k-10m-v1

$PYTHON scripts/run_skill_sft_campaign.py plan \
  --campaign "$MANIFEST" --run-dir "$RUN_DIR"

$PYTHON scripts/run_skill_sft_campaign.py run \
  --campaign "$MANIFEST" --run-dir "$RUN_DIR" \
  --generator-command-json \
    skills/skill-sft-campaign/assets/generator-command.json \
  --critic-command-json \
    skills/skill-sft-campaign/assets/critic-command.json \
  --tokenizer ../../models/Qwen3.5-4B

$PYTHON scripts/run_skill_sft_campaign.py audit \
  --campaign "$MANIFEST" --run-dir "$RUN_DIR" \
  --tokenizer ../../models/Qwen3.5-4B

$PYTHON scripts/run_skill_sft_campaign.py refill \
  --campaign "$MANIFEST" --run-dir "$RUN_DIR"
```

Read `references/contracts.md` before authoring generator or critic commands,
row schemas, verifier payloads, or skill-candidate scorecards.

## Gates

- Do not load benchmark, canary, or independent-holdout payloads.
- Do not edit the frozen skill while train shards are in flight.
- Do not let generators approve their own rows.
- Do not count rejected, development, duplicate, or unverified rows.
- Do not estimate tokens from characters or subagent usage reports.
- Do not claim 10k/10M completion unless the local audit says
  `training_unblocked=true`.
- Do not claim quality uplift from data completion or training loss.
- Preserve rejected rows and reasons in shard-local receipts; keep raw
  subagent outputs out of public commits.

## Self-Evolution

Use one bounded candidate cycle at a time. Score parent and candidates on the
same frozen synthetic development cases. Require:

- identical case IDs and scorer;
- candidate aggregate score strictly above the parent;
- no protected-family score below the parent;
- immutable skill content hash and scorecard;
- one selected candidate at most.

After promotion, record the selected skill hash in every shard receipt. If no
candidate passes, retain the parent and continue without inventing a promotion.
