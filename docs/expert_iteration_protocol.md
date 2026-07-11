# Evidence-Gated Expert Iteration Protocol

## Status

This document is the required protocol before starting another ExIt training
round. It replaces ad-hoc "collect some games, train, ladder" iterations.

The first r2 candidate is a **smoke-test artifact**, not a promoted model. It
was trained on 4,792 self-play trajectories only, without the human anchor used
for r1. Its public ladder gate was stopped. Do not use it as evidence about
whether strong-search ExIt works.

## What Literature Supports

The relevant recipe is not pure self-play behavioral cloning.

1. **Expert iteration / AlphaZero:** search improves a policy; policy targets
   come from search; the improved policy then improves later search. This makes
   the loop iterative, but does not guarantee every generation improves.
2. **Metamon / PA-Agent recipe:** warm start from human replay behavior,
   generate population data, fine-tune offline, and retain human data rather
   than annealing it to zero.
3. **AlphaStar PFSP:** train against a checkpoint/opponent pool, concentrating
   on roughly even matchups. This prevents a policy from only learning to beat
   its own latest behavior.
4. **piKL / Diplodocus:** regularization toward a human policy protects against
   the self-play distribution drifting away from the population metric that
   ladder GXE measures.

Sources and exact caveats are collected in `docs/research_ideas.md` B1-B3.
The reported useful Metamon round scale is 100k-500k full battles. We cannot
call a smaller batch literature-scale; use it only as a gated pilot.

## Terms

| Term | Meaning |
|---|---|
| Battle | One complete two-player Showdown game. |
| POV trajectory | One player's parsed trajectory from a battle. Normally two per battle. |
| Raw replay | Captured Showdown protocol JSON. It is not training-ready until parsed. |
| Strict battle | A battle in which required priors succeeded for every decision. |
| Candidate | A newly trained checkpoint that has not passed promotion gates. |
| Accepted checkpoint | The only checkpoint allowed to generate a subsequent ExIt round. |

All reports must state battles and POV trajectories separately. Never call a
raw replay count "games".

## Frozen Baselines

Before every round, lock and record:

- Engine/search revision and wheel hash
- Search budget: 500ms, parallelism 8, one search thread
- Prior server checkpoint and hash
- Human replay corpus hash and number of indexed trajectories
- Opponent-pool composition and random seed
- H2H opponent pool

The currently accepted public result is r1: **92.4-92.7 GXE at RD 25**. It is
the public-ladder reference, but every promotion must also beat a frozen H2H
pool at the same search budget.

## Data Integrity Gate

No data is admitted into a training round unless every condition holds.

1. Generator uses `METAGROSS_REQUIRE_PRIORS=1`; a failed root-prior fetch
   aborts the battle rather than recording fallback-FP behavior.
2. The prior server writes counters for root and opponent priors. Root-prior
   success must be 100%; opponent-prior availability must be measured and
   reported. Do not label data C1+C2 if opponent priors are absent.
3. Generated usernames cannot contain `_`; Metamon's parser uses `_` as a
   filename separator. The parser must produce two distinguishable POV files
   per successful battle.
4. Every parsed `.json.lz4` must decode, contain nonempty equal-length
   `states` and `actions`, and live below its canonical format directory.
5. The Metamon dataset loader must index exactly the manifest count before a
   GPU job starts.
6. Store `MANIFEST.json` per shard: generator git SHA, prior checkpoint hash,
   search budget, prior success counters, battle count, POV count, and parser
   validation count.

Mixed/fallback data is archived for debugging only. It is never silently mixed
into a promotion round.

## Round Protocol

### Pilot Round

Use this only to validate mechanics, not to claim literature-scale improvement.

- 5k strict full battles (about 10k POV trajectories)
- Opponent pool: accepted checkpoint vs itself, with both sides recorded
- Data mix: 90% strict self-play, 10% fixed human anchor
- Training: one variable from the accepted recipe; start with 1-2 epochs,
  KL-to-human anchor enabled
- Gate: 200 paired H2H games versus the accepted checkpoint. If CI contains
  50%, report no decision; do not promote.

### Promotion Round

Required before claiming iteration improvement.

- 25k strict full battles minimum (about 50k POV trajectories)
- Target 100k full battles if the pilot clears its H2H gate and budget allows
- 90% strict self-play / 10% fixed human anchor by sampling weight
- Fine-tune from the accepted checkpoint, not an unrelated base checkpoint
- Preserve a full trainable checkpoint plus a filtered deployment-policy
  checkpoint; record hashes for both
- Train 1-2 epochs initially. More epochs require a held-out reason, not hope.

### Opponent Pool (PFSP-lite)

Self-only collection is prohibited after the pilot. Generate from a fixed,
versioned pool:

- 35% accepted checkpoint
- 25% frozen base Kakuna-guided FP
- 20% previous accepted checkpoints
- 20% stock FP / public baseline

Reweight the pool toward opponents where the current generator wins roughly
40-60%. Keep a held-out subset of every pool member out of training evaluation.

## Evaluation And Promotion

1. First run the scorer sanity gate: frozen agent self-play at the target
   budget, paired, N >= 100, expected 50% within CI.
2. Candidate vs accepted checkpoint: >= 500 paired H2H games, both sides of
   each matchup. Promote only if the 95% CI is entirely above 50%.
3. Candidate vs fixed external pool: stock FP, base Kakuna-guided FP, and at
   least one older checkpoint. No regression beyond the predeclared noise band.
4. Public ladder is a bounded secondary validation only. Use a fresh account,
   stop at an RD target, and do not run it continuously. The public ladder is
   not a data generator.
5. A candidate that fails any gate is archived, not used to generate the next
   round.

## Stop Rules

- Three accepted-data rounds without a held-out H2H gain beyond noise: stop
  ExIt and revisit leaf evaluation/search.
- Any human-anchor or external-pool regression: reject candidate immediately.
- Any root-prior failure in a strict shard: reject affected battles.
- Never explain a regression by adding epochs. First audit data provenance,
  opponent pool, and the H2H gate.

## Current Required Repairs Before Resuming

1. Fix generator usernames to remove `_` and prove two parsed POV trajectories
   per strict battle.
2. Verify root prior success rate after the stateless sequence fix.
3. Repair/measure opponent-prior (C2) availability. Until then, label data C1
   only and do not compare it to a C1+C2 baseline.
4. Rebuild the pilot dataset from strict-only battles plus the fixed human
   anchor. The existing 4,792-trajectory r2 smoke batch remains non-promotable.
