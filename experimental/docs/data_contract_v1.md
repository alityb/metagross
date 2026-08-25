# Metagross Data Contract v1

## Decision

The current target is `gen9randombattle`. We will use three complementary
signals for the deployable model:

| Signal | Sampling weight | What it teaches |
|---|---:|---|
| Format-matched human replays | 40% | Human/population policy anchor and terminal outcomes |
| Format-matched search-improved roots | 40% | Visit policy, action-value advantages, and uncertainty |
| Format-matched league self-play | 20% | On-policy state coverage and exact terminal outcomes |

Before that multitask stage, initialize the shared representation with a 70/30
sample of pinned Metamon Gen 9 OU human trajectories and local Gen 9 Random
Battle human trajectories. The first Metamon sample is capped at 500,000 POV
trajectories. This is cross-format warm-start data, not evidence for Gen 9
Random Battle strength.

The machine-readable authority is
`experimental/configs/data_sources_v1.json`. Validate it with:

```bash
python -m srcs.metagross.data_source_registry \
  experimental/configs/data_sources_v1.json
```

## Why Metamon Is Added This Way

The current Metamon release advertises roughly 5.3M reconstructed human
trajectories. Its public self-play pile contains 11M `pac-base`, 7M
`pac-exploratory`, and 4M `pac-tauros` trajectories. The current released
formats are Gen 1-4 OU and Gen 9 OU (Tauros is Gen 1 OU only), not Gen 9 Random
Battle. Consequently:

- `metamon-parsed-replays@v6` Gen 9 OU is approved for non-commercial
  representation warm-start, human-policy anchoring, and auxiliary terminal
  value learning.
- It does not provide format-matched search-Q truth for Random Battle.
- Its reconstructed/backfilled opponent sets must never be treated as true
  hidden-state labels. Belief supervision uses genuinely revealed/censored
  information or exact simulator truth from our self-play.
- The Metamon code is MIT, but `metamon-parsed-replays` is CC BY-NC 4.0. Any
  model trained from it must be marked non-commercial and carry attribution.
- The `metamon-parsed-pile` dataset card declares no license. Its three
  self-play subsets remain quarantined until the authors or dataset metadata
  grant explicit permission. They may be inspected in a research-only
  ablation, but cannot enter a promoted or releasable checkpoint.

Pinned upstream identities:

- Metamon code: `0a00a759c9a4382a2877088d828302ec294a05a5`
- Parsed human trajectories v6:
  `7d82b873647dee35a62e7b63cd253e5d273cbe87`
- Raw replay v6: `792a86b5cdbcab9696032e947483b218453a0d4f`
- Parsed self-play pile:
  `4f800783dd2078928969f1515ac466af858f0e48`

Do not use an unpinned `main` revision in an experiment manifest.

Preview the approved Gen 9 OU human download (currently 20.4GB compressed)
without changing local state:

```bash
.venv-metamon/bin/python -m srcs.metagross.fetch_metamon_data
```

Add `--download` only when the cache location, disk budget, and non-commercial
use are confirmed. The fetcher validates the pinned LFS SHA-256 before emitting
its local source manifest. It refuses the unlicensed self-play subsets.

## Exact Targets

Every network input must be obtainable from the acting player's information
state at that turn. The allowed targets are:

| Source | Allowed target |
|---|---|
| Human replay | Observed human action; exact terminal result |
| Search root | Legal-action visit distribution; belief-averaged Q advantages; repeated-seed uncertainty |
| Local self-play | Acting policy/action; exact simulator terminal result; true hidden set only as a belief **label**, never an input |
| Revealed human set data | Censored/revealed-set likelihood target |

Search Q is a teacher signal, not the verifier. Terminal outcomes from the
exact simulator and independent paired games remain the non-circular evidence.

## Identity, Deduplication, And Splits

1. Derive `canonical_battle_sha256` from normalized immutable battle protocol,
   not a filename. Both POVs, reparses, roots, and relabels inherit it.
2. Deduplicate across every local and Metamon source before splitting. Prefer
   raw protocol plus the newest validated parser output when duplicate derived
   copies exist.
3. All historical sources are limited to train/development. Stable hash buckets
   allocate 95%/5% at the **battle** level.
4. Final confirmation is a new, post-freeze collection of at least 2,000 roots
   from distinct battles. It may be evaluated once after the checkpoint and
   decision rule are frozen.
5. H2H and public ladder battles are evaluation artifacts forever. They do not
   flow back into the replay buffer.

## Required Scale Before A Promotion Claim

| Corpus/evaluation | Minimum |
|---|---:|
| Unique format-matched human battles | 50,000 |
| Search-improved roots | 25,000; prefer 50,000 |
| Distinct battles behind search roots | 10,000 |
| League pilot | 25,000 battles |
| Full league round | target 100,000 battles |
| Fresh confirmation | 2,000 roots |
| Paired H2H | 1,000 games |
| Initial public ladder | 400 games |
| Strong public-ladder claim | 1,000 games |

The existing 411-root information-state set remains a development-only exporter
test. The existing 5,000-battle G5 league and historical MCTS/PFSP artifacts are
seed material after lineage audit; they do not satisfy the new promotion scale
by themselves.

## Search-Root Sampling

Search labeling is expensive, so select one or a small number of roots per
battle and stratify toward:

- close search margins and policy/search disagreement;
- high belief entropy or high value variance across particles;
- switches, tera choices, and other irreversible decisions;
- losses and high-regret decisions by previous agents;
- rare matchups, early/middle/late game, and rating strata.

Every target stores two independent search seeds at minimum, the full legal
mask, visit distribution, Q advantages relative to the legal-action mean,
particle/belief provenance, engine hash, and target uncertainty.

## Promotion Boundary

Training metrics and held-out teacher-Q accuracy can reject a model, but cannot
promote it. Promotion requires, in order:

1. better terminal counterfactual return on fresh confirmation roots;
2. no severe predeclared regression strata;
3. at least 1,000 paired H2H games with the confidence interval above 50%;
4. only then, a bounded public-ladder evaluation reporting GXE, rating, RD, and
   game count.

## Primary Sources

- Metamon repository and current dataset documentation:
  https://github.com/UT-Austin-RPL/metamon
- Metamon reconstructed human trajectories:
  https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays
- Metamon self-play pile:
  https://huggingface.co/datasets/jakegrigsby/metamon-parsed-pile
- Metamon paper: https://arxiv.org/abs/2504.04395
- Expert Iteration: https://arxiv.org/abs/1705.08439
- AlphaZero: https://arxiv.org/abs/1712.01815
