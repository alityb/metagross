# r2 Recovery Plan: Evidence Before Another Promotion

## Problem Statement

The first mixed r2 candidate is **not promotable**. It continued from r1
correctly (candidate policy is 1.7% relative L2 from r1 and 43.8% from base
Kakuna), but the experiment changed too many variables:

- r1 used `A_rating`; r2 used `ALL` (`A_rating + KL anchor + HL-Gauss`)
- r2 introduced a new strict-data distribution
- r2 changed training duration (6 epochs to 2)
- the new data was collected before all C2/visit logging checks existed

An early H2H deficit cannot identify which change caused it. Do not promote or
ladder this candidate.

## Fixed Reference

| Name | Meaning |
|---|---|
| Accepted r1 | `randbats_exit_r1` epoch 5, 92.4-92.7 GXE at RD 25 |
| Deployed budget | 500ms, parallelism 8, one search thread |
| Base policy | Kakuna / `randbats_D_hlgauss` |
| Human anchor | `data/parsed_replays`, 23,516 indexed trajectories |
| Legacy self-play | `data/selfplay_parsed_indexed`, 23,870 indexed trajectories |

The accepted r1 checkpoint is the only initialization for candidate rounds.
Never restart a candidate from base Kakuna while calling it an ExIt continuation.

## Evidence

1. **Expert Iteration** (Anthony, Tian, Barber 2017) supports iterating a
   search expert into an apprentice policy, but does not promise each round
   improves. A round must be promoted by held-out evidence.
2. **AlphaZero** (Silver et al. 2017) trains policy targets from MCTS visit
   distributions. Selected action alone is an information-poor target.
3. **AlphaStar PFSP** (Vinyals et al. 2019) supports an opponent pool rather
   than latest-self-only training.
4. **Metamon / PA-Agent** supports persistent human replay data during
   offline/self-play training. The target metric is human GXE, so human data is
   a target-distribution anchor, not contamination.
5. **piKL / Diplodocus** supports regularization toward the human policy to
   limit self-play population drift. This is indirect evidence for Pokémon but
   aligns with the TaurosV0 failure mode.

Full links and caveats are in `docs/expert_iteration_protocol.md`.

## Data Contract

No new training round starts until all conditions are true:

1. `METAGROSS_REQUIRE_PRIORS=1`; root-prior failure aborts the game.
2. One raw replay per battle; Metamon parser produces exactly two POV
   trajectories per battle.
3. Each decision JSON row has `state`, `selected_action`, normalized
   `mcts_visits`, and prior-coverage counts.
4. Root prior coverage is 100%.
5. C2 is recorded separately. C2 coverage must be reported, not assumed.
6. Parser and Metamon dataset index counts match the shard manifest.

## C2 Decision

The deployed best result was configured with C1+C2, but this does not prove C2
caused the improvement. The current C1-vs-C1+C2 paired ablation remains the
decision source.

- Run 300 paired games at 500ms/P8.
- If the Wilson 95% lower bound is above 50%, include C2 in the next pilot.
- Otherwise generate the pilot C1-only and report C2 as inconclusive.

This avoids blocking all work on C2 while preventing an unsupported causal
claim.

## Canonical Visit Targets

The current Metamon offline trainer consumes replay actions/outcomes, not MCTS
visit distributions. Therefore it is behavior-regularized offline RL, not yet
canonical AlphaZero policy distillation.

Before a promotion-scale round, add a dataset adapter/loss that trains the
policy against logged `mcts_visits` (cross entropy or KL over legal actions).
Until then, label experiments honestly as action-distillation / offline RL.

## Ablation Sequence

Every candidate starts from accepted r1 and changes one variable only.

### Gate 0: Reproducibility

- Load r1 as both sides in a paired 100-game H2H at 500ms/P8.
- Require zero void/unknown games and CI containing 50%.

### Gate 1: Data-only Pilot

Keep `A_rating` and all r1 optimizer/loss settings fixed. Change only data:

```text
70% legacy r1 self-play
20% new strict PFSP data
10% human anchor
```

- Start with one epoch.
- Candidate vs r1: 500 paired games.
- Promote only if Wilson 95% lower bound > 50%.

### Gate 2: Increase Fresh Data

If Gate 1 promotes, use:

```text
45% legacy r1 self-play
45% new strict PFSP data
10% human anchor
```

Again keep all model/loss settings fixed. This measures whether stronger fresh
search data helps beyond the legacy replay buffer.

### Gate 3: KL Anchor Ablation

Only after a data-mix candidate promotes:

- Compare `A_rating` vs `A_rating + KL` with identical data/initialization.
- Promote KL only if paired H2H clears the confidence gate.

### Gate 4: HL-Gauss Ablation

Only after KL has a decision:

- Compare existing value target vs HL-Gauss.
- Never combine this with a data or KL change in the same experiment.

## PFSP-lite Collection

After the C2 decision, strict pilot collection uses a frozen pool:

| Opponent | Base weight |
|---|---:|
| Accepted r1 | 35% |
| Base Kakuna-guided FP | 25% |
| Older accepted checkpoints | 20% |
| Stock FP | 20% |

After a 20-paired-game probe per opponent, favor 40-60% matchups while
retaining a 5% minimum quota for every pool member. Every matchup is role
balanced. The pool, seed, versions, and realized matchup counts belong in the
shard manifest.

## Promotion

For every candidate:

1. 100-game r1-vs-r1 scorer sanity gate.
2. 500 paired candidate-vs-r1 H2H games; Wilson lower bound must exceed 50%.
3. 200 paired games each vs base Kakuna FP and stock FP; no predeclared
   regression.
4. Only then use a short fresh public ladder gate. Stop at the declared RD;
   public ladder is never a data source.

## Stop Rules

- Three fully gated rounds without a held-out gain: pause ExIt and investigate
  leaf evaluation/search.
- A data-integrity failure rejects its shard, not merely the affected game.
- A candidate that fails promotion is archived and never generates the next
  round.
