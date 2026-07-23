# Tauros/Kakuna Autopsy Plan

## Goal

Build a postable Pokemon Showdown bot by reverse-engineering why public Metamon policies beat Foul Play in some settings, then using those findings to build a derived agent that is measurably stronger than stock Foul Play and not just a wrapper around borrowed checkpoints.

Target claim:

> I analyzed why a strong offline-RL Pokemon policy beats a public search bot, distilled the useful strategic regimes, and built a stronger agent with documented H2H and ladder results.

This is different from simply laddering TaurosV0 or Kakuna. Tauros/Kakuna are teachers and diagnostic tools. The final promoted agent must include our own measurable contribution.

## Current Evidence

- TaurosV0 beat Foul Play locally in Gen1OU: 15-5, 75%, CI [53.1%, 88.8%].
- TaurosV0 public Gen1OU ladder sample was only medium strength: about 63 GXE / 1420 ELO after mixed runs.
- Kakuna was slower and did not clearly outperform TaurosV0 in small public samples.
- Scratch learned eval and randbats generator-belief attempts did not beat stock Foul Play with significance.

Core tension:

- TaurosV0 appears strong against Foul Play head-to-head.
- TaurosV0 does not currently look like a top public ladder bot.

The project should explain that tension and exploit it.

## Research Questions

1. Where does TaurosV0 beat Foul Play?
2. Where does Foul Play beat TaurosV0?
3. Does Kakuna improve on TaurosV0 in specific state classes, or is it just slower/noisier?
4. Are Tauros/Kakuna strengths general strategic priors, or opponent-specific exploitation of Foul Play?
5. Can we distill the useful parts into a small CPU-speed model or decision module that beats stock Foul Play without running a huge borrowed policy at inference?

## Working Hypotheses

### H1: Tauros has better long-horizon Gen1OU strategic priors

Tauros may better handle positions where immediate material/eval is misleading:

- Sleep pressure and sleep sack timing.
- Paralysis spreading and speed-control value.
- Explosion trades.
- Chansey/Reflect/Soft-Boiled loops.
- Tauros endgames.
- Preserving checks instead of taking immediate damage value.

### H2: Tauros is exploitable on public ladder because it is policy-only

Tauros may beat Foul Play because it avoids Foul Play's hand-eval blind spots, but lose ladder EV because it lacks tactical verification or adapts poorly to unusual human lines.

### H3: Kakuna may be better in narrow regimes

Kakuna's public result was weak in a tiny sample, but it might still outperform Tauros in certain decision classes. We should compare by state bucket, not only aggregate win rate.

### H4: A distilled disagreement model can beat both raw Tauros and stock Foul Play

Instead of copying Tauros everywhere, train a model to identify when Tauros is likely right over Foul Play. Use it selectively.

## Deliverables

### Dataset

A local H2H trace dataset containing:

- Battle format.
- Battle ID and turn.
- Full observable state.
- Legal actions.
- TaurosV0 action.
- Kakuna action, when available.
- Foul Play action and visit distribution, when available.
- Final winner.
- Side to move, active Pokemon, HP, status, boosts, revealed moves, remaining Pokemon.
- Derived features: material, status advantage, sleep availability, paralysis spread, endgame class, speed control, boom availability.

### Analysis

A report that buckets disagreement positions and answers:

- In which buckets does Tauros outperform Foul Play?
- In which buckets does Foul Play outperform Tauros?
- Which buckets are common enough to matter for ladder EV?
- Which buckets are simple enough to distill or hard-code safely?

### Agent

At least one derived agent:

- `tauros_distilled_policy`: small model trained to imitate Tauros/Kakuna.
- `tauros_disagreement_gate`: choose Tauros-like action only when a learned model predicts Tauros is likely better than Foul Play.
- `tauros_foul_hybrid`: Foul Play base with a distilled RL prior or value model, not a direct runtime dependency on Tauros.

### Evaluation

- Derived agent vs stock Foul Play, paired H2H.
- Derived agent vs TaurosV0, paired H2H.
- Derived agent vs Kakuna if runtime permits.
- Public ladder only after local H2H passes a powered gate.

## Phase 0: Make Trace Collection Real

Goal: collect decision traces from TaurosV0, Kakuna, and Foul Play in comparable states.

### Steps

1. Confirm current local H2H runner can run TaurosV0 vs Foul Play deterministically enough for N >= 100.
2. Add decision logging for TaurosV0:
   - Log the action chosen.
   - Log action probabilities if accessible.
   - Log inference temperature/checkpoint/config.
3. Extend Foul Play decision logging:
   - Already captures MCTS visit distribution in some paths.
   - Ensure it works in Gen1OU H2H against Tauros.
   - Include selected move and considered move distribution.
4. Save all traces as JSONL shards under `data/traces/`.
5. Add a schema file for trace rows.

### Trace Row Schema

```json
{
  "run_id": "tauros_fp_trace_n200_...",
  "format": "gen1ou",
  "battle_id": "...",
  "turn": 17,
  "agent": "TaurosV0",
  "opponent": "FoulPlay",
  "side": "p1",
  "chosen_action": "move thunderwave",
  "legal_actions": ["move thunderwave", "move psychic", "switch chansey"],
  "policy": {"move thunderwave": 0.62, "move psychic": 0.21},
  "foul_play_visits": {"move thunderwave": 0.12, "move psychic": 0.74},
  "state_features": {},
  "state_string": "optional engine state string",
  "winner": "TaurosV0",
  "label": 1
}
```

### Exit Criteria

- N >= 100 TaurosV0 vs Foul Play games with turn-level decisions for both sides.
- No more than 5% missing decision rows.
- Reproduced TaurosV0 edge or explained why the original 15-5 was noise.

### Kill Criteria

- Cannot reliably log Tauros decisions.
- TaurosV0 no longer beats Foul Play in N >= 100 local H2H.

## Phase 1: Disagreement Autopsy

Goal: identify the actual state classes where Tauros and Foul Play differ and where those differences matter.

### Definitions

A disagreement state is a turn where:

- Tauros chosen action differs from Foul Play's highest-visit action, or
- Foul Play's top action has less than 60% visit share and Tauros chooses a different high-plausibility action.

### Buckets

Start with interpretable Gen1OU buckets:

- Opening sleep race.
- Sleep sack selection.
- Paralysis spread.
- Tauros reveal and Tauros endgame.
- Chansey mirror.
- Snorlax trade.
- Explosion opportunity.
- Recover/Soft-Boiled loop.
- Sacrifice decision.
- Forced switch.
- Low-HP endgame.
- Statused active vs healthy switch.

### Metrics

For each bucket:

- Frequency per game.
- Tauros win rate when disagreement occurs.
- Foul Play win rate when disagreement occurs.
- Delta in final outcome after controlling for side and matchup if possible.
- Average turn number.
- Average material/status state.

### Output

Generate:

- `experiments/tauros_autopsy/<run_id>/bucket_summary.csv`
- `experiments/tauros_autopsy/<run_id>/top_disagreements.md`
- `experiments/tauros_autopsy/<run_id>/examples/` with selected replay snippets.

### Exit Criteria

- At least 3 recurring disagreement buckets with N >= 30 examples each.
- At least 1 bucket where Tauros appears meaningfully better than Foul Play.

## Phase 2: Kakuna Comparison

Goal: determine whether Kakuna contributes distinct knowledge or can be ignored.

### Steps

1. Run smaller N due runtime cost:
   - Kakuna vs Foul Play, N=50.
   - Kakuna vs TaurosV0, N=50 if feasible.
2. Log Kakuna decisions in the same schema.
3. Compare three-way disagreement:
   - Tauros = Kakuna != Foul Play.
   - Kakuna != Tauros = Foul Play.
   - all three disagree.
4. Identify if Kakuna wins any specific bucket.

### Exit Criteria

- Decision whether Kakuna is useful as a teacher.
- If useful, define which bucket(s) Kakuna supervises.

### Kill Criteria

- Kakuna is too slow for trace collection.
- Kakuna does not add any bucket-level signal beyond Tauros.

## Phase 3: Distill the Teacher

Goal: build our own small model from Tauros/Kakuna behavior and outcomes.

### Candidate Models

#### A. Behavior Cloning Policy

Train a small model to predict Tauros action from state features.

Pros:

- Simple.
- Produces our own standalone policy.

Cons:

- May copy Tauros mistakes.
- May be weaker than Tauros.

#### B. Disagreement Gate

Train a binary model:

> When Tauros and Foul Play disagree, should we follow Tauros?

Pros:

- Uses Foul Play as a strong base.
- Only needs to learn high-value overrides.

Cons:

- Less pure than a standalone policy.

#### C. Bucket Specialists

Train or hard-code small specialists for specific buckets:

- Sleep sack model.
- Paralysis spread model.
- Explosion trade model.
- Tauros endgame model.

Pros:

- Interpretable and postable.

Cons:

- May not cover enough turns.

### Recommended First Model

Build the disagreement gate first.

Training examples:

- Include only disagreement states.
- Positive if following Tauros' action correlates with eventual win or search-improved replay outcome.
- Negative if Tauros line loses or Foul Play line is clearly better.

Features:

- Active species for both sides.
- HP fractions.
- Statuses.
- Known moves.
- Team alive counts.
- Sleep availability.
- Paralysis count.
- Boosts.
- Foul Play visit distribution entropy.
- Foul Play top move margin.
- Tauros action type: attack, status, switch, rest/recover, explosion.

### Exit Criteria

- Offline heldout accuracy above a majority baseline.
- Calibration curve is not pathological.
- The model overrides Foul Play on a small fraction of turns, ideally 5-20%, not everywhere.

## Phase 4: Build the Derived Agent

Goal: deploy a bot that is ours and can beat Foul Play.

### Agent Design

Base policy:

- Stock Foul Play.

Override module:

- If current state is a known Tauros-positive bucket and disagreement gate confidence >= threshold, choose Tauros-derived action.
- Otherwise use Foul Play.

Important rule:

- Do not call TaurosV0 at runtime in the final agent unless explicitly testing an oracle hybrid. Runtime Tauros calls make the final bot less ours and slower.

### Threshold Sweep

Sweep confidence thresholds:

- 0.55
- 0.60
- 0.65
- 0.70
- 0.80

Measure:

- Override rate.
- H2H win rate vs stock Foul Play.
- H2H win rate vs TaurosV0.
- Blunder rate in reviewed games.

### Exit Criteria

- N=100 paired vs stock Foul Play with point estimate >= 55% and no implementation voids.
- Then N=1000 paired if N=100 is promising.

### Promotion Criteria

- N >= 1000 paired vs stock Foul Play.
- CI lower bound above 50%.
- Public ladder ELO/GXE above stock Foul Play baseline under the same format and account discipline.

## Phase 5: Ladder and Writeup

Only ladder after local H2H passes.

### Ladder Protocol

1. Fresh account for derived agent.
2. Fresh account or historical baseline for stock Foul Play.
3. Same format, same search budget, same time window if possible.
4. Run enough games for GXE deviation to shrink.
5. Report ELO, GXE, W/L, and Glicko deviation.

### Post Structure

1. Problem:
   - Search is strong but misses strategic RL priors.
2. Teacher result:
   - TaurosV0 beats Foul Play locally, but does not dominate ladder.
3. Autopsy:
   - Show bucket-level disagreement analysis.
4. Method:
   - Distilled Tauros/Kakuna disagreement gate or bucket specialists.
5. Results:
   - H2H vs Foul Play.
   - H2H vs Tauros/Kakuna.
   - Public ladder ELO/GXE.
6. Ablations:
   - No gate.
   - Tauros-only teacher.
   - Kakuna-only teacher.
   - Per-bucket performance.
7. Limitations:
   - Format-specific.
   - Teacher dependence.
   - Ladder variance.

## Experiment Gates

### Gate A: Reproduce Tauros Edge

- TaurosV0 vs Foul Play, Gen1OU, N=100 paired if possible.
- If Tauros is not above 50%, stop and diagnose before any distillation.

### Gate B: Find Buckets

- At least 3 repeatable disagreement buckets.
- At least 1 bucket has positive Tauros signal.

### Gate C: Offline Gate Model

- Heldout disagreement prediction beats baseline.
- Override rate is controlled.

### Gate D: Local H2H

- Derived agent vs stock Foul Play, N=100.
- Continue only if point estimate is >= 55% or bucket analysis strongly explains a neutral result.

### Gate E: Powered H2H

- Derived agent vs stock Foul Play, N>=1000.
- CI lower bound above 50%.

### Gate F: Ladder

- Public ELO/GXE above stock Foul Play baseline.

## Immediate Implementation Plan

### Step 1: Add Tauros Decision Trace Logging

Files likely involved:

- `external/metamon/metamon/rl/evaluate/__main__.py`
- `external/metamon/metamon/rl/metamon_to_amago.py`
- `external/metamon/metamon/env/wrappers.py`
- `scripts/patch_metamon_cpu_public_ladder.py`

Output:

- `data/traces/tauros_vs_foulplay/<run_id>.jsonl`

### Step 2: Add Foul Play Trace Export

Files likely involved:

- `scripts/run_foul_play.py`
- `eval/run.py`

Use existing `METAGROSS_DECISION_LOG` as the base.

Output:

- Foul Play visit distributions aligned to battle and turn.

### Step 3: Run TaurosV0 vs Foul Play N=100

Record:

- Win rate and CI.
- Trace row counts.
- Missing row rate.

### Step 4: Build Autopsy Notebook/Script

Prefer script over notebook for reproducibility:

- `analysis/tauros_autopsy.py`

Outputs:

- bucket summary CSV.
- top disagreements markdown.

### Step 5: Decide First Distillation Target

Pick the highest-value bucket by:

- frequency times effect size.
- interpretability.
- ease of implementation.

Do not train a broad model until the first bucket is understood.

## Non-Goals

- Do not ladder TaurosV0 and call it ours.
- Do not continue Metamon environment plumbing unless it serves trace collection.
- Do not build another broad learned eval from scratch without a teacher signal.
- Do not claim progress from N<100 noisy H2H runs.
- Do not use AWS for public laddering; Showdown locks cloud IPs.

## What Success Looks Like

Minimum postable result:

- Derived agent beats stock Foul Play in local paired H2H with significance.
- Derived agent reaches higher public ladder ELO/GXE than stock Foul Play in the same format.
- Writeup includes actual disagreement analysis showing why the method works.

Strong result:

- Derived agent beats Foul Play and TaurosV0.
- Kakuna contributes a clearly identified specialist signal.
- Public ladder result is meaningfully above previous stock Foul Play baseline.

Best result:

- The distillation method transfers from Gen1OU to Gen9 randbats or Gen9 OU.
- The post becomes about a general recipe: offline-RL teacher autopsy plus selective distilled overrides on top of search.
