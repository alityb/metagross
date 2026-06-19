# Phase 0 Harness Gate

The N=20 self-play run in `experiments/phase0_baselines.csv` is a smoke check,
not a powered trust gate. It proves the harness can complete Foul Play self-play
and attribute winners by agent slot. It does not rule out a meaningful scoring
bias.

Do not use the harness for Phase 1 promotion decisions until this gate passes.

## Self-Play Gate

Run stock Foul Play self-play at the intended Phase 1 search budget. The default
Foul Play budget is `100ms`, so use `100ms` unless Phase 1 deliberately chooses a
different budget and records why.

Recommended command:

```bash
.venv/bin/python -m eval.run \
  --agent-a foul_play \
  --agent-b foul_play \
  --n-games 200 \
  --paired \
  --foul-play-search-time-ms 100 \
  --game-timeout-seconds 1200 \
  --client-finish-grace-seconds 30 \
  --run-id phase0_harness_selfplay_n200_s100 \
  --json-out experiments/phase0_harness_selfplay_n200_s100.json \
  --append-experiment-log experiments/phase0_baselines.csv
```

Minimum acceptable command if wall-clock is constrained:

```bash
.venv/bin/python -m eval.run \
  --agent-a foul_play \
  --agent-b foul_play \
  --n-games 100 \
  --paired \
  --foul-play-search-time-ms 100 \
  --game-timeout-seconds 1200 \
  --client-finish-grace-seconds 30 \
  --run-id phase0_harness_selfplay_n100_s100 \
  --json-out experiments/phase0_harness_selfplay_n100_s100.json \
  --append-experiment-log experiments/phase0_baselines.csv
```

Pass rule:

- `N_games >= 100` and paired.
- Point estimate for agent A is in `[0.45, 0.55]`.
- Wilson 95% CI contains `0.50`.
- Wilson 95% CI is contained in `[0.40, 0.60]`, ruling out a 60/40-or-worse scorer bias at this sample size.
- No ties or unknown winners unless manually explained from preserved logs.

Fail rule:

- Any of the pass criteria fail.
- The run sticks near 60/40 or worse.
- Winner parsing depends on one Foul Play process but not the other, or preserved logs disagree.

If the gate fails, stop and debug the scoring path before running Phase 1. Do not
interpret Phase 1 A/B numbers through a scorer that fails self-play.

## Phase 1 Budget Note

The existing Phase 0 local baselines used `--foul-play-search-time-ms 25` for
bootstrap throughput. That budget is not automatically representative of stock
Foul Play strength.

Phase 1 must lock the budget before training or A/B testing:

- Primary recommendation: use `100ms`, matching Foul Play's current CLI default.
- If using `25ms` for iteration speed, label it as a low-budget experiment and do
  not generalize the result to default-strength Foul Play.
- If using a higher ladder-like budget, record that number and re-run every stock
  baseline at the same budget.

Before the learned-eval A/B, re-record stock Foul Play baselines at the chosen
budget with the same harness options the A/B will use. For the recommended
`100ms` budget:

```bash
.venv/bin/python -m eval.run \
  --agent-a foul_play \
  --agent-b random \
  --n-games 100 \
  --paired \
  --foul-play-search-time-ms 100 \
  --game-timeout-seconds 1200 \
  --run-id phase0_foul_vs_random_n100_s100 \
  --json-out experiments/phase0_foul_vs_random_n100_s100.json \
  --append-experiment-log experiments/phase0_baselines.csv

.venv/bin/python -m eval.run \
  --agent-a foul_play \
  --agent-b max_damage \
  --n-games 100 \
  --paired \
  --foul-play-search-time-ms 100 \
  --game-timeout-seconds 1200 \
  --run-id phase0_foul_vs_max_damage_n100_s100 \
  --json-out experiments/phase0_foul_vs_max_damage_n100_s100.json \
  --append-experiment-log experiments/phase0_baselines.csv
```

The Phase 1 learned-eval A/B must compare against stock Foul Play at the exact
same search budget, side schedule, server mode, and timeout policy.

## Throughput Warning

The current harness starts fresh Foul Play subprocesses per game and runs games
serially. That is acceptable for Phase 0 validation but expensive for N~1000
Phase 1 experiments. Before large A/B runs, either add controlled concurrency or
keep persistent Foul Play processes alive across games.
