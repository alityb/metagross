# Experiments

Append one row per experiment run using the schema from `AGENTS.md` section 6.
Do not edit historical rows except to fix malformed CSV produced by tooling.

Schema:

```text
run_id | date | phase | format | change (ONE var) | baseline | N_games | winrate | CI95 | ladder_elo | gxe | belief_brier | decision(advance/iterate/rollback) | notes
```

Phase 0 baseline rows are recorded in `phase0_baselines.csv`.
