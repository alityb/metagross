# Eval Harness

Run paired head-to-head evaluations on `gen9randombattle` against the local
Pokemon Showdown server by default.

Fixed Phase 0 agents:

- `random`: poke-env `RandomPlayer`
- `max_damage`: poke-env `MaxBasePowerPlayer`
- `foul_play`: stock Foul Play at the pinned checkout documented in `SETUP.md`

Examples:

```bash
.venv/bin/python -m eval.run --agent-a foul_play --agent-b random --n-games 20 --paired
.venv/bin/python -m eval.run --agent-a foul_play --agent-b foul_play --n-games 20 --paired
.venv/bin/python -m eval.run --agent-a foul_play --agent-b random --n-games 2 --log-dir external/eval-debug
```

The harness reports agent A's win rate and a Wilson 95% confidence interval.
With `--paired`, `N` must be even; each pair runs once with A challenging B and
once with B challenging A.

Use `--log-dir` to preserve per-game Foul Play subprocess logs for debugging.

Before using the harness for Phase 1 decisions, run the powered self-play gate in
`AGENTS.md` §6. The N=20 self-play row in `SETUP.md` is only a smoke check.

Live ladder support is available for credentialed runs:

```bash
.venv/bin/python -m eval.run --mode ladder --server live --agent foul_play --username USER --password PASS --n-games 10
```
