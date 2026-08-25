# Experimental Archive

This directory preserves the research workspace that produced and tested the
accepted r1 agent. Nothing here is part of the production runtime.

- `docs/`: historical plans, surveys, reviews, and superseded setup notes.
- `runs/`: positive, negative, partial, smoke, and interrupted experiments.
- `src/`: training, evaluation, belief, search, and candidate-agent code.
- `engine/`: experimental engine forks and build tooling.
- `configs/`: training and evaluation schedules.
- `data/`: replay, self-play, and distillation datasets.
- `external/`: ignored local third-party or generated research dependencies.

The append-only research history is `runs/iteration_log.md`. Paths inside old
artifacts intentionally retain their historical names and may not be directly
runnable after archival.

The governing plan for the next research phase is
[`docs/alpha_zero_research_program.md`](docs/alpha_zero_research_program.md).
The executable first step is the
[`docs/online_rl_smoke.md`](docs/online_rl_smoke.md) staged collection and
guarded continuation runbook.

The current outcome-grounded residual branch supersedes that executable step.
Its frozen protocol and negative/positive evidence are in
`runs/outcome_residual_scale_20260814/`. The next admitted action is the local,
CPU-only 500-game schema-6 capture pilot:

```bash
experimental/src/scripts/run_fresh_schema6_pilot_500.sh \
  experimental/runs/schema6_fresh_pilot_500_<date> 8040 fresh
```

The wrapper fixes the 300/100/100 peer/direct-R1/unguided mixture, distinct seed
domains, exact checkpoint and causal-history contract, one observer per
physical battle, ≥95% capture audit, and atomic resume. It does not train or
open the 5,000-game scale gate; that still requires at least 50 roots after the
frozen 20k/50k four-way agreement screen.
