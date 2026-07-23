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
