# Benchmarking Plan

## Principles

- One variable per experiment.
- Paired games and both sides whenever possible.
- Always report N, decisive games, voids, win rate, Wilson CI, format, search budget, engine build.
- No promotion if CI includes 50%.
- Public ladder only after local H2H passes.

## Baseline Revalidation

Before any speculative-search run:

- Stock Foul Play vs stock Foul Play.
- Format: `gen9randombattle`.
- Search budget: same as planned MVP, initially 25ms and 100ms if time permits.
- N>=100 smoke gate; N=200 preferred.
- Pass criteria: point estimate in [45%, 55%], CI contains 50%, no unexplained voids.

## Measurement Benchmarks

For think-time/speculation instrumentation:

- N=20 local H2H stock Foul Play vs stock Foul Play.
- Record idle time between our `/choose` and next request.
- Record foreground decision latency.
- Record search iterations if available.
- Report median, p75, p90, max.

## MVP Gate

Speculative search versus stock:

- Format: `gen9randombattle`.
- N=100 paired.
- Same total foreground decision budget.
- Record cache hit/reject/miss rates.
- Record voids and timeouts.
- Kill if win rate is not positive or CI includes 50 without strong efficiency gain.

## Powered Gate

Only if N=100 is promising:

- N>=1000 paired.
- Same engine/build/search budget.
- CI lower bound must exceed 50%.

## Ladder Gate

Only if powered H2H passes:

- Fresh account or clearly separated account state.
- Same format and budget as H2H.
- Enough games for Glicko deviation to shrink.
- Report ELO, GXE, W/L, Glicko deviation, and time window.

## Metrics To Log

- Battle ID.
- Turn.
- Agent version.
- Engine build and feature flags.
- Search budget.
- Candidate moves and final action.
- Cache state: hit, miss, reject, stale.
- Time spent in background and foreground search.
- Opponent think-time.
- Winner.

## Invalid Run Criteria

- More than 5% voids.
- Unknown winners.
- Engine crashes.
- Wrong generation build.
- Mixed code versions in one run.
- Unpinned or undocumented model/engine state.
