# Powered mirrored replications (frozen 2026-09-03, before launch)

Both original screens ran before mirror enforcement existed; both were
decisive or supported but unpaired. Each replays at 200 games through the
fixed league harness (verified mirroring, activation checks, idempotent).

## R1 — Gate B (Gumbel completed-Q) replication
Candidate: causal + METAGROSS_GUMBEL_ROOT_PORTS (activation line
"GUMBEL_ROOT ACTIVE" mandatory in candidate arm only). Opponent: plain
causal. 200 games, base_seed 2026090301.
Frozen readings (Wilson 95%): clearly below 45% -> harmful CONFIRMED at
power; containing 50% -> original 10-40 overstated by small n (audit
revised to "harmful unconfirmed"); clearly above 55% -> original
CONTRADICTED (investigate seed/env differences before any claim).

## R2 — Flattening H2H replication (mirrored)
Candidate: causal + frozen temperature schedule (ACTIVE line mandatory).
Opponent: legacy-stateless. 200 games, base_seed 2026090302.
Frozen readings: clearly above 50% -> the 55.5% H2H win replicates
mirrored (instance-2 H2H leg confirmed); containing 50% -> H2H win
weakens toward parity (consistent with the retrodiction's 47.5%; writeup
revised to "H2H at-worst-parity vs population harm"); clearly below 45%
-> original contradicted; sign flip loses its H2H-win leg (reported).

Sequencing: R1 then R2, single wrapper loop, no cron. ~40h total.

## AMENDMENT (2026-09-03, before any valid R1 game): client correction
Initial R1 launch was INERT: production_r1_search_first runs the
production client, which lacks the Gumbel hook (the original Modal lanes
used the experimental harness client, where activation and ack-409s were
verified). Caught by activation audit at 30 games (measured plain
self-mirror 11-19; discarded). R1 re-specified with agent
foul_play_root_priors on BOTH arms (harness client, hook present) —
faithful to the original lanes' stack. Chain now enforces activation
fail-closed: no ACTIVE line within 25 min kills the stage.
