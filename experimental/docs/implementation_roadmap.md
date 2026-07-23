# Implementation Roadmap

## Milestone 0: Freeze Current Baseline

Estimate: 0.5 day.

Deliverables:

- Confirm stock Foul Play Gen9 H2H scorer gate in current environment.
- Record engine build, Showdown commit, Foul Play commit, Python binary, and search budget.

Exit gate:

- Stock Foul Play self-play N=100 or N=200 passes scorer gate.

## Milestone 1: Instrument Think-Time

Estimate: 1 day.

Deliverables:

- Log timestamp when Foul Play sends `/choose`.
- Log timestamp when next actionable request arrives.
- Output per-turn idle time, battle ID, turn, chosen move, timer status.

Gate:

- Report median/p75/p90 idle time over at least 20 local games.

## Milestone 2: Public-State Hash And Cache Shell

Estimate: 1-2 days.

Deliverables:

- Canonical public state hash for Foul Play battle state.
- In-memory speculative cache with TTL and exact-match validation.
- Cache logging: hit, miss, reject, stale.

Gate:

- Unit tests with synthetic state changes.
- Local smoke with no behavior change versus stock Foul Play.

## Milestone 3: Background Draft Worker

Estimate: 2-3 days.

Deliverables:

- Background process/thread started after our `/choose`.
- Draft top-k opponent actions using cheap search or heuristics.
- Prepare likely next-turn determinization bundles or search priors.
- Safe cancellation on next observed state.

Gate:

- No timeouts or protocol failures in N=20 local H2H.
- Nonzero cache hit rate.

## Milestone 4: Speculative Search Acceptance

Estimate: 2-4 days.

Deliverables:

- Use accepted cache as root prior or partial result for next decision.
- Strict fallback to stock Foul Play on miss/reject.
- Equal wall-clock budget accounting.

Gate:

- N=100 paired gen9randombattle versus stock Foul Play.
- Kill if CI includes 50 and no strong latency/search-efficiency gain.

## Milestone 5: Powered Evaluation

Estimate: 2-5 days runtime depending on hardware.

Deliverables:

- N>=1000 paired H2H if N=100 passes.
- Ladder run only after powered H2H.

Gate:

- CI lower bound above 50% versus stock Foul Play.

## Stop Conditions

- Think-time is too small for useful speculation.
- Cache hit rate remains below 10% after simple top-k draft.
- N=100 H2H is not positive.
- Implementation creates voids/timeouts above scorer-gate tolerance.
