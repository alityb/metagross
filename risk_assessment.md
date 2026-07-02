# Risk Assessment

## Highest Risks

### Speculative Cache Misses Too Often

Risk: opponent actions and random outcomes make precomputed states rarely match.

Impact: no performance gain.

Mitigation: measure cache hit rate before deep implementation; start with lightweight determinization/prep caches.

### Wrong State Hash Or Stale Cache

Risk: accepting stale speculative results produces illegal or bad moves.

Impact: catastrophic ladder performance and invalid experiments.

Mitigation: strict public-state hashing, TTL, legal-action validation, fallback to stock.

### Foul Play Monkey-Patch Fragility

Risk: additional patches interact badly with existing decision logging, randbats belief, Tauros gates, or protocol patches.

Impact: voids, hidden behavior changes, irreproducible results.

Mitigation: isolate the speculative-search variant in a minimal code path; do not combine with old failed patches.

### Engine Build Drift

Risk: wrong `poke-engine` generation feature is installed.

Impact: crashes or invalid mechanics.

Mitigation: dedicated venvs per generation; log build features and smoke on known species.

### Public Ladder Cloud Restrictions

Risk: AWS/cloud IPs are locked by Showdown.

Impact: cannot use cloud for public ladder.

Mitigation: use AWS only for local H2H/data generation; ladder locally.

## Medium Risks

### Search Budget Accounting Is Unfair

Risk: speculative search uses extra time that stock baseline does not get.

Impact: invalid comparison.

Mitigation: report both foreground and wall-clock budgets; compare under realistic live constraints.

### More Compute Does Not Help Bad Priors

Risk: speculative search allocates compute to wrong branches.

Impact: neutral or worse performance.

Mitigation: start with draft policies derived from stock Foul Play’s own cheap search, not a weak learned student.

### High Variance Masks Effects

Risk: N=100 is too small for small gains.

Impact: false negatives or false positives.

Mitigation: use N=100 as kill/promote-to-N1000 gate, not final claim.

## Low Risks

### Documentation Drift

Risk: dossier becomes stale after implementation.

Mitigation: update experiment log and design docs at each gate.

### Large Artifacts

Risk: traces/pools become too large for git.

Mitigation: keep raw data ignored; commit compact summaries and checksums.

## Current Project Risk Summary

The main risk is open-ended experimentation without a hard gate. The requested one-build-one-gate shield discipline is correct. Future speculative search should follow the same pattern: measure, build once, gate, kill fast if neutral.
