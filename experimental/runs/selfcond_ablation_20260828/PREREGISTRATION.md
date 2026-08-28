# Hardening run 1 — self-conditioning ablation (frozen 2026-08-28, before any implementation run)

## Question
The late-game entropy collapse (0.96 -> 0.58 nats by turn 30+, replicated
twice) occurs only in the causal-history serving view of checkpoint
policy_epoch_5.pt (sha c6a4c0f5...). Is the driver (a) the agent's OWN past
actions in context (self-conditioning), or (b) context length per se?

## Part 1a — offline mechanism probe ($0, no games)
Corpus (frozen): the 699 paired causal/stateless decisions of
r1_causal_vs_stateless_screen_20260814 — the same corpus that calibrated the
temperature schedule, replayed via the dual-variant offline rqid-correlation
contract (exact offline replay, max abs diff 0.0).

Conditions, all on identical states:
1. full causal-history        (sanity: must reproduce the known collapse)
2. truncate K in {5,10,20,40} (METAGROSS_HISTORY_TRUNCATE_STEPS, exists)
3. own-action-masked, full length (NEW serving variant: the agent's own past
   action entries in the trajectory are replaced by the step-0 null/pad
   action; opponent-observable history untouched)
4. legacy-stateless           (reference flat profile)

Metric: mean prior entropy by turn bucket (0-9/10-19/20-29/30+); collapse
magnitude C = bucket(0-9) entropy minus bucket(30+) entropy.

Frozen predictions:
- H-self (self-conditioning drives it): masked-arm C <= 33% of full-causal C,
  i.e. masked tracks the stateless profile; truncation reduces C
  monotonically with K.
- H-length (context length drives it): masked-arm C >= 66% of full-causal C.
- Between 33% and 66% = mixed drivers; report as mixed, no winner declared.

Validity (mandatory):
- Sanity condition must reproduce the known collapse (C within +-0.15 nats of
  the 2026-08-14 measurement) else the probe is INVALID (replay broken).
- The mask variant is env-gated (METAGROSS_MASK_OWN_ACTIONS=1), absent env =
  byte-identical, malformed = fail-closed raise, active = one-time
  "OWN_ACTION_MASK ACTIVE" line; unit-verified before the run.
- OOD caveat (frozen): masked inputs are off-distribution. Interpretation is
  limited to entropy structure, never strength. If the masked arm is
  degenerate (>1% NaN decisions, or near-uniform everywhere: mean entropy
  > 0.95*ln(13) in ALL buckets), the masked arm is INCONCLUSIVE and the
  self-conditioning claim rests on the truncation gradient alone.

## Part 1b — truncation outcome test (local, ~1 overnight, $0)
Gate C's never-run H2H, unchanged from dc_gates_modal_20260822 except venue:
causal+truncate-20 (arm A, prior-server env) vs plain causal (arm B).
Mirrored pairs, fail-fast, 100 games, mirror seed 2026082810, username
prefix gcl1, 500 ms / P8 / 1 thread / cpuct 2.0, engine pin 79bea0e4,
production servers 9023/9024, HISTORY_TRUNCATION ACTIVE line mandatory in
arm-A server log and absent from arm B.
Interpretation (frozen): Wilson interval containing 50% with point >= 47% =
strength preserved -> redundancy SUPPORTED (long history adds nothing that
search does not reconstruct; with 1a's H-self this completes the
self-conditioning story: history's marginal content is the pathology).
Clearly below 45% = long history carries real strategic signal -> redundancy
REFUTED (reported as-is; it complicates the redundancy claim and sharpens
the self-conditioning one: the same history that helps also miscalibrates).

## Sequencing
Nothing launches while the eval-harness root-cause session owns the machine.
Order after release: 1a (cheap, offline) -> baseline league completes ->
1b. Implementation of the mask variant + probe script may proceed anytime
(code only, unit tests only).
