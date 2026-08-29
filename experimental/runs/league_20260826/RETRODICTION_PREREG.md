# Hardening run 2 — league retrodiction of the flattening sign-flip (frozen 2026-08-28)

## Question
The league exists because head-to-head and population measurements flipped
signs twice. Instrument validation: would the league have caught the
flattening failure that the 200-game H2H screen (55.5%) missed and the
139-game ladder run (~-8 GXE) exposed?

## Design
Candidate: flattened-causal — plain causal-history serving plus client_env
METAGROSS_PRIOR_TEMP_SCHEDULE={"0":1.02,"10":1.03,"20":1.134,"30":1.689}
(the harness force-gates it to the candidate port; activation_check enforces
ACTIVE in candidate-arm logs only).
Pool, games, machinery: IDENTICAL to the baseline league (league_baseline.json),
including base_seed 2026082610 — same team sets against the same opponents,
so cells compare paired against the baseline reference vector.
Output: experimental/runs/league_20260826/flattened/.

## Frozen predictions (retrodiction SUCCEEDS iff both):
(i)  stateless cell: flattened point estimate >= the baseline reference
     stateless cell (directional reproduction of the 55.5% H2H result), and
(ii) at least one weak-opponent cell (vanilla-foulplay or max-damage):
     flattened clearly below the baseline cell — non-overlapping 95% Wilson
     intervals OR a point deficit >= 10 percentage points (the exploitation
     deficit that explains the ladder harm).
Pattern (i)+(ii) = the league reproduces the sign-flip locally in one
overnight -> instrument VALIDATED as a promotion gate.
Other outcomes (frozen readings): flattened weakly dominates everywhere =
retrodiction FAILS (the ladder harm needs another explanation; the league's
gate claim is weakened — reported as such). Flattened below everywhere =
PARTIAL (catches the harm, misses the H2H win; 40-game cell power noted).

## Status
This is instrument validation, not a promotion attempt; the promotion rule
in PREREGISTRATION.md is unchanged. Launches only after the baseline league
report exists and the root-cause session has released the machine.

## RESULT (2026-08-28) — PARTIAL: the harm is caught, the H2H win is not reproduced

Flattened vector (identical seeds as baseline, all activation-valid):
stateless 19-21 (47.5%, CI [32.9,62.5]) · vs plain-causal 9-11 (45.0%) ·
vanilla-foulplay 8-16 (33.3%) · max-damage 11-1 (91.7%).
Reference: 62.5 / 45.0(self) / 41.7 / 91.7.
Frozen criterion (i) FAILS (stateless cell 47.5 < 62.5, not >=);
criterion (ii) not met at the frozen bar (vanilla deficit 8.4pp < 10pp,
CIs overlap). Per the frozen readings: PARTIAL — the league detects
flattening as weakly dominated (below or equal on every cell, promotion
correctly BLOCKED under the weak-dominance rule) but does not reproduce
the 55.5% H2H win at 40-game power (CI contains both 47.5 and 55.5).
The deployment-relevant half of the retrodiction lands: the gate that
licensed flattening (single-opponent H2H) said ship; the league says
don't; the ladder said don't. Bonus measurement: m01 is the never-run
direct flattened-vs-plain-causal H2H — 45.0%.
Note accumulating across both leagues: causal-prior candidates are a
combined 18-30 (37.5%) vs vanilla foul-play head-to-head — the powered
stage-D run (priors_h2h_20260828) will settle that cell.
