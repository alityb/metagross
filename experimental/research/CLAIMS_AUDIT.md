# Claims audit (2026-09-01) — every major claim, its evidence, its checks, its residual risk

Discipline: for each headline claim, list the strongest attack, the check
run against it, and what could still overturn it. Updated as checks land.

| # | Claim | Primary evidence | Adversarial checks run | Residual risk |
|---|---|---|---|---|
| 1 | Elite human level; reproduces | 92.4–92.7 & 91.7 at RD-25; hashed rank snapshots (#105/#246); peak Elo 2320; scalps of then-#2/#6 | Rank claims snapshot-dated; decay/drift quantified; close-out snapshots hashed | Cross-era pool drift (stated); no verified bot leaderboard (stated) |
| 2 | History-induced sharpening (not self-conditioning, not temperature) | Ablation on 735 identical states, exact replay (0.0 diff); masked ratio 0.905; same-state stateless control | State selection (identical rows); mask narrowing (normalized entropy diverges harder); temperature (top-1 agree 65–70%, residual KL ≈0.13 after optimal τ) | Single checkpoint/single game; OOD nature of masked inputs (frozen caveat) |
| 3 | Completed-Q harmful under determinization | 10-40 (p = 1.2e-5 one-sided), 37% flips | Exact p computed; mechanism (strategy fusion) argued from telemetry | 200-game MIRRORED replication RUNNING (replications_20260903 R1, frozen readings incl. the contradicting one) |
| 4 | Flattening: wins H2H, loses population | 55.5% [48.6,62.2] (n=200); ladder 81.7 vs 89.4; league retrodiction rejects it | Mirror-integrity audited (screens unpaired but unbiased); league retro on verified-mirrored games; stratification SOFTENED after endogenous-matchmaking check (mix-standardization run; key cell Fisher p=0.032) | 200-game MIRRORED H2H replication RUNNING (replications_20260903 R2); cross-day pools (ratings invariant) |
| 5 | Priors: no H2H edge vs own skeleton | 88-112 (44.0% [37.3,50.9]), n=200, verified-mirrored | Powered replication of the league cell; activation/enforcement verified | Population side MEASURED (2026-09-03): vanilla converged ≈87.8 GXE → gap ≈4 GXE, not folklore's 7–10; instance 3 revised to measured form |
| 6 | Sign flip (H2H vs population anti-correlation), 3 instances | Claims 4+5 + causal/stateless parity-vs-gap | Each instance individually checked above | Instance 3 conditional on claim 5's population measurement; instances share one domain/one stack (external validity stated, PokeAgent cited) |
| 7 | Search overrides policy 48.1% | 3,962 telemetered decisions (peak push, stateless mode) | — | REPLICATED (2026-09-03): 40.9% in both arms of a second dataset (n≈14.8k) — mode-invariant; claim stated as 41–48% |
| 8 | Humans match no sampled world | 498 states, 36/31/33 | Held as discussion-grade, never a headline | Observational; modest n |

Standing rules: ratings > stratifications (matchmaking-invariant); every
rank snapshot dated and hashed; "mirrored" only when consumption-verified;
observable activation always; the reading most embarrassing to us is
written into each prereg before launch.
