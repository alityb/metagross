# Powered priors-vs-vanilla head-to-head (frozen 2026-08-28, before launch)

## Question
The baseline league's m02 cell showed the champion (plain-causal r1 priors)
at 10-14 (41.7%, CI [24.5, 61.2]) against vanilla foul-play — the identical
500 ms x8 pinned-engine search stack minus the neural root priors — despite
a ~7-10 GXE population-level gap (ladder-rank verified). If it replicates
at power, priors-vs-no-priors is the sign-flip's third and cleanest
instance: machinery with a large population edge and no head-to-head edge.

## Design
Runner: league.py (main; mirror enforcement verified) with a single-matchup
config (priors_h2h.json): candidate = plain-causal production_r1_search_first
(priors REQUIRED, port 9023), opponent = foul_play (no prior server),
n_games 200, base_seed 2026082830, output priors_h2h_20260828/run/.
All shared machinery as the league: 500 ms / P8 / 1 thread / cpuct 2.0,
engine pin 79bea0e4, mirrored pairs with consumption-enforced teams,
idempotent resume, stall watchdog via ensure_chain.sh stage D.

## Interpretation (frozen)
Wilson 95% interval on decisive games:
- clearly above 55%: priors DO carry a head-to-head edge; m02 was noise;
  the third-instance claim is dropped.
- containing 50%: no detectable H2H edge at 200-game power against a
  ~7-10 GXE population gap -> sign-flip third instance SUPPORTED; goes in
  the writeup as a measured claim.
- clearly below 45%: priors actively harmful head-to-head — the strongest
  form; reported with emphasis on its strangeness and a replication note.

## Sequencing
Launches automatically after the retrodiction league report exists
(ensure_chain.sh stage D). Developmental, owner-visible.

## RESULT (2026-08-30) — 88-112 (44.0%, CI [37.3, 50.9]): sign-flip third instance SUPPORTED

Frozen reading 2 applies: the interval contains 50 (upper bound 50.9), so
there is NO detectable head-to-head edge at 200-game power — against a
leaderboard-verified 7-10 GXE population gap. The interval also excludes
any meaningful positive edge: at best the priors are worth parity against
the identical search stack they guide past the entire ladder. Point
estimate (44.0%) sits below 45 but the interval does not clearly exclude
45, so the "actively harmful" strong form is NOT claimed. Supplementary:
combined with the two 24-game league cells the matchup totals 106-142
(42.7%, n=248, CI ≈ [36.7, 49.0]) — descriptively below parity.
Priors-vs-no-priors joins causal-vs-stateless and flattening as the third
measured instance of the population/head-to-head sign flip — the cleanest,
because the intervention (policy guidance itself) is the agent's entire
reason to exist.
