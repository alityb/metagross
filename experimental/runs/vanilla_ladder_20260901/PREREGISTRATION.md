# Vanilla foul-play ladder run (frozen 2026-09-01, before launch)

## Question — the missing pillar of sign-flip instance 3
The claim "the priors are worth 7-10 GXE at population level" has never
been measured by this project: it rests on community folklore about
vanilla Foul Play's ladder strength. The H2H side (44.0% [37.3,50.9]) is
measured at power; the population side must be too.

## Design
Agent: the harness `foul_play` agent (identical pinned engine 79bea0e4,
500 ms / P8 / 1 thread — the exact unguided stack from the H2H), laddering
live via eval.run --mode ladder on the roguefan55 account (its flattened
result is closed; rating restart from ~1780 Glicko noted). Bounded: up to
150 games or 14 h, resilient relaunch loop, no cron. Ratings polled to
ratings.jsonl; protocol via foul-play logs.

## Frozen readings (converged or near-converged GXE, RD <= ~28)
- 80-87: the 7-10 (or at least ~5+) GXE population gap is CONFIRMED as
  measured; sign-flip instance 3 stands in full.
- 87-90: gap smaller than folklore (2-5 GXE); instance 3 weakens to
  "population edge without H2H edge" — reported with the revised number.
- >= 90: the population-gap claim DIES; instance 3 collapses to "no edge
  anywhere" and the writeup is corrected accordingly. (This is the
  outcome that would most embarrass us; it is written down first.)
Rank/user snapshot fetched + hashed at close.

## RESULT (2026-09-03) — 87.7-87.9 GXE converged: reading 2, GAP WEAKENS

Vanilla foul-play (identical pinned engine/budget, zero priors) converged
at GXE ≈87.8, RD 25.0 over ~200 own games (stable 86.7-87.9 across the
final 100+). Frozen reading 2 applies (87-90): the priors' population
edge is ≈3.9 GXE (91.7 vs 87.8) — real, but roughly half the folklore
7-10. Sign-flip instance 3 is REVISED to its measured form: a ≈4 GXE
population edge alongside no head-to-head edge (44.0% [37.3, 50.9]) —
the direction of the flip stands; its magnitude shrinks. Also notable:
the unguided stack alone is a ~top-150-caliber agent (peak session Elo
2381 exceeded our own account peak; Elo is streak-sensitive, GXE is the
calibrated figure). Close-out user snapshot hashed (CLOSEOUT.json).
