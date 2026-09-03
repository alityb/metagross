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
