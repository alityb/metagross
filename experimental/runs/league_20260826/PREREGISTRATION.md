# League harness v1 — frozen pool and promotion rule (2026-08-26)

Motivated by the campaign's twice-observed population-vs-H2H sign flip: a
single-opponent H2H cannot license deployment. The league evaluates any
candidate serving configuration against a FROZEN pool spanning strength and
style, reporting the per-opponent vector, mirrored pairs throughout.

## Frozen pool (v1)
| Opponent | What it probes | Games |
|---|---|---|
| stateless (r1, legacy input) | strong systematic peer | 40 |
| plain-causal mirror | self-play sanity (expect ~50%) | 20 |
| vanilla foul-play (no priors) | mid-strength heuristic style | 24 |
| max_damage | weak hyper-aggressive (exploitation probe) | 12 |

## Promotion rule (frozen)
A candidate serving change is promotable only if its vector weakly dominates
the plain-causal reference vector: no opponent cell more than 1 CI-width
BELOW the reference cell, and at least one cell clearly above. The
max_damage and vanilla-foulplay cells are the exploitation guard — the cells
temperature flattening would have failed. Ladder confirmation remains the
final gate but is only earned by a league pass.

## Validity
Observable activation enforced per matchup: intervention ACTIVE line present
in candidate-arm logs, absent from opponent-arm logs, else the matchup is
graded INVALID. Idempotent: re-running skips completed matchups.

## Reference vector
The first run is candidate = plain-causal-r1 itself (its stateless cell
should reproduce ~48% from the August mirror; the self-mirror cell ~50%).
This vector is the standing baseline for all future candidates.
