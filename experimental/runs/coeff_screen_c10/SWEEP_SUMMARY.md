# Coefficient sweep summary

| Coefficient | Winrate vs r1 | Games | Verdict |
|---|---|---|---|
| 0.1 | 0.529 | 500 | Failed gate (best arm) |
| 0.3 | 0.450 | 101 | Rejected (SPRT accept-h0) |
| 1.0 | 0.407 | 27 | Rejected (futility stop) |

## c_puct sweep
| c_puct | Winrate vs r1 | Games | Verdict |
|---|---|---|---|
| 1.0 | 0.509 | 106 | No improvement over 2.0, stopped for futility |
| 1.5 | — | — | Deprioritized |
| 2.5 | — | — | Deprioritized |

## Conclusion
- coeff 0.1 is at or near the sweet spot; higher coefficients degrade the
  policy monotonically (dose-response points wrong way).
- c_puct 1.0 showed no benefit over the default 2.0.
- This dataset's distillation ceiling with the current approach is ~53%.
- Next levers: ExIt round 2 (compound the +3% with a new collection round
  using the 6k candidate) and/or learned-value-in-search.
