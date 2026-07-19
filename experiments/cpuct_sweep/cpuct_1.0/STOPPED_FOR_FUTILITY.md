# c_puct 1.0 arm — stopped for futility

- Match: mcts_v3_full_6k @ c_puct 1.0 (agent A) vs randbats_exit_r1 @ c_puct 2.0
  (agent B), paired local games at 500ms/P8, SPRT p0=0.50 p1=0.58 (cap 300).
- Stopped manually at 106 games: **54-52 (0.509)**, 0 voids.
- SPRT had not formally decided (LLR ~ -1.1), but the arm showed at best parity
  with the same candidate's 0.529 at c_puct 2.0 (500-game gate). The hypothesis
  ("sharper distilled prior prefers lower exploration") is unsupported at
  c_puct 1.0.
- Deviation note: manual futility stop before SPRT decision, to reallocate the
  single local eval box to higher-value coefficient-variant screens
  (mcts_v3_full_6k_c03 / _c10).
- Arms 1.5 / 2.5 deprioritized pending coefficient screen results.
