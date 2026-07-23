# Gate result: mcts_v3_full_6k vs randbats_exit_r1 — NOT PROMOTED

- Candidate: `mcts_v3_full_6k` (v3 MCTS distillation, 6,000 steps, batch 8, coeff 0.1,
  A10G, full 175,319-target verified dataset), served via prior server, deployed
  stack `foul_play_root_priors_opp` @ 500ms/P8/1 thread, c_puct 2.0 both sides.
- Baseline: `randbats_exit_r1` epoch 5 (accepted, 92.4-92.7 GXE).
- Protocol: 500 paired local games, promotion requires >=272 wins
  (Wilson 95% lower bound > 0.50).

## Result
- 500 games: candidate **264-235** (0.5291), 1 void.
- Wilson 95% lower bound: **0.4852** (< 0.50 required).
- **NOT PROMOTED** (needed 272 wins, got 264).

## Notes
- Run was split across two processes after a local Showdown/server crash at
  game 330 (`result.json.progress.jsonl` + `../mcts_v3_6k_gate_resume2/`);
  the aborted first resume attempt (170 all-void games, dead Showdown) is in
  `../mcts_v3_6k_gate_resume/` and is excluded.
- Zero gameplay errors/crashes across the 500 counted games.
- Screens before the gate: 1k arm 11/20 (55%), 3k arm 12/30 (40%),
  6k arm 20/30 (67%) — the 6k screen overestimated the true ~52.9%.
- Harness findings recorded during this gate: small acceptor advantage
  (~1.4pt, measured on 4,998 r1-vs-r1 games) cancelled by pairing;
  prior-server session leak (~1 session/game) — cosmetic, tags monotonic,
  no state reuse possible.

## Follow-ups
1. c_puct sweep (1.0/1.5/2.5) for this candidate vs r1 (per-slot cpuct,
   SPRT p0=0.50 p1=0.58).
2. Retrain on same dataset with mcts-v3-coeff 0.3 and 1.0 (6k steps, batch 8,
   A10G) and SPRT-screen.
