# coeff 0.3 screen — stopped for futility

- Match: mcts_v3_full_6k_c03 (v3 distillation, coeff 0.3, 6k steps, batch 8,
  same dataset/hardware as the 0.529 coeff-0.1 arm) vs randbats_exit_r1,
  paired local 500ms/P8, SPRT p0=0.50 p1=0.58.
- Split across two runs due to a local machine reboot at game 48
  (`coeff_screen_c03/` + `coeff_screen_c03_resume/`).
- Combined: **101 games, 45-55 (0.450), 1 void.**
- Combined SPRT LLR ~ -2.9 (rejection bound -2.94): decisive accept-h0.
- Conclusion: raising the distillation coefficient from 0.1 to 0.3 makes the
  policy WORSE than baseline r1, not better. Consistent with the aux loss
  amplifying MCTS visit noise / pulling the policy off the behavior-constrained
  optimum. Dose-response points the wrong way for coeff 1.0 as well.
