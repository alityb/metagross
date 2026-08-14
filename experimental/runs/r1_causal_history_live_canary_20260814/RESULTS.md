# Newly played causal-history canary

Date: 2026-08-14

Execution: local CPU only

Status: **PASS for live deployability; no strength claim**

Two identical causal-history production controllers played one mirrored pair at
500 ms/P8. The result split 1-1 with zero voids, as expected for a plumbing
canary.

The two isolated frozen-R1 servers recorded 70 and 68 decisions across both
battles. Independent offline recomputation grouped each shared dump by battle
and reproduced every one of the 138 live probability vectors exactly:

- prior A: 70 rows, 2 sessions, maximum absolute difference `0.0`;
- prior B: 68 rows, 2 sessions, maximum absolute difference `0.0`;
- tolerance: `1e-7`;
- checkpoint SHA-256:
  `c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93`.

Both battle trajectories grew monotonically with no silent reset, missing
intermediate action receipt, RL2 mismatch, time-index mismatch, fallback, or
game void. This satisfies the newly played live canary required by the causal
history protocol. It does not measure whether causal history is stronger than
the legacy stateless input; that treatment is evaluated separately in
`../r1_causal_vs_stateless_screen_20260814/RESULTS.md`.
