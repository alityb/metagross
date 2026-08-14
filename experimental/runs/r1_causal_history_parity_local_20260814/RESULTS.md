# R1 causal-history deployability result

Date: 2026-08-14

Execution: local CPU only

Status: **PASS for deployability; no performance claim**

## Result

The production R1 path can now reproduce a leak-free causal transformer
trajectory. The final controller action is joined to the exact private request
and acknowledged before it can be sent to Showdown. Public battle outcomes are
never used to infer our action. AMAGO reward/action inputs and absolute time
indices remain aligned across the live server, saved dump, and offline replay.

## Evidence

- The saved-protocol audit covered 40 real battle protocols, 1,207 private
  requests, and 1,051 outbound choices. All 1,051 choices mapped uniquely to
  their request and one of R1's 13 action indices. Every action index appeared.
  The 1,960 public move/switch events were counted as observations and ignored
  as labels.
- One complete captured battle was replayed through the actual frozen R1 server
  on CPU. Its 20 ordinary decisions produced 20 acknowledgements, monotonically
  growing inference/tensor sequences up to length 20, zero history resets, zero
  missing receipts, zero RL2 mismatches, and zero time-index mismatches.
- A separate process loaded the frozen 142.8M-parameter checkpoint and rebuilt
  every input sequence from the durable decision dump. All 20 policy rows
  matched the live probabilities exactly: maximum absolute difference `0.0`
  against the frozen `1e-7` tolerance.
- The checkpoint SHA-256 was
  `c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93`.

Primary artifacts:

- `saved_protocol_action_audit.json`
- `live_replay_report_v2.json`
- `live_replay_decisions_v2.jsonl`
- `offline_recompute_parity.json`

The earlier unsuffixed live-replay files are retained as superseded development
artifacts; the `v2` dump and report are authoritative.

## What changed

- Added an explicit `/action` receipt at the final controller-selection
  boundary, correlated by battle, request id, request hash, and decision index.
- Made duplicate receipts idempotent and conflicting, stale, unsupported, or
  missing receipts fail closed.
- Removed inference of our action from public `|move|` or `|switch|` events.
- Replaced the artificial zero observation with AMAGO's real time-zero
  observation and reward-first previous-action/reward layout.
- Preserved absolute AMAGO time indices when the live context window is
  truncated.
- Isolated mutable observation spaces per battle and repaired private active
  identity/move reconciliation without reading opponent-private information.
- Added exact handling for forced Recharge and Struggle boundaries.

## Interpretation and decision

This closes the missing production/offline history bridge. It proves that a
future history-conditioned candidate can be trained, replayed, and deployed
against the same causal information contract. It does **not** prove that R1 is
stronger, that a value model will rank actions, or that GXE will improve.

Do not start another generic action-Q or scalar terminal-value training run.
Metagross's two failed gates already show weak generalization from those
targets, and MegaGem independently demonstrates the same category of failure:
an offline-improving myopic selector can destroy long-horizon resource value in
live play. That result is a design warning, not Pokémon evidence.

The authorized next experiment is:

1. Run a small newly played local causal-history canary and require zero
   receipt, RL2, sequence, and time-index failures.
2. Freeze a battle-disjoint panel of informative roots and common
   determinizations/seeds.
3. Build a stronger long-horizon search expert whose rollouts value remaining
   HP, Tera, PP, tempo, revealed information, and switch resources through the
   rest of the battle.
4. Compare the expert with the current 500 ms controller on the frozen panel
   and then in a bounded paired live gate. Stop if it lacks an independently
   verified advantage.
5. Only after that pass, distill confident expert deviations into R1, mixed
   with pass-through R1 decisions and resource-stratified anchors. Evaluate the
   distilled policy at the same 500 ms budget.

This ordering makes the teacher's causal advantage the claim under test. It
prevents spending on a large distillation or league run before the proposed
improvement exists.
