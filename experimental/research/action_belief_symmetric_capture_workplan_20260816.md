# Workplan: symmetric-role action-evidence capture (apply ONLY after the screen ends)

DO NOT APPLY WHILE `belief-v1-d1-gate-attempt2` IS RUNNING: each game spawns
fresh agent processes from `experimental/src/scripts/run_foul_play.py`; editing
it mid-run silently changes the live experiment.

## Defect

Opponent-action evidence is captured only when the candidate plays p1. In
`run_foul_play.py`, inside `receive_with_public_action_history`, the
pending-action branch is guarded by `opponent_side == "p2"`:

    elif (opponent_side == "p2" and msg_type in ("move", "switch")
          and len(message_parts) >= 3 and message_parts[1].startswith(opponent_side)):

Stage 0 confirmed the consequence: 29/35 events captured in candidate-as-p1
games, zero in p2 games (uniform fallback). The treatment dose is halved.

## Root cause of the guard (why it is not just `!= ""`by accident)

Before the `|player|` line arrives, `opponent_side` is `""`, and
`message_parts[1].startswith("")` is True for every actor — an unguarded
branch would capture BOTH sides' actions, including our own. The original
author hardcoded `== "p2"` (safe for the common case) instead of excluding
the empty string.

## Fix (one line)

Replace the guard with a non-empty side check, preserving the hazard guard:

    elif (opponent_side in ("p1", "p2") and msg_type in ("move", "switch")
          and len(message_parts) >= 3 and message_parts[1].startswith(opponent_side)):

The rest of the closure is already side-generic: the pending-confirmation
branch (`move/switch/drag/turn/cant`), the `-terastallize` correlation branch,
and the forced-switch faint tracking all use `opponent_side` symmetrically.
Ordering note: with a p1 opponent, its action line precedes ours inside the
same message block, so the pending action is confirmed within the block —
still strictly pre-decision, no future information.

## Required validation before any gate use

1. Unit tests simulating message streams for BOTH orientations: p1 opponent
   and p2 opponent; assert captures only of opponent actions, none of own
   actions, empty-side hazard (no captures before `|player|` line), tera
   suffix correlation, forced-switch (post-faint) exclusion.
2. Stage-0-style smoke (4 mirrored games): expect nonzero captures in BOTH
   role orientations; likelihood availability roughly doubles versus the
   38.3% Stage-0 rate; zero voids.
3. This is a candidate CHANGE: it may only enter a freshly frozen gate
   (see CONFIRMATION_GATE_DRAFT.md option), never the running screen, and
   the single-variable claim of that gate must name the full-dose variant.
