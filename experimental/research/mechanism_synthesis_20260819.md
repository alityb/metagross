# Why the stateless-served stack wins: a mechanism synthesis

Date: 2026-08-19 · Status: explanatory synthesis assembled from existing
artifacts only (no new experiments). Every number below is from a frozen
run already recorded in the iteration log.

## The puzzle

Same weights, same search, same budget, same machine. Stateless serving
(the policy sees only the current observation): **92.4–92.7% GXE**.
Corrected causal-history serving (provably right, exact offline parity):
**86.4% GXE at 150 games** (continuation running). The correct input is
worth minus six points. Why?

## The evidence chain

1. **History collapses the prior's entropy, late.** Paired counterfactual
   inference on 699 identical live states (Aug-14 screen): stateless prior
   entropy is flat across the game (0.98 → 0.96 nats); causal prior
   entropy falls monotonically with history — 0.96 / 0.84 / 0.72 /
   **0.57** nats at turns 30+. The causal policy becomes *confident* as
   the game ages.
2. **That confidence is echo, not information.** By history length, the
   causal prior increasingly concentrates on the action the search itself
   selects (p(selected) 0.47 → 0.56; top-1 agreement with search 0.59 →
   0.62), while stateless stays flatter (0.40 → 0.48). In PUCT, a prior
   that echoes the searcher's own inclinations narrows the tree exactly
   where its errors correlate with the evaluator's errors.
3. **Late confidence is where the games are lost.** Ladder losses run
   median 32 turns vs 21 for wins; zero losses before turn 9; no
   operational taint (0 errors, 0 fallbacks, 0 timer losses in measured
   play). The corrected stack loses long games — the exact regime where
   its prior is sharpest.
4. **The confidence is miscalibrated there.** Independent corpus evidence
   (Gate A, 498 held-out states): R1's confident-miscalibration class —
   83 states where the human and deep search agree against R1, with R1 at
   median 0.61 on its own pick; and neither humans nor R1 track deep
   search better in late phases even as search itself sharpens.
5. **Training never certified those histories.** The checkpoint was
   trained on replay sequences, but every generation loop that shaped its
   behavior (ExIt r1, G2–G5 collections) *played* through stateless
   serving. Empty-history input is in-distribution at every trajectory's
   t=0; long live histories under the policy's own corrected-serving play
   drift from anything the training corpus certified. Serving stateless
   is, accidentally, serving the best-trained conditional of the model.

## The mechanism, in one paragraph

The deployed system's strength lives in the search (500 ms production
search beat the raw policy 17-3). A PUCT search is only as corrective as
its prior lets it be: a flat-enough prior lets the hand-leaf evaluator
explore and veto; a sharp prior spends the budget confirming itself. The
stateless defect acted as an **accidental entropy regularizer** — it held
the prior humble in late game, where the policy's history-conditioned
confidence is least calibrated, keeping the strong component (search) in
charge at the decisive moments. The causal repair handed control back to
the policy's least-trustworthy conditional, and the ladder priced that
transfer at ~6 GXE.

## Why every other approach failed — the same account

- **Equal-prior 8,192 root search (11/20):** removed the prior entirely —
  too flat; the search over-explored and lost the policy's real knowledge
  (165 exploratory overrides, no gain). Confidence floor, not ceiling.
- **Terminal-MCTS / one-deviation / LCB-gated overrides (72-78/150;
  28-30 override games):** injected *sharp, confident corrections* whose
  confidence didn't correlate with outcomes — the same miscalibration
  failure at the override layer. Cycle 41 and the July selective record
  found it independently: search-internal confidence does not identify
  safe corrections.
- **Equilibrium shared-root solving (6-18):** replaced the co-adapted
  prior↔search↔evaluator operating point with a robust-policy objective
  the population doesn't demand.
- **Tables / shallow policy-inference belief nulls:** tried to add
  information where there is almost none left (consistency filtering
  already captures it); no confidence distortion, but no signal either.
- **Filter v2 (+4.7 mass, the one offline win):** the exception that
  proves the rule — it adds *hard evidence* (item/ability/tera reveals)
  without touching the prior's calibration or the search's authority.

The unifying claim: **this system is a co-adapted ensemble whose live
strength sits at a particular prior-sharpness operating point. Every
intervention that moved the operating point — sharper (overrides,
causal-history late-game), flatter (equal priors), or differently
objective (equilibrium) — lost live, regardless of offline merit.**

## Falsifiable predictions (preregistered here, before any new run)

1. **History-truncation dose-response:** serving with history truncated to
   k ∈ {0, 2, 8, 32} steps will show live strength *decreasing* in k
   (monotone or near-monotone), while offline policy-parity metrics
   improve in k. Stateless (k=0) is not a bug but the argmax.
2. **Entropy intervention recovers most of the gap:** causal-history
   serving with late-game prior temperature flattening (temperature
   scheduled on history length or turn, calibrated offline on the Gate A
   corpus, frozen before play) recovers ≥ half of the six points — i.e.,
   causal information is not harmful; causal *overconfidence* is. If this
   fails, the OOD-history account dominates the calibration account.
3. **Controlled H2H:** corrected vs stateless serving, same checkpoint,
   mirrored pairs, powered — reproduces the ladder gap's sign with CI
   excluding 50% (converts the observational six points into a causal
   estimate).
4. **Retraining under corrected serving** (long horizon): a policy whose
   generation loop plays through causal-history serving closes the gap
   and can exceed 92.4 — the train/serve co-adaptation account's endgame.

Prediction 2 doubles as the cheapest path to a stack that beats 92.4:
keep the causal information, fix its calibration.
