# Metagross: Final Report

**Reproducing MegaGem's expert-iteration methodology in Pokémon, its
evidence-backed exhaustion, and what the corpus said instead**

Date: 2026-08-17 · Status: consolidated record of the 2026-07 → 2026-08
program · All claims below are backed by hash-frozen artifacts indexed in
`experimental/runs/iteration_log.md`.

---

## Executive summary

Metagross set out to reproduce the *transferable sequence* behind MegaGem's
auction result: prove a slow expert controller live against frozen
production, distill it once, and re-prove the student at the production
budget. After ≈48 numbered cycles the sequence is **exhausted by
measurement, not by fatigue**: every admitted expert family failed its
prospective live gate, and the final architecture candidate (a causal
interior search prior) failed its preregistered mechanics gate. The project
owner then redirected the assets toward improving the deployed agent
directly. One day of offline evaluation produced one real belief-layer win
(reveal-filter v2), three cleanly measured nulls, and a decisive
reorientation of the remaining strength hypotheses toward cross-world
aggregation and search-budget allocation. The strongest deployed agent
remains **corrected causal-history R1 + production 500 ms search** —
unbeaten by anything this program built.

**Addendum (2026-08-26, Part V):** the subsequent measurement campaign
re-measured both serving modes to matched RD-25 convergence (causal 89.4 /
stateless 91.7 GXE; the historical 92.4 substantially reproduces), confirmed
and replicated the late-game prior-overconfidence mechanism, closed two
serving-time gates (Gumbel-style decision rule: harmful; temperature
flattening: wins head-to-head yet loses ≈8 GXE at population level), and
established the campaign's central methodological finding — population and
head-to-head evaluations can flip signs, so only population measurement
licenses deployment. Part VI proposes the evaluation architecture that
follows.

---

## Part I — The MegaGem reproduction and its exhaustion

### The objective (frozen 2026-08-15)

1. A slow controller must prospectively beat frozen production R1/search in
   paired H2H with a confidence interval excluding 50%.
2. Only then, one Expert-Iteration/distillation round.
3. The distilled system must survive retesting at an equal 500 ms budget.

Offline accuracy, target stability, positive point estimates, and unpowered
canaries were preregistered as non-successes.

### The negative chain (all preregistered, all hash-frozen)

| Candidate expert | Result | Verdict |
|---|---|---|
| Terminal-MCTS recurrent controller | 72–78 / 150 | FAIL |
| Randomized one-deviation controller (early) | below frozen gate | FAIL |
| Equal-prior 8,192 root controller (Cycle 41) | 11/20 vs ≥13/20 gate; 165/575 overrides; internal confidence did not identify safe corrections | FAIL |
| One-deviation attribution reruns | Cycle 42 VOID (interim-look breach); Cycle 43 integrity-fail (Struggle semantics), protocol-retired | RETIRED |
| Counterfactual depth-one target bridge (Cycle 47) | 5.42% support, 8/64 battles | REJECTED |
| Causal interior prior, Gate A (Cycle 48) | 498/512 states admitted (97.3% row coverage) but <512 unique fingerprints; 80 fail-closed causal-integrity rows | FAIL — architecture stopped per program |

Cycle 48's failure was concentrated in three reproducible mechanics-tail
contracts (item-consumption authority, late-battle PP authority, R1 served-
action vocabulary). Its conditional positives — 86.95% top-1 agreement
between 8,192- and 20,000-iteration equal-prior search on observed states,
repeat JSD median 5×10⁻⁵ — establish that the search target itself is
highly reproducible; the expert built from it simply was never shown to be
stronger.

**Exhaustion finding (2026-08-16):** with every admitted family closed, the
controller-first / distill-second objective was declared exhausted on
current evidence rather than continued through repair cycles. MegaGem's
lesson survived intact in procedural form: *nothing was distilled from an
unproven expert at any point in this project.*

### What the reproduction attempt left behind (durable assets)

- **20,385 deterministically rematerializable public human battles** with a
  99.149% full-corpus admission gate and zero split leakage;
  **1,288,022 reconstructed causal information states**.
- A pinned-commit rematerializer with byte-exact request/prefix/ledger
  parity, typed causal reveal ledgers, slot-aware masks, and fail-closed
  mechanics contracts (Form, Ability, PP, Disable, Pressure, Struggle,
  switch-reactivation, request lineage).
- A dual-variant offline rqid-correlation contract making the 142.8M
  causal-history R1 exactly replayable offline (max abs diff 0.0).
- A paired/mirrored H2H harness with SPRT, outcome-blind watching, and
  counterbalanced scoring — hardened by its own recorded failures.
- The sealed 93-battle confirmation panel, **never opened**.

---

## Part II — The redirection: belief, measured

On 2026-08-16 the owner redirected the assets: improve the deployed agent,
outside the MegaGem framing. The archive audit surfaced two July threads
abandoned mid-gate; finishing them and mining the new Gate A corpus set the
agenda.

### Corpus findings (Gate A artifacts, 498 observed states)

1. **Disagreement with deep search is a belief-ambiguity phenomenon.**
   Where the 8 posterior worlds' searches agree unanimously, humans and R1
   match deep search ≈55–60%; where ≥3/8 worlds disagree, both collapse to
   30–39%. Search depth is not the binding constraint.
2. **Humans do not follow the majority world.** In ambiguous states the
   human action matched the majority-world recommendation only 36%
   (minority world 31%, *no* sampled world 33%) — human play embodies
   either sharper inference or robust aggregation that uniform-weighted
   determinization lacks.
3. **R1 has a confident-miscalibration class:** 83/498 states across 46
   battles where the human matches deep search but R1 confidently
   disagrees (median 0.61 on its own pick vs 0.16 on the human+search
   action).

### The Skat literature import

Buro et al. (IJCAI-09), Solinas/Rebstock/Buro (AAAI-19), and Policy
Inference (CoG-19) define the exact template: bias determinized-search
world sampling by likelihoods learned offline from human play, evaluated by
TSSR (how much likelier than uniform the true hidden state is sampled).
The project's rematerializer supplied what July lacked: true hidden teams
as **held-out evaluation labels** (never agent inputs), enforced by file
boundary.

### One day of offline gates (13,365 decisions, 248 held-out battles, $0)

| Belief candidate | Result |
|---|---|
| Generator-pool prior | mass 0.366 / top-1 0.432 (avg ≈3.5 candidate sets/species) |
| + moves-reveal filter (production equivalent) | mass 0.541 / top-1 0.589 |
| **Filter v2: + item/ability/tera reveals** | **+4.7 mass / +5.0 top-1 on the complete key space, at every reveal depth — PASS** |
| First-revealed-move evidence tables (107k instances) | null (top-1 −0.8); leaky-smoke lift was memorization |
| Learned action likelihood, shallow context (392k rows) | val NLL 1.3305 vs marginal 1.3318 — ≈ marginal; dead |
| Marginal-likelihood TSSR arm | +0.001 mass — null; closes the family |

**Conclusion:** gen9 randbats hidden-set inference beyond consistency
filtering carries little externally-inferable signal at shallow context —
candidate sets are few, reveal quickly, and within-set human move choice is
near-uniform given shallow context (1.33 of 1.39 nats). The belief
posterior is near its practical offline ceiling at filter v2. Corollary:
finding 2 above is more plausibly about **aggregation and robustness across
worlds than about sharper inference**, which moves the remaining live
strength weight to MAPLE-style shared-tree aggregation (2605.24139) and
dynamic determinization budget allocation (2607.13007) — precisely where
the recent IIG literature reports its largest effects.

### The D1 screen (closing an abandoned July thread)

The cumulative action-conditioned belief candidate (frozen-R1 likelihood)
ran a preregistered SPRT screen against its exact one-variable baseline.
After a void first attempt (an all-NaN policy at a Revival Blessing revive
prompt — fixed fail-closed, with regression tests, before relaunch) the
rerun reached, at this report's cutoff, **159-165 (49.1%) over 324
decisive games, SPRT LLR −2.23, zero voids** — descending toward the
futility boundary. Terminal update to be appended; the treatment/placebo
split (evidence fires only in candidate-as-p1 games) showed the treatment
games at ~coin-flip throughout. Whatever the final line, the thread closes
with a measured decision — its first.

---

## Part III — Evaluation architecture (built, portable, paused)

- **Offline-first gating:** TSSR-style calibration gates now filter belief
  candidates for pennies before any live game — three candidates were
  killed offline in one day that would previously have cost 25-hour runs.
- **Cloud gate farm (Modal):** the vendored patched engine builds and
  behaves correctly on Linux (M1 probe PASS); the full-stack image (M2) and
  the lane API (M3: one container = Showdown + two isolated prior servers +
  N mirrored paired games) are code-complete in
  `experimental/src/scripts/modal_game_farm.py`. Validity rests on
  platform-internal pairing; power becomes cheap (~$15–60 per 500–2,000-game
  gate, <1 h wall). Deployment is paused on a local network path that
  blocks sustained gRPC; it is one `modal deploy` away on a normal
  connection.
- **Open champion question, queued for the farm:** the G4 checkpoint beat
  r1 61-39 in the counterbalanced full-search pilot, yet lost decisively on
  the public ladder — and every cross-policy ranking from that era was
  measured under a serving path later shown broken (the causal-history
  repair changes R1's top action on 32% of decisions). A ~$30 powered gate
  under the corrected stack would settle the true champion.

---

## Part IV — Lessons this project paid for

1. **Prove the expert live before distilling.** Honored throughout; it is
   why 48 cycles produced no false success.
2. **Offline success fails live, repeatedly.** Locally calibrated
   corrections, equal-prior deep search, LCB-gated re-solving, and
   direct-policy gates all looked better offline than they played.
3. **Search-internal confidence does not identify safe corrections.**
   Found independently by Cycle 41 and the July selective re-solving record
   (overrides 28-30, LCB non-predictive).
4. **Evaluation design is a first-class experimental object.** A
   "symmetric" scorer measured 44% for a self-pair; counterbalancing
   exposed a 10-point slot effect; an interim peek voided a cycle; mirrored
   pairing, SPRT, outcome-blind watching, and fail-closed data admission
   were each earned from a specific recorded failure.
5. **Close threads with decisions.** The costliest pattern in the archive
   was not failure but abandonment-at-7-9: undecided gates compound into
   folklore. Every thread this program touched now ends in a measured
   verdict or a preregistered stop.
6. **Cheap gates change what is thinkable.** Once a candidate can die for
   pennies offline, exhaustive honesty becomes affordable.

## What would justify reopening

- **Strength:** the farm's first batch — champion selection (G4 vs
  corrected r1), the filter-v2 live gate, then single-variable gates for
  budget allocation and MAPLE-style aggregation on the ladder of champions.
- **MegaGem sequence proper:** only a genuinely new expert family (e.g., an
  aggregation-corrected searcher that first wins its own live gate) would
  reopen distillation — same ordering, same discipline.
- **Mechanics:** the three named Gate-A contract tails, as a bounded
  mechanics program, if observed-state corpora are ever needed at 100%.

---

*Artifact index: every experiment, hash, void, and decision referenced here
is recorded chronologically in `experimental/runs/iteration_log.md`; frozen
run directories under `experimental/runs/` carry protocols, manifests,
selections, raw measurements, and reports.*

## Part V — Addendum (2026-08-26): the measurement campaign

Everything above froze on 2026-08-17. The nine days after it produced the
project's strongest results, all preregistered, all in
`experimental/runs/iteration_log.md`.

### The matched pair: both serving modes, one standard

Two fresh accounts, same machine, same deterministic 472k-iteration search
budget, single variable = trajectory mode, both converged to the Glicko RD-25
floor — the exact standard of every historical number:

| Serving mode | Record | GXE | Glicko ± RD |
|---|---|---|---|
| causal-history | 149-81 | 89.4 | 1902 ± 25.0 |
| legacy-stateless | 174-83 | 91.7 | 1952 ± 25.0 |
| (reference) frozen-r1 causal | 141-85 | 86.6 | 1851 ± 25 |
| (reference) historical stateless era | — | 92.4–92.7 | RD 25 |

Findings: the historical 92.4 substantially reproduces (91.7, fresh account,
months later); the corrected causal path reproduces r1 and lands slightly
above it; the modes are ≈2 GXE apart at population level while a direct
200-game mirror measured 48% — parity. The six-point historical gap
decomposes into mostly era difference, a small real population edge, and no
head-to-head edge at all.

### The mechanism: late-game prior overconfidence

The causal prior's entropy collapses from ≈0.96 to ≈0.58 nats by turn 30+ —
measured offline in August, then replicated independently in cloud games
(0.98/0.73/0.72/0.58 by turn bucket, ≈550 decisions). The stateless prior
stays flat. This is the one confirmed mechanistic difference between the
serving modes.

### Serving-time gates: three interventions, three verdicts

**Gate B (Gumbel-style completed-Q root decision): decisively harmful.**
Replacing the visit-share decision rule with `log pi(a) + scaled Q(a)` flipped
37% of decisions and lost 10-40 against the unmodified rule. The visit rule's
implicit variance discipline — only trusting actions the search actually
vetted — carries real value that pooled Q estimates cannot replace.

**Gate C (history truncation): implemented, untested** (budget); remains a
serving-time flag.

**Gate D (entropy-matched temperature flattening): the campaign's deepest
result.** Three valid measurements that appear to conflict and do not:

1. Mechanism: the schedule verifiably restores late-game search breadth
   (activation-proven telemetry; it overshoots the entropy target).
2. Head-to-head vs stateless: **111-89 (55.5%, CI [48.6, 62.2])** over 200
   mirrored games — a ≈7-point recovery over the plain-causal baseline in
   the same matchup.
3. Ladder population: **≈81.7 GXE vs the 89.4 plain-causal reference** —
   roughly eight points of harm, persistent across ≈139 games,
   promotion-excluded by direct calculation before the run finished.

### The central finding: population and head-to-head measurements flip signs

Twice in one campaign, the two measurement families disagreed in *direction*:
causal-vs-stateless is parity head-to-head but a gap on the ladder;
flattening wins head-to-head and loses badly on the ladder. The consistent
account: interventions that trade decisiveness for robustness pay against
strong systematic opponents and bleed against the weak, erratic majority of
a real population. Neither measurement is "the truth"; they answer different
questions, and only the population answer licenses deployment. This rule is
now binding on all future promotion decisions.

It took five attempts to measure prediction-2 validly (a silently inert hook,
a wedged cloud container, a resume bug, and a dual-arm contamination that a
zero-activation-lines audit caught). The observable-activation rule — no
intervention result is valid without proof in the logs that it was on, and
off for the control — is the campaign's most transferable methodology lesson.

## Part VI — What better evaluation looks like (proposal)

The campaign's failures indict single-opponent H2H as a promotion gate. The
replacement, cheapest first:

1. **Offline paired-decision pre-gate** (exists): the 699-decision paired
   corpus plus the fail-closed n100 eval; extend with per-decision
   counterfactual scoring against the reference policy using the search's
   own Q estimates. Minutes, $0, thousands of effective samples.
2. **A frozen league instead of one opponent**: round-robin against a fixed
   pool spanning strength and style — plain causal, stateless, r1-no-priors,
   heuristic foul-play, and weaker archived checkpoints (`gate1`, `r2`
   candidates). Mirrored pairs per opponent; report the per-opponent vector,
   not one number. This reproduces population diversity locally and would
   have caught the flattening sign-flip in an afternoon.
3. **Luck-corrected, anytime-valid scoring**: mirrored pairs already halve
   variance; AIVAT-style baselines (cf. AV-AIVAT, arXiv 2608.06362) cut the
   games needed by an order of magnitude and give certified stopping — the
   fix for SPRT runs that hit the cap undecided.
4. **Strength-stratified ladder analysis**: when a ladder run is warranted,
   report win rate by opponent-rating bucket rather than only converged GXE;
   the flattening result predicts exactly this kind of strength-dependent
   effect structure.
5. **The tiered gauntlet as one command**: offline pre-gate → league →
   (only on a pass) fresh-account ladder block, with observable activation
   enforced at every tier. Codified, not remembered.
