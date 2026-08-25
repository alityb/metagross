# Belief-v1: redirected objective and bounded program

Date: 2026-08-16
Status: program definition after the Interior-v1 stop and the evidence-backed
exhaustion of the MegaGem controller-first objective. The project owner
redirected the assets on 2026-08-16: use the corpus/rematerializer/belief
infrastructure to improve the deployed agent, outside the MegaGem framing.

## Redefined objective

Improve the deployed agent — corrected causal-history R1 + production 500 ms
search (~92 GXE) — by any admissible mechanism, proven prospectively.

What no longer binds: the MegaGem sequence (slow controller first, one
distillation round, equal-budget retest) as the definition of success.

What still binds, unchanged: local CPU / $0; no fabricated or exposed hidden
information; causal information-state inputs only; battle/dependency-grouped
splits; preregistration before outcomes; outcome-blind progress watching (the
Cycle 42 lesson); zero interim looks; append every experiment and decision to
the iteration log; sealed93 stays sealed until a powered strength claim needs
confirmation; a strength promotion requires a prospective paired H2H whose
frozen criterion is met — never an offline metric; a GXE claim additionally
requires a bounded ladder block.

## Why this program

The July 2026 record contains two literature-backed strength candidates that
were fully built, leakage-audited, and then **abandoned mid-gate without a
decision** when the project pivoted to shinier architectures:

1. **Cumulative action-conditioned belief** (`foul_play_action_belief_root_priors_opp`):
   opponent actions update the generator posterior over their hidden randbats
   set via frozen-R1 likelihoods (log-space, tempered 0.5, fail-closed
   fallbacks). Exploratory screens: 12-12 (latest-action), 13-11 (cumulative);
   a confounded 60-39 screen was honestly withdrawn; the clean one-variable
   500-game gate was interrupted at 7-9 and "must not be cited as a failed or
   passed gate."
2. **Selective shared re-solving** (`foul_play_selective_shared_root_opp`):
   two-sided RM+ over posterior worlds, triggered only on high
   cross-determinization disagreement, LCB-gated confidence mixture. The
   iteration log's 44-35 snapshot is superseded by the fuller
   `selective_shared_root_gate_repaired_v2_20260727/STOPPED.md` record:
   **306 decisive games, 160-146 (52.29%), Wilson95 [46.7%, 57.8%], SPRT LLR
   -0.133 (no boundary), override-containing games 28-30, LCB magnitude not
   associated with outcomes, zero voids.** That is materially negative-leaning
   — the same search-internal-confidence failure Cycle 41 later found — and
   the July harness also lacked mirrored teams/RNG.

Corrected record for thread 1: the clean one-variable action-belief gate
(`action_belief_clean_gate_500`) stopped at 21 games, 9-12 (42.9%) — genuinely
undecided noise.

Finishing the D1 gate is the highest-value next work: zero new modeling, one
variable each, already-registered agents (`experimental/src/eval/run.py`),
passing unit tests, and honest precedent. Only if a thread promotes do
corpus-scale extensions (e.g., replacing the frozen-R1 action likelihood with
a model trained on the 1.29M causal states) become candidate follow-ups.

## Bounded program (three stages, no threshold changes after each freeze)

- **Stage 0 — compatibility smoke (development, not evidence).** Four paired
  local games per matchup on the current corrected causal-history stack.
  Requirement: zero unexplained voids, candidate belief endpoints actually
  exercised (nonzero valid likelihood/trigger counts), R1 priors served on
  every root. Harness repairs here are development work; nothing is scored.
- **Stage D1 — action-conditioned belief decision gate.** Candidate
  `foul_play_action_belief_root_priors_opp` versus baseline
  `foul_play_randbats_conditional_root_priors_opp` (identical pool,
  conditional generator, C1+C2 corrected R1 priors, engine, 500 ms/P8; the
  single variable is cumulative action-conditioned weighting at temperature
  0.5). Fresh teams/seeds/identities; paired mirrored roles; SPRT `H0=0.50`,
  `H1=0.55`, maximum 500 games; outcome-blind watcher only. Decision rule,
  frozen: promote only with zero unexplained voids and final Wilson 95% lower
  bound above 50% (or SPRT success boundary); SPRT futility or the boundary
  conditions failing closes this thread permanently.
- **Stage D2 — selective shared re-solving: demoted, contingent.** The
  306-game repaired_v2 record (52.29%, overrides 28-30, LCB non-predictive)
  is treated as strong prior evidence against this thread. It may be
  reconsidered only after D1 reaches a decision, only with the mirrored-pair
  harness, and only if a written case explains why the 306-game record does
  not already answer the question. Default: closed.

Stop rules: one run per stage; a substantive failure closes its thread with a
logged finding; only a pre-outcome harness defect found in Stage 0 may be
repaired; no threshold, trigger, temperature, or mixture tuning after a
freeze; no new architecture may be started while a stage is undecided.

## Honest expectations

Both threads are currently noise-compatible (all completed CIs include 50%).
A D1/D2 promotion would be an internal H2H improvement over the production
baseline at equal budget — a real, deployable gain — but not a GXE claim;
that requires a subsequent bounded ladder block. No numeric GXE promise is
made or implied.
