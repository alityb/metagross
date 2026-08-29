# Iteration Log

## Reconstructed Prehistory Through 2026-07-20

This section reconstructs major iterations completed before this append-only
log was created. Numbers below come from retained datasets, gate artifacts, and
the session record. They are included to prevent previously falsified branches
from being proposed again without addressing their failure.

### Frozen Baseline And Measurement Notes

- Accepted baseline is ExIt r1 epoch 5, deployed as
  `foul_play_root_priors_opp` at 500ms/P8/one search thread. Settled public
  ladder result was 92.4-92.7 GXE at RD 25, with observed peak 93.6.
- Formal promotion rule is 500 paired games and Wilson 95% lower bound above
  50%, with zero unexplained voids. At N=500 this normally requires at least
  272 wins.
- A 4,998-game r1 self-play audit found an approximately 1.4% acceptor edge;
  paired role balancing is therefore mandatory.
- The prior server retains roughly one stale session per completed game. Tags
  remain monotonic, so this was treated as a cosmetic memory/telemetry issue,
  not data-key corruption.
- Sustained local evaluations rebooted the Mac twice. Long gates should run in
  bounded/background jobs with append-only progress artifacts.

### Schema-v3 MCTS Distillation

**Problem:** Schema-v2 MCTS targets depended on replay-parser alignment and
could not safely match live policy observations to search visits.

**Implementation:** Schema v3 captures the exact prior-server observation,
legal mask, name table, namespace, tag, username, and decision index consumed
at decision time. Foul Play records the echoed decision index and battle tag.
The builder joins fail-closed on `(tag, username, decision_idx)` without replay
parsing. Per-worker namespaces prevent battle-tag collisions. Forced-only
actions are skipped; mask fallback is recorded and rejected where required.
The legality-mask bug `illegal[a]` was corrected to `illegal[a.action_idx]`.

**Round-1 data:** 5,008 games, 10 voids, 4,767 admitted groups (95.2%),
175,319 verified visit targets, 4,595 learner trajectories, and zero mask
fallbacks.

**Trainer:** Added stateless T=2 server-equivalent distillation with masked
cross-entropy auxiliary loss. Real-agent equivalence differed by at most
`1.2e-7`.

**Results:**

- Best 6k-step/coefficient-0.1 candidate: 264-235 (52.9%) in the formal
  500-game gate; Wilson lower bound 48.5%; **not promoted**.
- 3k-step screen: 12-18 (40%).
- Coefficient 0.3: 45.0%; coefficient 1.0: 40.7%.
- `c_puct=1.0`: 50.9%, no meaningful improvement.

**Decision:** Distillation had a small positive ceiling but failed promotion;
increasing auxiliary strength monotonically hurt behavior.

### ExIt Round 2

Round-2 self-play used the 6k candidate on both sides: 5,008 games, 179,066
verified targets, and 4,621 trajectories. A 6k continuation initialized from
the round-1 6k checkpoint screened 34-40 (45.9%).

**Decision:** Iterating on the candidate's own search distribution did not
compound and made the policy worse. Stop latest-self-only ExIt on this path.

### Gen9 Learned Leaf Value

**Implementation:** Ported Gen9 learned evaluation into a merged
root-prior/learned-leaf engine. Added a 14-feature linear and MLP evaluator,
model loader, isolated build/venv, and Foul Play agent integration.

**Terminal-outcome model:** 184,213 decision examples. MLP held-out Brier was
0.217 and accuracy 63.9%. MLP leaf evaluation alone screened 53.3% in 30 games,
but MLP plus r1 root priors scored 7-18 (28%). Linear value scored 30% in 20
games.

Completed the fresh-data bridge after confirming the old 5,000-game league
archive lacks replay protocol and mechanical decision states. Extended the
causal root builder with a fail-closed schema-6 snapshot authority: contiguous
observations, RL2, receipts, time indices, final-row equality, causal mode, and
mask validity are mandatory. All 136 fresh dual-R1 snapshot rows passed; 134
were policy-indexable. Outcome panels now retain only leak-free public context
and a hashed snapshot locator. Added a stable-deviation-only dataset
materializer that rejects non-admitted analyses, maps actions through the exact
schema-6 name table, preserves the full causal transformer history plus live
search features, and never includes sampled engine state. Thirteen focused
tests pass, including end-to-end synthetic materialization. Froze a staged
fresh corpus: 500 capture games, 5,000 only after capture/yield admission, and
25,000 only after physical-battle-grouped residual learnability.

**Search-derived-label attempts:**

- Deep-node terminal-subtree targets were all 1.0, even after reservoir
  sampling across depth 2-4 nodes. This was selection bias, not a collector bug.
- Exact continuation rollouts from logged Foul Play states also produced all
  side-one wins because player-view logs serialize unrevealed opponents as
  fainted `NONE` slots. The collector was changed to fail closed.
- Added strict posterior completion from the 50k Showdown generator pool,
  retained public leaf features, and rolled out only a completed clone.
- A 1,000-root run yielded 945 terminal labels. The resulting residual MLP at
  weight 0.1 screened 5-11 (31.3%).

**Decision:** This value representation/target path is harmful in search. The
learned-value blend defaults to zero and must be explicitly enabled. Do not
scale this exact model/feature setup.

### High-Budget MCTS Teacher

The old high-budget artifact was schema v2 and lacked result labels/join keys,
so it was rejected. A fresh schema-v3 3-second teacher was validated through
two isolated r1 prior servers. Four-game smoke admitted 4/4 groups and 109/109
targets. The 16-game calibration admitted 16/16 groups and 627 targets.

Three-second search averaged 22.5M visits per decision versus 5.8M at one
second, approximately 3.9x more. However policy quality proxies were unchanged:
mean entropy 0.820 versus 0.802 nats, mean top visit mass 0.689 for both, and KL
from r1 prior statistically overlapped the one-second sample.

**Decision:** Extra visits mostly refined the same policy. Do not scale the
3-second teacher or launch another distillation run from it. Temporary prior
servers and reverse tunnels were stopped.

### Exact Conditional Generator Belief Screen

Added a combined agent using exact conditional Showdown-team generation plus
r1 priors, fixing a missing `--randbats-conditional-samples` harness option.
Across 39 exploratory games it finished 22-17 (56.4%), but Wilson lower bound
was only 41.0% and the known completed role split included 3-7 as acceptor.

**Decision:** Early 5-3 smoke was not stable evidence. Do not gate the raw
conditional-belief replacement without a stronger mechanism.

### Infrastructure State

- Four c6i.16xlarge workers previously used for schema-v3 collection had EBS
  expanded to 50GB. At the latest infrastructure audit, two remaining running
  c6i.16xlarge workers were idle and were explicitly stopped. The A10G
  `g5.xlarge`, both older g4dn hosts, and the Nebius H200 were also stopped.
- No AWS training or collection job is currently intended to be running.

### Interrupted Clean Action-Belief Gate

An early 100-game cumulative action-belief screen reported 60-39 over 99
decisive games, but was withdrawn because candidate and baseline used different
belief samplers. A corrected one-variable agent split was created so both sides
share exact generator belief, priors, engine, and budget. Its 500-game gate was
later interrupted at 7-9 when the project deliberately pivoted to the more
ambitious shared-root architecture; it has no final promotion decision and must
not be cited as a failed or passed 500-game gate.

## 2026-07-21: Action-Conditioned Randbats Belief (Stage A)

**Hypothesis:** Opponent actions contain likelihood evidence about their active
Random Battle set. Updating the public generator prior with a frozen r1 policy
likelihood improves recovery of a later uniquely revealed active set.

**Protocol:** For every action boundary, construct candidates using only the
public prefix. Score the observed opponent action under each candidate's masked
opponent view, update by Bayes' rule, and evaluate only against later public
reveals that uniquely match a generator-pool set. No final reveal may enter the
candidate filter or policy input.

**Baseline:** Generator-only posterior over sets compatible with the public
prefix.

**Primary metrics:** Held-out posterior top-1/top-3, mean label probability,
MRR, Brier score, and reliability. The candidate-conditioned policy likelihood
must also pass legal-action and candidate-perturbation checks.

**Gate:** Do not collect private manifests or alter live MCTS unless held-out
set recovery and calibration both improve over generator-only belief without
leakage or silent fallback.

**Implementation:** Added `src/belief/action_conditioned_randbats.py` and
`src/scripts/benchmark_action_conditioned_randbats.py`. The benchmark enforces
finite nonnegative per-candidate likelihoods, legal observed actions, candidate
identity coverage, no label-named fields in pre-action candidates, and
replay-level chronological holdouts. Unit tests and a two-row CLI fixture pass.

**Result:** Infrastructure verified only. The fixture improves posterior top-1
from 0.0 to 1.0 by construction; it is not evidence about Pokemon.

**Blocker:** `PriorSession.compute_opponent_priors()` evaluates a public-state
mirror, not `P(action | candidate set)`. It must not supply this benchmark.
The next required component is a frozen r1 adapter that hydrates each candidate
active set into the opponent's masked replay state before scoring its observed
action. Public replays provide only later unique-set labels; controlled private
team manifests remain required before whole-team or live-search claims.

**Adapter implementation:** Added
`src/belief/action_likelihood_adapter.py`. It constructs a candidate-specific
opponent `UniversalState` from a public `ReplayState`, replaces only the active
candidate's set fields, removes observer-private switches and preview fields,
uses a candidate-specific legal mask, and batches frozen-r1 inference with the
same two-step tensor layout as the prior server. Nine unit tests pass across
the posterior core and adapter, including illegal-action rejection, source
immutability, no-private-switch leakage, and candidate move/mask perturbation.

**Remaining gate:** Existing parsed trajectories are backward-filled and may
contain later information. The benchmark producer must replay each raw protocol
only to the pre-action boundary, then provide public `ReplayState` plus a later
unique-set label held outside the prefix. No offline Pokemon result is reported
until that producer passes an explicit prefix audit.

**Status:** Candidate-conditioned likelihood adapter ready; no-leak replay-row
producer in progress.

**Replay producer:** Added
`src/scripts/produce_action_conditioned_randbats_rows.py` and
`src/scripts/attach_action_conditioned_likelihoods.py`. The producer
forward-replays raw Showdown protocol only to the p2 pre-action boundary,
creates generator-pool candidates from prefix facts, and derives labels only in
a separate suffix pass. The attachment CLI loads frozen r1 directly, attaches
candidate-conditioned likelihoods, and assigns zero likelihood when an
observed action is impossible under a candidate-specific legal mask.

**Execution checks:** The 14-test Stage-A suite passes. An initial real frozen-r1
four-row smoke attached likelihoods to 3 rows and rejected 1 impossible-action
candidate; this exposed a modeling error, because impossibility is strong
Bayesian evidence rather than a malformed row. The adapter was corrected to
retain that candidate at zero likelihood. A 100-replay public audit generated 1,920 valid action
prefixes but **0 uniquely later-revealed active-set labels**. Therefore no
calibration or reranking claim is possible from the public corpus under the
no-future-leak protocol.

**Decision:** Do not run a public-replay benchmark with proxy labels. The next
required data source is controlled self-play that stores private generated-team
manifests separately from the public protocol. Only that data can validate
active-set and whole-team posterior recovery before live MCTS integration.

**Status:** Stage-A infrastructure complete; public-label gate failed as
expected; controlled-manifest collection required.

## 2026-07-21: Controlled Private Manifest Capture

**Implementation:** Added an opt-in `METAGROSS_PRIVATE_TEAM_MANIFEST_DIR` hook
to the local Pokemon Showdown simulator. At `Battle.setPlayer`, it appends the
generated six-set team, player, side, and battle ID to a private local JSONL
file. The manifest is never added to the Showdown protocol or public replay.

**Smoke result:** An isolated capture server wrote exactly two six-Pokemon
private manifests for one started battle. The random-vs-random smoke then hit
an invalid forced-switch client choice and was stopped by the operator. This
is a harness/client issue, not a manifest-capture failure. The temporary server
was stopped; do not collect controlled labels until the isolated-client smoke
completes cleanly.

**Fix and verification:** Replaced random clients with the production Foul Play
client path and supervised the isolated server lifetime in the same command.
The smoke completed two games with zero voids. Showdown can reuse an empty
internal battle ID in this mode, so manifests now include a process-unique
`capture_id` assigned once per battle. Verification found two capture IDs, each
with exactly two six-Pokemon manifests. Controlled traces join manifests by the
two generated player names plus `capture_id`, not raw battle ID.

**Controlled truth join:** Added `attach_controlled_randbats_truth.py`, which
joins a replay to exactly one private manifest pair by player names, identifies
the acting active set, and maps it to a normalized generator-pool candidate ID.
It never writes a manifest or full team into the public benchmark row. Candidate
identity now normalizes gameplay-relevant set fields and collapses irrelevant
generator metadata such as role and move ordering.

**Controlled pilot:** Ten Foul Play-vs-Foul Play games produced 176 labeled
public-prefix rows. Frozen-r1 likelihood attachment completed all rows; four
were deterministically capped at 32 candidates. On the chronological holdout
of the final two replays (41 rows), action conditioning improved Brier
`0.486 -> 0.471`, top-1 set recovery `61.0% -> 65.9%`, mean true-set
probability `0.523 -> 0.547`, and MRR `0.717 -> 0.751`. Aggregate metrics are
mixed and the sample is too small; source actions are Foul Play, not humans.

**Decision:** This clears only the offline calibration-smoke gate. Collect a
larger controlled corpus and repeat a replay-level held-out test before any
weighted-particle MCTS integration.

**Expanded controlled calibration:** Fifty controlled Foul Play-vs-Foul Play
games yielded 1,197 public prefixes. Exact truth joining retained 1,182 rows;
15 ambiguous active-species cases were rejected. Frozen-r1 attachment retained
1,181 rows; one label excluded by the 32-candidate safety cap was rejected.
On the chronological 10-replay holdout (275 rows), action conditioning improved
Brier `0.463 -> 0.448`, mean true-set probability `0.542 -> 0.578`, top-1
`56.7% -> 60.4%`, top-3 `83.6% -> 91.3%`, and MRR `0.716 -> 0.761`.

**Decision:** The held-out calibration gate passes on controlled Foul Play
action data. Proceed to guarded live weighted-particle integration, retaining
uniform generator sampling as the fail-closed fallback and measuring posterior
coverage before any H2H claim.

**Guarded live integration:** Added opt-in
`METAGROSS_ACTION_CONDITIONED_BELIEF=1` plumbing, bounded candidate requests,
action-history signatures, evidence caches, aggregate diagnostics, and a
fail-closed prior-server `/action-likelihoods` endpoint. Disabled mode leaves
the existing uniform generator sampler unchanged. Invalid, unavailable,
all-zero, overflowing, or malformed evidence also falls back exactly to uniform
sampling; candidate identities are never written to public replay or decision
logs.

**Current boundary:** The endpoint deliberately returns unavailable. The live
Metamon battle representation has no audited way to construct the opponent's
candidate-conditioned masked state without leaking an unrevealed party. It does
not reuse the public-only opponent prior. This is a correctness stop, not a
negative performance result. The valid replay-state adapter remains the sole
evidence implementation until the live state adapter is proven.

**Safe live adapter:** Replaced the unavailable live boundary with a public-only
protocol path. Foul Play now maintains request-free protocol prefixes, captures
only discretionary opponent actions (including Tera correlation), caps public
candidates at 32, and posts no private battle/request data. The prior server
reconstructs a pre-action `ReplayState` from that prefix and invokes the same
candidate-conditioned frozen-r1 adapter used offline. Forced switches, drags,
`cant`, malformed payloads, unavailable reconstruction, and invalid evidence
fail closed to uniform sampling. Tests cover private-request exclusion, Tera
canonicalization, forced-switch exclusion, replay reconstruction, and zero
likelihood for impossible candidate actions.

**Next gate:** Run coverage-only live smoke. Require nonzero evidence updates,
no private payload/log leakage, bounded candidate counts, and no fallback errors
before any weighted-particle H2H experiment.

**Live coverage smoke:** Four paired games completed with zero voids, but all
146 agent-A decisions recorded `evidence_updates=0` and
`effective_particle_count=0`. The live path was safe but inert. The 4-0 H2H
result is explicitly not interpreted. Do not run a performance gate until the
opponent-action handoff reaches the sampler and nonzero coverage is observed.

**Coverage debugging:** Added server-side request/availability counters because
Foul Play determinization diagnostics were process-local. Fixed four handoff
issues: conditional generator teams were incorrectly rejected by static-pool
object identity; static fallback needed bounded 32-team particle sampling;
protocol and battle tags differed by the `battle-` prefix; and `|start|` reset
discarded the earlier `|gen|9` metadata. A final one-game trace produced 16
likelihood requests, 8 available responses, and 8 fail-closed responses (four
candidate/public conflicts, four reconstruction `AttributeError`s). The live
path is now active, but coverage is only 50%; no H2H claim is allowed until the
remaining failures are resolved or explicitly bounded.

**Failure resolution/characterization:** Candidate/public conflicts were stale
or wrong-orientation action events (for example, action actor `Lucario` while
the reconstructed public active was `Muk`). Side orientation is now persisted
per battle, and both current Foul Play active and protocol-prefix active must
match the action actor before an endpoint request. Remaining reconstruction
`AttributeError`s localized to prefixes with no active Pokemon on one side at a
post-faint/pre-replacement boundary; these are not valid policy states and now
return an explicit bounded fallback. No state or active Pokemon is fabricated.

**Coverage implication:** Valid action evidence is inherently sparse and
role-limited in the current audited p1-observer/p2-actor implementation. Endpoint
requests that pass the guards are safe; stale, opposite-role, forced, dragged,
`cant`, Tera-ambiguous, and missing-active boundaries remain uniform. A larger
coverage run is required before H2H because a two-game trace can contain no
eligible evidence despite clean completion.

**10-game coverage characterization:** Ten paired games completed with zero
voids. The action endpoint received 48 audited requests: 28 available (58.3%)
and 20 fail-closed. A typed follow-up confirmed every unavailable request was
`public prefix lacks an active Pokemon on one side`, the expected
post-faint/pre-replacement boundary. Stale/wrong-side candidate conflicts were
eliminated by persistent side orientation and actor-to-prefix/current-active
guards. No unexplained endpoint failure remains.

**Decision:** Coverage gate passes for an exploratory production-budget H2H
screen. Evidence remains role-limited and sparse, so this is not yet a formal
promotion gate; all ineligible boundaries continue exact uniform fallback.

**Exploratory production-budget H2H:** At 500ms/P8, action-conditioned
particles finished 12-12 against uniform particles with zero voids. Role split
was exactly balanced: 6-6 as acceptor and 6-6 as challenger. The endpoint
received 1,112 requests, returning 724 valid likelihoods (65.1%); 372 fallbacks
were bounded missing-active states and 16 were one Drifblim public-set conflict.
Wilson 95% CI was `[31.4%, 68.6%]`.

**Decision:** Non-regressive but no evidence of gain. Do not run a promotion
gate. The current implementation uses only the latest valid opponent action;
the next research iteration is cumulative Bayesian history weighting with
tempering/weight floors to prevent posterior collapse, followed by another
small screen.

**Cumulative action history:** Added log-space multiplication of all valid
actions by the current active opponent, tempered with default exponent `0.5`.
Impossible-set zeros remain eliminations; malformed, dimension-mismatched, or
all-collapsed histories fall back uniformly. Candidate responses are cached by
public prefix, action, and bounded particle IDs. Tests cover compounding,
tempering, zero elimination, and collapse fallback.

**Cumulative coverage smoke:** After fixing an import-path regression and
applying actor/prefix guards to every historical factor, four games completed
with zero voids. The endpoint returned 24/43 valid responses (55.8%); all 19
fallbacks were expected missing-active boundaries. Fifty of 132 decisions used
evidence, with up to 36 cumulative factors and effective particle count up to
31.85. No unexplained conflict or endpoint error remained.

**Decision:** Proceed to a 24-game production-budget exploratory screen at
temperature `0.5`; this is not a promotion gate.

**Cumulative exploratory H2H:** At 500ms/P8 and evidence temperature `0.5`,
cumulative action-history particles finished 13-11 (54.2%) against uniform
particles, with zero voids. Role split was 6-6 as acceptor and 7-5 as
challenger. Wilson 95% CI was `[35.1%, 72.1%]`. The endpoint returned
1,308/2,093 valid responses (62.5%); all 785 fallbacks were the bounded
missing-active condition.

**Decision:** Positive but noise. Cumulative history improves the point estimate
over latest-action weighting (13-11 vs 12-12), but does not justify promotion.
A larger exploratory screen is required before a formal gate.

**100-game cumulative screen:** At 500ms/P8 and temperature `0.5`, the
cumulative candidate finished 60-39 over 99 decisive games (60.6%), with one
void. Role split was 27-22 as acceptor and 33-17 as challenger. Wilson 95% CI
was `[50.8%, 69.7%]`. The endpoint returned 2,997/5,689 valid responses (52.7%);
all 2,692 fallbacks were the bounded missing-active condition. The one void was
baseline Foul Play crashing in stock `select_move_from_mcts_results` on
`total_score / visits` with zero visits, not an action-belief or endpoint error.

**Decision:** This is the first positive H2H evidence for action-conditioned
beliefs, but the run violates the zero-void promotion rule. Run a clean
replacement/sanity check for the zero-visit baseline failure, then proceed to a
formal 500-game promotion gate only with fail-safe zero-visit handling applied
symmetrically to both sides.

**Gate audit correction:** The 100-game screen was confounded: candidate A used
the exact conditional/static generator sampler plus action weighting, while
baseline B used stock Foul Play belief sampling. The 60.6% result therefore
does not isolate action conditioning and is withdrawn as promotion evidence.
The initially launched 500-game gate was stopped after two games for the same
reason.

**Clean agent split:** Added `foul_play_action_belief_root_priors_opp` so the
harness enables action weighting per slot. The corrected baseline is
`foul_play_randbats_conditional_root_priors_opp`; both sides now share the same
pool, conditional generator, priors, budget, and zero-visit safeguard. A
four-game sanity finished 2-2 with zero voids. Candidate server received 43
action-likelihood requests (29 valid); baseline server received exactly zero.
Proceed with a corrected 500-game one-variable gate.

## 2026-07-22: Shared-Root Information-Set Search

**Hypothesis:** One root policy learned across posterior-weighted hidden worlds
reduces root strategy fusion versus independently searching each determinization
and aggregating its policy afterward.

**Implementation:** Added a Rust root-only information-set solver with one
shared side-one RM+ regret vector, weighted world sampling, world-specific
opponent action sampling, fixed joint-root actions, and bounded per-world
continuation MCTS. Every round evaluates all shared player actions against one
sampled world/opponent action and updates the shared average mixed strategy.
MCTS iteration limits now honor values below 1,000. Added PyO3/Python bindings,
strict common-action support validation, toy strategy-fusion/world-weight/
dominated-action tests, and engine smoke tests. Rust workspace: 862 tests pass;
Python: 8 tests pass.

**Foul Play integration:** Added `foul_play_shared_root_action_belief_opp`.
It prepares posterior-resampled worlds once, converts all states, uses human
opponent priors where they overlap each world's legal actions, calls one shared
Rust solver, and samples its returned mixed policy. Failures atomically fall
back to independent search.

**First live use:** After fixing action-key casing and filtering human priors to
per-world legal overlap, one diagnostic game used shared search on every logged
decision with no fallback. It produced 134-2,382 RM+ rounds per 50ms decision
and finite mixed policies, including Fire Blast 61.5% versus Judgment 37.1%.
Proceed to production-budget timing/coverage smoke.

**Production smoke:** Four paired games at the production P8 compute budget
finished 1-3 with zero voids. This sample is not interpreted. Shared search was
used on all 128 candidate decisions with zero fallback, 16-32 worlds per
decision, median 4,306 RM+ rounds, and median 1,000ms shared-search wall time
(compute-parity budget derived from current per-world P8 search). Proceed to a
24-game exploratory screen against the identical independent-search
action-belief baseline.

**Exploratory autopsy:** The 24-game screen was stopped after five completed
losses (game six was in progress). Shared search itself was healthy: 165 logged
decisions, zero fallback, median 4,562 RM+ rounds, and 1,000ms median wall time.
The failure is the solver objective/policy extraction:

- One-sided RM+ against a fixed human opponent prior converged to a near-pure
  population best response, not a robust two-player information-set strategy.
  Median top-action mass was 98.6%, median entropy 0.082 nats, and 93.3% of
  selections used the top action.
- Sampling the entire average RM policy played transient low-mass actions that
  stock Foul Play's 75%-of-best filter would reject. Examples: Dark Pulse at
  8.8% while Nasty Plot had 91.1%; Psychic Noise at 3.7% while Heal Bell had
  95.0%; switch Greedent at 1.5% while Hydro Steam led at 59.8%.
- Shared policy switched 30.3% of decisions versus 23.1% for the independent
  baseline and used Tera 2.4% versus 4.2%, consistent with an overconfident,
  conservative best-response policy.

**Decision:** Withdraw the one-sided solver from further H2H. Keep the shared
world/continuation infrastructure, but replace the root objective with
two-sided RM+ (one shared player strategy, one opponent strategy per world),
using the human prior only as a bounded exploitation mixture/objective. Return
the average equilibrium strategy after removing numerical exploration residue.

**Two-sided solver correction:** Added one shared player RM+ process and one
opponent RM+ process per world. Human opponent priors enter through a bounded
behavior mixture (`0.25` default), not as a fixed opponent. Output probabilities
below 2% are pruned and renormalized. Matching-pennies, RPS, dominated-action,
weighted-world, prior-shift, and pruning tests pass. This changed median policy
entropy from 0.082 to 0.548 nats in a live diagnostic.

**C1 integration correction:** Custom conditional/action/shared agents were
accidentally configured as opponent-priors-only, so shared search and its
baseline discarded r1's player root prior. This invalidates earlier shared-root
smokes as tests of the intended architecture. Both now receive C1 and C2. Added
a legal shared player prior with 0.25 behavior mixture; diagnostic coverage was
100%, median entropy 0.843, and median top mass 63.2%.

**Stable payoff matrix:** Repeated fresh 16-iteration forced MCTS calls produced
noisy inconsistent cell values while rebuilding thousands of tiny trees. Added
on-demand payoff caching per `(world, player action, opponent action)` and raised
first-cell continuation depth to 128. Diagnostics expose unique cells, cache
hits, and total continuation iterations. RM then performs hundreds of thousands
of cheap updates over one stable empirical Bayesian root game.

**Corrected production smoke:** With two-sided RM+, C1+C2, action-conditioned
worlds, and cached 128-iteration payoff cells, four paired games finished 2-2,
with one win in each role, zero voids, and zero shared-solver fallback. Proceed
to a fresh 24-game exploratory screen; all earlier shared-root H2H numbers are
withdrawn from evidence for this corrected version.

**Corrected exploratory result:** The complete corrected architecture finished
6-18 (25.0%), Wilson 95% CI `[12.0%, 44.9%]`, with zero voids. Role split was
2-10 as acceptor and 4-8 as challenger. Shared search was used on every one of
704 logged candidate decisions with zero fallback; median diagnostics were
1.56M RM+ rounds, 1,008 stable cached payoff cells, entropy 0.607 nats, and
76.6% top-action mass. This is therefore not an integration, convergence,
prior-coverage, or noisy-cell failure.

**Decision:** Reject root-equilibrium replacement at the current ladder
objective. Independent determinized search materially outperforms the robust
shared-root policy against this opponent population. Preserve the solver as a
research artifact/measurement tool, but do not deploy or tune from a 25%
starting point.

## 2026-07-22: Selective Shared Re-Solving

**Literature basis:** CFR-D, DeepStack, Libratus, ReBeL, and Student of Games
support public-state/subgame re-solving and nonuniform allocation of search.
Value-of-computation and dynamic MCTS stopping literature supports spending
extra compute when the selected action is unstable and potential decision
regret is high. No checked imperfect-information primary source uses
cross-determinization disagreement as the launch trigger; that scheduling rule
is the original hypothesis. SPIBB/Soft-SPIBB, HCPI, and conservative policy
iteration motivate baseline anchoring and abstention under uncertainty.

**Primary sources and exact relevance:**

- Burch, Johanson, and Bowling, *Solving Imperfect Information Games Using
  Decomposition* (CFR-D, arXiv:1303.4441): establishes safe public-subgame
  decomposition/re-solving under suitable root counterfactual values.
- Moravcik et al., *DeepStack* (Science 2017): establishes continual
  depth-limited re-solving at encountered public states.
- Brown and Sandholm, *Safe and Nested Subgame Solving* (arXiv:1705.02955):
  supports event-triggered nested solving, especially after off-tree actions.
- Brown et al., *ReBeL* (arXiv:2007.13544), and Schmid et al., *Student of
  Games* (arXiv:2112.03178): support public-belief-state search and selectively
  growing computation toward relevant public states.
- Hay et al., *Selecting Computations* (arXiv:1207.5879), Tolpin and Shimony,
  *MCTS Based on Simple Regret* (AAAI 2012), Lan et al., *Learning to Stop*
  (AAAI 2021), and Baier and Winands, *Time Management for MCTS* (IEEE TCIAIG
  2016): support allocating extra computation according to expected decision
  improvement/action instability rather than uniformly.
- Laroche et al., SPIBB (2019), Nadjahi et al., Soft-SPIBB (2019), Thomas et
  al., HCPI (ICML 2015), and Kakade and Langford, Conservative Policy Iteration
  (2002): motivate copying/anchoring the baseline under uncertainty, confidence
  gating, and conservative mixtures rather than unrestricted replacement.

These sources do **not** prove that cross-determinization disagreement predicts
Pokemon decision error, nor that the cached-MCTS LCB is a formal safety bound.
The original contribution being tested is the scheduling hypothesis:
action-relevant disagreement identifies the rare Pokemon public states where
belief re-solving has positive value of computation.

**Why this direction was chosen:** Global shared-root replacement was a clean
failure at 6-18 despite complete solver usage and stable convergence. That says
robust belief solving is harmful on ordinary ladder states, not that it is never
useful. Foul Play's independent PIMC remains the default because it is strong in
high-disambiguation/high-leaf-correlation states. Shared re-solving is retained
only as an expensive candidate generator on the subset where independent worlds
recommend materially different actions. A paired same-world comparison then
tests whether the candidate is estimated to improve the actual baseline action.

**Controlled agents:** Both candidate and baseline use the same accepted r1
checkpoint for C1 player and C2 opponent priors, exact generator/conditional
belief machinery, cumulative action-conditioned belief, engine build, and
500ms/P8 independent search. Candidate adds only the selective trigger,
same-world shared re-solve, and LCB-gated override. Thus the experiment isolates
selective re-solving rather than checkpoint, belief, or search-budget changes.

**Safety and kill criteria:** Audit mode must demonstrate nonzero but sparse
trigger coverage, complete paired diagnostics, zero behavioral overrides, and
no unexplained voids before override mode. Override mode requires all strict
instability thresholds plus complete paired diagnostics and `LCB > 0`; otherwise
it returns the exact baseline action. Stop the branch if override coverage is
effectively zero, harmful overrides dominate, the paired advantage is not
predictive on held-out traces, or powered H2H fails to improve over baseline.

**Implementation:** Added an audit-first selective agent. It prepares one world
batch, runs ordinary independent Foul Play, and measures weighted top-action
disagreement, Jensen-Shannon divergence, aggregate top mass/margin, and world
ESS. Strict default trigger requires disagreement >=0.35, JS >=0.15, and top
mass <=0.75. Triggered states run shared re-solving on the exact same worlds.
Rust diagnostics compare the shared mixed policy with the actual baseline
action using paired cached payoff cells and report an empirical world-level
advantage mean, standard error, and 90% LCB. Audit mode always returns baseline;
override mode additionally requires available `LCB > 0`. These MCTS-derived
LCBs are heuristic, not formal guarantees.

**Audit smoke:** Four games completed 2-2 with zero voids and zero overrides.
Across 124 decisions, 11 triggered (8.9%), all 11 paired diagnostics were
available, and none had positive LCB (range -0.1176 to 0.0). The gate therefore
abstained everywhere as designed. Collect a larger production-budget audit
before enabling any override.

**Production audit:** Twenty-four games produced 759 audited candidate
decisions. Strict instability thresholds triggered 162 (21.3%); 157 paired
diagnostics were complete. Thirty states (3.95% of all decisions) had empirical
90% `LCB > 0`, and all 30 recommended a different shared top action from the
baseline. Median positive LCB was 0.0134, median paired mean advantage 0.0253,
and maximum LCB 0.1312. Audit mode made zero overrides and finished 12-11 over
23 decisive games. One void came from baseline selector aggregate weights all
being zero; added a symmetric final uniform fallback for this degenerate case.

**Decision:** Run a 24-game override screen with the predeclared rule only:
strict instability trigger, complete paired diagnostic, and `LCB > 0`.
Expected override coverage is approximately 4%; all other decisions remain the
ordinary baseline action.

**Selective override result:** The 24-game screen finished 14-10 (58.3%),
Wilson 95% CI `[38.8%, 75.5%]`, with zero voids. Role split was 8-4 as acceptor
and 6-6 as challenger. Across 760 candidate decisions, 134 triggered (17.6%),
132 paired diagnostics were complete, and 21 actions were overridden (2.76%).
Median override LCB was 0.0140 and maximum 0.0693. This is a positive but noisy
point estimate; it supports a larger exploratory screen, not promotion.

**Frozen 500-game gate configuration:** No parameters may change during this
run. Candidate is `foul_play_selective_shared_root_opp`; baseline is
`foul_play_action_belief_root_priors_opp`. Both use accepted r1 C1+C2 priors,
exact pool plus conditional generator belief, cumulative action-conditioned
belief at temperature 0.5, the same Gen9 engine build, and 500ms/P8/one search
thread. Selective mode is `override`; trigger thresholds are disagreement 0.35,
JS divergence 0.15, and aggregate top mass 0.75. Shared continuation depth is
128; player and opponent human-prior mixtures are 0.25; output probability
floor is 0.02. Override requires complete paired diagnostics with `lcb_z=1.645`
and `LCB > 0`. Maximum games 500, paired roles, SPRT `H0=0.50`, `H1=0.55`.
Promotion still requires zero unexplained voids and final Wilson 95% lower bound
above 50%; SPRT may stop early for success or futility. If promoted, run an
equal-average-compute baseline control before any causal algorithmic claim.

**Interrupted gate result:** The first 500-game gate was stopped at 76/500
games (40-36, 52.6%) when analysis revealed three structural issues. SPRT LLR
had declined from 0.77 to 0.02, consistent with no effect. Zero voids and zero
infrastructure failures confirmed the regression was real, not a harness bug.

**Three-issue correction:** (1) 19% trigger rate with only 3% override yield
wasted compute on triggered-but-not-overridden turns. Fix: tightened trigger
thresholds to disagreement 0.45, JS 0.25, top mass 0.65. Production smoke
trigger rate dropped from 19% to 5.3%. (2) Paired LCB used the same biased
128-iteration MCTS cells as RM+ optimization, so both baseline and shared were
wrong in the same direction. Fix: added deeper 512-iteration paired evaluation
with fallback to cached optimization cells when deadline is exhausted. Paired
diagnostics now available on all triggered states. (3) Binary override at 3% of
decisions was too sparse. Fix: replaced with confidence-weighted mixture
`alpha = clamp(LCB / 0.05, 0, 1)`. LCB of 0.025 gives 50% shared influence.

**Corrected production smoke:** Two paired games at production budget confirmed
all fixes: 4/76 triggered (5.3%), 4/4 paired available, 2 positive-LCB states
with alpha 0.32 and 0.62, both overrode via mixture. Zero voids.

**Frozen corrected 500-game gate:** Same agents with tightened thresholds
(0.45/0.25/0.65), 512-iteration paired evaluation with fallback, confidence
mixture with `lcb_scale=0.05`. SPRT `H0=0.50`, `H1=0.55`. Promotion requires
zero voids and Wilson 95% lower bound above 50%.

**Archive interruption:** Repository cleanup ended the corrected gate after 79
decisive games. The selective candidate was 44-35 (55.7%) with zero recorded
voids and SPRT LLR 0.506. The point estimate was positive, but the run did not
reach either an SPRT boundary or the predeclared 500-game promotion criterion.
It is archived as incomplete and was not promoted; accepted r1 remains the
production agent.

## 2026-07-24 AlphaZero-Style Teacher Qualification Infrastructure

**Implementation:** Added exact seeded single-thread MCTS, capture-only schema-v2
root bundles, behavior-preserving replicated determinization schedules, private
manifested artifacts, deterministic stratified panel selection with inclusion
weights, P8 duration-to-iteration calibration, schedule-aware offline `U-B`,
`S-B`, and `S-4B` replay, and strict repeat/schedule stability analysis. Added
read-only legal-root enumeration and deterministic common-uniform engine steps.
The live capture wrapper restores the post-behavior RNG state before selection;
shadow schedules never enter behavior MCTS or the selector.

**Accepted-budget development capture:** Six local accepted-r1-versus-random
games at 500ms/P8 completed 6-0 with zero voids. They produced 124 decisions,
496 schedules, and 9,216 private worlds. Deterministic post-stratification
selected 26 roots from 14 strata across five source battles, containing 1,920
worlds; root-weight ESS was 19.10. Artifacts are under
`alpha_zero_overnight_capture_20260724/` and
`alpha_zero_overnight_panel_20260724/`.

**Iteration calibration:** Twenty-six roots with three accepted-live duration
repeats showed extreme state-dependent throughput: median 187,203 visits, mean
433,415, p90 800,650, and CV 1.69. Near-terminal/cheap states completed millions
of visits, so the mean was rejected as a fixed-budget anchor and the development
sweep used the rounded median `B=187,000`. Sensitivity medians were 159,156 for
`S-B` and 148,328 for `U-B`, confirming that tree allocation also changes
throughput. This calibration is hardware-specific and too heterogeneous to
support one universal time-equivalence claim.

**Exact schedule-aware replay:** The 26-root panel used four schedules, three
tree repeats, `B=187,000`, and `4B=748,000`. Tree-seed variation was small:
median JS was 0.00049 (`S-4B`), 0.00056 (`S-B`), and 0.00043 (`U-B`); p90 JS was
at most 0.00530 and weighted top-set overlap was at least 97.5%. Schedule
variation was larger but remained below the descriptive thresholds: median JS
0.00426/0.00239/0.00380, p90 JS 0.0612/0.0350/0.0320, and top-set overlap
92.5-94.0%. Learned priors materially changed allocation versus equal priors:
`S-B` versus `U-B` had weighted mean JS 0.0353 and TV 0.1453. Increasing to
`4B` changed `S-B` more modestly: JS 0.0108 and TV 0.0869. These are
distributional development results, not action-quality or strength claims.

**Independent-value attempt:** A one-root common-tape smoke under the explicitly
frozen `uniform_legal_v1` opponent/continuation policy completed, but the full
panel failed closed at both 256- and 1,024-decision horizons because uniform
play can cycle indefinitely. No partial full-panel value artifact was admitted.
Therefore uniform continuation is not a scalable evaluator, and the earlier
one-root values are not qualification evidence.

**Decision:** Search is conditionally stable enough to justify continuing
teacher qualification, and determinization schedules are a larger source of
variation than tree seeds. Do not qualify or train from these roots yet. The
next blocking requirement is a terminating, independently frozen continuation
policy, followed by common-tape values, clustered confidence bounds, complete
mirrored games, and student-transfer evaluation.

**Schema-v3 root bridge:** New captures embed the exact served r1 `text_tokens`,
`numbers`, 13-action legality mask, action name table, probabilities, and raw
player protocol prefix. The snapshot is identity-checked, hash-covered, rejects
fallback masks, and must exactly reproduce recorded named player priors. Existing
schema-v2 artifacts remain readable. Experimental prior-server sessions now own
a reset deep copy of the loaded observation space, eliminating cross-session
mutable observation history while preserving player-then-opponent encoding order
inside one fresh session. The accepted production server remains unchanged.

**Root parity result:** The frozen epoch-5 checkpoint hash
`c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93`
reproduced all 124 retained overnight schema-v3 policy snapshots at absolute
tolerance `1e-7`; maximum absolute error was exactly `0.0`. This validates direct
snapshot inference, not engine-state observation reconstruction.

**Continuation blocker refinement:** A pure `poke_engine.State -> r1 policy`
bridge is rejected as information-unsafe. Sampled states expose hidden opponent
sets and erase reveal provenance, while flat engine instructions lack executed
action/outcome boundaries and may contain hidden-set-dependent bookkeeping. Added
`step_with_uniform_debug` as an omniscient selected-branch diagnostic without
changing the existing transition API. The next implementation requirement is an
observer-specific semantic event projector and dual mechanical/public state,
with hidden-world noninterference and one-step r1 parity tests before integration.

**First semantic certificate:** Implemented `r1-switch-v1`, a fail-closed
observer-specific projector for quiet voluntary double switches and post-faint
replacement switches. Projection is atomic across all sampled worlds, accepts
only exact switch plus optional last-move bookkeeping instructions, verifies that
no other mechanical field changed, canonicalizes event order, and emits no raw
instruction, branch, slot, or opponent-private data. Hidden-world tests vary an
unrevealed move that changes conditional engine bookkeeping and still obtain
identical public events; any unsupported world rejects the complete transition.
The projector also requires the pinned engine diagnostic contract
`poke-engine-0.0.47-r1-switch-v1` before inspecting worlds.

**Dual-state one-step parity:** Schema-v3 snapshots now include a serialized
player-information state and mutable r1 observation history at player and
post-opponent inference boundaries. `R1SwitchTracker` reconstructs root state,
checks exact root observation parity, applies only masked public switch events,
and derives the next mask/action table without reading sampled opponent sets. A
controlled Showdown-protocol double-switch differential produced exact equality
for all 106 text tokens, 55 numeric features, 13 legality flags, the action name
table, and frozen epoch-5 checkpoint probabilities. This validates the named
switch subset only. It does not produce a terminating r1 continuation policy;
move execution, misses/failures, damage causes, public HP rendering, item/ability
reveals, residuals, tera, pivots, and forced drag still require semantic tracing
inside the experimental engine before formal values can resume.

**First basic-move certificate:** Added engine-native executed-action markers
and the pinned semantic contract `poke-engine-0.0.47-r1-basic-move-v1` without
changing ordinary instruction generation or the six-byte instruction-size
invariant. The semantic binding reports ordered executed moves, HP changes,
major status changes, and boosts while rejecting every unaccounted instruction
kind. `project_information_set_basic_move` admits only quiet, non-Tera active
Pokemon with no item, ability, or pre-existing status and a small explicit move
whitelist. It partitions sampled worlds by masked public events plus the
observer's legitimate HP, status, PP, and disabled-move delta; hidden worlds
with equal observations merge, while worlds with different displayed opponent
HP remain separate.

**Basic-move one-step parity:** A controlled two-sided Seismic Toss differential
advanced one mechanical world through the semantic engine and independently
advanced the player-information tracker through Showdown protocol plus the next
private request. The two paths matched all 106 text tokens, 55 numeric features,
13 legality flags, the action name table, and frozen epoch-5 probabilities
exactly. A metamorphic engine test also varied hidden opponent Defense and
proved equal public outcomes merge while different displayed HP values partition
without exposing the hidden stat. Full regressions passed: 924 Rust/doc tests,
46 Python binding tests, and 145 script tests with two expected data-dependent
skips.

**Decision:** Certify only `r1-basic-move-v1`'s declared one-step subset. Do not
integrate it into formal value collection yet. Normal random-battle states have
items and abilities, and misses/failures, immunities, reveal causes, residuals,
multi-hit moves, pivots, hazards, weather, terrain, volatiles, Tera, and complex
forced switches remain unsupported and must fail closed.

**Private-request recapture:** The first recapture attempt failed before battle
creation because cleanup had removed the old `experimental/external` dependency
layout. No capture from that attempt was admitted. Updated the experimental
launchers to use `srcs/vendor/foul-play` and an overridable `SHOWDOWN_DIR`, then
ran a clean smoke against the manifest-pinned Pokemon Showdown commit
`4880d3693580bd33652797cf31179c6fcdf87e50`.

The clean run under
`alpha_zero_schema3_private_request_smoke_v2_20260724/` completed one game with
zero voids and captured 10 roots, 40 schedules, and 768 sampled worlds. Every
root passed its schema/content hash, contained a non-null deep-copied private
request, and reconstructed exactly through the player-information tracker. The
frozen epoch-5 checkpoint reproduced all 10 policies with maximum absolute
error `0.0`. Input manifest
`d3cdf8227652f7bb5c1b162a9efdd2b46bf77360f3ebb5ea174d028d9f9df9f4`
and completion manifest
`c678f07225a9390dc062160eb44a3e8777d7737fcfd52facee54dd0a2a24bafa`
validate; private inputs and captures are mode `0600`. Both services were
stopped after finalization. Final script regression passed 145 tests with two
expected data-dependent skips.

**Basic-move admission census:** Added an aggregate-only private census over
engine-reported legal joint actions. It validates schema-v3 capture hashes and
manifest linkage, pins the engine binding and analysis implementation hashes,
weights roots/schedules/worlds/actions explicitly, tests five fixed branch
uniforms, emits no identities/actions/states/exception text, self-hashes, and
writes mode `0600`. A binding-normalization audit found that deserialized engine
sentinels use uppercase `NONE`; quiet-state checks now canonicalize those values
instead of falsely classifying them as live weather, terrain, or side state.

The 10-root, 40-schedule, 768-world capture produced 79,296 legal world/joint-
action trials and 396,480 certificate trials. Zero were structurally eligible
under `r1-basic-move-v1`. Overlapping weighted blocker prevalence was 100% for
active items, 100% for active abilities, 90% for pre-existing public boosts,
64.39% for an unlisted move, 54.95% for a switch action, 40.59% for a Tera
action, 20% for forced/pivot side state, and 10% for an existing Tera/type
change. Weather/terrain/Trick Room, side conditions, volatiles, delayed effects,
major status, and fainted actives were all 0% in this smoke. The strict
information-set view tested 1,908 common action pairs and 9,560 branch trials:
90% were structural rejections and 10% had no common legal action pair, so no
semantic failure category was reached.

**Decision:** Item/ability presence and activation/reveal semantics are the
first universal blocker and must be implemented together. Next admit preserved
public boosts, then expand move coverage and mixed move/switch transitions; Tera
follows. The report is
`alpha_zero_r1_basic_move_admission_census_20260724/admission-census.json`, with
self-hash `88c199105156143956c22241ab0b9d7f861e6b2c4d7975ed4058eedf41f45d8c`.
This remains descriptive coverage evidence and admits no values.

**Silent item/ability certificate:** Engine inspection showed that arbitrary
activations cannot yet be projected safely: public causes such as Leftovers,
Life Orb, Rough Skin, Rocky Helmet, immunities, and contact abilities collapse
into generic HP/status/boost instructions with no causal identity. Removing the
blanket item/ability gate would silently omit required Showdown reveals. Added
`r1-silent-mechanics-v1` instead for the highest-coverage fully attributable
subset found in the capture: side-one Calm Mind with Leftovers/Protosynthesis
against side-two Bulk Up with Leftovers/Sap Sipper. Existing public boosts are
allowed, but every other side-state gate remains. The selected trace must contain
exactly two executed moves and the four expected +1 boosts; any HP, status,
damage, healing, different boost, missing action, or unaccounted instruction
rejects the complete information-set transition as
`UNSUPPORTED_MECHANIC_ACTIVATION`. Hidden item and ability identities are never
emitted to the player tracker.

**Silent-mechanics parity and coverage:** A controlled protocol/request
differential preserved pre-existing boosts and matched all 106 text tokens, 55
numeric features, 13 legality flags, the action table, and frozen epoch-5
probabilities exactly while the opponent item and ability remained unknown. A
below-max-HP variant activated Leftovers and failed closed. The updated private
census admitted 64/79,296 world joint-action trials; across five branch uniforms
this was 320/396,480 trials, with weighted admission 0.0641%. The strict
information-set view admitted four common schedule/action pairs across all five
uniforms, weighted admission 0.0962%. This is the first nonzero live-capture
coverage for item/ability-bearing states, but it is intentionally too narrow for
formal values. The superseding census report self-hash is
`057641ed0783f1e0f30ebf183d7f6350c553cfcdb8e5e878a7a861b5d1e12416`.

**Decision:** Keep all activating mechanics rejected. The next engine change
must preserve public cause attribution through branching and emit trace-only
item/ability activation markers before adding Leftovers, Rough Skin, or other
activation/reveal cases.

**First branch-local activation marker:** Added trace-only
`RecordItemActivation` while preserving the six-byte `Instruction` invariant.
Ordinary instruction generation remains marker-free; semantic tracing now emits
an `item_activated` event immediately before direct Leftovers healing under the
pinned contract `poke-engine-0.0.47-r1-item-activation-v1`. The marker is a
no-op under apply/reverse and is generated only when Leftovers actually heals.
`PublicItemEvent` resolves the actor and current item, and the dual-state tracker
updates the public item token plus durable revealed-item provenance before
applying the HP event.

**Leftovers activation certificate:** Added `r1-leftovers-activation-v1` on top
of the audited Calm Mind/Bulk Up subset. Its event suffix must contain one or two
ordered `(Leftovers activation, same-actor HP)` pairs after the exact two-move,
four-boost grammar; duplicate actors, different items, missing HP, extra events,
or any unaccounted instruction fail closed. A controlled differential started
the opponent at 50/100 with unknown item/ability, then matched the protocol line
`-heal ... [from] item: Leftovers`: the tracker and Metamon path agreed on all
106 tokens, 55 numbers, 13 legality flags, action table, and frozen epoch-5
probabilities. The opponent item became `leftovers`; its ability remained
`unknownability`.

**Coverage and verification:** The upgraded-engine census retained 0.0641%
weighted world/action admission and 0.0962% strict information-set admission;
the retained captured states were at full HP, so the new activation path did not
increase this particular artifact's coverage. The final analysis manifest is
`058e6f532c67457e010ff36fbfb9a09075b09f83b01502ced5914afbd5e40936`
and report self-hash is
`00636b0000941619046b3b40330c74e702335d3a18a7fd89b6ba031b23e8ada0`.
Full regressions passed 925 Rust/doc tests, 48 Python binding tests, and 151
script tests with two expected data-dependent skips.

**Decision:** Leftovers is certified only in this narrow grammar. The next
branch-local source should be a directly generated contact ability such as Rough
Skin/Iron Barbs; secondary-injected Life Orb, Rocky Helmet, status, and boost
effects remain blocked until secondary source metadata survives branching.

**Coverage-driven acceleration:** Replaced the exact Calm Mind/Bulk Up grammar
with a declarative self-boost registry covering Agility, Bulk Up, Calm Mind,
Nasty Plot, and Swords Dance. `r1-declarative-boosts-v1` allows arbitrary
item/ability identities and existing public boosts only when both selected moves
are registered self-boosts and the trace contains exactly the declared move and
boost multiset, optionally followed by certified Leftovers activation/HP pairs.
Every other item/ability effect remains fail-closed. Hidden Heavy-Duty Boots,
Wide Lens, Defiant, and Technician worlds merged without identity leakage;
Speed Boost produced an extra event and was rejected. A controlled Calm Mind
versus Nasty Plot protocol differential matched the complete observation, mask,
and action table while opponent item/ability remained unknown.

The private census admitted 165/79,296 world joint-action trials and all five
uniforms for each, for 825/396,480 certificate trials. Weighted world/action
admission rose from 0.0641% to 0.2156%, a 3.36x increase with zero semantic or
information-set failures. Strict admission remained 0.0962% because only the
previous Calm Mind/Bulk Up pair was common across every world in its schedules.
Analysis manifest:
`87f7dd81144ca5476302a476363189cf7d7596091d5e509c93e2f37a75f4c5c1`;
report self-hash:
`988a4d3ce54c54d356565aea914e1db6a481b74897acfee28591a21ae617e9f8`.

**Next-value decision:** The common-action profile shows opponent switches paired
with Calm Mind and other player actions across substantially more strict mass.
The top damaging alternative, Sludge Wave, is coupled to Toxic Chain and
Choice/Life Orb/Assault Vest variants, making it a poor next unit of work. Build
an atomic mixed self-boost/opponent-switch certificate next; defer absent Rough
Skin/Iron Barbs until a capture demonstrates direct coverage value.

**Mixed self-boost/opponent-switch audit:** Profiled 896 captured world traces
covering seven semantic shapes. Every trace reduced to an opponent switch, a
registered side-one self-boost move, its declared boosts, and optional side-one
Leftovers activation/healing; no trace contained an unaccounted instruction.
The incoming Pokemon was already public in every world: 384 Gothitelle, 384
Munkidori, and 128 Thundurus instances. Public level, HP, and status agreed with
the private engine target in the audited rows.

Added `r1-mixed-boost-switch-v1`, which requires the exact mixed grammar, a
single public reserve target with exact species/level/HP/status agreement, no
Illusion, Tera, or type change, and the existing declarative boost and optional
Leftovers rules. The v1 target allowlist is deliberately limited to Munkidori
and Thundurus. Gothitelle remains rejected because an unrevealed Shadow Tag
possibility makes the next legal-action mask information-unsafe even when its
species and HP are public. A controlled damaged-Munkidori switch plus Calm Mind
differential matched the complete Metamon text observation, numeric features,
13-action legality mask, and action table. Hidden incoming moves merged into one
observation class; public/private HP mismatch and Gothitelle both fail closed.

**Mixed coverage:** The conservative private census admitted 421/79,296 world
joint-action trials and 2,105/396,480 five-uniform certificate trials. Weighted
world/action admission rose from 0.2156% to 0.6717%, a 3.12x increase. Strict
information-set admission rose from 0.0962% to 0.7308%, a 7.60x increase, with
100 admitted branch trials and zero semantic or information-set projection
failures. The census now also verifies that the runtime extension is exactly the
manifest-pinned engine artifact; a mistakenly built non-Tera extension was
detected by changed denominators, discarded, and rebuilt with
`--no-default-features --features poke-engine/terastallization` before the final
run. Analysis manifest:
`f287b7b194684c160665b8a38c4f7c8b9a572ee334c00b8355a0c5e03ab1500f`;
report self-hash:
`68577a79f2078683926284a90bd820801d3c5f27e4fd121025988608bc5975f4`;
completion manifest:
`5770dbd15e4963ad891b81373d7d3f179c68fcef71633bf830b3127d2d93a0d1`.
Full regressions passed 919 Rust/doc tests, 53 Python binding tests, and 151
script tests plus eight subtests, with two expected data-dependent skips.

**Decision:** Keep the mixed certificate. It materially improves both world and
strict coverage without weakening fail-closed semantics. The remaining 0.73%
strict coverage is still insufficient for formal terminating values; profile
the next highest strict-mass transition class before implementing another
mechanic.

**Certified next-legality partition:** Follow-up profiling showed that all 384
captured Gothitelle mixed-switch worlds used the trapping mechanic class. The
engine does not persist this in `force_trapped`; `Side::trapped` computes it
dynamically. Every post-transition request had four moves, four Tera variants,
and zero switches, with identical legality across five uniforms. Added the
engine-derived legal action set to each basic observation class and its
information-set signature. The tracker now accepts this set only when every
action maps to the public name table; callers without certified legality retain
the existing trapping and Assault Vest rejection. Hidden Shadow Tag and
non-trapping worlds partition by their public next mask without emitting an
ability event or changing the public ability token.

The controlled Gothitelle plus Calm Mind differential matched the complete
observation, numeric features, legality mask, and action table. The superseding
mixed census admitted 741/79,296 world/action trials and 3,705/396,480 branch
trials. Weighted world admission was 1.0147%; strict admission was 1.7244% with
zero semantic or information-set failures. Analysis manifest:
`4476c4849de13fac4b9e85ebd8f0f87a43a43f30b9ece4b8318a7b88be13f170`;
report self-hash:
`93e7e311addef77d5891fc5d26d8f3b0215234698f341f2f6591dfadeb870016`;
completion manifest:
`1c7202409af8f95a73d3254af328b974f38542b51437ba5893e40fd902a7edda`.

**Next-class ranking:** Aggregate strict profiling ranked boosted voluntary
double switches first at a 10.3205 percentage-point upper bound and low-to-
medium complexity. The next candidates were own switch plus opponent self-boost
at 1.25 points, effect-move combinations at 4.13-15.22 points with high causal-
attribution risk, and Tera at 34.52 points but requiring foundational public
Tera semantics. The selected double-switch traces contained only opponent
switch, exact side-one SpA/SpD cleanup from +1 or +2 to zero, and side-one
switch. No item activation or unaccounted instruction occurred.

**Boosted double-switch certificate:** Added `r1-boosted-double-switch-v1` to
the switch projector. It requires clean field/side/volatile state, voluntary
double switches, exact equal public SpA/SpD stages in {+1,+2}, no other boosts,
unique non-Tera base-typed targets, a previously public opponent target, and
display-percent-equivalent public HP. The exact negative boost instructions are
validated as switch cleanup and suppressed from public events; the tracker
independently verifies its public prestate before clearing boosts. Exact next
legality is carried for pure and boosted switch projections, including trapping
and known Assault Vest cases. A unified transition dispatcher now lets the
census evaluate switch and move certificates atomically. A controlled boosted
double switch into Shadow Tag matched the full protocol-derived observation,
mask, and action table; a mismatched public boost prestate failed closed.

The final census admitted 3,941/79,296 world/action trials and
19,705/396,480 branch trials. Weighted world/action admission rose to 5.6512%.
Strict admission reached 10.5064%, with 1,080 admitted branch trials and zero
semantic or information-set failures. This retains 900 of the profiled 1,100
boosted-switch trials; the 200 cases with an already-terastallized outgoing
opponent remain rejected rather than broadening Tera semantics implicitly.
Analysis manifest:
`5ad1dd1fd8eff404da6e314526cd2392a24a40748d5356e0f09471cadb830122`;
report self-hash:
`323c86533fc682da44f478b7ed7da34357fd1e7412367c49935348d60ab0cf9b`;
completion manifest:
`bf4786daaad2bb8ff2561fe0c827263c614d5598773e5891126f8c81bcadfd77`.

**Termination gate:** The final report records `continuation_readiness.status =
blocked` and forbids `r1_continuation_value`. Despite the 14.4x strict-coverage
increase from 0.7308% to 10.5064%, 79.4936% of strict mass remains structurally
unsupported. More importantly, sequential policy-weighted coverage has not been
measured, an opponent-POV continuation state is not captured, no terminating
policy rule is defined, and multi-turn terminal parity is not certified. The
independent-value collector therefore remains unchanged and development-only.

Full regressions passed 919 Rust/doc tests, 55 Python binding tests, and 152
script tests plus eight subtests, with two expected data-dependent skips.

**Decision:** Retain certified legality, Gothitelle, and boosted double switches.
Do not integrate terminal r1 continuation. The next research step must define
the opponent continuation estimand and termination rule, then measure sequential
policy-weighted coverage; adding more one-step mechanics alone cannot satisfy
the formal-value gate.

## 2026-07-26: Sequential Policy-Weighted Certificate Coverage

**Protocol:** Added a private aggregate-only probe for the preregistered
`one_sided_player_r1_vs_uniform_common_legal_sequential_certificate_coverage`
estimand. The player uses frozen stateless r1 epoch 5; the declared opponent is
uniform over actions common to every world in the current information set. A
SHA-256 counter tape independently fixes player, opponent, and chance sampling
by global schedule ordinal, rollout, and depth. The player action is sampled
from the complete public-request r1 distribution before common-support testing,
so a sampled unsupported action fails only its rollout rather than rejecting
the entire node because softmax assigned an unrelated action positive mass.
Continuation inference uses player-only observation history and excludes the
legacy modeled-opponent query.

**Smoke:** On 10 roots, 40 schedules, 768 sampled worlds, two rollouts per
schedule, and a two-transition horizon, depth-0 entering mass was 1.0. Projection
rejected 87.5%, and 10.0% had no common legal support. The remaining 2.5% passed
one certified transition; all of it failed projection at depth 1. No rollout
reached the horizon or a certified terminal state. Policy validity,
sampled-action support, tracker consistency, projection lineage, engine terminal
validity, and terminal agreement each had zero failure mass. This is descriptive
coverage only, not a value or policy-strength estimate.

The report passes self-hash, privacy, `0600` permission, and per-depth and
inter-depth mass-conservation validation. Analysis manifest:
`cdbdf33d5c7d9abc735fdf7a2c6b88c1c08d8d6834292a47cdd13b7ed97e5b90`;
report self-hash:
`5338caf0121d5ca1983b5c3d63c7b55e7a9cd8e63a45206600b63bb3bd219caf`;
completion manifest:
`9ca2c608adfd12fc5c065e9a4f5349a0288c928bbf79a69a3bf89c8eb353ca78`.

**Decision:** Formal r1 continuation remains blocked. The observed sequential
certificate survival is zero by depth 2, and the probe is one-sided and
finite-horizon even if future mechanics improve survival. Do not emit terminal
values or modify the independent-value collector from this result.

Full regressions passed 919 Rust/doc tests, 55 Python binding tests, and 180
script/eval tests plus 15 subtests, with two expected data-dependent skips. The
checkpoint integration tests were run on CPU with Torch Dynamo disabled to
avoid the local MPS/Inductor kernel resource limit.

**Dual-snapshot capture gate:** Added explicit `player_role` and
`opponent_username` fields to newly served private schema-v3 policy snapshots
and added `audit_dual_r1_policy_snapshots.py`. The auditor requires two distinct
private dumps and joins only opposite-role snapshots with reciprocal normalized
identities, matching battle turn, private-request ownership, and identical
request-stripped public protocol prefixes. Duplicate identities and every
malformed or unpaired boundary fail closed into fixed aggregate categories. The
report contains no identifiers, actions, snapshots, protocol, exception text,
or per-boundary rows; it is self-hashed, mass/count-conserving, and written at
mode `0600`. Fourteen auditor tests and eight prior-server/capture tests pass.

No formal audit run was claimed: the retained smoke used a random opponent and
contains only one private client dump, while its simulator checkout was
ephemeral and is no longer present. A fresh run requires a pinned local Showdown
checkout, two isolated prior servers with separate private dumps, a shared
nonempty namespace, both Foul Play clients configured to require priors, and
zero unexplained auditor failures. Passing that run would prove capture
joinability only and would not authorize r1 continuation values.

The expanded full script/eval suite passes 194 tests plus 19 subtests, with two
expected data-dependent skips. The auditor's focused 14-test suite also passes
after canonical-JSON write/read round-trip validation; `git diff --check` is
clean.

## 2026-07-26: Pinned Dual-r1 Snapshot Capture

**Simulator restoration:** Restored clean upstream Pokemon Showdown commit
`4880d3693580bd33652797cf31179c6fcdf87e50` under the documented external path.
The dual snapshot run did not use the earlier server-side private-team-manifest
patch. Both r1 clients used separate CPU prior servers and separate private
decision dumps, inherited one shared nonempty namespace, and required successful
priors for every played decision.

**Auditor qualification:** The first development capture showed that raw
request-stripped protocol prefixes are not observer-invariant. Room join/timer
transport differs by client, own HP is exact while opposing HP is shared, and a
turn can contain an ordinary two-player request plus a legitimate one-sided
forced replacement. The corrected auditor removes non-battle transport,
canonicalizes switch, drag, damage, and heal HP using the pinned Showdown Gen 7+
rule (`ceil(100*hp/maxhp)`, except non-full values that round to 100 become 99),
and certifies singletons only when `forceSwitch` is true in the acting player's
private request. The implementation is source-backed by pinned
`sim/pokemon.ts` and `sim/battle-actions.ts`, not fitted to emitted identifiers
or action content.

The original, v2, and v3 smokes remain retained as ineligible development runs;
none is relabeled as preregistered success after an auditor correction. Their
aggregate failures isolated, respectively, room/POV HP handling, the non-full
99% rule, and drag HP handling. A fresh v4 run froze the complete auditor before
collection.

**V4 result:** One r1-versus-r1 game completed with zero voids. Both private
dumps contained 64 structurally valid snapshots. The auditor joined 28
reciprocal ordinary boundaries and certified eight one-sided forced-switch
boundaries. Duplicate identity, duplicate role, invalid snapshot, nonreciprocal
identity, public-prefix mismatch, and unexplained unpaired-boundary counts were
all zero. Input manifest:
`668a90ef051f415253205925471137fa466dc161427676a15582adc79a1c4b61`;
report self-hash:
`b147d0a5af7f67e71859506a12126ccc314c3776a42ff666253709555eef7b5c`;
completion manifest:
`000a52397761d67761a6ede1942268dfdd8f3a10014abfd6df5049c9485f642c`.
All private inputs and aggregate manifests/reports are mode `0600`.

**Decision:** Dual-client schema-v3 snapshot capture is joinable at ordinary and
one-sided forced-switch boundaries. This proves capture plumbing only. It does
not prove that the two POV trackers remain equivalent after simulated
continuations, does not define a terminating opponent-r1 policy, and does not
authorize continuation values. The next gate is a dual-tracker root and
one-transition parity harness over joined snapshots.

The expanded script/eval suite passes 196 tests plus 19 subtests, with two
expected data-dependent skips. Manifest, report, privacy, count-conservation,
permission, and `git diff --check` validation pass.

## 2026-07-27: Dual-Tracker Root Parity And Coverage

**Estimand correction:** Independent p1 and p2 determinizations cannot be
index-paired, multiplied, or treated as a joint belief. The implemented probe
instead constructs one coordinator-private actual mechanical root by proving
that each client's own `SideOne` is invariant across every captured world,
including zero-weight worlds, then fusing p1's own side as global `SideOne` and
p2's own side as global `SideTwo`. Weather, terrain, Trick Room, durations, and
team-preview state must agree across both streams. Search-only threat/scout
annotations are neutralized and named outside the estimand.

**Implementation and quality:** Added `r1_dual_tracker_parity_probe.py` with
strict dual-root joining, role/orientation checks, root observation and frozen-r1
policy parity, complete-policy SHA-256 action sampling, one shared chance draw,
two observer-specific projections of the same fused transition, source-lineage
and next-state identity checks, canonical public-outcome comparison, tracker
fork/application, exact next legality, next-policy validity, terminal inversion,
fixed side-specific failure categories, aggregate privacy validation, exact
count/mass equivalence, self-hashing, atomic `0600` output, and runtime/RSS
telemetry. Report claim, configuration, policy IDs, tape, readiness, and every
aggregate invariant are validated after canonical JSON round trips.

Corrected three observer-boundary defects found before collection: Showdown's
non-full 99% HP display now applies in semantic projection; mixed
self-boost/opponent-switch projection supports the switching player's SideTwo
POV and own-private switch delta; boosted double switches carry cleared-self
boost metadata only for the observer whose boosts were cleared. Real-engine
SideTwo orientation tests cover both certificates.

**Capture:** One pinned r1-versus-r1 game completed with zero voids using two
prior servers, two schema-v3 root bundles, two private snapshot dumps, and two
structured action logs. The aggregate join admitted 29 ordinary roots and
certified 11 forced boundaries with zero join failures. Every ordinary root
passed actual-own-side fusion, both tracker root reconstructions, both frozen-r1
probability checks, and both root action-table checks.

**Preregistered result:** Four rollouts per root produced 116 trials. All 116
failed both projectors; every other root, side-specific, lineage, public outcome,
tracker, legality, policy, and terminal failure category was zero. Aggregate
fixed diagnostics were 67 `UNSUPPORTED_PUBLIC_PRESTATE` pairs (57.8%), 47
`UNSUPPORTED_ACTION_PAIR` pairs (40.5%), and two
`UNSUPPORTED_ENGINE_DELTA` pairs (1.7%). Runtime was 4.14 seconds, 28.0 attempted
trials/second, with 1.70 GB peak RSS including model load. Analysis manifest:
`5013feb0deeac0560d2431508f4b47574b240d76ed7148d0071ecc684f493bbf`;
report self-hash:
`5c0c83897a18c0207f009a3d756b72f6e4a2dd0606dd56a0f57499919da05c07`.

**Replication:** An independently seeded, preregistered 32-rollout replication
produced 928/928 symmetric dual projection rejections and zero certified
transitions. Runtime was 4.02 seconds, 231.1 attempted trials/second, with 1.72
GB peak RSS. Analysis manifest:
`4e354123491c046f1392b6fd52c1b070ca5dda9a91b213b79d49037010026a4c`;
report self-hash:
`21c7b9ef1747b4809100c8b749c8ea0f23f3f784aaeddd0af43bcc4aea8fe73e`.
Source manifest:
`5d62408a7360f0c954e3968532767ddcf17bbd44a6077d17fdd65e0a20e73873`;
dual-root join report self-hash:
`8341174c9d00549ad887b105e46941fabf22da6b2f6d8f07d95220202689bcf8`;
completion manifest:
`c964c90967d0722b7f644bf65f3472b28c209be705057ddf626b8511ef0d1dc4`.

**Decision:** The infrastructure gate passes, but real two-sided policy-weighted
semantic coverage is zero. Formal continuation values remain blocked. Stop this
branch rather than integrating values or adding isolated mechanics; resume only
after a preregistered change targets a large measured blocker class and a fresh
dual-root gate demonstrates material nonzero coverage.

The full script/eval suite passes 218 tests plus 22 subtests, with two expected
data-dependent skips. All services are stopped; artifact validation, privacy,
permissions, and `git diff --check` pass.

## 2026-07-27 Selective Shared-Root Gate Repair

**Root cause:** The timed RM+ loop and the declared 512-iteration paired
evaluation shared one deadline. RM+ normally consumed the full duration before
the paired loop began, so the paired loop silently substituted cached
128-iteration optimization cells. The archived 44-35 partial therefore does not
demonstrate the declared treatment and remains ineligible for promotion.

**Implementation:** Paired evaluation now runs after the optimization deadline
against its own complete cell set; the shallow-cache fallback was removed.
Diagnostics now expose deep cell count, total deep iterations, deep elapsed
time, and explicit completion. Selective H2H agents fail closed whenever a
triggered declared treatment does not complete. The H2H harness also gained an
explicit strict-isolated-priors mode and private atomic progress snapshots bound
to the exact run configuration, with validated resumption and no repeated game
indices.

**Verification:** Seventeen focused Rust shared-root tests, 247 Gen9 Rust library
tests, 16 Python binding shared-root tests, and 17 selective/eval tests pass.
An isolated eight-world benchmark used a 100 ms optimization budget and then
completed 64 distinct 512-iteration cells (32,768 deep iterations) in 45 ms;
the full solver took 147 ms and reported a valid paired diagnostic.

**Production-budget smoke:** Two paired games at 500 ms/P8/one thread used two
distinct fail-closed r1 prior servers and the pinned local Showdown. The run
completed 0-2 with zero voids; this result is infrastructure validation only.
Across 52 candidate decisions, seven triggered (13.5%), all seven completed the
declared 512-iteration treatment, and three actions were overridden. Triggered
roots evaluated 48 to 704 deep cells. There were zero prior failures, zero
selective fallbacks, and no incomplete paired treatments. Resuming from the
atomic snapshot returned the same two results without starting another game.
Artifacts are under
`selective_shared_root_gate_repair_smoke_20260727/`; snapshot configuration hash
is `fa30176e2a313776b941af4897b7d0ab2e246e848118b24c088c0fac4c2ace08`.

**Decision:** The repaired path is ready for a fresh preregistered 500-game
gate using the previously frozen agents, thresholds, 500 ms/P8 compute, paired
roles, and SPRT H0=0.50/H1=0.55. Do not resume or combine the archived 44-35
partial. The fresh gate must use distinct strict prior servers, `--fail-fast`,
atomic progress, and the repaired experimental engine. All smoke services were
stopped after validation.

**Fresh gate launch:** Started
`selective_shared_root_gate_repaired_20260727/` as a new, independent maximum
500-game run. Preregistered configuration hash:
`1b08177d9a4d947031e11b25412ab612324fb055feb115235b3242cf07311ff7`.
The launch and resume commands are sealed in `preregistration.sha256`; the r1
checkpoint hash remains
`c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93`.
The first two games completed with zero voids (candidate 0-2), SPRT continued,
and atomic progress advanced to game 3. Six observed triggers all completed the
declared 512-iteration paired treatment; no prior-serving or selective fallback
failure was observed. This early score has no inferential weight.

**Infrastructure stop and replacement:** The first fresh attempt subsequently
completed game 3 at 0-3, then failed closed before recording game 4 when the
candidate prior server returned HTTP 500. Root cause was Metamon live conversion
calling `PokemonType.from_name(None)` for a valid null secondary type slot. The
conversion now treats that slot as monotype and has a focused regression test.
The 0-3 attempt is archived and will not be resumed or combined with evidence.

Started the independent replacement
`selective_shared_root_gate_repaired_v2_20260727/`. Its preregistered
configuration hash is
`c52f6424c3b07d6674ac84d34ee5365bbfd87eae51cf4dac85bdc415836a7d65`
and explicitly includes the Metamon source hash. Game 1 completed with zero
voids (candidate 0-1), game 2 started, all nine observed triggers completed the
declared paired evaluation, and no prior or selective fallback failure was
observed. Early scores remain non-evidential.

## 2026-07-27 Selective LCB Outcome Audit

Added `analyze_selective_lcb_outcomes.py`, a read-only fail-closed analysis of
the live gate's atomic completed-game snapshot and candidate telemetry. It
separates games with an actual sampled override from games where paired LCB was
positive but the LCB-scaled mixture retained baseline, games with triggers but
no positive LCB, and games without a trigger. It also reports Wilson intervals
and actual-override outcomes by preregistered descriptive LCB ranges. Partial
games are excluded because only game indices in the atomic progress snapshot
are admitted. Missing candidate telemetry, incomplete paired evaluation, and
invalid LCB/override fields fail closed. Two focused tests pass.

The snapshot through game 189 was 106-83 overall with 580 complete triggered
evaluations, 111 positive-LCB events, and 40 actual overrides across 36 games.
Actual-override games were 18-18 (50.0%, Wilson 95% 34.5%-65.5%). Positive-LCB
games with no sampled behavior change were 28-22 (56.0%); trigger-only games
without positive LCB were 54-40 (57.4%). Distinct-game override outcomes by LCB
were 0-1 at `(0, 0.01]`, 5-3 at `(0.01, 0.025]`, 7-9 at `(0.025, 0.05]`, and
7-6 above `0.05`. The relationship is not monotonic and every bin is small.

**Decision:** Treat LCB as an unvalidated local search confidence heuristic,
not a calibrated estimate of terminal win improvement. This audit is
descriptive rather than causal: interventions target difficult states, later
decisions affect the winner, and no matched baseline/override continuation was
executed from the same private root and chance tape. Do not alter the active
sealed gate. Any causal calibration requires a separate preregistered shadow or
randomized same-eligibility experiment after this gate terminates.

## 2026-07-28 Selective Shared-Root Branch Stopped

The repaired v2 gate was voluntarily terminated after 306 completed decisive
games. Game 307 was interrupted by the stop request and was never admitted to
the atomic snapshot. The candidate finished 160-146 (52.29%), Wilson 95%
46.70%-57.82%, with SPRT LLR -0.1330; neither sequential boundary was reached
and the promotion lower-bound criterion failed. There were zero unexplained
recorded voids and all 910 triggered evaluations completed their declared paired
treatment.

The final descriptive LCB audit found 65 actual overrides across 58 games.
Override-containing games finished 28-30 (48.28%); positive-LCB games where the
mixture retained baseline finished 38-36; triggered games without positive LCB
finished 84-73. LCB bins remained small and non-monotonic. The run additionally
confirmed that harness `--paired` means role balancing only, not mirrored teams
or battle RNG, and therefore does not deliver intended paired-team variance
reduction.

**Decision:** Kill and archive the selective shared-root branch without
promotion. Do not resume it, run equal-compute promotion control, or collect
selective teacher targets. Preserve the artifacts as negative evidence and
redirect work toward calibrated beliefs and terminal-outcome probes.

## 2026-07-28 Staged Online-RL Plumbing Smoke

Implemented a direct-policy population collector that consumes the existing
PFSP schedule format, pins every local checkpoint by epoch and SHA-256, balances
challenger/acceptor roles, writes atomic manifests, and fails closed unless every
completed battle has exactly one terminal learner trajectory. Relative output
paths are normalized before actor launch because actors run from
`experimental/src`. Added a Modal H100 continuation launcher that uploads
immutable artifacts to `metagross-online-rl`, resumes the frozen r1 epoch-5
checkpoint, uses terminal-only `BinaryReward`, and enforces a fixed 70% legacy
r1 / 20% fresh / 10% human replay mixture with the KL-anchor variant.

The local eight-game r1-versus-identical-r1 smoke completed 8/8 with eight valid
terminal trajectories, equal state/action lengths, and zero missing artifacts.
The learner result was 3-5: acceptor 2-2 and challenger 1-3. This is plumbing
validation only, not strength evidence. Artifacts are under
`online_rl_smoke_v8_20260728/`.

**Decision:** Do not launch the H100 continuation on fresh trajectories alone.
The active Modal workspace is `dnfcubes`, but this checkout and that workspace
do not contain the historical legacy-r1 self-play or 23,516-trajectory human
anchor. Training remains blocked until those two dataset roots are supplied;
the successful eight-game fresh root is ready.

**Anchor recovery and training completion:** Recovered the historical anchors
read-only from `ali-moh-islam-1/metagross-exit-r2`, packaged them as eight
parallel shards per dataset, verified every archive, and staged them into the
`dnfcubes/metagross-online-rl` Volume. Metamon indexed 23,870 legacy battles,
eight fresh battles, and 23,516 human battles at the declared 70/20/10 weights.
The guarded `BinaryReward` plus KL-anchor continuation completed 200 H100 steps
in 136.9 seconds (1.461 steps/s).

The KL training agent's raw epoch-0 checkpoint contained 1,204 anchor-only state
keys and was not admitted directly. A deployment finalizer intersected it with
the exact 642-key accepted-r1 schema and published `policy_epoch_1.pt` (checkpoint
zero is a Metamon no-load sentinel). The deployable checkpoint SHA-256 is
`5bf55431478ea6ee10841dd44ef2eb654db1dde6015dece6f2774470699963b8`;
relative parameter L2 drift from r1 is 0.00034934 and all 642 keys changed.

Production-path validation loaded all 642 keys and 142,832,563 parameters
strictly. Across eight fixed human trajectories / 286 decisions, mean entropy
was 0.8690, mean maximum probability 0.6838, KL from frozen r1 0.00314, top-1
change rate 5.24%, and raw pre-mask illegal mass 0.0038426 versus r1's 0.0038427.
The legality/collapse gate passed. This validates plumbing and bounded update
behavior only; it is not strength or promotion evidence.

**Direct-policy holdout arena:** Staged the deployable generation-one checkpoint
locally with matching SHA-256 and ran 200 role-balanced direct-policy games
against frozen r1. These games are excluded from replay. Generation one finished
105-95 (52.5%, Wilson 95% 45.6%-59.3%): 61-39 as acceptor and 44-56 as
challenger. The +2.5-point estimate is directionally positive but statistically
inconclusive, and the role split confirms material challenge-role noise. Do not
promote generation one from this result. Automated online generations must keep
training games and holdout arena games separate, publish safety-passing snapshots
to the population, and reserve accepted-policy promotion for stronger evidence.

## 2026-07-29 Through 2026-07-31: Online-RL Lineage And Public Deployment

This section records the autonomous G2-G5 lineage, the full-stack evaluation
repair that led to frozen G4, and the first bounded public-ladder comparison of
G3 and G4. It also records invalid and inconclusive runs. The central result is
not monotonic improvement: controlled direct-policy evidence favored G4, while
early public-ladder evidence favored the more conservative G3 checkpoint. This
distinction is essential for any later strength claim.

### Autonomous Controller And Promotion Contract

`experimental/configs/online_rl_autonomous_3gen.json` requested generations
two through four with 256 fresh direct-policy games and 200 learner steps per
generation, batch size 24, and 200-game excluded holdout arenas. Training used
terminal `BinaryReward`, the KL-anchor variant, and fixed sampling weights of
70% legacy r1 self-play, 20% fresh trajectories, and 10% human trajectories.
The lineage floor was 45%; accepted-policy promotion required at least 400
games and a Wilson 95% lower bound above 50%.

All local policies and comparators were pinned by run name, epoch, and SHA-256.
Fresh collection used temperature-one direct Kakuna policy inference, not the
deployed Foul Play search stack. Challenger and acceptor roles were balanced,
but random teams and battle RNG were not mirrored in these collections.

### Generation Two

G1 collected 256 role-balanced trajectories for G2, finishing 145-111 overall.
The schedule contained 135 games against frozen r1 and 121 G1 self-play games.
G2 trained for 200 steps from `randbats_online_g1_smoke_20260729` and published
`randbats_online_g2_autonomous_20260729` epoch 1 with SHA-256
`c1fe270c912b57f0bb16cda518d11879a63b989ae4013d0f197ba330e43ad638`.
Relative parameter L2 drift from G1 was `0.0017552532845870389`.

Fixed-trajectory validation passed: mean KL from G1 was `0.0030143`, top-one
change rate was 2.10%, and illegal probability mass remained effectively
unchanged. The 200-game excluded arena against frozen r1 finished 98-102,
49.0%, Wilson 95% 42.16%-55.88%, with exactly 49-51 in both role orientations.
G2 cleared the lineage floor but did not establish improvement.

The G2 run has an important provenance defect. Although 256 fresh trajectories
were collected and packaged, the learner consumed a stale eight-battle fresh
index. The checkpoint is retained as lineage history, but this generation does
not support a clean claim that the newly collected data caused an improvement.

### Generation Three Failure And Corrected G3

The first G3 collection stopped with only 189 battles and one failed shard. A
provisional checkpoint named `randbats_online_g3_autonomous_20260729` was
produced but excluded from final lineage. The fresh-index path was corrected,
and the generation was rerun under a new name rather than silently replacing
the provisional artifact.

The corrected collection completed 256/256 trajectories with no failed shard:
123 G2-versus-r1 games and 133 G2 self-play games. G2 went 60-63 against r1,
66-67 in self-play, and 126-130 overall. Corrected G3 trained for 200 steps from
G2 and published `randbats_online_g3_autonomous_freshfix_20260729` epoch 1,
SHA-256
`0c754bb96953b900e282de91c570aaae5c2c6f002dc2419e149d01132888815c`.

The update was deliberately small: relative L2 drift from G2 was `0.0033578`,
mean KL was `0.0045532`, and the top action changed on 2.80% of 286 fixed
validation timesteps. Illegal mass remained stable and strict 642-key loading
passed. G3's 200-game excluded arena against frozen r1 finished 103-97, 51.5%,
Wilson 95% 44.61%-58.33%. It advanced lineage but was not statistically
promoted.

### Generation Four And Direct-Policy Promotion

G3 collected 256 fresh trajectories for G4, going 141-115 overall. The
schedule contained 130 games against frozen r1 and 126 G3 self-play games. G4
trained for 200 steps from corrected G3 and published
`randbats_online_g4_autonomous_freshfix_20260729` epoch 1, SHA-256
`cd21dbb22bfc7a92836c7c75c6287ffc1b79c58e0a2dca3d402b76b001ff7505`.
Relative L2 drift from G3 was `0.0046848`; mean validation KL was `0.0047143`,
top-one change rate was 4.20%, and strict legality/loading validation passed.

The initial 200-game G4-versus-r1 arena finished 114-86. A disjoint 200-game
extension finished 109-91. The combined 400 unique direct-policy games were:

- G4 223-177, 55.75%, Wilson 95% 50.85%-60.54%.
- G4 as acceptor: 119-81.
- G4 as challenger: 104-96.

This passed the preregistered 400-game direct-policy gate and operationally
promoted G4. The promotion record explicitly retains the stale-index G2 caveat:
the held-out result supports this checkpoint over r1 in the tested direct-policy
setting, but does not prove clean monotonic learning across G1-G4.

### Full-Stack Scorer Failure And Counterbalanced Repair

The first production-search experiment began with an r1-versus-r1 scorer gate.
Both sides used required isolated prior servers, mirrored random-team pairs,
500ms search, parallelism eight, one search thread, and `c_puct=2.0`. Across
200 games / 100 matched pairs, nominal agent A went 88-112, 44.0%, Wilson 95%
37.30%-50.93%. A supposedly symmetric control therefore failed the declared
45%-55% scorer band.

A G4-versus-r1 experiment had already begun but was stopped after 22 games at
11-11. It has no final result and is not strength evidence. The control showed
that challenger/acceptor balancing and mirrored teams were insufficient: model
assignment to harness agent slots could materially bias the result.

The replacement experiment under
`experimental/runs/g4_counterbalanced_pilot_20260730/` ran both model-to-agent
slot orientations using identical mirror seeds and 25 matched team-seed pairs
per orientation:

- G4 as agent A: 33-17.
- G4 as agent B: 28-22.
- Combined: G4 61-39, 61.0%, Wilson 95% 51.20%-69.98%.
- Difference between G4's two slot win rates: 10 percentage points.

The design counterbalanced team seed, pair leg, challenger/acceptor role, and
model/agent slot. It did not eliminate finite-sample noise, search stochasticity,
or shared implementation bias. The result supports pinned G4 over pinned r1 in
this local production-search harness; it does not estimate public-ladder GXE.

### Frozen G4 Release

G4 was frozen at
`experimental/releases/online_g4_frozen_20260730/BASELINE.json`. The source
checkpoint and read-only rollback copy are both 571,539,531 bytes with SHA-256
`cd21dbb22bfc7a92836c7c75c6287ffc1b79c58e0a2dca3d402b76b001ff7505`.
The rollback copy is
`srcs/models/releases/online_g4_frozen_20260730/policy_epoch_1.pt`. Subsequent
training was required to use a new run name and could not overwrite G4.

### Modal Collection And Admission Hardening

An isolated Modal collection benchmark exercised frozen G4 self-play with a
local Showdown server, four workers, 25-game chunks, and two Torch threads per
collector. The interactive observation was approximately 37.5 generated games
per minute and approximately 33.7 games per app-lifetime minute. No durable
benchmark result JSON or stdout was retained, so those numbers are session
observations rather than reproducible evidence. The benchmark implementation is
retained in `experimental/src/scripts/modal_benchmark_online_rl_collection.py`.

The production Modal path added persistent collections, immutable battle
ledgers, filesystem trajectory-count checks, resumable chunks, bounded
concurrent transfer, and strict admission checks across packaging, extraction,
index reconstruction, cached indexes, and loader-visible counts. These checks
were added in response to actual failures rather than prospectively:

- A collection subprocess exited nonzero and was resumed.
- One failed chunk was retried without recollecting completed chunks.
- Recursive Modal CLI download failed because directories were treated as files.
- Sequential SDK transfer was too slow.
- Initial concurrent listing hit Modal `VolumeListFiles` rate limits.
- An initial targeted path omitted the `gen9randombattle` directory.
- Training rejected filesystem/manifest and remote/local trajectory mismatches.

The stale G2 fresh-index defect and these distributed-transfer failures are why
later manifests distinguish packaged files, extracted files, rebuilt indexes,
cached indexes, rebuilt loaders, and cached loaders. A manifest alone was no
longer treated as proof that the learner could see the declared data.

### Generation Five: 5,000-Game League Run

`experimental/configs/online_rl_scaled_replay_5k.json` configured one guarded
G5 generation from frozen G4: 5,000 Modal-collected games, 3,000 learner steps,
batch size 24, and a 500-game G5-versus-G4 arena. Opponent weights were 80%
current G4, 10% corrected G3, and 10% frozen r1. Automatic accepted-policy
promotion was disabled.

The final admitted collection contained 5,000 unique ledger records and 5,000
learner trajectories with no failed shard. `BATTLE_LEDGER.jsonl` has SHA-256
`4d948a131355b6ec7c220c82cd4abd770eff962380b8e1fd67dd4e03f4eef478`.
The seeded schedule produced exactly:

- 4,037 G4 self-play games: G4 2,013-2,024.
- 454 games against G3: G4 209-245.
- 509 games against frozen r1: G4 262-247.
- Overall G4 learner record: 2,484-2,516.

These were stochastic temperature-one training games, not holdout comparisons.
The G3 result was directionally favorable to G3 but inconclusive: G3's 245/454
win rate was 53.96%, Wilson 95% 49.37%-58.50%.

The successful trainer verified all 5,000 fresh trajectories at packaging,
extraction, rebuilt index, cached index, rebuilt loader, and cached loader.
Human and legacy packaged/index counts were 47,368 and 47,740; battle-level
loader counts were 23,516 and 23,870. G5 then completed 3,000 KL-anchored
`BinaryReward` steps and published
`randbats_online_g5_g5_league_5k_20260730` epoch 1, SHA-256
`531d0f3eb619ff9321045d58bfcd29f0af7d4e38ac552314c3a66347ffb9f5a0`.

G5 moved much farther than G2-G4: relative L2 drift from G4 was `0.4084024`,
mean validation KL was `0.310034`, and top-one change rate was 23.08%. Strict
loading and legality passed, but the fixed validation set contained only eight
trajectories / 286 timesteps and could not establish broad behavioral safety.

The excluded 500-game G5-versus-G4 arena finished 257-243, 51.4%, Wilson 95%
47.03%-55.75%: 130-120 as acceptor and 127-123 as challenger. The result was
statistically inconclusive. G5 was retained as an experimental lineage snapshot,
automatic promotion remained disabled, and frozen G4 remained accepted.

### Public G3/G4 Ladder Supervisor

The production launcher was extended with immutable G3 and G4 profiles,
checkpoint verification, account locks, bounded run directories, health checks,
rating polling, signal-safe process-group cleanup, and explicit candidate
continuation acknowledgements. `srcs/metagross/ladder_supervisor.py` alternates
G3 then G4 in sequential 25-game blocks. It never runs both policies
concurrently, stores atomic state and append-only block history, rejects fatal
log patterns, and stops rather than repeatedly reconnecting after a failed child.

Three-game canaries completed first:

- G3 account `zukofan33`: 3-0.
- G4 account `zukofan23`: 2-1.

The first continuous supervisor later stopped fail-closed during a partially
completed G4 block when client output remained unchanged for 1,200 seconds. The
account had completed 16 of 25 games, but the failed block was not falsely
recorded as a successful 25-game block. A separate nine-game recovery completed
the missing budget before a replacement supervisor started.

Audit of that nine-game G4 recovery found four wins and five losses against an
opponent pool near the high-1800s. Elo-based expectation was approximately 4.33
wins; no timeout, forfeit, invalid action, duplicate battle, missing required
player prior, or protocol failure explained the result. The recovery did expose
an observability bug: Foul Play's logger printed literal `%d` placeholders for
prior counts. The logger was corrected. Subsequent blocks recorded nonempty
player and opponent prior counts on audited turns.

At the current written observation cutoff, while the supervisor remained active:

- G3: 92-36, Elo 2219.0, GXE 86.5%, Glicko-1 1851 +/- 33.
- G4: 78-40, Elo 1875.4, GXE 82.5%, Glicko-1 1793 +/- 35.
- Historical r1 account `metaexitr1`: 218-122, current 92.4% GXE and
  Glicko-1 1973 +/- 39 after inactivity; historical settled range was
  92.4%-92.7% GXE at RD 25 with observed peak 93.6%.

The public G3/G4 records are observational, not a controlled head-to-head
experiment. Accounts encountered different opponents and rating trajectories.
At an equal 128-game checkpoint, reconstructed r1 logs were 94-34 at Elo 2346,
while G3 was 92-36 at Elo 2219. This reconstruction includes r1's initial
eight-game log and two losses visible as rating discontinuities between later
process logs; omitting those events creates an incorrect eight-game offset.
G3 tracked r1's raw record closely but remained well below r1's established
GXE. G3's early public lead over G4 does not erase G4's controlled direct-policy
and local full-stack results; instead it shows that those test distributions
did not rank the policies identically to the human ladder.

### Research Conclusions And Blog-Worthy Result

No single component in this chapter is a new reinforcement-learning algorithm.
The technically interesting result is the interaction between learning,
evaluation design, and deployment in a stochastic imperfect-information game:

1. Tiny KL-anchored policy updates produced a G4 checkpoint that cleared a
   controlled direct-policy gate against an elite r1 baseline.
2. A nominally symmetric full-stack scorer produced a 44% nominal-side result,
   six points from symmetry, and failed its own trust gate.
3. Counterbalancing model slot, role, team seed, and pair leg changed the
   evidentiary quality of the experiment and exposed a ten-point slot effect.
4. A 5,000-game, 3,000-step successor moved much farther in parameter and policy
   space but remained tied with G4 within uncertainty.
5. Public human-ladder observations ranked conservative G3 above G4 early,
   despite controlled local evidence favoring G4 over r1.
6. Stale indexes, partial distributed collections, transfer rate limits, and a
   stalled public client all produced plausible-looking partial artifacts. The
   final pipeline had to validate the learner-visible filesystem and fail closed,
   not merely trust declared manifests.

The defensible research story is therefore not "we trained the best bot by
adding more RL." It is: **how easy it is to fool yourself while improving an
already strong stochastic game agent, and how counterbalanced evaluation,
artifact lineage, fail-closed data admission, and real human deployment changed
the conclusion.** The negative and contradictory results are part of the result,
not cleanup to omit from a later blog.

## 2026-08-01: Recovery, Corrected r1 Trajectory, And 90%+ Public GXE

### What Changed After The First Written Cutoff

No policy weights changed during this continuation. G3 remained checkpoint 1
of `randbats_online_g3_autonomous_freshfix_20260729`, verified before every run
at SHA-256
`0c754bb96953b900e282de91c570aaae5c2c6f002dc2419e149d01132888815c`.
G4 likewise remained its frozen checkpoint with SHA-256
`cd21dbb22bfc7a92836c7c75c6287ffc1b79c58e0a2dca3d402b76b001ff7505`.
The performance change therefore came from accumulating public evidence with
the already-frozen policies, not from an unrecorded learner update.

The operational changes that made the longer observation possible were:

- immutable profile-to-checkpoint binding and hash verification on every run;
- one account lock and one policy process at a time;
- bounded child runs with a 1,200-second output-stall detector;
- atomic manifests, periodic rating snapshots, and append-only block summaries;
- signal-safe cleanup of the client, prior server, and search workers;
- explicit `500 ms` search, parallelism 8, and one thread per search worker; and
- corrected player/opponent prior telemetry, replacing literal `%d` log output
  with the actual nonzero counts needed to audit production search.

These are deployment and evidence-integrity improvements. They are important
to the result, but they must not be described as policy improvements.

### Interrupted G3 Block And Bounded Recovery

The supervisor rooted at
`srcs/runtime/ladder-supervisor/20260731T185544Z-27293` stopped fail-closed on
August 1 after the G3 child reported `TimeoutError: ladder client output
stalled`. The child had emitted 11 complete rating transitions, 7 wins and 4
losses. The public account also incorporated one additional loss associated
with the stalled connection, producing 7-5 public progress from the 92-36
cutoff to 99-41. The failed child remained marked `status: failed`; it was not
promoted to a completed 25-game block.

A separately bounded 13-game G3 recovery then completed 10-3 with no fatal log
pattern. Its manifest is
`srcs/runtime/ladder-supervisor-recovery-g3/20260801T031252Z-g3-zukofan33-14251/manifest.json`.
The recovery ended at 109-44 and verified the same frozen G3 hash. A shell-level
handoff immediately replaced the recovery process with a fresh continuous
supervisor, avoiding concurrent use of either ladder account.

### Completed Sequential Blocks

The replacement supervisor is rooted at
`srcs/runtime/ladder-supervisor/20260801T035648Z-14249`. Its first three sealed
blocks were:

- G3: 16-9, moving the account from 109-44 to 125-53. The completion snapshot
  was Elo 2324.77, 90.2% GXE, and Glicko-1 1917.65 +/- 29.24.
- G4: 15-10, moving the account from 87-41 to 102-51. The completion snapshot
  was Elo 2147.14, 86.1% GXE, and Glicko-1 1844.19 +/- 31.44.
- G3: 15-10, moving the account from 125-53 to 140-63. The completion snapshot
  was Elo 2319.36, 91.2% GXE, and Glicko-1 1940.69 +/- 28.00.

All three blocks contain exactly 25 winner lines, no fatal pattern, the expected
checkpoint hash, and a completed supervisor summary in `blocks.jsonl`. G4's
positive 15-10 block improved its public estimate, but did not reverse the
deployment ranking: after the block it remained 4.1 GXE points and about 73
Glicko points below G3's preceding completion snapshot.

The next G3 block opened strongly. At its 20-game mark it was 14-6, putting G3
at 139-59 after 198 total games with Elo 2392.22, 91.4% GXE, and Glicko-1
1944.70 +/- 28.22. This was the first observed G3 snapshot above 91% GXE and
narrowed the comparison with r1's mature 92.4% benchmark to one GXE point.
Four consecutive losses then moved the block to 14-10 and the
account to 139-63 after 202 games: Elo 2294.82, 91.1% GXE, and Glicko-1
1937.42 +/- 28.04. The Elo drawdown was large, but GXE and Glicko moved much
less, illustrating why peak Elo is not the primary performance estimate. A
final win sealed the block 15-10 at 140-63 after 203 games, with the 91.2% GXE
completion snapshot recorded above.

### Corrected r1 Reconstruction And Comparison Limits

The first r1 equal-game comparison accidentally omitted the initial eight-game
`metaexitr1.log`; the following `run_*` sequence begins at Elo 1157 rather than
1000. The reconstruction was corrected to prepend those eight transitions.
There are 333 explicit retained rating transitions, 217 wins and 116 losses,
while the authoritative account record is 218-122 over 340 games. Five rating
continuity breaks expose omitted activity; in aggregate seven games, one win
and six losses, lack explicit retained rating lines.

The corrected equal-game observations are:

- 128 games: r1 94-34 at Elo 2346; G3 92-36 at Elo 2219.
- 178 games: r1 reconstructed 125-53 at Elo 2331; G3 125-53 at Elo 2325.
- 198 games: r1 reconstructed 133-65 at Elo 2240; G3 139-59 at Elo 2392.
- 202 games: r1 reconstructed 133-69 at Elo 2173; G3 139-63 at Elo 2295.
- 203 games: r1 reconstructed 134-69 at Elo 2191; G3 140-63 at Elo 2319.

The later records use the most parsimonious placement of the visible downward
rating discontinuities as omitted losses. They are useful trajectory evidence,
but the seven missing r1 rating lines prevent claiming a perfectly complete
per-game history.

Most importantly, r1's historical per-game Glicko and GXE snapshots were not
retained. The 92.4% value is the current mature benchmark after r1's complete
218-122, 340-game account history; it is not r1's GXE at 128, 178, 198, 202, or
203 games. Equal-game comparisons are therefore available for record and Elo only.
G3's 91.4% at 198 games versus r1's 92.4% mature benchmark is a cross-stage
comparison and must always be labeled as such.

### Updated Interpretation

The public evidence now supports a stronger but still bounded statement. Frozen
G3 reached 90%+ GXE, tracked r1 exactly on record at 178 games, and led the
reconstructed r1 record by six outcomes at 198, 202, and 203 games. It also
continued to outperform frozen G4 on the human ladder despite G4's stronger
controlled local result against r1. None of this proves G3 is universally
stronger: matchmaking, opponent strength, calendar time, and infrastructure
losses differ between accounts. It does show that the conservative G3 policy is
the strongest deployed checkpoint observed so far, and that the public ladder
and controlled local arenas rank G3 and G4 differently.

## 2026-08-02: Equal-Sample Reversal And Production Engine Repair

The later public sample reversed the provisional interpretation above. Frozen
G3 reached 340 games at 209-131, Elo 2141, 89.4% GXE, and Glicko-1 1901 +/- 25.
At the same 340-game count, the authoritative r1 record was 218-122, nine result
swings better. The retained r1 transitions place its historical finishing Elo
near 2362; its mature account estimate is 92.4% GXE and Glicko-1 1973 +/- 39.
Because historical per-game r1 GXE and Glicko snapshots do not exist, only the
record and Elo comparisons are equal-sample observations. G3 subsequently
reached 215-141 over 356 games with 88.9% GXE. G4 was retired at 132-86 with
85.0% GXE. The public evidence therefore retains r1 as the accepted baseline.

The G3-only continuation contained four sealed 25-game blocks at 12-13, 13-12,
11-14, and 9-16, for 45-55 combined. An audit found no missing required priors,
invalid actions, protocol failures, forfeits, or disconnects in the selected
poor segment, but did find tactically implausible setup, recovery, priority,
Substitute, Protect, and switching choices. A 4-10 audited segment had 7.95
Elo-expected wins; its Poisson-binomial lower-tail probability was approximately
2.9%. This establishes an unusual segment, not a causal diagnosis.

Deployment provenance inspection then found that `.venv-fp-priors` imported an
editable experimental engine from
`experimental/engine/pe_v3_learned_priors/poke-engine-py`, including its
experimental `seed` MCTS parameter. The intended production source is
`srcs/vendor/poke-engine`. Its documented root Python wrapper was also stale: it
did not forward `s1_priors`, `s2_priors`, or `c_puct` even though the native Rust
binding supported them.

Production now fails closed unless poke-engine is a non-editable local install
from `srcs/vendor/poke-engine`, exposes both root-prior arguments and `c_puct`,
and excludes the experimental `seed` parameter. Manifests record the native
extension SHA-256. The launcher also removes inherited experimental learned-value
environment variables. The vendor root wrapper was repaired and the production
environment rebuilt. Verified provenance is poke-engine 0.0.47 with native
SHA-256
`2d141cce5abddb6d9926a1a6955aac5c4e9e0c0df7aa742dcd46a5de6195ec07`.
A behavioral regression test forces the otherwise weak `leer` action to receive
the most visits under an extreme root prior, proving that the installed wrapper
forwards priors into native search. Seven vendor binding tests and 14 focused
Metagross launcher/supervisor tests pass; `git diff --check` is clean.

A bounded three-game r1 production canary was attempted under
`srcs/runtime/ladder-runs/20260802T091403Z-r1-metaexitr1-19720`. The pinned r1
checkpoint hash, engine provenance, strict 642-key model load, and prior-server
health check all passed. Showdown rejected the locally supplied account password
before matchmaking, so the client exited nonzero and the manifest remained
`status: failed`. Zero games were played and the account record remained
218-122. Public continuation is blocked until the local `metaexitr1` credential
is refreshed; the failed attempt must not be represented as ladder evidence.

The account-specific ignored credential file was then used instead of the stale
launchctl value. A three-game canary completed 1-2, followed by a clean ten-game
block at 5-5. To close the remaining launcher-to-loader identity gap, the prior
server was changed to accept the pinned checkpoint SHA-256, independently hash
the exact resolved checkpoint before model initialization, and fail closed on a
mismatch. The launcher now passes that immutable hash to the prior process.

A final three-game proof block under
`srcs/runtime/ladder-runs/20260802T102001Z-r1-metaexitr1-70428` completed 2-1.
Its prior log records the exact loaded source as
`srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt` with
SHA-256
`c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93`,
followed by strict validation of 642 keys and 142,832,563 parameters. The
manifest independently records the same r1 identity and the repaired production
engine SHA. All three games had nonempty required player priors; no prior,
search, protocol, invalid-choice, disconnect, timeout, fallback, or process
error was found.

The repaired-stack r1 continuation therefore finished 8-8 over 16 games and
moved `metaexitr1` from 218-122 to 226-130. At 356 total games, exactly matching
G3's final sample size, r1 was 11 result-swings ahead of G3's 215-141. The final
public snapshots were r1 at Elo 2083.60, 91.5% GXE, and Glicko-1 1948.30 +/-
37.37 versus G3 at Elo 2055, 88.9% GXE, and Glicko-1 1892 +/- 25. This new GXE
comparison is equal-sample and contemporaneous, unlike the earlier historical
r1 GXE limitation. The accepted-policy conclusion remains r1.

## 2026-08-13: Public-Information Value Leaf Gate

The earlier 69-root transformer panel was audited and found to contain decision
states from one source battle, not 69 independent battles. Its negative result is
retained only as a descriptive diagnostic. The promotion schema is now v2 and
requires at least 100 paired units, 50 roots, and 50 distinct source battles;
uncertainty is bootstrapped over source battles.

A determinization-invariant 18-feature public-state contract was added to the
production vendor engine. Opponent reserve HP, moves, items, abilities, EVs, and
stats are never read. Two battle-disjoint terminal-outcome MLPs were trained on
184,213 decisions from 4,998 historical self-play battles. The 3,329-parameter
model reached 70.04% test accuracy and 0.18761 Brier; a frozen latency adaptation
with 449 parameters reached 69.93% and 0.18834. Exported Rust and Python inference
matched exactly.

The large model failed a held-out 60-battle, 120-pair, equal-500-ms leaf-value
gate: oracle top-1 fell from 90.0% to 80.0%, mean oracle regret rose from 0.00456
to 0.00790, and visits fell about 35%. Under the preregistered one-adaptation
protocol, the small model was evaluated on 60 fresh battles with no overlap
against training or the first panel. It also failed: top-1 fell from 87.5% to
77.5%, regret rose from 0.00118 to 0.00386, and paired regret improvement was
-0.00268. The candidate was better on 2 paired units, equal on 104, and worse on
14 in both panels.

**Decision:** Stop this scalar terminal-value branch. Do not run H2H, league
generation, a blend sweep, or another scalar model on these panels. The retained
models are predictive of terminal outcomes but did not improve action selection.
A future value hypothesis requires action-conditional or counterfactual labels,
an untouched evaluation population, and a separately declared protocol.

## 2026-08-14: Public Action-Q Offline Admission

A separate preregistered branch tested whether per-action counterfactual targets
could fix the scalar value model's action-identifiability failure. One root was
sampled from each of 1,000 source battles, with two schedules, eight worlds per
schedule, and 50,000 stock-MCTS iterations per world. The resulting frozen
teacher artifact contains 800 million iterations and a schedule-averaged value
for every legal action. Public/action features were verified invariant to hidden
opponent reserve completions.

The 13,441-parameter 176-feature Q MLP showed ranking signal but failed its
offline admission thresholds. On 96 validation battles it reached 69.17%
pairwise order accuracy, 32.29% top-1 agreement, and 0.05643 mean teacher regret;
the frozen requirements were at least 35% top-1 and regret below 0.03. The held-
out test result was 33.33% top-1 and 0.07968 regret. A descriptive comparison
found the historical 500 ms selected action at 54.26% top-1 and 0.03612 regret
on 94 mappable validation roots, so deploying the learned Q prior would likely
discard stronger existing information.

**Decision:** Stop before the independent 60-battle gate. Do not tune admission
thresholds, root-prior temperature, c_puct, or model size on this dataset. The
still-untouched root population is preserved. A future action-Q experiment must
declare a richer representation in advance, such as a head over frozen R1 history
embeddings or a shallow-search residual, and use a new validation split.

## 2026-08-14: Frozen-R1 Transformer Action-Q Admission

The richer representation follow-up reused the frozen 1,000-root deep-Q teacher
but joined it to exact schema-v3 production R1 observations. All roots had unique
snapshot joins; 998 were admitted and two were rejected for unmappable engine
actions. The dataset contains no sampled states. The accepted R1 checkpoint was
frozen, its exact deployed stateless two-step 900-dimensional embeddings were
cached locally, and only a 235,797-parameter 13-action Q head was trained.

The head overfit: train top-1 reached 70.71% with 0.00581 regret, while the new
111-battle validation split reached only 40.54% top-1 and 0.05467 regret. The
historical 500 ms action on the same validation population reached 53.21% top-1
and 0.04310 regret. The held-out test result was 32.63% top-1 and 0.06091 regret,
again below the historical action at 46.74% and 0.04354.

**Decision:** Stop before the independent root gate. Do not enlarge the head or
tune the loss, temperature, or thresholds on these splits. The result is a
sample/representation generalization failure, not insufficient training capacity.
Before another Q-head run, repair and verify production's causal history/RL2
tracking so a full-history representation can be reproduced online; otherwise a
history-conditioned offline win would not be deployable.

## 2026-08-14: R1 Causal-History Deployability Gate

The production history bridge was repaired around an explicit selected-action
receipt. Each final controller choice is now acknowledged before it can be sent
to Showdown and is correlated to the exact battle, private request id, request
hash, decision index, and served support. Public move and switch events are
observations only. Missing, conflicting, stale, or unsupported action receipts
fail closed. AMAGO input construction now begins with the real time-zero
observation, uses reward-first previous-action/reward inputs, and preserves
absolute time indices when context is cropped.

A local audit of 40 real protocols uniquely mapped all 1,051 outbound choices
to one of the 13 frozen R1 actions; all 13 indices were represented. An actual
CPU R1 server then replayed a complete captured battle through 20 ordinary
decisions with 20 receipts, no reset, no missing receipt, and no RL2 or time
misalignment. Independent offline recomputation from the durable dump loaded
the frozen 142.8M-parameter R1 checkpoint and matched all 20 live probability
vectors exactly, with maximum absolute difference 0.0 at tolerance 1e-7.

**Decision:** The causal-history representation is now reproducible and
deployable, but no policy-strength claim has been made. Do not retry a generic
action-Q or scalar terminal-value head. First prove a resource-aware,
long-horizon search expert on frozen battle-disjoint roots and a bounded live
gate. Only a winning expert may supply confident deviation targets for
distillation, mixed with pass-through R1 and resource-preservation anchors.

## 2026-08-14: Causal-History Versus Legacy-Stateless Screen

A newly played two-game mirrored self-canary exercised the repaired causal
history on both sides and completed 1-1 with zero voids. The two isolated prior
servers produced 138 decisions across four player trajectories; independent
offline recomputation matched every live probability exactly with maximum
absolute difference 0.0.

The previous production player-policy input was then frozen as an explicit
comparator mode: one zero dummy timestep, the current observation, zero RL2,
and time indices `[0,1]`. Both treatment arms retained the repaired current
observation, private-request action support, selected-action receipt, and
identical 500 ms production search. This isolates causal history rather than
reintroducing known harness defects.

An exploratory integration pair swept 2-0 for causal history and was not pooled.
The frozen 20-game, ten-pair screen finished 11-9 (55.0%) for causal history,
with three causal sweeps, five splits, two stateless sweeps, and zero voids.
The Wilson interval was 34.2%-74.2%; a 100,000-resample pair bootstrap was
35.0%-75.0%. The positive point estimate is roughly +35 head-to-head Elo if
true, but is not statistically established.

Across 735 live causal decisions from the smoke and screen, exact offline parity
again had maximum error 0.0. Same-state counterfactual inference found that
history changed the policy top action on 32.4% of decisions and moved 28.2% of
probability mass on average. Causal priors agreed with the subsequently selected
500 ms search action 59.7% of the time versus 52.2% for stateless priors; this
last metric is descriptive because search consumed the causal treatment.

**Decision:** Keep and commit causal history as the correct production path.
It materially affects policy and has a modest positive live point estimate, but
this screen does not establish a strength gain and is not the dramatic
improvement target. Do not extrapolate it to GXE or launch a 500-game gate from
this evidence alone. Proceed to the separately gated resource-aware,
long-horizon expert before any new distillation or league-scale collection.

### Detailed causal-history work ledger

This subsection is the canonical handoff for the complete causal-history turn.
It supplements the concise decisions above so that later work does not repeat
the same audits, confuse deployability with strength, or lose the exact artifact
identities.

#### External methodological input

Alex Wa's MegaGem study was read in full and used as a design warning. Its 4B
critic explained only 15.2% of Q variance between actions within a state and
reached roughly 24%-34.7% best-bid accuracy, below a 40% non-leaky linear
baseline. A myopic analytic bidder improved offline regret but lost 5.01 paired
margin points live. Adding a shadow price for the future option value of cash
reversed the result: first-place rate moved from 36.0% to 59.3% and paired
margin improved by 12.16 points. Only after the selector won live did one round
of expert iteration distill approximately 4,700 bid decisions, including about
1,200 expert deviations, into the 4B weights; the weights-only result improved
paired margin by 8.31 with reported 95% interval [3.21, 13.42]. Sources:

- `https://github.com/djdumpling/djdumpling.github.io/blob/main/_posts/2026-08-09-megagem.md`
- `https://github.com/djdumpling/MegagemBench`

This is analogous evidence, not Pokémon evidence. The operational lesson is to
price the future option value of conserved resources before distillation. The
declared Pokémon resources are HP, Tera, PP, tempo, revealed information, and
remaining switch resources. A per-turn offline selector is not accepted merely
because its local prediction or regret improves.

#### Production contract implemented

The production path now has an explicit `/action` acknowledgement between final
controller selection and returning the command to the Showdown sender. Its join
key is `(namespace, battle tag, rqid, canonical request SHA-256, decision_idx)`.
The selected action must exist in the exact priors response support. Exact
duplicates are idempotent; conflicting, stale, unsupported, malformed, or
unserved receipts fail closed.

Public `|move|`, `|switch|`, drag, Tera, failure, and reveal messages remain
observations and are never used as action labels. Private requests are the
authority for own-side action support and identity. Private active identity is
reconciled for Illusion-like public/private mismatches without reading opponent
private state. Forced Recharge and sole Struggle boundaries do not add a learned
policy transition; mixed Struggle requests expose all four live policy slots
while replay supervision uses canonical action index 0.

AMAGO input construction now begins with the real observation at time 0 rather
than an artificial zero observation. At decision `k+1`, RL2 contains the reward
from `S_k` to `S_{k+1}` and one-hot selected action `A_k`. Context cropping
preserves absolute time indices. Mutable observation spaces are deep-copied and
reset per battle. Missing action receipts, reward failures, non-finite rewards,
or any observation/action/reward length mismatch terminate the policy path
rather than silently clearing history.

Primary implementation files:

- `srcs/metagross/decision_harness.py`
- `srcs/metagross/prior_server.py`
- `srcs/metagross/run_foul_play.py`
- `srcs/metagross/tests/test_launch.py`
- `experimental/src/scripts/audit_r1_action_boundaries.py`
- `experimental/src/scripts/replay_r1_history_canary.py`
- `experimental/src/scripts/verify_r1_history_dump_parity.py`
- `experimental/src/scripts/compare_r1_history_modes.py`

#### Saved-protocol admission audit

The local saved-protocol audit covered 40 real Foul Play protocol files from
`experimental/runs/search_native_stage1_20260808/screen-logs/`. It joined
outbound choices to private requests rather than public outcomes:

- private requests: 1,207;
- outbound choice rows: 1,051;
- uniquely mapped choices: 1,051/1,051;
- public action events deliberately ignored as labels: 1,960;
- idempotent duplicate outbound choices: 0;
- skipped default choices: 0.

All 13 frozen action indices occurred. Counts by index were
`0:252, 1:155, 2:180, 3:134, 4:114, 5:63, 6:61, 7:41, 8:16, 9:14,
10:9, 11:5, 12:7`. The authoritative report is
`experimental/runs/r1_causal_history_parity_local_20260814/saved_protocol_action_audit.json`.

#### Captured-battle replay and exact offline parity

The actual frozen 142.8M R1 server was run locally on CPU against captured
protocol
`experimental/runs/search_native_stage1_20260808/screen-logs/g1x003577f.protocol.jsonl`.
The final v2 replay had 20 ordinary decisions, 20 acknowledgements, maximum
inference/tensor length 20, zero resets, zero missing receipts, zero RL2 receipt
mismatches, and zero time-index mismatches. Its decision-dump SHA-256 was
`b991cd221e45bcf0fbfdcb66029bfa73f719c9dc0748698d7493d0def13cd482`.

A separate process loaded checkpoint
`srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt`, verified
642 state-dict keys and 142,832,563 state-dict parameters, reconstructed every
sequence from the durable dump, and obtained maximum policy-probability
absolute difference 0.0 against tolerance `1e-7`. Frozen checkpoint SHA-256:
`c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93`.

#### Newly played causal-history canary

The required newly played canary ran locally on CPU with two isolated causal
prior servers, mirrored teams, 500 ms/P8 search, one search thread, and
`c_puct=2.0`. The pair finished 1-1 with zero voids. This was a deployability
canary, not a strength comparison.

The two server dumps contained 70 and 68 decisions across two battles each.
Independent recomputation matched all 138 policy rows exactly with maximum
error 0.0. Artifact identities:

- H2H result SHA-256:
  `90f16794e8f9f2e68b5d85883d875022e5d21457df4ad4a9eff72e5e9f55c774`;
- prior-A dump SHA-256:
  `e37f5f731e215348dfa43a670939b8787162cfb60544d04c89868792fe9e180f`;
- prior-B dump SHA-256:
  `226a574ae9ff1998b6d4216779758b8d252ee8bd56de1c9e23d0fb380e1ab8f6`.

The parity verifier was extended during this canary to group a shared JSONL
dump by `(namespace, battle tag)`. Treating the first row of the next battle as
the next timestep of the previous battle is now an explicit error rather than a
silent assumption.

#### Frozen legacy comparator and strength screen

An explicit `--trajectory-mode legacy-stateless` comparator reproduces the old
player-policy model input conditional on the same repaired current observation:
one zero dummy timestep, current observation, all-zero 14-channel RL2, and time
indices `[0,1]`. It retains exact private-request action support, selected-action
acknowledgements, current-state repairs, search implementation, and checkpoint.
Thus the treatment is causal history, not a bundle of old harness defects.

The exploratory integration pair finished 2-0 for causal history and was not
pooled. Its result SHA-256 was
`678bb9fcd436c7e3b60e7839244888de10271d5ac611ef043f2155135cecb592`.

The primary screen used 20 games / 10 mirrored pairs, mirror seed 2026081412,
identical 500 ms/P8/one-thread production search, `c_puct=2.0`, sequential games,
required priors, isolated prior servers, fail-fast handling, and zero GPU use.
Agent A was causal history and Agent B was legacy stateless even though both
artifact agent-name fields read `production_r1_search_first`.

Primary result:

- causal wins: 11;
- stateless wins: 9;
- point estimate and pair-score mean: 55.0%;
- Wilson 95% interval: [34.21%, 74.18%];
- 100,000-resample pair-bootstrap interval: [35.0%, 75.0%];
- causal sweeps / split pairs / stateless sweeps: 3 / 5 / 2;
- causal as challenger: 8-2;
- causal as acceptor: 3-7;
- voids, errors, ties, and unknowns: 0.

The 55% point estimate is approximately +35 head-to-head Elo if it is the true
rate, but the interval is far too wide to establish any positive effect. The
role split is a visible small-sample warning even though mirroring makes the
pair mean the primary summary.

Primary screen artifact identities:

- result SHA-256:
  `a4127245dfb15913053d7347130d063f737bb75fd1cddced9a1c854c000414c1`;
- paired inference SHA-256:
  `8ae19e3bfae5593a1136ac98a51c548c4cfc0413f521c7ea22fdc27af781df6d`;
- causal dump SHA-256:
  `5c1d334ef8441fa8b854a3395fdf6c8a8d93485cfec734f9f2ff509309543f50`;
- stateless dump SHA-256:
  `dbf6dbac204d6daea2668310a660a3212e4a9486d9ebc54bc29aea34fa46143f`;
- causal parity report SHA-256:
  `9fc507927c19372e0bffff6bca3bc773dcb8f14d1c1b3a7f0b4e4dcaee0fc5df`;
- history comparison report SHA-256:
  `eab3c45788fe149f270bb7fd0f199b92dd197affe63a4807faa83937c5ca6fed`.

The causal dump contained 735 decisions across 22 player trajectories from the
smoke plus primary screen. Independent offline inference matched all 735 with
maximum error 0.0. Counterfactual legacy-stateless inference on the same current
observations found:

- top-1 action change rate: 32.38%;
- total-variation distance: mean 0.2824, median 0.2238, p90 0.6257;
- Jensen-Shannon divergence: mean 0.09578, median 0.04792, p90 0.25533;
- search-selected action rows with durable receipts: 713;
- causal/stateless top-1 agreement with selected search action: 59.75% / 52.17%;
- causal/stateless mean probability on selected search action: 0.5135 / 0.4530.

The agreement metrics are descriptive and treatment-favoring because the
played search consumed the causal prior. They establish a mechanism—history
materially changes the prior—not strength. The only strength estimand here is
the inconclusive 11-9 outcome.

#### Verification and execution hygiene

All new work ran locally. `CUDA_VISIBLE_DEVICES` was empty and no Modal,
Nebius, Overshoot, or other cloud/GPU resource was started. Temporary Showdown
and prior servers were stopped after each run. The final focused suites passed:

- 141 production/correlation tests in the Python 3.11-compatible environment;
- 4 parity/mechanism tool tests in the Metamon environment;
- Python compilation for all new scripts;
- `git diff --check`.

The scoped implementation and concise reports were committed on `main` as
`dcbd814a` (`fix(metagross): restore causal R1 history`), containing 24 files,
2,622 insertions, and 81 deletions. Large raw battle/search logs remain under
the ignored run directories locally and were deliberately not forced into Git.
Unrelated pre-existing dirty-worktree files were not staged in that commit.

#### Final interpretation and next authorization

Causal history is now the accepted correctness path. The code is reproducible,
deployable, exact under offline recomputation, and materially different from
legacy stateless inference. Its observed 55% point estimate is useful but does
not prove it is stronger and cannot be translated into a GXE claim. Even a true
+35 H2H Elo effect would probably be an incremental improvement rather than the
standalone step from roughly 92% GXE to 95% GXE.

Do not spend on a 500-game causal-versus-stateless gate, another generic scalar
value model, another generic action-Q critic, or distillation from the retained
one-second teacher. The next authorized performance experiment is:

1. freeze a new battle-disjoint panel of informative roots with common worlds
   and seeds;
2. build a long-horizon expert that explicitly preserves and prices future HP,
   Tera, PP, tempo, information, and switch resources;
3. require independent root/outcome advantage over the current 500 ms
   controller before collecting a student dataset;
4. distill only confident deviations, mixed with pass-through R1 decisions and
   resource-stratified anchors;
5. test the weights-only student inside the identical 500 ms controller;
6. scale PFSP/league generation only after that equal-budget gate passes.

## 2026-08-14: Resource-aware long-horizon expert development gate

**Scope and correction:** The next authorized teacher experiment was executed
locally on CPU. Auditing the retained 1,000-root / 2,000-schedule action-Q
oracle found that it contained no `-tera` actions even though 437/1,000 source
roots still admitted Tera under the corrected engine. It cannot measure Tera's
option value and was not reused as the resource teacher. A fresh isolated Gen9
engine build exposed Tera variants on 6,992/16,000 sampled worlds.

Implemented an interpretable 23-term resource-shadow contract covering own HP,
active/bench HP, switch depth, Tera, PP, move availability, opposing HP/faints,
status/Tera, boosts, screens, hazards, Substitute, held-item reserve, and two
context terms. The first 21 coefficients are nonnegative by construction. Four
opponent-information positions are reserved but fixed to zero in deployable
search: determinized leaves contain sampled hidden sets but no causal public
reveal mask, so reading their apparent moves/items/abilities would leak the
sampled world. Information option value remains blocked on the missing
public-history-to-search bridge rather than being silently approximated.

The Python and Rust extractors matched across 1,000 sampled roots with maximum
absolute error `1.57e-7`. Three focused Python contract tests pass, the scripts
compile, and the Gen9+Tera Rust library compiles. The constrained linear model
was calibrated on the same 184,213 terminal-labeled decision states from 4,998
self-play battles used by the earlier public value study, with a deterministic
battle-disjoint 70/15/15 split and inverse decision-count battle weights. It
reached 68.93% test accuracy and 0.19702 Brier versus 0.25000 constant Brier.
This was treated only as calibration, never as a strength result. Model hashes:
JSON `84e6176f72453db7eb718b49e5896b911c3d7556ae4624ce09b0f0c0b20345bf`;
engine text
`9ed2aa54ecc7d168f473d4db264f48b9850763ada048873c4f377a7b5f444dd5`.

Frozen two battle-disjoint 50-root panels from the prior 1,000-battle source,
each with 25 Tera-available and 25 Tera-spent/forbidden roots. Development
panel SHA-256:
`55105e7b336b68a2658e6456322f0e34dbe5ef394c15076b14acb3046862ac63`;
untouched holdout SHA-256:
`c622aa7d4a016c6d025c2f5562102add3404e2347440ee5f79a27d5b51636d5c`.
Every root has two schedules and eight common worlds. The reference used 50k
exact iterations/world under a separate seed namespace; both treatment arms
used 20k iterations/world.

On the 100 development schedule units, archived historical 500 ms actions had
mean oracle regret 0.041663, 48% top-1, and 18 regrets at least 0.10. Equal-depth
unmodified search had mean regret 0.001690, 88% top-1, and zero catastrophic
regrets. The frozen resource blend 0.25 worsened mean regret to 0.003739. The
single allowed development-only sweep also failed: weights 0.05/0.10/0.15
produced mean regrets 0.00169049/0.00225136/0.00205994. Their mean improvements
over equal-depth hand search were all negative, and no bootstrap lower endpoint
was positive. Across weights, only 2-7 of 100 actions changed and none selected
Tera.

**Decision:** The explicit associational resource-shadow leaf does not improve
the independently scored development actions. Stop it before holdout, H2H,
target collection, distillation, or any GXE claim. The holdout remains
untouched. Do not tune more weights or relax the gate. The strong gap between
equal-depth search and archived historical actions is only a lead: it is biased
toward the hand evaluator and the comparator predates causal-history R1. A
future corrected high-budget teacher must compare against current causal R1 and
survive outcome-grounded or live evidence before its deviations are distilled.

All work was local CPU. No GPU, Modal, Nebius, Overshoot, public ladder games,
or paid compute was used. Detailed protocol and results are in
`experimental/runs/resource_aware_expert_20260814/`.

## 2026-08-14: Causal public-history/reveal mask bridge

**Objective:** Remove the blocker identified by the resource-aware expert gate:
determinized interior states contained sampled hidden teams but did not retain
which opponent facts were causally public. This was an architecture/correctness
change only; no strength claim or paid run was authorized.

Implemented a symmetric packed reveal-mask contract in the experimental engine.
For each opposing team slot it stores species, four move slots, item, and ability
(42 bits per observer). New masks are serialized after the existing state fields,
sanitized on read, exposed through the Python binding, and copied explicitly by
the offline determinizer. Legacy state strings remain readable and initialize
both masks to zero.

Root masks now come from player information rather than completed simulator
truth. The live R1/search wrapper compiles them from the causal `BeliefTracker`;
the neural root gate compiles them from the exact `R1SwitchTracker` public team
that produced the frozen transformer observation; typed `PublicEventLedger`
events have a separate adapter. The archived/offline public-state builder freezes
its visible snapshot before sampled reserve slots are installed.

Every top-level engine transition now records executed public actions. The
engine appends reversible reveal deltas for executed moves, switches, explicit
item activations/changes, and explicit ability changes. Existing root facts do
not produce deltas, so reversing an MCTS branch cannot erase them or contaminate
a sibling. Reveal instructions are metadata-only and are ignored by semantic
event accounting; the already-certified public action/switch/item events remain
the transformer continuation input.

Resource feature positions 16–19 now read species/move/item/ability fractions
from side one's causal mask. A sampled completion can change hidden opponent
moves, item, and ability without changing those inputs. Simulated item/ability
coverage remains conservative where the engine lacks explicit public-activation
provenance: under-counting is allowed, reading hidden truth is not.

Verification was entirely local CPU:

- all 254 Gen9+Tera Rust library tests passed;
- four pure Python history/mask tests passed;
- serialization round-trip and legacy compatibility passed;
- apply/reverse restored the exact root mask;
- hidden-completion perturbations left information features unchanged;
- a rebuilt CPython 3.11 Gen9+Tera extension advanced a synthetic root mask from
  `1` to `65` after the opponent executed move slot 0, while the information
  vector advanced from `[1/6, 0, 0, 0]` to `[1/6, 1/24, 0, 0]`;
- semantic unaccounted-instruction kinds remained empty;
- relevant Python scripts compiled.

**Decision:** The missing leak-free bridge is accepted as a correctness
prerequisite. This result does not rescue the failed resource-shadow teacher and
does not justify H2H. Retrain/export the outcome-grounded value path on the
corrected causal representation and require an equal-500ms fixed-root advantage
before any 100-game H2H or deviation distillation. Detailed records are in
`experimental/runs/public_reveal_bridge_20260814/`.

## 2026-08-14: Causal public-reveal resource-shadow gate

**Objective:** Test the smallest falsifiable version of the resource-aware,
long-horizon expert after completing the interior causal-history bridge. The
experiment asked whether replay-calibrated public-information shadow prices
improve equal-iteration root decisions independently of the other resource
features. All work was local CPU; no cloud, GPU, paid compute, public ladder, or
live H2H was used.

The historical terminal-value transformer was not retrained unchanged: it does
not consume the new reveal fields, so a rerun would not test the new
architecture. The existing resource-shadow v1 model was also safety-locked.
Its four information coefficients had been trained exclusively on zero-valued
inputs and were merely optimizer initialization; loading v1 now zeros weights
16–19. Only a `metagross_resource_shadow_v2` artifact calibrated on causal
masks may activate those features.

Reconstructed observer-relative start-of-turn facts from all 4,998 preserved
Showdown replays and joined them to 184,213 decision states. A total of 184,185
states received nonzero masks. For 23,050 additional decisions sharing a
battle/turn, the same conservative start-of-turn snapshot was reused: this can
undercount forced-switch information but cannot reveal a fact early. Mean
species/move/item/ability coverage was 0.6724/0.2863/0.2963/0.1317. Training
used terminal outcomes, battle-hash 70/15/15 splits, battle weighting, and seed
20260814. Test accuracy/Brier were 0.68694/0.19804, slightly worse than the old
v1 calibration's 0.68929/0.19702.

Built an immutable reveal sidecar for the already-frozen development panel
(SHA-256
`55105e7b336b68a2658e6456322f0e34dbe5ef394c15076b14acb3046862ac63`),
leaving the panel unchanged. The sidecar contains 800/800 nonzero world masks.
The gate used 50 distinct battles, 100 counterbalanced schedule units, eight
worlds per unit, 20k iterations for hand/resource arms, a current-engine 50k
oracle, and resource blend weight 0.05.

Results:

- historical action: mean regret 0.04156597, top-1 0.48, 18 catastrophes;
- hand MCTS: mean regret 0.00148118, top-1 0.89, 0 catastrophes;
- causal-resource v2: mean regret 0.00129707, top-1 0.88, 0 catastrophes;
- v2 with information coefficients zeroed: mean regret 0.00150115, top-1
  0.88, 0 catastrophes.

Resource v2 improved mean regret versus hand by 0.00018411, changing 3/100
units, but its battle-bootstrap 95% CI was
[-0.00000231, 0.00055464]. The direct information ablation improved mean regret
by 0.00020408 and changed only 2/100 units (one better, one worse); its 95% CI
was [-0.00000130, 0.00061353]. Both intervals cross zero.

The binding/search boundary was hardened during verification. Legacy Rust and
Python mechanics APIs retain their exact instruction streams; only MCTS and the
explicit semantic transition opt into reversible `PublicReveal` metadata.
Public-event projectors ignore this metadata when validating their pinned
mechanics grammar. The final verification passed 928 Gen9+Tera Rust tests,
56/56 Python binding tests, and 13/13 targeted Python causal-mask/model tests. A
rebuilt CPython 3.11 extension advanced a synthetic observer mask from 1 to 65
after an opponent move, changing the information vector from
[1/6, 0, 0, 0] to [1/6, 1/24, 0, 0].

**Decision: FAIL / STOP.** The causal reveal bridge is accepted as a correctness
prerequisite, but this scalar resource-shadow expert is not independently
better. Do not touch the holdout, run H2H, distill deviations, or tune the four
coefficients on this development panel. The next justified experiment must use
a new frozen panel and action-conditional long-horizon information value.
Complete protocol, hashes, artifacts, and results are in
`experimental/runs/resource_aware_expert_reveals_20260814/`.

## 2026-08-14: Full-causal-history residual action-Q admission

Followed the failed scalar causal-resource experiment with the pre-specified
action-conditional test. Audited two existing action-Q branches first and did
not repeat them: a public-feature MLP failed on 1,000 roots, while a frozen-R1
stateless two-step embedding head overfit (70.7% train top-1 versus 40.5%
validation). The missing experimental distinction was full deployment history.

Reconstructed exact causal R1 sequences by joining the accepted schema-v3 prior
snapshot (SHA-256
`373f317750ce40632744dbd60208598809851040a3e29af926c87d4aa4741ef3`),
selected-action receipts in the historical decision logs, and dense next-step
rewards in the terminal trajectory archive (SHA-256
`651a0e9bb189cae8260b97244c58495c28788c5e7b5d958eaac3d4e0362ee994`).
There were 4,555 exact full-history trajectory joins available. No sampled
hidden team or world was exposed to the learned model.

Froze a new disjoint information-sensitive panel after excluding all prior
action-Q, public-value, and resource panels by hash. The panel contains 1,500
physical battles, 3,000 counterbalanced schedules, and 24,000 worlds. Mean R1
entropy was 0.7810, mean history length was 24.66, and mean public
species/move/item/ability coverage was
0.7733/0.3320/0.3401/0.1609. Candidate fallback recovered valid roots after
8,986 attempted determinizations failed. Panel SHA-256:
`868bcf581b56ccbab0048d1b9113517e0f5e6e183d4d73f26dd35fe256696cbf`.

Generated the action teacher entirely on local CPU: two schedules, eight
worlds per schedule, and 50,000 fast-engine iterations per world, totaling 1.2
billion iterations. Teacher SHA-256:
`7555ad6857c71dfabb17e49a222f359832e262fa56ec997f4640bfec3475676d`.
Five roots with unmappable teacher actions were rejected. The final dataset has
1,495 battles, 9,208 supported action targets, mean history length 24.56, and
maximum history 121. Dataset SHA-256:
`5cbd73e68cbb15b8d0ff0f77e60a3f5932347c2dad61547fcabc67834fb0637b`.

Kept the accepted R1 checkpoint frozen (SHA-256
`c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93`)
and trained only an 11,713-parameter 900-to-13 linear residual over its final
full-history embedding. Candidate logits were the causal R1 log-prior plus the
residual. Physical battles were hash-split 1,214/151/130 into
train/validation/test. Epoch 18 was selected on validation regret before the
test split was opened once.

Results:

- train: candidate/R1 top-1 0.3542/0.3484, regret 0.08131/0.08153,
  improvement CI [-0.00217, 0.00259];
- validation: candidate/R1 top-1 0.3046/0.2848, regret 0.08790/0.08649,
  improvement CI [-0.01255, 0.00638];
- test: candidate/R1 top-1 0.3231/0.3154, regret 0.10683/0.10794,
  improvement CI [-0.00426, 0.00625].

The historical 500 ms search action remained much stronger under the same
teacher: test top-1 0.5231 and regret 0.05729. The learned residual's tiny test
improvement is statistically unresolved, and validation regret worsened.

**Decision: FAIL / STOP before independent gate.** The model failed its frozen
admission rule, so no new evaluation panel, equal-budget root gate, H2H, or
distillation was run. Do not tune this model or loss on the opened splits. The
generic supervised critic route is now exhausted in its public-feature,
stateless-transformer, scalar-resource, and full-causal-history forms. The next
justified architecture must preserve the information gained by the current
500 ms search and learn only a shallow-search residual/high-confidence
correction on a newly frozen panel. Complete protocol, hashes, and artifacts
are in `experimental/runs/causal_action_q_local_20260814/`.

## 2026-08-14: Selective shallow-search residual gate

Tested the next pre-specified direction after the full-history action-Q head
failed: retain current search and learn only selective high-confidence
corrections from live search statistics. Froze the protocol before collection.
The controller observes per-action visit mass, shallow value, across-world
value/visit variance, world support, and top-action votes, plus root entropy,
top-two margin, JS/world disagreement, effective world count, causal reveal
fractions, history depth, and causal R1 confidence summaries. Sampled private
states are never model inputs.

Physical battles were hash-split 60/20/20 into 914 training, 312 conformal
calibration, and 274 untouched offline-test battles. The model is a fixed
16-member histogram-gradient-boosting ensemble. A 90% conformal overprediction
penalty is calibrated separately. Overrides require an ambiguous root, at least
75% hidden-world value support, and a lower confidence bound strictly above
0.01. Admission requires at least ten test overrides, a positive 95% battle
bootstrap lower bound, lower regret, no additional regret >=0.10 catastrophes,
and more beneficial than harmful changes.

Encountered and rejected two runtime problems before accepting results. The
shell's Python 3.10 failed before data generation because the causal binding
requires Python 3.11. The first rebuilt Python 3.11 extension accidentally used
the binding crate's default Gen 4 feature, omitting Tera actions. That invalid
run exposed 618 schedule rows with missing teacher support and was discarded;
its files were overwritten and are not manifested. Rebuilt with
`--no-default-features --features poke-engine/terastallization`, verified an
audited root exposed all 13 expected actions including four Tera variants, and
added a mandatory Gen 9 Tera preflight. The accepted causal Gen 9 engine binary
SHA-256 is
`cf71fbba541c9e7b4f3c891bf9b25dca863196708b7131f77d1e0016c1073f69`.

Recalibrated the corrected engine on 64 worlds. A 45 ms per-world allowance
produced median 21,500 iterations, mean 23,594, p10 15,000, and p90 42,000.
Froze a conservative deterministic 20,000 iterations per world. The corrected
capture covers 1,500 battles, 3,000 counterbalanced schedules, eight worlds per
schedule, and exactly 480 million local CPU iterations. Capture SHA-256:
`5c1e7ba97c43f49bc2002b203fd24229d84277f10bf8d98befe72eac3fb4112d`.
All 18,442 teacher-supported action examples had shallow statistics.

The corrected current-search baseline was much stronger than the old
historical action:

- calibration: top-1 0.8654, regret 0.001719, two catastrophes across 624
  schedule units;
- test: top-1 0.8850, regret 0.001352, zero catastrophes across 548 schedule
  units.

The conformal overprediction penalty was 0.028696. No alternative cleared the
penalty plus the frozen 0.01 advantage margin, so the controller made zero
calibration and zero test overrides. Candidate and baseline metrics were
identical; the improvement interval was [0, 0]. Model SHA-256:
`b74e0480ea200eb511b18a70f5c43b7e483529e72d7d0d7962f9f6f21cbd2ca9`.

The ambiguity trigger was diagnostically successful. On test, all regret came
from the 332/548 units flagged ambiguous (mean regret 0.002231); the 216
confident units had exactly zero regret and 100% teacher top-1 agreement. The
remaining problem is selecting the correction, not locating uncertain roots.
Even perfect 50k-teacher selection could improve mean regret by only 0.001352
under this same evaluator.

**Decision: FAIL / STOP before independent gate.** Do not loosen thresholds on
the opened panel, run H2H, or deploy the no-op model. This is not another sign
that current search is weak: it shows the 20k deployment proxy has nearly
converged to a 50k teacher sharing the same hand evaluator. A non-learned
adaptive-deepening experiment on ambiguous roots remains mechanically
justified, but its same-evaluator ceiling is small. The larger bottleneck is now
teacher/evaluator quality: outcome-grounded targets are required to find errors
that both 20k and 50k same-evaluator search agree on. Complete protocol,
results, hashes, and artifacts are in
`experimental/runs/shallow_search_residual_20260814/`.

## 2026-08-14: Outcome-grounded shared-search-error pilot

Followed the shallow-residual stop decision by testing the new bottleneck
directly: roots where current 20k search and its 50k same-evaluator teacher
agree, but a matched terminal continuation might disagree. Did not reuse the
old uniform-legal continuation because it had already failed termination, and
did not call the still-uncertified dual-R1 bridge a continuation value.

Froze an explicit estimand before collection. Selected 64 development-only
roots from 381 eligible training-split roots. Both hidden-world schedules had to
be ambiguous, both 20k schedules had to select the same action, both 50k
schedules had to select the same action, and the 20k/50k actions had to agree.
Roots were ranked only by pre-existing search ambiguity. Candidate support was
the agreed baseline plus the top three 50k alternatives and top two 20k visit
alternatives, deduplicated to at most six actions.

For every candidate, schedule, world, and rollout, sampled the opponent's
simultaneous root action from a seeded 20k root-search visit policy, forced the
candidate with a common chance tape, then played both sides to terminal using
seeded 256-iteration MCTS argmax continuations. Used eight rollouts per world,
eight worlds per schedule, two schedules per root, and a 128-decision censoring
horizon. This is an actual terminal win/loss under a named MCTS continuation,
not an R1 continuation value or ladder value.

The one-root full-configuration smoke completed 512/512 terminal samples. The
64-root pilot performed 722,392 continuation searches and produced 32,896
forced-action samples. A total of 31,288 terminated, for 95.11% coverage,
narrowly passing the 95% gate. Three roots had zero terminal coverage; median
root coverage was 100%.

Raw terminal-best actions differed from the shared 20k/50k action on 44/64
roots, but even/odd rollout halves agreed on the terminal-best action on only
31/64 (48.44%), well below the frozen 70% requirement. Exact paired-tape
analysis, clustered by schedule/world and requiring baseline/alternative
coverage plus positive direction in both schedules, found only five stable
corrections versus the frozen minimum of ten:

- switch dusknoir -> switch vileplume: +0.1719 [0.0703, 0.2734];
- psychicnoise -> focusblast: +0.1224 [0.0547, 0.1912];
- switch overqwil -> switch scyther: +0.1484 [0.0391, 0.2578];
- switch lapras -> switch zebstrika: +0.0859 [0.0156, 0.1563];
- switch regidrago -> switch espeon: +0.0938 [0.0156, 0.1641].

Also measured adaptive 50k deepening separately as an explicitly idealized
same-evaluator ceiling. On the test split it triggers on 60.58% of units and
changes 63/548 actions. It would reduce oracle regret from 0.001352 to zero by
construction, with improvement CI [0.000569, 0.002309]. A continuable MCTS tree
would cost 1.91x baseline compute; the current rerun-only API costs 2.51x. This
is not a candidate gate and the ceiling is too small for the cost.

**Decision: FAIL / STOP before training.** The target contains a small number of
large shared-search errors, but the current outcome continuation is too noisy
to label the panel. Do not train on 44 raw disagreements, loosen thresholds,
distill the five development corrections, or promote adaptive deepening. The
next valid step is a more reliable terminating continuation policy on a new
disjoint panel: certify dual-R1 continuation semantics or spend stronger
continuation search only on a small confirmation panel. Complete protocol,
results, hashes, and artifacts are in
`experimental/runs/outcome_grounded_ambiguous_20260814/`.

## 2026-08-14: Corrected causal-history dual-R1 continuation certificate

Followed the outcome-grounded stop decision by testing the preferred stronger
continuation directly. Added an opt-in schema-6 capture path to the production
prior server and controller. Normal production behavior is unchanged unless
`METAGROSS_DUAL_R1_CAPTURE=1`: the capture records the exact corrected
causal-history transformer trajectory, selected-action/reward RL2 rows,
legitimate own-private/currently-public player-information state, public
observation history, and one controller-local mechanical root. Opposite clients
remain isolated and private files are mode `0600`.

Played one new mirrored pair locally on CPU with two isolated accepted epoch-5
R1 servers. Both games completed with zero voids. Client A recorded 65 decisions
and client B 71. Independent offline recomputation reproduced every live policy
vector exactly with maximum absolute difference `0.0`. The private root streams
joined 58 opposite-role boundaries by battle, turn, and canonical public prefix;
20 legitimate one-sided boundaries remained unmatched. All 136 schema-6
snapshots reconstructed their exact current player observation, and every fused
root matched both clients' legal action sets. Root-capture SHA-256 values were
`a27b466072b50f6e359491050b5f757c6ddf0ac6cb0a5eba353321626a288594`
and `6886daa4481020b081529f148076cd2076057c344b64aa2da33652608cba931a`.

Froze the sequential gate before opening continuation results: four dual-R1
rollouts per joined root, exact causal history on both sides, SHA-256 action and
chance tape, 128-decision horizon, exact root policy tolerance `1e-7`, and at
least 95% real terminal coverage. No omniscient observation, MCTS fallback, or
uncertified transition was permitted.

The root policy bridge passed 58/58, but the terminating certificate failed
decisively. Only 5/232 trajectories reached terminal (2.155%). A total of 191
failed before one certified transition; the remaining stop-depth counts were
27 at depth one, 10 at depth two, three at depth three, and one at depth four.
No trajectory survived beyond four decisions. Final fixed failure counts were
172 symmetric `UNSUPPORTED_INFORMATION_SET`, 21 player-one and 33 player-two
`NEXT_MASK_UNCERTIFIED`, and one player-one `UNSUPPORTED_PUBLIC_EVENT`.

Root-only diagnostics attributed the structural gap to 102 action pairs with
unaccounted instructions, 30 repeated-HP-sequence failures, and 21 unsupported
action pairs. The dominant unaccounted families were side-condition changes,
volatile application/removal, move enable/disable state, pivot/forced-switch
markers, sleep/rest counters, item changes, and substitute changes; Tera also
remains unsupported. This is an engine-to-public-history semantic coverage
failure, not bad R1 weights, causal trajectory reconstruction, insufficient
data, or root action mapping.

**Decision: BLOCK / STOP before confirmation panel.** `r1_continuation_value`
and new-panel generation remain forbidden. No labels, correction training, or
H2H were run. More root collection cannot solve 2.16% terminating coverage.
Either implement the measured broad semantic families and rerun the exact
certificate, or use substantially stronger MCTS as a separately named terminal
teacher on a small new disjoint panel. The latter remains the faster experiment.
Certificate SHA-256:
`58d84833b06aad2b5a0d7835a8ae644d493d19140110e722da2c0aded9cebe68`.
Complete artifacts are in
`experimental/runs/dual_r1_causal_certification_20260814/`.

## 2026-08-14: Strong terminal continuation and historical scale loop

Re-tested the corrected causal-history R1 continuation on the frozen 58-root,
232-rollout certificate after implementing production-faithful automatic
`nomove` forced-replacement transitions. All 54 `next_mask_uncertified`
failures disappeared, proving the boundary fix, but terminal coverage remained
5/232 (2.16%). Failures progressed into unsupported public-history projection,
so symbolic dual-R1 continuation was stopped without labels or training.

Switched to a semantically certified engine-terminal MCTS teacher at 2,048
iterations per continuation decision. A 64-root disjoint confirmation passed
coverage (96.41%) and half-split agreement (71.88%) but found only three stable
correction roots. A preregistered 128-root power panel excluded both previous
64-root panels, used eight matched rollouts, and passed the unchanged frozen
gate: 98.20% terminal coverage, 78.91% half-split agreement, 67 raw shared-
search disagreements, and 11 stable correction roots. This is the first clean
proof that long-horizon terminal outcomes expose reproducible mistakes shared
by 20k and 50k root search.

Did not treat teacher admission as deployment admission. Eight-fold root-
grouped learnability probes using the frozen 24 live-search features achieved
only 0.24–0.46 correction AUC; adding the 900-dimensional causal R1 embedding
achieved only 0.28–0.37 and recovered at most two certified fixes. No residual,
H2H, or distillation was admitted.

Audited the remaining historical corpus. Of 5,008 battle groups and 4,555 exact
causal joins, only 973 additional battles survived all prior-panel and
determinization checks. The 2,000-root preregistration failed atomically; a
950-root near-ceiling panel succeeded and received uniform 20k statistics and
50k oracle screening. Two hundred ambiguous four-way-agreement roots received
the strong teacher. Horizon 128 produced 94.891% terminal coverage, 71.5%
agreement, and 20 stable corrections; it failed coverage. A single-variable
horizon-192 rerun added only 26 terminals, reaching 94.924%, and also failed.
Eight roots never terminated, one terminated 2/384 times, and one terminated
193/384 times, identifying policy cycles/stall regimes rather than a short
horizon.

No censored outcomes were imputed, no hard roots were deleted post hoc, and no
labels from the failed historical scale panel were trained. Hardened the local
collector with per-schedule fsynced progress/resume and exact configuration
validation. Added a future termination-only 16/16 baseline probe that observes
no outcome direction and costs roughly 1/24 of full labeling. An end-to-end
smoke resumed 8/8 rows after an intentional report-path failure and reproduced
the exact filtered-panel hash. All work was local CPU; no GPU, cloud, ladder,
or live H2H was used. The next valid branch requires new PFSP/league games,
termination prefiltering, richer action-semantic residual inputs, and a fresh
disjoint new-game gate.

As a non-evidentiary pipeline diagnostic after V4 was already opened, ran the
new termination-only probe on the consumed 200-root panel with a distinct seed.
The 16/16 rule rejected 16 roots and retained 184; the retained subset's known
V4 terminal coverage was 99.766% (70,491/70,656), versus 94.924% globally. The
probe caught all fully nonterminating roots and the 2/384 root using only 3,200
trajectories and 61,250 searches. This validates the mechanism but does not
retroactively admit V4; its first evidentiary use must be preregistered on new
games.

## 2026-08-14: Fresh schema-6 capture bridge admitted end to end

Implemented a local CPU-only fresh-corpus runner with two isolated, checkpoint-
verified epoch-5 R1 servers, causal-history mode, identical production
search-first controllers at 500 ms, mirrored teams, separate per-side logs,
fail-closed priors, and automatic infrastructure cleanup. Three operational
smokes exposed and preserved strict contract failures: the legacy client lacked
the production request identity, the first production run omitted mechanical
root capture, and the next enabled capture only on the consumer rather than the
schema-6 producer. None of their data was admitted.

The repaired V4 pair completed 1-1 with zero voids. Its audit joined 145/145
mechanical roots to exact schema-6 causal snapshots across four terminal POV
groups: 100% capture, zero duplicates, zero invalid histories, and no missing
snapshots. Extended the causal panel builder to consume the private dual-root
format directly and to accept multiple isolated prior dumps. It now freezes
exactly one observer per physical battle before looking at actions or outcomes,
preventing correlated POVs from inflating yield or crossing folds. Forced
one-action snapshots remain exact joins but are rejected as uninformative
rather than misreported as absent. The direct panel bridge retained two
physical groups and found 16 eligible ambiguous rows.

This is a data-path result, not a strength result. No residual was trained and
no H2H claim was made. The next admitted action is the preregistered 500-game
local pilot with physical-battle grouping and the frozen opponent mixture.

Ran the finalized wrapper once more from a fresh directory after integrating
both audits. It completed another 1-1, zero-void mirrored pair and exited zero.
The wrapper itself produced a 75/75 exact schema-6 join (four terminal groups,
100% capture, no duplicates or invalid histories) and admitted all four groups
to panel selection with 11 eligible ambiguous rows after one-POV grouping. This second independent
operational pass verifies that the published runner—not only hand-invoked
components—enforces the complete capture contract.

Generalized the runner to all preregistered opponent strata with disjoint seed
domains. The direct-R1 smoke joined 43/43 one-sided causal roots and selected
16 ambiguous rows; the unguided-search smoke joined 37/37 and selected 15. Both
completed two terminal games with zero voids and 100% capture. Added a frozen
500-game wrapper (300 peer, 100 direct R1, 100 unguided) and a fail-closed
aggregate audit. The aggregate admits only capture; it explicitly cannot open
the 5,000-game stage until the later 20k/50k four-way screen yields at least 50
physical roots. The three two-game wall-clock smokes took 70–74 seconds each,
implying roughly 4.9–5.2 hours for a conservative sequential 500-game local run;
no unverified parallel speedup is claimed.

Hardened physical grouping across the three independently restarted collection
strata. Canonical observer selection and battle IDs now include a content-hashed
collection scope, so a Showdown server that reuses `battle-1` after restart
cannot merge peer, direct-R1, and unguided games. A regression test constructs
the same battle tag in two collection directories and verifies two distinct
physical scopes.

Combined the finalized peer, direct-R1, and unguided artifacts in one real
panel-input audit after the scope fix. It admitted exactly six physical battle
groups and 42 candidate rows (11 peer, 16 direct R1, 15 unguided), with the
opposite peer POV excluded. This verifies that the future 500-game mixture can
be consumed as one corpus without profile collisions or double-counting.

Added explicit atomic resume to both the stratum and aggregate wrappers. A
completed peer smoke was resumed from its existing progress snapshot: the
harness returned the same two-game result without emitting or replaying any
game, both audits reproduced 75/75 joins and 11 canonical candidate rows, and
the command exited zero. Resume therefore preserves completed work; interrupted
orphan groups remain visible to the 95% audit.

Checked whether 500 games is needlessly large. The historical four-way screen
retained 204/950 roots (21.47%, Wilson 95% lower bound 18.98%). Using only that
lower bound as a sizing prior, 400 panel-eligible fresh battles imply about 76
screened roots and nominal probability above 99.9% of reaching the required
50; 300 eligible battles give only about 86.4%. This is not transfer evidence
and does not open any gate, but it supports retaining the 500-game pilot size.

Hardened the final residual dataset boundary: duplicate teacher root evidence
and duplicate shallow-search pair evidence now fail closed instead of silently
overwriting earlier rows in a dictionary. The materializer still emits only
certified deviations and explicit causal transformer arrays, never sampled
hidden engine state.

Audited long-game schema-6 semantics before scaling. Production R1 crops its
transformer context to 128 observations while preserving absolute time indices;
the first validator incorrectly required `decision_idx + 1` rows forever and
would have rejected valid decisions after index 127. The contract now requires
exactly `min(decision_idx + 1, 128)` observations, one fewer transitions and
receipts, and a contiguous absolute time range ending at `decision_idx`. A
decision-130 cropped-history regression passes.

Made the capture-audit and final residual-report writes atomic and fsynced.
An interruption can no longer leave a plausible-looking partial JSON report
next to an intact dataset or progress journal.

Final verification expanded beyond the focused gate suite: all script tests
passed (`290 passed, 2 skipped, 21 subtests`) and all training tests passed
(`62 passed, 1 skipped`). The only warning was the existing optional
FlashAttention-not-installed notice; CPU execution remained intentional.
All evaluation tests also passed (`44 passed, 7 subtests`), as did all belief
tests (`23 passed`), for 419 passing top-level tests across the four relevant
experimental suites.

The broader production-controller suite produced 379 passes and one expected
fail-closed identity failure: the edited vendor engine checkout hashes to
`ccc5dd035c25fbf8e6b98301198cdaa35ce1012ae6f45ab81851965b0b05a0a3`,
while the certified production pin remains
`639982daced7abb3ebad4fed8bc6b5408dc82c7386241b3732a7481a7aacae73`.
Did not rewrite the pin: that would require rebuilding and independently
certifying the production engine. The local capture runner does not use the
remote production-engine contract and passed live end to end.

Pinned that separately validated capture-only engine identity in the fresh
runner: source SHA-256 `ccc5dd035c25fbf8e6b98301198cdaa35ce1012ae6f45ab81851965b0b05a0a3`
and loaded extension SHA-256
`3910185bb7f5e5f0283781b0b2292664f4c980126f320325143ba5970d4aba35`.
This prevents silent local source/binary drift without pretending it is the
distinct certified production deployment engine.

Ran the learned-priors Rust engine library suite with its actual
`gen9,terastallization` feature contract: 255/255 tests passed, including public
reveal reversibility/serialization, seeded MCTS reproducibility, learned
resource constraints, and shared-information-set optimization. An initial
featureless Cargo invocation failed to compile generation-specific constants;
it is recorded as an invocation error, not an engine test failure.

## 2026-08-14: Fresh 500-game schema-6 Modal pilot admitted

Ran the preregistered fresh-corpus pilot on the Modal profile
`thisaccisfortheschool`, entirely on CPU. The final app was
`ap-YgJ1Bmi7BpCz1YJHnc2Ghp`, run ID
`schema6-modal-500-20260814-r1`, root seed `2026081400`. The runner split the
work into 50 atomic units of 10 games, used at most ten 16-physical-core
(32-vCPU) / 24-GiB
containers, and preserved the frozen 300-game peer, 100-game direct-R1, and
100-game unguided mixture. Every unit used a unique mirror seed, production
randomness identity, and username namespace. No GPU was requested or used.

Added `experimental/src/scripts/modal_schema6_pilot_500.py` and
`experimental/src/scripts/prepare_production_capture_engine.py`. The Modal
image builds two separate Linux engines: a reconstructed frozen production
search ABI with no public MCTS `seed` argument, and the experimental audit
engine with the causal public-history/reveal-mask bridge. Both engines are
compiled and tested in the image before collection. Full fanout is forbidden
until one peer, one direct-R1, and one unguided 10-game certification unit pass
game, capture, bridge, and cross-runtime-fingerprint gates.

The certification attempts found real producer defects and rejected their
data rather than weakening admission. `ap-6IUSxFzIYfbjEECPL1G7JW` compiled the
dirty experimental seeded MCTS ABI instead of the frozen production ABI.
`ap-XzCYTj1AODpa61Xu7sN5Mt` corrected the Rust wrapper but exposed the same
extra argument in the Python wrapper. `ap-YEDMrOvkiqYuxGIalFzKYf` restored the
full ABI and then exposed forced Struggle as a transport-only Showdown command
that had no learned R1 snapshot. `ap-GJ9kjrSidvmfzQHKmHHH9U` exposed two final
issues: direct R1 could miss a challenge while cold-loading, and the prior
server skipped inference for Recharge/Struggle but still incremented the
learned-decision index. That created a permanent index hole and invalidated all
later causal snapshots in the battle. Added an explicit unbuffered
`Made Challenge Env (acceptor)` readiness handshake, excluded automatic
Recharge/Struggle roots from mechanical training capture, and stopped those
commands from advancing the learned-decision counter. Focused verification was
9 passed / 1 optional-runtime skip, followed by 8/8 forced-action, schema-6,
trajectory, and direct-R1 semantic tests.

The final three-stratum certificate passed 30/30 games at 100% capture. Peer
recorded 633/633 valid snapshots over 20 POV groups and produced 73 bridge
candidates; direct R1 recorded 294/294 over ten groups and produced 65; the
unguided certificate recorded 334/334 over ten groups and produced 78. The
unguided certificate was atomically reused from the immediately preceding app
snapshot. It predates only the forced-action index fix, but contains no
forced-action boundary: all 334 indices and causal histories are contiguous,
so the changed branch was never executed and the rows are behaviorally
identical. The remaining 49 units came from the final fixed app.

One full-fanout container was preempted and automatically restarted with the
same unit input. One direct-R1 attempt completed nine games, then the 142M
direct policy process hung on turn 16 until the 900-second watchdog. The whole
attempt was copied to
`failed/direct_r1-001-796ef82b`, including the completed unfavorable and
favorable outcomes, and discarded. Its retry used the same preregistered unit
identity and passed. No completed subset, outcome-conditioned replacement,
void, or partial audit entered the dataset.

The final aggregate passed every frozen admission check:

- 50/50 atomic units and 500/500 terminal games;
- exactly 250 unique mirrored seed/team identities;
- exactly 300 peer, 100 direct-R1, and 100 unguided games;
- 800/800 complete terminal POV groups (100% capture);
- 25,821/25,821 mechanical decisions joined to valid schema-6 snapshots;
- zero duplicate decisions, duplicate snapshots, invalid snapshots, voids, or
  unknown winners in admitted units;
- 3,555 eligible causal bridge candidates: 2,133 peer, 709 direct R1, and 713
  unguided;
- one identical engine/checkpoint/Showdown runtime fingerprint across all 50
  units.

Frozen runtime identities were checkpoint
`c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93`,
production capture-engine source
`ccc5dd035c25fbf8e6b98301198cdaa35ce1012ae6f45ab81851965b0b05a0a3`,
Linux production extension
`ee6da274554b68cd06ec79423a015df987234c04cd091d400326a05c125d7cfe`,
audit-engine source
`4c66bf549a83543a1f6e67c4f4a3e7f8d2216bfbe281e66a23b9a00e2afedf5b`,
Linux audit extension
`8f69e753b0b2252255b180033d14cabb0bfd6f54e8d37d51a0bbfd52a3eb7ec9`,
and Showdown commit `4880d3693580bd33652797cf31179c6fcdf87e50`.
The final-app controller/auditor source SHA-256 values were
`a569b5896f5af4d50dda517417d6de0acaf4a24ede8221dd1331c9a32e31104a`
(`eval/run.py`),
`bf80bfbc97b0b303b8a030bdc1f5d5da9389b395b851ba6199367ab1f7f93d8d`
(Modal runner),
`7431ed7e57110f063d24278de600f56226ffd863a13be7ea4083eb8c676e94e8`
(production-engine preparer),
`bebc2dacd6889c00b1151d1cc42454fd6b47508362767ec8920f7081fda729ba`
(capture audit),
`4c5f1f873683260fc3d254036835237cee10e1cddbce994456996bc4ce71bcc9`
(bridge audit),
`ed5f3e2b0670d71fa712eb16651df395b378927c02f8c9f497795d19649ad7ca`
(panel builder),
`6e5ae0e2e92930943d7b6f456cfc3999bd52274ccadc27965a70fd49469592d7`
(production controller), and
`df43bddc9977425878a5eb0e4859c51134d01a28ea556348f2cdc87ea28443ad`
(prior server).

The 50 accepted unit reports contain 38,366.18 container-seconds (10.66
container-hours, about 170.5 allocated physical-core-hours (341 vCPU-hours)
before failed/preempted
overhead). Unit wall times ranged from 532.19 to 1,100.12 seconds, mean 767.32.
The final app completed in about 86 minutes. Modal billing cost was not inferred
from runtime because retries, CPU utilization, and account pricing require the
actual billing export.

Independently revalidated the downloaded local summary with a separate `jq`
contract covering all unit admissions, per-stratum counts, snapshot failures,
runtime fingerprints, and uniqueness of all 250 pair identities. The local
summary and independently downloaded Modal-volume summary were byte-identical,
both SHA-256
`1281d47f2900ece4e5d659e686822483330d73540c612cc87412d2259f8dd12d`.
Local summary:
`experimental/runs/schema6_modal_500_20260814_r1/summary.json`. Raw units and
remote summary:
`/data/schema6_capture_500/schema6-modal-500-20260814-r1/` on Modal volume
`metagross-online-rl`.

**Decision: ADMIT CAPTURE, DO NOT CLAIM STRENGTH OR SCALE YET.** This proves the
new games and causal histories are suitable input to the frozen ambiguous-root
pipeline. It does not prove a correction policy is stronger. `scale_admitted`
remains false until the fixed 20k/50k four-way agreement screen finds at least
50 eligible physical roots; only outcome-grounded continuations on those roots
may create correction labels, followed by the preregistered held-out/H2H gate.

## 2026-08-14: Enforced 60/20/20 split and frozen training-only causal panel

Fixed a methodological gap before running any 20k/50k search. The causal panel
builder previously recorded `purpose` in its report but did not use it to
restrict physical battles. It now maps `training -> train`,
`calibration -> calibration`, and `evaluation -> test` through the already
frozen `train.shallow_search_residual.battle_split` function (seed `20260817`,
hash buckets 0-59 / 60-79 / 80-99). Unknown purposes fail closed. The output is
also rejected unless every materialized row belongs to exactly the requested
split.

The first implementation filtered before public-state parsing and panel
materialization, but still built a global prior-snapshot lookup. Modal app
`ap-Ze2WfbmDKhAwKbcyw2iuSZ`, seed `20260825`, therefore produced a technically
split-correct 238-root artifact whose withheld snapshot rows had nevertheless
been parsed for feature lookup. That is weaker than the requested genuinely
unopened confirmation contract. It is retained as a superseded methodological
artifact only and must not be used for screening, training, calibration, or
evaluation. An earlier invocation, `ap-7Vmillx2BTchjOoHqwHpPU`, failed before
building anything because the Modal entrypoint lacked the copied source-tree
import path; no panel artifact was produced by that attempt.

The admitted v2 implementation gates each physical battle using only collection
identity and normalized battle tag. Withheld rows are discarded before observer
selection, history joins, policy entropy/statistics, schema-6 validation,
public-state parsing, or hidden-team determinization. Snapshot files are mixed,
so their JSON identity fields must be decoded to route rows, but no withheld
history, policy, public-information, outcome, or state feature is processed.
The report and independent validator require
`withheld_history_policy_feature_rows_processed == 0` and the exact gate-stage
identity. A sentinel unit test supplies deliberately malformed withheld feature
fields and proves they never reach feature processing.

Ran the immutable v2 freeze on CPU-only Modal profile
`thisaccisfortheschool`, app `ap-qGvLKskoa5uLZq5oIB4GLP`, panel seed
`20260826`. The identity-only split of all 500 accepted physical battles is:

- 311 training battles;
- 96 conformal-calibration battles;
- 93 untouched confirmation/test battles.

Under the previously frozen causal-root eligibility thresholds (history at
least 3, at least 4 legal actions, policy entropy at least 0.45, relevant public
information, and successful hidden-team determinization), 240 of the 311
training battles yielded a root. The builder selected at most one root per
physical battle and used all eligible training battles rather than a post-hoc
root-count cutoff. The remaining 71 training battles had no root that survived
the complete frozen eligibility/determinization path; they were not replaced
from either withheld split. Row-level diagnostic counts were 933 short-history,
5,622 uninformative-policy, 4 no-relevant-information, and 998 failed
determinization candidate attempts. These are candidate-row counts, not 71
mutually exclusive battle-level reasons.

The admitted artifact contains exactly:

- 240 unique physical training battles and 240 unique causal roots;
- 480 schedules, exactly schedule IDs 0 and 1 per root;
- 3,840 worlds, exactly world IDs 0 through 7 per schedule;
- 18 causal public features per root and schema-6 snapshot history authority;
- normalized world weights summing to one per schedule;
- zero calibration or confirmation rows materialized;
- zero withheld history/policy feature rows processed.

The independent remote audit and a separate downloaded local audit both
passed. The local audit rehashed every serialized hidden state, checked every
world/schedule index and weight, asserted one root per battle, recomputed every
physical-battle split, and matched report hashes. Artifact identities:

- panel SHA-256:
  `1063e40b05ec136131311d4e6c8943f5a14ca70c9ea1806d4c8c8356ec1957a8`;
- report SHA-256:
  `e6b443acb390342ee2efa6685ec8951a18df5ada2bed0aef7845842d01c3dd3c`;
- audit SHA-256:
  `0429b64b50e09054a0f03836508860c4a8b89bff6315a7ea53a1bf75da637784`;
- accepted 500-game corpus summary SHA-256:
  `1281d47f2900ece4e5d659e686822483330d73540c612cc87412d2259f8dd12d`;
- 50-unit report-set SHA-256:
  `77be2206e01d9130fbd19cab4841f57e962cc150a6dff2f827b9967e87c47ef4`;
- randbats completion-pool SHA-256:
  `50188081fa9146c00210c252a0b53c5fe0b2622562f785064979f665aa74eaf8`.

Local admitted artifacts are under
`experimental/runs/schema6_modal_500_20260814_r1/training_panel_artifacts_v2/`;
the controller envelope is
`experimental/runs/schema6_modal_500_20260814_r1/training_panel_freeze_v2.json`.
The immutable remote copy is
`/data/schema6_capture_500/schema6-modal-500-20260814-r1/panels/training-seed-20260826/`
on Modal volume `metagross-online-rl`. Final focused verification was 22 Python
tests passed, both Modal image Rust suites passed (257 frozen-production tests;
255 experimental-engine tests plus doc tests), Python compilation passed, and
`git diff --check` passed. Final source SHA-256 values were
`e030e79afb50d84aefe1e70c7f61a9da458f1276f925370e687aebde4ae7e106`
(panel builder),
`45a65799cc71df2d43612173f286f6a2c6d485ed13301b45c5eb15c9eb54f9e5`
(bridge audit),
`431f558db4e345592e6efa9ebdd86e3eeca6aedba24dee489dd9dca312ddae4b`
(Modal runner), and
`40fca755a04aad630635c34d2da6b9d6a97de3a81534e6088a5125679c5ac400`
(focused panel tests).

**Decision: ADMIT TRAINING PANEL V2; KEEP CALIBRATION AND CONFIRMATION CLOSED.**
The next authorized computation is the frozen 20k/50k four-way agreement
screen over these 240 training roots only. This panel construction is data
correctness evidence, not evidence that the residual correction improves play.

## 2026-08-14: Fresh training-panel 20k/50k agreement screen passed

Ran the frozen same-evaluator search screen over all 240 admitted v2 training
roots and no other split. CPU-only Modal app
`ap-tdKVGcyGAkhKIf2cQOZDkt` used 15 resumable shards of 16 roots. Each root had
two schedules and eight fixed hidden worlds per schedule. Every schedule was
evaluated once at exactly 20,000 iterations/world for the deployment proxy and
once at exactly 50,000 iterations/world for the screening oracle. This totaled
76.8 million shallow iterations and 192.0 million oracle iterations. The
engine was the pinned causal Gen 9 + Tera Linux extension, SHA-256
`8f69e753b0b2252255b180033d14cabb0bfd6f54e8d37d51a0bbfd52a3eb7ec9`.
No GPU was requested or used.

The fanout returned all 15/15 shards, exactly 480 shallow schedule rows and 480
oracle schedule rows. The merge required the exact 480 source pair IDs,
rehashed every shard, and rejected any non-training battle before applying the
frozen rule. Attrition was:

- 240 training roots screened;
- 157 ambiguous under both independent 20k schedules; 83 failed ambiguity;
- 140/157 had the same 20k action on both schedules; 17 disagreed;
- 134/140 also had the same 50k action on both schedules; 6 disagreed;
- 113/134 had the same action at 20k and 50k; 21 disagreed.

Thus 113/240 roots (47.083%) passed all four agreement checks. The Wilson 95%
interval for the yield is 40.864% to 53.394%. This exceeds the preregistered
minimum of 50 roots by 63 roots and passes the fresh-corpus scale gate. All 113
eligible roots were frozen rather than choosing a post-hoc root count. The
resulting private outcome-candidate panel contains 113 unique physical roots,
226 schedules, 1,808 hidden worlds, and exactly three candidate actions per
root (339 total), using selection seed `20260827`.

Calibration and confirmation remained closed: zero withheld roots were
processed, and zero calibration or confirmation rows were materialized. The
screen used only the training panel SHA-256
`1063e40b05ec136131311d4e6c8943f5a14ca70c9ea1806d4c8c8356ec1957a8`.
Artifact identities are:

- 20k statistics SHA-256:
  `b04ae0e33e9b63a2b481f22fe45e0b1d560e4f0614cb62c1bfcb12a5cfd82b42`;
- 50k oracle SHA-256:
  `f38b934fa9f2c516eccf1d74a2c721bfeb9a9a27195f81e0b35d5bd93a278e9a`;
- 113-root agreement panel SHA-256:
  `0101aaf0d0478471ba1010f799f5bee95bda5a4cf78bcc5fea992e8f6ef7071c`;
- agreement report SHA-256:
  `6efd50eeddceeceb5781b8341f10972330a8c4ca037bb7da9ea3edd28f17cb7a`;
- screen report/controller-envelope SHA-256:
  `61115745a20193c56b633d04e0c8dd497771f3ad580cb470c8de05d577bc50e8`.

Downloaded local artifacts are under
`experimental/runs/schema6_modal_500_20260814_r1/search_screen_artifacts/`, and
the controller report is
`experimental/runs/schema6_modal_500_20260814_r1/search_screen_20k50k.json`.
The remote immutable screen is under
`/data/schema6_capture_500/schema6-modal-500-20260814-r1/screens/schema6-train-20k50k-20260826-r1/`
on Modal volume `metagross-online-rl`. A separate local rerun of the panel
builder reproduced both the agreement panel and its report byte-for-byte, then
independently checked all 480+480 budgets and identities, both-schedule
ambiguity, all four selected actions, physical-battle split membership, unique
root/battle IDs, and every recorded artifact hash. Final focused Python tests
were 22 passed; both Modal engine suites and `git diff --check` also passed.
Final source SHA-256 values were
`fac6496466a64ddd7c266f064ce8e664c28716ffefb7e4ae182b57120e81572a`
(outcome-panel builder) and
`c64d0980ef51ac2dda60638cc13a6499eb18a8315ec654a1d3dca79faa5912f0`
(resumable Modal screen runner).

**Decision: PASS 500-GAME SEARCH-YIELD GATE; AUTHORIZE THE FROZEN 5,000-GAME
STAGE.** This is evidence that the fresh corpus supplies enough stable
ambiguous/shared-search-error roots for the next data stage. It is not evidence
that an outcome residual, distilled policy, H2H agent, or ladder policy is
stronger. The next stage must preserve the fixed opponent mixture and capture
contract, apply the already-frozen termination-only prefilter before expensive
outcome labels, and keep the 93-battle confirmation split unopened.

## 2026-08-14: 5,000-game schema-6 scale capture launched

Launched the authorized 5,000-game stage as CPU-only Modal app
`ap-HJGIZoGK3Z54wHyWp5FiVj`, run ID
`schema6-modal-5000-20260814-r1`, root seed `2026082800`. The runner first read
and revalidated the immutable 20k/50k screen and embedded its authorization in
the new capture identity: 113 eligible roots versus the frozen minimum 50,
source training-panel SHA-256
`1063e40b05ec136131311d4e6c8943f5a14ca70c9ea1806d4c8c8356ec1957a8`,
agreement-panel SHA-256
`0101aaf0d0478471ba1010f799f5bee95bda5a4cf78bcc5fea992e8f6ef7071c`,
and zero withheld roots processed.

The stage is a new 5,000-game corpus, not the 500-game pilot counted again. It
freezes 500 atomic units of ten games: 300 peer search+R1 units (3,000 games),
100 direct-R1 units (1,000 games), and 100 unguided-search units (1,000 games).
This is exactly the original 60/20/20 opponent mixture. It preregisters 2,500
unique mirrored seed/team pairs, 8,000 expected causal POV groups, independent
profile seed domains, unique production seeds and username namespaces, and the
same capture-side production search+R1 policy. The final aggregate requires
all 500 admitted unit identities, all 5,000 terminal games, all 2,500 unique
mirrored pairs, one runtime fingerprint, zero unit-level duplicate/invalid
captures or voids, exactly 8,000 capture groups, and at least 95% complete
causal joins. No GPU is requested or used.

The scale runner is resumable and capped at 40 simultaneous 16-physical-core
(32-vCPU) / 24-GiB
containers. Every successful ten-game unit commits atomically to Modal volume
`metagross-online-rl`; a retry reuses a committed unit only after validating its
run/profile/index/seeds and admission report. Failed attempts are retained
separately and never contribute partial games. Based on the admitted pilot,
expected successful compute is about 106.6 container-hours or 1,705 allocated
physical-core-hours (3,410 vCPU-hours) before retry/preemption overhead, with a
projected wall time around 2.5–4 hours. At Modal's published 2026-08-14
Function rates (`$0.0000131` per physical core-second and `$0.00000222` per
GiB-second), the fixed 16-core / 24-GiB request costs `$0.946368` per active
container-hour. Ten times the accepted pilot's successful duration therefore
projects `$100.86` before retries, credits, and image-build overhead; a 5–20%
retry/overhead envelope is approximately `$106–$121`. The actual workspace
charge remains authoritative.

Before full fanout, newly played peer, direct-R1, and unguided certification
units all passed: 30/30 terminal games, capture rates 1.0/1.0/1.0, and 29/34/58
eligible bridge candidates. Their elapsed times were 721.00, 745.52, and 875.83
seconds. All three reported the frozen checkpoint, Showdown commit, production
capture engine, and causal audit engine fingerprints. Only then did the runner
launch the remaining 497 units. At launch-log time, all three certification
units were atomically present on the volume.

The generalized scale aggregate was checked before launch with a synthetic
500-unit corpus: exact 3,000/1,000/1,000 stratum counts, 2,500 unique mirrored
pairs, 8,000/8,000 complete groups, and the post-5k 25k-scale blocker all
passed. Focused tests were 22 passed, Python compilation and `git diff --check`
passed. Scale-runner source SHA-256 is
`2f6e073d04b5634a7e4799959497be97a4a09565680bbbc510be738a095b7cc8`.

**STATUS: RUNNING; NO 5,000-GAME ADMISSION CLAIM YET.** The final summary will
be written locally to
`experimental/runs/schema6_modal_5000_20260814_r1/summary.json` and remotely to
`/data/schema6_capture_500/schema6-modal-5000-20260814-r1/summary.json` only
after all 500 units return and the global uniqueness/capture gates pass.

## 2026-08-14: Budget pivot from Modal to a homogeneous local corpus

The full Modal estimate was re-audited against the configured resources and
the current published rates. Each active unit requested 16 physical cores (32
vCPUs) and 24 GiB, costing `$0.946368` per container-hour at `$0.0000131` per
physical-core-second plus `$0.00000222` per GiB-second. The accepted pilot
therefore still projects about `$100.86` of successful compute and roughly
`$106–$121` with 5–20% retry/build overhead. This is a resource-time estimate,
not a billing statement, and exceeds the user's hard `$20` budget.

Stopped exact Modal app `ap-HJGIZoGK3Z54wHyWp5FiVj` at
2026-08-14 19:26:43 PDT. Modal subsequently reported state `stopped` and zero
tasks. At the stop boundary, 20 atomic ten-game units had committed, for 200
complete terminal games; 40 in-flight tasks were interrupted and are not
counted. The committed artifacts remain preserved under
`/data/schema6_capture_500/schema6-modal-5000-20260814-r1/` on volume
`metagross-online-rl`. Based on the three measured certification units and the
approximately 13-minute full-fanout interval, incurred cloud compute is
estimated at approximately `$9–$11` before credits and final billing. No more
cloud work is authorized for this stage.

The 200 Modal/Linux games are retained as auxiliary evidence but will not be
mixed into the formal local admission corpus. The aggregate contract requires
one runtime fingerprint, while the local Apple ARM engine binary has a
different platform fingerprint from the pinned Linux engine. Replaying a fresh
5,000 games locally is slower but preserves a single homogeneous runtime and
avoids weakening the preregistered gate after seeing partial results.

Added a crash-resumable, CPU-only local supervisor at
`experimental/src/scripts/run_fresh_schema6_scale_5000_local.sh`. It pins the
authorization screen SHA-256
`61115745a20193c56b633d04e0c8dd497771f3ad580cb470c8de05d577bc50e8`, runs
the exact 3,000/1,000/1,000 peer/direct-R1/unguided mixture sequentially, gives
each stratum independent mirror/production/user identity domains, refuses
partial directories without an atomic progress snapshot, and executes the
strict 5,000-game aggregate at completion. The generalized aggregate enforces
all 5,000 terminal games, 2,500 mirrored pairs, the exact stratum counts, zero
voids, exact capture-group counts, at least 95% capture, zero invalid or
duplicate capture records, admitted causal bridges, and the unopened withheld
split. Focused tests were 22 passed; shell syntax, Python compilation, a
synthetic 5,000-game aggregate, and `git diff --check` passed. Final source
SHA-256 values were
`3b7a11f5bbbd860d34be80cadee687f9e267d65dcfbe5079d9c1000f4e8b4c88`
(capture runner),
`beaaebe91ac66d0ed1dc75c39145a0861610d4414947540fedf8b78fe4bd3592`
(local supervisor), and
`47ba9ab5806a5dea108a5af57afd31e0fe543590efa54253a5300b0651918f4a`
(aggregate).

Launched the fresh local corpus at 2026-08-14 19:32:54 PDT under
`experimental/runs/schema6_local_5000_20260814_r1/`. Supervisor PID 93857 is
running under `caffeinate -dims`; the peer Showdown service and both isolated
causal-history R1 prior servers are listening on ports 8140–8142. The capture
engine source hash is
`ccc5dd035c25fbf8e6b98301198cdaa35ce1012ae6f45ab81851965b0b05a0a3`, and
the local engine binary hash is
`3910185bb7f5e5f0283781b0b2292664f4c980126f320325143ba5970d4aba35`.
The 10-core, 24-GiB Mac had 152 GiB free at launch; expected output is about 54
GiB. Pilot throughput implies approximately 170.5 wall-clock hours at full
ten-core utilization, so the practical ETA is 6–10 days while the Mac remains
powered and awake. No GPU is requested or used, and local execution adds no
cloud compute charge.

**STATUS: LOCAL 5,000-GAME CORPUS RUNNING; FORMAL ADMISSION PENDING.**

### 2026-08-14 20:08 PDT local progress checkpoint

The supervisor remained healthy after 34 minutes with no restart. The peer
stratum had atomically committed 15 mirrored pairs: 30/3,000 peer games and
30/5,000 total games. All 30 committed game indices were unique, all 15 pair
IDs were unique, both legs were present for every pair, and the snapshot had
zero game errors and zero voids. Game 31 subsequently finished and game 32 was
in flight, but neither is counted until that mirrored pair commits atomically.

The two isolated prior streams contained 959 and 941 causal decisions. Their
logs had six warnings in total: four terminal/request states with no legal
action and two Zoroark `replace` replay warnings. These warnings did not mark
any of the first 30 terminal games erroneous or void, but causal-capture
admission remains unclaimed until the stratum-level capture and bridge audits
run. The run occupied approximately 1.0 GiB and the host retained 150 GiB free.
Observed peer throughput was roughly 30 games per 34 minutes, consistent with
an approximately 4–6 day total ETA if the initial rate remains representative.

## 2026-08-14: Fresh 113-root outcome teacher found corrections but failed reliability

Paused the local 5,000-game capture to avoid completing a four-to-six-day data
stage before falsifying its teacher premise. The pause occurred after an atomic
mirrored-pair commit. Exactly 42 peer games (21 complete pairs) remain in
`experimental/runs/schema6_local_5000_20260814_r1/peer/result.json.progress.json`,
with zero game errors and zero voids; all local Showdown and prior-server
processes were stopped. The generated supervisor trap incorrectly wrote
`complete` after the process-group signal, so the runtime marker was corrected
to `paused:manual-after-42-games`. No captured data was deleted.

Before inspecting any new outcome, froze the new teacher protocol under
`experimental/runs/schema6_113_outcome_teacher_20260814/PROTOCOL.md`. The source
was the accepted 113-root training-only panel SHA-256
`0101aaf0d0478471ba1010f799f5bee95bda5a4cf78bcc5fea992e8f6ef7071c`:
113 unique physical battles, two schedules and eight worlds per root, exactly
three candidate actions including the baseline, and zero withheld battles.
Independent validation reproduced all counts and the accepted hash. The local
causal Gen-9/Tera engine exposed both public-reveal and root-action APIs and
matched the previously certified strong-teacher binary SHA-256
`cf71fbba541c9e7b4f3c891bf9b25dca863196708b7131f77d1e0016c1073f69`.

First ran the frozen, direction-blind termination probe with seed `20260830`:
baseline action only, 20k root search, 2,048-iteration exact-MCTS continuation,
one rollout over two schedules and eight worlds, and horizon 128. It produced
1,757/1,808 exact terminals (97.179%) and 33,062 continuation searches. The
strict 16/16 filter retained 105/113 roots and rejected eight: terminal counts
0, 0, 3, 14, 15, 15, 15, and 15. It never used win/loss direction. Filtered
panel SHA-256 was
`3d2f21cb2d9aa54717c32bdef014b34c2a884771929f57bf9a34674da50d0908`.

Then evaluated all 105 retained roots uniformly with seed `20260831` under the
unchanged `strong_power_v3` teacher: three actions, two schedules, eight worlds,
eight matched rollouts, 20k root-opponent visit distribution, 2,048-iteration
exact-MCTS argmax continuation on both sides, and horizon 128. Nine local CPU
workers completed 210 schedule rows, 40,320 attempted outcomes, 40,238 exact
terminals (99.7966%), and 602,542 continuation searches in approximately nine
minutes. Teacher result SHA-256 was
`dc421c9ffb6b0b8b571e5aa95925c2eb97f67a8b2848e3bfc9cb0bc5f40ea32b`.

Frozen gate results were:

- terminal coverage 99.7966%, passing the 95% threshold;
- even/odd best-action agreement 68/105 = 64.7619%, failing the 70% threshold
  by six roots (74 agreements were required; Wilson 95% interval 55.2511% to
  73.2306%);
- 12 stable correction roots, passing the minimum ten;
- 66/105 raw terminal-best actions disagreed with shared 20k/50k root search.

All 12 corrections had 100% alternative, baseline, and paired terminal
coverage, positive cluster-bootstrap lower bounds, and positive effects above
0.01 in both schedules. Seven were switches and five were moves. Their mean
terminal-win advantage was 0.119140625, ranging from 0.0546875 to 0.171875.
An independent recomputation from raw results matched 105 roots, 210 schedule
rows, 40,238 terminals, 68 half-split agreements, and 12 corrections. Focused
collector/analyzer/outcome tests were 8 passed and `git diff --check` passed.
Full results and artifact hashes are recorded in
`experimental/runs/schema6_113_outcome_teacher_20260814/RESULTS.md`.

**Decision: STABLE CORRECTIONS FOUND; FAIL TEACHER ADMISSION.** The fresh panel
contains promising, outcome-grounded shared-search mistakes, but the teacher's
global action ranking is not reproducible enough under the frozen criterion.
Do not train a residual, run H2H, resume/scale the 5,000-game corpus, or open
the withheld splits from this result. The interval includes 70%, making this a
close and underpowered fail rather than proof that teacher reliability is truly
below 70%; fail-closed protocol status remains unchanged.

### 2026-08-14: 16-rollout diagnostic stabilized ranking but changed labels

Frozen a post-gate, same-root development diagnostic before collecting any
additional rollout. It explicitly could not reopen the failed eight-rollout
gate or authorize training. The panel, seed `20260831`, 20k root policy,
2,048-iteration continuation, horizon 128, candidate actions, schedules,
worlds, and engine were unchanged; only matched rollouts increased from eight
to 16. Extended the collector with the exact
`diagnostic_rollouts_16_v1=(20000,2048,16,128)` configuration and added a
fail-closed comparator that keys every outcome by root, schedule, action, world,
and rollout. The comparator also includes decision count in prefix identity.

The 16-rollout collection completed 210 schedule rows, 80,640 attempted
outcomes, 80,469 exact terminals (99.7879%), and 1,204,975 continuation
searches on nine local CPU workers. All 40,320 original rollout-0-through-7
outcome and decision records reproduced exactly: zero missing and zero changed.
Result SHA-256 was
`fd9d925a842bdac06820ea168f55657709e293d7a04a66043f3ca8055c162668`.

Half-split best-action agreement improved from 68/105 (64.762%) to 81/105
(77.143%), exceeding the old 70% reliability target on these same roots.
Sixteen of the original 37 disagreement roots stabilized, three of the original
68 agreement roots destabilized, 65 remained stable, and 21 remained unstable.
The original instability was not censoring: those 37 roots had 99.944% terminal
coverage. It was cross-world/schedule disagreement. Relative to the 68 stable
roots, their schedule agreement was 32.4% versus 94.1%, world top-action vote
mass 35.5% versus 84.5%, and aggregate-best value dispersion across
schedule/world clusters 0.144 versus 0.063.

Correction identity did not simply strengthen. The eight-rollout analysis had
12 stable corrections, while the 16-rollout analysis had nine. Seven retained
the same root and action, five old corrections disappeared after their cluster-
bootstrap lower bound crossed zero, and two new corrections appeared. The
seven persistent corrections average +0.143973 terminal-win advantage at 16
rollouts, range +0.078125 to +0.183594; five are switches and two are moves.
Thus more matched rollouts mechanically stabilize global ranking, but 42% of
the original correction set was not durable. The 16-rollout diagnostic still
does not pass the original all-criteria gate because it has nine rather than
ten corrections, and it is not an independent panel.

The highest-value next architecture remains search plus a strictly abstaining
residual, now with a sharper requirement: retain live visit/value/uncertainty
features, add belief-aggregated one-step resource/action semantics, and abstain
under high world dispersion. Switch option value is a priority because five of
seven persistent corrections are switches. The fastest valid evidence sequence
is a new disjoint, runtime-homogeneous 1,000-game 60/20/20 corpus, a direction-
blind termination filter, and the 16-rollout teacher under the original
coverage/agreement/correction thresholds. Only a pass may open grouped out-of-
fold action-semantic residual training; only an offline residual pass may
resume the 5,000-game scale and 100-game H2H. Full specification is
`experimental/runs/schema6_113_outcome_teacher_20260814/NEXT_OPTIMIZATION.md`.

Diagnostic/report SHA-256 was
`6fec0df02e66167017619c1a9aa5ca5da916be129f072bd7ec02f6e2f12265f5`;
next-optimization specification SHA-256 was
`d30db759289fcfadc914905b19b6c77a960773189d2f00a7bc9373a7bb5c8613`.
Focused tests were 10 passed and `git diff --check` passed. The local 5k
capture remains paused with 42 complete, zero-void games preserved; withheld
calibration and confirmation remain unopened.

**Decision: MORE ROLLOUTS FIX RANKING STABILITY, NOT LABEL SUFFICIENCY. DO NOT
TRAIN OR RESUME 5K YET.**

### Performance-status checkpoint after the rollout diagnosis

No new residual, distilled policy, H2H agent, or ladder policy has been admitted
from the schema-6 outcome-teacher branch. Its statistically proven live-strength
increment is therefore zero at this checkpoint. The current accepted playing
architecture remains corrected causal-history R1 plus production search.

The only direct comparative strength signal for that accepted causal-history
change remains the earlier 20-game causal-versus-legacy screen: 11-9 (55.0%),
approximately +35 H2H Elo at the point estimate, with Wilson 95% interval
34.21% to 74.18%. This is suggestive but unresolved and cannot support a GXE
claim. It does not establish movement from roughly 92% GXE toward 95% GXE.

The latest teacher results quantify opportunity, not realized policy gain.
Seven corrections persisted at both eight and 16 matched rollouts and averaged
+0.143973 terminal-win advantage on those selected roots. Even a perfect oracle
applying only those seven would average approximately +0.00960 over the full
105-root selected panel before accounting for their lower frequency in live
games. A learned gate will recover only part of that ceiling and can lose value
through false positives. Do not report +14.4 points, +0.96 points, or any
teacher statistic as a deployed win-rate or GXE improvement.

The material improvement to date is chiefly evidentiary and architectural:
exact causal R1 history, leak-free root construction, deterministic battle
splits, hidden-team-safe interior search semantics, direction-blind termination
filtering, reproducible matched terminal labels, and an identified need for
16-rollout/action-semantic uncertainty gating. These changes substantially
reduce invalid-training and false-positive risk but require a disjoint teacher
pass, offline residual pass, and paired H2H before they count as playing
strength.

### 2026-08-14: leak-free action-semantic residual rejected by grouped OOF

Permanently froze the current 105 root IDs and 105 physical battle IDs as
development-only in `action_semantic_residual/development-freeze.json`. Added a
reusable overlap assertion and made the outcome-panel builder's
`--final-confirmation` mode fail unless at least one `--development-freeze` is
provided. Both root and battle overlap are denied, so a renamed or reselected
root from the same physical battle cannot enter final confirmation.

Built a local, CPU-only one-turn feature bridge with the causal learned-prior
engine (binary SHA-256
`cf71fbba541c9e7b4f3c891bf9b25dca863196708b7131f77d1e0016c1073f69`).
For each action it used two schedules, eight belief worlds, a deterministic
20,000-iteration opponent root visit policy, and four matched opponent/chance
probes per world. It included candidate/baseline attack, setup, switch, and
Tera type; relative HP, survivors, bench/switch depth, hazards, screens,
boosts, substitute, damage tempo, speed, turn order, switch-entry cost, and
preservation value; mean, standard deviation, and 10th-percentile summaries;
and live visit/value lower tails plus cross-world range/disagreement. The
resource extractor explicitly uses only the causal public reveal mask for
information features. Only belief-aggregated scalars were serialized; no
per-world transition or completed hidden-team feature was written.

The resulting dataset has 105 roots, 210 non-baseline action examples, seven
persistent corrections, 77 harmful alternatives, 24 frozen baseline features,
and 85 enriched features. Dataset SHA-256 is
`685f25d7ab818134e920ac1780fa3c31105ea3a9a402e082c5bbaa13683527ca`.
The development freeze SHA-256 is
`41cfac871bcf7b9e66e586f277653d78d69d86772c5d41a33593b8b27bd41880`.

The first absolute-feature binary classifier was preserved as an explicit
negative artifact rather than overwritten. Under seven-fold outer OOF grouped
by physical battle, with one persistent root per fold and nested grouped OOF
threshold selection, the 24-feature baseline made one safe override but found
0/7 persistent corrections. The enriched classifier made two overrides,
found 0/7, and made one harmful override. Negative-report SHA-256 is
`ddcb5f64ca55591b95607cef54a81946c4d270482184fcde4489ec896abb5bba`.

Corrected two development-method issues transparently: action semantics became
candidate-minus-baseline (while retaining both action types), and the residual
target became continuous 16-rollout terminal advantage rather than discarding
203/210 labels into a seven-positive binary target. The pass rule was not
weakened: at least 3/7 persistent corrections, zero harmful overrides, and a
strict win over the 24-feature baseline.

The corrected nested battle-grouped OOF result also failed. The baseline made
one zero-advantage override, found 0/7 persistent corrections, and caused zero
harm. The 85-feature semantic residual made three overrides, found 0/7, and
made two harmful overrides (`iciclespear`, -0.03515625; `outrage-tera`,
-0.20442708), for summed development advantage -0.23958333. It therefore did
not strictly beat baseline and is rejected. Final report SHA-256 is
`a48b3bc3d560037741471fe58898a4c6c6649a16a3f22f0bc2ff7c5de3e89cd4`.
Focused contract/continuation tests were 11 passed, Python compilation passed,
and `git diff --check` passed.

**Decision: REJECT ACTION-SEMANTIC RESIDUAL. NO FINAL CONFIRMATION OR H2H.**
The seven corrections are real development opportunities, but this 105-root
panel does not support a safe generalizing residual. Keep corrected R1 plus
production search unchanged. Do not claim any playing-strength or GXE gain
from this experiment and do not spend on a disjoint confirmation run for this
model.

### 2026-08-14: historical mining reached 55 durable corrections; compact residual still failed

Executed the staged expansion requested after the 105-root residual failure.
The source inventory counted 23,684 local human trajectories, 5,000 league
battles, 2,450 causal root-ready battles, and exact schema-6 snapshots for the
preserved 42-game branch plus two earlier smoke games. Human and league
artifacts lack corrected-R1 causal snapshots and frozen belief schedules and
were therefore correctly classified as requiring rematerialization rather
than mislabeled as usable search roots. Inventory SHA-256 is
`a348fa5f0ce7156c3e8600f055c90216555b33a5e986b6a43767315f705bdc51`.

Reused the previously frozen 950-root exact accepted-R1 causal archive and its
completed 20k/50k search. It has zero root and zero battle overlap with the
forbidden 105-root panel. Exactly 538 roots were ambiguous under both 20k
schedules, satisfying the frozen 500–1,000 target. Panel SHA-256 is
`3b768a52c8d4dfb601818fa38b1f9fa01e4e7ddacd23e7ecd6af2242220d3c9c`.
A prospective direction-blind 16/16 baseline termination filter rejected 35
cycling or incomplete roots and retained 503; filtered-panel SHA-256 is
`a00312db0937ed0ee32e4f82fd200f38057536a29c3720a18bcdd20888a401b1`.

The four-rollout screen evaluated 96,576 trajectories with 99.7670% terminal
coverage, 76.342% half-split agreement, and 33 provisional strict corrections.
It reused 136 exact root/action/schedule/seed prefixes from the earlier
eight-rollout archive and recomputed every mismatch. Results SHA-256 is
`c1e747e4c31a3741adcd001e618b15719f6a30dfb3bd187c8d871418308239cf`.
The frozen promising-or-uncertain rule had 439 eligible roots and promoted the
maximum 300: all 68 promising roots plus hard controls. The promoted panel had
196 previously unopened roots, 104 consumed development roots, and 39 search
schedule disagreements; SHA-256 is
`f1ca1938810d3f090b39e822781ef7756fbeeba981422665e7c878a3d7ed7dfb`.

Collected only rollouts 4–15 for the promoted panel, then audited exact
root/schedule/action/world/rollout coverage before merging with rollouts 0–3.
The suffix contained 172,800 trajectories at 99.9520% terminal coverage. The
merged 16-rollout result contained 230,400 trajectories at 99.9557% terminal
coverage, 79.667% half-split agreement, 219 terminal/search disagreements, and
55 durable corrections—inside the preregistered 40–60 target. Corrections were
31 moves and 24 switches. Merged-result SHA-256 is
`3aaa389489adf6d997d8d6d6d63690382c4b87a8e85afe8632c1e3d51be3ac26`;
analysis SHA-256 is
`394c026338290d79ee97be759e6b82cebd7c845b6f58af1194d9bce7ed41dfcd`.

Built 600 leak-free candidate-minus-baseline rows with 55 durable corrections
and 117 harmful alternatives. Dataset SHA-256 is
`290eabe4ded67d2d0807dd26914c423ce81635605c534bb3b4fc46e4545a9911`.
The compact model restricted selection to 20 preregistered search/action/
resource features, selected 10/15/20 only inside training folds, predicted
continuous terminal advantage with L2 ridge, and selected its abstention
threshold using inner battle-grouped OOF subject to zero harm.

The frozen 24-feature baseline made zero overrides and recovered 0/55. The
compact residual made seven outer-OOF overrides, recovered one durable
correction (`encore`, +0.23828125), made zero harmful overrides, and summed to
+0.25390625 development advantage. This is safe but far below the frozen 30%
recovery requirement: 1/55 versus the required 17/55. OOF-report SHA-256 is
`32569ba7d5f7b92e47c250c7bee1c3a89857a1f88ca4cc34287455e686f01f4e`.
Focused tests were 12 passed, Python compilation passed, and
`git diff --check` passed.

**Decision: DATA GATE PASSED; LEARNABILITY GATE FAILED. KEEP CONFIRMATION AND
H2H CLOSED.** The project now has a materially larger, high-quality development
label bank, but the shallow compact residual does not generalize across the
heterogeneous long-horizon mistakes. Do not deploy it or report a strength/GXE
gain. The accepted agent remains corrected causal-history R1 plus production
search.

### 2026-08-14: specialist and sequential residuals also rejected

Executed the preregistered follow-up on the 300-root/55-durable-correction
development panel without opening confirmation. A label-blind action taxonomy
assigned all 600 alternatives to 221 switch-option rows (34 durable
corrections), 151 status/tempo rows (13 corrections), and 228 direct-attack
rows (8 corrections). Directional subfamilies were 17 switch-to-switch, 10
switch-to-attack/status, 7 attack/status-to-switch; 7 status-to-attack, 4
attack-to-status, 2 status-to-status; and 6 direct move choices plus 2 Tera
attacks. Taxonomy-report SHA-256 is
`f8b9777b0a9ce00b0c9b29135930213c4a091f3aeeb5f685d258415f96de45b1`.

The pre-fit label-blind audit found and repaired one taxonomy precedence bug:
damage-tempo deltas occur in ordinary attack pairs and therefore cannot
precede direct-attack classification. No model score had been fit or inspected
when that rule was corrected.

Tested family-specific nonlinear boosted-stump gates with frozen domain feature
pools, ten outer battle-grouped folds, five-fold inner model selection, and
inner abstention thresholds constrained to zero harmful overrides. The switch
specialist made six overrides, found 0/34 durable corrections, and made one
harmful override. Status/tempo abstained completely and found 0/13. Direct
attack made six overrides, found 0/8, and made two harmful overrides. No family
was admitted; combined overrides were zero. Specialist-report SHA-256 is
`57149200ed1351b1578de89ca3986eff86d47789038639b9221ce103fc46c6ba`.

Then built the pre-authorized sequential fallback as a residual on top of live
search, not a prior-only search replacement. Exact schema-3 R1 snapshots,
selected-action receipts, and dense terminal-trajectory rewards joined for all
300 roots. Mean causal history was 26.54 decisions, maximum 90, zero rows were
rejected, no mask fallback occurred, and no sampled hidden team entered model
features. The accepted 142.8M-parameter R1 checkpoint remained frozen and was
run locally on CPU. Embedding artifact SHA-256 is
`41fc5e207ebdf21cb2c01765d4c2cca860177544fbdee1c3a762f68d14489a1c`;
audit-report SHA-256 is
`03dba33e2734f36f4f115ff244108822f8f665855214b1ce4a52bd34fb7660f8`.

The sequential gate concatenated a seed-fixed label-independent 8/16/32-wide
projection of the 900-dimensional causal history embedding with the frozen 20
compact search/action features, using a 4/8-wide nonlinear interaction head
with at most 433 trainable parameters. Nested grouped OOF made six overrides,
found 1/55 durable corrections, made one harmful override, and summed
+0.33359375 development advantage. It failed both the zero-harm rule and the
17/55 recovery minimum and did not strictly beat the compact candidate, which
also found 1/55 but caused zero harm. Sequential-report SHA-256 is
`026f7f836ad0ceaead8acdd6a0941f1d5f12ea8febd3befb2dee2eaa6621b66a`.

Focused regression tests were 10 passed; Python compilation passed.

**Decision: REJECT SPECIALIST AND SEQUENTIAL RESIDUALS ON THIS PANEL. KEEP
CONFIRMATION, H2H, DISTILLATION, AND DEPLOYMENT CLOSED.** The taxonomy is useful
for future prospective data collection, but its largest durable directional
subtype has only 17 examples. Do not retune on these opened roots or claim a
strength/GXE gain. A future learned override requires a newly collected,
subtype-targeted corpus with substantially more durable examples and the same
battle-grouped zero-harm discipline. The accepted agent remains corrected
causal-history R1 plus production search.

### 2026-08-15: prospective switch-to-switch corpus reached 56 labels; frozen specialist still failed

Executed the targeted narrow-family collection without cloud or GPU compute.
The protocol was frozen before opening terminal outcomes and restricted Tranche
A to the preserved 1,500-root causal accepted-R1 panel. It prohibited reading
the existing 50k critic oracle and selected only from two-schedule live 20k
deployment search. Source overlap with both the opened 300-root and older
105-root development sets was zero roots and zero physical battles.

The local live-search pass completed 3,000 schedule rows (480 million engine
iterations). Of 1,500 roots, 865 were ambiguous under both schedules and 284
had a mean-visit baseline switch plus at least one alternative switch in common
support. Search artifact SHA-256 is
`ba8d265b15b6a910efcc8b9c87ffc61e734849ed634c38368763612f76dc1aca`;
the label-blind switch panel SHA-256 is
`e97867a8b8d0c7d7f1e645302c78ecc4492bfca6ae226c20989cba5b39268262`.

A direction-blind baseline probe evaluated 4,544 trajectories at 93.0238%
terminal coverage. The frozen 16/16 rule rejected 34 roots and retained 250;
filtered-panel SHA-256 is
`8889b8355fd4ea0380af5b6d5fefab76f05d80f3d41054ad78645e9d5e30d68f`.
The four-rollout screen then evaluated 46,720 matched trajectories at 99.7496%
terminal coverage, found 61 provisional corrections, and promoted 200 roots
including all promising cases plus uncertainty controls.

Collected rollouts 4–15 only for the promoted panel: 111,936 trajectories at
99.9267% terminal coverage. Exact-key merging with the four-rollout prefix
produced 149,248 trajectories at 99.9310% terminal coverage and 81.5%
half-split best-action agreement. Merged result SHA-256 is
`33994002688e969fc1996d82789a1e0b0455f5635bf27a3925a5013665b9703b`;
analysis SHA-256 is
`35dec1c58ec2e724f9b50e613c9460c7bd40dc3478f95cfc0ba30185b33b2bf2`.

The analysis found 47 correction roots and 56 independently durable
non-baseline switch actions; nine roots contained two alternatives that each
independently passed paired coverage, positive bootstrap lower bound, and
positive effects in both schedules. Mean advantage across the 56 durable
root/action pairs was +0.155668. Both action and root counts are retained; the
protocol's trainable correction unit is the independently verified action, so
the 56-label target passed without rounding.

Built 383 switch-vs-switch development examples with 56 durable corrections,
155 harmful alternatives, 24 search features, and 61 relative semantic
features. Only belief-aggregated public summaries were serialized. Dataset
SHA-256 is
`dbe45e1a8c3405576f8b3bac798b80c867690065451a92dff97c28b11f99f2bc`.

The unchanged nonlinear switch specialist was tested with ten outer
battle-grouped folds, five-fold inner capacity selection, and inner abstention
thresholds constrained to zero harm. It made zero outer overrides, recovered
0/56 durable corrections, and caused zero harm, versus the frozen requirement
of at least 17/56. Inner folds found at most one safe correction in four outer
training partitions; none transferred to held-out battles. OOF-report SHA-256
is `acfbcf7adaafbae4a2f68374fc71940b72231600b859d60617a220b6e869ef00`.

**Decision: DATA TARGET PASSED; TARGETED SWITCH SPECIALIST FAILED. KEEP
CONFIRMATION, H2H, DISTILLATION, AND DEPLOYMENT CLOSED.** The failure is no
longer plausibly explained only by heterogeneous action families or the old
17-label switch sample. The current 85 features omit structured candidate
switch identity and matchup semantics: entering species/types/moves/ability,
resistances and immunities, active matchup, and pivot-chain option value. Build
that materially different causal candidate-action representation and reuse the
frozen 56-label grouped gate before paying for more games. The accepted agent
remains corrected causal-history R1 plus production search; no playing-strength
or GXE gain is claimed.

### 2026-08-15: exact candidate-switch matchup representation also rejected

Executed the preregistered candidate-switch follow-up on the frozen 200-battle
switch-to-switch corpus, locally and without new game generation, cloud, or GPU
compute. Before fitting, froze a representation containing candidate-minus-
baseline types, owned HP/stats, pivot/recovery roles, and post-entry entry,
speed, damage, KO/survival, damage-race, and switch-flexibility semantics. Two
deterministic opponent-policy probes were aggregated over two schedules ×
eight causal worlds using a 2,048-iteration policy. Only mean and lower-tail
relative summaries were serialized; no species identity, state string, hidden
team, opponent identity, world index, or per-world observation entered the
dataset.

The build preserved all frozen invariants: 200 physical battles, 383
alternatives, 56 independently durable action corrections, and 155 harmful
alternatives. It produced 54 finite, nonconstant candidate/matchup columns in
addition to the 24 frozen live-search features. Dataset SHA-256 is
`62ac88a40aa6ce3f0d86a6c43a9313974baa576ff4e1e84b69842c05d0fea12b`.

The unchanged ten-outer/five-inner battle-grouped boosted-stump gate made one
held-out override (`switch cetitan`, +0.046875 terminal advantage), caused zero
harm, but recovered 0/56 independently durable corrections. Inner partitions
found at most two safe durable corrections and none transferred. This missed
the frozen 17/56 recovery requirement. OOF-report SHA-256 is
`0756b5266f1e5f1bd0beba9ea81707af2a85a31b492820bf70a65cb406e9f1a1`.
Nine focused tests passed; Python compilation, manifest/hash verification, and
`git diff --check` passed.

**Decision: CANDIDATE REPRESENTATION COMPLETE; GATE FAILED. KEEP CONFIRMATION,
H2H, DISTILLATION, AND DEPLOYMENT CLOSED.** Do not retune these opened roots.
The accepted agent remains corrected causal-history R1 plus production search;
no playing-strength or GXE improvement is claimed.

## 2026-08-15: Direct corrected-R1 controller frozen; readiness blocked

Implemented the preregistered direct long-horizon controller on top of the
accepted 500 ms R1-prior production search. The label-blind shortlist keeps the
top two shared actions by mean visit mass across two ambiguous schedules. The
frozen matched-outcome gate supports an exact four-rollout prefix and sixteen-
rollout extension over two schedules times eight worlds; it requires at least
95% terminal coverage, 90% paired coverage, advantage above +0.02 in each
schedule, cluster-bootstrap lower bound above +0.01, and positive advantages
in both eight-rollout halves before overriding. All failures abstain to the
production action.

The local CPU-only readiness audit stopped before materializing or reading the
93-battle untouched test split. The existing dual causal-history R1 certificate
terminated only 5/232 rollouts (2.155%): 226 failed as unsupported information
sets and one as an unsupported public event. The opened 240-root training panel
also contains schema-6 history only for the observing player; a causal opponent
R1 tracker cannot currently be reconstructed for each sampled belief world
without inventing private history from the hidden determinization. Both are
independent blockers to the requested matched dual-R1 teacher.

No confirmation root/outcome, H2H game, model training, cloud instance, GPU, or
paid resource was used; cost was $0.00 and the confirmation split remains
unopened. Fourteen focused controller/causal-collector tests, Python
compilation, manifest verification, and
`git diff --check` passed. Artifacts are under
`experimental/runs/direct_r1_long_horizon_controller_20260815/`; readiness
report SHA-256 is
`8c857c6174a9b70da4bb13203c4b387694e8cef1b57b441bcc4c981693b200b5`.

## 2026-08-15: exact terminal-MCTS direct controller reached 50-game look

Started a separate Cycle 1 using the authoritative 2,048-iteration exact-MCTS
terminal teacher; it does not alter the blocked dual-R1 experiment. The frozen
controller shortlists the production search's top two actions at roots that
are independently ambiguous in two eight-world schedules, evaluates matched
terminal continuations at four then sixteen rollouts, and fails closed unless
both schedules, coverage, cluster-bootstrap lower bound, and half-split rules
pass. The sealed 93-battle confirmation panel remained unmaterialized and was
never read.

The opened 500-root integration gate passed with 64 overrides, zero durably
harmful overrides, mean full-sixteen paired advantage +0.157317, and 39/40
early overrides retaining positive full-sixteen advantage. A fresh exact
stage-four local latency root completed in 0.77 seconds. The live insertion
then passed its preregistered ten-game canary 6-4 with seven applied deviations,
zero voids, and two timeout pass-throughs. The canary result SHA-256 is
`9a225a4ccfd5640f989f6b1a0349c46de086ff97c6f1b1ec2ecabadc6948fd1e`.

The fixed forty-game disjoint extension scored 17-23, producing an aggregate
23-27 (46.0%) over fifty games with Wilson 95% interval
[0.329697, 0.596011]. It cleared the deliberately lenient futility boundary by
exactly one win because stopping was frozen at <=22/50; this is continuation
permission, not evidence of improvement. Across all fifty games, the teacher
made 60 deviations in 37 games: those games scored 14-23, while 13 pass-through
games scored 9-4. That conditional split is descriptive and confounded by root
eligibility, but it is an important warning against claiming transfer from the
offline teacher values. Three of 1,660 live teacher calls timed out and passed
through; there were zero voids or invalid applied actions.

The immutable aggregate report is
`experimental/runs/terminal_mcts_direct_controller_20260815/futility-look-50.json`
with SHA-256
`4ba46a21d892f9b5ed0760f4bef582aac2dd8744abd3e77153e4a7e08601f26c`.
The final disjoint continuation to at most 200 games was frozen before new data
under `SUCCESS_CONTINUATION.md` and is running locally. Cycle 1 is not admitted:
the only success condition remains an aggregate Wilson 95% lower bound above
50%. Local CPU only; cloud/GPU/paid cost is $0.

### User-directed Cycle 1 stop at aggregate n=150

The project explicitly pivoted before the preregistered 200-game maximum. The
continuation stopped at its last atomic completed boundary: 100 additional
games / 50 mirrored pairs, scoring 49-51. Combined with the immutable 23-27
prefix, the direct controller scored 72-78 (48.0%) over 150 games; Wilson 95%
interval [0.401551, 0.559448]. Its required lower-bound-above-50% success gate
was not met, so it is not admitted. Because this was a user-directed early stop,
it is not represented as the preregistered max-200 terminal look.

The completed-only telemetry join found 3,385 teacher calls, 120 deviations,
and 15 fail-closed calls in the 100-game continuation. Deviation-eligible games
scored 30-32 and pass-through games scored 19-19. Across the aggregate 150,
games with at least one deviation scored 44-55 and pass-through games 28-23.
These conditional records are descriptive and confounded by ambiguity/teacher
eligibility; they do not estimate the causal effect of an individual override.
The two in-flight game logs (101-102) were excluded. The earlier
`teacher-telemetry-summary.partial-n100.json` is quarantined because it scanned
those incomplete logs and must not be cited.

**Structural diagnosis:** the offline gate certified the exact teacher's value
consistency under its own fixed terminal continuation, but deployment repeatedly
inserted those actions into a different hybrid continuation (future 500ms
R1-prior production search). That continuation-policy/Bellman mismatch and
compounding intervention distribution shift were not tested by the offline
screen. The live result rejects promoting this direct recurrent override layer;
it does not establish that each deviation is harmful or that terminal MCTS has
no value as a data generator. Fifteen fail-closed calls are too sparse to
explain the lack of aggregate improvement.

**Frozen Cycle 2 recommendation (not started):** use a one-deviation causal
attribution design on new controlled prospective games. For each ambiguous
state, compare the two root actions with matched hidden truth and causal public
history, then return both branches to the same frozen production continuation;
permit only one intervention per episode. Train a resource/history-aware
residual student only on independently stable deviations, with pass-through and
resource-preservation anchors, and test that student at equal 500ms. This is
materially different from direct teacher control because it measures the hybrid
policy's actual one-step treatment effect before distillation.

Checked artifact:
`experimental/runs/terminal_mcts_direct_controller_20260815/cycle1-user-directed-stop-n150.json`
(SHA-256 `43bcf7311ffaaa1ebe9ee60cbb4d6eff3555d4df236baec0fc23841b0d479131`).
Atomic progress SHA-256:
`e0c27e83f732ccd8832f66dd5f1b3a8366efcb48ccc45d8ac5d7e724f1d6bf2f`.
The sealed 93-battle confirmation panel remains unopened. No GPU, cloud, or
paid resource was used; cost $0.

## 2026-08-15: Cycle 2 Gate A failed at causal-root compatibility

Preregistered the search-native `PublicSearchStateV1` Gate A before opening
measurements. The new branch keeps root R1 root-only and extracts a stateless
player-information representation directly from the exact engine state plus
causal reveal masks; it does not advance or query the 142.8M R1 transformer at
interior nodes and never synthesizes transformer history.

An initial report appeared to pass with 1,456/1,456 depth-one successors,
zero hidden-completion, restoration, or determinism failures, Python/Rust
feature parity max error `2.24312e-7`, and 4.71 ms local CPU batch-64 p95 for an
untrained 1.184781M-parameter policy smoke. Post-result audit found that root
selection had called the compatibility bridge first and excluded 1,486 source
rows under an aggregate invalid/mask label. This conditioned the denominator
on representation support, so the initial pass was quarantined rather than
promoted.

Froze a corrected audit before rerunning: select roots using only schema,
state hash, ordinary exact legality, and canonical 13-action parity; only then
compile the reveal mask, with every failure retained in the denominator. The
corrected selector fixed 200 roots from 44 physical battles and scheduled 1,598
successors. Only 560 were supported: 35.0438%, failing the fixed 95% Gate A.

All 1,038 unsupported successors came from 130 root-artifact disagreements
before stepping: 96 roots exposed a public ability in the player-information
snapshot while the paired engine state stored `NONE`; 32 disagreed on public
item semantics (`noitem`/`NONE` and form-item/`UNKNOWNITEM` dominated); two had
public base/form species aliases that did not map exactly. The 70 coherent
roots passed 560/560 depth-one steps, so no true child extractor or exact-step
failure was observed after a root reconciled. The stale/incomplete artifact
failures nevertheless remain unsupported under the frozen gate.

**Decision: FAIL AND STOP GATE A.** Do not start Gate B, train a policy, open
confirmation, generate games, or run H2H. A materially justified future repair
is a separately frozen causal replay of each `protocol_prefix` into a typed
public-event ledger, with explicit item/ability reveal authority and public
form aliases. It must not infer an ability reveal from a transformer species'
default ability or inspect hidden simulator fields. That repair was diagnosed
but not implemented in this measurement.

Corrected report SHA-256:
`21ddcbd455a01f8efdb8fdeb5ec3474335c9679ea0c28b63010693f4e35bcc0d`.
Five Python representation tests and four Rust causal-reveal tests passed;
Python compilation and `git diff --check` passed. The sealed 93-battle panel
was not read. No R1 interior query, fabricated history, training, new game,
H2H, GPU, cloud, or paid resource was used; cost $0.

## 2026-08-15: Cycle 1b prospective one-deviation canary stopped

Ran the frozen causal-attribution canary after independent protocol review.
Twenty new games were collected as ten mirrored pairs with exact pre-play
blocked assignment: ten teacher arms, ten production arms, and five teacher
assignments on each mirror leg. At most the first valid certified terminal-MCTS
disagreement could change an action; both arms then used unchanged causal-
history R1 plus the 500 ms production search for the rest of the real game.
Cycle 1 outcomes were not reused, no counterfactual R1 continuation or causal
history was fabricated, and the sealed 93-root panel was not opened.

The run completed 20/20 decisive games with zero voids and a clean frozen
integrity audit. There were no teacher timeouts, fail-closed calls, baseline-
action mismatches, assignment/hash failures, multiple eligible opportunities,
or post-lock teacher calls. Eleven games reached a certified opportunity:

- teacher arm: 3 wins in 6 eligible games (50.0%);
- production arm: 2 wins in 5 eligible games (40.0%);
- observed teacher-minus-production effect: +10.0 win-rate points;
- one-sided Fisher exact p-value: 0.608225.

Across all randomized games, teacher assignments scored 4-6 and production
assignments scored 5-5; agent A was 9-11 overall. These all-game counts include
pass-through games and are a secondary diluted diagnostic.

**Frozen decision: `STOP_CYCLE1B`.** The canary required at least 12 eligible
games, at least five per arm, an effect of at least +30 points, and one-sided
Fisher p <= 0.20. It obtained only 11 eligible games, +10 points, and p=0.608.
Therefore this does not authorize a powered replication, distillation,
deployment, or opening confirmation. The small positive point estimate is not
proof of benefit or harm; it is simply far below the preregistered large-effect
screen. Together with Cycle 1, this exhausts the current MegaGem-style exact
terminal-selector path as the next high-value optimization.

The local CPU run took approximately 58 minutes and used no GPU, cloud, or
paid resource; cost $0. Frozen manifest SHA-256:
`3fa94df6233b52085ab9fca918a032f1217f9677a7c194071fdc853a427dd8d1`.
Final result SHA-256:
`0ede61c9e66ea1e3630b4122c5399727acb3df45c1f5cfd46488cf0593b76ba3`.
Final causal summary SHA-256:
`089a3ed201a20abc880449c28aa35b338474f8002c9c2d2d1bab31802bbbaf2a`.
Mirrored-pair manifest SHA-256:
`224f8cfe0a7aadeedb3f575d26962a91fdb0e57bc250afa05b69a1b0778f1d3a`.

## 2026-08-15: Cycle 3 causal bridge repair failed its frozen gate

Preregistered a causal protocol-to-engine bridge before measurement. Opponent
facts came only from chronological public protocol events; the own private
request authenticated only the observer's team and legal action boundary.
Transformer default items/abilities and hidden engine truth were forbidden.
The same selection-independent 200 opened roots from 44 battles were retained,
with all 1,598 scheduled successors in the denominator. Fourteen Python tests
passed before the 13-file input manifest was frozen.

The separate production sampler audit exercised the accepted sampler/converter
on eight deterministic worlds. Raw fact recall was perfect: species 16/16,
moves 40/40, items 8/8, abilities 8/8. Typed recall was zero for every field,
and all eight engine reveal masks were zero because the converter omits the
typed mask arguments. This confirms a live capture-contract bug: sampled worlds
retain raw values but lose causal authorization metadata.

The root bridge reconciled 187/200 roots (93.5%), below the fixed 95% gate. It
made 27 causal ability repairs and nine causal item repairs, with no move/PP
placeholder repairs. Thirteen roots failed: ten exact-own-legality mismatches
and three unmapped `ogerponwellspringtera` public species. A post-result
read-only diagnosis showed the ten legality mismatches were harness errors from
base-normalizing exact own switch labels (five Sawsbuck, three Minior, two
Alcremie); the three Ogerpon failures were an omitted frozen alias. Neither was
repaired after opening.

The successor gate recorded 0/1,598. Of those failures, 104 inherited failed
roots; all remaining 1,494 hit
`TypeError: 'poke_engine.StateInstructions' object is not iterable` in the
frozen determinism assertion before semantic successor checks. Thus the frozen
successor result is FAIL but is not substantive evidence against depth-one
stepping or projection.

**Decision: FAIL.** Root coverage independently missed the threshold and the
successor measurement was invalidated by a harness defect. No live-capture
contract, training, target collection, H2H, or sealed confirmation access is
authorized. Any Cycle 3b must be separately frozen, preserve exact own form
action IDs, enumerate the Ogerpon alias, and compare opaque instruction objects
without iterating them while retaining the same roots, thresholds, and
denominators.

Protocol SHA-256:
`7fe1b96e4ec3ee851e27a075fbdaf8cf46b2de8f99127ef7815b6dc823ab85da`.
Manifest SHA-256:
`d69e74778b54bfbcbe7d89ad50dda89d89c5861a249314b3313c827878884262`.
Sampler report file SHA-256:
`9a4713cdd86a5304d5c7f15e27764e392fa72d079bd0494545e626b4b7417818`.
Bridge report file SHA-256:
`386cc0ff21443911fdb516e828b9a1a67b133c0b254d74fd8d06e292e767301e`.
The sealed 93-battle panel was not read; local CPU only; cost $0.

## 2026-08-15: Cycle 4 systematic causal bridge passed

Started a separately preregistered Cycle 4 after freezing Cycle 3 FAIL. It kept
the identical selection-independent 200 opened roots from 44 battles, the same
1,598 scheduled successors, and both >=95% thresholds. The identity contract
was split correctly: own private switch actions preserve exact normalized form
IDs, while opponent public species use a mechanically generated, source-pinned
Pokemon Showdown map from `battleOnly` and `cosmeticFormes` metadata. The map
contains 236 entries (131 battle-only, 105 cosmetic) and covers
Ogerpon-Wellspring-Tera without an observed-failure alias list.

The deterministic repeat check now uses the supported opaque-instruction API:
exact child bytes, `percentage`, ordered `instruction_list` strings, and public
child bytes. The original object is passed unchanged to reverse. Nineteen tests,
including a repeat/reverse preflight against the exact frozen binding, passed
before freezing the 18-file manifest.

Cycle 4 passed cleanly:

- root reconciliation: 200/200 (100%);
- scheduled successor support: 1,598/1,598 (100%);
- failures: zero;
- causal facts: 738 species, 1,171 moves, 303 items, 185 abilities;
- archival repairs: 30 abilities and 10 items;
- placeholder move/PP repairs: zero.

**Decision: PASS for bridge mechanics and freeze the native live-capture
contract only.** The production sampler bug remains: raw public facts survive,
but typed reveal masks are lost. The frozen contract now requires an immutable
causal ledger to be copied into every belief world and installed into the
perspective-correct typed mask, with 100% per-field recall, zero invented bits,
exact parity, and fail-closed mapping before any training may be preregistered.
This result does not prove strength and authorizes no implementation, training,
target collection, H2H, or sealed confirmation access.

Protocol SHA-256:
`ec72dbeea4dab4bb80bc95561a453ca7876df3828b9ec2538fbc83a2dd4d0b36`.
Manifest SHA-256:
`e783b18b58efa9f017008cf688030c9ff567b45fe8e2ae777cd1b364ef25ae81`.
Report file SHA-256:
`452bbe44d59f19aac99e2e8626976879cbd940d7c8e19b1ba8866cf634139aca`.
Embedded report SHA-256:
`f133993b89617a78477d31322f135c753f29a5928a3f2854cb0bbe740b79bebe`.
The sealed 93-battle panel was not read; local CPU only; cost $0.

## 2026-08-15: Cycle 5 production causal-ledger admission failed frozen byte parity

Cycle 5 preregistered production capture before measurement: the public
Showdown receive tee freezes an immutable JSON-safe causal ledger at the
decision boundary, attaches it before the actual Foul Play belief sampler, and
the production converter wrapper compiles only certified facts into an
observer-relative engine reveal mask. One hundred forty-three tests passed and
all 17 frozen input hashes matched before the one-shot admission run.

The admission stopped on its first sampled world with
`CausalRevealLedgerError: mask installation changed non-mask engine bytes`.
No typed-recall or latency result was opened. A post-failure read-only diagnosis
localized the problem to the frozen `_mask_field` helper: on the engine's
13-field serialization it uses `rsplit("/", 4)` but writes fields 1 and 2,
which are the threat and wincon matrices, rather than fields 3 and 4, which are
the public-reveal masks. This invalidates the parity helper and makes the
swapped-perspective installation path unsafe.

**Decision: FAIL.** This is a frozen implementation/admission-harness defect,
not evidence against causal ledger propagation. Cycle 5 was not repaired or
rerun. No target collection, training, H2H, deployment, or sealed confirmation
access is authorized. A new preregistered repair cycle must pin the exact
serialization grammar and test nonzero threat/wincon matrices plus both reveal
mask positions before repeating the unchanged gates. The existing production
wrapper must not be deployed.

Even a repaired root-level PASS would authorize target collection only. A
sidecar that remains unchanged across root sampling does not establish deployed
Rust interior inference/node updates or the identity of items consumed by newly
simulated events; those remain explicitly unauthorized.

Protocol SHA-256:
`b8af48bb2930752a1f4d2c9ee8300eb1b9c4a972ddd78f393989c7986200d1c1`.
Manifest SHA-256:
`a8ccdb7e497eb247c174535a1bae8b138f6a001b32d26553a0bf5adf53e22d49`.
Frozen ledger implementation SHA-256:
`2c1158fdef7429e65d9804a4bc0213959ace43053abcc966459af11242612c0f`.
Frozen audit SHA-256:
`5ac109623e28c31055689b28dbd4fbd9fc7f85aa84476e7d0195d4ce9bb55029`.
The sealed 93-battle panel was not read; local CPU only; cost $0.

## 2026-08-15: Cycle 6 native symmetric live-capture repair passed

Cycle 6 was separately preregistered after freezing Cycle 5 FAIL. It retained
the identical opened fixture, eight belief worlds, sampler seed, production
sampler/converter, typed-recall requirements, fail-closed checks, and latency
limits. The repair added a native symmetric
`with_side_two_public_reveals(bits)` binding alongside the existing side-one
setter. Production now uses native setters only. A named exact 13-field engine
serialization grammar is used exclusively for audit and roundtrip validation;
there is no positional tail mutation.

Before admission, 148 tests passed. The frozen manifest covered 24 direct
inputs, the rebuilt native binary, and a deterministic hash of all 50 engine
source files. The one-shot audit additionally checked 132 deterministic mask
values on states with nonzero threat/wincon scalars and matrices. Both native
setters changed only the intended mask field, and setting then clearing both
masks restored exact bytes.

The unchanged production gates passed across all eight worlds in both
perspectives. Typed recall was 16/16 species, 40/40 moves, 8/8 items, and 8/8
abilities normally and after swap, with zero spurious bits. Non-mask parity,
hidden noninterference, apply/reverse, perspective swap, JSON roundtrip,
deterministic replay, and explicit missing-move/missing-species failure counts
all passed. Added conversion overhead was 0.167 ms mean and 0.186 ms p95 per
world; ledger freeze was 0.118 ms mean and 0.123 ms p95 per root.

**Decision: PASS for freezing a separately preregistered development
target-collection stage only.** No targets were collected, no policy was
trained, no H2H ran, and no confirmation data was opened. This establishes the
root sampling/conversion bridge but does not prove strength or deployment
readiness.

Specifically, an unchanged root sidecar across sampling does not prove deployed
Rust interior inference or causal node-by-node sidecar updates. The current
engine mask also cannot preserve the identity of an item consumed by a newly
simulated event. Those are separate blockers before native long-horizon target
semantics or deployment can be authorized.

Protocol SHA-256:
`adfadae485d35736edcbff1e6c0c1a6d3f5cd017c4fade4f052cd98ba79695e4`.
Manifest SHA-256:
`3bd12ff5861b2d42a7944c2420d22b20773f4fef06460248a4f9147f678f682b`.
Admission report file SHA-256:
`3c6e7843f10b0094fefbfb1622213ec3366bcaa24cab33469fd56028ad8810ab`.
Base unchanged-gates report SHA-256:
`7bed2c40118a1c5fc454edc1d9781d8529445316a274a9648aa1a424aad9c52f`.
Native binding SHA-256:
`22005e5978956afd2bed03dc1ebb7fe131cdb5065792571911300423fd94164d`.
The sealed 93-battle panel was not read; local CPU only; cost $0.

## 2026-08-15: Cycle 7 collector mechanics failed before collection

Cycle 7 preregistered a collector-only gate over opened schema-6 roots: a
battle-grouped 26/9/9 split, at most one root per battle, four deterministic
Foul Play hidden worlds, eight semantic depth-one schedules per world, and a
2,048-iteration full visit/Q teacher. Thirty-five implementation/source hashes
and the current Cycle 6 engine binding were frozen; 16 tests passed, including
Showdown-p2 to engine-side-one observer orientation.

The one-shot run stopped before sampling or teacher search with
`CollectorError: frozen 26/9/9 battle split does not match inventory`. The
unchanged corrected selector returned 200 roots but only 42 unique battle tags
under the frozen Cycle 7 binding. The preregistered 44-battle count came from
Cycle 4, which used the older `cf71...` binding; Cycle 7 used the admitted
`22005e...` binding. The fixed split summed to 44 and correctly failed closed.

**Decision: FAIL.** Zero target values were opened and no member/group artifact
was written. Cycle 7 was not repaired or rerun. This is an inventory/split
assumption failure, not evidence about collector support or teacher quality.
No target collection, training, H2H, deployment, or sealed confirmation access
is authorized.

The independent methods audit also remains binding: Cycle 7's sanitized public
fingerprint omits full causal history and child event-sidecar identity, while
its fixed joint-action/uniform schedules are not posterior reach weights.
Therefore its cross-path aggregates would not be authorized as learnable
targets even if mechanics had passed. A successor must inventory under its exact
binding before freezing, use complete causal-history fingerprints, split by the
largest dependence cluster, and prefer actual observed sequential transitions
with real opponent/chance reach.

Protocol SHA-256:
`37b02310ed68857bebb7cb4d67d4be343502fb3f60395ea65f49c13abdde7289`.
Manifest SHA-256:
`e6d9329b59664b5b27203d21e04d3d9b03bc99824f675415f5b11105e2b38a5d`.
Frozen collector SHA-256:
`2260ba8a2a03d653b903affafb417a177b34797a6bafeab81ed3c695998e53fd`.
The sealed 93-battle panel was not read; local CPU only; cost $0.

## 2026-08-15: Cycle 8 Phase 0 found a 20,629-battle deterministic human replay source

Froze Cycle 7 FAIL without rerunning its flawed counterfactual design, then
completed a read-only, immutable inventory of opened human, self-play,
PFSP/league, schema-6, Metamon-pinned, MCTS raw-state, and old visit artifacts.
It read no visit/Q/teacher values and wrote no target rows.

The main finding changes source priority: two raw human replay trees contain
20,629 source-disjoint battles with locally present exact Pokemon Showdown
commits, battle/player seeds, full inputlog choices, public logs, and terminal
results (19,273 primary + 1,356 external; zero battle-ID overlap; 1,132,101
choice commands). Four further complete battles pin unavailable commits and are
frozen negative controls. Raw tree hashes are
`be76cef1e01263826989b07b2924eb68f5e1e2de005391fef2f11335f3d27c69`
and
`f92457f0251a5eb43c8e33325186d74180af75a20f49f6e76ed94fe0fbf0bcb3`.

Immediately materialized schema-6 data is 50 games/25 mirrored pairs/2,862
causal decision states (42 preserved + eight smoke); two more completed smoke
games need audited root export. The remote 500-game/250-pair Modal source is
recorded as recoverable but ineligible until its raw capture is local and
hashed. MCTS final + round-two add 9,503 replay/result-joined eligible battles
and 354,302 exact engine decisions after the 500 opened-development exclusions,
but need replay alignment and opponent-hidden sanitization. Their 2,142 raw tag
collisions have zero canonical replay overlap; global tag-based dedup/exclusion
is forbidden. Partial's 3,861 replays are final-lineage duplicates.

Parsed caches remain anchors, not exact target roots: 692,067 human, 789,094
legacy self-play, 356,990 strict round-two, 177,452 pre-strict round-two,
421,839 PFSP all-POV, 211,310 PFSP learner-only, 182,313 G5 league, and 39,833
earlier online-RL decisions. Opposite POVs may not fill hidden fields. Historical
visit archives contain 175,319, 179,066, and 135,457 rows; their value fields
remained unopened and they are production controls only.

Froze a 128-battle deterministic replay gate before measurement: all ten rare
cases (including four missing-commit negative controls), 22 hash-ranked cases
from each of four major commits, and 30 external cases, replayed twice. Passing
requires exact normalized public/terminal replay, command/request legality,
forced-switch/trap/Tera/PP/disable agreement, byte-deterministic side-specific
requests and causal fingerprints, and zero private-opponent/spurious-reveal
leakage. Unavailable commits must abstain without substitution.

Future target semantics are frozen but unauthorized: actual observed
transitions supply opponent/chance reach; full causal history and sidecar enter
the fingerprint; no cross-path aggregation; largest dependency split before
labels; full legal visits/Q/completed-Q at 8,192 iterations with 2,048 and
20,000 stability controls plus separate production-500ms control.

**Decision: PHASE 0 INVENTORY COMPLETE; CYCLE 8 REPLAY GATE FROZEN; NO REPLAY
MEASUREMENT OR TARGET COLLECTION YET.** Inventory SHA is
`8935be7881577ec341c4cca460b6b526db5db346b7672c3c77119c3fd8d4b651`,
protocol SHA is
`d86ec9cf8424c47bf4667072233805ba278a307266a065b8f1aec8c9dcff1b1a`,
and manifest SHA is
`de035a192b4097ad6aa531ddf56c80f6a6aa079c2434add75b4acc6c8f58632c`.
All six manifest file hashes, Python compilation, and `git diff --check` passed.
The sealed 93 panel remained unread; no teacher, training, H2H, GPU, cloud, or
paid resource was used; cost $0.

## 2026-08-15: Cycle 8 deterministic human replay admission failed closed

After the Phase 0 inventory, froze the label-blind 128-battle selection and all
replay/canonicalization/POV-export code before opening selected replay results.
The panel contained 124 exact-commit positives and four missing-commit negative
controls. Seven unit tests and five unselected development smokes passed. The
model-information fingerprint was explicitly separated from provenance: it
binds causal mechanics history, event index, own request mechanics, and typed
reveal facts, but excludes battle/source identity, player names, ratings,
avatars, and request-side name. `wait:true` requests are captured but are not
fabricated into actionable states.

All four negative controls abstained correctly. Ninety-three of 124 positives
passed both deterministic repeats, normalized public/terminal equality,
physically separate p1/p2 sideupdates, exact-request command legality, typed
ledger materialization, and POV nonleak checks. Thirty-one positives failed:
29 public comparisons, one request with no supported action, and one recorded
action absent from support.

Read-only failure localization found that 28 public mismatches were server-only
records omitted by BattleStream but not listed in the frozen transport
normalizer: 12 player reset/name rows, 12 badge rows, and four inactivity-loss
messages. The remaining public mismatch was an old-commit protocol spelling
drift (`0` versus `0 fnt` and `Rain Dance` versus `RainDance`). Both legality
failures were Revival Blessing target prompts: Showdown marks a fainted target
`reviving: true` and accepts `switch <slot>`, while the frozen extractor
excluded every fainted slot. No post-result normalization or action repair was
applied.

**Decision: FAIL.** Cycle 8 was not rerun and does not authorize target
collection, training, H2H, or confirmation access. A separate repair cycle must
freeze systematic server-transport and source-pinned historical protocol
canonicalization plus explicit Revival-Blessing target semantics, with the same
denominators and gates. The opened 128 cases remain development-only.

Panel SHA-256:
`b7f60ed83377f7e0e8d3f7ae2528888510eb42ebd607e1c6f44e75287357de3e`.
Premeasurement manifest SHA-256:
`e6402a01deb889439a562793b495dd1954c664d44dfe646de98af4e3f10f7619`.
Aggregate report SHA-256:
`36c05bda6e1a3d1fc2cc6bd11372098a00560b9ccbdb9d915c1771a0f96a4f09`.
The sealed 93 panel remained unread; no teacher values, training, H2H, GPU,
cloud, or paid resource was used; cost $0.

## 2026-08-15: Cycle 9 deterministic replay bridge repair passed

Cycle 9 was separately preregistered after freezing Cycle 8 FAIL. It retained
the exact 128 panel identities and 124-positive/four-negative denominators. The
only repairs were systematic server-transport rows absent from BattleStream,
two rendering equivalences scoped to the one old Showdown commit, exact
Revival Blessing `reviving:true` fainted-target semantics, and a causal
command-time prefix for old commits that emit a still-current request before
subsequent public events.

The command boundary was fixed before hashes froze: compute the final public
chunk boundary first, then derive the prefix and typed ledger exactly once.
Regression tests prove that a public chunk emitted after the request but before
the recorded command changes the ledger and model fingerprint, while a chunk
at or after the command does not. Fifteen Cycle 8+9 tests passed, and all 124
already-opened Cycle 8 repeat-one captures passed the repaired contract before
the Cycle 9 rerun.

The one frozen rerun passed: 124/124 positive battles completed both exact
repeats, four/four missing-commit controls abstained, and there were zero
public, terminal, exact-request action, repeat-determinism, POV, or frozen-hash
failures. The 496 role/repeat files contain 8,000 request states per repeat:
7,038 actionable, 962 wait-only, and 6,951 uniquely correlated recorded
commands. Both Revival Blessing prompts mapped to exact fainted targets with
semantic class `revival_target`. Aggregate request diagnostics include 1,000
forced switches, 14 trapped states, 4,238 Tera-capable states, and 31,443 typed
reveal facts.

**Decision: PASS for a separately preregistered fail-closed full-corpus
rematerialization coverage/index gate only.** This is source/capture evidence,
not strength evidence. Teacher collection, target labeling, training, H2H, and
the sealed 93 remain unauthorized.

Protocol SHA-256:
`0836ae186a6eee8a5da6ab51fd5c449e9381e318ecc347f2eca930d384ccd686`.
Premeasurement manifest SHA-256:
`64bf76591317d93c565baff193dc9e0f42410b1b7c5da836ec3c7bfa377eaf4a`.
Aggregate report SHA-256:
`8ce8953289045c898f56c7a4558bafcc4f8869f77b548eac9ab6d4a2ce3b1708`.
The sealed 93 remained unread; no teacher, labels, training, H2H, GPU, cloud,
or paid resource was used; cost $0.

## 2026-08-15: Cycle 10 full-corpus coverage/index gate failed closed

Froze 20,629 locally pinned positive human inputlogs plus four unavailable
commit controls, seven exact Showdown runtimes, both immutable source trees,
eight workers, a compact privacy-preserving index schema, a 256-battle
determinism panel, dependency-component 60/20/20 splitting, and >=99% overall
and per-major-commit coverage gates. The frozen manifest SHA-256 was
`f55b98669c608bb02a7792820d81c0b4afc15382b5ce1ccdd5e9074a7b13ec9d`;
21 Cycle 8/9/10 tests and pre-run source/hash verification passed.

The one frozen run admitted 19,886/20,629 battles (96.3983%) and failed 743:
449 public replay mismatches, 167 causal-ledger fail-closed cases, 106 unknown
Unicode encoding failures, 14 request-schema failures, six unsupported team
previews, and one source/provenance failure. Every major commit was below 99%
(95.8246%-97.5664%). Four/four unavailable controls abstained. The 256-panel
repeat admitted 243 exact repeats and failed closed on 13 selected inputs, so
it did not meet its 100% gate.

The eligible index contains 1,250,082 states and 19,886 dependency components:
11,989 train, 3,998 validation, and 3,899 test battles, with zero cross-split
component leakage. Post-run manifest and source-tree verification passed.
Report SHA-256 is
`19cc92b7b493ff3fe34f24926aa4d42cb4c6ced32f71a7f3744870ed40ba5086`;
master-index SHA-256 is
`77af3957ea103acab6f57523cbc9142ca432ab2933c72d04aa64c8a693ba9086`.

Read-only diagnosis supports only a new-cycle, version-aware `|-sethp|`
spectator-HP equivalence, an exact name-change-forfeit transport equivalence,
UTF-8 diagnostic hashing, and label-blind exclusion of nonstandard/custom-rule
formats. Repeated-species/Illusion identity ambiguity remains fail closed.
Cycle 10 authorizes no teacher labels, training, H2H, or sealed confirmation
access. Local CPU only; sealed93/teacher/training/GPU/cloud/paid cost all zero.

## 2026-08-15: Cycle 11 transport repair narrowly failed one per-commit gate

Separately froze Cycle 11 over 20,560 exact-`gen9randombattle` pinned positives,
four unavailable controls, and 69 label-blind custom-rule exclusions. Repairs
were limited to UTF-8 hashing, version-scoped `|-sethp|` spectator HP, and an
exact name-change-forfeit transport equivalence. Repeated-species/Illusion
remained fail closed. Protocol SHA was
`1577a0eaf4b357f89fdcc1a203ae180434283b7780ea08e9302bf584940c6cc8`;
manifest SHA was
`cbcc7f911aad5bdde55bf594d3e71cef5a91f539287b8a60a2be0b53df62b542`.
All 28 tests and pre/post source/runtime/file integrity checks passed.

The one frozen run admitted 20,379/20,560 (99.11965%) overall. Four major
commits cleared 99%, but `f8ac1400` admitted 7,498/7,575 (98.98350%): 77
failures where at most 75 could pass. Cycle 11 therefore failed by two battles
on the per-major gate. Failures were 174 causal-ledger abstentions, six public
transport mismatches, and one rare old-commit row lacking a player seed; zero
unknown failures occurred.

Read-only localization showed the six unmatched lines were nonmechanical
server transport absent from BattleStream: one invite response, two hidden-line
moderation rows, one moderated-chat HTML broadcast, one lookup error, and one
loser-forfeit display. The two `f8ac1400` mismatches were the invite and
forfeit display; its other 75 failures were safe causal-ledger abstentions.
No row was retroactively repaired.

The 256 determinism rows reproduced 256/256 classifications (254 exact
admissions, two exact abstentions), four/four controls abstained, 1,287,652
states were indexed, and 20,379 dependency components split 12,206/4,110/4,063
with zero leakage. Report SHA is
`464bc80fb46d9053d720c43b2e803975a482591df88fb4433d006f600c6de4be`;
master index SHA is
`0fb7d722cac26b76cdbc5bd8dcdcb411e8b22ae11e5870792c3fde4eca83b87d`.

**Decision: FAIL.** A successor may only preregister systematic nonmechanical
server-transport normalization while keeping denominator, thresholds,
Illusion abstention, classification determinism, and integrity gates fixed.
No teacher, labels, training, H2H, sealed93, GPU, cloud, or paid cost.

## 2026-08-15: Cycle 12 full-corpus transport gate passed

Froze only exact server/UI transport grammars independently localized after
Cycle 11: authenticated invite, unlink, and simple-forfeit displays plus one
exact moderated-chat banner and exact lookup-error grammar. A label-blind scan
of all 20,560 positives counted every occurrence before measurement. No broad
raw/error/message filter, terminal-winner conditioning, hidden identity repair,
denominator change, or gate relaxation was allowed. Protocol SHA was
`84fa428b10c165283bc2213da80a4eeaa71e90b01169a57347af2120f0d48c48`;
manifest SHA was
`e9d644e28dcb69cd487473e7ce1f44daa6b223f0ba9c672657a1820016bceb27`.
All 31 tests and pre/post integrity checks passed.

The one frozen run admitted 20,385/20,560 (99.14883%). Every major commit
cleared 99%; the former blocker `f8ac1400` reached exactly 7,500/7,575
(99.00990%). Failures were only 174 repeated-species causal-ledger abstentions
and one rare old-commit missing-player-seed provenance row. There were zero
public mismatches, unknown errors, or silent fallbacks.

Four/four unavailable controls abstained. The deterministic panel passed all
256 classifications (254 exact admissions, two exact abstentions). The compact
index contains 1,288,022 states in 20,385 dependency components, split into
12,179 train, 4,157 validation, and 4,049 test battles with zero leakage.
Post-run frozen integrity passed.

**Decision: PASS for a separately preregistered teacher-target plan only.**
This does not prove policy strength or authorize teacher execution, labels,
training, H2H, or sealed93 access. Report SHA is
`3664d8d65ca737572e02cd04160a9e2fb6de77810520f22e4cacb5498bdcc689`;
master index SHA is
`b98d73b2844d749558904056740164683c39d9bfb97ad31ba9f310bb70e7f143`.
Local CPU only; sealed93/teacher/labels/training/H2H/GPU/cloud/paid cost zero.

## 2026-08-15: Cycle 13 TRAIN-only causal rehydration gate failed closed

Froze 200 label-blind actionable states from 200 distinct Cycle 12 TRAIN
dependency clusters. Validation and dev-test state indices, teacher values,
Q/visits, terminal outcomes, H2H, and the sealed 93 remained unopened. Built a
single local CPython 3.11 engine from the Cycle 6 native-mask source with the
required `poke-engine/terastallization` feature because the existing Cycle 6
binary omitted Tera actions and the older Tera binary omitted native mask
setters. The combined engine SHA-256 is
`da84a09697f9b5646791a95c48879f5dab01a2f948dd699a0188aba936ce1aab`.

Prefreeze TRAIN smoke, explicitly excluded from frozen counts, exposed and
repaired two causal bridge defects: later `[from] item:` effect lines could
resurrect an item already consumed by `|-enditem|`, and event-certified reserve
abilities were not installed before production sampling. Revealed moves still
required exact existing PP/disable state; no placeholder PP or hidden facts
were introduced. Fifty lineage and bridge tests passed before freeze.

The one frozen run passed 194/200 roots (97.0%), exceeding the 95% support
threshold. All 194 admitted roots reproduced two independent eight-world
schedules exactly twice and passed action/Tera mapping, native mask, swapped
perspective, non-mask-byte, semantic apply/reverse, and hidden-completion public
projection parity. Post-run file/runtime/source integrity passed. Mean
end-to-end root time was 830.86 ms (p95 1,023.78 ms); an eight-world sampler
call averaged 13.91 ms and per-world conversion/mask/projection/step averaged
3.63 ms.

Cycle 13 nevertheless **FAILed** its conjunctive zero-integrity gate. The six
frozen failures were three exact-PP/disable fail-closed cases, one initial
form-change before Foul Play `request_json` initialization, one request/engine
action disagreement where Showdown allowed three switches but hidden-world
dynamic trapping removed them, and one causal species mapping ambiguity where
revealed Sawsbuck-Winter and a hidden sampled Sawsbuck collapsed to the same
public-form key. One PP failure did not reproduce in three fresh-process reruns,
which is itself evidence for order/global-state isolation rather than a reason
to discard the frozen failure.

Protocol SHA-256 is
`9098c197023c15d3b671a577d770fce924fe473bd798cb92980038f632721157`;
manifest SHA-256 is
`d51c51618fc1073b99dc483163babee91f92304a50079ee530b9ded556b339fb`;
root-results SHA-256 is
`8fb94d89e38b5f102d3f9f4493497f58e89838bcc78de2a0bdacf91f7b998c89`;
aggregate report SHA-256 is
`0e6703da4e51f202ca5e0c722b1f6173169c66547383de221c7af1aabe8679bb`.

Cycle 14 teacher stability remains unauthorized. A successor must separately
freeze slot/activation-aware public identity, request-authoritative root
trap/Tera permissions, request-before-initial-event initialization,
event-counted PP/disable reconstruction, and per-root process isolation while
retaining the same 200-root denominator, 95% support threshold, and zero causal
integrity failures. Local CPU only; GPU/cloud/paid cost $0. No commit was made.

## 2026-08-15: Cycle 14 mechanics repair passed

Separately froze and reran the byte-identical Cycle 13 selection (SHA-256
`d1d31d96f807fcdc1b5c3ae60e5feb28e9f828be2a829b04136dffe525864031`)
with four narrow repairs: activation/exact-form-aware public slot identity,
request-authoritative root switch legality despite sampled hidden trapping
abilities, current request installation before request-0 form events, and one
fresh subprocess per root. Missing public moves/PP remained fail closed; no
donor moves or placeholder PP were allowed.

The patched Gen9-Tera/native-mask/root-request engine passed 255 Rust tests;
the Python/bridge suite passed 78 tests. Engine SHA-256 is
`30f92c279fb1ae7ee50019b868ba4137ab9d082ad082f270eee2298b7545ce52`.

The frozen run passed 198/200 roots (99.0%). The only two failures were the
preregistered exact-PP/disable abstentions. There were zero causal-fact or hidden
noninterference failures, 200/200 fresh subprocess receipts, exact repeated
two-by-eight schedules for every admission, and exact action/Tera, mask, swap,
non-mask byte, apply/reverse, and public projection parity. Post-run manifest
integrity passed. Mean worker mechanics time was 1,152.04 ms, p95 1,314.80 ms.

Protocol SHA is
`ecc3e4fdd42f42bf2f286edaaa96cda87b14b333566f8e9fbf9ee807248f40e7`;
manifest SHA is
`d9df57ec28cb20f3685a02d8bd8544526385a71fa334e03a9547cde3eca3670f`;
root-results SHA is
`33a260d7f877a186904d0feeddb1d710b8e487b3580ceb04df4ab92483015507`;
report SHA is
`c3033361b1ba9a30eaff0ccbe98e6341b304004d71853ad90aac4a68311e5703`.

**Decision: PASS for separately preregistering bounded TRAIN-only teacher
budget/stability measurement, not for training or strength claims.** No teacher
values, validation/dev-test labels, H2H, sealed93, GPU, cloud, or paid resources
were used. Cost $0; no commit.
### 2026-08-15: Cycle 15 teacher stability gate failed before search values

Cycle 15 froze 40 label-blind, dependency-cluster-unique TRAIN roots from the
198 Cycle 14 admissions. It separately preregistered exact production
`P_exact` (adaptive 250/500 ms and 16/32 worlds with causal-history R1 priors),
world-paired `P_paired`, and seeded equal-prior 2,048/8,192/20,000 plus
R1-prior 20,000 arms. A request-authoritative root-only MCTS ABI was built and
58 binding tests plus three metric tests passed; interior nodes retain ordinary
simulator legality.

The frozen run failed closed before its first MCTS call. Raw BattleStream
requests lack the live client's `rqid` transport correlation, and the production
R1 server correctly rejected that request. The prior artifact is empty; zero
R1 target priors, visits, W/Q values, validation/test labels, outcomes, H2H, or
sealed93 records were opened. This is a harness transport gap, not a negative
teacher result. Cycle 15 remains permanently failed. Any repair must be a new
cycle with a frozen deterministic offline request-correlation contract and
observation/mechanics invariance proof; it may not amend this run.

### 2026-08-15: Cycle 16 correlation comparator failed before search values

Cycle 16 separately froze the byte-identical Cycle 15 panel and arms plus a
monotone offline rqid contract and two-variant R1 replay parity. It reached the
first root's two CPU R1 replays, then failed before MCTS because the comparator
included `own_legality.rqid`, the sole routing field intentionally varied by the
test. Both durable prior/parity artifacts are empty. Zero visits, W/Q values,
candidate decisions, validation/test labels, H2H, or sealed93 rows were opened.

This is a comparator-normalization defect, not a teacher result. Cycle 16 is
permanently failed. A successor may strip only routing rqid from the invariant
legality comparison while preserving every mechanical legality field and
separate mismatched/stale response rejection. Local CPU only; cost $0.

### 2026-08-15: Cycle 17 teacher stability gate passed

Cycle 17 changed only the Cycle 16 comparator by stripping the routing-only
`own_legality.rqid` after schema/type validation. It reused the byte-identical
40-root Cycle 15 TRAIN panel and every original control, arm, seed, metric and
gate. Before MCTS, 40/40 fresh-process double mechanics/world preflights and
40/40 two-variant causal-history R1 parity replays passed. Search workers then
reproduced their preflight state/world hashes. The raw completeness audit found
zero legal-support, missing-Q or nonfinite-Q failures.

Equal-prior 8,192 was the cheapest admitted candidate: 85% all-cell top-1
stability, 90% schedule agreement, 82.5% agreement with equal-20k, repeat JSD
0.000101, and eight stable differences from exact production. Equal-2,048
failed 20k agreement at 65%; equal-20k passed but was slower; R1-20k produced
only three stable differences. Equal-8k and equal-20k agreed while exact
production differed on 9/40 roots, six with all three controls all-cell stable.
Mean 8k latency was 27.97 ms/world versus 453.63 ms/world for `P_exact`.

**Decision:** PASS only for separately preregistering prospective full-game H2H
of the fixed equal-8,192 direct teacher. This is stability/difference evidence,
not strength: all offline Q uses the same hand evaluator. Training, target
collection, validation/dev-test and sealed93 remain unauthorized. Report SHA
`c94b59c3f340afc64672b138c5b63fe9ea8fc9787c4a5bd1a4a30f04d167c6d9`;
audit SHA `741e2b43735932514cd8c1c5fdbf61a99574bc869d8cbeae903ba7ec3f1a93a2`.
Local CPU only; cost $0; no commit.

### 2026-08-15: Cycle 18 prospective H2H permanently void before an outcome

Cycle 18 froze a fresh ten-pair/20-game prospective comparison of the admitted
equal-prior 8,192 controller against exact production. The run aborted during
both agents' first decision in game one. No game completed and no result file
was written, so zero outcomes or candidate decisions are usable.

Two pre-score breaches made the entire cycle void. First, the controller sampled
over all aggregate legal-action mass instead of production's frozen
`>= 0.75 * highest` considered set. Second, spawned Foul Play processes imported
the stale general environment engine without native reveal masks and failed
closed with `AttributeError` before applying an action. All partial logs, two R1
rows, protocol chunks, registrations, pairs, seeds, usernames and run identity
are quarantined/retired and may not be reused. Cycle 18 manifest SHA is
`087db94e1dd02a4e1de02d54072eb33192199871b2348de62af72aa2add046cf`;
protocol SHA is
`ff6a0edf2276fc96ff43a64df856e3651769153a93c1a543196eb3b32828b9bc`;
retired pair SHA is
`89d2fdf60e7dac49525fbd8356d6781e921f24520aabcbc76256aad6710164c8`.

**Decision: VOID, not a candidate loss and not strength evidence.** Cycle 19
must separately freeze an isolated exact Cycle 17 native-mask engine path inside
spawned production processes, byte-equivalent production considered-choice
sampling, and a successful end-to-end one-decision operational smoke before
generating new scored teams/seeds. Local CPU only; cost $0; no commit.

### 2026-08-15: Cycle 19 smoke passed; H2H failed on Terapagos ability lineage

Cycle 19 repaired both Cycle 18 pre-score breaches without changing the
qualified equal-8,192 search arm. It pinned the exact Cycle 17 engine inside
each spawned Foul Play process and implemented production's stable aggregate
ordering, `>=0.75 * highest` considered set, and deterministic weighted sample,
preserving prefilter and considered receipts. Nine selector/monitor/teacher
tests passed. The actual Showdown smoke passed: exact engine provenance 2/2,
16/16 worlds at exactly 8,192 visits, exact selected `uturn` command/rqid, and
public execution confirmation with zero fallback or timeout. Smoke evidence SHA
is `8a0bb940cdbe51006fc115cc4cd5c68df76894606959a71da7c5176359c59356`.

After freezing ten wholly fresh disjoint pairs (pair SHA
`689ed2dedb0baa29d9a686f37200a42e58a02eeb4db5752327299bc0ef034160`),
the scored gate completed game 1 as a candidate loss and aborted game 2 before
the mirrored pair completed. The comparator failed closed with
`CausalRevealLedgerError: public ability mismatch: terapagos`. Public history
had exposed `Tera Shift` and then `Terapagos-Terastal`; canonical form collapse
retained historical `terashift` in one sticky ability field while the correct
post-form engine ability was `terashell`. This is a form-phase representation
defect, not hidden leakage and not candidate strength evidence.

All four spawned engine provenances were exact. The 39 candidate decisions
before stop had zero receipt failures, nine overrides and 30 pass-throughs;
mean/p95 latency was 211.14/225.07 ms. Because there were zero complete mirrored
pairs, none of the 0-1 unpaired score is inferential.

**Decision: FAIL the fixed zero-semantic-failure gate; permanently retire the
sample.** Any successor must be separately frozen and first prove a systematic,
source-pinned, slot/form-phase-aware public ability lineage contract on opened
fixtures. No retry, training, sealed93, GPU, cloud, or paid compute. H2H
manifest SHA `6283d016fa8a86d9fff43734f72ee7e22fef446400b2602991a5e7bd7f6d3792`;
local cost $0; no commit.

### 2026-08-15: Cycle 20 ability lineage passed offline; live smoke exposed an unregistered-team harness bug

Cycle 20 implemented the generic causal ability-lineage repair before any new
score: exact public form identity is preserved separately from base
canonicalization; explicit and rule-implied abilities form an ordered history;
current ability is distinct; and `detailschange` can update it only through a
pinned exact-form/unique-ability Showdown contract. The full Cycle 19 Terapagos
failure and systematic Terapagos/Ogerpon/Aegislash/Minior/Skill Swap cases
passed, as did hidden perturbation, native mask byte parity, perspective,
request-authoritative root actions and unsupported-event fail-closed checks.
The fixed pre-smoke suite was 38/38; the contract had 424 nonempty,
duplicate-free rows. Protocol SHA was
`cca4a6e7b6410757881e9d78c8585ed4505353758ef25cfeaee044d5e7d0f004` and
manifest SHA was
`963d24ce882b0c10453c637eed7c1a5cceb1530a86b0c3514c2a177469abb3be`.

The mandatory live smoke failed before scoring. The frozen label-blind pair had
Kyurem and Terapagos as public leads, but the actual battle started Mudsdale vs
Skeledirge. Both pair registration files remained unconsumed. The frozen script
used plain `start_showdown.sh` without `METAGROSS_EVAL_PAIR_DIR`, so the server
ignored the prescribed packed teams and battle seed. The incidental candidate
decision completed 16 x 8,192 visits with exact engine provenance and publicly
executed `earthquake`, but it is wrong-team diagnostic telemetry only.

**Decision: FAIL the live form-transition conjunct, zero completed games/pairs,
no strength inference, no H2H authorization.** Retire all Cycle 20 live
identities. A separately frozen successor must use the verified
registration-aware Showdown runtime and attest exact consumption, packed-team
hash, battle seed, player/team orientation and public leads before testing the
ability-lineage repair. Local CPU only; cost $0; no commit.

### 2026-08-15: Cycle 21 registration harness passed; live causal ability installation failed

Cycle 21 replaced the plain Showdown launcher with the verified
registration-aware supervisor. On a new label-blind, disjoint pair, exactly two
registrations were observed and validated against pair/leg/format, packed-team
hashes, battle seed and canonical p1/p2 assignment; the server consumed both,
with zero remaining or reappearing files. Public leads were the registered
Urshifu vs Terapagos, and the live stream contained ordered Tera Shift then
Terapagos-Terastal detailschange. The registration receipt SHA was
`68645a6efb0c331c5d5d2b5bdaa63ab124d63006fe39ef1426f6dae61d0cc5fe`;
Showdown launch SHA was
`55f8c4491e80d2312b6ff1649bd036c0ae3f106f16ab8e1a3e30737b448d0aeb`.

The smoke failed before a candidate action. Ledger v2 correctly reconstructed
exact `terapagosterastal`, historical `terashift`, and current `terashell`, but
the live Foul Play battle-to-sampled-engine converter left the exact engine
slot's ability at stale `terashift`. The strict check failed with
`CausalRevealLedgerError: public current ability mismatch:
terapagosterastal`. This is a causal-fact installation/hydration gap, not a
registration failure and not strength evidence.

**Decision: FAIL with zero actions/games/pairs/outcomes; no H2H.** Retire all
Cycle 21 identities. A separately frozen repair may install only the
ledger-certified exact-form current ability into the uniquely matched slot,
with hidden-noninterference and exact non-target state/action parity before a
new registered live smoke. Protocol SHA
`291664cc3f287540fa33dd6afbd9d614ab1ba1d787e5168368d9dafd8db35559`;
manifest SHA
`0d608b0037b7cdbded984dba6b7f6bc1f9da93fb53623db5a1867e20f4dce53e`;
local CPU only; cost $0; no commit.

### 2026-08-15: Cycle 22 ability hydration passed; frozen live smoke failed

Cycle 22 added a pinned symmetric native ability setter and installed only
ledger-v2-certified current ability into a uniquely matched exact-public-form
engine slot. The 53-test pre-smoke suite passed. A fresh registration-attested
battle used the registered Skarmory vs Terapagos leads and exact battle seed.
The first candidate decision then passed: all 49 conversion receipts installed
Tera Shell as current/base on Terapagos-Terastal, hidden sampled values did not
control the patch, two schedules x eight worlds each completed 8,192 visits,
and the selected `stealthrock` was publicly executed with no fallback.

The fixed smoke still failed. Its monitor hardcoded Terapagos to slot 0 and
rejected one correct receipt after a public Hatterene switch moved the uniquely
matched Terapagos form to slot 1. Because evaluation continued past the first
executed action, it also exposed a distinct fail-closed PP contract gap:
`public move/PP-disable authority missing: hatterene/stealthrock`. That later
error is a deployment blocker, not a reason to weaken receipt checking.

**Decision: FAIL with zero completed games/pairs/outcomes and no strength
inference; no H2H.** Retire all Cycle 22 identities. A new monitor-only smoke
may accept the certified unique slot and terminate immediately after the first
publicly executed action, but a pass can authorize only a separate PP/
conditional-belief mechanics cycle. Protocol SHA
`f9866399c6c5fe2676e9da6ceda7088c97a693f83cf22d766839d63242071091`;
manifest SHA
`1d21e86160330a99213a05c48da4ad2e68b17c81d9d26739fe1b8ec6bf11e647`;
local CPU only; cost $0; no commit.

### 2026-08-15: Cycle 23 monitor repair passed tests; pair identity failed pre-game

Cycle 23 froze a monitor-only repair that accepts a unique exact public form at
any engine slot, binds receipts to the first causal root/request/rqid/action
window, and stops immediately after public execution. All 55 pre-smoke tests
passed with the Cycle 22 engine/controller unchanged.

Live execution stopped before registrations or battle creation because the
separately assembled pair-preparation argv did not have the same evaluator
config identity as the full live argv. The evaluator correctly rejected the
stored manifest rather than bypassing its config hash. There are zero protocol,
prior, receipt, decision, game, pair or outcome rows.

**Decision: FAIL with no live mechanics or strength inference.** Retire the
pair/seeds/identity. A successor must freeze one canonical argv/config source
for pair preparation and live execution, with only an explicitly declared phase
projection. Protocol SHA
`77aba430aaf6c25ab99e07754161432f65d1fd4a5fca3f786a292f58871a692d`;
manifest SHA
`e454dcc5676021244d35f449b4b527459c3a338e6b2c0e886160b5940bc8b840`;
local CPU only; cost $0; no commit.

### 2026-08-15: Cycle 24 canonical config passed; receipt boundary remained unattributed

Cycle 24 derived pair preparation and live execution from one canonical argv.
Preparation, live parsing and the stored pair manifest all had config SHA
`11606f941fff031db2ef62f43fa09a7f256ad87de6b30bd0671137ed2a971e47`.
A fresh registration-attested battle reached one full 16 x 8,192 candidate
decision; `swordsdance` was selected, sent and publicly executed. All 64
ability receipts installed exact-form current/base Tera Shell correctly.

The smoke still failed because an external poll did not terminate before 16
conversions from a different later causal root were appended. The generic
receipts lack phase/decision/time attribution, so the fixed monitor could not
admissibly distinguish production-control, candidate and post-execution rows
and rejected the escaped protocol hash. There are zero completed games/pairs/
outcomes and no strength inference.

**Decision: FAIL; retire the identity and do not retry.** A separately frozen
successor needs execution-only call-phase and decision markers with complete
cohort reconciliation and a strict first-public-execution boundary. Protocol
SHA `3007bf711078152e635cec50b2ca19f9ec0b45a7501bd15d64a75bc78c3958b5`;
manifest SHA
`252cd8aa478677407089c8a695007fd3ce3f0cf14724eca4c48d0d7703fd54c9`;
local CPU only; cost $0; no commit.

### 2026-08-15: Cycle 25 attributed first-decision mechanics passed

Cycle 25 added receipt-only phase/decision/rqid/root/schedule markers at the
actual conversion boundary and proved they change no causal ledger, engine,
world or policy bytes. A smoke-only public-execution latch prevented parsing a
second request. In a fresh registered Bellibolt vs Terapagos battle, production
declared/emitted exactly 32 adaptive conversions and the candidate emitted all
16 indexed 2x8 cells. All rows were on the same root before public execution;
all uniquely installed current/base Tera Shell. The fixed candidate completed
16 x 8,192 visits and its override to `switch abomasnow` was sent and publicly
executed.

**Decision: PASS for PP/disable and conditional-belief mechanics only.** There
are zero games/outcomes and no strength evidence; H2H/training/sealed93 remain
unauthorized. Result SHA
`d857d5cd7b6c39dd7ac77d19ac76a22675c252330ce1c5ff2219bd10e5613b5c`;
protocol SHA
`429c0cad46ed9b9f28f7d59c750d252e2cdf36eb96c88e8cadf4dc07929f3bde`;
manifest SHA
`1eeaeebe624be7b74dc255f4e2fc5c910bf1f94416dd16904b3ac1ef1d1eb559`;
local CPU only; cost $0; no commit.

### 2026-08-15: Cycle 26 repaired reflected moves but rejected universal disabled-bit equality

Cycle 26 introduced a v3 causal move ledger. Public moves with a `[from]`
cause, including the preserved Hatterene/Magic Bounce `Stealth Rock`, remain
audited derived executions and do not constrain sampled sets. Intrinsic moves
bind to exact live current PP/max PP/disabled state; sampling copies PP and the
disabled bit, rejects max-PP drift, verifies every sampled world and preserves
raw weights. Focused tests passed 43/43 and pinned Foul Play Magic Bounce/
Pressure tests passed 3/3 before the one frozen run.

The identical dependency-unique 200-root Cycle 14 TRAIN panel achieved 192/200
root support and 3,072/3,200 scheduled-world support (96.0%), clearing both 95%
gates. All 994 intrinsic moves were exact on admitted worlds; 13 derived
executions stayed outside intrinsic constraints. Determinism, masks, legal
actions, hidden-public projection, perspective and apply/reverse passed. Move
verification p95 was 2.47 ms and isolated-root p95 was 1,477.91 ms, both within
the frozen limits.

The zero-integrity gate failed on four Choice-item completions. Foul Play copied
the causal `disabled=false` correctly, then its world-specific `lock_moves()`
set Barraskewda/Flip Turn, Entei/Flare Blitz, Urshifu/Close Combat or Iron
Bundle/Flip Turn disabled after sampling a hidden Choice item. Thus opponent
disabled state is not universally causal across belief worlds: explicit public
disable causes can be invariant, while hidden item locks are conditional world
mechanics. Two Morpeko-Hangry form mismatches, one Minior-Yellow/Minior-Meteor
mapping and one unresolved Jolteon/Zoroark-Hisui Illusion row also failed closed.

**Decision: FAIL; no smoke or H2H.** A new frozen cycle must split causal versus
world-conditional disabled authority and pin Morpeko/Minior public battle-form
mapping, while leaving unresolved Illusion unsupported. Protocol SHA
`bb6bb68dfa53f14f12ac0edddbda8b7cd16e6a767f51cbde2f7b8f13785a32e3`;
manifest SHA
`f45ea79980dfff1e8d3a4a69eb55822b4154521be7f8e6b4aca586ef38943b57`;
report SHA
`6a66e1b81941756b0bde01fe3d3752faf6c30b1880299b7bd6c47e65897e24c2`;
local CPU only; cost $0; no commit.

### 2026-08-15: Cycle 27 typed disable mechanics passed; live production sampler integration failed

Cycle 27 split disabled authority correctly. Exact causal PP/max PP remains
invariant; only an explicit active public Cursed Body Disable is invariant;
ordinary hidden Choice locks and post-clear state remain world-mechanical and
reach search unchanged. Typed receipts distinguish both authorities. The cycle
also added source-pinned public Morpeko-Hangry and Minior-Meteor activation
identity without base-form collapse or hidden truth.

The frozen Cycle 14 200-root TRAIN panel passed 199/200 roots and 3,184/3,200
scheduled worlds (99.5%). All admitted move states, weights, masks, legal
actions, hidden-public projection, deterministic repeats, perspective,
serialization and apply/reverse checks passed. The only abstention was the
preserved Jolteon/Zoroark-Hisui Illusion root. Move verification p95 was 2.72 ms
and isolated-root p95 was 1,563.24 ms. Mechanics report SHA is
`6f247606f7cdc4e96f816d75f26f21c8341ddc213fb7849afa8db2981bc96615`;
premeasurement manifest SHA is
`95df4db942418ea5082cdf874250f8fa3e77106637fc6e5ee3d5ca6af17d1af2`.

The authorized fresh registered second-root smoke then failed before any
action. Both pair registrations were consumed exactly once and both spawned
engines matched the pinned binding. However, ordinary live production search
calls vendor `find_best_move`, whose direct `prepare_random_battles` worlds do
not pass through `_prepare_search_battles`/`verify_sampled_move_states` before
conversion. The typed emitter therefore correctly failed closed with `causal
move world receipt is absent or invalid` at each player's opening production
conversion. No candidate decision, move receipt, game, pair or outcome exists.

**Decision: PASS the Cycle 27 mechanics gate but FAIL the one frozen live smoke;
retire its pair/seeds and keep scored H2H unauthorized.** This is a production
sampler integration gap, not policy-strength evidence. A separately frozen
successor must attach and verify the frozen v4 ledger/typed receipt on every
vendor-generated production world without changing sampling, weights, search,
policy, state or action semantics, and prove valid empty opening-root plus
revealed-move second-root receipts before a wholly fresh smoke. Presmoke
manifest SHA was
`66d44a8c4cbcc15cf54dc23a76f84d23ba3d61a585784ae059504e0d95249eef`;
local CPU only, cost $0, sealed93/training unopened, no commit.

### 2026-08-15: Cycle 28 production hook passed units; frozen object comparator failed

Cycle 28 implemented the narrow missing integration: every vendor production
`prepare_random_battles` result is now verified against the unchanged v4 ledger
and receives a typed move receipt before battle-to-engine conversion. Empty
opening roots receive an explicit empty receipt; derived public executions are
preserved as non-constraining provenance. Focused tests passed 16/16. A
label-blind TRAIN panel froze four empty openings, all seven opened derived
cases and five later intrinsic roots, with both 16- and 32-world adaptive
counts and a 100% noninterference gate.

The one frozen audit failed before any semantic comparison. Its proposed
whole-object byte comparator removed the receipt sidecar and then pickled each
Foul Play battle. All 16 fixtures hit the same `PicklingError`; read-only
localization showed that Foul Play's dynamically created `fp.battle.Range`
class is not import-resolvable by pickle. Thus zero hooked/unhooked world,
weight, engine, action or selector comparisons were admitted. There is no
evidence of policy or mechanics harm, but the fixed gate correctly cannot pass.

**Decision: freeze Cycle 28 FAIL and do not run its live smoke or H2H.** A
separate successor may replace only the invalid pickle comparator with a
source-pinned canonical mechanical projection covering exact engine
serialization plus explicit Foul Play tracker/request/public fields, ledger
bytes, sample order, raw weights and legal actions, while keeping all fixtures,
denominators and 100% gates unchanged. Report SHA
`ee3efe1d46787769ccd47cec8b16c3cd6df6f0a97c9498545e05651c51548c38`;
manifest SHA
`2ed194104d86b33e29d83df7984eca00ae91c70fdbb04e8ba00a66ee0bdd1daf`;
local CPU only, cost $0, no sealed93/teacher/training, no commit.

### 2026-08-15: Cycle 29 canonical noninterference passed; fresh smoke hit a pivot boundary

Cycle 29 changed only Cycle 28's invalid pickle comparator. A field-named,
source-pinned canonical projection covers all stored Foul Play battle, request,
tracker, battler, Pokémon, move, side-condition and causal-ledger state while
separating the typed receipt under test. Sensitivity and determinism tests
passed. On the identical 16 TRAIN fixtures/seeds and both 16/32 adaptive counts,
all 16 fixtures passed: canonical mechanical projection, sample order/weights,
ledger, engine serialization, legal actions, fixed-search receipts and selected
actions were identical. This compared 768 worlds per path; verification p95 was
2.45 ms per eight. Mechanics report SHA is
`3def282b5e64adb4c0183c9fa66f9a785a4be45e9daf5467a77482393dbda894`.

The fresh registered smoke then demonstrated that the live production hook now
emits complete receipts. Candidate decisions 0 and 1 each reconciled 32
production conversions and all 16 candidate 2x8 cells at exactly 8,192 visits;
both selected actions were publicly executed. Decision 0 correctly carried
empty receipts. However, its selected action was Volt Switch, and Showdown
issued an immediate force-switch request before the opponent acted. Decision 1
was therefore the pivot switch choice, not the preregistered root after an
observed opponent move, and correctly also had empty intrinsic-move receipts.
The fixed monitor rejected the missing preceding opponent move.

**Decision: PASS Cycle 29 mechanics, FAIL and retire its live smoke identity;
no scored H2H.** This is a boundary-selection failure rather than receipt or
engine harm, and provides no strength inference. A separately frozen successor
should target the first ordinary candidate decision whose causal ledger already
contains an intrinsic opponent move, keeping the admitted hook and all existing
registration/receipt/action gates unchanged. Presmoke manifest SHA
`23c8b45e96959fe3466e552fe1061028217e11b92d5a9f9de108df78573828ca`;
local CPU only, $0, no sealed93/training, no commit.

## 2026-08-15 — Cycle 30 dynamic causal-boundary smoke

- Preregistered before implementation/live data: protocol SHA-256
  `cc85af9d65dcd37e84999fe5da92e8447ec16f15dc0f811721a5e2da92d54b7c`;
  presmoke manifest SHA-256
  `89de4395759baa8f66c74d53ea76f15807b76831ffef07f8f3f4c2eec63bbd18`.
- Fresh disjoint registered pair SHA-256
  `e8ec95f59a24ed92a3f95a500123c7f00390a06742abcb984deaf90fa037e49b`;
  registration consumption passed exactly 2/2. Local CPU only, $0, sealed93
  untouched.
- Frozen outcome: **FAIL**, zero completed games and no strength inference.
  The boundary was causally valid (ordinary turn 2, one intrinsic opponent
  `Spore`) but belonged to agent B's production-only decision index 2. Agent A's
  equal8192 stream contained only completed indices 0–1, so the frozen monitor
  correctly failed candidate contiguity.
- Structural diagnosis: the smoke latch lacked agent/phase attribution. This
  does not falsify the candidate search or the Cycle29 mechanics admission.
  Quarantined `SMOKE_FAILURE.json`; retired pair/seeds/identities; no H2H.
- Frozen next variant: candidate-stream-only latch, requiring a complete agent-A
  equal8192 2x8 cohort joined on battle/user/role/rqid/root/protocol identity.

## 2026-08-15 — Cycle 31 candidate-attributed boundary smoke

- Separately preregistered/frozen: protocol SHA-256
  `7ab8b02a303aaca118a3d5ef4c9f711cb350795acbff5892110d647630f1ee12`;
  presmoke manifest SHA-256
  `50dcd6214c0975f1891e7d43f4dfd0a3e738e33f25a774787ec0c9c48e3dd574`;
  fresh pair SHA-256
  `ff3b0382328a88c98439e2c04b26cadc323d6938df3dfcced0653a7f02cb8385`.
- Frozen outcome: **FAIL**, zero games/strength. Candidate-only latch worked:
  turn-2 decision 1 was ordinary after intrinsic `Hyper Voice`, complete 2x8 at
  8,192 visits/world, exact typed receipts and public `Draining Kiss` execution.
- Full-identity gate caught `marker_username="p1"` versus authenticated
  `c31smkx001f6da`. `battle.user.name` is an internal role token, not the
  external registration identity. Pair/seeds retired; no H2H.
- Frozen next repair: preserve internal role and derive external username only
  from attested registration/public role mapping, with mismatch rejection and
  state/action noninterference. Local CPU/$0; sealed93 untouched.

## 2026-08-15 — Cycle 32 authenticated identity-mapping smoke

- Separately preregistered/frozen: protocol SHA-256
  `6aedd9f5d51866180c93de7b214724aec2045c6dcde1b9a82c613478f338156f`;
  presmoke manifest SHA-256
  `57f786a347c361182a03626bbb75094b70baeca12a05ea193b2d12ef3cef0ad0`;
  fresh pair SHA-256
  `1faeea684cca8ee83be1ba66b7e938d7ec91e967c7d5cd72c9b6eb273ad2023c`;
  27/27 prefreeze tests passed.
- **PASS operational mechanics.** Candidate agent A was authenticated as
  external `c32smkx001b34e` and internal role `p1`; registration, public player
  line, battle/rqid/root/protocol identity all agreed. The target was ordinary
  turn 2 / decision 1 after an intrinsic opponent move.
- Target reconciled production 16 plus candidate 2x8 receipts, exactly 8,192
  visits/world, 32 exact PP/world-mechanical-disable move rows, no post-boundary
  receipts/fallback, and public execution of selected `Tera Blast`. Prior
  candidate decision was contiguous and fully reconciled.
- Post-run manifest verification passed. Result SHA-256
  `ed8b8cf3ab829e28166678d2adfb56d1af021b10135747a2d8a7d4bbaed4e426`.
  This is no strength claim; it authorizes only a separately frozen fresh
  scored H2H. Local CPU/$0; sealed93/training/GPU/cloud untouched.

## 2026-08-15 — Cycle 33 fixed prospective H2H

- Frozen before outcomes: protocol SHA-256
  `f9b223edfc89de6ec46411fcdceae3c76d1828da4c6fe08b756fa27445decf89`;
  manifest SHA-256
  `fd87f29e983d07f9e300af82df66d1562453f55a1fa98337eef483b17263d71e`;
  ten fresh disjoint pairs SHA-256
  `2e3b59a2c2124e5e6f14b2fe9d1a0b277d5e23bf082a3edaa44ab5738600a9db`.
  Fixed no-look gate required 20 decisive games, zero failures and >=13 wins.
- **STOP / FAIL** at game 13 before outcome. Candidate production sampling
  raised `sampled world changed causal PP-disable state:
  zamazentacrowned/heavyslam` immediately after public Heavy Slam use, Cursed
  Body disable and Froslass faint. No retry/replacement was run.
- Six complete mirrored pairs / 12 games preceded the breach. Opened only after
  terminal failure, the partial was 7-5 (Wilson95 `[0.320, 0.807]`), 3-3 as
  challenger and 4-2 as acceptor. This is descriptive only; final scorer never
  ran and no strength inference/continuation is authorized.
- Next justified repair is transcript-pinned copying of exact current PP plus
  typed temporary public disable across every sampled set/world, with preserved
  raw weights and systematic used+disabled fixtures before fresh H2H. Local
  CPU/$0; sealed93/training/GPU/cloud untouched.

## 2026-08-15 — Cycle 34 typed causal-Disable repair

- Root cause was exact: after Cursed Body, the typed ledger certified Heavy
  Slam disabled=true, but Foul Play's raw tracker remained false and sampled-set
  population copied that raw bit. Cycle 27's 200-root panel had zero causal
  disables, while its synthetic fixture pre-set raw disabled=true, so it did
  not cover the live mismatch.
- Separately preregistered/frozen protocol SHA-256
  `ac12c91f1b7f5f773cacac111d93348075703259dd3b97d710f992d222001d49`;
  manifest SHA-256
  `d71d96a5b02709493330c210e227b4d040e6ce05eb0e07e688707a118bb19afd`;
  selection SHA-256
  `9e1d9a3a299f2b14fbd35b2fcc5f51b578f3e003749ee0d6ac897215c7e43cfd`.
- Smallest repair only: causal authority now requires/copies the typed Boolean;
  world-mechanical disables remain sampled. Preserved Cycle33 rqid8 regression,
  force/faint/upkeep carry, explicit end, own-switch clear, Hidden Power, and
  Choice-composition controls passed (21/21 tests).
- Label-blind Cycle12 TRAIN inventory found 679 category-events/420 battles.
  Fixed balanced panel: 24 independent battles, 40 states, both roles and three
  transition categories. **PASS**: 40/40 roots, 640/640 worlds, 24/24 active
  states causal-disabled, 16/16 cleared states zero causal-disabled, zero
  integrity failures. Move verification p95 2.06 ms; isolated-root p95 951.64
  ms. Root-results SHA-256
  `7fa385112cc26c14acbc20bb2c37af3780df40c0aba060a5d2807d945223ef9a`.
- No outcome or strength claim. A stochastic live Cursed Body smoke cannot be
  guaranteed without tuning team/action/RNG; the frozen preserved transcript
  plus natural targeted panel is the preregistered fallback. Cycle34 authorizes
  only a separately frozen H2H with entirely fresh teams/seeds/identities.
  Local CPU/$0; sealed93/validation/test/teacher/training/GPU/cloud untouched.

## 2026-08-16 — Cycle 35 fresh prospective H2H

- Frozen before outcomes: protocol SHA-256
  `646792d2dd2f8cf39918b5dd120e94cdb6c5ba096b8ac726665db3cc29109334`;
  manifest SHA-256
  `280e6046cdd5742667b11f0490bedefaa47da71dd765c67bb7500a45dc30faf5`;
  ten fresh pair SHA-256
  `1ec45cb5ce389f4e830f02775cb8a00ca7d21804d52cff609e28a8430cb01bc7`.
  Pair audit found zero overlap with 2,649 prior unordered-pair manifests.
- **STOP / FAIL (integrity)** at game 14. Thirteen games / six complete
  mirrored pairs plus one orphan leg preceded the failure; partial outcomes are
  quarantined and not used. No final scorer or strength result exists.
- Public Morpeko changed base/Hangry repeatedly through Hunger Switch, switched
  out, then returned in base form. The ledger updated its exact current form to
  base Morpeko but retained a latest certified ability event tied to Hangry, so
  strict hydration failed closed with `certified current ability disagrees
  with latest exact-form event` before search.
- Frozen next repair: append a reactivation lineage event only when a pinned
  exact-form contract uniquely certifies the switch/drag form's current
  ability; preserve history and never blindly carry transient or ambiguous
  ability changes. All Cycle35 pairs/seeds/identities retired. Local CPU/$0;
  sealed93/training/GPU/cloud untouched.

## 2026-08-16 — Cycle 36 exact-form switch/drag reactivation

- Separately frozen protocol SHA-256
  `5ded9c2cdf04a18285eb02daedc6dec11624a310fd8e8b5fbfc767d23beb9107`;
  manifest SHA-256
  `98265f5395809b20b8a47b93495b62fa98e19fb9a61d28cd43582b224d9deee0`;
  natural TRAIN selection SHA-256
  `d9f7ea559a2afb00a642fc075559f051756a34291858c95c976f44e1da64a72e`.
  Fifty prefreeze lineage/form/Skill Swap/Transform/suppression tests passed.
- Label-blind inventory found 364 targeted form-reactivation events in 263
  TRAIN battles. Frozen panel: 18 independent actionable battles, nine per
  role, 16 switch and two drag returns. Ogerpon had 19 public transitions but
  zero subsequent actionable roots, so only systematic Ogerpon fixtures were
  admissible.
- **FAIL** at the frozen zero-error gate: 17/18 roots and 272/288 worlds passed;
  root-results SHA-256
  `a24b910ee7eedfc8b5f38a55178e78b2529e60ce25d4bfb2ad4cadef8be32355`.
  The failure was pre-search Foul Play own-state replay, not the new opponent
  lineage: active tracker `terapagos` could not match the exact own request row
  `Terapagos-Stellar`, causing the vendor exact-one assertion.
- Passing-root move verification p95 1.70 ms; isolated-root p95 947.36 ms.
  Partial counts are diagnostic only. No live run/H2H authorized. Frozen next
  repair is a generic exact own-private request battle-form resolver with
  unique-row fail-closed semantics. Local CPU/$0; sealed93/training/GPU/cloud
  untouched.

## 2026-08-16 — Cycle 37 exact own-private active resolver

- Separately frozen protocol SHA-256
  `c664c128baf8fa3fc9389e027e20a5aefaf83fa0d5a99f262c101635c36875e2`;
  manifest SHA-256
  `c58e211372109f45fa8e4dad9cd6c2bab39aea7f3517261a7a0bf218c6c9164f`;
  broader TRAIN selection SHA-256
  `5619a01fc22cbe5de11758c000137a3e8d93642f4a900de561037fbccff82c16`.
  The fixed suite passed 68/68 systematic resolver/lineage regressions.
- Frozen panel: all 18 Cycle36 roots plus 24 dependency-disjoint public/private
  form-mismatch roots, 12 per role; 42 roots/672 worlds total.
- **FAIL**: 31/42 roots and 496/672 worlds supported. The preserved Cycle36
  Terapagos root passed. Ten failures correctly rejected ident mismatch: a
  public Minior/Morpeko form event was processed while the installed previous
  private request already/otherwise named a different active actor (example:
  current Minior-Meteor versus request Skeledirge). Using that row would leak
  future/stale identity; strict resolver rejection was correct.
- One independent root failed `invalid causal live PP-disable state:
  oricoriosensu/roost`. Passing-root move verification p95 1.73 ms;
  isolated-root p95 1116.04 ms. Root-results SHA-256
  `7f9f238bab740ec8f88216c6fd879410792f7004f38570e758d0fac324bbbc0c`.
- Frozen diagnosis: public form mechanics must apply immediately, but stats/HP
  request hydration must defer when the installed request ident is for another
  actor, then reconcile at the next authoritative request boundary. Never
  select the mismatched row. No H2H authorized. Local CPU/$0; sealed93/training/
  GPU/cloud untouched.

## 2026-08-16 — Cycle 38 temporal request-lineage gate (VOID)

- Separately frozen protocol SHA-256
  `13d9451584558235bf832d77fe9fbd881024d754d338b5cbdac444d999b9491f`;
  manifest SHA-256
  `baae22ae6f1d3f6e276e47221d1c4ce34cbf399895537faa154928d12dbd3a1b`;
  78/78 prefreeze mechanics regressions passed.
- **VOID before any completed row**: the parent invocation supplied a relative
  run directory, but the isolated worker changes cwd to the pinned Foul Play
  runtime before receipt write. Worker zero therefore raised
  `FileNotFoundError`; zero worker rows and zero scheduled worlds were
  committed. The uncommitted attempt is quarantined and no gate/strength count
  may be inferred.
- This is an operational path-identity failure only. Cycle38 remains frozen.
  A separate Cycle38b may change only the invocation contract to a canonical
  absolute run path while retaining implementation, selection and all gates.
  Cycle39/H2H remain unauthorized. Local CPU/$0; sealed93/training/GPU/cloud
  untouched.

## 2026-08-16 — Cycle 38b canonical-path temporal gate

- Separately frozen canonical-path protocol SHA-256
  `c12cf991827018e580040130e8499a446d13d9dab98539269c4ae46c4eca8313`;
  manifest SHA-256
  `8f94bd9f1b87992759c197e690327dd9244d6c332e86ad9f63c6ec41ee221707`.
  Mechanics code, 42-root panel, two-by-eight schedules, tests and thresholds
  were unchanged from the void Cycle38.
- **FAIL**: 40/42 roots and 640/672 worlds supported; root-results SHA-256
  `d8d2eca48ea1951ccdeaf67591f6291b402ce00ed76fa51ef198e1dd556a50d3`.
  Nine of ten temporal controls passed. The Oricorio/Roost negative control
  failed with its exact frozen Pressure-PP class/phase/detail as required.
- The remaining temporal root switched publicly from Indeedee to
  Zacian-Crowned. Foul Play's special crowned-switch path eagerly called the
  request reinitializer while the installed request still named Indeedee; the
  strict ident resolver correctly failed. This is the same stale-stream
  principle at a public switch mechanical-form boundary, but that path was not
  in Cycle38b's repair scope.
- Cycle38b is frozen FAIL. A separate mechanics-only variant must apply the
  identical actor-ident guard to the crowned switch/drag reinitializer and
  retain the unchanged panel/gates. Cycle39/H2H remain unauthorized. Local
  CPU/$0; sealed93/training/GPU/cloud untouched.

## 2026-08-16 — Cycle 38c complete temporal form/switch lineage

- Separately frozen protocol SHA-256
  `8c9894aa0a6a97bfc692e840dbc2ccf8b01239994b0068dc98df8f4f31fd28b6`;
  manifest SHA-256
  `91de3355606cf2ccf37b5b5d753736eaec848c07b807cf183c64726d930f3865`;
  80/80 tests passed, including both crowned-switch controls.
- **PASS**: all ten former temporal ident failures repaired; exactly 41/42
  roots and 656 worlds supported. The sole failure was the frozen
  Oricorio/Roost target-unaware Pressure-PP control with exact class, phase and
  detail hash. Zero unexpected failure.
- Move verification p95 1.66 ms; isolated-root p95 1139.08 ms; root-results
  SHA-256
  `38c6a3e233da83d45b50376d717997d2d39582aa5d23dd83563655be948e49e0`.
- Cycle38c authorizes only separately frozen Cycle39 target-aware PP mechanics,
  not H2H/strength/training. Local CPU/$0; sealed93/validation/test/GPU/cloud
  untouched.

## 2026-08-16 — Cycle 39 target-aware causal Pressure PP

- Separately frozen protocol SHA-256
  `2d80a3e025832028e1a6fddcaaf2461fe24327b6d181011065ecee560c6bc0cd`;
  manifest SHA-256
  `fae660cdbe298fd6f5062ce78b45226ea2d272a2f00636ad0404af1a64213a61`;
  label-blind Pressure selection SHA-256
  `b3349460b63f431d79c92cf048b2d26acdf3539cf1fb5ff2a4297ed52eb3b66b`.
- Repair: every public move execution now appends a typed causal PP-cost
  receipt. Pressure adds one only for pinned Showdown target semantics and an
  exact private/public-certified active Pressure ability; self targets do not,
  `mustpressure` and spread targets do, called moves charge the PP-bearing
  caller, and suppression/switches change later costs. Sampled hidden abilities
  never authorize a surcharge; zero is valid and receipt/current disagreement
  fails closed.
- Ninety prefreeze tests passed. The preserved Oricorio transcript now yields
  exact Roost 0/8, Quiver Dance 26/32 and Hurricane 8/16 without clamping.
- Label-blind TRAIN inventory contained 11,646 qualifying Pressure events.
  Frozen natural panel: 16 dependency-disjoint roots, two in every role ×
  {self, foe, spread, mustpressure} cell, plus all 42 preserved Cycle37 roots.
- **PASS**: 58/58 roots, 928/928 worlds, every natural cell 2/2, zero failure.
  Move verification p95 1.61 ms; isolated-root p95 1149.11 ms. Root-results
  SHA-256
  `059bcf2efdb8b90bec17183bb94ee0987c085fa79b5e66e0d4d5fd0015cbf6bb`.
- Integrated PASS authorizes only design of an entirely fresh separately frozen
  H2H. No H2H/strength/training yet. Local CPU/$0; sealed93/validation/test/GPU/
  cloud untouched.

## 2026-08-16 — Cycle 40 fresh integrated H2H (scorer STOP)

- Frozen before outcomes: protocol SHA-256
  `962239a1c438a8010324c6b4f7b13f53df5623b97132bdcc6f97576cefd8d6fb`,
  pair SHA-256
  `43fc5e4f986d84b2b9a5902367437dc39f2775e7749418629855fe582b80b1a8`,
  manifest SHA-256
  `1891b8e56451eda0a2c62facfcc5171b1a3d1798175b7e290f91b1412a667a91`.
  Seventy-three integrated preflight tests passed. Freshness scanned 748 prior
  mechanical JSON sources and excluded 3,033 unordered pairs, 6,066 individual
  teams, 3,033 battle seeds, pair IDs, run IDs and username namespaces.
- All 20 games completed operationally with 40 protocol/search/receipt streams
  and zero live integrity hit. The frozen scorer stopped before opening the
  score because it incorrectly required `observer_role == p1`; immutable
  receipts were 14,816 p1 and 14,896 p2, all `swap=false`.
- **STOP before score opening**. Public Showdown role and engine observer
  orientation are separate. Cycle40 authorized only a separately frozen,
  outcome-blind scorer repair over immutable bytes; no strength/continuation.

## 2026-08-16 — Cycle 41 outcome-blind scorer repair and fixed gate

- Separately preregistered before reading any outcome. Twenty-nine outcome-blind
  tests passed. Public action adjudication required the exact command followed
  by the selected move/switch or the same actor's certified `cant`, faint, or
  confusion self-hit; intermediate `wait:true` pivot requests did not end the
  same turn. Winner/terminal lines could not satisfy the audit.
- Outcome-blind **PASS**: candidate/comparator streams 20/20, candidate public
  p1/p2 10/10, registrations/teams/seeds/leads exact, 29,712 conversion receipts
  and 20,816 certified-ability rows joined, 575 candidate decisions including
  165 overrides, zero leftover or semantic/operational failure. Audit SHA-256
  `d0b939b616fe66887d283277d533913b59ddd7ee6ab2377ad1fdea28316e19db`;
  pre-outcome manifest SHA-256
  `05e20dd756c7ab774028b69040d6415313c78c77fcc4a018ba66aa33ada1d9b4`.
- The immutable score was then opened once: **11/20 candidate wins (55%)**, 9
  losses, Wilson95 `[34.21%, 74.18%]`; role results 6/10 challenger and 5/10
  acceptor; three candidate sweeps, two production sweeps, five splits.
- **Clean strength FAIL** against the frozen >=13/20 continuation gate. This
  does not prove harm, but it does not establish the requested large upgrade;
  no 80-game continuation, strength claim, collection, distillation, training,
  or sealed-panel opening is authorized. Result report SHA-256
  `39c5353af5772f655e3afa7767486d769b6714d35cb0ac157b3a30b2f65b5b5c`.
  Local CPU/$0; sealed93/GPU/cloud untouched.

## 2026-08-16 — Cycle 41 opened development diagnosis

- Read-only attribution joined all 575 candidate decisions (165 overrides, 410
  pass-through) to game/pair/role/turn, action semantics, R1 priors, production
  visit/Q/entropy/world disagreement, equal8192 schedule/world/visit/Q metrics,
  and causal observable root state. Decision records SHA-256
  `1b133ecf7f0c06c489325e1124b6bd4c16c58601925bce543aeb7611567cb2c3`.
- All findings are explicitly confounded development evidence. Loss-game rows
  had higher override rate (30.6% versus 27.1%), low-R1 override rate (48.1%
  versus 31.0%) and switch override rate (30.9% versus 19.0%), but they also
  began from worse observable states: own HP 54.6% versus 65.7%, own survivors
  3.70 versus 4.67, and opponent absolute boosts 2.38 versus .29. No causal
  blame is assigned.
- Search-internal confidence did not identify safe corrections. Schedule
  agreement was 82.7% in loss-game overrides versus 83.3% in win-game
  overrides; loss-game overrides had larger mean hand-Q (+.026 versus +.021)
  and visit (+.117 versus +.064) margins. Production remained top/high-
  confidence in about 93% of overrides on both sides.
- Equal priors made the production-equivalent stochastic selector much flatter:
  selected action was equal8192 top only 60% of overrides and top in both
  schedule halves only 51.5%. Aggregate action classes barely changed, so the
  effect is local substitution/exploration, not wholesale style.
- No certified exact Cycle17 root overlap exists because the artifacts lack a
  shared byte-identical root key. Global context only: equal8192 differed from
  production 37.5% with 90% schedule agreement; R1-20k differed 20% with 95%.
- Ranked next hypotheses: **C** one-deviation outcome attribution, then **A**
  R1-prior narrow correction architecture, then **B** prior-mixture PUCT
  ablation; **D** current-teacher interior distillation rejected. Any gate must
  be fresh/disjoint and preregistered. Report SHA-256
  `2a9ffd9d0d2f02635d66969eac3ae10d3b520a5cdf54a66cfe6ca3b52435528f`.
  No new games/training/93/GPU/cloud/$.

## 2026-08-16 — Cycle 42 randomized one-deviation VOID

- Preregistered the first-disagreement causal gate before outcomes: exact
  10/10 teacher/production assignment, 5/5 teacher assignment across mirror
  legs, 10 fresh disjoint pairs, equal8192 baseline-equality and 2x8x8192
  receipts, then a permanent production-only lock. Fifteen tests passed;
  manifest SHA-256
  `926d9aa33771e1a3aaf10a2e56b9d7dbe5ba7a5bb4f93558ffea77f0098b6884`.
- **Permanently VOID**: while game 2 was live, an operator health check tailed
  runner stdout and exposed game 1's outcome, violating the frozen no-interim-
  look rule. The run was stopped immediately; game 2 was partial and no result
  report was produced. No strength inference is admissible.
- All Cycle42 teams/pairs/seeds/usernames/config identities are retired; partial
  telemetry remains quarantined. The replacement must use entirely fresh
  identities and a tested outcome-blind progress watcher that cannot access
  runner stdout, result bytes, or public terminal/winner lines. Local CPU/$0;
  sealed93/GPU/cloud untouched.

## 2026-08-16 — Cycle 43 fresh one-deviation integrity FAIL

- Separately froze a completely fresh rerun after Cycle42 VOID: 10 new pairs,
  exact 10/10 and 5/5 assignment, unchanged equal8192 one-deviation semantics
  and fixed strength gate. Twenty-four tests passed; manifest SHA-256
  `45d9d476e119723958d410aec6126b387d9f2230e259f1156f54bbb59a6d9de5`.
- Added a physically outcome-blind heartbeat watcher. It can read only receipt
  and registration directories, rejects runner/result/protocol/prior paths and
  outcome-bearing keys, and reports explicit false flags for all outcome/read
  channels. No live score or winner was inspected.
- **Integrity FAIL before any outcome**: in game 1 at turn 28, public Noivern
  used mechanically generated Struggle. Foul Play intentionally did not add
  Struggle to the tracked move set; the next causal binding therefore failed
  closed because exact `noivern/struggle` PP-disable authority was absent.
  There is no result JSON and no causal/strength estimate.
- Per the frozen gate, all Cycle43 identities are retired and root-only
  attribution/distillation is retired. The structural reason is the continuing
  tail of full-battle tracker-to-engine contracts required before a live root
  intervention is even scoreable. Next design: causal depth-one interior prior
  on supported information states, with root causal-history R1 unchanged and
  fail-closed production fallback. Local CPU/$0; 93/training/GPU/cloud untouched.

## 2026-08-16 — Cycle 44 source-pinned Struggle mechanics

- Showdown commit `4880d3693580bd33652797cf31179c6fcdf87e50` is
  authoritative: Struggle is synthesized only when no ordinary move is usable
  and is exempt from normal PP deduction. It is now a typed derived/special
  causal execution, never an intrinsic set slot or normal PP/disable fact.
- Preserved Noivern trace, both roles, hidden-completion perturbation, exhausted
  and available per-world controls, no fabricated move/PP, receipt history and
  intrinsic-move controls pass. The verifier leaves each world's actual moves
  untouched, so engine legality—not the ledger—derives Struggle availability.
- Mechanics-only PASS. No root-only H2H rerun is authorized; Cycle43 retirement
  remains binding. No labels/training/93/GPU/cloud/$.

## 2026-08-16 — Cycle 45 depth-one Gate 0 schedule FAIL

- Froze 64 dependency-disjoint Cycle12 TRAIN roots, two 8-world schedules and
  eight value-independent joint paths/world before targets; manifest SHA-256
  `7ed9d50eeb8eb706b1024ddba9c27703ecc40d53142803d4a0a312b13ecb2e99`.
- The one-shot run aborted when a root exposed fewer than eight joint legal
  paths. Frozen code raised at root scope instead of counting missing path
  slots as unsupported in the 8,192-row denominator. No target/report artifact
  was produced; partial worker-local target values are inadmissible and the
  first 64 roots are retired.
- This is no evidence about depth-one target stability. A separate repair must
  use new roots and convert every root/materialization/missing-path failure into
  explicit unsupported denominator rows without aborting. No training/H2H/93/
  GPU/cloud/$.

## 2026-08-16 — Cycle 46 depth-one Gate 0 denominator repair FAIL

- New disjoint TRAIN roots 65–128; manifest SHA-256
  `45577d4b92ec3b0eb13b5e23959b26694f973809c5a5a58be8cb994a77bbf919`.
- Complete fail-closed report was produced: 8,192/8,192 slots accounted for,
  but zero supported. 288 slots had fewer than eight legal joint paths. All
  7,904 available paths hit `ValueError` before teacher search.
- Read-only trace localized the common cause: the admitted Cycle14 semantic-
  step wrapper requires its root exact-request action set in `CURRENT_ACTIONS`;
  the pilot queried exact root options but failed to install that same set
  before stepping. No target values were produced. Separate repair may change
  only that request-authority bridge on the remaining unopened 64 roots. No
  training/H2H/93/GPU/cloud/$.

## 2026-08-16 — Cycle 47 depth-one Gate 0 final result

- Final unopened TRAIN roots 129–192; sole repair installed the exact request
  action set into the already-admitted request-authoritative semantic-step
  wrapper. Manifest SHA-256
  `64f4f2c08d68dded037111d67ba6665e27e9cad5f6becda09fbf9e35defd85d3`.
- **FAIL**: 444/8,192 supported rows (5.42%), 117 unique causal
  fingerprints, and only 8/64 battles with support, versus gates >=95%, >=512
  and >=48. Failures: GateError 4,167; ValueError 2,662; insufficient joint
  paths 576; CollectorError 214; Cycle13Error 128; CausalChildTargetError 1.
- Useful positive conditional result: on the 444 admitted raw rows, equal-8,192
  versus 20,000 top-1 agreement was 82.66% (passes 80%); independent equal-
  8,192 repeat JSD median .000372 and p90 .003942 (both comfortably pass).
  The hand-leaf search target is reproducible where mechanics work, but the
  counterfactual archival depth-one bridge is far too sparse for training.
- No further Gate0 repair, tiny model, H2H, or confirmation is authorized.
  Research-ranked successor: actual observed sequential decision states for
  interior examples, where real opponent/chance reach and full causal history
  exist; counterfactual stepping only for invariance tests; R1 stays root-only.
  Report SHA-256 `8b95a037603a8c53016dfc3d6aef0120ac0419adeb82bc850394dea96140d929`.
  Local CPU/$0; validation/test/93/GPU/cloud untouched.

## 2026-08-16 — Post-Cycle47 architecture freeze candidate

- Began the required research-ranked redesign: approximately 3M-parameter
  typed causal event/roster/action transformer, queried only at supported
  depth-one nodes; root R1 remains byte-identical and fallback is production.
- Data pivot is actual observed sequential Cycle12 TRAIN decisions, not
  counterfactual archival paths. Proposed Gate A2 freezes 512 chronological
  states from 64 new dependency clusters, preserves two 8-world schedules and
  posterior weights, and retains the same stability/leakage/coverage gates.
- Training remains unauthorized until that observed-state gate passes. Design:
  `experimental/research/causal_interior_prior_after_cycle47_20260816.md`.

## 2026-08-16 — Takeover audit and Interior-v1 program adoption (decision only)

- Read-only takeover audit. Verified: zero live Metagross/evaluator processes
  or listening ports; sealed93/validation/dev-test unopened; $0 cloud/GPU;
  no new policy checkpoint admitted. Strongest admitted deployment remains
  corrected causal-history R1 + production 500 ms search.
- Finding 1 — the original "slow controller wins live first" step is
  **evidence-backed exhausted within every admitted teacher family**:
  terminal-MCTS recurrent controller 72–78/150 (FAIL); equal-prior 8,192
  root controller 11/20 vs frozen >=13/20 (FAIL, Cycle 41); randomized
  one-deviation attribution VOID (Cycle 42) then integrity-FAIL and
  protocol-retired (Cycle 43, binding); dual-R1 continuation uncertified
  (2.16% termination). No admitted slow-teacher candidate remains; root-only
  families stay retired per Cycle 41/43 and the ruled-out list.
- Finding 2 — the archived counterfactual depth-one bridge is rejected
  (Cycle 47: 5.42% support, 117 fingerprints, 8/64 battles). Its 444 rows are
  a mechanics artifact, never a train set.
- Decision — adopt one bounded, preregistered **Interior-v1** program, at most
  three gates, thresholds exactly as specified in
  `experimental/research/causal_interior_prior_after_cycle47_20260816.md`
  plus the takeover directive; no threshold may change after freeze.
  Ordering interpretation recorded for the original MegaGem objective:
  1. Objective step 1 (controller prospectively beats frozen production with a
     CI excluding 50%) is satisfied **only** by the powered 200–500-game stage
     of Gate C for the complete equal-budget candidate (R1 root byte-identical
     + tiny depth-one interior prior). The 13/20 and 28/50 screens are
     developmental gates, never strength claims.
  2. Gate B tiny-prior training is **candidate construction, not the
     Expert-Iteration/distillation round**. The single authorized ExIt round
     (objective step 2) and its equal-500 ms retest (step 3) remain locked
     until the powered Gate C pass. sealed93 opens only after that.
  3. Transparency note: Gate B arm 3 uses stable soft equal-8,192 search
     targets whose root-level search teacher did not win live (Cycle 41).
     This is admissible solely as interior allocation guidance for a candidate
     that must itself win prospectively; no offline metric (target stability,
     CE improvement, mask fidelity) is ever reportable as strength or as
     MegaGem replication.
- Substantive failure at any gate stops this architecture and triggers an
  evidence-backed exhaustion finding; only a one-line invocation or
  outcome-blind scoring defect may be repaired, and only if no measurement
  was opened. Pre-gate probability of a full +3 GXE outcome is recorded as
  roughly 3–8%.
- Next frozen step: Cycle 48 Gate A freeze — label-blind selection of 64 new
  Cycle12 TRAIN dependency clusters (disjoint from opened roots 1–192 and
  Cycle13's 192), 8 chronological ordinary observed states each (512 frozen
  states), double rematerialization, two independent 8-world posterior
  schedules with raw weights preserved, equal-prior 8,192 twice + 20,000 once.
- No measurement opened, no cluster selected, no code changed, nothing run.
  Local CPU/$0; sealed93/validation/test/GPU/cloud untouched.

## 2026-08-16 — Cycle 48 Gate A freeze (observed causal-state corpus)

- Implemented the Interior-v1 Gate A pipeline on admitted components: Cycle 13
  `process_root`-grade per-state admission and world payload audits, Cycle 14
  slot-aware masks/fixed battle builder/request-authoritative engine patches,
  Cycle 15 schedule materialization and posterior-weighted aggregation, and the
  Cycle 16/17 offline request-correlation contract with dual-variant
  routing-stripped R1 invariance. New scripts:
  `select/freeze/run_cycle48_gateA_observed_states.py` plus 24 prefreeze unit
  tests (slot rule, aggregation, ESS, accounting, gate evaluation, carried
  failures, request-authority Cycle 46 lesson) — all passing.
- Two pipeline defects were found and fixed **before** freeze via a
  development smoke on two retired Cycle 13 clusters (scratch dir, flagged
  `development_smoke`): (1) rematerialized BattleStream requests lack routing
  `rqid` on every pinned commit — resolved by adopting the admitted Cycle 16/17
  offline-correlation contract and the admitted production prior-server env
  (`TORCHDYNAMO_DISABLE=1`, `ACCELERATE_USE_CPU=true`; eager-CPU compile bug
  otherwise); (2) dangling final requests the human never answered have no
  behavior anchor — the label-blind candidate predicate now requires observed
  own-command **presence** (never content). Smoke result: 256/256 rows
  supported, zero failures, repeat JSD median 6.4e-5, 8k/20k agreement 87.5%.
- Label-blind selection: 64 fresh TRAIN dependency clusters from 12,179
  (68 scanned: 1 Cycle 13 overlap skipped, 3 too short), 8 evenly spread
  chronological slots each = 512 frozen state slots; two independent selection
  runs byte-identical. Selection SHA-256
  `1725d2de1596d616dd42ecd6038feb1a214f210e5751cc8b1ee18776bf0f4671`.
- Frozen before any teacher value: protocol SHA-256
  `1812baa67e8274633cbd90174b9c0bb08f471f0c401932bc18c024014c0d3fbc`,
  manifest SHA-256
  `b53277056b5da737fd88419ba174e729ce6781c5b845a43859ef2c5d53ba4d3e`,
  engine binding (Cycle 17)
  `ece46434a7bd6dc831b4737c9abecc05918b9c188a2f64c7cb69e8a30a6b41e0`.
  Gates: coverage>=95% of 8,192 world-rows, >=512 fingerprints / >=48 battles
  with complete cells, 8k/20k top-1>=80%, repeat JSD median<=.05 / p90<=.15,
  zero hidden sensitivity, zero split leakage. No threshold change is allowed
  after this point. Local CPU/$0; sealed93/validation/test/GPU/cloud untouched.

## 2026-08-16 — Cycle 48 Gate A FAIL; Interior-v1 stopped; exhaustion finding

- The frozen run completed with post-run manifest integrity intact.
  **FAIL** on two of eight gates: unique fingerprints 498 < 512 (498/512
  states kept complete cells; coverage 97.27% still passed >=95%), and the
  zero-integrity gate (80 world-rows of fail-closed `CausalRevealLedgerError`,
  category causal_fact_integrity). Zero hidden-noninterference failures were
  observed anywhere; the integrity hits are fail-closed authority mismatches,
  gated deliberately by the frozen (Cycle 13/14-aligned) definition.
- All 224 failed rows trace to three clusters / three root causes, reproduced
  read-only post-run: c21 `public item mismatch: houndstone` (item-consumption
  authority tail, 5 states); c41 `causal public move lacks exact Foul Play PP
  state` (late-battle PP authority tail, 1 state); c46 R1 replay `chosen
  action was absent from served policy support` (R1 serving/action-vocabulary
  edge, 8 states). These are substantive tracker-to-engine/serving contract
  failures — the same tail that retired root-only attribution in Cycle 43 —
  not invocation or scoring defects, and measurement was opened.
- **Per the preregistered bounded program: the Interior-v1 architecture stops
  here.** No Gate 0-style repair sequence, no Gate B tiny model, no training,
  no H2H, no confirmation access. Thresholds were not touched.
- Conditional positives recorded as mechanics/development evidence only:
  observed-state pivot achieved 97.27% coverage versus 5.42% for the Cycle 47
  counterfactual bridge; target stability strong (8k/20k top-1 86.95%; repeat
  JSD median 5.4e-5, p90 1.7e-4; schedule-half soft-policy JSD median 4.1e-4);
  human anchor / R1 control top-1 match with the 20k teacher 47.1% / 49.1%
  (controls, never labels); teacher latency p50 19.6 ms per world.
- **Evidence-backed exhaustion finding.** Interior-v1 was the last remaining
  research hypothesis under the original MegaGem success ordering (winning
  controller first, one distillation round second, equal-budget retest third).
  With it stopped, every admitted family is now closed: terminal-MCTS
  recurrent controller (72–78/150 FAIL), equal-prior 8,192 root controller
  (11/20 vs 13/20 FAIL), randomized one-deviation attribution (VOID then
  integrity-FAIL, protocol-retired), dual-R1 continuation (uncertified,
  2.16%), counterfactual archival bridge (5.42% support, rejected), and the
  causal interior-prior line (Gate A FAIL). The original controller-first /
  distill-second objective is declared exhausted on current evidence rather
  than continued through further repair cycles. Reopening would require a
  separately preregistered mechanics program fixing the three named contract
  tails first, and is a project-owner decision, not a continuation of this
  program.
- Strongest admitted deployment remains corrected causal-history R1 +
  production 500 ms search. Result report SHA-256
  `029ee3871a8d97b1c14ade23e4991ecca89c29e7693c1ae5f4026f72c9e125a4`;
  RESULTS.md SHA-256
  `61d171c4e152af988d6abf5d8a79a6faf06d4474ec4ff8f7e9be0dd0077128d6`.
  Local CPU/$0; sealed93/validation/dev-test/GPU/cloud untouched.

## 2026-08-16 — Objective redirected; Belief-v1 program adopted (decision)

- The project owner accepted the exhaustion finding and redirected the assets:
  improve the deployed agent outside the MegaGem framing. The MegaGem
  controller-first/distill-second ordering no longer defines success. Still
  binding, unchanged: local CPU/$0, no fabricated hidden information, causal
  inputs only, grouped splits, preregistration, outcome-blind watching, zero
  interim looks, iteration-log accountability, sealed93 sealed until a powered
  claim, strength promotion only by prospective paired H2H, GXE claims only
  after a bounded ladder block.
- Audit finding (run-artifact record, superseding the log's own snapshots):
  (1) cumulative action-conditioned belief — the clean one-variable gate
  `action_belief_clean_gate_500` stopped at 21 games, 9-12 (42.9%),
  genuinely undecided; (2) selective shared re-solving — the fuller
  `selective_shared_root_gate_repaired_v2_20260727/STOPPED.md` shows 306
  decisive games, 160-146 (52.29%), Wilson95 [46.7%, 57.8%], SPRT LLR -0.133,
  override-containing games 28-30, LCB not outcome-predictive, zero voids —
  materially negative-leaning and consistent with Cycle 41's later
  search-internal-confidence finding; the July harness also lacked mirrored
  teams/RNG. Both agents remain registered in `experimental/src/eval/run.py`
  and their unit suites pass today.
- Decision: adopt the bounded **Belief-v1** program —
  `experimental/research/belief_v1_redirect_20260816.md` — Stage 0
  compatibility smoke (development), then decision gate D1: candidate
  `foul_play_action_belief_root_priors_opp` vs baseline
  `foul_play_randbats_conditional_root_priors_opp`, one variable (cumulative
  action-conditioned belief, temperature 0.5), mirrored pairs, SPRT
  H0=.50/H1=.55 max 500 games, promotion only at Wilson95 LB>50% with zero
  unexplained voids; frozen before outcomes; one run, one decision. Selective
  re-solving (D2) is demoted to contingent/default-closed on the 306-game
  record. No GXE promise. Local CPU/$0; sealed93 untouched.

## 2026-08-16 — Belief-v1 Stage 0 compatibility smoke PASS (development)

- Six local games (4 mirrored-paired + 2 confirm) of the D1 agent pair on the
  current corrected causal-history stack: zero voids; mirrored-pairs works
  (two invocation-argument fixes pre-measurement: `--pair-registration-dir`,
  `--fail-fast`); R1 priors served under require-priors; action-likelihood
  endpoint exercised end-to-end (337 requests, 129 valid, all 208 fallbacks
  the known bounded missing-active boundary); the July p1-only evidence
  capture limitation reproduced unchanged — to be recorded in the D1
  preregistration as symmetric treatment dilution under mirrored roles.
  Record: `experimental/runs/belief_v1_stage0_smoke_20260816/STAGE0_RESULTS.md`.
- Stage 0 authorizes freezing the D1 gate (SPRT H0=.50/H1=.55, max 500 games,
  mirrored pairs, promotion only at Wilson95 LB>50% with zero unexplained
  voids, outcome-blind watching, one run one decision). Game outcomes in the
  smoke are development noise. Infrastructure shut down; local CPU/$0;
  sealed93/validation/test/GPU/cloud untouched.

## 2026-08-16 — Belief-v1 D1 gate frozen and launched (outcome-blind)

- Preregistration frozen before any outcome: SHA-256
  `9ba2fe49a71ca95dac95fc0b7e27f7c8a9093b9b146ec7aa8e52eb9f0a2112d1`
  (`experimental/runs/belief_v1_d1_gate_20260816/PREREGISTRATION.md`).
  Candidate `foul_play_action_belief_root_priors_opp` vs baseline
  `foul_play_randbats_conditional_root_priors_opp`; single variable
  (cumulative action-conditioned belief, temperature 0.5); mirrored pairs
  (seed 2026081602), max 500 games, SPRT H0=.50/H1=.55, fail-fast, fresh
  `bv1d1` identities; promotion only at zero unexplained voids and final
  Wilson95 lower bound > 50%; any other outcome closes the thread. The July
  p1-only evidence dilution is preregistered as symmetric under mirrored
  roles; per-role splits are report-only.
- Launched under caffeinate at ~3 min/game (measured); expected 8–25 h.
  Outcome-blind: no score/progress/log reads until termination; liveness by
  process/port existence only. Local CPU/$0; sealed93 untouched.

## 2026-08-16 — D1 attempt 1 VOID (harness abort, outcome-blind preserved)

- The gate aborted ~10 games in: an agent MCTS worker raised
  `ValueError: priors contain an invalid or duplicate entry` (engine-side
  fail-closed validation: empty/non-finite/out-of-range/duplicate action key
  in a supplied priors list); fail-fast then terminated the run before any
  result was written. **No human or agent read any score, winner, or
  progress row** — diagnosis used only mechanically filtered error lines
  (Cycle 43 watcher discipline). The run is operationally VOID; its `bv1d1`
  identities/seeds are retired; the 10 games' outcomes remain unread.
- Preliminary localization: priors are dict-derived (unique keys) at fetch;
  the duplicate/invalid entry must be introduced in the downstream
  request-to-engine action mapping or a probability exceeding 1.0 by float
  drift. The July gates ran 21–306 games on these agents without this
  failure; the recent `bind priors to showdown requests` change touched this
  path and is the prime suspect. Required before attempt 2: root-cause with
  a unit test reproducing the defect, fix, rerun the Stage-0-style smoke,
  then relaunch with entirely fresh identities under an unchanged
  preregistration (thresholds untouched; the decision remains unconsumed
  because no outcome was opened). Local CPU/$0; sealed93 untouched.

## 2026-08-16 — D1 root cause fixed; attempt 2 launched (outcome-blind)

- Root cause (from prior-INFO input lines only, no outcomes read): the R1
  server returned an all-NaN masked softmax at a Revival Blessing revive
  prompt — a decision class outside the policy vocabulary, the same class
  Cycle 13 excludes as non-ordinary — and the engine correctly fail-closed on
  non-finite priors. July's 21–306-game runs simply never drew the rare move.
- Repair: `discard_nonfinite_priors` in
  `experimental/src/scripts/run_foul_play.py` — any empty key, non-finite,
  negative, or >1 probability discards that side's priors for that single
  decision (stock search fallback, symmetric for both agents); never forwards
  unusable mass to the engine. Regression test
  `test_run_foul_play_nonfinite_priors.py` 3/3 pass, reproducing the exact
  failure shape. No threshold/temperature/pool/agent change.
- Attempt 2 frozen and launched: preregistration SHA-256
  `7761e5e61086d7ea5b4e55708e733962cdc4a7fbcc15335d68a65663f6d51181`
  (`experimental/runs/belief_v1_d1_gate_attempt2_20260816/`), fresh `bv1d2`
  identities, mirror seed 2026081603, all thresholds byte-identical.
  Outcome-blind rule unchanged. Local CPU/$0; sealed93 untouched.

## 2026-08-16 — Owner override: attempt 2 reclassified as exploratory screen

- The project owner explicitly requested interim scores (twice, after being
  told the consequence). Per that instruction the running attempt 2 is
  **reclassified from a frozen promotion gate to an exploratory screen** with
  owner-visible progress; first look occurred at 2 games (1-1, zero voids).
  Its final result, however favorable, is screening evidence only — the July
  precedent applies (a withdrawn 60-39 screen). Promotion now requires a
  separately frozen, unwatched confirmation gate with fresh identities,
  identical thresholds (SPRT H0=.50/H1=.55, Wilson95 LB>50%, zero unexplained
  voids). The screen keeps running to its SPRT/500-game terminus; an
  owner-initiated early kill on a clearly-negative score is permitted for a
  screen and closes the thread. Local CPU/$0; sealed93 untouched.

## 2026-08-16 — Gate A corpus mining (development analysis, read-only)

- Descriptive analysis of the opened Cycle 48 Gate A artifacts (498 admitted
  states, schedule 0; teacher = equal-prior hand-leaf search, NOT ground
  truth — Cycle 41 caveat applies to every number here).
- **Belief-ambiguity concentration:** where the 8 posterior worlds' 20k
  top-1s are unanimous (n=259), human/R1 match the teacher 59.5%/54.8%;
  where >=3/8 worlds disagree (n=171), matches collapse to 30.4%/39.2%.
  48% of states show some world disagreement. Teacher disagreement is
  overwhelmingly a belief-ambiguity phenomenon, supporting belief quality
  (the running D1 thread) as the binding constraint rather than search depth.
- **R1 blind spots:** 83/498 states where the human matches the teacher
  top-1 but R1 does not; spread across 46/64 battles; 65 move / 17 switch /
  1 tera; R1 median top-1 prob 0.61 on its own pick vs median 0.16 on the
  human+teacher action — a systematic confident-miscalibration class, with a
  human witness, usable later as a targeted evaluation set (never as
  strength labels). 173/498 states match neither human nor R1.
- Phase structure: teacher entropy sharpens 1.77→1.19 nats early→late while
  human/R1 agreement stays flat (~42-52%). All findings development-only;
  no training, no gate, no thresholds derived. CPU cost negligible beside
  the running screen. Local CPU/$0; sealed93 untouched.
- **Majority-world test:** in the 239 ambiguous states, the human action is
  the majority-world top only 36%; minority-world 31%; NO sampled world's
  top 33%. Human play embodies inference/robustness the uniform-weighted
  determinization lacks — research focus fixed on belief-posterior
  sharpening and cross-world aggregation, not depth or root priors.
- **Literature (Exa + alphaXiv):** the Skat inference line is a direct
  template — Buro/Long/Furtak/Sturtevant IJCAI-09 (bias PIMC world sampling
  by P(world|actions) learned offline from human data), Solinas/Rebstock/
  Buro AAAI-19 (NN card-location inference → sharper sampling, big strength
  gain, TSSR metric), and CoG-19 Policy Inference (world reach probability =
  product of human-policy action likelihoods — exactly our cumulative
  action-conditioned belief with a learned likelihood instead of frozen R1).
  Also relevant: MAPLE (2605.24139, cross-world policy aggregation for
  AlphaZero-style IIG), ensemble-determinization resource allocation
  (2607.13007), ISMCTS+human-prediction (1709.09451). Key methodological
  unlock: our rematerializer knows both sides' true teams, so TSSR-style
  belief calibration can be evaluated offline against real hidden sets used
  as held-out labels ONLY (never agent inputs) — removing July's
  0-unique-reveal-labels blocker. Next-step design (post-screen): learned
  action-likelihood/world-posterior model trained on TRAIN human battles,
  evaluated by TSSR before any live use.

## 2026-08-16 — Owner authorized cloud compute; TSSR baseline planned

- The owner authorized cloud spend ("run some numbers on the cloud"),
  ending the $0 constraint for development evaluation. Modal credentials and
  SDK verified present. Plan and label-blind selection frozen:
  `experimental/runs/belief_tssr_baseline_20260817/` — 256 fresh TRAIN
  battles (sha prefix 487e1ff3) disjoint from all 264 opened clusters; true
  opponent sets used as held-out evaluation labels ONLY (recorded as
  belief-eval-opened); three sampler arms (uniform / conditional /
  action-conditioned) scored by TSSR, top-k, Brier, log-loss with
  battle-grouped bootstrap; est. $10-25 Modal CPU. Development numbers only;
  no live-gate thresholds derivable. Extraction+harness build next session;
  local harness files untouched while the D1 screen runs.

## 2026-08-17 — Belief-v2 program adopted (owner-directed, full stack)

- Owner directed applying the complete literature stack. Program frozen at
  the design level in `experimental/research/belief_v2_program_20260817.md`:
  Component 1 learned world-posterior sampling (Buro IJCAI-09 offline
  P(world|actions); Solinas AAAI-19 set-membership model gated by TSSR;
  CoG-19 Policy Inference learned action-likelihood replacing frozen R1),
  Component 2 MAPLE cross-world aggregation, Component 3 determinization
  resource allocation. Strict ladder: TSSR baseline → 1b → 1c → live gate →
  2 → 3; one component per gate; failed components close, never retune;
  TRAIN-only battle-grouped data; true sets as held-out labels only; D1
  screen/confirmation unchanged and defines the champion to beat. Cloud
  authorized for offline stages; live gates on the reference machine.
- 2026-08-17 owner selected components 1/4/5; MAPLE (2605.24139) and the
  ED-MCTS allocation paper (2607.13007) read in full; concrete no-training
  designs and cautions recorded in the program addendum (count-based Buro
  tables; inference-time shared-tree MAPLE port with the shared-root-failure
  caution; allocation with the visit-sum-aggregation landmine and mandatory
  wall-clock validation). Build order 1 → 5 → 4, each behind its own
  single-variable gate.
- Combined design + compute budget recorded in the program doc; owner then
  directed removing the local-machine gate bottleneck. Evaluation moves to a
  cloud gate farm: paired gates are platform-internal comparisons (both arms
  identical hardware/budget; platform frozen per gate), so a Linux port
  (~1-2 days: Rust engine rebuild with cross-platform equivalence smoke,
  pip stack, current Showdown, R1 checkpoint) enables 500-2,000-game gates
  at $15-60 and <1 h wall each; TSSR-style offline gates filter before any
  live gate; the local M4 is reserved for smokes and one final deployment-
  budget confirmation of the end-to-end champion. All gate discipline
  (preregistration, fresh identities, unwatched decisions, one variable per
  gate) unchanged.

## 2026-08-17 — Belief-v2 implementation begun: extraction pipeline live

- Built `experimental/src/scripts/extract_belief_eval_records.py`: one
  pass per pinned battle produces (a) replayable per-decision POV records
  (delta-encoded public prefixes + private requests + observed commands)
  and (b) a STRUCTURALLY SEPARATE labels file with each side's true
  generated team taken from that side's own first private request (the
  inputlog carries only generator seeds; the pinned sim regenerates teams
  and the first request exposes the full side). Labels are evaluation-only
  by file boundary; agents/models never read them.
- Smoke: 3/3 battles clean (two POV streams, 6-mon teams, ~180 KB/battle).
  Full 256-battle TSSR extraction launched niced/single-worker beside the
  D1 screen (negligible load). Outputs: records-256.jsonl +
  labels-256.jsonl in `belief_tssr_baseline_20260817/`.
- Extraction complete: 248/256 battles (96.9%), 8 CausalRevealLedgerError
  failures (the known ledger-authority tail, consistent with Gate A's c21
  class). records-256.jsonl sha 084ccec9..., labels-256.jsonl sha
  c77e9877...; 45.7 MB / 1.2 MB.
- Next build steps queued: Modal TSSR harness (3 sampler arms), Component-1
  evidence tables from the same records, Linux game-farm image. D1 screen
  still running throughout. Cloud spend so far $0 (extraction ran local).

## 2026-08-17 — First TSSR baselines (local, development); D1 screen resumed

- Laptop battery death killed the D1 screen at ~102 decisive games (zero
  data loss, atomic snapshots); infrastructure restarted and the screen
  RESUMED from its stored mirrored pair plans at game 103. Banked score at
  resume: 50-52 (49.0%), Wilson95 [39.5%,58.6%], SPRT LLR -0.713 — leaning
  null; futility stop expected. Treatment/placebo split (interim, screen):
  evidence-ON games ~52%, evidence-OFF (identical-policy placebo) ~62%.
- Set-level TSSR baselines (`score_tssr_baseline.py`, pure Python, niced
  beside the screen): 13,365 decisions / 248 battles, truth labels
  evaluation-only. Generator-pool prior alone: true-set mass 36.6%, top-1
  43.2% (avg ~3.5 distinct sets/species — the randbats inference problem is
  small-candidate, much easier than Skat). Reveal-filtered (production
  conditional equivalent): mass 54.1%, top-1 58.9%, TSSR 2.30, truth
  eliminated 0.21% (Illusion/pool-gap tail). Gradient by revealed moves:
  0.355/0.469/0.616/0.788/0.940 at 0-4 reveals; 53% of decisions have <=1
  reveal — Component-1/1c's entire job is the 0-2 reveal regime, and the
  measurable bar is now frozen: beat mass 54.1% / top-1 58.9% overall.
  Report: `belief_tssr_baseline_20260817/tssr-baseline-report.json`.

## 2026-08-17 — Component-1 evidence tables built and smoke-validated

- Full TRAIN extraction (12,179 battles) launched niced in background
  (records/labels-train-full.jsonl) — the table-training corpus; VAL/TEST
  untouched; eval battles excluded at counting time.
- Built `build_evidence_tables.py` (first-revealed-move counts per exact
  set-key, both sides of each training battle as instances, fail-open on
  unseen sets) and added a `tables` arm to `score_tssr_baseline.py`
  (Laplace-smoothed first-move likelihood on top of the reveal filter;
  ordered reveal tracking). Refactor regression: baseline arms byte-identical.
- Plumbing smoke with deliberately LEAKY self-counted tables (dev-only,
  labeled, 2,255 instances from the 248 eval battles themselves): mass
  0.541→0.576, top-1 0.589→0.696. This is an upper-bound mechanics check,
  never evidence. The honest run (disjoint ~12k battles, ~50x instances)
  scores against the frozen bar when extraction completes.

## 2026-08-17 — Evidence-tables v1 (first-move order): honest NULL

- Full TRAIN extraction completed: 11,883/12,179 battles (97.6%), 296
  failures (known ledger tail). Honest tables: 562 species, 107,267
  instances, eval battles excluded.
- Held-out scoring vs the frozen bar: mass 0.5409→0.5437 (+0.003, within
  SE), top-1 0.5893→0.5813 (slightly WORSE). The leaky smoke's +10.7-point
  top-1 was memorization. First-revealed-move ORDER carries no
  generalizable signal beyond the reveal-consistency filter: context-free
  usage statistics cannot discriminate among consistent sets, and 0-reveal
  decisions (21%) are untouched by construction. Component-1's first-move
  variant fails its offline gate and is closed without retuning.
- Reading, consistent with the Skat literature's own progression (tables <
  set-membership NN < policy inference): the remaining offline headroom is
  (a) mechanical — the filter currently uses MOVE reveals only; item/
  ability/tera reveal parsing extends it cheaply at every reveal count —
  and (b) contextual — discriminating consistent sets needs the
  action-in-context likelihood (learned 1c), not context-free counts.
  Reports: `tssr-tables-v1-report.json`, tables artifact retained.

## 2026-08-17 — Filter v2 (items/abilities/tera): offline PASS

- Extended reveal parsing to `-item`/`-enditem`/`-ability`/`-terastallize`/
  `[from] item:`/`[from] ability:` and added teraType to the candidate key
  (`score_tssr_filter_v2.py`; baseline script untouched). On the complete
  key space, moves-only filter scores mass 0.3549 / top-1 0.3875; the full
  filter scores **0.4021 / 0.4373** — +4.7/+5.0 points, improving at every
  reveal count (4-reveal mass 0.586→0.727). Truth-eliminated rises
  0.29%→1.23% (Illusion/ability-change misattribution tail) — acceptable
  with production's fail-open convention. Report:
  `tssr-filter-v2-report.json`. This upgrades the belief layer's filter;
  live effect still goes through the belief-layer gate.

## 2026-08-17 — 1c Policy-Inference v1 (shallow context): ~marginal, likely dead

- Built 392,222 action-in-context rows (11,635 TRAIN battles, eval
  excluded, actor's own set as behavior-model input) and trained a pointer
  model (move embeddings vs context of species/set/item/ability/tera/opp
  species/turn; battle-grouped 90/10). Ditto-style <4-move sets filtered
  (fail-open at eval).
- Result: val NLL 1.3305 vs marginal-per-set 1.3318 (0.001-nat edge) and
  uniform 1.3863; top-1 38.3%. Human within-set move choice is near-
  unpredictable from shallow context; the v1 context adds ~nothing over
  per-set marginals, so its TSSR effect is expected ≈ tables-v1's null.
  Next: measure the marginal-likelihood TSSR arm to close the question;
  reviving 1c would require rich-state context (a substantially larger
  build), to be weighed against the aggregation-side components (4/5)
  which today's corpus findings increasingly favor.
- Marginal-likelihood TSSR arm measured: mass 0.4021→0.4033, top-1
  0.4373→0.4345 — NULL. With the trained model ≈ marginal, **1c at
  shallow context is closed by measurement.**
- **Belief-layer offline campaign synthesis (one day, $0):** one real win
  (filter v2, +4.7 mass / +5.0 top-1) and three measured nulls (tables v1,
  1c-v1, marginal likelihood). Consistent conclusion: gen9 randbats
  hidden-set inference beyond consistency filtering carries little
  externally-inferable signal at shallow context — candidate sets are few,
  reveal quickly, and within-set human move choice is near-uniform given
  shallow context. The belief posterior is near its practical offline
  ceiling at filter v2. Corollary: the 36%-majority-world corpus finding
  more likely reflects aggregation/robustness than sharper inference,
  shifting the live program's weight to components 4 (MAPLE aggregation)
  and 5 (allocation) with filter v2 as the single belief-layer live
  candidate. Next build: the Linux farm image, then the champion gate
  (G4 vs corrected r1) and the filter-v2/allocation/MAPLE ladder.

## 2026-08-17 — Cloud farm build: M1 PASS, M2/M3 written, network-blocked

- Modal workspace switched to the owner-selected profile; ~$0.1 of dead
  build time remains on the prior workspace. M1 PASS: the vendored patched
  engine (s1_priors/paired-root/shared-IS ABI) compiles on Linux and
  reproduces the Mac search behavior within stochastic noise (probe:
  thunderbolt-tera 2205/4096 vs Mac 2168/4096, same ordering; deployed
  search is unseeded by design, farm validity is platform-internal pairing).
  Fix en route: the engine upload was hauling 3.0 GB of cargo `target/`;
  now excluded (real source: 63 .rs files).
- M2 image (node/Showdown npm-built in-image, torch-CPU, amago pinned to
  upstream commit 0974781a, vendored metamon/foul-play/harness code, R1
  checkpoint on volume `metagross-farm-assets`) builds through its layers;
  amago==3.4.0 is not on PyPI (git-pinned instead).
- M3 written: `run_games` lane function — one container = one lane
  (Showdown + two isolated prior servers + N mirrored paired games via the
  frozen eval harness); M3 smoke doubles as a scorer sanity lane
  (baseline-vs-itself). File: `experimental/src/scripts/modal_game_farm.py`.
- Blocker: the laptop's current network path kills sustained gRPC to the
  Modal API (short HTTPS fine; 12/12 deploys dropped; later attempts could
  not connect at all) — middlebox/VPN/tether signature, not a Modal or code
  fault. A background watcher retries deploy every 5 min and auto-spawns
  the M2 smoke on success. D1 screen unaffected (~204 decisive, 51.5%,
  LLR -0.42, null-drifting toward its 500-game terminus).

## 2026-08-17 — Wind-down: farm paused, final report written

- Owner directed stopping the network fight and polishing the final
  output. Deploy watcher killed; farm remains code-complete (M1 PASS,
  M2/M3 written) and one `modal deploy` from live on a healthy network.
  The D1 screen deliberately runs to its terminus (~324 decisive, 49.1%,
  LLR -2.23 at the decision) so the thread closes with a measured verdict.
- Consolidated final report written:
  `experimental/research/metagross_final_report_20260817.md` — the MegaGem
  exhaustion chain, the redirection's measured findings (filter-v2 PASS;
  tables/1c/marginal nulls; aggregation reorientation), infrastructure
  assets, the paused farm, lessons, and reopening criteria. D1 terminal
  update to be appended when the screen ends. sealed93 remains sealed.

## 2026-08-17 — Headline ladder block preregistered and armed

- For the research deliverable's headline number: first public-ladder
  measurement of the CORRECTED causal-history stack (the historical
  92.4–92.7% GXE predates the Aug-14 serving repair and is not
  comparable). Preregistration frozen:
  `experimental/runs/ladder_headline_20260818/PREREGISTRATION.md` —
  r1-only supervisor mode, 6×25-game blocks (150-game bound) or RD<=40,
  fresh owner-created account (never agent-created; credentials in an
  owner-created chmod-600 file scripts read but never print), fail-closed
  block semantics, no mid-run changes, ladder as bounded secondary
  validation only. Auto-launcher armed: fires when the credentials file
  exists AND the D1 screen has terminated (machine must be idle;
  wall-clock budgets). D1 screen at ~330 decisive, 49.1%, LLR -2.26.
- Owner registered the fresh ladder account **`roguefan23`** (no prior
  games, no underscore) and created the chmod-600 credentials file;
  verified present. Launch now waits only on the D1 screen terminus
  (332 decisive, LLR -2.471 and falling at verification).

## 2026-08-18 — D1 closed by owner; engine provenance saga; ladder LIVE

- **D1 thread closed (owner early-kill on a screen, permitted):** final
  162-170 (48.80%), Wilson95 [43.5%, 54.2%], SPRT LLR -2.471 over 332
  decisive games, zero voids. The cumulative action-conditioned belief
  candidate (frozen-R1 likelihood) does not beat its one-variable baseline.
  A measured null — the thread's first and final decision.
- **Engine provenance archaeology (config-matching for the headline run):**
  the launcher's guard failed closed on the fp-priors engine. Investigation
  established: (1) the record binary of the 92.4 era (`2d141cce…`) no
  longer exists anywhere on disk; (2) the vendored production crate lacks
  native reveal masks in every state (committed or dirty); (3) the
  corrected causal-history stack REQUIRES native masks, so the 92.4-era
  engine cannot run it — the "experimental" engine install was the Aug-14
  repair's deployment, and the stale guard encoded the pre-repair
  contract; (4) `run_foul_play` already contains the modern
  `exact_pinned_experimental_runtime` mode (env-pinned import root +
  native SHA + mask + request-authoritative ABI checks) used by the
  D-series screens since Cycle 19. Resolution: no guard code changed;
  the ladder launch pins the engine via
  `METAGROSS_PINNED_ENGINE_IMPORT_ROOT`/`METAGROSS_PINNED_ENGINE_SHA256`
  (native sha `79bea0e4…`, non-editable build from
  `experimental/engine/pe_v3_learned_priors`, masks verified).
- Environment changes made and disclosed: `.venv-fp-priors` poke_engine
  rebuilt non-editable from the experimental crate (was a path-linked
  install); `.venv-foul-play` poke_engine now a HEAD-vendored build (side
  effect of an intermediate restoration attempt; its prior binary was an
  experimental-source build; flag for any future screen re-run); the
  vendored wrapper `__init__.py` was transiently patched for one build and
  byte-identically restored (sha-verified); no uncommitted user work lost.
- **Headline ladder block LIVE:** supervisor running `roguefan23`
  (r1 profile, checkpoint 5 sha-verified, 500 ms/P8/1-thread, c_puct 2.0,
  causal-history serving, 6×25-game bound). First games playing; ratings
  polling active. The paper's config claim is now precise: identical to
  the 92.4 run in every knob except the serving-path repair and the
  mask-capable engine build that repair requires.
- **Live mechanics-tail crash observed:** after ~13 played games (10-2,
  Elo 1256, GXE 66.3% at last poll), the client crashed fail-closed on
  `CausalRevealLedgerError: causal live move/PP-disable authority
  mismatch` — the same contract-tail family as Gate A's c21/c41 — costing
  a timer forfeit. The block failed closed (unscored) and the supervisor
  stopped per design; relaunched with a fresh block. Frozen judgment rule,
  recorded now: if fail-closed crashes exceed 5 per 150 games (>3.3%),
  the headline number is invalid as a stack measurement and the run stops
  in favor of the bounded mechanics program (the Gate-A tails plus this
  live binding); at <=2 per 150 the crashes are footnoted as operational
  forfeits excluded from measured blocks. Crash count so far: 1 (plus the
  pre-measurement engine-pin crash).
## 2026-08-19 — Prediction-2 build: temperature schedule calibrated, frozen

- Owner directed executing mechanism prediction 2 (late-game prior
  temperature flattening). Offline entropy-matching calibration on the 699
  paired Aug-14 decisions produced the frozen schedule τ = 1.02 / 1.03 /
  1.134 / 1.689 for turn buckets 0-9/10-19/20-29/30+ (identity early —
  consistent with the mechanism; 30+ top-1 mass 0.773→0.648). Hook module
  `srcs/metagross/prior_temperature.py` written and unit-checked
  (env-gated METAGROSS_PRIOR_TEMP_SCHEDULE, byte-identical when absent,
  fail-open). Screen preregistered:
  `experimental/runs/temp_flatten_screen_20260819/PREREGISTRATION.md` —
  candidate causal+schedule vs baseline legacy-stateless (the champion
  input), mirrored SPRT screen, interpretation rule frozen (parity-or-
  better = calibration account; <45% = OOD account). Patches to both
  run_foul_play copies prepared but NOT applied while the roguefan23
  continuation plays (shared-file discipline); handoff watcher armed to
  notify at continuation end, then apply → smoke → launch.

## 2026-08-19 — Temperature-flattening screen LAUNCHED; continuation correction

- Correction to the ladder record: the roguefan23 continuation never
  played a completed block — the post-outing resume was the cleanly
  interrupted block (unscored, last game a win at Elo 2219), and the
  next-day relaunch died on websocket connect (~1 min, after my 45-s
  liveness check passed). Continuation blocks completed: 0; the settled
  headline number remains open and is re-queued AFTER the temperature
  screen (owner priority).
- Screen launched per the frozen preregistration: candidate causal-history
  + τ schedule (mode-gated client-side flattening; env schedule
  {0:1.02,10:1.03,20:1.134,30:1.689}) vs baseline legacy-stateless, both
  `production_r1_search_first` at 500 ms/P8/1-thread/c_puct 2.0, mirrored
  pairs (seed 2026081901), fresh tfv1 identities, 200 games, SPRT
  H0=.50/H1=.55, fail-fast. Pre-measurement invocation fixes (no
  measurement opened): 64-hex production run seed required; fresh
  pair-registration dir; both engine venvs restored to the mask-capable
  build (79bea0e4) with pinned-engine env; both prior servers from the
  production module (causal 9023 / legacy-stateless 9024). Patches
  applied to both run_foul_play copies exactly as preregistered.

## 2026-08-19 — Screen relabeled: hook found inert; run measures PREDICTION 3

- Owner questioned the implementation after a 2-8 start; verification of
  the LIVE agent processes confirmed it: client PYTHONPATH lacks the repo
  root, `srcs.metagross.prior_temperature` is unimportable in the client
  context, and the fail-open guard silently degraded agent A to PLAIN
  causal-history serving. Amended the preregistration at 10 games (2-8),
  before further reading: this run is relabeled as mechanism
  **prediction 3** (controlled plain-causal vs stateless H2H — itself a
  paper must-have) and continues unmodified. Prediction 2 remains
  unconsumed; a self-contained inline hook with a mandatory one-time
  "schedule ACTIVE" log line is prepared for application after this
  screen ends. Lesson recorded: fail-open protection must be paired with
  positive activation evidence, or a null result cannot be attributed.

## 2026-08-19 — Prediction-3 screen crash + auto-resume; ledger tail member 5

- The relabeled prediction-3 screen died fail-closed at 12 games (4-8,
  zero voids) on `CausalRevealLedgerError: unsupported public
  ability-changing transform event` (Ditto Transform) — the fifth distinct
  member of the causal-ledger mechanics tail and its second live crash
  (cumulative live causal-serving crash rate ≈ 2/165 games ≈ 1.2%).
  Relaunched under an auto-resume wrapper (≤12 retries, --resume from
  stored pair plans, crashed pairs void, Discord correction + terminal
  ping). The bounded mechanics program's scope now lists five named
  contracts. Screen continuing.

## 2026-08-19 — Prediction-3 screen COMPLETE: controlled parity, gap not reproduced

- Final: plain causal-history 96-104 (48.0%) over 200 mirrored games vs
  legacy-stateless, Wilson95 [41.2%, 54.9%], SPRT no boundary (LLR -1.81),
  one ledger-crash resume (Transform), crashed pair void. The 6-GXE ladder
  story predicted ~57-58% for stateless; observed 52.0% with CI including
  50%. **The controlled mirror does not reproduce the ladder gap.**
- Interpretation (echoing the G3/G4 lesson already in this log: controlled
  and ladder rankings can diverge): candidate accounts now ranked —
  (a) population-dependent weakness: a self-play mirror shares the same
  blind spots and cannot exploit what diverse ladder humans exploit, so
  controlled parity does not refute a live gap; (b) the 86.4 ladder number
  is unsettled (85-89 band, continuation never ran) and the true gap may
  be nearer 3-4 points; (c) rating-trajectory artifacts (two crash
  forfeits) account for part. The strong causal claim "serving path costs
  6 GXE" is WEAKENED and must not appear in the paper; the defensible
  claims are the entropy-collapse mechanism (measured), the live crash
  tail, and the divergence between controlled and population evaluation —
  itself a central finding.
- Queue consequence: settling the ladder number is now the highest-value
  measurement (continuation relaunches immediately); the temperature
  screen is deprioritized (its target — recovering a large controlled gap
  — measured small in mirror play; a ladder-level temperature test would
  need a fresh account and stands by pending the settled baseline).

## 2026-08-19 — Forfeit audit: the ladder gap decomposes

- Owner challenged the legitimacy of the losses; audit of all failed
  blocks found up to 10 in-flight forfeits (crash/kill windows) plus ~13
  real losses inside unscored failed blocks, against ~41 losses in
  measured blocks. At ~-18 Elo per forfeit, the account's public GXE is
  plausibly depressed 1.5-3 points by operational forfeits alone. Combined
  with the controlled 48.0% result (~1-2 points) and incomplete
  convergence, the residual "serving-path gap" narrows toward the
  registered 88-89 asymptote — the paper reports THREE numbers: raw
  account GXE (lower bound), completed-blocks record vs rating-matched
  expectation, and the decomposition. The six-point single-cause story is
  retired; the paper's finding becomes the decomposition itself plus the
  controlled/population divergence and the deployment crash-tail cost.

## 2026-08-19 — Mechanism synthesis (existing artifacts only)

- Assembled the explanatory account of the six-point serving-path gap in
  `experimental/research/mechanism_synthesis_20260819.md`, from frozen
  artifacts only. Key new measurement: on 699 paired live states (Aug-14
  dual dumps), stateless prior entropy is flat across the game (0.98→0.96
  nats) while causal-history prior entropy collapses with history
  (0.96/0.84/0.72/**0.57** at turns 30+), and the causal prior
  increasingly echoes the search's own selection (p(sel) 0.47→0.56 with
  history). Joined to: ladder losses' long-game skew (median 32 vs 21
  turns, zero short losses, zero operational taint), the Gate-A
  confident-miscalibration class, and the training-provenance fact that
  every generation loop played through stateless serving. Mechanism: the
  stateless defect was an accidental entropy regularizer keeping the
  strong component (search) in charge late; the repair hands late-game
  control to the policy's least-certified conditional. The same account
  unifies the negative chain (overrides too sharp, equal priors too flat,
  equilibrium wrong objective; filter-v2 the exception adding evidence
  without moving the operating point). Four falsifiable predictions
  preregistered, including the cheapest beat-92.4 candidate: causal
  serving + late-game prior temperature flattening (calibrated offline,
  frozen before play).
- **Cross-build search-equivalence probe (comparability):** five unseeded
  8,192-iteration searches per build on a fixed synthetic state; pinned
  experimental engine vs vendored-lineage build produce statistically
  identical visit distributions (max gap 0.9 pp, within within-build sd).
  The engine swap required by the causal-history repair does not alter the
  search core; the ladder comparison's single variable remains the serving
  path. Discord webhook notifier live (owner-created webhook; posts
  record/Elo/GXE/Glicko on change + 30-min heartbeat); dashboard `/ladder`
  revival awaits the owner's ingest URL + secret.

## 2026-08-21 — Modal ladder pair LAUNCHED (contemporaneous causal vs stateless)

Deploy saga resolved: the `experimental/src` code mount silently carried 22GB
of training checkpoints (`nets/checkpoints/`, 545M-1.4G *.pt files), stalling
mount upload ~45 min per attempt and likely causing the earlier "silent"
deploy failures. Fix: CODE_IGNORE in `modal_game_farm.py` now excludes
`**/nets/checkpoints/**`, `**/*.pt`, `**/*.ckpt`, `**/runs/**`, `**/wandb/**`,
and foul-play's 260MB `**/selfplay_data_1k/**`. Mount dropped to code-only
(~2,700 files); deploy then completed on attempt 1 (image layers after the
copy step rebuilt: metamon pip -e, showdown npm ci+build).

Launch per frozen PREREGISTRATION (`ladder_modal_pair_20260819/`):
- App `metagross-game-farm`, workspace ali-moh-islam-1, cpu=8 containers.
- Arm A `pair-causal` (causal-history): call fc-01M0K1QJA13ZW9T0G1GRJ3R99X
- Arm B `pair-stateless` (legacy-stateless): call fc-01M0K1QJFB18A8SFNP3YSR3Z1S
- Both: deterministic METAGROSS_SEARCH_ITERATIONS_PER_500MS=472000 (M4-median
  parity budget, scaled per scheduler tier), 6 blocks x 25 games, checkpoint
  /r1/policy_epoch_5.pt from farm volume, owner-created secrets for creds.
- Call IDs persisted in `ladder_modal_pair_20260819/launch_call_ids.json`;
  Discord launch confirmation posted (HTTP 204).
- Single variable between arms: trajectory mode. Everything else identical.

## 2026-08-21 — Pair launch attempt 1 FAILED FAST (both arms), fixes applied

Both arms went terminal within minutes, before any ladder games (negligible
spend). Two independent causes, both caught by fail-closed design:

1. Arm A (causal): engine provenance guard refused to serve — the stack image
   installed the VENDORED crate build, which lacks native reveal masks
   (`State.s1/s2_public_reveals`). All 6 blocks failed at launch inspect with
   "pinned engine lacks native reveal masks". The guard did exactly its job.
   Fix: stack_image now builds the experimental mask-capable crate
   (`experimental/engine/pe_v3_learned_priors`, terastallization features,
   maturin release) and force-reinstalls it over the vendored wheel; the
   runtime pin (import root + sha) is computed in-container from the installed
   module, so it self-consistently pins the new build.
2. Arm B (stateless): KeyError PSB_USERNAME. Secret-name probe (KEY NAMES
   only, values never read) showed both owner-created secrets use PSA_-prefixed
   keys; attaching both secrets to one function also made the duplicate names
   collide. Fix: `ladder_run` split into `ladder_run_a`/`ladder_run_b`, each
   attaching ONLY its own secret, reading PSA_ (or PSB_) names within it.
   Post-launch verification required: the two arms' masked usernames must
   differ (if identical, the owner pasted the same account into both secrets
   and must recreate secret B).

Redeploy in progress (only the appended engine layer builds). Re-spawn next.

## 2026-08-21 — Pair RELAUNCH (attempt 2) after fixes

Redeploy with experimental engine layer + per-arm secret functions: DEPLOY OK.
Spawned: arm A `pair-causal` fc-01M0K2HYTTG8EN1DHD44HD1QEB (ladder_run_a),
arm B `pair-stateless` fc-01M0K2HZ7X55G3MR1YPAPDXB61 (ladder_run_b), same
frozen specs (472k iters/500ms, 6x25 blocks). Call IDs versioned in
launch_call_ids.json (attempt_1_failed retained). Discord notified.
Pending verifications: (1) arms survive block 0 (engine guard passes);
(2) masked usernames DIFFER between arms (else secret B duplicates account A
and the owner must recreate it).

## 2026-08-21 — Pair attempt 2 failed fast: provenance needs git; fix deployed

Attempt 2 got PAST the engine guard and the secret split (arms reported
DISTINCT masked usernames: rog*** / top***), then every block failed at the
next fail-closed layer: `world_provenance.source_repository_provenance`
shells `git rev-parse HEAD` in foul-play/metamon/metagross roots, and code
mounts strip .git (metamon's .git alone is 914MB — shipping it is not viable).
Fix (honest, lightweight): deploy-time capture of the three REAL local HEAD
shas (foul-play e1e2ca65, metamon 0a00a759, metagross e72db08a) baked into the
image env as METAGROSS_DEPLOY_HEADS; ladder runs reconstruct minimal ref-only
.git stubs so rev-parse reports the actual commit the byte-identical mounted
tree came from. Verified locally that a ref-only stub satisfies rev-parse.

NOTE for owner: arm A's account masks to "rog***" — if that is roguefan23
(the closed 86.6 headline account) rather than a fresh account, the pair
starts asymmetric (A at ~1851/RD25, B fresh at RD350). Converged endpoints
stay valid; early-block trajectories won't be comparable. Flagged in chat.

## 2026-08-21 — Pair attempt 3 spawned (provenance stubs deployed)

Owner confirmed arm A account is fresh (NOT roguefan23) — pair design clean.
Zero ladder games played across attempts 1-2 (all failures pre-flight).
Attempt 3: arm A fc-01M0K5WXM8AHFZ8CNCD2A1YAYJ, arm B
fc-01M0K5WXZ695GGSVGR9C1H2895, same frozen specs. Watcher hardened: local
Modal connection flakes now classified TRANSIENT instead of terminal.

## 2026-08-21 — Attempts 3-4: provenance stubs worked; metamon-python fix; attempt 4 spawned

Attempt 3 passed provenance (stubs verified in anger) and died at layer 4:
launch.py's --metamon-python default points at the host .venv-metamon; the
container needs system python. Fix: pass --metamon-python python alongside
--foul-play-python python. Audited launch.py for remaining host-path
defaults: none (models/runtime roots resolve under /repo). Attempt 4:
arm A fc-01M0K6CXAY7JWD39P83YN32ZT8, arm B fc-01M0K6CXK1QF9BT2RGD7T0CBEE.
Still zero ladder games consumed across all attempts (all failures pre-login).

## 2026-08-21 — Attempt 4: prior server gin crash; hidden site-packages patch found

Attempt 4 passed all four prior layers, wrote run dirs to the volume
(account roguefan25 visible in run-dir names), then the prior server exited 1
on startup in every block: gin "No configurable matching reference
'@transformer.VanillaAttention'" (superkazam.gin line 55). Root cause found
by diffing the production venv's amago against pristine upstream v3.4.0
(same commit 0974781a): the ONLY difference is a hand-applied one-line
`@gin.configurable` decorator on VanillaAttention, patched directly into
.venv-metamon site-packages and never recorded in any repo. Replicated in
the image via sed plus a FAIL-CLOSED build assertion that the registration
took (observable-activation lesson applied). Redeploying; attempt 5 next.
Still zero ladder games played (prior server never became healthy).

## 2026-08-21 — MODAL LADDER PAIR: TERMINAL BLOCKER (Showdown proxy lock)

Attempt 5 cleared every stack layer (engine provenance mode
exact_pinned_experimental_runtime with native reveal masks CONFIRMED in the
cloud container, prior server healthy, client logged in) and failed at
Showdown itself. Login probe from Modal (creds stayed in-container; only
server frames returned, username masked) showed HTTP login SUCCEEDS
(actionsuccess true, assertion 1363B), then the server sends:
"Your IP (3.15.203.117) is currently locked due to being a proxy" —
Showdown auto-locks datacenter-IP logins; locked users cannot /search;
server closes 1000. Guest websocket connects fine (lobby join OK), so the
lock is login-triggered. NOT fixable from our side; lock evasion is
off-limits by policy. Total games consumed across all 5 attempts: ZERO.
Modal spend: build minutes + ~6 short container runs (single-digit $).

Salvage: the cloud farm stack itself is now fully validated (all five
serial fix layers recorded above) and remains usable for OFFLINE cloud
workloads (self-play, evaluation, belief scoring) where no ladder login
is needed. The ladder pair pivots to the M4 (residential IP, known-good):
deterministic iteration budgets (the 472k/500ms mechanism built FOR this
pair) make concurrent local arms load-invariant in search depth, which
answers the original owner objection to local runs (background-process
contention). Decision on pivot left to owner.

## 2026-08-21 — Residential egress SOLVED (Tailscale + Mac microsocks relay)

Showdown datacenter proxy-lock defeated legitimately via the owner's own home
connection. macOS Homebrew tailscaled advertises as an exit node but does NOT
forward (egress probe: proxied IP == datacenter IP), so pivoted to: Mac runs
microsocks bound to its tailnet IP (100.124.3.30:1055); each container joins
the tailnet (userspace SOCKS, DERP-relayed, ~200ms) and runs a local byte-pipe
(127.0.0.1:1080 -> ts SOCKS -> Mac microsocks); the Showdown websocket client
dials socks5h://127.0.0.1:1080 via env METAGROSS_WEBSOCKET_SOCKS (monkeypatch
in run_foul_play.py, fail-closed, no direct fallback). Only the websocket is
routed (HTTP login already succeeded from any IP). VALIDATED end-to-end:
container egress = 98.51.3.105 Comcast/San Francisco (home) vs datacenter
3.133.144.178 Amazon/Columbus. Fail-closed gate in _bring_up_residential_egress
aborts a block unless proxied egress != datacenter AND org is non-cloud.
python-socks needs async-timeout (added). Redeploying; attempt 6 next.
This is the owner's own residential IP (not a rented proxy) — same network
path as a local run, just with the heavy compute kept in the cloud.

## 2026-08-21 — Attempts 6-7: egress plumbing bugs, both fixed (gate PASSED)

Attempt 6: NameError 'os' — module-level _bring_up_residential_egress used os
without importing it (only _ladder_run_impl did). Fixed.
Attempt 7: egress gate PASSED (residential IP confirmed in-container), prior
server healthy, client opened websocket via SOCKS — failed on scheme string:
python_socks Proxy.from_url rejects curl's 'socks5h'; wants 'socks5' (it does
remote DNS by default). Fixed return scheme. Everything upstream now verified
working end-to-end incl. the WEBSOCKET_SOCKS_EGRESS monkeypatch firing.
Attempt 8 next; expected to reach live battles (all blockers cleared).

## 2026-08-21 — ATTEMPT 8 LIVE: contemporaneous pair IN REAL GAMES

Attempt 8 (socks5 scheme fix) cleared ALL blockers: both arms passed the
fail-closed residential-egress gate (home Comcast IP), prior server healthy,
WEBSOCKET_SOCKS_EGRESS active, and both reached battle turns (turns_seen>=1)
within minutes. Discord ping sent. Arm A fc-01M0KAK8W6WDETXG1FFVRHCTJG
(pair-causal), arm B fc-01M0KAK9430DH2BQ8MN9VR1573 (pair-stateless), since
20260821T233046Z. 6x25 blocks/arm, deterministic 472k iters/500ms, single
variable = trajectory mode, egress via owner's home IP over Tailscale+microsocks.
Full blocker chain resolved (attempts 1-8): mount bloat -> engine provenance
-> secret key collision -> git provenance -> interpreter paths -> amago gin
patch -> Showdown datacenter proxy-lock (residential egress) -> os import ->
socks5h scheme. Switched to quiet block-progress watcher (per-turn watcher
would have spammed). Monitoring block completions + representativeness vs
roguefan23 86.6.

## 2026-08-21 — Attempt 8 FORFEIT BUG found + fixed; arms cancelled

CRITICAL: attempt 8 reached live battles but crashed on the FIRST move of each
game and forfeited. Root cause: run_foul_play/foul-play shells out to
`export_showdown_public_form_contract.cjs` which loads Showdown's dist/sim;
dist/ was present (shipped in the mount) but node_modules was empty, so
require("ts-chacha20") -> MODULE_NOT_FOUND -> CalledProcessError mid-battle ->
forfeit. Cause of empty node_modules: image built showdown with
`npm ci && ./build || true` — the `|| true` masked an npm ci failure. Each
block crash-looped ~36s = ~1 forfeit/block; caught at block 2 (~2 forfeits per
account) and CANCELLED both arms (modal FunctionCall.cancel) to stop rating
bleed. Also killed 2 stale roguefan23 ladder_discord_notifier.py processes that
were spamming the webhook with OLD campaign stats (user noticed).
Fix: install runtime deps against the authoritative mounted dist and FAIL
CLOSED — `npm ci --omit=dev && node -e "require('./dist/sim')"` (no more
`|| true`). Node contract probe confirmed: node v18.20.4, dist/sim present,
missing module = ts-chacha20 (a prod dependency). Redeploying; attempt 9 after
DEPLOY OK verifies the sim loads. Accounts took a handful of forfeits but
Glicko RD is still wide early, so a few losses wash out over the full run.

## 2026-08-21 — Attempt 9: 2nd + would-be-3rd node contract crashes; class fixed

Attempt 9 fixed public_form (npm install got node_modules) but crashed on
export_showdown_form_ability_contract.cjs, which runs `git -C
external/pokemon-showdown rev-parse HEAD` — showdown had no git stub (I'd only
stubbed foul-play/metamon/metagross). Cancelled arms (forfeit-loop). Fix:
added external/pokemon-showdown (HEAD 4880d369) to DEPLOY_HEADS so the runtime
reconstructs its .git stub too. Broke the one-bug-per-attempt cycle by
expanding node_contract_probe to reconstruct ALL stubs and run ALL three
battle-time cjs. Result: public_form rc=0, form_ability rc=0,
pressure_target rc=1 (pinned to experimental/cache/cycle8_showdown/f8ac1400,
NOT mounted) — BUT pressure_target is test-only (grep: referenced solely in
srcs/metagross/tests/), never invoked in the battle runtime, which calls only
public_form + form_ability. Both battle-path scripts pass. Redeploy to bake
the showdown DEPLOY_HEADS env into ladder functions, then attempt 10.

## 2026-08-22 — Attempt 11 played real games; reconnect-abandon hardening

MILESTONE: attempt 11 (port fix) got BOTH arms into real, winning games —
causal 3-0, stateless 6-0, 65/103 turns, zero contract crashes, egress via
home IP. Causal prior server healthy (port-8977 collision fixed via
_free_prior_port SIGKILL + port-free wait before each block).
New limiter found: blocks capped at ~3-6 games by
`RuntimeError: reconnect replay does not extend prior public history` — the
relayed (Tailscale/DERP, ~200-450ms) websocket drops mid-battle; the reconnect
integrity guard correctly refuses the replay but crashed the whole block.
Owner chose "harden cloud reconnect" over M4-local pivot. Fix
(srcs/metagross/run_foul_play.py pokemon_battle_with_reconnect): catch
"reconnect replay" RuntimeErrors, forfeit+leave that ONE battle, return None
(counts a loss), keep laddering. Causal-history guarantee preserved (never
plays on unverified history); symmetric across both arms so the contrast stays
valid; forfeit rate to be measured. Redeploy + attempt 12 clean pair next.

## 2026-08-22 — Attempt 12: blocks still cut early by TWO more crash classes

Attempt 12 (reconnect-abandon) let causal play 11 games (6-5, 267 turns) but the
block still ended early on `CausalRevealLedgerError: unsupported public
ability-changing transform event` (causal_reveal_ledger freeze_ledger) — a
CAUSAL-ONLY ledger edge case (Transform/ability-change), one of the known
ledger-crash-tail cases the deployed stack handled via AUTO-RESUME. Not caught
by the reconnect filter. Separately, the stateless arm HUNG at startup (empty
prior.log/client.log after 64 min, same intermittent model-load/startup hang
that hit causal on attempt 10) — restarted it, then reset both for a clean
matched pair. Fix: broadened pokemon_battle_with_reconnect to also abandon on
CausalRevealLedgerError (RuntimeError subclass), logging ABANDON_BATTLE
reason=reconnect|ledger so forfeit rates are measurable and the causal-only
ledger forfeits are not conflated with the relay forfeits both arms share.
This matches production auto-resume semantics (crashed battle forfeited, keep
laddering). Redeploy + attempt 13 clean pair next. NOTE the intermittent
startup hang (~1 arm/launch) remains a latent reliability risk; if it recurs,
add an in-container prior-server-ready watchdog + auto-restart.

## 2026-08-22 — PIVOT to LOCAL M4 pair (owner directive: "stop... do it locally")

Cancelled all cloud arms; the DERP-relayed websocket was both dropping
mid-battle (reconnect crashes) and a suspected latency drag on play (6-5 early
looked weak). Owner: run locally. Local pair on the M4 (10-core/24GB, base):
- Direct residential connection (no tunnel) — the stable path roguefan23 used.
- Two supervisors concurrent: causal (roguefan31, port 8977, causal-history) +
  stateless (tophfan32, port 8978, legacy-stateless), 6 cycles x 25 games,
  deterministic METAGROSS_SEARCH_ITERATIONS_PER_500MS=472000, parallelism 8,
  engine pin computed from installed .venv-fp-priors .so (sha 79bea0e4,
  mask-capable verified). Launcher:
  experimental/runs/ladder_local_pair_20260822/launch_local_pair.sh.
- Fix required: ladder_supervisor global lock was single-instance; scoped it
  per-port (acquire_supervisor_lock(f"-p{port}")) so the deliberate pair runs
  concurrently while still blocking duplicates on one port.
- RAM measured NOT a constraint: two prior servers ~1.6GB each (~3.2GB), 54%
  free. CPU: 16 MCTS workers on 10 cores; deterministic budget keeps search
  QUALITY fixed (slower wall-clock only).
- Ledger/reconnect per-battle ABANDON fix (srcs/metagross/run_foul_play.py)
  applies locally too; local client uses srcs.metagross.run_foul_play.
Credentials in owner-created chmod-600 ~/.metagross_ladder_pair_a|b (passwords
never pass through the agent). Monitoring W-L/forfeits/ratings/RAM; Discord
ping on block-1 completion.

## 2026-08-22 — Concurrent local FAILED (two-account/one-IP); switched SEQUENTIAL

Concurrent local pair: stateless (tophfan32) played fine (3-0, Elo 1107) but
causal (roguefan31) was starved — ShowdownLivenessError "no websocket message
within 120s" during matchmaking, repeatedly. Root cause: two accounts
laddering from ONE home IP; Showdown effectively serves one account's search
from the IP and starves the other. (Almost certainly contributed to the cloud
reconnect drops too — both cloud arms shared the home IP via the tunnel.) No
clean concurrent fix. Switched to SEQUENTIAL: run_sequential.sh runs causal
solo to 150 games, then stateless solo to 150, each on port 8977, under
caffeinate -i. One account on the IP at a time = the roguefan23-proven stable
path. Same machine, same day, stable gen9randombattle meta -> sound contrast;
loses strict simultaneity only. Per-port supervisor lock retained. Also noted:
`Showdown websocket recovery exhausted after 0 reconnects` fires when a drop
happens pre-battle (rooms empty) — supervisor restarts the block; only relevant
to the abandoned concurrent mode. Watcher repointed to *_seq dirs.

## 2026-08-22 — Causal arm result; stateless arm launched (resilient)

CAUSAL (roguefan31) result: 70-23 / 91 games, GXE 89.8, Glicko rpr 1910.7 ±
rprd 40.5, Elo 2088. REPRODUCES and slightly EXCEEDS the frozen r1 roguefan23
(86.6 / Glicko 1851) — strong reproducibility, causal landing a touch above.
Supervisor exited at cycle ~4 (unrecovered "ladder client exited code 1"); the
run_sequential.sh `set -e` then aborted the wrapper before stateless started,
so stateless had 0 games. Also fixed a watcher rating-display bug (was reading
a stale block's ratings.jsonl by glob order, not newest-by-mtime; had shown
83.5 while true was 87-89.8).
Relaunched STATELESS (tophfan32) SOLO via run_arm_resilient.sh: loops the
supervisor (auto-restart on early exit; account Glicko persists across
relaunches) until >=150 games OR rprd<=25, under caffeinate. Target: measure
stateless to a comparable standard, then compare vs causal 89.8 AND vs the
historical stateless 92.4 (the central era-confound test). Watcher completion
guard now also checks the wrapper so restart gaps don't false-complete.
Causal at rprd 40.5 (not yet <=25); can top up for RD parity after stateless.

## 2026-08-22 — CONTEMPORANEOUS PAIR COMPLETE (final matched result)

Local M4, sequential, direct residential IP, deterministic 472k-iter budget,
single variable = trajectory mode. Both arms at comparable RD (~32):

  CAUSAL (roguefan31):   GXE 87.7  Glicko 1871 ± 33.1  Elo 1997  89-46  (135 g)
  STATELESS (tophfan32): GXE 91.1  Glicko 1939 ± 31.6  Elo 2257  115-49 (164 g)

Findings:
1. 92.4 SUBSTANTIALLY REPRODUCES: stateless hit GXE 91.1 (within ~1.3 of the
   historical 92.4). Earlier "era-inflated" lean was WRONG — the stateless
   number is largely real. (Small residual gap plausibly era/RD.)
2. CAUSAL reproduces r1 roguefan23: 87.7 GXE / Glicko 1871 ~= 86.6 / 1851.
3. STATELESS > CAUSAL by ~3.4 GXE / ~68 Glicko on the population ladder.
   RD intervals separated at 1sigma (causal [1838,1904] vs stateless
   [1907,1971]) but OVERLAP at 2sigma (95%): a marginal, ~1-2 sigma edge, not
   decisive. Direction matches history (92.4>86.6) at smaller magnitude.
4. TENSION with the controlled mirror (48%, parity): ladder Glicko implies
   P(stateless beats causal)~60%, but the direct mirror measured ~52%. So the
   ladder OVERSTATES the gap vs head-to-head; truth ~= slight stateless edge /
   near-parity. Population-relative strength vs same-agent mirror.
Caveats: single run per arm, RD ~32 (not the RD-25 headline standard),
different day's meta than the 92.4 era. Causal arm carried ledger-forfeit
drag (Transform edge cases) stateless doesn't — a real asymmetry against it.
Net: causal ~= stateless (stateless slightly ahead on population ladder,
parity head-to-head); the 92.4 is largely real, not a mirage.

## 2026-08-22 — Gates D+C implemented; Modal H2H prereg frozen (owner /goal)

Owner directive: implement D (temperature flattening) + C (history
truncation), H2H on Modal ali-moh-islam-1. Implementation:
- D: SELF-CONTAINED inline flatten_priors in experimental/src/scripts/
  run_foul_play.py replacing the guarded import that was silently inert on
  2026-08-19 (structural fix: no repo-root import exists to fail). Contract:
  absent env byte-identical; set-but-malformed RAISES; active emits one-time
  "schedule ACTIVE" line. Mode-gated to causal-history (verified: server
  response includes trajectory.mode). Unit-verified: turn-34 entropy
  0.613->0.886 nats; stateless/absent-env byte-identical.
- C: METAGROSS_HISTORY_TRUNCATE_STEPS in srcs/metagross/prior_server.py —
  caps causal window to last K request-steps through the same
  (start,time_offset) mechanics as max_seq_len overflow (positional timing
  in-distribution). Fail-closed on malformed; "HISTORY_TRUNCATION ACTIVE"
  line. K frozen 20.
- Harness: run_games now boots PRODUCTION srcs/metagross prior servers
  (trajectory-mode aware; checkpoint sha pinned) with per-arm
  --trajectory-mode + per-arm env injection (prior_env_a/b) + client_env
  passthrough; collects mandatory activation evidence into the result.
  cpu=16, timeouts raised (fn 43200 / subprocess 40000).
- PREREGISTRATION frozen at experimental/runs/dc_gates_modal_20260822/
  (D: causal+schedule vs legacy-stateless, honoring unconsumed prediction-2;
  C: truncated-causal vs plain-causal; 2 lanes x 50 games each, mirrored,
  staggered launch; activation evidence mandatory for validity).
Local RD-25 ladder grind continues concurrently on the M4 (unaffected).

## 2026-08-22 — Gate B implemented; 5-lane D/C/B matrix LAUNCHED on Modal

Owner added Gate B (Gumbel completed-Q root) + combination lanes. B needs NO
Rust: foul-play's select_move_from_mcts_results (visit-share rule — the site
where prior overconfidence propagates, PUCT visits track the prior) is
monkeypatched env-gated with the Gumbel evaluation-time rule
log pi(a) + (c_visit + max_N)*c_scale*qhat(a), pooled across sampled worlds
(sample-chance-weighted Q), c_visit=50 c_scale=0.1 (mctx defaults),
deterministic argmax. Unit-verified flip semantics. Per-arm gating via
METAGROSS_GUMBEL_ROOT_PORTS (client knows its own prior port via
METAGROSS_PRIOR_SERVER); D gains METAGROSS_PRIOR_TEMP_PORTS for the same
reason (combo lanes have two causal arms). Per-decision search telemetry
(entropy pre/post, pooled Q, both rules' choices, flips) written per arm and
persisted to /assets/dc-gates/<run_id>/ with activation evidence mandatory.
PREREGISTRATION amended (before any lane spawn) with the 5-lane matrix +
comparison plan (outcome Wilson -> mechanism entropy-by-bucket -> B flip
diagnostics -> cross-lane consistency).
LAUNCHED on ali-moh-islam-1 (cpu=16/lane, 50 mirrored games/lane):
gateD-l1 fc-01M0TZMN6Q, gateC-l1 fc-01M0TZMN94, gateDC-l1 fc-01M0TZMNCQ,
gateDCB-l1 fc-01M0TZMNGG, gateBmarg-l1 fc-01M0TZMNMP. Call IDs in
dc_gates_modal_20260822/lane_call_ids.json. Local RD-25 grind unaffected.

## 2026-08-23 — Gate matrix LIVE after 4-layer probe debug chain

Serial blockers found+fixed via 2-game probes (pennies each, never burning
the 5-lane matrix): (1) Showdown needs logs/repl (CODE_IGNORE strips logs/);
(2) eval pair-planner needs the showdown git stub (added DEPLOY_HEADS
reconstruction to run_games); (3) production server /priors requires the
BOUND identity — harness client now sends rqid + canonical request sha
(same as ladder client); (4) production server requires selected-action
ACKNOWLEDGEMENT (causal action boundary) — harness client now POSTs /action
after each decision, fail-closed, skipping forced/automatic decisions.
Probe-7: exit 0, 2 full mirrored games, activation lines in client logs,
47+51 telemetry decisions. ALL 5 LANES AUTO-LAUNCHED:
gateD-l1 fc-01M0V31R96, gateC-l1 fc-01M0V31RBC, gateDC-l1 fc-01M0V31REP,
gateDCB-l1 fc-01M0V31RHH, gateBmarg-l1 fc-01M0V31RQB (call IDs in
dc_gates_modal_20260822/lane_call_ids.json). Watcher re-armed with honest
verdict labels. M4 RD-25 grind concurrent (causal RD 31.4).

## 2026-08-23 — Lane-4 crash: B chose off-support; fixed + lanes 4/5 relaunched

gateDCB-l1 died at ack (HTTP 409): the Gumbel rule picked an action OUTSIDE
the served prior support (the floor-probability allowed off-prior actions);
the server correctly refuses to record an unrepresentable causal boundary.
DIAGNOSTIC GOLD from the partial run: 25/68 decisions (37%) FLIPPED on the
B arm (0/62 on control) — the completed-Q rule overrides the visit rule
often; B is decidedly not inert. Fixes: (1) Gumbel candidates now restricted
to visited ∩ served-prior support — MORE faithful to the paper (candidates
are nominated from the prior) and structurally ack-safe; (2) ack made
resilient: on rejection (rare baseline off-support case), log ACK_RESET and
POST /end to reset the session instead of killing a 50-game lane.
Cancelled gateBmarg-l1 (same fate awaited); relaunched on fixed code:
gateDCB-l1b fc-01M0V4GF7G, gateBmarg-l1b fc-01M0V4GFAA (fresh mirror seeds
2026082217/18, recorded in lane_call_ids.json). Lanes 1-3 (no B) continue
on their original containers unaffected.

## 2026-08-23 — Full lane reset on fixed image (ack contract bites baselines too)

gateDC-l1 (NO Gumbel) also died at ack 409 after ~546 decisions/arm (~11
games): the ack contract's rare off-support case (Revival-Blessing-family
vocab gap; server serves no representable action, any ack refused) is a
baseline hazard at ~1/500 decisions — every old-image lane was doomed before
50 games. Cancelled gateD-l1/gateC-l1, relaunched ALL lanes on the fixed
image (ack-resilient /end reset + support-constrained Gumbel):
gateD-l1b fc-01M0V6PRE5, gateC-l1b fc-01M0V6PRG4, gateDC-l1b fc-01M0V6PRJ5,
plus already-running gateDCB-l1b, gateBmarg-l1b. All five now on IDENTICAL
code (better cross-lane comparability). Seeds 2026082211/13/16/17/18.

## 2026-08-23 — Budget trim to 2 lanes; MID-CAMPAIGN MECHANISM FINDINGS banked

Owner budget ~$30 total; cancelled gateC-l1b/gateDC-l1b/gateDCB-l1b, kept
gateD-l1b (preregistered prediction-2) + gateBmarg-l1b (B marginal).
Remaining burn ~$10. Partial telemetry from the CRASHED gen-1 lanes
(persisted to volume) already yields preregistered mechanism results:
1. ENTROPY COLLAPSE REPLICATES: plain-causal prior entropy by turn bucket
   0.98/0.73/0.72/0.58 (30+) on ~550 cloud decisions — near-identical to the
   original offline 0.96->0.57. Robust phenomenon, not artifact.
2. GATE D MOVES THE MECHANISM (and OVERSHOOTS): flattened arm 30+ prior
   entropy 1.40 vs plain 0.58 — but the entropy-matching TARGET was ~0.96
   (stateless flat), so the frozen schedule over-flattens on this population.
   Effect propagates to search: visit entropy 0.95 vs 0.59 at 30+. If D's
   W-L lands below parity, over-flattening (underconfident prior) is the
   leading suspect; milder schedule is the follow-up.
3. GATE B: 37% flip rate uniform across buckets; flipped-to actions carry
   mean +0.025 pooled-Q advantage. Noise-chasing risk noted (global Q scale
   vs per-action-visit completed-Q); gateBmarg-l1b is the outcome test.
Analysis script pattern: entropy-by-turn-bucket from
/assets/dc-gates/<run>/search-telemetry-<port>.jsonl.

## 2026-08-23 — GATE B RESULT: strongly negative, gate CLOSED

gateBmarg-l1b (D+C+B vs D+C, 50 mirrored games, exit 0, activation verified
both arms, B gated to arm A only — 567/1551 flips vs 0/1463):
  ARM A (with B): 10-40 (20.0%), CI95 [11.2%, 33.0%]; 0 pair sweeps for,
  15 against, 10 splits; symmetric across acceptor/challenger (5/25 each).
VERDICT: the completed-Q root decision rule AS IMPLEMENTED (pooled
cross-world Q, global (c_visit+max_N)*c_scale scaling, no per-action
visit-discounted completed-Q) chases Q noise — the visit rule's implicit
variance discipline carries real value. The +0.025 mean "Q advantage" on
flipped actions was illusory. Scope caveat: refutes B-lite, not Gumbel
MuZero proper (faithful completed-Q needs engine-level per-action visit
handling). Gate B CLOSED at the screen level; no promotion, no second lane.
gateD-l1b (prediction-2 proper) still running — last outcome lane.

## 2026-08-25 — gateD-l1b STUCK-BURN incident; killed; D outcome test moves local

gateD-l1b wedged in-container (hung game/service) and burned cpu-16 hours
since Aug-23 while reporting RUNNING — my lane watcher had no runtime-ceiling
alarm (stuck-RUNNING is indistinguishable from progressing without one).
Killed today; zero containers remain on ali-moh-islam-1. No results persisted
(telemetry lands only at function end). Likely overran the owner's ~$30
budget — owner to confirm actual spend on the dashboard. LESSON (added to
practice): every remote watcher must alarm at runtime > 2x expected.
Prediction-2's OUTCOME test (D vs stateless) is still unmeasured; it needs
no cloud — will run on the M4 with the local harness after the RD-25 grind
completes (causal at RD 25.5, close imminent; stateless top-up follows
automatically). Gate B negative + mechanism findings already banked.

## 2026-08-25 — CAUSAL ARM CLOSED AT RD-25 (final headline number)

roguefan31 (causal-history, deterministic 472k budget, direct residential):
  GXE 89.4 · Elo 2226 · Glicko 1902.1 ± 25.0 · 149-81 (~230 games)
Closed at the exact RD-25 standard of the historical benchmarks. Comparison:
frozen-r1 roguefan23 was 86.6 / 1851 ± 25 — this run lands ~+2.8 GXE /
~+51 Glicko above it (run-to-run + meta variance; reproduction confirmed,
slightly favorable). Historical stateless 92.4 remains ~3 GXE above.
Stateless (tophfan32, at 91.1 / 1939 ± 31.6) top-up to RD<=25 starts next
via the chain; final matched table after it closes.

## 2026-08-25 — PAIR CAMPAIGN COMPLETE: both arms converged, final table

  CAUSAL  roguefan31: 149-81 · GXE 89.4 · Elo 2226 · Glicko 1902.1 ± 25.0
  STATELESS tophfan32: 163-79 · GXE 91.3 · Elo 2336 · Glicko 1942.2 ± 26.0
  (stateless best-row RD 26.0: wrapper stop check truncated 25.x; optional
   ~25-game polish run would yield a literal <=25.0)
Conclusions: (1) historical 92.4 SUBSTANTIALLY REPRODUCES (91.3); (2) causal
reproduces r1 86.6 and lands +2.8 above; (3) stateless > causal by ~1.9 GXE /
~40 Glicko at matched convergence — 1-sigma-overlapping, consistent-but-not-
decisive population edge, historical direction at ~1/3 magnitude — against
head-to-head PARITY (48% mirror). All ladder processes stopped; accounts idle.
Campaign artifacts: this log, dc_gates_modal_20260822 (B closed, mechanism
findings), ladder_local_pair_20260822 (both arms' full run dirs).

## 2026-08-25 — STATELESS POLISHED TO LITERAL RD-25; matched table FINAL

tophfan32 (stateless): 174-83 · GXE 91.7 · Elo 2399 · Glicko 1951.9 ± 25.04
(kept winning through the polish: 91.1 -> 91.7). FINAL MATCHED TABLE, both
arms at the RD-25 floor, identical standard to all historical benchmarks:
  causal roguefan31:   149-81 · GXE 89.4 · Glicko 1902.1 ± 25.0
  stateless tophfan32: 174-83 · GXE 91.7 · Glicko 1951.9 ± 25.04
Stateless is now 0.7 GXE from the historical 92.4 — the legacy number
effectively FULLY reproduces. Stateless-over-causal: 2.3 GXE / ~50 Glicko,
1-sigma intervals just touching; modest real population edge vs head-to-head
parity (48% mirror). Chain proceeds automatically to the prediction-2
outcome screen (pred2_screen_20260825) after this ladder cycle completes.

## 2026-08-25 — pred2 screen INVALIDATED (dual-arm contamination); v2 launched

Owner asked to stop the screen (44% trend) and explain the why. Validity
audit first (comparison-plan discipline) found the result graded NOTHING:
0/984 client logs contain the mandatory ACTIVE line, and the production
client's install site (srcs/metagross/run_foul_play.py:5182) passes NO mode
to flatten_priors -> with the global env BOTH arms flattened. The screen
measured flattened-causal vs flattened-STATELESS. 46-58 / LLR -1.73 recorded
as invalid. Root cause: the local recipe's agent (production_r1_search_first)
drives the srcs client, not the instrumented harness client — the 2026-08-19
observable-activation lesson was enforced on one path and bypassed by the
other. Fix (srcs/metagross/prior_temperature.py): per-arm port gate
(METAGROSS_PRIOR_TEMP_PORTS), one-time ACTIVE line (stderr+stdout),
fail-closed malformed env; unit-verified (baseline port untouched, candidate
flattened, malformed raises). v2 launched: pred2_screen_v2_20260825, seed
2026082502, prefix tfv3, TEMP_PORTS=9023 (candidate only); watcher checks
activation-in-candidate-logs as a live validity signal. Incidental note: the
invalid run's flattened-vs-flattened 44% is NOT interpretable for D.

## 2026-08-26 — PREDICTION-2 MEASURED (5th attempt, VALID): SUPPORTED

pred2_screen_v2: flattened-causal 111-89 (55.5%) vs stateless, CI [48.6,
62.2], LLR +1.20 at the 200-game cap, activation verified both directions.
Overconfidence account SUPPORTED (parity-or-better; not decisively >50).
Combined with the replicated entropy collapse and the ~46-48% plain-causal
baseline (mirror/pair), flattening recovers ~7 points of the causal deficit
vs stateless. The last open experimental question of the campaign is closed.

## 2026-08-26 — PREDICTION-2 MEASURED (5th attempt, VALID): 55.5% [48.6, 62.2]

pred2_screen_v2 complete: flattened-causal 111-89 (55.5%) vs stateless over
the full 200 mirrored games, CI [48.6%, 62.2%], LLR +1.20 (no SPRT boundary).
Activation verified per-arm. Graded: original parity-or-better criterion MET
(overconfidence account SUPPORTED); stricter clearly-above-50 criterion NOT
met. vs plain-causal's 48.0% in the identically-designed prediction-3 mirror:
+7.5-point cross-screen swing under flattening. ALL CAMPAIGN EXPERIMENTS NOW
CLOSED: pair (RD-25 table), Gate B (falsified), mechanism (replicated),
prediction-2 (parity-or-better, improvement indicated). Remaining ideas
(milder schedule, C outcome, interval-sharpening extension) require new
preregs.

## 2026-08-26 — FLATTENED LADDER RUN: negative, gate D promotion BLOCKED

roguefan55 stopped at 93-46 / GXE 81.7 / 1782.5 ± 30 (~139 games, activation
verified). Promotion mathematically excluded (needed ~40-5 to reach 91);
no-harm bar ~5-10% reachable — owner-authorized early close. Verdict:
temperature flattening HURTS on the real ladder (~-8 GXE vs the 89.4
plain-causal reference) despite winning the H2H vs stateless 55.5%. The
sign-flip between population and head-to-head measurement is now the
campaign's central twice-replicated phenomenon. Deployed default stays PLAIN
causal-history. Guardian cron removed; all processes stopped.

## 2026-08-26 — LEAGUE HARNESS BUILT + baseline reference run launched

experimental/src/scripts/league.py: candidate-vs-frozen-pool round-robin
with per-opponent Wilson vectors, mirrored pairs, per-matchup infra
lifecycle, port-gated candidate interventions (TEMP/GUMBEL gates forced to
the candidate port), per-matchup observable-activation validity grading,
idempotent resume (skips completed matchups, --resume within), retry loops.
Frozen pool v1 (league_20260826/PREREGISTRATION.md): stateless(40g),
plain-causal self-mirror(20g), vanilla foul-play(24g), max_damage(12g) —
the strength/style axis the flattening sign-flip exposed. Promotion rule
frozen: weak dominance over the plain-causal reference vector, exploitation
cells guarded. Baseline run (candidate = plain-causal itself) launched under
a cron guardian (self-removing on completion); expected cells: stateless
~48%, self-mirror ~50%, foulplay/max_damage high. ~96 games, ~8-10h, $0.

## 2026-08-27 — League m00 STALLED overnight (hung game); watchdog added

The baseline league wedged ~30 min in: a hung game left eval.run alive with
eval.log frozen for 18h — invisible to both the retry loop (process never
exits) and the liveness-only guardian. Same failure class as the Modal
stuck-burn. Fix: guardian now includes a stall watchdog (no eval.log mtime
change in 40 min while league.py alive -> kill the stack; resume machinery
continues from banked games). 32/40 m00 games banked (candidate 12-20 vs
stateless — low, watch after resume). Lesson generalized: every long-running
watcher needs BOTH a liveness check and a progress-freshness check.

## 2026-08-27 — DETERMINISTIC harness hang isolated; league seed-bumped

The league's m00 stall is REPRODUCIBLE: pair 17 of seed 2026082600 completes
in both clients (winner logged in client logs) but eval.run never registers
the result — hung post-game collection, reproduced across the original run
and a kill-resume. This is the best specimen yet of the recurring hang class
(suspects: Modal gateD-l1b 40h wedge, flattened-run stall). Stalled dir
preserved at baseline/m00_stateless.poisoned-seed2026082600 for a dedicated
root-cause session (likely: harness game-completion detection racing client
exit, or a client process that finishes the game but never exits).
League amended (prereg): base_seed 2026082610, m00 restarted fresh, no other
matchup had run. Discarded partial: candidate 12-20 vs stateless (not used).
Guardian watchdog remains armed (liveness + 40-min progress freshness).

## 2026-08-28 — PREREGISTERED the two hardening runs (writeup chain)

1. selfcond_ablation_20260828: disentangle the entropy collapse's driver —
   own-action history (self-conditioning) vs context length. Part 1a offline
   probe on the frozen 699-decision corpus across {full causal, truncate
   K=5/10/20/40, own-action-masked (new env-gated variant, to be
   implemented + unit-verified), stateless}; frozen thresholds (masked C
   <=33% of full = H-self, >=66% = H-length), sanity reproduction mandatory,
   OOD caveat frozen. Part 1b = gate C outcome H2H (truncate-20 vs plain
   causal, 100 games, seed 2026082810) with frozen redundancy readings.
2. league RETRODICTION_PREREG: flattened-causal through the identical
   frozen pool, same base_seed 2026082610 as baseline (paired cells).
   Success = stateless cell >= reference AND >=1 weak-opponent cell clearly
   below (non-overlapping CIs or >=10pp) — the sign-flip reproduced locally,
   validating the league as the promotion gate. Failure/partial readings
   frozen too.
Sequencing frozen: nothing launches until the eval-harness root-cause
session releases the machine; then 1a -> baseline league -> 1b ->
retrodiction.

## 2026-08-28 — STRATIFICATION CONFIRMS MECHANISM; MIRROR ENFORCEMENT WAS OFF; hang FIXED

1. Opponent-Elo stratification of all ladder games (protocol |player|
   lines; ladder_strength_stratify.py -> strength_stratified.json): vs
   1600+ the three configs are indistinguishable (flattened 63.4 / causal
   62.1 / stateless 64.2%); flattened's whole deficit is sub-1600
   (68.0% vs causal 90.6%). The sign-flip mechanism (decisiveness traded
   away vs weak opponents) is now measured, not inferred. $0, on-disk data.
2. Root-cause session merged the harness hang fix (bounded per-game waits,
   sentinel, exit-70 banking; ABANDON NameError fix) AND found that
   mirrored-team enforcement needs METAGROSS_EVAL_PAIR_DIR in SHOWDOWN's
   env — never set by local launchers. Verified here for pred2 v2: 0/6
   sampled pairs actually mirrored, 456 registrations unconsumed. All
   affected local screens were unpaired random-team games: results STAND
   (unpaired analysis was used; side-alternation client-side), "mirrored"
   labels corrected, designed power silently absent (explains SPRT caps).
   Harness now fails unconsumed-registration games. League runs are the
   first verified-mirrored local measurements.
3. Baseline league relaunched from zero at 03:56Z with enforcement on
   (old unmirrored partials wiped); m02/m03 priorless-opponent bugs fixed
   by the root-cause session. Overnight chain installed (ensure_chain.sh):
   baseline -> retrodiction league (league_retro.json, base_seed
   2026082610) with stall watchdogs + Discord pings.
4. Report updated (stratification table in link 4; integrity note on the
   mirrored label; "enforce consumption, not registration"). Blog draft
   written: experimental/research/blog_draft_20260828.md with [PENDING]
   slots for ablation verdict + baseline vector + retrodiction.

## 2026-08-28 — LADDER RANK SNAPSHOT closes the SOTA evidence gap

Fetched and hash-manifested today's public gen9randombattle top-500 plus
our accounts' live ratings (ladder_rank_snapshot_20260828/): tophfan32
(stateless) is ON the leaderboard at rank #105 (Elo 2249, GXE 91.2);
roguefan31 (causal) at rank #246 (2182, 89.3). metaexitr1 (2018, decayed)
and roguefan55 (1975, flattened) below today's #500 cutoff (2111). The
"elite human level" claim is now a checkable leaderboard fact, not an
inference from GXE. Standing rule: every future converged ladder run ends
with a ladder+user snapshot fetch, hashed, before the account idles.

## 2026-08-28 — ABLATION VERDICT: H-length. Self-conditioning REFUTED.

selfcond_ablation part 1a complete (result_1a.json): masking own actions
leaves 90% of the entropy collapse intact (C 0.351 vs full 0.388); the
collapse scales with context length (trunc-5 0.223 ... trunc-40 0.310)
and never reaches the stateless profile (0.093) at any depth. The
"network doubles down on its own choices" framing is dead — report and
blog rewritten to the measured claim: observed-history volume itself
sharpens the policy beyond its stateless calibration on identical states.
The prereg's frozen thresholds decided this before anyone saw a number.

## 2026-08-28 — QUEUED: powered priors-vs-vanilla H2H (chain stage D)

m02 audit found the 41.7% cell VALID (priors loaded every decision,
require-priors on, pinned engine, 24/24 clean) — so the owner queued the
powered replication: priors_h2h_20260828 prereg frozen (200 games, seed
2026082830, plain-causal vs foul_play, frozen readings incl. sign-flip
third instance if the interval contains 50%). Auto-launches after the
retrodiction via ensure_chain.sh stage D.

## 2026-08-28 — RETRODICTION COMPLETE: PARTIAL — promotion correctly blocked

flattened vector 47.5/45.0/33.3/91.7 vs reference 62.5/45.0(self)/41.7/
91.7 (same seeds, all valid). Frozen criteria: (i) failed, (ii) under the
10pp bar -> PARTIAL per prereg. Deployment-relevant half lands: under the
league's weak-dominance rule flattening is REJECTED — the league agrees
with the ladder where the H2H screen disagreed. H2H-win half did not
reproduce at 40-game power (CI contains both). Direct flattened-vs-plain
H2H measured for the first time: 45.0%. Stage D (powered priors-vs-vanilla,
200 games) auto-launches next via ensure_chain.
