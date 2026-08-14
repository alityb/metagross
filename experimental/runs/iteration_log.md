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
