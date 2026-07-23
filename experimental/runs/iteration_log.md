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
