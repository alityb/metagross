# Cycle 2: Search-Native Player-Information Network

Status: architecture decision and pre-outcome protocol

Date: 2026-08-15

## Decision

Do not distill, retune, or continue the Cycle 1 terminal-MCTS residual
controller. Its completed 150-game boundary is 72-78 (48.0%) despite frequent,
offline-certified action changes. The accepted agent remains corrected
causal-history R1 plus production 500 ms search.

The next high-ceiling branch changes the search architecture itself. Build a
small, fast network that consumes a leak-free player information state and
supplies soft policy priors at interior search nodes. Search remains the
deployed decision-maker. The first causal experiment keeps the exact simulator,
root R1 prior, belief sampler, and hand leaf evaluator unchanged.

This is not another root residual, scalar critic, resource-shadow blend,
selected-action distillation, or root-only checkpoint substitution.

## Why Cycle 1 Did Not Transfer

The exact terminal teacher answered a valid but narrow question: under its own
frozen continuation, which of the production search's top two actions has the
larger matched terminal return at this root? It did not establish that applying
many such one-step interventions creates a stronger sequential policy.

At the completed 150-game boundary:

- candidate score was 72-78;
- Wilson 95% interval was [40.16%, 55.94%];
- 180 certified deviations were applied in 99 games;
- deviation-containing games scored 44-55;
- pass-through games scored 28-23;
- the last 50 games would have needed a 42-8 finish to satisfy the frozen
  200-game success rule;
- there were zero voids and the sealed 93-battle panel remained unopened.

The conditional game records are descriptive and confounded by root
difficulty. They do not prove each deviation was harmful. They do show that a
large local counterfactual advantage did not become a large whole-game effect.
That falsifies the target for the intended purpose and blocks distillation.

## Remaining High-Ceiling Hypothesis

Production strength comes from search: retained endpoint evidence has
500 ms root-prior search beating direct R1 17-3. The current search nevertheless
uses the 142.8M-parameter R1 policy only at the root and relies on a hand-written
leaf evaluator deeper in the tree. Independent perfect-information searches
also cannot share a learned player-information representation across hidden
worlds.

The Cycle 2 hypothesis is:

> A search-native player-information policy, queried inside the tree and
> trained on regularized search strategies, will improve the entire sequential
> search policy enough to produce a large equal-budget H2H effect.

This mechanism is materially different from every rejected local controller:
it changes priors and action allocation inside the simulated tree before the
root decision is formed.

## PublicSearchStateV1 Contract

The input is constructed from the acting player's legitimate information, not
from a serialized sampled hidden team. It contains:

- exact own team, HP, status, boosts, moves, PP, item, ability, and Tera state;
- public opponent species, HP, status, boosts, and causally revealed moves,
  items, abilities, and Tera state;
- field, weather, terrain, hazards, screens, turn, forced-action state, and
  exact legal actions;
- packed causal reveal masks already carried and reversibly updated by the
  experimental engine;
- aggregated posterior summaries and uncertainty over hidden worlds, never a
  raw per-world hidden field;
- candidate-action semantics, including switch entry, damage range, priority,
  setup, recovery, pivot, status, Tera, and preservation features.

Required invariants:

1. Perturbing an unrevealed hidden completion cannot change the serialized
   player-information input.
2. Apply/reverse restores the input byte-for-byte.
3. Perspective swap obeys the declared zero-sum transformation.
4. Legal actions and canonical 13-action names exactly match the live request.
5. Legacy states without reveal metadata fail closed for formal training or
   use an explicitly named compatibility path; they never infer reveals from
   hidden truth.

This contract does not advance the 142.8M causal-history R1 transformer inside
the tree. That semantic-history path is already known to have inadequate
coverage. `PublicSearchStateV1` is extracted directly from the exact mechanical
state plus its causal reveal masks, exposing own-private fields and masking all
unrevealed opponent fields. R1 remains a root-only anchor.

## Initial Network

Start with a 1M-5M parameter set/sequence policy, selected by measured batched
CPU latency rather than parameter ambition. It has one shared trunk and these
initial outputs:

- player policy over exact legal actions;
- opponent-view policy through the same perspective-safe input contract;
- optional action-kind auxiliaries used only if they improve the frozen
  policy-target validation metric.

R1 remains the frozen root anchor in the first ablation. The new model is first
used only for depth-one interior priors, which isolates the missing capability
and preserves the known-strong root representation. The existing hand leaf
evaluator remains unchanged because multiple learned leaf/value branches have
already failed. Distributional value, opponent-specialist, deeper-node, and
root-replacement heads are separate later ablations unlocked only by a winning
interior-policy agent.

## Existing Data Before New Collection

Inventory and battle-deduplicate the retained sources before training:

- 23,516 human battles;
- 23,870 legacy R1/self-play battles;
- 5,000 admitted league battles;
- 175,319 exact schema-v3 search-policy targets from 4,767 battles;
- complete terminal outcomes and causal observations already retained for the
  named trajectory sources.

This is roughly 52,000 source battles before overlap auditing. The exact
decision count, source intersection, observation compatibility, and split
counts must be emitted by the new builder; estimates are not training
authority. The 93-battle confirmation panel and every held-out H2H seed remain
excluded.

Targets are named by provenance:

- behavior or human action for imitation warm-start;
- production search strategy for policy improvement;
- actual terminal win/loss and exact simulator-derived auxiliary outcomes,
  retained for separately gated later heads rather than the first policy-only
  pilot;
- no Cycle 1 local-deviation label is treated as an improved-policy target.

Split by physical battle and keep source families stratified. No decision-row
split is allowed.

## Cheap Falsification Funnel

### Gate A: representation and throughput

- all leakage, apply/reverse, perspective, legality, and schema tests pass;
- exact Python/Rust feature parity on at least 1,000 states;
- on roughly 200 already-opened roots, at least 95% of ordinary depth-one
  successors support a legitimate acting-player information state and policy
  query;
- deterministic inference parity after export;
- measured batching supports the declared 500 ms search budget;
- no model training beyond a tiny overfit smoke.

Failure stops the architecture implementation. It does not authorize a weaker
or leaky input.

### Gate B: existing-data pilot

Replay roughly 500 already-opened roots to collect 0.5M-2M depth-one
information-state/action rows. Train three local CPU seeds for two matched
arms: regularized soft search-policy targets and the same search's one-hot
selected action. Both use the same R1/human pass-through anchor. Require:

- battle-grouped held-out policy KL/cross-entropy superiority for soft targets
  over one-hot targets;
- at least 90% policy fidelity on the frozen R1/human anchor states;
- no legality, collapse, or source-family regression;
- interior-node inference can actually be called during exact search with no
  raw hidden-world field crossing the boundary.

Offline metrics establish plumbing and learnability only.

### Gate C: direct equal-budget screen

Compare the complete depth-one-prior search agent against corrected R1 plus
production search at identical 500 ms, hardware, worlds, hand leaf evaluator,
and role/team schedule. Run a fixed 20 mirrored games first.

- At least 13/20 with zero semantic or operational failures: continue.
- Otherwise: stop and diagnose before any new generation. This configuration
  did not show the large developmental effect the project is seeking.

This is a development gate, not a strength claim. A survivor runs a fresh,
frozen 50-game mirrored stage and must score at least 28/50 with no
protected-stratum regression before any data scale-up. A powered 200-500 game
gate remains mandatory before a strength claim.

### Gate D: compounding loop

Only after Gate C passes:

1. generate 25,000 search-guided PFSP/league battles;
2. train the interior policy on regularized search strategies while retaining
   human/R1 anchors;
3. reanalyze high-disagreement and high-value-surprise positions with the new
   search network;
4. retain human and historical anchors;
5. promote at most one generation after equal-500 ms H2H against a frozen
   opponent panel;
6. scale toward 100,000 battles only if the first generation wins.

The pure-equilibrium shared-root RM+ treatment is not repeated: it previously
scored 6-18. Any later public-belief solver must include the learned opponent
model and a frozen bounded-exploitation objective, and must beat the complete
independent-search parent.

## Success and GXE

A 20- or 50-game internal win rate is not a GXE estimate. The architecture is
admitted only after a powered equal-budget opponent-panel H2H. The sealed
confirmation panel opens only after that pass. A claim of movement from roughly
92% to 95% GXE additionally requires a bounded public-ladder evaluation or a
validated calibration against a broad frozen ladder-like opponent population.

There is no defensible guarantee of a three-point GXE gain. This branch has a
higher ceiling than residual selection because it improves the search process
at every node, but it is also a new model/search system. The aggressive 20-game
gate exists so a merely incremental implementation does not consume the full
training budget. Based on the retained local failures and the strength of the
root-search mechanism, the current research estimate is roughly 5%-15% for a
full three-point GXE gain before Gate A, rising to roughly 15%-25% only if Gate
A passes and the fresh 50-game screen reaches at least 56%. These are planning
ranges, not calibrated probabilities or a promise.

## Resource Boundary

Cycle 2 starts with local CPU engineering and existing data. No cloud, paid
compute, GPU training, new large collection, confirmation opening, or ladder
games are authorized by this document. Any later accelerator request must name
the measured pilot throughput, exact dataset, maximum cost, expected wall time,
and the Gate C result it is intended to exploit.
