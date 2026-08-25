# Search-Native Public-Belief Architecture

Status: research synthesis and proposed architecture

Date: 2026-08-08

## Executive Conclusion

The highest-ceiling architecture supported by the current evidence and game-AI literature is a Pokemon-specific Student-of-Games system using:

- Public-belief game-theoretic search.
- The exact Pokemon simulator.
- Learned policy, opponent, belief, and counterfactual-value networks.
- Expert Iteration with search-generated soft targets.
- Replay-buffer reanalysis.
- Population and league training with dedicated exploiters.
- Independent verification re-search rather than a positive-admission gate.

The network should serve search rather than compete with it. Search should remain the deployed decision-maker.

## Validated Local Evidence

The completed direct-policy versus search-guided H2H strongly supports a search-native direction:

- `foul_play_root_priors` defeated `direct_r1` 17-3.
- Search won 85% of 20 games.
- Seven mirrored pairs were search sweeps.
- Three mirrored pairs split.
- Direct policy swept zero pairs.
- There were zero void games and zero void pairs.
- Direct-policy win-rate 95% CI: 5.2% to 36.0%.
- Search used 500ms, P8, one thread, and `c_puct=2.0`.

Result artifact:

```text
experimental/runs/direct_vs_search_gate_probe_v1/results-20.json
```

This H2H compares direct frozen r1 against frozen r1 used as root priors for Foul Play. It does not directly measure the current certified gate against historical search selection. It does establish that search provides a large policy-improvement effect over the direct policy.

The historical 92% GXE deployment was also search-driven:

```text
frozen r1 policy -> root priors -> Foul Play/poke-engine search -> selected action
```

The direct policy did not independently achieve 92% GXE.

## Target Architecture

```text
Public battle history
        |
        v
Belief and opponent model
  P(hidden team, set, Tera, style | history)
        |
        v
Weighted particle population
        |
        v
Public-belief game tree
  + simultaneous-action regret solving
  + exact Pokemon simulator
  + chance enumeration or stratified sampling
  + policy priors at every node
  + counterfactual value network at leaves
        |
        v
Search mixed strategy
        |
        v
Independent verification re-search
        |
        v
Final action sampled from search strategy
```

This architecture combines the strongest applicable ideas from Student of Games, ReBeL, DeepStack, Expert Iteration, AlphaZero, MuZero Reanalyze, KataGo, and PSRO-style population learning.

## Current Architecture Limitations

The existing stack already demonstrates the value of search, but it leaves major gains available.

### Root-Only Priors

Metamon priors are supplied only at the root. Deeper poke-engine nodes use ordinary UCB1 without neural policy guidance.

Relevant implementation:

```text
srcs/vendor/poke-engine/src/mcts.rs
srcs/vendor/poke-engine/poke-engine-py/src/lib.rs
srcs/metagross/run_foul_play.py
srcs/metagross/prior_server.py
```

### Independent Determinizations

Foul Play samples complete hidden worlds and searches each as perfect information. Aggregating independent world recommendations can create strategy fusion: different hidden worlds recommend incompatible actions even though the player cannot distinguish those worlds.

### Uniform World Weights

Production hidden-world candidates are generally assigned equal weights rather than maintaining a persistent posterior updated throughout the battle.

### Decoupled Simultaneous MCTS

The current engine independently maximizes marginal UCB or PUCT statistics for each side and indexes the selected action pair. This is not minimax, Nash, expectiminimax, or regret-minimizing search.

### Hand-Written Leaf Values

Production Gen 9 MCTS evaluates nonterminal leaves with a hand-written material and board-state heuristic. Metamon's critic is not used by the live prior server.

### Policy-Default Controller

The current certified controller usually falls back to direct policy when search and policy disagree. The H2H result shows that this default discards a large amount of search value.

## Public-Belief Search

The primary search should operate on one shared public-belief tree rather than independent perfect-information trees.

A public-belief node should contain:

- Public battle state and action-observation history.
- Our known private state.
- A posterior over opponent hidden worlds.
- Reach probabilities or particle weights.
- Opponent policy or style hypotheses.
- Search regrets, strategy sums, and counterfactual values.

The strategy must be shared across hidden particles that are indistinguishable to the player. Hidden particles may influence expected values, but they must not independently select our action.

Student of Games uses growing-tree counterfactual regret minimization and counterfactual value-and-policy networks. ReBeL performs search from public-belief states and trains value and policy networks from the resulting subgame solutions. DeepStack uses continual re-solving with counterfactual value vectors.

Pokemon's information-state space is too large to enumerate. A practical implementation therefore requires particle or compressed-belief approximations. This loses the clean asymptotic guarantees of exact public-belief solving but remains structurally preferable to independent determinization.

## Simultaneous-Action Solving

Each simultaneous turn should estimate a payoff tensor:

```text
Q[our_action, opponent_action, belief_particle]
```

The posterior-weighted action matrix should be solved with game-theoretic updates such as:

- Regret Matching+ at the root.
- Growing-tree CFR for deeper public states.
- Monte Carlo CFR when full traversal is too expensive.
- Mixed strategies rather than deterministic marginal maxima.
- Neural policy priors as warm starts and expansion guidance.

The experimental repository already contains a shared information-set RM+ implementation:

```text
experimental/engine/pe_v3_learned_priors/src/shared_information_set.rs
```

This is the strongest starting point for a vertical slice. It needs to become particle-weighted, recursive, persistent across turns, and integrated without monkey patches.

## Exact Dynamics and Chance

The real simulator should remain authoritative. A learned MuZero-style dynamics model would introduce unnecessary model error because poke-engine already implements the game rules.

MuZero ideas remain useful for:

- Search-generated policy targets.
- Search-generated value targets.
- Replay-buffer reanalysis.
- Distributional value prediction.
- Batched neural inference.
- Prioritized replay.

Chance should be handled by fidelity level:

- Enumerate high-impact or near-root outcomes exactly when feasible.
- Use stratified or quasi-Monte Carlo sampling deeper in the tree.
- Use common random numbers when comparing root actions.
- Preserve exact terminal values.
- Track uncertainty from finite chance sampling separately from belief uncertainty.

## Neural Architecture

Use a large shared history-and-belief transformer with specialized heads.

| Head | Purpose |
| --- | --- |
| Player policy | Priors at every search node |
| Opponent policy | `P(action | history, candidate hidden set, opponent style)` |
| Counterfactual value | Values for hidden-state particles or compressed information-state groups |
| Belief model | Posterior over sets, items, abilities, moves, Tera, EVs, and teammates |
| Distributional value | Win/loss distribution and calibrated uncertainty |
| Auxiliary outcomes | Remaining HP, faint counts, hazards, status, Tera, next reveal, and turns to terminal |
| Search confidence | Predict where additional simulations have highest value |

The counterfactual value head is particularly important. It should replace the current hand-written leaf evaluator and return values suitable for the public-belief solver, not just one perfect-information scalar.

Use an ensemble or equivalent uncertainty-aware model so search can distinguish:

- Expected game value.
- Aleatoric uncertainty from Pokemon randomness.
- Epistemic uncertainty from unfamiliar states.
- Posterior uncertainty over hidden teams and sets.

KataGo's auxiliary ownership and score targets provide a useful lesson: predicting informative subcomponents of the terminal result greatly improves credit assignment. Pokemon has many exact simulator-derived auxiliary targets that can play the same role.

## Persistent Belief Model

Maintain particles for:

```text
team composition
movesets
items
abilities
nature and EV hypotheses
Tera type
opponent policy or style latent
```

Update particle weights using:

- Exact compatibility with public events.
- Damage likelihood.
- Speed-order likelihood.
- Move, item, ability, and Tera revelations.
- Team-generator and usage priors.
- Candidate-conditioned opponent action likelihood.
- Resampling and rejuvenation when effective particle count collapses.

Production opponent priors currently rely on a flipped-view approximation and are often unavailable until six Pokemon and four active moves are public. The replacement should model opponent actions conditional on each candidate hidden set.

Use a distributionally robust ambiguity set around the posterior. This reduces the chance that search exploits a small belief-model error by choosing an action that is excellent under the estimated posterior but catastrophic under a plausible alternative.

## Expert Iteration Flywheel

1. Bootstrap from the existing Metamon human, offline-RL, and synthetic datasets.
2. Generate complete games using search on every decision.
3. Record the complete search strategy, counterfactual action values, root belief, particles, and final outcome.
4. Train policy heads on soft search strategies rather than selected actions.
5. Train value heads on terminal outcomes and high-budget counterfactual search targets.
6. Use current deployment-policy roll-ins so training covers states the current agent actually reaches.
7. Reanalyze old positions with the newest network and a larger search budget.
8. Prioritize policy-search disagreements, value surprises, catastrophes, unusual sets, and exploiter-discovered states.
9. Promote complete search agents, not direct policy checkpoints.

Expert Iteration separates planning and generalization: tree search acts as the expert, while the network generalizes search results and improves future search. Its tree-policy targets were stronger than chosen-action targets despite similar move-prediction accuracy.

The direct network remains important. Its roles are:

- Search prior.
- Leaf value approximation.
- Opponent prediction.
- Belief inference.
- Search warm start.
- Emergency fallback when search infrastructure cannot execute safely.

It should not normally override completed search.

## Search Target Design

Do not distill only the final selected action.

Preferred targets include:

- Regret-average mixed strategy.
- Regularized improved policy computed from search Q-values.
- Counterfactual action values.
- Action gaps or cost-sensitive losses.
- Search uncertainty and visit allocation.
- Candidate-specific catastrophe distributions.

Policy distillation literature shows that KL targets over a sharpened teacher distribution can outperform one-hot action imitation and raw value regression. KataGo also shows that exploration visits should be pruned from policy targets when they do not represent genuine policy improvement.

Gumbel AlphaZero and regularized-policy-optimization interpretations of MCTS provide principled alternatives to raw visit-count targets, especially when not all actions receive equal search effort.

## Reanalysis

Retain a replay buffer containing exact observations, masks, beliefs, particles, search statistics, engine provenance, and outcomes.

Run a continuous reanalysis service that:

- Rebuilds beliefs with the latest belief model.
- Re-runs search with the latest policy and value network.
- Uses more computation than original self-play when valuable.
- Refreshes policy and value targets.
- Prioritizes states with large target drift.
- Preserves original and reanalyzed targets for auditability.

MuZero Reanalyze used fresh search policies for most updates and substantially improved sample efficiency. Our repository retains many of the required artifacts but does not yet have a replay-buffer reanalysis service.

## League and Population Training

Naive self-play can overfit to itself and forget strategically important responses. The Metamon paper observed that self-play fine-tuning improved self-play results but transferred inconsistently to humans because the model learned to expect its own policy.

Maintain a league containing:

- Current main search agents.
- Historical main agents.
- Main exploiters targeting current agents.
- League exploiters targeting the archive.
- Goal-conditioned exploiters targeting specific tactical weaknesses.
- Human imitation policies.
- Search variants with different risk assumptions.
- Adversarial set and team distributions.
- Public ladder agents when reproducible implementations are available.

Use PFSP or a PSRO meta-strategy to emphasize opponents near the learning frontier, forgotten weaknesses, and strategically important counter-policies. Pokemon strategies are likely non-transitive, so one scalar Elo ordering is insufficient for population construction.

The deployed policy can expose a strategy latent, but the deployed search should optimize against the inferred opponent and robust population rather than blindly selecting one style.

## Verification Re-Search

The verifier should not decide whether search is allowed to improve the policy.

Use two search stages:

```text
Primary search
  -> mixed strategy and top candidate actions

Independent verification search
  -> disjoint particles, chance tapes, and larger tactical budget
```

If the searches disagree:

- Allocate more computation.
- Add counterexample particles or action responses to the primary search.
- Re-solve the public subgame.
- Select a robust search action.

Do not blindly fall back to the direct policy. The verifier should trigger more search, not suppress search.

Common random numbers should be used across root actions so each action is compared under the same hidden particles, opponent responses, and chance tapes. The existing paired holdout engine already implements part of this variance-reduction approach.

The verifier remains useful for:

- Integrity and provenance failures.
- Illegal or impossible actions.
- Guaranteed no-ops.
- Forced and terminal actions.
- Independent catastrophic-risk estimation.
- Shadow auditing and regression detection.

## Search Improvements Before Full SOG

The following improvements can deliver value before a complete public-belief Student-of-Games implementation:

1. Restore search-first action selection with objective deterministic safeguards.
2. Put neural priors at every MCTS node.
3. Replace the hand evaluator with a batched learned value ensemble.
4. Replace uniform world weights with a persistent posterior.
5. Replace decoupled simultaneous UCB with particle-weighted RM+.
6. Share one information-set root across hidden worlds.
7. Reuse trees and beliefs after observed turns.
8. Evaluate root actions with common particles and chance tapes.
9. Use Gumbel sequential halving or regularized policy optimization for search allocation.
10. Train on completed Q-values and mixed strategies rather than raw visits alone.
11. Batch thousands of leaf evaluations across searches and games on accelerators.
12. Use exact iteration budgets for experiments so wall-clock contention does not confound comparisons.

## Speculative Extensions

These are promising only after the sound search-and-learning loop works:

- Learn simulation allocation and backup rules using MCTSnet-style meta-learning.
- Train goal-conditioned tactical exploiters.
- Add test-time opponent adaptation with an exploitability budget.
- Use a learned fast search as a proposal generator and a slower exact-simulator search as adjudicator.
- Precompute reusable finite-state policy graphs for frequent belief states.
- Search over an ensemble posterior rather than averaging network logits.
- Train a search-confidence head to allocate computation according to expected value of information.

MCTSnet demonstrated that learned simulation policies, vector search memories, and learned backups can outperform handcrafted MCTS in a smaller planning domain. Applying it to Pokemon is substantially more speculative than Expert Iteration or public-belief re-solving.

## Proposed Build Order

1. Archive and audit the completed 17-3 H2H as the search-first baseline evidence.
2. Restore historical search-first action selection while retaining objective deterministic safeguards.
3. Promote the experimental shared-information-set RM+ root into a production-quality particle-weighted simultaneous solver.
4. Add persistent hidden-world beliefs and candidate-conditioned opponent priors.
5. Add neural guidance at every node and train a counterfactual value-and-policy network.
6. Build asynchronous Expert Iteration data generation and training.
7. Add replay-buffer reanalysis.
8. Extend root solving recursively into public-belief GT-CFR continual re-solving.
9. Add PSRO or PFSP league training and goal-conditioned exploiters.
10. Add independent verification re-search and distributionally robust beliefs.

Each stage should be evaluated against the previous complete search agent with mirrored teams, fixed iteration budgets, common provenance, and intention-to-treat accounting.

## Time-Optimized Implementation Plan

### Operating Principle

Optimize for reliable information gained per wall-clock hour, not number of experiments launched.

Use one narrow vertical slice at a time. Every stage follows the same funnel:

```text
static and unit checks
        -> deterministic offline replay
        -> fixed-root mechanism test
        -> four-game local smoke
        -> small predeclared mirrored screen
        -> powered H2H only if promising
        -> bounded ladder canary only after local evidence
```

Cheap failures stop expensive work. No stage may start a large collection, training run, H2H, or ladder block merely because its code compiles.

### Critical Path

| Order | Deliverable | Cheapest decisive evidence | Expensive work unlocked |
| ---: | --- | --- | --- |
| 1 | Search-first controller | Exact replay plus existing 17-3 endpoint H2H | Three-game operational canary |
| 2 | Particle-weighted RM+ shared root | Toy exploitability and frozen-root bundles | Small mirrored H2H |
| 3 | Persistent randbats posterior | Held-out calibration and particle invariants | Belief-aware H2H |
| 4 | Learned counterfactual leaf value | Held-out calibration and fixed-root search value | Search-agent H2H |
| 5 | Expert Iteration pilot | Data-contract smoke and one small training proof | 5k strict-battle pilot |
| 6 | Reanalysis and league | Accepted single-generation gain | Multi-generation scaling |
| 7 | Recursive public-belief GT-CFR | Root solver and learned leaves already promoted | Full Student-of-Games program |

The critical path intentionally defers recursive GT-CFR, MCTSnet, a learned verifier, and a large league. Those components have poor information-per-hour until search-first selection, shared-root solving, beliefs, and learned leaves each show independent value.

## Stage 0: Freeze Baseline And Reuse Existing Evidence

Do not rerun work that has already answered the question.

Freeze the following baseline artifacts:

- Direct r1 versus root-prior search H2H: 3-17 from the direct-policy perspective.
- Search configuration: 500ms, P8, one thread, `c_puct=2.0`.
- Frozen r1 policy SHA-256: `c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93`.
- Existing 664-decision audited block corpus.
- Existing teacher-root bundles and exact seeded engine fixtures.
- Latest accepted production and engine source hashes from the relevant manifest.

Required output:

```text
baseline manifest
17-3 H2H result digest
capture list and digests
exact test commands
accepted rollback configuration
```

Gate:

- Every referenced artifact exists and its hash is recorded.
- The 20-game H2H has 20 decisive games, 10 complete mirrored pairs, and zero voids.
- No new baseline games are run unless an artifact cannot be reproduced or the deployed budget changes.

Failure action: repair provenance only. Do not begin architecture changes with an ambiguous baseline.

## Stage 1: Restore Search-First Selection

This is the fastest high-value implementation because the strength mechanism is already demonstrated.

### Scope

- Add one explicit search-first controller through the existing `DecisionHarness` controller seam.
- Make the valid search result the default action on policy-search disagreement.
- Preserve request legality, forced actions, guaranteed no-op prevention, terminal corrections, and hard protocol failures.
- Remove positive-superiority certification from the action-selection path.
- Retain the current verifier in shadow mode for telemetry and independent risk analysis.
- Preserve the old controller only as a frozen A/B comparator while the decision is being validated.
- Avoid adding another selector monkey patch.

### Unit Tests

Add a table-driven selector test covering:

- Policy and search agree.
- Search and policy disagree with a valid search action.
- Search action is illegal under the current request.
- Forced move or forced switch.
- Guaranteed no-op search action.
- Terminal correction.
- Search timeout, malformed result, missing result, and provenance mismatch.
- Tera and non-Tera canonical-action mapping.
- Old certification says reject but search-first still plays a valid search action.
- Telemetry reason exactly identifies search selection, deterministic correction, or infrastructure fallback.

Required invariants:

- A valid search result is never replaced by direct policy merely for lacking positive-superiority evidence.
- Every emitted action is request-valid.
- Infrastructure failures fail closed under the declared fallback contract.
- Deterministic safeguards do not use hidden information unavailable to the player.

### Offline Replay Tests

Replay all captured block decisions through old and new controllers without running new battles.

Report:

- Total decisions.
- Policy-search disagreements.
- Actions changed by the new controller.
- Actions unchanged because policy and search agree.
- Deterministic safeguard activations.
- Illegal or unmappable outputs.
- Exact old and new action for every delta.

Gate:

- 100% decision joins.
- 100% canonical-action mappings.
- Zero illegal outputs.
- Zero unexplained action changes outside policy-search disagreements and declared safeguards.
- All historical forced-action and terminal fixtures remain unchanged.

### Fixed-Root Tests

Use retained exact root bundles to prove that search-first selection returns the canonical search choice for every valid bundle. Re-run the same bundle twice with the same seed and exact iteration count.

Gate:

- Byte-identical result artifacts across deterministic repeats.
- Search choice matches the frozen search baseline on all roots where no new hard safeguard applies.
- Every safeguard-induced difference is listed and manually auditable.

### Local Smoke

Run four mirrored local games with candidate and comparator occupying both challenger and acceptor roles.

Gate:

- Four completed games and two complete mirrored pairs.
- Zero voids, unknown winners, timer failures, or identity mismatches.
- Every outbound command joins to exactly one logged decision.
- Prior server and search are required and demonstrably active.

This smoke tests execution only. Win rate has no authority.

### Developmental H2H Screen

Run one fixed-N 20-game mirrored screen against the current gate at the same 500ms/P8 budget. Do not repeatedly peek at an ordinary interval.

Decision rule:

- Candidate wins 12 or more: promising; proceed to the operational canary.
- Candidate wins 8 or fewer: kill or debug before more games.
- Candidate wins 9-11: inspect predeclared mechanism metrics once; extend to fixed N=40 only if search activation, latency, and action mapping are all healthy.

This is a development screen, not a promotion claim. The existing 17-3 direct-versus-search result supplies separate endpoint evidence, but it does not replace exact candidate-versus-current-gate evaluation.

### Exit

Stage 1 exits only after unit, replay, fixed-root, local smoke, developmental H2H, and `jeanfan29` canary gates pass. Search-first can then become the candidate baseline for Stage 2. A historical-best or promotion claim still requires the formal promotion standard below.

## Stage 2: Particle-Weighted Shared RM+ Root

### Scope

- Move the existing experimental shared-information-set root behind a first-party engine API.
- Accept normalized particle weights.
- Build one shared own-action support across hidden worlds.
- Maintain world-specific opponent action support where required.
- Solve the posterior-weighted simultaneous root with RM+.
- Return a mixed strategy, counterfactual values, exploitability diagnostics, iterations, and provenance.
- Keep deeper search unchanged for this stage so the root solver is the only variable.

### Engine Tests

Test exact matrix games before Pokemon states:

- Rock-paper-scissors converges toward uniform play.
- Matching pennies converges toward uniform play.
- A dominated action receives asymptotically negligible probability.
- A pure saddle point converges to the correct action pair.
- Regret and exploitability decline as iterations increase under exact deterministic payoffs.

Test Pokemon integration invariants:

- Particle-order permutation invariance.
- Duplicating one particle and splitting its weight leaves the result unchanged within tolerance.
- Zero-weight particles have no effect.
- Identical particles reduce to the corresponding single-world result.
- Illegal actions receive zero probability.
- Strategy probabilities are finite, non-negative, and sum to one.
- Same seed and exact iterations reproduce the same artifact.
- Publicly indistinguishable particles cannot select different own policies.

### Fixed-Root Gate

Use a frozen panel containing:

- Policy-search agreement roots.
- High cross-world disagreement roots.
- Switch-heavy roots.
- Tera roots.
- Forced-action roots.
- Known lock-in and no-op pathology roots.

Pass criteria:

- Zero legality, normalization, or reproducibility failures.
- Lower exploitability than decoupled marginal UCB on toy games.
- Stable root strategy under 2x iterations on at least 95% of panel roots, using a predeclared total-variation threshold.
- No unexplained collapse to direct-policy argmax.

### Game Gate

Use the same four-game smoke and fixed-N 20-game developmental screen. Run N=100 only if the 20-game point estimate is positive and mechanism diagnostics are healthy. Kill immediately on any strategy-fusion leak, non-finite regret, or action-support mismatch.

### Canary

Run the three-game `jeanfan29` canary only after the N=100 screen is positive. Root-solver research should not consume public ladder games before controlled evidence exists.

## Stage 3: Persistent Randbats Beliefs

Start with deterministic and generator-derived inference. Do not train a belief network until the exact public-generator baseline is correct.

### Scope

- Persist particles across turns.
- Initialize from the public random-battle generator distribution.
- Filter impossible teams and sets from public events.
- Apply damage and speed-order likelihoods.
- Normalize weights and track effective sample size.
- Resample deterministically under a recorded seed.
- Add candidate-conditioned opponent-action likelihood only as a separate later ablation.

### Unit And Property Tests

- Initial weights sum to one.
- Impossible revealed moves, items, abilities, and species receive zero posterior mass.
- Damage ranges and speed order never increase incompatible particle weight.
- Equivalent event orderings produce equivalent posteriors when the public information is equivalent.
- Resampling preserves normalized mass and is reproducible under seed.
- Effective sample size is correct on uniform and degenerate fixtures.
- Rejuvenation never creates a world inconsistent with public history.
- Empty posterior fails closed and emits a diagnostic rather than silently becoming uniform.
- No opponent-private ground truth reaches the deployed policy or search API.

### Calibration Gate

Use held-out complete randbats replays and reconstruct the posterior at multiple turns.

Report by turn and reveal count:

- Negative log likelihood of the true completed set.
- Brier score.
- Expected calibration error.
- Top-k coverage.
- Effective sample size.
- Fraction of empty or repaired posteriors.

Pass criteria:

- Zero impossible-world mass after definitive reveals.
- No calibration regression versus the exact generator and compatibility-filter baseline.
- Better likelihood or top-k coverage before full reveal.
- No hidden-truth leakage under an explicit adversarial test.

### Downstream Gate

Only after calibration passes, run the frozen root panel with uniform versus posterior weights. Advance to games only if posterior weighting changes uncertain roots in a stable direction under held-out completed-world evaluation.

Run N=20, then N=100 only if promising. Do not collect large self-play data for a belief model that has not improved held-out calibration and fixed-root decisions.

## Stage 4: Learned Counterfactual Leaf Values

### Scope

- Start with the smallest network that can batch efficiently.
- Keep the exact simulator and shared-root solver fixed.
- Train a value distribution plus policy and selected auxiliary heads.
- Evaluate the learned value only at leaves; do not simultaneously change search allocation or beliefs.
- Preserve the hand evaluator as a frozen comparator and optional bounded blend during development.

### Data Tests

- Exact schema and feature-version validation.
- Train, validation, root-panel, H2H, and ladder splits are disjoint by source battle.
- No future public event or hidden opponent truth appears in deployed inputs.
- Perspective flips preserve the declared zero-sum target transformation.
- Terminal targets equal exact game outcomes.
- Legal masks match the prior-server mapping.
- Every dataset shard has immutable source, engine, model, and transform hashes.

### Model Tests

- Output values are finite and in the declared support.
- Perspective and side-swap metamorphic tests pass.
- Batched and single-example inference agree within tolerance.
- Saved and reloaded checkpoints produce identical outputs.
- Calibration metrics are reproducible from the frozen validation manifest.
- Search never queries a model with an incompatible feature schema.

### Offline Gate

Report:

- Log loss and Brier score.
- Reliability curves by turn and advantage bucket.
- Error on terminal-near states.
- Error by hidden-state entropy.
- Search-root ranking accuracy against high-budget independent values.
- Inference throughput and p50/p95 batch latency.

The model advances only if it improves high-budget root ranking or calibrated value error over the hand evaluator. Lower training loss alone has no authority.

### Game Gate

Run four-game smoke, fixed-N 20 screen, then N=100. A formal promotion round uses at least 500 paired games and the external opponent panel. More epochs are not a response to failed H2H; first audit value targets, calibration, and search interaction.

## Stage 5: Expert Iteration Pilot

### Mechanical Smoke

Before generating 5k strict battles:

- Generate 32 full search-guided battles.
- Parse both POV trajectories where expected.
- Build one training shard.
- Train for a small fixed number of steps.
- Reload the checkpoint through the real prior-server and search path.
- Reanalyze a small frozen sample.

Gate:

- 100% manifest and hash coverage.
- Zero train/evaluation contamination.
- Zero missing legal masks or malformed targets.
- Search-policy targets normalize correctly.
- Reanalysis is deterministic under the frozen configuration.
- The checkpoint loads through deployment without key or schema mismatch.

### Proof-Of-Signal Run

Use at most 1k battles to verify that the student moves toward soft search targets without regressing the fixed human anchor. This run cannot promote a checkpoint.

Pass criteria:

- Search-policy KL improves on held-out strict roots.
- Human-anchor KL remains inside its predeclared bound.
- Value calibration does not materially regress.
- Direct-policy and search-agent diagnostics remain finite and reproducible.

### Strict Pilot

Only then run the existing 5k strict-battle pilot with a 90/10 search/human mix, one to two epochs, and the current accepted search agent as parent. The promotion round remains at least 25k strict full battles and is not started until the pilot passes paired H2H.

## Stage 6: Reanalysis And League

Implement reanalysis before scaling raw collection because it extracts more value from already expensive search games.

Reanalysis tests:

- Old and refreshed targets remain separately auditable.
- Latest search cannot read holdout outcomes.
- Priority computation is deterministic.
- Reanalysis cannot silently cross feature or engine versions.
- Replay sampling weights are normalized and logged.

League tests:

- Opponent sampling matches the versioned PFSP distribution.
- Held-out opponent instances never enter training.
- Historical checkpoints remain immutable.
- Exploiters cannot overwrite the accepted lineage.
- Meta-game payoffs include uncertainty and complete identity manifests.

Do not launch a large league until one single-generation Expert Iteration candidate passes the formal search-agent promotion gate.

## Stage 7: Recursive Public-Belief GT-CFR

This stage begins only after shared-root RM+, persistent beliefs, and learned counterfactual values have independently passed.

First implementation slice:

- Depth-one public-belief continuation after the simultaneous root.
- Exact public observation update.
- Leaf counterfactual value query.
- Warm-started regrets from policy priors.
- Tree reuse after the observed turn.

Tests:

- Tiny fully enumerable games match exact CFR values.
- Public-belief updates match brute-force Bayes fixtures.
- Increasing iterations lowers exploitability in toy games.
- Increasing depth converges toward an exact small Pokemon subgame fixture.
- Tree reuse and fresh solve agree within tolerance.
- No private-state-conditioned own policy appears at shared information sets.

Expand depth only after the previous depth shows a stable fixed-root and H2H gain. Do not build the full recursive system in one pass.

## Quality Gate Matrix

| Gate | Cost | Required result | Failure action |
| --- | --- | --- | --- |
| Static | Seconds | Compile, format, `git diff --check`, schema checks | Fix before tests |
| Unit/property | Minutes | Zero failures; deterministic seeds | Fix before replay |
| Offline replay | Minutes | 100% joins, zero illegal actions, explained deltas | Kill behavior change |
| Fixed roots | Minutes | Reproducible mechanism and stability | Tune only declared mechanism |
| Four-game smoke | Small | 4/4 complete, zero voids | Fix infrastructure |
| N=20 screen | Moderate | Positive predeclared development signal | Kill, debug, or one fixed extension |
| N=100 screen | Higher | Positive effect and healthy mechanism metrics | Do not canary later stages |
| `jeanfan29` canary | Bounded | Operational integrity, not significance | Immediate rollback on attributable failure |
| N>=500 promotion | Expensive | CI and panel gates pass | Archive candidate |
| Ladder block | Most variable | Secondary population confirmation | No historical-best claim |

## Statistical Discipline

- Use complete mirrored pair or evaluation block as the independent unit.
- Use fixed N for the first implementation stages; do not build new sequential-statistics infrastructure merely to save one small run.
- If repeated architecture experiments make early stopping valuable, implement one tested confidence-sequence or e-value gate and use it consistently.
- Never repeatedly inspect an ordinary confidence interval and stop when it looks favorable.
- Treat candidate-attributable crashes, timeouts, and malformed actions as candidate non-wins.
- Use intention-to-treat results for formal decisions.
- Predeclare practical improvement and non-inferiority margins before N>=500 promotion runs.
- Report effect sizes, pair sweeps/splits, role balance, voids, latency, and intervals.

## `jeanfan29` Canary Protocol

### Account Selection

Resolve "latest jeanfan account" by the most recent successfully completed and audited manifest timestamp, not by the largest numeric suffix.

At the time of this plan, the latest successful audited account is exactly:

```text
jeanfan29
```

Do not use the misspelled historical `jenfan29` path or username.

Last captured account state on 2026-08-08:

| Metric | Value |
| --- | ---: |
| Wins | 24 |
| Losses | 11 |
| Elo | 1354.66 |
| GXE | 67.0 |
| Rating deviation | 55.77 |

Source:

```text
srcs/runtime/modal-v5-r1-qualified-safeguards-block-jeanfan29/
  20260808T030147Z-r1-jeanfan29-47498/ratings.jsonl
```

This state is reference evidence only. Query and record live account state immediately before every canary because the ladder may change.

### Preflight

- Confirm no other ladder process owns the account lock.
- Confirm exact username, format, and credentials without writing secrets to logs.
- Freeze and hash policy, engine, source tree, randbats dataset, and configuration.
- Verify prior and search preflights against the exact deployment contract.
- Record live Elo, GXE, W/L, and RD.
- Run one local request-to-command smoke with the exact deployment environment.
- Set automatic continuation to disabled.
- Set a hard cap of three games and a bounded runtime.

### Execution

Run exactly three `gen9randombattle` games or stop earlier on a quality failure. Capture protocol, decisions, search results, beliefs when enabled, ratings, source manifest, and process logs.

Do not inspect W/L to decide whether to continue. The three-game canary is an operational test, not a strength estimate.

### Canary Pass

- Three of three target games complete unless an opponent-side legitimate interruption is explicitly audited.
- Zero candidate-attributable voids, forfeits, timer losses, or protocol stalls.
- 100% decisions reconstruct from protocol.
- 100% selected actions match outbound commands.
- Zero illegal or unmappable actions.
- Zero silent policy fallback when search is required.
- Every fallback, safeguard, and correction has an explicit reason.
- Engine, model, and source hashes match the frozen manifest.
- Candidate p95 end-to-end decision latency is no worse than the predeclared baseline tolerance and no decision approaches the Showdown timer limit.

Normal combat losses do not fail an operational canary. One implementation-attributable loss, illegal action, identity mismatch, unexplained fallback, or integrity failure causes immediate rollback.

### After Canary

- Stop automatically regardless of result.
- Run the existing canary audit and shadow replay.
- Publish a compact audit before any additional ladder game.
- Return to controlled local evaluation.
- Run a larger bounded ladder block only after the formal H2H gate passes.

Because `jeanfan29` is reused and already has ladder history, it cannot establish a clean candidate-versus-parent ladder improvement. It is appropriate for operational canaries. A historical-best claim still requires a fresh or properly interleaved account design.

## Latency And Resource Gates

Time is both an engineering constraint and part of the deployed agent definition.

- Mechanism tests use exact iterations and fixed seeds.
- Deployment comparisons use identical hardware and per-decision latency contracts.
- Record inference, belief update, search, verification, and total decision latency separately.
- Compute baseline p50, p95, p99, and max from the latest valid capture before setting candidate tolerances.
- Reject any candidate that gains strength only by violating the deployment timer or using unreported compute.
- Batch neural leaf evaluations and independent games rather than reducing correctness checks.
- Cache immutable model loads, generator data, and root bundles across tests.
- Do not cache mutable battle beliefs across process or battle identity boundaries.

The latest `jeanfan29` block observed search RPC mean latency around one second and occasional multi-second holdout calls. Search-first Stage 1 should remove synchronous positive-admission holdouts from the action path, so it should improve rather than worsen live latency.

## Test Commands

Use repository-native suites instead of introducing a second test runner for the same layer.

Production Python suite:

```bash
.venv-fp-priors/bin/python -m unittest discover \
  -s srcs/metagross/tests \
  -p 'test_*.py'
```

Focused experimental Python suites:

```bash
.venv/bin/python -m pytest \
  experimental/src/search/tests \
  experimental/src/belief/tests \
  experimental/src/eval/tests
```

Engine tests:

```bash
cargo test \
  --manifest-path experimental/engine/pe_v3_learned_priors/Cargo.toml
```

Python binding tests when the engine API changes:

```bash
.venv/bin/python -m pytest \
  experimental/engine/pe_v3_learned_priors/poke-engine-py/python/tests
```

Every stage should add focused tests next to the modified layer. Run the focused suite during development and all affected production, experimental, and engine suites once before its first H2H.

## Engineering Quality Rules

- One behavior variable per experiment.
- One first-party controller or solver seam; no new patch stack.
- Every candidate has a feature flag and one-command rollback to the accepted stack.
- Never overwrite accepted checkpoints or manifests.
- Every output includes source, model, engine, dataset, seed, and configuration hashes.
- No test or training process may read formal holdout outcomes.
- All random behavior accepts and logs an explicit seed.
- All empty, malformed, non-finite, and incompatible states fail closed with a typed reason.
- New telemetry fields are schema-versioned and tested for replay compatibility.
- Unrelated worktree changes are left untouched.
- No large collection or training starts without a written gate result from the previous stage.

## Stop And Pivot Rules

| Evidence | Decision |
| --- | --- |
| Search-first fails exact replay or smoke | Fix controller; do not revisit verifier thresholds |
| Search-first is healthy but loses the developmental screen | Audit exact current-gate versus search choices once; do not start RM+ |
| RM+ fails toy exploitability | Fix solver mathematics before Pokemon integration |
| RM+ passes toys but is unstable on roots | Improve payoff estimation or particle support |
| Beliefs fail calibration | Keep generator baseline; do not train an action model |
| Learned leaf improves offline loss but not root ranking | Reject target/model recipe before H2H |
| Learned leaf improves roots but not games | Audit search interaction and calibration, not epochs |
| Student learns search targets but search agent does not improve | Improve value/policy coupling or data distribution |
| Self-play improves only against itself | Add historical and goal-conditioned exploiters before scaling |
| One stage fails twice after one declared fix | Stop that branch and return to the last accepted stack |

## Required Stage Artifact

Each stage ends with one compact, immutable report containing:

```text
objective and one changed variable
candidate and parent manifests
test commands and results
offline and fixed-root metrics
H2H schedule and result
latency and resource metrics
canary audit when applicable
pass, kill, or inconclusive decision
next permitted action
```

This report is the authorization boundary for expensive work. An undocumented verbal conclusion does not unlock the next stage.

## Promotion Standard

A candidate network or solver should be promoted only as part of the complete deployed search stack.

Required evidence should include:

- Exact fixed-budget mirrored H2H against the previous search agent.
- Pair-level outcomes and confidence intervals.
- Zero unexplained voids or integrity failures.
- Search scaling curves as computation increases.
- Belief calibration on held-out complete battles.
- Value calibration by hidden-state and game phase.
- Exploiter and local-best-response testing.
- Reanalysis stability and target-drift reports.
- A small audited ladder canary followed by a larger rating-band block.

Direct-policy performance remains a useful diagnostic and latency fallback metric, but it is not the promotion objective.

## What Not to Prioritize

Do not prioritize the following before the search-native foundation:

- Training a learned positive-admission verifier.
- Further tuning a policy-default superiority gate.
- Replacing the exact simulator with learned MuZero dynamics.
- Distilling only one-hot search actions.
- Training only against the latest self-play policy.
- Treating independent determinization aggregation as a sound information-set policy.
- Evaluating model checkpoints without the search stack that will deploy them.

## Research Caveats

- No cited system has solved Gen 9 Random Battles with this exact architecture.
- Student of Games and ReBeL have strongest guarantees in two-player zero-sum settings; Pokemon battles fit the zero-sum outcome structure but add enormous hidden-state and chance complexity.
- Particle beliefs, neural values, action pruning, and approximate CFR weaken theoretical guarantees.
- A minimax-robust strategy may sacrifice ladder win rate against weak or biased populations. Safe opponent exploitation should therefore be studied explicitly.
- The 17-3 H2H proves a large search benefit for the current frozen r1 and search stack. It does not prove every proposed architecture component will improve strength.

## Key References

- Anthony, Tian, and Barber. [Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/abs/1705.08439). Expert Iteration, soft tree-policy targets, and distributed search-teacher training.
- Silver et al. [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815). AlphaZero search-based policy iteration.
- Schrittwieser et al. [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265). MuZero and replay-buffer reanalysis.
- Grill et al. [Monte-Carlo Tree Search as Regularized Policy Optimization](https://arxiv.org/abs/2007.12509). MCTS as an improved regularized policy and direct Q-based targets.
- Danihelka et al. [Policy Improvement by Planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO). Sequential halving, completed Q-values, and policy improvement with limited simulations.
- Brown et al. [Combining Deep Reinforcement Learning and Search for Imperfect-Information Games](https://arxiv.org/abs/2007.13544). ReBeL public-belief search and learning.
- Schmid et al. [Student of Games](https://arxiv.org/abs/2112.03178). Growing-tree CFR, sound self-play, recursive query solving, and counterfactual value-and-policy networks.
- Moravcik et al. [DeepStack](https://arxiv.org/abs/1701.01724). Continual re-solving and learned counterfactual values.
- Wu. [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565). KataGo playout-cap randomization, target pruning, and auxiliary targets.
- Rusu et al. [Policy Distillation](https://arxiv.org/abs/1511.06295). Soft KL policy targets and online distillation.
- Bighashdel et al. [Policy Space Response Oracles: A Survey](https://arxiv.org/abs/2403.02227). Population learning, meta-strategy solvers, exploiters, and non-transitivity.
- Guez et al. [Learning to Search with MCTSnets](https://arxiv.org/abs/1802.04697). Learned simulation policies, backups, and search memories.
- Grigsby et al. [Metamon: Human-Level Competitive Pokemon](https://arxiv.org/abs/2504.04395). Pokemon offline RL, synthetic self-play, search comparisons, and self-play transfer limitations.

## Bottom Line

The architecture worth pursuing is not Metamon plus a clever positive-admission verifier. It is:

> Exact-simulator public-belief Student of Games, trained by distributed Expert Iteration and a PSRO-style league, with Metamon transformed into the policy, belief, opponent, and counterfactual-value network that powers search.

This preserves the mechanism that already produced the strongest observed results while creating a path to improve every weak part of the current search stack.
