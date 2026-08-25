# Causal interior prior after Cycle 47

## Decision

Keep causal-history R1 and all production safeguards byte-identical at live roots. Do not train on Cycle47's 444 counterfactual rows: they cover only 8/64 battles and therefore select a mechanically easy subpopulation.

The research-backed architecture remains a small search-native prior, but the data path must change from counterfactual archival stepping to **actual observed sequential decision states**. A genuine later-turn decision already contains the realized opponent action/chance transition, exact own request, full causal public prefix and terminal provenance. It does not need us to reconstruct thousands of hypothetical child transitions with a 5.4% bridge.

## Proposed model (about 2–4M parameters)

- Typed tokens for six own roster slots (private mechanics), six opponent public slots (reveal-mask gated), field/side conditions, active matchup, legal 13-action candidates and chronological causal event tokens.
- 4-layer, width-192 causal/set transformer with separate event-order and roster-slot embeddings, approximately 3M parameters.
- Candidate-action cross-attention head emits masked logits over the exact engine/request action table. No value head in the first experiment.
- At deployment, query only at supported ordinary depth-one nodes. Root R1 prior is unchanged. Any missing typed history, ambiguous mapping, PP/disable authority or unsupported event causes immediate fallback to current interior equal/simulator priors.

## Cycle 48 Gate A2: observed-state target corpus

1. Label-blind select 64 new Cycle12 TRAIN dependency clusters with at least eight chronological ordinary actionable states each; freeze the 512 states before teacher values. Do not use Cycle13's already opened/retired 192 roots.
2. Rematerialize every state twice from its pinned inputlog/Showdown commit. Require exact own request, command-time public prefix, typed ledger, mechanical state, action mask and identity-sanitized full causal fingerprint. Dependency cluster is provenance only.
3. Generate two independent eight-world belief schedules per state. Preserve schedule membership and raw posterior weights; aggregate teacher soft policies only within the same causal information fingerprint and schedule, then compare schedules. Never expose hidden-world state fields to the model.
4. Run equal-prior 8,192 twice and 20,000 once on every admitted observed state. Preserve all legal N/W/Q and null unvisited Q. Human observed action is a behavior anchor, never a strength label; R1 output is a separate root-control anchor.
5. Retain the Cycle47 gates: >=95% coverage, >=512 fingerprints from >=48 battles, >=80% 8k/20k top1, repeat JSD median<=.05/p90<=.15, zero hidden sensitivity/split leakage. Add schedule-half soft-policy JSD and effective sample size reporting.

Cycle47 suggests target stability is plausible (82.66% top1 agreement; repeat JSD p90 .00394) if the observed-state mechanics achieve coverage. Gate A2 must still pass before training.

## Cycle 49 Gate B: tiny CPU learnability

Three dependency-grouped TRAIN seeds. Compare:

- human-anchor only;
- same-teacher one-hot action;
- human anchor plus stable soft 8k search policy.

Require grouped held-out soft-target cross-entropy improvement over one-hot, >=95% exact action-mask fidelity on held-out human/R1 anchors, zero high-confidence illegal action, batch-64 CPU p95 <=5ms, byte-identical root R1 with feature disabled, and no 500ms fallback/timeout increase in integration replay. Offline metrics authorize only a fresh equal-500ms 20-game H2H, never strength or data scale-up.

## Why this is the justified pivot

Expert Iteration supports a slow planner/fast apprentice only after the target is reproducible. Regularized-MCTS work supports soft policies, not one-hot argmaxes. Belief-planning work requires player-information-state inputs, not completed hidden worlds. MegaGem's transferable lesson is procedural: prove the live expert before distillation and preserve pass-through/anchor behavior. Cycle41 rejected recurrent equal-prior root replacement; Cycle47 rejected the current counterfactual archival child bridge. Neither falsifies a small prior that improves fixed-budget allocation only inside the tree, trained from high-coverage observed causal states.
