# Reproducing MegaGem's winning methodology in Metagross

Date: 2026-08-15  
Status: research and experiment specification; no controller or training code changed  
Recommendation: proceed, but reproduce the *sequence of evidence*, not the auction formula

## Executive conclusion

The transferable result from MegaGem is not “train a resource critic” or “add a shadow-price feature.” Metagross has already tested close versions of both ideas and rejected them. The transferable method is:

1. preserve the strongest existing policy as the behavior policy;
2. use abundant local interaction records to construct an interpretable decision-time expert;
3. retain uncertainty as a distribution rather than collapsing it to a point estimate;
4. charge for downstream consequences, then test the complete intervention in live, paired games;
5. change only decisions for which the expert has a conservative advantage;
6. distill only after the expert itself wins, mixing corrections with pass-through and resource-preservation anchors;
7. test the weights-only student again at the original inference budget.

For Metagross, the correct expert is **not** another learned residual. It is the already certified, outcome-grounded long-horizon continuation planner applied selectively to ambiguous live roots. Its terminal continuations price HP, Tera, PP, tempo, information, and switch flexibility through their actual downstream consequences instead of a fitted linear shadow cost. The immediate experiment should therefore compare:

> current causal-history R1 + 500 ms production search

against

> the same controller, with an abstaining top-two long-horizon terminal-continuation override only at ambiguous roots.

This is a teacher-strength test, not yet a production-latency test. If the slow controller cannot win prospectively, stop: there is no justified target to distill. If it wins, export its corrections, pass-throughs, and resource anchors into one conservative expert-iteration round and test the student at the same 500 ms budget as the baseline.

## Research method and source coverage

I used Exa for four workstreams: (1) Alex Wa's post, repository, datasets, and implementation; (2) expert iteration and search distillation; (3) shadow pricing, belief-state planning, and resource option value; and (4) offline-to-online failure, safe improvement, abstention, and paired evaluation.

- `sources_reviewed: 320`, defined exactly as the sum of requested `numResults` over all Exa search calls, including the deliberate second audit pass.
- The audit pass returned 154 URL-bearing records and deduplicated them to 139 unique URLs (15 duplicates removed).
- Ten primary sources were deeply read and used below. Search snippets from derivative summaries were not used as technical evidence.
- The MegaGem repository was also inspected read-only at commit `6e5b082e0718f16351907f9e2e08bce0fddd4459`; this matters because the current repository includes post-publication selector infrastructure beyond the narrative in the August 9 post.

The strongest sources are the author's own [technical post](https://djdumpling.github.io/2026/08/09/megagem.html) and [repository at the inspected commit](https://github.com/djdumpling/MegagemBench/tree/6e5b082e0718f16351907f9e2e08bce0fddd4459). They are first-party practitioner evidence with code, negative results, effect sizes, and released weights. They are not independent replication, and MegaGem is a different game, so its effect sizes must not be carried over to Pokémon.

## What MegaGem actually did

### 1. Establish a competent behavior policy

The base Qwen3-4B-Instruct policy was first trained on 6,153 examples from 150 teacher games, with 420 examples from ten held-out games. Teachers were a 70/30 mixture of Gemini 3 Flash and Claude Opus 4.6. SFT raised the benchmark rating from 551 to 1158. This created a coherent behavior policy whose state distribution was worth improving; the later expert did not have to invent play from scratch. These numbers and the complete pipeline are documented in the [post](https://djdumpling.github.io/2026/08/09/megagem.html#data-and-sft) and [training runbook](https://github.com/djdumpling/MegagemBench/blob/6e5b082e0718f16351907f9e2e08bce0fddd4459/docs/training.md).

### 2. Treat self-play and critic failures as diagnostics, not hidden successes

Seven GRPO configurations were audited. The two configurations measured throughout against the stationary SFT anchor stayed flat; some checkpoints improved substantially against a scripted heuristic, which showed exploitation of that opponent rather than transfer. In the critic probe, only 15.2% of candidate-Q variance separated actions within a state. A 4B critic reached only 24.0%–34.7% full-set best-bid accuracy, below a 40% non-leaky linear-bid baseline, despite confident pair filtering. The author stopped rather than paying the estimated 328 H200-hours for the next scale-up.

This is directly analogous to Metagross's generic value, action-Q, sequential-embedding, and residual failures. It argues against “more RL” until decision-level supervision has independently demonstrated value.

### 3. Reinterpret rollout logs as local market data

MegaGem games produced few noisy terminal labels but thousands of clean local bid observations. The successful branch separated the decision into:

- a value estimate `V-hat` for the lot, using only actor-visible information; and
- an opponent bid distribution `F-hat`, rather than one predicted clearing price.

The bid model was a histogram gradient-boosted regressor on 16 public features, fit to about 3,600 opponent-bid rows with folds grouped by paired seed. It stored out-of-fold residuals and shifted their empirical distribution around the current predicted mean. That construction preserved the probability mass around discrete win/loss thresholds. A separate point-prediction diagnostic achieved 1.10-coin MAE but still put 26.1% of auctions on the wrong side of win/loss; 55% cleared within one coin. Retaining residual uncertainty improved offline regret by another 0.456 coins per decision over the point version. The formula and validation are documented under the post's [bid-distribution model](https://djdumpling.github.io/2026/08/09/megagem.html#bid-distribution-model).

### 4. Let offline success fail live

The first selector maximized immediate expected surplus and overrode the SFT bid only for at least one coin of estimated improvement. It still changed live paired margin by **-5.01 points** with 95% interval `[-10.04, +0.02]`. It bought more valuable gems and completed more missions, but depleted cash, incurred financing costs, and lost investment returns. This is the crucial negative result: a locally calibrated selector was not a full-game improvement.

### 5. Price future flexibility and prove the complete selector

The corrected objective charged each spent coin an additional constant liquidity premium:

`U(b) = ((V-hat - delta) - (1 + lambda_coin) * b) * P(win | b)`.

The author screened 18 pacing/de-bias/gate arms in a simulator, chose `lambda_coin=0.5`, `delta=2`, and gate `1.0`, and then evaluated that frozen choice on 150 paired live seeds. First-place rate moved from 36.0% to 59.3%; paired mean margin improved **+12.16 `[7.61, 16.71]`**. After refitting the opponent price law on about 1,800 SFT-policy bids, the policy-matched selector still improved raw paired margin by **+8.47 `[3.99, 12.94]`**. The current code implements the gate and pacing in the [selector](https://github.com/djdumpling/MegagemBench/blob/6e5b082e0718f16351907f9e2e08bce0fddd4459/src/megagem/environment/ev_selector.py); the published experiment is in [pricing liquidity](https://djdumpling.github.io/2026/08/09/megagem.html#pricing-liquidity).

There is an important scope distinction. The post's successful experiment used a constant premium. The repository now also contains state-dependent pacing schedules. Those later schedules are implementation leads, not evidence from the reported live effect, and should not be silently attributed to the post.

### 6. Distill only the proven deviations

One expert-iteration round generated about 4,700 bid decisions from 400 fresh mixed-policy games, including about 1,200 selector deviations. The export contained:

- expert deviations at selector-controlled treasure decisions;
- pass-through responses when the expert did not establish enough improvement; and
- loan/investment anchors to protect the financing behavior the myopic model had damaged.

Games were split by seed; the post reports a 15% held-out split. The student recovered 0.760 of the expert's held-out utility gain versus 0.316 for a fresh SFT response and moved in the expert direction on 87.8% of bids versus 52.3%. With no selector active, the distilled weights improved paired margin by **+7.97 `[2.70, 13.25]`**; style-matched targets improved it by **+8.31 `[3.21, 13.42]`**. The exact current exporter makes deviations, pass-throughs, and financing anchors explicit in [distill_export.py](https://github.com/djdumpling/MegagemBench/blob/6e5b082e0718f16351907f9e2e08bce0fddd4459/scripts/training/distill_export.py).

The blog describes the chosen mixture and deviation repetition, while the current exporter exposes these as parameters and has defaults that are not necessarily the historical run of record. A reproduction must save the exact command/configuration rather than assuming today's defaults recreate the published checkpoint.

### 7. Evaluate the student, not merely its imitation loss

The final rating used a balanced 624-game, 13-model Steiner triple schedule and Plackett–Luce fit. A separate 450-game top-three panel rotated all seats over 150 independent deal clusters. This is unusually good evaluation hygiene for a game-agent blog post. It still establishes MegaGem performance only, not a general guarantee about this recipe.

## What transfers to Metagross, and what does not

| MegaGem component | Metagross analogue | Transfer decision |
|---|---|---|
| Competent SFT behavior policy | Corrected causal-history R1, 142.8M parameters | Already present; freeze it as the behavior policy |
| Opponent bid distribution `F-hat` | Opponent action distribution from causal R1/search, aggregated over legal belief worlds | Transfer the distributional treatment, not a point opponent move |
| Lot value `V-hat` | Terminal win probability after forcing a candidate action | Use outcome-grounded continuations, not the hand evaluator |
| Empirical residual distribution | Paired action-difference distribution over schedules, worlds, and rollout seeds | Preserve mean, lower tail, disagreement, and paired sign |
| Cash shadow price | Future option value of HP, Tera, PP, tempo, information, and healthy switches | Do **not** fit another static scalar; terminal continuation already prices these jointly |
| One-coin gated selector | Conservative lower-confidence-bound override | Transfer as a fail-closed abstention rule |
| Paired live selector test | Mirrored, seed-paired H2H against current production controller | Transfer exactly, with role/team rotation |
| Deviation/pass-through/financing corpus | Teacher corrections, baseline pass-throughs, and rare resource-state anchors | Transfer after the selector wins |
| Weights-only evaluation | Student inside identical 500 ms production search | Required before deployment |

The closest research foundation is Expert Iteration: a slow planning expert improves states reached by a fast apprentice, and the apprentice then generalizes those plans ([Anthony, Tian, and Barber, 2017](https://proceedings.neurips.cc/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf)). MCTS visit distributions can also be understood as an approximate regularized policy-improvement step rather than arbitrary counts ([Grill et al., 2020](https://proceedings.mlr.press/v119/grill20a/grill20a.pdf)). In partial observability, planning should be over beliefs and observations, not a sampled hidden state exposed to the policy; BetaZero is a useful primary example of learned approximations combined with belief-state planning ([Moss et al., 2024](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_27.pdf)).

The auction-specific pieces do **not** transfer literally:

- Pokémon actions are not an ordered scalar like bids.
- A logged opponent action cannot be replayed counterfactually without changing the next state.
- Action value depends on simultaneous move resolution, hidden sets, speed, damage rolls, switching, and later policy responses.
- Resources interact discontinuously: one HP can preserve a switch, Tera changes typing and move power, and information changes the belief rather than a score component.
- There is no reason to expect MegaGem's `lambda=0.5`, one-unit gate, data ratio, or effect size to apply.

Therefore the long-horizon terminal planner must replace the analytic bid equation. An interpretable resource decomposition can be logged for diagnosis, but it must not determine actions until it independently predicts terminal effects.

## Local evidence that constrains the reproduction

Metagross already contains the following results, all of which must be treated as prior negative evidence rather than repeated:

1. The corrected causal-history R1 path has exact offline/live probability parity and is the accepted policy representation. Its only small live comparison was 11-9 against the legacy stateless path; that is suggestive, not proof of strength.
2. A 23-term resource-shadow leaf covering HP, switch depth, Tera, PP, hazards, items, and tempo failed the equal-depth root gate. Adding causal public-reveal information also failed. Another hand-tuned shadow blend is not justified.
3. The semantically certified 2,048-iteration terminal-continuation teacher **did** pass a disjoint reliability panel: 128 roots, 49,152 matched outcomes, 98.197% terminal coverage, 78.906% half-split agreement, and 11 stable corrections. This establishes that the slow expert can find reproducible mistakes shared by 20k and 50k root search.
4. On the later 105-root development panel, increasing to 16 matched rollouts raised half-split agreement to 77.143%. Seven corrections persisted with mean terminal-win advantage +0.143973. These are opportunity estimates, not deployed improvement.
5. Historical mining produced 55 durable corrections; targeted switch-to-switch collection then produced 56 durable action corrections from 200 roots and 149,248 matched trajectories, with mean advantage +0.155668. Compact, specialist, sequential, generic semantic, and exact candidate-matchup residuals all failed the grouped zero-harm gate. The latest candidate-conditioned model recovered 0/56.
6. Corrected dual-R1 sequential continuation was not usable in the last certification: only 2.16% of trajectories terminated because public-history projection failed. Until a new certificate supersedes that artifact, the valid continuation expert is the exact engine-terminal MCTS teacher, not an asserted R1-on-both-sides rollout.

The conclusion is strong: the project does not mainly lack more correction labels or more root features. It lacks a reliable cheap predictor of when those corrections apply. The next test should use the expensive teacher directly and ask whether its selective decisions improve complete games.

## Precise replication plan

### Phase 0 — Freeze the claim and namespaces

Before a result is opened, write a protocol that freezes:

- baseline artifact hashes: R1 checkpoint, engine binary, production search settings, and public-history schema;
- ambiguous-root trigger;
- candidate-set construction;
- world and rollout seed namespaces;
- continuation policy, iterations, horizon, and termination rules;
- override confidence rule;
- H2H schedule, role/team rotation, early-stop boundaries, and primary estimand;
- a development battle exclusion list and an untouched prospective H2H seed range.

Search/teacher randomness and played-battle randomness must have disjoint namespaces. No H2H outcome may alter a threshold.

### Phase 1 — Build the slow abstaining controller

At each live root:

1. Run the unchanged 500 ms production search and preserve its selected action as `a0`.
2. Trigger only on label-blind ambiguity visible in the existing search output: small top visit/value gap, high cross-world disagreement, or high uncertainty. Freeze the exact thresholds from search telemetry, not teacher labels.
3. Form a maximum two-action set: `a0` and the highest-supported alternative common to the required schedules/world support. Three actions may be used only if preregistered and separately costed.
4. For each candidate, use the same two schedules, eight causal belief worlds, opponent/chance tapes, and rollout indices. The policy never sees the completed hidden team; only the engine inside a sampled world does.
5. After forcing the candidate action, continue with the semantically certified exact MCTS policy at 2,048 iterations per decision, horizon 128, to engine terminal win/loss.
6. Estimate paired action differences, not two independent means. Preserve the distribution across world/schedule clusters: mean, lower tail, schedule-specific sign, half-split ordering, termination, and paired coverage.
7. Override only if all frozen checks pass; otherwise play `a0`.

A defensible initial rule is:

- at least 95% terminal coverage for baseline and alternative;
- at least 90% paired coverage;
- positive effect in both schedules;
- same best action in the two rollout halves;
- one-sided world/schedule-cluster bootstrap lower bound above zero;
- minimum absolute effect of 0.05 for a live override, stricter than the old 0.01 label threshold;
- no action-support, history-projection, or hidden-information exception.

The 0.05 margin is a proposed safety threshold, not a research fact. It must be frozen before prospective play and may be made stricter after a label-blind latency/coverage smoke, never after viewing wins.

An adaptive sample schedule is reasonable but must itself be frozen. Start with two rollouts per world/action. Abstain immediately when the upper bound cannot clear the gate; extend promising unresolved roots to four and then eight. Do not continue sampling merely until the desired action wins.

### Phase 2 — Semantic and mechanical admission

Use opened development artifacts only as an integration suite. They cannot support a new strength claim.

Require:

- exact state/hash/action-support replay for every tested root;
- deterministic reproduction under an identical seed tape;
- zero hidden-team or per-world data crossing the controller boundary;
- at least 95% terminal completion and zero silent fallback;
- exact reproduction of a preregistered subset of existing teacher decisions;
- bounded intervention rate, provisionally 2%–30% of ordinary decisions;
- latency and iteration accounting per intervention.

Recovering old corrections here verifies implementation. It is not out-of-sample predictive evidence because the same teacher created those labels.

### Phase 3 — Prospective direct-controller H2H

Run the slow selector directly against the unchanged production controller on new mirrored games:

- identical generated teams/deals within each mirror;
- controller roles and player sides swapped;
- independent teacher-rollout randomness from played-game randomness;
- no ladder opponents and no adaptive opponent changes;
- raw paired win outcome as the primary estimand;
- pair-cluster bootstrap or an exact paired test as the confidence procedure;
- intervention telemetry analyzed only after the frozen primary result.

Suggested sequential schedule:

- 20 games: harness canary only; require zero voids and no gross side/role fault.
- 100 games: stop for clear harm or futility. Do not claim success from a favorable wide interval.
- 200 games: continue only if the point estimate is positive and both roles are non-harmful.
- up to 500 games: pass only if the preregistered 95% confidence interval for the paired win rate excludes 50% and the raw point estimate is positive in both role directions.

No resource-component or teacher-rollout metric may replace the played-game result. This follows the central MegaGem lesson: the myopic selector looked better locally and was worse live.

### Phase 4 — Export an expert-iteration corpus only after a live pass

Generate fresh games from the admitted slow controller against fixed production-search opponents. Preserve three row types:

1. **deviations**: the expert action plus its baseline action, paired terminal difference, uncertainty, search visits/values, and legal support;
2. **pass-throughs**: ordinary and ambiguous states in which the expert retained production search;
3. **resource anchors**: rare but important Tera, forced-switch, low-HP, PP-constrained, status/tempo, information-reveal, and switch-preservation states where the baseline is retained.

Split by physical battle before row expansion. Do not place decisions from one battle in different splits. Keep at least 20% of battles unopened for final student evaluation; use a separate confirmation set for H2H.

The first corpus should target **1,000–2,000 independently proven deviations**, close to MegaGem's approximately 1,200 but appropriate for Metagross's larger categorical action/state surface. At an intervention rate of 5%–15% and roughly 15–30 ordinary decisions per battle, this likely requires about **400–1,500 generated battles**. That range is an inference; a 50-game admission canary should replace it with a measured decisions/deviations rate.

Do not copy MegaGem's sampling mixture mechanically. Start from a preregistered 1:2:1 deviation/pass-through/resource-anchor unit ratio, upweight deviations only inside the training split, and run an anchor-only no-op control. Freeze the ratio after a training-loss and coverage smoke, before held-out teacher-advantage evaluation.

### Phase 5 — Distill once, conservatively

Train one supervised student from the accepted causal-history R1 checkpoint. This is behavior-policy improvement, not unrestricted RL. The target should be either the expert action or a KL-regularized distribution that retains baseline support. Expert Iteration supports this slow-expert/fast-apprentice division; safe policy-improvement work supports falling back to the behavior policy under uncertainty ([SPIBB](https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf)).

The primary offline student metric should mirror MegaGem's EV closure:

`closure = (Q(student_action) - Q(baseline_action)) / (Q(expert_action) - Q(baseline_action))`

on held-out deviation battles, using fresh teacher seeds. Also report:

- direction agreement;
- harmful-override count and summed harm;
- pass-through agreement;
- action-mask legality;
- resource-anchor retention by subtype;
- calibration of student confidence against teacher advantage.

Proposed admission thresholds are at least 50% median closure, zero high-confidence harmful overrides, at least 95% pass-through agreement, and no resource subtype with a statistically visible regression. These are conservative project choices, not universal thresholds.

### Phase 6 — Equal-budget student test

Remove the slow teacher. Put the student prior inside the **identical 500 ms production search** and compare against the frozen baseline. Use new mirrored games and the same sequential H2H framework. Require:

- confidence interval excluding 50%;
- positive point estimates in both role directions;
- no increase in voids, unsupported actions, or inference mismatches;
- at least half of the direct teacher's estimated paired improvement retained, with uncertainty reported.

Only this stage constitutes a production-speed improvement. If it fails, retain the slow controller as research evidence and do not deploy the student.

## Data and compute estimate

| Stage | Data | Dominant compute | Estimate and decision rule |
|---|---:|---|---|
| Integration | Existing opened roots only | CPU exact MCTS | No new game generation; benchmark 100 interventions before predicting wall time |
| One initial top-two intervention | 2 actions x 2 schedules x 8 worlds x 2 rollouts = 64 terminal trajectories | CPU/Rust engine | Existing schema-6 data averaged 14.94 continuation searches per terminal trajectory; at 2,048 iterations/search this is about **1.96M MCTS iterations** per intervention |
| Extended intervention | Same, 4–8 rollouts | CPU/Rust engine | About **3.91M–7.83M iterations** per intervention; cap adaptively and abstain when unresolved |
| 100-game H2H | Approximately 150–600 interventions if 1.5–6 ambiguous decisions/game | Parallel CPU cores | Roughly 0.3B–4.7B engine iterations depending on extension rate; measure, do not price from this wide bound |
| 500-game maximum | Same controller | Parallel CPU cores | Approximately five times the measured 100-game cost; embarrassingly parallel by mirrored pair, but games within a battle remain sequential |
| Corpus generation | 400–1,500 battles, aiming for 1k–2k deviations | CPU for expert labels | Stop once deviation and subtype coverage targets are met; do not collect to a round battle number |
| Distillation | Likely 10k–50k sequence rows after anchors/expansion | One modest GPU | The student is 142.8M parameters, so an H200 is unnecessary; benchmark one epoch. A single L4/A10/A100-class GPU should plausibly finish a small supervised round in hours, but this is an inference until measured |
| Final student H2H | 200–500 mirrored games | CPU production search; GPU optional only for batched prior service | Exact same 500 ms per move budget in both arms |

The iteration calculation is grounded in the local teacher report: 602,542 continuation searches for 40,320 attempted terminal samples, or 14.94 searches/sample. Wall time is deliberately not fabricated because no reliable elapsed-time field was recorded in those artifacts and iteration throughput depends strongly on worker count and root complexity.

GPU acceleration does not solve the expensive stage if the bottleneck is the Rust MCTS engine. More CPU cores and battle-level parallelism do. The GPU becomes useful only for student training or batched neural-prior inference.

## Falsifiable gates

The project should stop or advance on these claims:

| Claim | Pass condition | Failure action |
|---|---|---|
| Teacher semantics are valid | 100% hash/action-support parity; >=95% terminal; no information leak or silent fallback | Fix semantics; no H2H |
| Runtime selector is stable | Both schedules positive, half-split agreement, bootstrap LCB > 0, effect >=0.05 | Abstain at that root |
| Direct expert improves games | New paired H2H CI excludes 50%, positive in both roles | Stop distillation branch |
| Corpus contains learnable change | 1k–2k battle-disjoint proven deviations with subtype coverage | Collect targeted states only; do not pad with correlated rows |
| Student imitates safely | >=50% held-out closure, zero high-confidence harmful overrides, >=95% pass-through retention | Reject student; do not retune on confirmation |
| Student improves production | Equal-500 ms paired H2H CI excludes 50%, both roles positive | Keep current R1 + search |

High-confidence policy-improvement methods formalize the principle that a candidate should be returned only when it clears a user-chosen lower performance bound ([Thomas, Theocharous, and Ghavamzadeh, 2015](https://proceedings.mlr.press/v37/thomas15.pdf)). Those guarantees do not apply automatically to this neural, partially observed setting, but the experimental discipline—baseline fallback and a frozen lower bound—is the correct design analogue.

## Major methodological risks

1. **Opponent dependence.** Long-horizon Q values are values against a specified continuation policy, not universal action truths. MegaGem observed action-rank changes when the continuation opponent changed. Start with the exact production mirror matchup, then require a second fixed opponent mixture before broad deployment.
2. **Information leakage.** A determinized world contains sampled hidden sets. The policy/controller must receive only the causal public-history belief and aggregated outcomes. Any per-world species/move/item field crossing the boundary invalidates the result.
3. **Teacher model bias.** Exact terminal scoring does not make the continuation policy optimal. It establishes the outcome under that frozen continuation. Candidate comparisons must use identical continuation policies and common random tapes.
4. **Common-random-number failure.** Matched worlds/seeds often reduce variance, but this is not automatic in non-monotone systems. The classic analysis explicitly conditions the benefit on structural properties ([Glasserman and Yao, 1992](https://business.columbia.edu/sites/default/files-efs/pubfiles/4261/glasserman_yao_guidelines.pdf)). Report paired and unpaired uncertainty diagnostics.
5. **Post-selection bias.** Do not tune ambiguity thresholds, rollout counts, or confidence margins on H2H wins. Freeze them before prospective play.
6. **Repeated offline optimization.** Iteratively fitting against estimated Q can exploit its errors and amplify distribution shift; a strong empirical warning comes from one-step offline policy improvement outperforming iterative variants ([Brandfonbrener et al., 2021](https://proceedings.neurips.cc/paper/2021/file/274a10ffa06e434f2a94df765cac6bf4-Paper.pdf)). Run one distillation round and recollect before any second.
7. **Resource-price misspecification.** In operations research, bid prices arise as approximations to a dynamic-programming value function and can vary over time ([Adelman, 2007](https://pubsonline.informs.org/doi/10.1287/opre.1060.0368)). Metagross's resources are more state-dependent than auction capacity. A constant or linear shadow is a diagnostic at most.
8. **Teacher latency mistaken for product strength.** A slow teacher win proves a useful target exists; it does not prove a 500 ms agent has improved. Only the weights-only equal-budget stage closes that gap.
9. **Multiple comparisons and optional stopping.** One primary estimator, one pass boundary, and preregistered futility rules are required. Report exploratory component analyses as exploratory.
10. **GXE extrapolation.** Internal H2H against one frozen controller does not prove movement from roughly 92% to 95% GXE. That requires ladder evaluation or a validated calibration from H2H strength to GXE against a sufficiently broad opponent field.

## Expected value and confidence

Confidence that the **slow selector will make different, outcome-improving decisions at some roots** is high: the disjoint 128-root terminal teacher passed, and later panels found dozens of durable corrections.

Confidence that those decisions will improve full games when applied online is moderate, not high. The current teacher has real signal, but correction frequency, continuation-policy bias, and strategic interaction can erase local gains. The proposed 100-to-500-game direct H2H is exactly the test that resolves this uncertainty.

Confidence that one distillation round will retain a useful fraction of a proven selector is moderate if the live effect is large and at least about 1,000 diverse deviations are collected. MegaGem retained roughly three quarters of its selector effect, but its action was a scalar bid and its student was a 4B language model. Metagross must measure its own closure.

There is no defensible numerical promise of reaching 95% GXE from this experiment. The strongest justified forecast is conditional:

- if the direct selector wins by less than roughly 2–3 percentage points, distillation is unlikely to yield a dramatic production gain;
- if it wins by roughly 5–10 points with stable interventions and the student retains at least half, the branch could produce a material improvement worth a broad H2H/ladder gate;
- if it does not win directly, stop this line and preserve the result rather than training another proxy.

## Key sources and evidentiary role

| Source | What it supports | Quality / limitation |
|---|---|---|
| [Learning MegaGem, from self-play to price discovery](https://djdumpling.github.io/2026/08/09/megagem.html) | Complete experiment sequence, negative results, model/data sizes, live and distillation effects | Primary practitioner report with unusually detailed failures; not independently replicated |
| [MegagemBench repository, inspected commit](https://github.com/djdumpling/MegagemBench/tree/6e5b082e0718f16351907f9e2e08bce0fddd4459) | Executable environment, selector, artifacts, exporter, trainer, evaluation code | Primary implementation; current tree includes evolution beyond the exact blog run |
| [Expert Iteration](https://proceedings.neurips.cc/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf) | Slow planning expert / fast generalizing apprentice loop | Primary peer-reviewed paper; fully observed Hex differs from Pokémon |
| [MCTS as regularized policy optimization](https://proceedings.mlr.press/v119/grill20a/grill20a.pdf) | Interpretation of search visits as regularized policy improvement | Primary peer-reviewed paper; does not validate this engine or belief model |
| [BetaZero](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_27.pdf) | Belief-state representation and online planning under partial observability | Primary research; benchmark evidence, not Pokémon-specific |
| [SPIBB](https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf) | Fall back to a baseline under insufficient evidence | Primary peer-reviewed batch-RL work; guarantees do not directly carry to neural POMDP play |
| [High Confidence Policy Improvement](https://proceedings.mlr.press/v37/thomas15.pdf) | Frozen lower-bound admission principle | Primary peer-reviewed work; practical sample cost can be high |
| [Offline RL Without Off-Policy Evaluation](https://proceedings.neurips.cc/paper/2021/file/274a10ffa06e434f2a94df765cac6bf4-Paper.pdf) | Distribution shift and iterative error exploitation; motivation for one conservative round | Primary peer-reviewed empirical/theoretical analysis |
| [Common random numbers](https://business.columbia.edu/sites/default/files-efs/pubfiles/4261/glasserman_yao_guidelines.pdf) | Conditions and caveats for matched simulation seeds | Primary Operations Research analysis; structural conditions must be checked empirically |
| [Dynamic bid prices in revenue management](https://pubsonline.informs.org/doi/10.1287/opre.1060.0368) | Shadow prices as approximations to dynamic continuation value | Primary peer-reviewed operations-research source; auction/inventory setting is narrower |

## Final recommendation

Implement and freeze the selective slow controller now. Spend CPU only after a 100-intervention latency/semantic benchmark. Use prospective mirrored H2H as the deciding evidence. Do not train, distill, open confirmation, or claim a GXE gain until that controller itself wins. If it wins, one deviation-focused expert-iteration round is the highest-value next optimization; if it fails, the honest conclusion is that the currently certified long-horizon teacher is diagnostically useful but not a stronger playing policy.
