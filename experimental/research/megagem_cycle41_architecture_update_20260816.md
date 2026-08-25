# MegaGem replication after Cycle 41

Date: 2026-08-16  
Scope: research update only; local CPU / $0 / sealed-93 unopened / no hidden-state leakage  
Decision: stop direct root-teacher distillation; cheaply exhaust one-deviation attribution, then put the main architecture bet on causal search-native interior priors while retaining R1 at the root

## Executive decision

Cycle 41 is a clean negative gate, not an equivalence result. The equal-prior
8,192-iteration controller completed all 20 prospective games with zero
semantic or operational failures, changed 165 of 575 candidate decisions
(28.70%), and scored 11-9. Its Wilson 95% interval was wide
`[34.21%, 74.18%]`, but it missed the frozen `>=13/20` continuation threshold.
The immutable result is
`experimental/runs/search_native_v2_cycle41_blind_scorer_repair_20260816/RESULT_REPORT.json`
(SHA-256
`39c5353af5772f655e3afa7767486d769b6714d35cb0ac157b3a30b2f65b5b5c`).

This rules out using the Cycle 41 controller as a proven MegaGem-style expert.
MegaGem distilled only after the complete live selector won. Metagross now has
two relevant failures: the exact terminal-MCTS direct controller scored 72-78
over 150 games, and the semantically clean equal-prior 8,192 controller missed
its large-effect gate. No Cycle 41 action, visit distribution, or override is
therefore an admitted improved-policy label.

One cheap question remains: were isolated equal-8,192 deviations useful under
the production continuation, but damaged by applying 165 changes recursively?
A prospective randomized one-deviation canary can answer that for roughly one
or two local CPU hours. It is an attribution experiment, not the main
architecture and not permission to retune Cycle 41.

The only remaining architecture with a defensible three-GXE ceiling is:

> Keep corrected causal-history R1 as the exact root prior and production
> fallback; add a small causal public-information policy only at depth-one
> interior nodes; train it on full causal human behavior anchors plus stable,
> regularized soft search policies aggregated across posterior worlds; admit it
> only by equal-budget full-game H2H.

This is genuinely new. Cycle 41 replaced the root policy with equal-prior
high-budget decisions at every eligible root. It did not test whether a
history-correct neural proposal can allocate a fixed search budget better at
interior information states while leaving the known-good root R1 prior intact.

## Evidence update

No new web search was needed. This update reuses the ten primary sources deeply
reviewed in the Exa-backed MegaGem memo and adds the local Cycle 1, 1b, 4, 6,
7, 12, 17, and 41 evidence. The earlier source search reviewed 320 returned
records and deduplicated the audit pass to 139 URLs; the primary sources used
here are listed below.

Evidence is ranked as follows:

1. **Direct prospective local evidence.** Cycle 41 is the highest-relevance
   result because it tested the exact controller in complete games with clean
   causal mechanics. It overrides favorable self-referential search-Q and
   stability diagnostics.
2. **Earlier local causal evidence.** Cycle 17 established stability and
   nontrivial differences, not strength. Cycle 1/1b reject the current
   terminal-selector family. Cycles 4/6/12 establish that a no-leak causal
   representation and a 20,385-battle human corpus are now technically
   possible. Cycle 7's design audit remains a warning against stateless public
   fingerprints, equal world aggregation, and one-hot same-teacher targets.
3. **Peer-reviewed architecture evidence.** Expert Iteration supports a slow
   planner/fast apprentice loop; regularized-policy and Gumbel work supports
   soft search-policy improvement and better finite-budget allocation; belief
   planning work supports player-information-state rather than hidden-world
   inputs. None proves that this Pokémon engine's hand-evaluator search is a
   stronger teacher.
4. **MegaGem practitioner evidence.** It strongly supports the experimental
   sequence—prove the complete expert live, then distill deviations and
   pass-throughs—but its auction formula and effect sizes do not transfer.

## Ranked remaining architectures

| Rank | Candidate | Research case after Cycle 41 | Cheapest falsifier | Planning chance of a real +3 GXE |
|---:|---|---|---|---:|
| 1 | **R1-root + causal search-native interior prior** | Highest ceiling and genuinely untested. It improves allocation inside the tree without replacing the accepted root representation. Full causal human streams provide a non-search anchor; stable soft search policies can refine it. | Existing-TRAIN depth-one mechanics/learnability gate, then 20 fresh mirrored games at equal 500 ms | **3%-8% before the gate; 15%-25% only after a clean `>=13/20` screen** |
| 2 | **R1/uniform prior mixing with regularized or Gumbel root allocation** | Defensible as a low-cost variance/allocation experiment. It may preserve R1 where Cycle 41's uniform teacher overrode too often. It does not repair the hand evaluator, belief model, or missing interior guidance. | Forty opened TRAIN roots at matched exact iterations, followed by one-deviation play only if stable | **2%-5%** |
| 3 | **High-budget search retaining pure R1 root priors** | Valid control, but low ceiling. In Cycle 17, R1-prior 20k produced only three stable differences on 40 roots and failed the minimum-difference gate. More of the existing root mechanism is unlikely to yield three GXE points. | Add R1-8,192 to the existing Cycle 17 root panel; require at least four stable differences and 80% agreement with R1-20k before any game | **1%-3%** |
| diagnostic | **Prospective one-deviation equal-8,192 attribution** | Can distinguish harmful compounding from useful isolated actions under the actual production continuation. It is not a deployable architecture and a null result does not test interior priors. | Fixed 20-game randomized canary below | **2%-5% chance of exposing a root-only effect large enough to justify power** |

These percentages are subjective planning ranges, not calibrated probabilities.
Moving from approximately 92.5 to 95 GXE is a large population-level change;
no 20-game internal result establishes it. A branch must ultimately beat a
broad frozen opponent panel and pass a bounded ladder evaluation.

## Why each branch does or does not survive

### 1. Retaining R1 root priors at high budget

This is methodologically clean: keep the same PUCT rule, determinizations,
evaluator, and exact iteration budget while changing only the prior. It tests
whether the learned policy allocates finite visits better than equal legal
mass. Cycle 17 already supplied the crucial warning: R1-20k was stable but
differed from exact production at only three of 40 roots, whereas equal-8,192
produced eight stable differences. A pure R1 high-budget arm may be a safer
controller, but it is probably too similar to production to create a dramatic
upgrade.

Use it as a control in every remaining root-allocation experiment. Do not run a
standalone 100-game study unless the cheap root panel first demonstrates both
stable differences and a non-circular, independently attributed advantage.

### 2. Prior mixing, PUCT, and Gumbel policy improvement

A frozen mixture

`P_mix(a|I) = (1 - lambda) P_R1(a|I) + lambda U_legal(a|I)`

can interpolate between production's learned root prior and Cycle 41's uniform
prior without adding a learned value. PUCT makes the role of this prior
explicit, and the regularized-policy interpretation of MCTS supports treating
the search output as a soft improved policy rather than a one-hot action.

Gumbel sequential halving is research-backed for finite simulation budgets,
but its policy-improvement statement depends on the algorithm's completed
Q-values and setting assumptions. The current engine records Q only for visited
actions, uses a hand evaluator, aggregates determinizations, and independently
selects simultaneous sides. A root sampler that merely adds Gumbel noise to raw
visits is **not** Gumbel AlphaZero and inherits no improvement guarantee.

The branch is defensible only if it first implements and audits:

- exact legal-action logits and a declared completed-Q rule;
- identical particles, chance tapes, evaluator, and exact iteration counts
  across R1, uniform, mixture, and Gumbel arms;
- repeated-search stability and no missing/nonfinite action values;
- a selection rule frozen without looking at played-game outcomes;
- prospective one-deviation or complete-game confirmation.

This is worth trying as a cheap allocation control, but not as the main +3 GXE
bet. It changes which flawed estimates receive compute; it does not create a
better state representation or evaluator.

### 3. One-deviation matched attribution

Cycle 41 applied equal-8,192 at every eligible decision, so its 165 overrides
changed the continuation distribution. The played result cannot tell whether
individual first deviations were beneficial and later deviations compounded
badly. A separate one-deviation design can estimate the action treatment under
the actual production continuation.

This survives despite Cycle 1b because the treatment is different. Cycle 1b
tested the terminal-MCTS selector and stopped at 11 eligible games, a +10 point
estimate, and one-sided Fisher `p=0.608`. Reusing that treatment, threshold, or
outcome is prohibited; equal-8,192 must receive wholly fresh teams, seeds,
identities, and assignment.

Even a positive one-deviation result would authorize only a powered attribution
study or a frozen conservative root gate. It would not authorize distilling all
Cycle 41 visits, because most of those decisions were never causally shown to
improve production continuation.

### 4. Search-native interior priors

This is the highest-ceiling survivor because it attacks a missing capability:
the deployed 142.8M policy is queried only at the root. Interior nodes revert
to unguided search even though finite-budget allocation is where policy priors
are most valuable. The first version should be deliberately narrow:

- unchanged causal-history R1 prior and safeguards at the root;
- a 1M-5M parameter CPU-batched policy at depth one only;
- input is the acting player's exact own private request/current state plus the
  chronological typed public-event ledger and reveal masks;
- no completed sampled opponent team, per-world hidden feature, terminal
  outcome, or future reveal enters the model input;
- targets are aggregated by the full causal player-information state across
  posterior-weighted worlds;
- human actions from the admitted full-causal TRAIN corpus are the behavior
  anchor;
- search refinement uses stable soft visit/regularized-Q policies, never only
  the selected equal-8,192 action;
- root replacement, learned leaf value, deeper-node rollout, and opponent head
  remain separate locked ablations.

The Cycle 41 failure lowers confidence in search targets but does not falsify
this architecture. Cycle 41 tested a uniform-prior root policy applied
recurrently. It did not test whether a causal neural proposal improves the
efficiency of otherwise unchanged R1-root search. The branch must nevertheless
earn admission through full games; offline KL or agreement cannot rescue it.

## Prior local families now ruled out

Do not repeat any of the following without a genuinely new falsifiable
mechanism:

- exact terminal-MCTS recurrent overrides (72-78/150) or a powered repeat of
  its failed Cycle 1b canary;
- direct equal-prior 8,192 recurrence or generic “more equal-prior visits”
  after Cycle 41's 11-9 failure to clear `13/20`;
- distillation from Cycle 41 selected actions or visits before a live expert
  win;
- generic value heads, action-Q critics, sequential embeddings, ridge/GBDT
  residuals, specialist gates, or candidate-matchup residuals that already
  failed grouped zero-harm gates;
- another static HP/Tera/PP/hazard/tempo resource shadow or hand-tuned leaf
  blend;
- exact dual-R1 continuation until its causal-history termination certificate
  exceeds the old 2.16%;
- stateless root/interior representations that omit causal event identity;
- equal averaging across hidden worlds when the sampler supplies posterior
  weights;
- one-hot same-teacher targets, raw visits without allocation correction, or
  hidden-world-specific training rows;
- pure shared-root RM+/unchanged regret treatment (previously 6-18), old Exp3,
  naive latest-self RL, or additional RL without a stronger admitted policy;
- increasing 8,192 to 20,000 merely because it is deeper: Cycle 17 found the
  equal arms already largely agreed, and more iterations do not correct model
  bias.

## Cheapest falsification protocol

### Gate 0: exhaust the root-only causal question

Freeze a new 20-game / ten-pair canary before pair generation.

1. Both arms play unchanged corrected R1 + production search until the first
   label-blind root at which equal-8,192 is stable in both schedules and selects
   a different legal action.
2. Assignment to equal-8,192 action or production action is blocked ex ante by
   mirror leg and hidden from outcomes. At most one action changes in a game.
3. Immediately afterward, both arms use unchanged production for the rest of
   the actual game. No post-lock teacher call is allowed.
4. Use new teams, battle seeds, search seeds, usernames, run IDs, and pairs;
   reuse no Cycle 41 outcome or seed. Require exact current baseline-action
   compatibility and every Cycle 41 semantic receipt.
5. A teacher timeout, invalid assignment, baseline mismatch, second
   intervention, receipt failure, or post-lock call voids the fixed canary.

Large-effect continuation gate, frozen in advance:

- at least 14 eligible games and at least six eligible games per arm;
- equal-8,192 minus production win rate at least +25 percentage points;
- one-sided Fisher exact `p <= 0.20`;
- zero integrity failures and a nonnegative effect in both public-role
  directions.

If any condition fails, stop all root-only equal-8,192 distillation and powered
attribution. The thresholds deliberately ask whether a large missed effect is
present; they are not a strength claim. A pass authorizes only a separately
powered, frozen one-deviation study.

### Gate A: interior representation and target mechanics

Run only if pursuing the higher-ceiling architecture; Gate 0 need not pass,
because it tests a different mechanism.

Use 64 dependency-disjoint Cycle 12 TRAIN battles; validation, test, sealed-93,
and prior H2H data remain unopened. Select roots without teacher results. For
each root, build two independent eight-world posterior schedules and ordinary
depth-one successors.

Require:

- 100% exact action/request/state-hash join for admitted roots and children;
- zero hidden-completion sensitivity in the serialized public model input;
- public-event identity, order, actor, move/target, form, item, ability, PP,
  disable, Tera, switch, and request boundary retained rather than collapsed
  into a stateless fingerprint;
- posterior weights preserved and normalized; report effective sample size;
- same public information states merged across worlds, never emitted as
  hidden-world rows;
- exact perspective/apply/reverse/repeat parity and zero split overlap;
- at least 512 unique public-state fingerprints from at least 48 battles,
  otherwise stop for insufficient target diversity;
- repeated 8,192/20,000 target top-action agreement at least 80%, median repeat
  JS at most 0.05 nats, and 90th percentile at most 0.15;
- if the Gumbel arm is claimed, a complete finite Q for every legal action and
  an audited completed-Q implementation; otherwise label it ordinary
  regularized/visit policy improvement.

### Gate B: local learnability and integration

Train three CPU seeds of a 1M-5M parameter model on battle-grouped TRAIN only.
Compare soft regularized search targets against a same-data one-hot selected-
action control and a human-anchor-only control.

Require simultaneously:

- grouped held-out soft-target KL/cross-entropy improvement over one-hot with a
  battle-bootstrap lower bound above zero;
- at least 95% action/legality fidelity on untouched human/R1 anchor states;
- zero high-confidence illegal action and zero source-family collapse;
- byte-identical root R1 inputs/outputs with the interior feature disabled;
- actual depth-one engine calls use only the public model record;
- batch-64 local CPU inference p95 at most 5 ms and complete equal-500 ms search
  with no timeout/fallback increase.

Failure stops before a game. Offline metrics prove representation and
learnability only.

### Gate C: equal-budget played falsification

Only after Gates A and B pass, freeze 20 fresh mirrored games:

- candidate: unchanged R1 root prior plus depth-one interior policy;
- comparator: unchanged corrected R1 + production search;
- identical 500 ms, hardware, worlds, leaf evaluator, role/team rotation, and
  semantic gate;
- continue only at `>=13/20`, with zero failures and no protected-role
  regression; otherwise stop before data scale-up;
- a separately frozen 50-game stage must reach at least 28/50 before any
  25k-game collection, GPU request, sealed-panel access, or ladder test.

The `13/20` and `28/50` thresholds are aggressive developmental gates, not
confidence-interval strength claims. A real promotion still requires a powered
opponent-panel H2H with its interval above 50% and then a bounded ladder block.

## Compute boundary

All recommended pre-H2H work is local CPU. Cycle 17 measured equal-8,192 at
27.97 ms per world and R1 replay/mechanics already pass in fresh subprocesses.
The 20-game one-deviation canary should be budgeted as a measured one-to-two
hour diagnostic, not assumed from search iterations alone. The 64-root
interior mechanics/target pilot should be capped at two wall-clock hours and
stopped if its measured throughput cannot support the 500 ms search envelope.
A 1M-5M parameter three-seed pilot is small enough for CPU; no GPU is justified
before the played Gate C passes.

## Research basis

- Alex Wa, [Learning MegaGem, from self-play to price discovery](https://djdumpling.github.io/2026/08/09/megagem.html), and the inspected [MegagemBench implementation](https://github.com/djdumpling/MegagemBench/tree/6e5b082e0718f16351907f9e2e08bce0fddd4459): prove the complete selector before distillation; retain deviations, pass-throughs, and resource anchors.
- Anthony, Tian, and Barber, [Expert Iteration](https://proceedings.neurips.cc/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf): slow planning expert and fast apprentice; supports soft tree-policy targets, not an unproven teacher.
- Grill et al., [MCTS as regularized policy optimization](https://proceedings.mlr.press/v119/grill20a/grill20a.pdf): principled soft policy-improvement target from policy and Q.
- Danihelka et al., [Policy Improvement by Planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO): sequential halving and completed Q for limited simulations; assumptions and completed-Q mechanics must be implemented rather than borrowed rhetorically.
- Moss et al., [BetaZero](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_27.pdf): online planning over beliefs/player information in partially observable problems.
- Laroche et al., [SPIBB](https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf): behavior fallback under insufficient evidence; guarantees do not directly transfer to this neural POMDP.
- Brandfonbrener et al., [Offline RL Without Off-Policy Evaluation](https://proceedings.neurips.cc/paper/2021/file/274a10ffa06e434f2a94df765cac6bf4-Paper.pdf): warns against iterative exploitation of estimated values and supports a single conservative round only after a strong teacher.

## Final recommendation

Do not distill Cycle 41 and do not continue generic equal-prior search. If the
root-only hypothesis must be exhausted, run exactly one fresh randomized
one-deviation large-effect canary and stop on failure. In parallel only at the
design level, specify the causal depth-one interior-prior Gate A.

The build priority is the interior-prior architecture because it preserves the
component that demonstrably works—R1-guided production search at the root—while
adding guidance where none exists. It should receive no large data collection,
GPU, sealed-panel, or ladder budget until the tiny local model wins the fixed
equal-budget 20- and 50-game gates.
