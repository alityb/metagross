# Belief-v2: the full inference-search program (owner-directed 2026-08-17)

Owner directive: apply the complete literature stack, not one component.
The five sources compose into three stacked components; each enters play only
through its own single-variable, preregistered H2H gate against the current
champion (ladder of champions, no simultaneous changes). Cloud compute is
authorized for offline stages; live gates stay on the reference machine.

## Component 1 — learned world-posterior sampling (the Skat trio, unified)

Replaces the weak link identified by all three local findings: uniform/
frozen-R1 belief weighting.

- 1a (Buro IJCAI-09 lesson): estimate P(world | observed actions) OFFLINE
  from human data — never runtime 0/1-brittle P(move|world) from our own
  deterministic policy.
- 1b (Solinas AAAI-19): a set-membership model — predict components of the
  opponent's hidden set (species/roles/items/moves) from causal public
  history; posterior over candidate sets by feature-product. Fast, first to
  build. Gate: TSSR/top-k/Brier vs the frozen baselines from
  `belief_tssr_baseline_20260817` on held-out battles.
- 1c (Policy Inference CoG-19): world reach probability = product over the
  opponent's observed actions of a LEARNED human-policy likelihood
  P(action | world, public state) — our cumulative action-conditioned
  belief with the frozen-R1 likelihood replaced by a policy trained for
  exactly this conditional. Expected to beat 1b (as in Skat); costlier per
  world. Gate: same offline metrics; then the better of 1b/1c goes to a
  fresh live H2H gate vs production sampling (D-series rules: SPRT
  H0=.50/H1=.55, Wilson95 LB>50%, unwatched).
- Training data: TRAIN split human battles only; true sets appear only as
  held-out evaluation labels; all splits battle/dependency-grouped; every
  model input passes the causal/leak checks (hidden-completion
  insensitivity).

## Component 2 — cross-world aggregation (MAPLE, 2605.24139)

After Component 1 settles the posterior, replace uniform posterior-weighted
visit-mass averaging with principled multi-state aggregated policy
evaluation. Local motivation: humans matched the majority-world top only 36%
in ambiguous states — aggregation, not just weighting, is part of the gap.
Requires a careful read of MAPLE first (method details TBD); enters as its
own single-variable gate on top of the Component-1 champion.

## Component 3 — determinization resource allocation (2607.13007)

Allocate the fixed 500 ms budget nonuniformly across worlds by posterior
mass/uncertainty instead of equal per-world search. Read paper first; local
precedent exists (the selective-shared-root trigger family) but as
scheduling, not solving. Own gate on top of the running champion.

## Addendum 2026-08-17: owner selected components 1, 4, 5; papers read in full

- **Component 1, no-deep-training form (Buro IJCAI-09 / Kermit-style):**
  count-based conditional evidence tables from the TRAIN corpus — for
  randbats the generator pool already fixes the set space, so the new layer
  is P(observed early actions | set features) estimated by counting over
  20,385 battles, multiplied into the existing conditional-generator
  posterior (log-space, tempered, fail-closed like the current action
  belief). No neural net. Gate: TSSR/top-k/Brier vs the
  `belief_tssr_baseline_20260817` arms. The learned 1b/1c models remain the
  program's later upgrade path.
- **Component 4 (MAPLE, 2605.24139), read:** single shared PUCT tree over
  information-set nodes; k worlds share the tree; per iteration the selected
  action path is applied to every world, illegal worlds discarded; leaf
  policy = legality-masked mean over valid worlds, leaf value = mean;
  aggregated value backpropagated. k x N network-eval cost control. Their
  Elo gains (+291 Phantom Go, +136 Dark Hex) came with training in the
  loop, BUT their own ablations show inference-time aggregation with more
  worlds (k_E) improves fixed checkpoints — so a no-training port is
  legitimate: shared tree over our 8 posterior worlds using hand-leaf
  values and existing priors. Engine-level (Rust) build. Explicit local
  caution, recorded: shared-root equilibrium solving failed here twice
  (RM+ 6-18); MAPLE is best-response aggregation, not equilibrium — a
  materially different mechanism, but the burden of proof sits on a fresh
  20-game screen. Siamese sampling (their trained component) is skipped in
  v1; Component 1's posterior does that job.
- **Component 5 (2607.13007), read:** purely algorithmic, no training.
  Two axes: dynamic determinization count (margin-based, turn to turn) and
  dynamic per-world simulation allocation (greedy contestedness with
  Hoeffding locks; across-tree UCB; move pruning). Their strongest pattern:
  across-tree UCB with voting aggregation (+2.6 to +4.7 pp); their
  landmine: tree-level UCB with visit-sum aggregation was severely harmful
  (up to -16 pp) because extra budget skews the aggregate. Our production
  aggregation normalizes per-world visits before posterior weighting, which
  partially shields us, but the paper's lesson is binding: any allocation
  variant must be validated under the wall-clock deployment budget, and the
  final-selection rule is part of the frozen variant. Gains are small and
  game-dependent — sized as a cheap incremental gate, not a headline bet.
- **Revised build order (cost-ascending): 1 (count tables + TSSR, cloud) →
  5 (allocation scheduling, engine orchestration) → 4 (shared-tree MAPLE
  port, largest build).** Each still enters play only through its own
  single-variable gate against the current champion.

## Combined design (2026-08-17)

Target stack, outermost to innermost — each layer independently gateable and
independently removable:

    R1 root prior (unchanged champion anchor)
      └─ BELIEF: conditional generator pool
           x Component-1 evidence tables (count-based, offline)
           x cumulative action-conditioning (D1 thread; frozen-R1 now,
             learned 1b/1c later)            → posterior worlds + weights
      └─ SEARCH: Component-4 MAPLE shared tree over the k posterior worlds
           (legality-masked mean policy/value, hand-leaf evals, R1 priors);
           fallback form: current independent per-world ensemble
      └─ BUDGET: Component-5 dynamic allocation
           - in MAPLE form: dynamic k (world count) + world-eval locking
           - in ensemble form: per-world iteration allocation
             (across-tree UCB / greedy contestedness + Hoeffding locks)

Shared infrastructure economy: ONE extraction pass over the 20,385 TRAIN
battles (the TSSR Phase-L extractor) yields (a) TSSR evaluation records,
(b) Component-1 counting data, and (c) later 1b/1c training data. Build it
once, on cloud, with true sets stored label-only.

Integration points: evidence tables → `belief/` alongside
`randbats_determinize` with the action-belief's log-space/tempering/
fail-closed conventions; allocation → chunked per-world scheduling in the
production search orchestration (engine already supports iteration-budgeted
calls); MAPLE → new Rust engine mode reusing `shared_information_set`
scaffolding, PyO3-bound, behind a feature flag with byte-identical
fallback.

Gate ladder (one variable per gate, champion-relative, non-additivity
expected per Kowalski — the final combined configuration gets its own fresh
confirmation gate):

    C0 champion = production (or D1 winner after its confirmation)
    G1 = C0 + evidence tables          (belief-layer variable)
    G2 = G1-champion + allocation      (budget-layer variable)
    G3 = G2-champion + MAPLE tree      (search-layer variable)
    G4 = combined-config confirmation  (unwatched, Wilson LB > 50%)

## Compute budget (honest)

Cloud (Modal, CPU unless noted):
- One-time extraction of all 20,385 battles: ~11-23 core-hours of node
  rematerialization; ~64-way parallel ≈ 30-60 min wall, **$5-15** (requires
  porting node harness + pinned Showdown worktrees into the image — the one
  real porting task; no Rust engine needed).
- TSSR baselines (3 sampler arms, 256 battles): **$10-25** (R1 likelihood
  arm dominates).
- Component-1 table building: counting, **<$2**; its TSSR eval: **$5-10**.
- Later learned 1b/1c training (small nets over 1.29M states): **$20-80**,
  optionally one small GPU.
- Near-term cloud total ≈ **$25-50**; with the learned-model phase ≈
  **$100-150**. All development-stage; live gates stay local ($0).

### Revised evaluation architecture (owner directive: no local gate bottleneck)

Gates move to a **cloud gate farm**. Validity argument: a paired H2H gate
requires both arms on identical hardware/budget — fairness is internal to
the comparison, so a Linux gate is as decisive as a Mac gate for A-vs-B.
Platform becomes part of each gate's frozen config ("at cloud budget").
Only the single final end-to-end winner needs one confirmation run on the
deployment machine at the true deployment budget.

- One-time Linux port (the real cost, ~1-2 engineering days): rebuild the
  Rust engine from the in-repo source (equivalence smoke: fixed-seed
  searches compared across platforms; if not bit-identical, platform-
  internal consistency suffices since both arms share it), pip-able
  foul-play/metamon/torch-CPU stack, current Showdown, R1 checkpoint in an
  image/volume. Pinned OLD Showdown worktrees are needed only by the
  extraction image, not the game image.
- Cost per powered gate: a game-pair needs ~16-24 vCPUs; 500 games ≈
  400-600 vCPU-h ≈ **$15-30 spot**; at 100 concurrent games ≈ **20-40 min
  wall**. Doubling to 1,000-2,000 games for tight CIs is still <$100 —
  power becomes cheap, so gates can demand CI-excluding-50% instead of
  aggressive small-N thresholds.
- Ladder G1→G4 wall time collapses from ~4-5 machine-days to ~1 day of
  cloud runs; offline gates (TSSR) filter candidates before any live gate
  so most losers never cost a game.
- Local reference machine: development smokes + the one final deployment-
  budget confirmation of the end-to-end champion.
- Discipline unchanged: preregistration per gate, fresh identities,
  unwatched decision gates, one variable per gate, iteration-log entries.

Failure economics: components are ordered cost-ascending, so a dead cheap
layer never blocks the expensive one behind it; any layer that fails its
gate is dropped from the stack without redesigning the others.

## Ordering and discipline

TSSR baseline (Modal) → 1b → 1c → live gate → 2 → 3. The running D1 screen
is unchanged and, if it promotes via its confirmation gate, simply updates
the champion Component 1 must beat. One component change per gate; a failed
component is closed, not retuned; iteration log records every step; sealed93
stays sealed; GXE claims still require a bounded ladder block.
