# Metagross — Resume Raw-Material Brief (2026-08-26)

Purpose: complete, verified factual material about the metagross project for a
resume-writing agent. Every number in "VERIFIED CLAIMS" is measured and
defensible in an interview. Items in "QUALIFY IF USED" need the stated
caveat. Do not invent numbers beyond these.

## One-line project description

A research-grade Pokémon Gen 9 Random Battle agent: a 142M-parameter
transformer policy (fine-tuned with expert iteration on search-generated
self-play) combined with determinized PUCT Monte-Carlo tree search for
online planning under partial observability, evaluated with
preregistration-driven controlled experiments on the public Pokémon Showdown
ladder.

## Technologies (all actually used)

- Python (PyTorch, AMAGO RL framework, transformer policy serving over HTTP)
- Rust (custom MCTS poke-engine with learned-prior extensions; built via
  maturin into Python wheels; feature-flagged cargo builds)
- Reinforcement learning: expert iteration (ExIt) — policy distilled from
  MCTS visit distributions over self-play; RL2-style sequence inputs
- Search: PUCT with learned root priors (c_puct 2.0), determinized search
  over 16–32 belief-sampled worlds per move, 500 ms/move wall-clock or
  deterministic 472,000-iteration budgets, parallelism 8
- Belief modeling: conditional team generator sampling opponent sets
  consistent with revealed moves/items/abilities from a 50,000-team pool
- Infra: Modal (serverless containers; image builds compiling Rust in-image,
  torch CPU, Node/Showdown simulator), Tailscale (userspace networking,
  SOCKS relay), macOS local orchestration, GitHub
- Node.js (Pokémon Showdown simulator embedded for hermetic evaluation;
  protocol-level integration)
- Statistics: Glicko-1/GXE ladder ratings, Wilson intervals, SPRT sequential
  testing (H0=0.50/H1=0.55), mirrored-pair game designs, preregistration

## VERIFIED CLAIMS (safe to use, measured)

Model & system:
- 142,832,563-parameter transformer policy (AMAGO MultiTaskAgent,
  Perceiver-style timestep encoder + transformer trajectory encoder,
  max sequence length 128), served on CPU for real-time play.
- Custom Rust MCTS engine with native reveal masks and prior-conditioned
  root search (s1/s2 priors), built as manylinux wheels in CI-like Modal
  image builds.

Ladder results (public Pokémon Showdown, gen9randombattle):
- Historical peak: 92.7 GXE (~2362 Elo era), documented; the strongest
  documented Gen 9 Random Battle bot at the time.
- 2026-08 contemporaneous re-measurement, BOTH accounts run to the Glicko
  RD-25 convergence floor (the strictest standard the ladder supports):
  - Stateless serving arm: 174-83, GXE 91.7, peak Elo 2399, Glicko 1952±25.
  - Causal-history serving arm: 149-81, GXE 89.4, Elo 2226, Glicko 1902±25.
  - ~470 total rated games across the pair; single-variable design (only
    the policy's input trajectory mode differed).
- Benchmarks REPRODUCED across eras: historical 92.7/92.4 vs contemporary
  91.7 (Δ≈0.7–1.0 GXE); earlier frozen-model measurement 86.6 GXE/1851±25
  reproduced at 89.4/1902±25 by an identically-configured rerun.

Experimental methodology (differentiating for research roles):
- Preregistered every experiment before launch (frozen hypotheses,
  interpretation rules, seeds, stopping criteria); results graded against
  the preregistration including honest MISS/INVALID grades.
- Controlled A/B ladder study: two fresh accounts, identical machine,
  identical checkpoint/engine/config, deterministic search budgets; run
  sequentially to eliminate same-IP matchmaking interference (discovered
  empirically: concurrent same-IP laddering starves one account).
- Mirrored-pair head-to-head harness: both agents play both sides of the
  same generated team matchup (variance reduction), SPRT early stopping,
  fail-fast, resumable from progress snapshots.
- Convergence standard: all headline numbers quoted at Glicko RD ≤ 25 (the
  ladder's rating-deviation floor), matching standards across eras so
  numbers are directly comparable.

Diagnosis & interventions (research-loop story):
- Instrumented per-decision search telemetry (prior entropy, pooled visit
  entropy, per-action Q across sampled worlds, decision-rule agreement).
- Diagnosed a late-game PRIOR ENTROPY COLLAPSE in causal-history serving:
  mean prior entropy 0.96→0.58 nats by turn 30+, independently replicated
  on a second platform (cloud) months of games apart.
- Built three serving-time interventions, each env-gated, fail-closed on
  misconfiguration, with mandatory observable activation logging:
  1. Entropy-matched temperature flattening (turn-bucketed tau schedule
     calibrated offline on 699 paired decisions).
  2. Causal-history window truncation (last-K-steps, reusing the model's
     in-distribution windowing mechanics).
  3. Gumbel-style completed-Q root decision rule (log pi(a) +
     (c_visit+max_N)*c_scale*Q̂(a) pooled across sampled worlds; candidates
     nominated from prior support; mctx-default constants).
- Cleanly FALSIFIED the Gumbel-lite rule via a marginal-contribution A/B:
  10-40 (20%, CI [11,33]) vs identical stack without it — despite the rule
  flipping 37% of decisions toward higher pooled-Q actions; demonstrated the
  visit rule's implicit variance discipline carries real value.
- Verified the flattening intervention moves its mechanism (30+ turn prior
  entropy 0.58→1.40 under the schedule; propagates into search visit
  entropy 0.59→0.95) before testing outcomes; final outcome screen run
  with per-arm activation evidence (in progress at ~56% over 168 games,
  final number pending).

Infrastructure engineering (systems story):
- Containerized the entire stack into a Modal evaluation farm: in-image
  Rust engine compilation (maturin), 142M model serving, embedded Pokémon
  Showdown Node simulator, per-arm prior servers with trajectory-mode
  isolation; per-lane results and telemetry persisted to a shared volume.
- Deterministic search budgets (iterations-per-move instead of wall-clock)
  to make search depth load- and platform-invariant — enabling fair evals
  on shared/cloud CPUs and concurrent local runs.
- Request-bound prior-serving contract: priors bound to the exact Showdown
  request (rqid + canonical request SHA-256) with mandatory selected-action
  acknowledgements, so the causal input trajectory is exactly
  request-aligned (no silent desync).
- Solved cloud-to-ladder access legitimately: Showdown proxy-locks
  datacenter IPs; built a Tailscale userspace-networking relay through the
  owner's own residential connection with a FAIL-CLOSED egress gate (run
  refuses to start unless verified exiting via a non-cloud IP).
- Hardened long-running eval loops: resilient supervisors (auto-resume with
  persistent ladder ratings), per-battle abandon semantics instead of
  block-crashes, stall alarms (no-log-growth detection) after a wedged
  container incident, activation-evidence checks as validity gates.

## QUALIFY IF USED (true but needs the stated framing)

- "92.7 GXE and 2399 Elo": both real measured peaks, but from different
  runs/eras (92.7 historical GXE peak; 2399 Elo peak from the 2026-08
  stateless arm). Fine to state together as peaks; don't claim both from
  one run.
- "Strongest documented Gen 9 Random Battle bot": defensible for the
  documented-bot claim at time of measurement; phrase as "strongest
  documented" not "provably best".
- Stateless-vs-causal gap (91.7 vs 89.4): a consistent ~2 GXE population-
  level edge, but head-to-head mirrored play measured parity (48%);
  describe as "characterized the gap" not "stateless is better".
- Temperature-flattening outcome: final screen still running; do NOT claim
  a win/loss number yet. Safe claim: "diagnosed the pathology, verified the
  intervention moves the mechanism, outcome evaluation in progress."
- Modal farm: fully validated end-to-end (played real games in-container);
  the LADDER-from-cloud path was blocked by Showdown policy, hence the
  residential relay; offline self-play/eval works purely in-cloud.

## Numbers cheat-sheet (for quick bullet assembly)

- 142M params (142,832,563); max seq len 128
- 16–32 sampled worlds/move; 500 ms/move; parallelism 8; c_puct 2.0
- 472,000 iterations/500ms deterministic budget (M4-calibrated)
- 50,000-team belief pool
- 92.7 GXE historical peak; 91.7 GXE + 2399 Elo (2026); 89.4 GXE causal arm
- Glicko 1952±25 and 1902±25 (both at the RD-25 floor); ~470 paired-study games
- Entropy collapse 0.96→0.58 nats (turn 30+), replicated; intervention moves
  it 0.58→1.40 (prior) / 0.59→0.95 (visits)
- Gumbel-lite falsification: 10-40 (20%), 37% decision-flip rate, 567 flips
- 699 paired decisions (calibration set); SPRT H0=0.50/H1=0.55; ≤200-game screens
- 5-condition intervention matrix (D, C, D+C, D+C+B, B-marginal), one
  container per condition, 50 mirrored games each, per-arm telemetry

## Angle suggestions for the resume agent

- RL/research angle: expert iteration at 142M scale + a genuine
  instrument→diagnose→intervene→preregistered-evaluation loop with an
  honest falsification. Rare and credible.
- Empirical-rigor angle: preregistration, SPRT, mirrored pairs, matched
  Glicko-RD-25 convergence, cross-era benchmark reproduction.
- Systems angle: Rust engine + Python serving + Node simulator containerized
  into a cloud eval farm; deterministic cross-platform search budgets;
  request-bound serving contracts; fail-closed networking gates.
- Keep 3–5 bullets; lead with scale (142M) and the strongest measured
  numbers; use one methodology bullet and (optionally) one diagnosis bullet.

---

# DEEP FRAMING 1: metagross as an RL RESEARCH project

Use this vocabulary; every mapping below is genuine (the work actually
instantiates the concept, not just resembles it).

## Problem formalization
- Two-player zero-sum POMDP / imperfect-information game: hidden opponent
  team (species/moves/items/abilities), simultaneous-move turn structure,
  stochastic transitions (damage rolls, secondary effects).
- The agent maintains an explicit BELIEF over opponent private state: a
  conditional generative sampler over a 50,000-team pool, filtered by
  revealed evidence (moves seen, items/abilities inferred, tera types).
- Planning = determinized search: sample 16–32 worlds from the belief
  (particle-style), solve each as a perfect-information game with PUCT,
  aggregate root statistics across worlds (lineage: PIMC / Information-Set
  MCTS; Cowling et al. 2012).

## Learning algorithm
- Expert Iteration (ExIt; Anthony et al. 2017 / AlphaZero family): the
  142M transformer policy is distilled from MCTS visit distributions over
  search-generated self-play, then redeployed as the search prior —
  iterated (checkpoint lineage r1 → r2 exists).
- Sequence-model policy: AMAGO (UT-Austin) MultiTaskAgent — RL2-style
  reward-first trajectory conditioning, Perceiver timestep encoder +
  transformer trajectory encoder, max context 128 decisions. The policy is
  context-conditioned on the full episode history (memory-based meta-RL
  input format).
- Serving-time ablation axis: "causal-history" (full trajectory
  conditioning, matching training) vs "legacy-stateless" (two-step
  amnesiac input) — an input-sufficiency ablation of the memory channel.

## Research questions actually posed and answered
1. IS HISTORY REDUNDANT GIVEN BELIEF? (ReBeL sufficiency argument — Brown
   et al. 2020: the belief state is a sufficient statistic of history.)
   Tested via serving-mode A/B at scale: head-to-head mirrored play showed
   PARITY (48%); population-level ladder showed a small stateless edge
   (~2 GXE). Conclusion: the policy's history channel adds ~nothing net
   over belief-conditioned search — consistent with belief sufficiency.
2. WHY DOESN'T HISTORY HELP? Instrumented per-decision telemetry and found
   LATE-GAME PRIOR ENTROPY COLLAPSE: mean root-prior entropy 0.96→0.58
   nats by turn 30+ under history conditioning while stateless stays flat
   — a train/serve distribution-shift miscalibration (the policy grows
   overconfident on out-of-distribution long histories). Independently
   replicated on a second platform.
3. CAN POST-HOC CALIBRATION FIX IT? Built entropy-matched temperature
   flattening (turn-bucketed tau schedule, offline-calibrated on 699
   paired decisions — the RL analog of temperature scaling in calibration
   literature). Verified the intervention MOVES THE MECHANISM before
   testing outcomes (prior entropy 0.58→1.40 at 30+; propagates into
   search visit entropy 0.59→0.95). Outcome screen: preregistered SPRT.
4. CAN THE SEARCH BE MADE ROBUST TO PRIOR MISCALIBRATION? Implemented an
   evaluation-time Gumbel AlphaZero-style completed-Q root decision rule
   (Danihelka et al., ICLR 2022; mctx-default constants) — and cleanly
   FALSIFIED it: 10-40 (20%) in a marginal-contribution A/B despite a 37%
   decision-flip rate toward higher pooled-Q actions. Empirical finding:
   the visit-count decision rule performs implicit variance regularization
   (low-visit Q estimates are noise; trusting them loses games). A real
   negative result with a mechanism, honestly reported.
5. DO HISTORICAL BENCHMARKS REPRODUCE UNDER POPULATION DRIFT? The eval
   environment (public ladder) is NONSTATIONARY (meta shifts). Ran a
   contemporaneous matched A/B — both arms to the Glicko RD-25 convergence
   floor, single variable, same machine/window — decomposing a 6-point
   historical gap into: mostly era drift + ~2 GXE real population edge +
   zero head-to-head edge.

## Methodological rigor (say this explicitly)
- Preregistration for every experiment: frozen hypotheses, interpretation
  rules, seeds, stopping criteria BEFORE launch; results graded against
  the prereg including honest MISS and INVALID grades (one screen was
  invalidated and rerun when activation evidence was absent).
- Sequential testing: SPRT (H0=0.50, H1=0.55), Wilson intervals, mirrored
  pairs (both agents play both sides of identical matchups — paired-sample
  variance reduction), fail-fast + snapshot-resume.
- Validity engineering: interventions are env-gated, FAIL-CLOSED on
  misconfiguration, and must emit observable activation evidence; a result
  without activation proof is invalid by prereg. (Two historical screens
  were caught measuring the wrong condition by exactly this rule.)
- Mechanism-before-outcome discipline: an intervention must demonstrably
  move its target statistic before its win-rate is interpreted.

---

# DEEP FRAMING 2: metagross as an ML SYSTEMS project

## Model serving
- Real-time CPU inference server for the 142M transformer (HTTP, health
  endpoints, per-battle session state, checkpoint SHA-256 verification at
  boot; 642-key state-dict validation; ~1.6–2.2 GB RSS/instance; serves a
  500 ms/move decision loop).
- REQUEST-BOUND CONSISTENCY CONTRACT: each served prior is bound to the
  exact game request (rqid + canonical-JSON SHA-256); the client must
  ACKNOWLEDGE the selected action before the next request is served —
  guaranteeing the model's input trajectory is exactly request-aligned
  (no silent desync between what the game asked and what the model saw).
  Idempotent acks, forced-action short-circuits, cached-response replay.
- Per-arm server isolation for A/B: two servers, per-arm trajectory modes
  and env injection, strict client-server binding, port-gated
  interventions (one global env can target exactly one arm).

## Hermetic builds & provenance
- Rust native extension (custom MCTS engine) compiled IN-IMAGE with
  maturin (manylinux wheels), feature-flagged cargo builds; engine binary
  pinned by SHA-256 with a fail-closed runtime provenance guard (server
  refuses to play if the engine lacks required native capabilities).
- Source provenance manifests: git HEAD SHAs captured at deploy time and
  reconstructed in-container (mounts strip .git); every run's manifest
  records exact commits, checkpoint SHA, dataset SHA, RNG scheme.
- Reproducibility: deterministic run seeds, frozen configs, byte-identical
  no-op guarantees for absent feature flags.

## Heterogeneous stack orchestration
- One container runs: Node.js Pokémon Showdown simulator + two Python
  model servers + Rust-engine search workers (16 processes) — process
  supervision, port lifecycle management, readiness gating, log capture.
- Cross-language integration: Python↔Rust via PyO3 wheels; Python↔Node via
  the Showdown websocket protocol and protocol-line teeing into the causal
  ledger; Python↔Python via the bound-prior HTTP contract.

## Determinism & fair benchmarking
- Replaced wall-clock search budgets with DETERMINISTIC ITERATION BUDGETS
  (472,000 iterations/500 ms, calibrated to the reference machine's
  median): search depth becomes load- and platform-invariant, enabling
  fair A/Bs on shared CPUs, concurrent local runs, and cloud/local
  comparability. (Analogy: fixed-work benchmarking / deterministic replay.)

## Cloud infrastructure (Modal serverless)
- Evaluation farm: one experimental condition per container (cpu=16),
  spawn/poll async job management, per-lane artifact + telemetry
  persistence to shared volumes, image-layer caching strategy.
- Deploy-time optimization: diagnosed a 45-minute deploy stall as 22 GB of
  training checkpoints silently swept into a code mount; ignore-pattern
  surgery cut mounts to code-only and deploys to minutes.
- Probe-driven debugging methodology: before burning full experiment
  fleets, cheap 1–2-game probe containers validated each fix; an 11-layer
  serial failure chain (mount bloat → engine provenance → secret schema →
  git provenance → interpreter paths → hidden site-packages patch →
  simulator deps masked by `|| true` → missing runtime dirs → API contract
  gaps → action-ack protocol → decision-rule support constraint) was
  resolved at ~$0.50/probe instead of ~$15/fleet-launch.
- Cost engineering under a hard budget: lane triage by information value,
  cancellation economics, incident postmortem of a wedged container
  (stuck-burn) that produced a new watchdog rule (runtime > 2x expected
  alarms loudly).

## Networking
- Cloud-to-service access via the owner's own residential connection:
  Tailscale userspace networking (no TUN privileges in containers) + a
  SOCKS5 relay chain (container-local asyncio byte-pipe → tailnet → host
  microsocks), with a FAIL-CLOSED EGRESS GATE — the run refuses to start
  unless a live probe confirms egress from a non-datacenter IP (IP + ASN
  checks), so misconfiguration can never leak the wrong network identity.
- Websocket-level client instrumentation: monkeypatched connect for
  SOCKS routing, reconnect-with-replay verification (refuses to continue
  a game on unverifiable public-history replay), per-battle abandon
  semantics instead of process crashes.

## Reliability & observability
- Resilient supervisors: auto-resume loops keyed on externally persistent
  state (ladder ratings survive process death), snapshot-resume for
  mirrored-pair evals (progress JSONL), per-battle graceful degradation.
- Observability: per-decision telemetry (JSONL) for every arm of every
  experiment; activation-evidence lines as machine-checkable validity
  gates; log-growth stall detectors; alarm-bearing monitors for every
  long-running job.
- Incident discipline: every failure got a written postmortem in an
  append-only iteration log (100+ entries) with root cause, fix, and the
  generalized lesson (e.g., "fail-open must pair with positive activation
  evidence", "every remote watcher needs a runtime ceiling").

---

# Ready-to-use bullet sets

## RL-research flavored (pick 4–5)
- Fine-tuned a 142M-parameter transformer policy with expert iteration
  (AlphaZero-style distillation from MCTS visit distributions over
  search-generated self-play)
- Planned under partial observability with belief-conditioned determinized
  PUCT: 16–32 opponent worlds sampled per move from a 50,000-team
  conditional generator (ISMCTS/PIMC lineage)
- Reached 92.7 GXE and 2399 peak Elo on the public ladder — the strongest
  documented Gen 9 Random Battle bot
- Diagnosed late-game prior miscalibration (entropy collapse 0.96→0.58
  nats) via per-decision search telemetry; verified an entropy-matched
  temperature intervention moves the mechanism before outcome testing
- Falsified a Gumbel-AlphaZero-style completed-Q decision rule in a
  preregistered marginal-contribution A/B (20% vs identical baseline),
  isolating visit-count decision-making as implicit variance regularization
- Ran contemporaneous matched A/Bs to Glicko RD-25 convergence (~470
  games) with SPRT screens and preregistration, reproducing cross-era
  benchmarks within ~1 GXE

## ML-systems flavored (pick 4–5)
- Served a 142M transformer for real-time (500 ms/move) decisions with a
  request-bound consistency contract: priors bound to request rqid +
  canonical SHA-256 with mandatory selected-action acknowledgements
- Containerized a heterogeneous stack (Rust MCTS engine via maturin,
  PyTorch model servers, Node.js game simulator) into a serverless
  evaluation farm with per-condition lanes and volume-persisted telemetry
- Made search depth platform-invariant with deterministic iteration
  budgets (472k/500 ms), enabling fair A/Bs across cloud and local CPUs
- Enforced provenance fail-closed: SHA-pinned engine binaries, deploy-time
  git-commit manifests, checkpoint hash verification at server boot
- Built fail-closed residential egress for cloud clients (Tailscale
  userspace networking + SOCKS relay + live IP/ASN verification gate)
- Cut cloud deploys from 45 min to minutes by eliminating 22 GB of mount
  bloat; debugged an 11-layer failure chain with $0.50 probe containers
  instead of $15 fleet launches

---

# ML-SYSTEMS TECHNICAL INVENTORY (the stuff screeners probe for)

## Performance & latency (real numbers)
- HARD REAL-TIME BUDGET: 500 ms/move decision loop containing one 142M
  transformer forward (context ≤128 decisions) PLUS 16–32 determinized
  MCTS searches, under a game clock with forfeit consequences.
- Rust engine throughput: ~944k MCTS iterations/sec/core (472k iterations
  in 500 ms single-thread, release build) — measured, and used as the
  calibration constant for cross-platform budgets.
- Measured cloud/local perf ratio (0.55x on shared cpu vs Apple M4) and
  engineered around it with fixed-work budgets instead of accepting skew.
- CPU-only inference deliberately (TORCHDYNAMO_DISABLE=1 after Inductor
  compile failures — eager-mode serving; OMP_NUM_THREADS tuned; ~1.6–2.2 GB
  RSS per server instance; two servers + 16 search workers inside a 24 GB
  laptop or a cpu=16 container).
- Throughput at fleet level: ~12 games/hour/lane, 5 experimental conditions
  in parallel containers; ~3,000 telemetry'd decisions per 50-game lane.

## Concurrency & correctness
- Per-battle session locks, fcntl-flock account locks, per-port supervisor
  locks (deliberate-concurrency escape hatch: lock scoped by port so a
  designed A/B pair can run while duplicates on one port are refused).
- ThreadingHTTPServer model servers; process-pool (fork) search workers;
  module-level picklable/forkable search entrypoints (fork-safety-aware).
- Idempotency & exactly-once: cached-response replay keyed by (rqid,
  request-SHA); idempotent action acknowledgements; snapshot-resume for
  evals (progress JSONL); reconnect-with-replay verification that REFUSES
  to continue a game whose replayed public history doesn't extend the
  verified prefix (correctness over availability, per-game blast radius).
- Numerical robustness: all-masked softmax → NaN prior detection and
  symmetric discard (with regression tests); nan_to_num on model inputs;
  finite-reward assertions in trajectory assembly.

## Feature flags & config discipline
- Every intervention is an env-gated feature flag with a BYTE-IDENTICAL
  no-op default, fail-closed parsing (set-but-malformed = crash, never
  silent degradation), and machine-checkable activation evidence.
- Per-arm targeting of global flags via port-scoped gates (one env var,
  exactly one treatment arm — A/B contamination structurally impossible).
- Execution-identity capture on every run: full argv, environment, and a
  config SHA-256 recorded into results (experiment tracking without a
  tracking service).

## Data engineering
- Protocol-ingestion pipeline: raw Showdown protocol lines teed in
  real-time into a causal reveal ledger (evidence extraction for
  moves/items/abilities/tera with authority rules and contradiction
  handling), feeding the belief sampler.
- Schema-versioned JSONL datasets everywhere: per-decision search
  telemetry, decision dumps, holdout metrics, protocol captures, rating
  streams — all append-only, resumable, and diffable.
- Dataset/checkpoint provenance: SHA-256 of the random-battle dataset,
  checkpoint hash verified at server boot (642-key state dict, param-count
  cross-check), git commits of four repos recorded per run.
- Post-hoc analytics on the telemetry (entropy-by-turn-bucket, decision
  flip attribution, win-rate-by-game-length) drove every research verdict.

## Security & secrets hygiene
- Credentials never in argv or logs: passwords passed via env into child
  processes only; owner-created chmod-600 cred files; cloud secrets in
  Modal's secret store with key-NAME-only introspection during debugging
  (values never read); per-arm secret attachment to prevent collisions.
- Masked identifiers in all reporting; egress identity verified before any
  authenticated connection (fail-closed IP/ASN gate).

## Build & release engineering
- Multi-stage container images with layer-cache-aware ordering; native
  Rust extension built in-image (maturin, manylinux, feature-flagged
  cargo); npm runtime-only installs with a fail-closed load check
  (`node -e "require('./dist/sim')"` as a build assertion) after a masked
  install failure caused silent runtime forfeits.
- Reproduced a hidden hand-patch to a site-packages dependency (found by
  diffing installed tree vs pristine upstream at the same tag) and
  codified it as an image build step with a build-time assertion — turning
  tribal knowledge into infrastructure.
- Deploy retry loops keyed on exit codes (not log greps) after observing
  false-success detection; deploys verified by functional probes, not
  build success.

## Incident engineering (tell these as stories)
- Stuck-container burn: a wedged cloud lane billed idle cpu-16 for a day
  while reporting RUNNING; killed, postmortem'd, and every subsequent
  watcher got a runtime-ceiling alarm (>2x expected = loud alert).
- The 22 GB mount: deploys stalling 45 min traced to training checkpoints
  swept into a code mount; ignore-pattern surgery → minutes. Lesson
  codified: code mounts must never carry artifacts.
- Silent-inert instrumentation (twice): a fail-open hook that measured the
  wrong condition invalidated two experiments; response was structural —
  observable activation lines became MANDATORY validity gates checked by
  the harness itself.
- 11-layer serial failure chain on cloud bring-up, each layer found by a
  ~$0.50 probe container before risking a ~$15 fleet launch.

## Extra bullet candidates (systems-flavored, pick to taste)
- Sustained a 500 ms/move real-time loop combining 142M-transformer
  inference with ~1M-iteration/sec/core Rust MCTS across 16 workers
- Enforced experiment validity in infrastructure: byte-identical-default
  feature flags, fail-closed misconfiguration, machine-checked activation
  evidence, execution-identity (argv+env+config SHA) stamped on every run
- Built an append-only protocol-to-dataset pipeline (schema-versioned
  JSONL telemetry, provenance-hashed datasets) powering all analyses
- Designed idempotent, resumable evaluation: snapshot-resume, cached
  response replay keyed by request hash, reconnect replay verification
  with per-game blast radius
- Ran incident-driven reliability: postmortems for every failure class,
  codified into watchdogs (runtime ceilings, stall detectors, build-time
  load assertions)

---

# PURE-RL FRAMING (use this alone if the resume slot is "RL research project")

Rule for the agent: in this framing, infrastructure is INVISIBLE — every
sentence is about agents, policies, search, beliefs, learning, and
evaluation. No containers, no networking, no deploys.

## The RL story arc (one paragraph)
Built a game-playing agent for a two-player, simultaneous-move,
imperfect-information stochastic game (Pokémon Gen 9 Random Battles):
a 142M-parameter sequence-model policy trained by expert iteration on
search-generated self-play, deployed as the prior inside belief-conditioned
determinized MCTS. Then treated the deployed agent as a research subject:
measured it against the strongest historical baselines under population
drift, discovered and replicated a policy miscalibration phenomenon
(late-game prior entropy collapse under long-context conditioning), designed
calibration- and search-side interventions grounded in the literature
(temperature scaling, Gumbel policy improvement, belief-sufficiency
truncation), and adjudicated each with preregistered sequential tests —
including a clean falsification.

## RL-technical inventory (all real; vocabulary matters)
- POLICY: 142M-parameter memory-based policy (AMAGO MultiTaskAgent) —
  Perceiver-style per-timestep observation encoder + transformer trajectory
  encoder over an RL²-style reward-and-action-augmented sequence (context
  up to 128 decisions). Mixed offline/online objective (offline loss weight
  1.0, online 0.25). 13-action discrete space with legality masking.
- TRAINING LOOP: Expert Iteration — self-play games generated WITH search,
  policy distilled toward the search's visit distributions, redeployed as
  the search prior; multi-round checkpoint lineage (r1 → r2, plus control
  and ablation trains incl. an HL-Gauss-style value-loss variant and
  visit- vs action-target distillation students).
- SEARCH AS POLICY IMPROVEMENT: PUCT with learned root priors for BOTH
  players (own priors s1 + opponent-model priors s2), c_puct 2.0; the
  engine also exposes paired-root policy evaluation and
  shared-information-set root search variants (used in earlier gates).
- PLANNING UNDER PARTIAL OBSERVABILITY: belief over opponent private state
  = conditional generative sampling from a 50,000-team pool filtered by a
  causal reveal ledger (evidence-consistent sets only); 16–32 sampled
  worlds per decision, determinized perfect-information search per world,
  cross-world aggregation at the root (PIMC/ISMCTS family).
- OPPONENT MODELING: opponent root priors from the same policy (role-
  flipped inference); an action-belief opponent-conditioning extension was
  built and measured to a CLOSED NULL (162-170, LLR −2.47) — reported
  honestly, not shelved.
- CREDIT/VALUE: terminal win/loss reward; reward stream threaded through
  the RL² input; leaf evaluation via engine rollout/eval with a hand-
  crafted leaf evaluator component; learned-value and value-horizon
  variants explored in earlier training rounds.
- EXPLORATION/DECISION RULES: PUCT visit-count decision rule vs Gumbel
  completed-Q evaluation-time rule (implemented per Danihelka et al. 2022,
  mctx constants) — falsified head-to-head (20% vs identical baseline),
  yielding a transferable finding: visit counts act as implicit variance
  regularization over noisy Q estimates in determinized search.
- DISTRIBUTION SHIFT / CALIBRATION: identified train-serve context
  mismatch as the driver of late-game prior overconfidence (entropy
  0.96→0.58 nats by turn 30+, replicated cross-platform); intervention =
  entropy-matched per-turn-bucket temperature scaling calibrated on 699
  paired decisions (post-hoc calibration, Guo-et-al-style, applied inside
  the search prior); verified the intervention shifts both prior AND
  induced search-visit entropy before interpreting outcomes.
- BELIEF SUFFICIENCY (ReBeL question): tested whether raw history is
  redundant given the belief state — serving-mode A/B (full-context vs
  amnesiac policy input) found head-to-head parity (48%) and only a ~2 GXE
  population-level gap: the belief-conditioned search, not the policy's
  history channel, carries the hidden-information value.
- EVALUATION SCIENCE: nonstationary evaluation environment (live ladder
  meta drift) handled by contemporaneous matched A/B to a fixed
  convergence standard (Glicko RD-25 floor, ~470 games); mirrored-pair
  head-to-heads (both agents play both sides of identical matchups) for
  paired-sample variance reduction; SPRT (H0 .50/H1 .55) sequential
  stopping; preregistration with frozen interpretation rules; results
  graded honestly including MISS, NULL, and INVALID.

## Numbers for the RL framing
- 142,832,563 params; context 128; 13 actions; offline/online loss 1.0/0.25
- 16–32 worlds/decision; c_puct 2.0; 472k search iterations per decision
  (deterministic budget) or 500 ms
- Entropy collapse 0.96→0.58 nats (turn 30+), replicated; intervention
  moves prior 0.58→1.40 and visit entropy 0.59→0.95
- Gumbel falsification 10-40 (20%, CI [11,33]); 37% decision-flip rate
- Serving-mode A/B: 48% head-to-head; 91.7 vs 89.4 GXE population-level,
  both at Glicko RD 25 (1952±25 vs 1902±25), ~470 games
- Benchmarks: 92.7 GXE historical peak / 2399 Elo peak; strongest
  documented Gen 9 Random Battle bot
- 699-decision calibration set; 200-game SPRT screens; closed nulls
  reported (action-belief D1: 162-170, LLR −2.47)

## Pure-RL LaTeX bullet set (pick 4–5)
\resumeItem{Trained a \textbf{142M-parameter} sequence-model policy by \textbf{expert iteration} — distilling MCTS visit distributions from search-generated self-play over multiple rounds — and deployed it as the prior in \textbf{PUCT} search}
\resumeItem{Planned under \textbf{partial observability} with belief-conditioned determinized MCTS: 16--32 opponent worlds per decision sampled from an evidence-filtered generative belief over a 50{,}000-team pool, with learned priors for both players}
\resumeItem{Reached \textbf{92.7 GXE} and \textbf{2399 peak Elo}, the strongest documented Gen 9 Random Battle agent; reproduced cross-era benchmarks within $\sim$1 GXE via matched A/Bs run to \textbf{Glicko RD-25} convergence}
\resumeItem{Discovered and replicated a \textbf{prior miscalibration} phenomenon (late-game entropy collapse, 0.96$\to$0.58 nats) from per-decision search telemetry; designed an entropy-matched \textbf{temperature-scaling} intervention and verified it shifts both prior and induced search entropy}
\resumeItem{Implemented and \textbf{falsified} a Gumbel-AlphaZero-style completed-Q decision rule in a preregistered \textbf{SPRT} ablation (20\% vs identical baseline), isolating visit counts as implicit variance regularization in determinized search}
\resumeItem{Tested the \textbf{ReBeL belief-sufficiency} hypothesis with a serving-time memory ablation: head-to-head parity showed the belief-conditioned search, not the policy's history channel, carries the hidden-information value}

## VERIFY-WITH-OWNER before using (from checkpoint/run artifacts, details
   not re-confirmed this session)
- Exact self-play game counts per ExIt round (artifacts show 1k/3k/6k-game
  training runs; confirm which fed r1)
- The HL-Gauss variant and visit- vs action-target students exist as
  checkpoints; confirm they were evaluated before citing results for them
- "Multiple rounds" of ExIt: r2 checkpoints exist; confirm r2 was completed
  and evaluated if claiming more than one deployed round
