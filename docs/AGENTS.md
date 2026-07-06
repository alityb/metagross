# AGENTS.md — Pokémon Showdown Singles Agent

> Playbook for any agent (human or LLM) working in this repo. Read this top to
> bottom before touching code. The goal is not a clever demo; it is the highest
> ladder **ELO / GXE** achievable, validated by thousands of timed games.

---

## 0. Mission

Build the strongest Pokémon Showdown **singles** agent measured by ladder **ELO**
and **GXE** (Glicko-derived expected win rate vs. the ladder population).

Bar for success, in order:
1. Beat the public **Foul Play** (search) and **Metamon / PA-Agent** (RL) baselines head-to-head on a fixed format.
2. Reach the top of the human ladder for that format.
3. Approach or exceed **elite** human GXE — the gap the PokéAgent Challenge report says is still open.

### Target formats — attack in this order
The order is dictated by *hidden-information size* and *engine maturity*, not preference.

| Priority | Format | Why this order |
|---|---|---|
| 1 | `gen1ou` | Smallest hidden-info space (no items/abilities, tiny viable roster), densest replay coverage per matchup, fastest exact engine (`libpkmn`). RL won this at the challenge — we should win it with **search on top of** a learned eval. |
| 2 | `gen9randombattle` | The team generator is **public code** → set probabilities are *known* → beliefs can be near-exact. This is the cleanest place on the entire ladder to validate the belief module. |
| 3 | `gen9ou` | The prize. Combinatorial breadth (items, tera, huge roster), sparse data relative to state diversity, and engine gaps (dynamax / z-moves not yet covered by `poke-engine`). Last because it is hardest, not because it is least important. |

Do **not** start on `gen9ou`. Prove the loop where it is cheap, then port.

---

## 1. The thesis (one sentence)

> **Belief-conditioned expert iteration:** an Oak-style loop — ultra-fast
> simulation (`libpkmn` / `poke-engine`), information-set MCTS with **Exp3** at
> simultaneous-move nodes, and a small CPU-speed value/policy net continually
> retrained on its own search-improved self-play data — **warm-started on the
> Metamon human-replay corpus**, with determinizations sampled from a **learned,
> archetype-bucketed belief network** over opponent sets, and a **bounded
> exploitation dial** tilting the equilibrium policy toward best-response against
> the ladder population.

Everything below is in service of building and falsifying that sentence, one
component at a time.

---

## 2. Why this design (the empirical grounding)

These are the facts the architecture is built on. Full sources in §9.

- **The two benchmark winners split by paradigm.** At the NeurIPS 2025 PokéAgent
  Challenge, Gen 9 OU was won by **Foul Play (search, zero learning)** and Gen 1
  OU by **PA-Agent (RL, no search)**. This architecture is each winner's missing
  half: search gets a learned eval; the learned policy gets test-time search.
- **The weld already works at toy scale (Oak).** A small network trained on
  search-generated self-play data beat Monte-Carlo eval **95–5**; Monte-Carlo
  eval itself beats Foul Play's *hand-crafted* eval. The engine (`libpkmn`) runs
  **~1000× faster** than Showdown's JS sim with exact behavior matching,
  sustaining **~1M MCTS iterations in seconds** on CPU.
- **Naïve simultaneous-move search is unsound.** Joint-node MCTS with standard
  UCB does **not** converge to low-exploitability play regardless of iteration
  count. **Exp3** (adversarial bandit) provably converges to equilibrium. Use it.
- **The objective is a population metric, and humans are the population.**
  Metamon-lineage agents rate **90–99th percentile** vs. humans and were trained
  on a mix of **~1M human + ~4M self-play** battles. GXE = expected win rate vs.
  the ladder, so the human anchor is not optional flavor — it *is* the target.
- **Belief distributions are the principled hidden-info state.** ReBeL's theory
  says the correct state expansion for imperfect-info games is the common-knowledge
  belief distribution. Exact posteriors are infeasible at Pokémon's type-space
  size, so we **amortize beliefs into a net** and bucket sets into archetypes.
- **The gap to elite humans is still real.** The challenge report documents it
  explicitly. "Best bot" is genuinely unclaimed — this is a research bet, not a
  reimplementation.

---

## 3. Architecture (components & contracts)

```
                    ┌─────────────────────────────────────────────┐
                    │              EXPERT ITERATION LOOP            │
                    │  self-play(search) ─► data ─► train net ─►──┐ │
                    │        ▲                                    │ │
                    │        └────────── new net ◄────────────────┘ │
                    └─────────────────────────────────────────────┘
                                       │ net (value + policy)
                                       ▼
  ┌────────────┐   beliefs    ┌──────────────────┐   moves   ┌──────────────┐
  │ BELIEF NET │ ───────────► │  ISMCTS + Exp3   │ ────────► │ SHOWDOWN I/O │
  │ (sets→arch)│  determinize │  (root solver)   │  mixed σ  │  (poke-env)  │
  └────────────┘              └──────────────────┘           └──────────────┘
        ▲                              │ leaf eval                   │
        │ replay labels                ▼                             │
   ┌─────────┐                 ┌──────────────┐              live ladder / H2H
   │ REPLAYS │                 │  FAST ENGINE │
   │(Metamon)│                 │libpkmn/poke- │
   └─────────┘                 │   engine     │
                               └──────────────┘
```

| Component | Responsibility | Initial choice | Replaced by |
|---|---|---|---|
| **Engine** | Simulate moves at exact Showdown semantics, fast | `libpkmn` (early gens) / `poke-engine` (gen9) | — |
| **Search** | Pick a mixed strategy per turn under hidden info | ISMCTS + **Exp3** at sim-move nodes | + CFR re-solve at pivotal turns (Phase 4) |
| **Eval net** | Leaf value + policy prior, CPU-fast | Small MLP / NNUE-style, value as classification | bigger net only if it pays |
| **Belief net** | Posterior over opponent sets given revealed info | usage-stat / known-generator prior (baseline) | learned, archetype-bucketed (Phase 3) |
| **Exploitation dial** | Tilt σ from Nash toward best-response vs. population | scalar `λ` (0 = Nash, 1 = greedy BR) | restricted-Nash-response (Phase 4) |
| **Trainer** | Expert iteration: gen data → train → repeat | warm-start from replay BC | — |
| **Eval harness** | Measure ELO/GXE honestly | H2H vs. fixed pool + live ladder | — |

**Hard interface rule:** search **never** calls the Showdown JS sim. All
rollouts go through the fast engine. (This is the single most important reason
Oak beats Showdown-bound projects; see §7.)

---

## 4. Build order — phases, hypotheses, kill criteria

Each phase has a **Goal**, a **Hypothesis (H)**, a **Kill criterion** (when to
declare it false and stop), an **Exit criterion** (when to advance), and
**Steps**. Do not advance on a result inside the noise band (see §6).

### Phase 0 — Infrastructure & baselines
**Goal:** Be able to run, play, and measure before changing anything.
**Steps:**
1. Stand up the fast engine (`libpkmn` for gen1; `poke-engine` for gen9) and a
   Showdown client (`poke-env` or equivalent). Confirm engine output matches
   Showdown on a battery of scripted turns.
2. Reproduce **Foul Play** and a **Metamon** baseline as opponents. Pin versions.
3. Build the **eval harness** (§6): head-to-head with paired teams + a live-ladder
   runner that logs ELO/GXE over time.
4. Record baseline ELO/GXE for Foul Play and Metamon on `gen1ou` and
   `gen9randombattle`. **These numbers are the yardstick for every later claim.**
**Exit:** harness produces stable, reproducible ELO with confidence intervals and
passes the powered self-play scorer gate in §6.

### Phase 1 — THE FIRST TEST: learned eval inside Foul Play  ⬅ START HERE
**Goal:** Cheapest possible falsification of the whole thesis.
**H:** Replacing Foul Play's hand-crafted leaf eval with a **learned value net**
raises ladder ELO with *no other change*.
**Why first:** It reuses Foul Play's entire (working) search + inference stack,
isolates exactly one variable, and is already the project's queued next step. If
search + a learned eval is the right direction, this moves ELO before you build
anything novel. If it doesn't, the thesis is in trouble — find out now.
**Steps:**
1. Train a small value net on available targets (start by distilling Oak/MC-eval
   self-play, or BC value targets from replays). Value as **classification**
   (win/loss/■ buckets), not raw regression — more stable.
2. Swap it in as Foul Play's eval. Keep determinization, inference, search
   identical.
3. Lock the search budget **before** the A/B and re-record stock Foul Play at
   exactly that budget. Default to Foul Play's current CLI default (`100ms`); if
   using `25ms` for throughput, label it as a low-budget experiment and do not
   generalize it to default-strength play. Learned-eval gains are budget-dependent.
4. A/B vs. stock Foul Play: ≥ ~1000 paired games (both sides of each matchup),
   report win rate + binomial CI.
**Kill:** learned eval ≤ hand eval (CI includes 50%) after honest tuning → the
"search + learned value" leg is weaker than assumed; re-examine before Phase 2.
**Exit:** statistically significant ELO gain over stock Foul Play.

### Phase 2 — Stand up the expert-iteration loop (gen1ou)
**Goal:** Own the full Oak-style loop end to end on the easiest format.
**H:** Iterating *self-play(search) → train net → stronger net → repeat*, with
**Exp3** at simultaneous-move nodes and a **replay BC warm-start**, produces a
net that beats the Phase-1 agent and climbs each iteration.
**Steps:**
1. Warm-start the policy/value net with behavior cloning on Metamon replays
   (skips the cold-start phase that makes self-play expensive).
2. Implement ISMCTS with **Exp3** (not joint-UCB) at sim-move nodes. Verify
   exploitability drops with iterations on a toy subgame.
3. Run the loop on `gen1ou`. After each generation, eval vs. **held-out**
   opponents (Foul Play, Metamon ladder of skills) — never only vs. previous self.
4. Track ELO per generation; expect monotone-ish improvement that plateaus.
**Kill:** ≥ 3 successive generations with no held-out ELO gain beyond noise →
loop is not compounding; suspect data quality, target leakage, or eval-net bias.
**Exit:** loop beats the Phase-1 agent and Foul Play on `gen1ou` with significance.

### Phase 3 — Learned belief network (validate on randbats first)
**Goal:** Replace usage-stat determinization with a **calibrated** posterior.
**H:** Sampling determinizations from a learned, **archetype-bucketed** belief net
beats sampling from raw usage stats, because better worlds → better search.
**Steps:**
1. **Start on `gen9randombattle`**, where the generator is public → ground-truth
   set probabilities are known. This isolates belief *quality* from belief
   *learning* and gives a clean upper bound.
2. Train the belief net **supervised** on replays: input = revealed info at turn
   *t*; label = the *eventually revealed* full set. Bucket sets into archetypes
   (e.g. "Scarf Lando" vs. "defensive Lando") and predict over buckets.
3. **Calibrate separately from win rate** (Brier / reliability curves). A
   miscalibrated belief net poisons every downstream search — calibration is a
   first-class metric here, not an afterthought.
4. Only after randbats validation, port to `gen1ou` / `gen9ou` (adversarial,
   nonstationary meta — harder, expect drift).
**Kill:** learned beliefs ≤ usage-stat beliefs on win rate **and** no calibration
edge → determinization is not the bottleneck; redirect effort to search/eval.
**Exit:** belief net improves both calibration and downstream ELO over the
usage-stat baseline.

### Phase 4 — Equilibrium refinement & safe exploitation
**Goal:** Reduce exploitability where it matters and *capture* population EV.
**H(a):** CFR/regret **re-solving at pivotal turns** + **exact solving of
late-game subgames** (Pokémon's hidden info decays toward perfect info as sets
reveal) lowers exploitability without hurting ladder EV.
**H(b):** A **bounded** exploitation dial (restricted-Nash-response style) toward
best-response vs. the *human population* raises GXE over pure-Nash play.
**Steps:**
1. Flag pivotal turns (high value-variance across determinizations); re-solve those
   subgames with CFR instead of trusting determinized MCTS.
2. Solve fully-revealed endgames exactly.
3. Add scalar `λ` exploitation dial; sweep `λ` on the live ladder; keep the
   exploitability cost bounded.
**Kill:** equilibrium machinery costs ladder EV (humans punish unexploitable but
"timid" lines), or exploitation dial gains nothing → ladder population is already
well-handled by Phase 2–3; lock `λ` and move on.
**Exit:** measurable GXE gain at acceptable, bounded exploitability.

### Phase 5 — Scale & format expansion
**Goal:** Turn a working loop into a strong bot across formats.
**Steps:** distributed/volunteer data generation (Stockfish/Leela model); port to
`gen9ou` as `poke-engine` matures (dynamax/z-move coverage gates this); add team
co-optimization (beyond Oak's 1v1 PPO toy) once the battle agent is strong.
**Exit:** state-of-the-art ELO/GXE on ≥ 2 formats; documented gap to elite humans
shrinking.

---

## 5. The first thing to try (expanded)

If you read nothing else, do **Phase 1**: drop a learned value net into Foul Play
in place of its hand-crafted eval, change nothing else, and A/B on ladder.

- It is the **highest information-per-hour experiment** in the whole plan.
- It tests the load-bearing assumption ("search + learned value > search + hand
  eval") in isolation, with a stack that already works.
- Oak already showed the *direction* (search-trained net ≫ MC eval ≫ Foul Play
  hand eval) at toy scale; Phase 1 checks it **on ladder, at format scale**.
- A clean win green-lights Phases 2–4. A clean loss is the single most valuable
  negative result you can get, and you get it cheaply.

---

## 6. Evaluation methodology (read before reporting any number)

Pokémon has **high per-game variance** (crit rolls, damage rolls, accuracy).
Sloppy evaluation will manufacture phantom wins. Rules:

1. **Paired matchups, both sides.** For any A/B, play each team matchup from both
   sides to cancel team-strength bias.
2. **Sample size.** A ~3% effect needs ~1000+ games before its binomial CI clears
   50%. State **N** and the **confidence interval** on every comparison. No CI, no
   claim.
3. **Powered self-play scorer gate.** Before trusting a harness for A/B decisions,
   run stock-agent self-play at the intended search budget with `N >= 100`, paired.
   The point estimate for side A must be in `[45%, 55%]`, the 95% CI must contain
   `50%`, the full 95% CI must be contained in `[40%, 60%]`, and there must be no
   unexplained ties or unknown winners. A small run whose CI merely straddles 50%
   is a smoke check, not a trust gate. If this fails, debug scoring before running
   Phase 1.
4. **Held-out opponents.** Always evaluate vs. a *fixed external pool* (Foul Play,
   Metamon skill ladder) and the **live human ladder** — never only vs. your own
   previous generation. Self-play-only metrics overstate progress.
5. **Noise band = no decision.** If the CI includes 50% (or the prior best), the
   result is noise. Do not advance a phase, do not promote a model.
6. **Calibration is its own metric** (Phase 3): Brier score / reliability curves
   for the belief net, separate from win rate.
7. **GXE the way it's scored:** enough laddered games for the Glicko deviation to
   shrink; report GXE alongside raw ELO.
8. **One variable per experiment.** If you changed two things, you learned nothing.

### Experiment log schema (append-only, one row per run)
```
run_id | date | phase | format | change (ONE var) | baseline | N_games |
winrate | CI95 | ladder_elo | gxe | belief_brier | decision(advance/iterate/rollback) | notes
```

---

## 7. Guardrails — do not do these

- **Never run search on the Showdown JS simulator.** It is orders of magnitude
  too slow; search dies. Use `libpkmn` / `poke-engine`. (This is *the* reason Oak
  outruns Showdown-bound projects.)
- **Never use plain joint-UCB at simultaneous-move nodes.** It is unsound — no
  number of iterations gives low exploitability. Use **Exp3** / regret matching.
- **Never train pure self-play without a human anchor.** Self-play optimizes
  against itself; ELO/GXE is paid by humans. Keep replays in every training mix
  and regularize toward the BC anchor (the ~1M human / ~4M self-play precedent).
- **Never trust a belief net you haven't calibrated.** Bad posteriors compound
  through every rollout.
- **Never optimize Nash for its own sake.** The objective is *population EV*, not
  unexploitability. Keep the exploitation dial; Nash is the floor, not the goal.
- **Never advance on noise.** See §6.5.
- **Never let eval opponents leak into training.** Held-out means held-out.
- **Never chase a bigger net first.** NNUE-style small CPU nets are the proven
  win; scale the net only after it demonstrably pays.

---

## 8. Repo layout & conventions

```
/engine        # fast sim bindings (libpkmn / poke-engine), exactness tests
/search        # ISMCTS, Exp3, CFR re-solver, endgame solver
/nets          # value/policy net, belief net; arch + checkpoints
  /checkpoints # model cards: train data, params, eval ELO, date
/train         # expert-iteration loop, BC warm-start, data gen
/belief        # belief net training, archetype bucketing, calibration
/eval          # harness: paired H2H, ladder runner, CI stats
/io            # poke-env / Showdown client glue
/data          # replays, self-play shards (compressed), team sets
/experiments   # append-only log (schema in §6), one dir per run_id
AGENTS.md
SETUP.md       # reproducible local setup log
```

**Conventions**
- Every promoted checkpoint ships a **model card**: training data, params,
  held-out ELO/GXE, calibration (if belief), date.
- Every experiment appends one row to `/experiments` per §6 schema. No exceptions.
- One variable per experiment; the change is named in the log row.
- Pin baseline (Foul Play, Metamon) versions; record them in the run.

---

## 9. References

- **PokéAgent Challenge (NeurIPS 2025)** — challenge, baselines (PokéChamp /
  Metamon), retrospective noting RL specialists' dominance and the remaining gap
  to elite humans. https://pokeagent.github.io ;
  Smogon writeup: https://www.smogon.com/forums/threads/3772550/
- **Oak — "Stockfish for RBY"** — fast-engine + ISMCTS + Exp3 + self-play-trained
  CPU nets; search-net beats MC eval 95–5; MC eval beats Foul Play hand eval;
  `libpkmn` ~1000× faster than Showdown; unsoundness of joint-UCB and the Exp3
  fix. https://www.smogon.com/forums/threads/stockfish-for-rby.3770936/ ;
  https://github.com/lab-oak/oak
- **libpkmn engine** — https://github.com/pkmn/engine
- **Foul Play** — determinized MCTS w/ hand-crafted eval + revealed-info inference
  (the strongest public search agent; Gen 9 OU challenge winner).
- **Metamon** — "Human-Level Competitive Pokémon via Scalable Offline RL and
  Transformers" (RLC 2025); winning RL baseline (PA-Agent lineage), 90–99th
  percentile vs. humans, ~1M human + ~4M self-play training mix; ReplayPredictor
  team completion. https://github.com/UT-Austin-RPL/metamon ;
  teams dataset: https://huggingface.co/datasets/jakegrigsby/metamon-teams
- **PokéChamp** — LLM-in-minimax agent; LLM/opponent-modeling baseline.
  https://github.com/sethkarten/pokechamp
- **ReBeL** — "Combining Deep RL and Search for Imperfect-Information Games";
  belief-distribution state expansion (the theory behind the belief net).
- **CFR / Exp3 background** — Zinkevich et al. (CFR); Deep CFR (Brown et al.);
  Exp3 adversarial bandit / equilibrium convergence for simultaneous-move games.

> Note: all ELO/GXE/percentile figures above are baselines to **reproduce in
> Phase 0**, not assumptions to inherit. Verify on your harness before citing.
