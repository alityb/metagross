# Research Ideas: Recent Techniques for the gen9randombattle Ladder Goal

**Date:** 2026-07-02
**Goal these serve:** sustained GXE > 82.8 (stock Foul Play's measured number) on the live
human gen9randombattle ladder, per the pinned Phase-1 plan.
**Sourcing:** three parallel arXiv/web surveys (offline-RL fine-tuning; policy+search
hybrids for imperfect-info games; self-play/population training), then **every
load-bearing paper verified via OpenReview** (scores + reviewer criticisms; API v1 and
review-dump mirrors were used where OpenReview's anonymous access was Cloudflare-gated).
All 20 checked papers exist; two venue claims were corrected; reviewer caveats are
quoted where they matter. Papers older than 2023 are marked FOUNDATIONAL.

**House rules still apply (AGENTS.md §6):** one variable per experiment, paired gates
with CIs, never promote on pool-relative results — only live-ladder GXE decides.

---

## A. Phase 1 — improvements to the current offline fine-tune (cheap, do-now candidates)

Ranked by expected ELO per engineering hour. All run on the existing metamon/AMAGO
stack; each is a ≤1-day change and independently gateable.

### A1. Rating-conditioned training, prompt high rating at test time
- **From:** Maia-2 (arXiv 2409.20553, **NeurIPS 2024 poster, scores 3;4;6;7 — borderline**).
- **Mechanism:** our dataset's mean winrate is exactly 50% by construction (both POVs of
  every game) and spans ~1000–1900-rated players. Plain imitation converges to the
  *average* player. Add the player's rating as a conditioning token; at test time
  condition on top-band rating. Keeps all data (low-rated games still teach dynamics
  and what the mid-ladder opponents we must beat actually do) while aiming the policy
  at high-skill play.
- **OpenReview caveat, taken seriously:** reviewers found only ~2% move-prediction gain
  over per-bucket models, no significance tests; the "prompt high rating → play
  stronger" step extrapolates beyond what the paper demonstrates. Treat as a cheap,
  plausible bet, not proven.
- **Fit here:** our scraped replays carry the `rating` field (verified in our JSONs).
- **Cost:** hours (extra token/embedding in obs text + parser passthrough).
- **Gate:** A/B same fine-tune ± rating conditioning, N=200 paired vs stock FP, then ladder.

### A2. Policy-extraction upgrade: behavior-regularized policy gradient instead of pure AWR
- **From:** "Is Value Learning Really the Main Bottleneck in Offline RL?" (arXiv
  2406.09329, **NeurIPS 2024 poster, 6;7;7;8**).
- **Mechanism:** AWR (metamon's default) can only *reweight dataset actions* and
  under-uses the critic. With 13 discrete actions we can compute the exact expected
  advantage over the policy's softmax and maximize it directly plus a BC/KL term —
  full critic utilization, no sampling.
- **OpenReview caveat:** 4 seeds, no variance; reviewers note the bottleneck is
  *sometimes* value, sometimes policy — i.e., this may or may not be our binding
  constraint. Continuous-control evidence only.
- **Cost:** ~half a day (one loss term + coefficient sweep).
- **Gate:** same-data A/B vs the AWR baseline checkpoint.

### A3. Advantage-filter shape ablation (exp vs binary vs clipped)
- **From:** CRR (arXiv 2006.15134, FOUNDATIONAL) + RCSL theory (arXiv 2206.01079,
  **NeurIPS 2022 poster, 6;6;6;7**).
- **Mechanism:** binary advantage filtering `1[A>0]` is more robust than `exp(A/β)` when
  the critic is noisy — and Pokémon's crit/roll variance makes critics noisy. One-line
  variants, sweep as a battery.
- **Warning encoded in the same literature:** do NOT use win/loss outcome filtering or
  Decision-Transformer-style return conditioning as the primary objective — RCSL's
  reviewed result: return-conditioning is only sound under near-deterministic dynamics
  (reviewers confirmed it underperformed baselines on 3/4 datasets). In a stochastic
  game it credits luck. TD-advantage filtering is the sound version.
- **Cost:** hours. **Gate:** checkpoint battery vs stock FP N=200 each, promote best.

### A4. KL-anchor to the frozen pretrained Kakuna (not only the EMA anchor)
- **From:** RLHF standard practice; WSRL analysis (arXiv 2412.07762, **ICLR 2025
  poster, 6;6;6;8**).
- **Mechanism:** metamon's EMA anchor drifts with the run; a reverse-KL penalty to the
  *frozen* pre-fine-tune checkpoint hard-bounds how far 23k games can pull 142M params.
  Our biggest Phase-1 risk is overfitting/forgetting, and this is the best-understood
  insurance. Optional: brief critic-only warmup before unfreezing the actor.
- **OpenReview caveat (WSRL):** its "don't retain offline data" headline is broader than
  its evidence; we use only its collapse diagnosis, which reviewers did not dispute.
- **Cost:** hours (one frozen forward per step; fine on the H200).
- **Gate:** sweep λ ∈ {0, small, medium}; degenerate-checkpoint check + N=200 vs stock FP.

### A5. Value-classification detail: HL-Gauss vs two-hot
- **From:** "Stop Regressing" (arXiv 2403.03950, **ICML 2024, PMLR v235** — ICML reviews
  not public).
- **Mechanism:** among value-as-classification variants, Gaussian histogram smoothing
  (HL-Gauss) beats two-hot, with gains growing with model size and target noise — our
  regime exactly (one of their demos is a chess transformer). Check what metamon uses;
  swap if it's two-hot.
- **Cost:** ~1 hour. Low ceiling, near-zero risk, compounds with A2.

### A6. More data (the boring lever that usually wins)
- **From:** PokéChamp dataset (arXiv 2503.04094, **ICML 2025**; note its earlier **ICLR
  2025 rejection 3;5;6;6** — criticisms were about the LLM agent, not the dataset:
  3M+ battles, 500k+ high-Elo; verify gen9randombattle coverage) + our own scraper
  (~13k replays and growing ~4k/night).
- **Mechanism:** at 23k trajectories vs a 142M model, dataset size is plausibly our
  binding constraint; Metamon's own ablations showed data scale dominating objective
  tweaks.
- **Cost:** ~1 day to filter/dedup/parse any usable external randbats data; scraper is
  already compounding daily.

**Deliberately excluded from Phase 1** (evidence-based): Decision Transformer /
return-conditioning as primary objective (see A3); IQL/X-QL machinery (built for
continuous OOD-action problems; 13 enumerable actions make it moot; X-QL has known
instability); LoRA/PEFT (no strong result beats full FT + KL anchor at 142M scale);
diffusion-policy O2O literature (continuous-action tooling, irrelevant here);
Reinformer (ICML 2024, real, but return-conditioning risk in stochastic games —
watchlist as auxiliary head only).

---

## B. Phase 2 — self-play / population training (only if the Phase-1 ladder gate passes)

### B1. The backbone: iterative offline RL, annealed human mix, hot collection
- **From:** PA-Agent (gen1ou challenge winner) + Kakuna recipe — PokéAgent Challenge
  report (arXiv 2603.15563, NeurIPS 2025 competition track) + Metamon (arXiv
  2504.04395, **RLC/RLJ 2025**).
- **Recipe, validated twice on our exact stack and metric:** bootstrap from human BC →
  rounds of population battle generation → offline-RL fine-tune per round; anneal
  human-data weight 100% → ~10% but **never 0**; collect at raised sampling temperature
  (exploration + better value targets). Useful round size ≈ 100k–500k battles.
- **Cost:** ~0 engineering (it *is* metamon's finetune/dataset-config pipeline).

### B2. Prioritized checkpoint-pool opponents (PFSP-lite), curated not emergent
- **From:** AlphaStar PFSP (FOUNDATIONAL) + TStarBot-X negative result (arXiv
  2011.13729: faithful league training FAILS at reduced compute) + 4thLesson
  (challenge report App. E.1.3).
- **Mechanism:** generate each round's data against the full pool — heuristics, public
  Metamon baselines, stock FP, all our past checkpoints — sampling opponents weighted
  toward ~50% winrate matchups. Refresh the pool between rounds (the documented jumps
  came from pool changes, not more same-flavor data).
- **Anti-goal:** this is the direct countermeasure to the TaurosV0 trap (80-20 vs FP
  head-to-head, ~1420 on the human ladder — population overfitting).

### B3. KL-to-human-anchor during self-play rounds (piKL-style), rating-conditioned
- **From:** piKL/DiL-piKL → Diplodocus (arXiv 2210.05492, NeurIPS 2022 — the only
  human-population tournament win in this literature) + HR-PPO (arXiv 2403.19648,
  **RLJ/RLC 2024**, published title "...driving agents...").
- **Mechanism:** during self-play fine-tunes, penalize KL from the (rating-conditioned)
  human BC policy. Keeps the policy on the human strategy manifold that the ladder
  actually pays for. Sweep λ; ladder-gate it.
- **Caveat:** Diplomacy is general-sum where the drift theory is stronger; in our
  zero-sum setting this is empirically motivated (TaurosV0), not theoretically forced.

### B4. Reanalyse-style value relabeling of old battle piles
- **From:** MuZero Reanalyse (arXiv 2104.06294, FOUNDATIONAL) + RaE (arXiv 2311.15951,
  **ICLR 2024 poster, 6;6;6;8** — reviewers: contribution is "don't discard data,"
  gains can be marginal).
- **Mechanism:** each round, recompute critic targets over ALL past battles with the
  current critic — converts idle H200 hours into effective sample size without new
  battles (battles are our scarcest resource at ~8s each).
- **Cost:** ~1 day; pure GPU spend; low downside.

### B5. Offline→online handoff hygiene (for the first online-RL round, if ever)
- **From:** Cal-QL (arXiv 2303.05479, **NeurIPS 2023 poster, 5;5;6;6**), PEX (arXiv
  2302.00935, **ICLR 2023 poster, 6;6;5;8**), WSRL (see A4).
- **Mechanism:** calibrated critics (Cal-QL) or a frozen-offline-policy + new-policy
  selector (PEX) to avoid the characteristic collapse at online-fine-tune onset.
  Reviewer caveats on all three (heuristic, domain-narrow) — treat as defaults, not
  gospel; our own gates decide.

### B6. Fast exact simulator for randbats (force multiplier, not an algorithm)
- **From:** the environment-generation paper containing PokeJAX (arXiv 2603.12145,
  **preprint only, no venue** — verification status matters here) ; AGENTS.md §7 already
  mandates fast-engine-only rollouts.
- **Mechanism:** ~8s/battle on Showdown-JS caps us at ~10k battles/day → 3–6 meaningful
  self-play rounds per quarter. A verified fast randbats sim multiplies every Phase-2
  technique. HOWEVER: unverified artifact, gen9 mechanics + generator coverage unknown,
  and our Phase-0 exactness discipline applies. Budget a verification week before
  trusting; poke-engine (Foul Play's Rust sim) is the nearer-term option.

**Down-ranked for Phase 2** (evidence-based): full PSRO/league machinery (all 2024–26
variants — Global PSRO, Simulation-Free PSRO, JBR-PSRO — evaluated on Kuhn/Leduc-class
exploitability, none on human-ladder populations; compute-infeasible at our battle
budget); diversity-regularized populations (Conflux-PSRO's own ablations show diversity
bonuses degrade best-response quality); learned world-model rollouts (no adversarial-
scale evidence; strictly dominated by an exact fast sim); per-opponent online adaptation
(anonymous one-shot ladder makes it near-moot; the in-context transformer already does
the within-battle version); FXP (AAMAS 2023 + JMLR 2025 — sound but designed for
mixed cooperative-competitive; the B2 pool covers our need at lower cost).

---

## C. Endgame — policy+search hybrid (Phase 3; the path to actually *exceeding* stock FP)

The external literature's consensus recommendation collides with this repo's own
falsification record in one place; stated explicitly:

### C-DEAD. Distill the transformer into a leaf VALUE net inside FP's search
- Every survey source (Oak lineage, MAV, AlphaZero heritage) ranks "replace FP's
  hand-crafted leaf eval with a learned value net" #1. **Our repo already falsified
  this five ways** (label source, capacity, iteration, features, representation) and
  the oracle probe found a ~68% contested-accuracy ceiling *with ground-truth inputs*:
  the information isn't in the determinized leaf. External priors do not override our
  own N=1000+ of paired evidence. Anyone re-proposing this must first explain the
  oracle result away. Status: **dead on arrival, do not rebuild.**
- Note: the Oak project's v0.2.0 is publicly attempting exactly this integration for
  FP (gen1). Their result, either way, is free information — watch the repo/thread.

### C1. Transformer policy as PRIOR at decoupled search nodes (PUCT-style) — ALIVE
- **From:** MAV (arXiv 2412.12119, **ICML 2025**): policy+small-search >> policy alone
  at tiny budgets; searchless-chess (arXiv 2402.04494, **NeurIPS 2024 poster 5;6;6;7**,
  camera-ready "Amortized Planning with Large-Scale Transformers"): pure policy
  plateaus *below* policy+search — reviewers confirmed it stays weaker than engines.
- **Why it escapes the C-DEAD ceiling:** priors change *where simulations go*, not leaf
  values. And it's the only original technique in this repo's ledger with a positive
  point estimate (gen1 PUCT prior: 55–56%, CIs straddling 50).
- **Cost:** distill policy head into a small CPU net (142M is ~1000× too slow in-loop),
  ~1–2 weeks. **Gate:** N=400 paired vs stock FP (a +5% effect needs that N).

### C2. Transformer as OPPONENT model at root nodes (the exploitation dial) — ALIVE
- **From:** GenBR blueprint (arXiv 2302.00797, IJCAI 2025) + Metamon's human-anchor
  results; no published ablation exists — we'd be first.
- **Mechanism:** FP's search assumes the opponent searches like FP. Our transformer is
  literally a model of the human population's action distribution. Blend
  λ·(human-predicted opponent action) into opponent-node priors: population
  best-response inside sound search, AGENTS.md's λ-dial finally implemented.
- **Cost:** ~1–2 weeks (root-only batched inference). **Gate:** ladder GXE, since the
  entire point is population EV.

### C3. MAPLE-style shared-tree aggregation across sampled worlds — WATCHLIST
- **From:** arXiv 2605.24139, acceptance claim "IEEE CoG 2026" rests on the arXiv
  comment only (proceedings not out); +Elo results are vs the authors' own baseline in
  small domains. Mechanistically attractive (removes root-parallel strategy fusion,
  amortizes NN evals) but 3–4 weeks of core search rework on one modestly-evidenced
  paper. Revisit if C1/C2 pay.

### C4. Exp3/regret-matching at the root — CHEAP A/B ONLY
- Theory is unambiguous (joint-UCB is unsound; FOUNDATIONAL, Lisý et al.) but the
  empirical counterweight is decisive for priorities: FP won the gen9ou championship
  50–14 *with unsound DUCT*, and our own gen1 Exp3 A/B lost 15.2% [9.4, 23.5]. Ladders
  don't best-respond. Days of work, expect null; only value is unpredictability on
  mixed-strategy turns.

**Known-dead from external evidence, kept for the record:** ReBeL/Student-of-Games
public-belief search at Pokémon scale (SoG itself underperforms AlphaZero-style search
per unit compute; no 2024+ paper made public-belief search cheap — the absence is the
finding); LLM-in-the-loop play (PokéChamp ~1300–1500 Elo, 700+ below stock FP; ICLR
reviewers flagged its reliance on domain tools and weak opponent prediction);
GO-MCTS-style search through a learned observation model (arXiv-only, uncompetitive at
real-time budgets, and we own an exact simulator).

---

## D. Verification summary (OpenReview pass)

- 20/20 papers resolved to real documents; none fabricated.
- Corrections made: "Grandmaster Chess Without Search" is **NeurIPS 2024**, not ICML,
  and is the same paper as "Amortized Planning with Large-Scale Transformers";
  PokéChamp was **rejected at ICLR 2025 (3;5;6;6)** before ICML 2025 acceptance;
  "PokeJAX" is an artifact inside an unpublished preprint, not a venue paper;
  GO-MCTS is CoRR-only; HR-PPO is published at RLJ/RLC 2024 under a changed title.
- Weakest-reviewed load-bearing papers: Maia-2 (3;4;6;7) and Cal-QL (5;5;6;6) — both
  used here only for cheap, individually-gated bets, consistent with their evidence.

## E. Recommended immediate adoption (Phase 1, current fine-tune)

In order, one variable at a time: **A1 (rating conditioning) → A4 (frozen-KL anchor) →
A3 (filter-shape battery) → A5 (HL-Gauss check)**, with A2 and A6 queued behind the
first ladder gate. Everything in B and C waits for the Phase-1 gate number, per the
pinned plan.
