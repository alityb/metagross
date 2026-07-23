# First-Principles Analysis & Plan: gen9randombattle

**Date:** 2026-07-04. Derived from two verified research briefs (game-theory formalization;
empirical ceilings). Citation verification status marked per claim in the underlying briefs;
key sources listed at bottom.

---

## Part 1 — What kind of game this actually is

**1. Formal class.** Two-player, zero-sum, finite-horizon, simultaneous-move stochastic game
with two-sided hidden information — a factored-observation stochastic game (Kovařík et al.,
arXiv:1906.11110), and in Harsanyi's exact sense a Bayesian game whose **common prior is
literally executable public code** (the team generator). Randbats is possibly the only real
ladder game where Harsanyi's common-prior assumption holds *exactly*.

**2. Hidden information is large but evaporates.** ~60–67 bits at turn 0 (6 species from 508,
~3.8 bits of set entropy each; computed from the generator data), decaying monotonically:
~8–9 bits per switch-in reveal, 1–2 per move. By midgame ~10–20 bits remain; by endgame ~0.
Contrast poker: only ~10 bits hidden, but they stay load-bearing to the terminal node.

**3. The decisive theory result — Long et al. (AAAI 2010, DOI 10.1609/aaai.v24i1.7562):**
determinized search (PIMC — exactly Foul Play's architecture) is near-optimal in games with
high *disambiguation* (info revealed fast) and high *leaf correlation* (outcomes depend on
public state). Randbats scores high on both. **The theory predicts FP's paradigm is
near-optimal on this format.** This retroactively explains our entire falsification record —
we spent weeks trying to beat an architecture that theory says is close to the right answer
for this specific game.

**4. Where determinization provably leaks EV** (Frank & Basin, AIJ 1998 — strategy fusion &
non-locality). The residual is a short, enumerable list:
   a. **Information hiding** — fusion makes the agent assume its own hidden info is transparent
      (undervalues concealing tera/unrevealed mons);
   b. **Information gathering** — scouting lines are worth exactly 0 to a determinized search
      (in every sampled world it already "knows");
   c. **Inference from opponent behavior** — non-locality: opponent actions are evidence about
      their sets AND their intentions; i.i.d. world sampling ignores this channel.

**5. The luck ceiling is real and measured.** Top-500 ladder (fetched live): #1 human =
**95.2 GXE / 2586 Elo**; the whole top-10 sits at 90–95 GXE. Even the best player on earth
loses ~5% to the average opponent. Practical GXE ceiling for ANY agent: **~93–96**.

**6. The incumbent's true strength — our baseline was mis-calibrated.** The FP maintainer's
own published numbers (Smogon thread, verified): **88 GXE / 2341 peak at ~7 s/decision with
8–14 parallel determinized worlds**. Our "stock" baseline (2208/82.8) ran FP at **100 ms with
1–2 worlds** — a fraction of its designed strength. The real bar is 88 GXE, and ~5.6 of the
"missing" GXE points between our baseline and the incumbent is *configuration, not research*.

**7. FP's verified gaps** (from reading its source + maintainer statements):
   - gen9 eval has **no PP/stall/wincon/speed/weather terms** (the gen3 eval HAS a low-PP
     penalty; gen9's simply omits it) — maintainer confirms stall/long-horizon blindness;
   - the modeled opponent is a **mirror with full information** — it sees FP's entire team
     incl. unrevealed mons/sets/tera (source: the determinized state passes our full side);
   - **zero value on information** (hiding or scouting) — pathologies 4a/4b above, confirmed;
   - unrevealed-teammate sampling is a **hand-rolled approximation** of the generator, not
     the exact distribution (we own the exact pool);
   - DUCT at simultaneous nodes (unsound per Shafiei et al. 2009; Exp3 fix per Lisý et al.,
     arXiv:1310.8613 — but our gen1 A/B showed Exp3 loses at FP budgets, and FP won the
     PokéAgent gen9 title with DUCT: ladders don't best-respond).
   - Known eval bug-class: no Unaware handling in boost terms (documented Kingambit-vs-Dondozo
     failure in the Smogon thread).

**8. Why our H1 failed, in theory terms.** We injected priors on **our own** action selection
— the side where search + eval is already near-optimal (point 3). The theory locates the
leak on the **opponent side** (point 4c: non-locality; point 7: mirror+full-info opponent).
And there's a direct existence proof that opponent modeling pays in this exact format:
**Athena** (arXiv:2212.13338) hit rank #33 in gen7 randbats using an ML opponent-move
predictor + best response. Maia (KDD 2020) proves human move distributions are learnable.
Our 142M policy — 46–48% top-1 on 2000+-rated human decisions — IS a human model. We wired
it to the wrong side of the tree. (`s2_priors` is already plumbed in our Rust patch.)

---

## Part 2 — The plan (each step: theory basis → verified gap → gate)

**Step 0 — Re-baseline at true config. [1–2 days, CPU]**
Measure FP at maintainer config (≈7s/decision, parallelism 8+) on our harness and ladder
protocol. All subsequent gates compare against THIS, not the 100 ms strawman. Also decide
our compute budget honestly: ladder timer permits ~10–15 s/turn.
*Expected outcome: FP-at-true-config reproduces ~86–88 GXE; the research target becomes
+2–7 GXE on top of that, toward the 93–96 ceiling.*

**Step 1 — Close the verified eval gaps (FP's paradigm, zero learning). [3–5 days each, gated]**
   1a. Port gen3's low-PP penalty to gen9 eval + stall/Toxic awareness (maintainer-confirmed gap).
   1b. Fix Unaware (and audit sibling ability interactions) in boost valuation (documented bug).
   1c. Concealment value: small eval bonus for own unrevealed mons/unused tera (counters
       pathology 4a at the leaf, the only place determinization can see it).
   Gate each at N=400 vs Step-0 baseline (effects will be small; N=200 can't see +3%).

**Step 2 — Exact-generator determinization, retested at true config. [2–3 days, built already]**
belief0's null (52% @ 25 ms, 2 worlds) doesn't bind at 7 s × 8–14 worlds where world quality
compounds. Machinery exists (`randbats_pools`, conditional sampler). Gate N=400.

**Step 3 — Opponent-side human priors (the H1 reframe). [1–2 weeks, ~80% built]**
Feed `s2_priors` from the 142M policy evaluated from the *opponent's POV* (their side is fully
known in each determinized world; our side masked to what they've seen). This attacks the
mirror+full-info opponent model (gap 7) and non-locality (4c), with Athena as the format-
specific existence proof. Sweep the mixing weight (population-vs-search dial, AGENTS.md λ).
Gate N=400; this is the highest-theory-support original experiment remaining.

**Step 4 — Endgame exact re-solving where beliefs collapse. [2–4 weeks]**
Theory (ReBeL/SoG, arXiv:2007.13544 / 2112.03178) says sound belief-state solving is feasible
exactly when the belief support is small — which randbats' disambiguation guarantees by the
endgame, precisely where FP is weakest (maintainer-confirmed; our ladder data: 33–38% win in
25+ turn games). Detect small-support states → switch to exact expectiminimax /CFR re-solve.

**Step 5 — Only if Steps 1–4 stack to a ladder-verified edge: the measurement papers.**
(i) Fraction of turns with genuinely mixed equilibria (open question, C2.5); (ii) luck/skill
variance decomposition for randbats (open, C5.3); (iii) the PIMC-vs-sound-search EV gap on a
real ladder format (open, C6.4). These are publishable regardless of bot supremacy.

**Kill criteria:** each step gates independently at N=400 vs Step-0 baseline; CI including
50% = revert and proceed to next step. Ladder run (130+ games, registered account) only for
configurations that pass H2H. The luck ceiling (95.2 GXE by the #1 human) is the sanity bound
on all ambitions.

---

## Key sources
- Kovařík, Schmid, Burch, Bowling, Lisý — FOSGs — arXiv:1906.11110
- Long, Sturtevant, Buro, Furtak — when PIMC succeeds — AAAI 2010, DOI 10.1609/aaai.v24i1.7562
- Frank & Basin — strategy fusion / non-locality — AIJ 100:87–123, 1998
- Lisý, Kovařík, Lanctot, Bošanský — SM-MCTS convergence — arXiv:1310.8613 (+ 1804.09045)
- Brown, Bakhtin, Lerer, Gong — ReBeL — arXiv:2007.13544; Schmid et al. — Student of Games — arXiv:2112.03178
- McIlroy-Young et al. — Maia — KDD 2020, arXiv:2006.01855
- Sarantinos — Athena (gen7 randbats, rank #33, opponent modeling) — arXiv:2212.13338
- Grigsby et al. — Metamon — arXiv:2504.04395; Czarnecki et al. — Spinning Tops — arXiv:2004.09468
- Duersch, Lambrecht, Oechssler — skill vs chance — EER 2020; Getty et al. — SIAM Rev 2018
- pmariglia — foul-play/poke-engine source + Smogon thread 3767378 + blog (config, eval, gaps)
- Live ladder top-500 (fetched 2026-07-03): #1 = 95.2 GXE / 2586 Elo
