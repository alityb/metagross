# Belief, Search, and the 92nd Percentile

## How we built a Pokémon Showdown bot that ranks ~18th in the world

*July 2026*

---

Pokémon Showdown is the largest competitive Pokémon battle platform in the world. Players queue up for random battles — 6v6 singles, teams assigned by a public generator, no team preview. You see your team, your opponent sees theirs, and neither of you knows what the other has until Pokémon are sent onto the field.

It is, in the language of game theory, a **game of imperfect information**. And that single fact determined almost everything about what worked, what failed, and why.

This is the story of how we went from a stock search bot at 75 GXE (roughly 1100 ELO) to 92.4 GXE (ranked ~18th globally), what we tried that didn't work, and the game-theoretic reasons why.

---

## The landscape

Two paradigms dominate competitive Pokémon AI:

**Foul Play** is a search-based agent. It runs determinized MCTS — it samples possible opponent teams from a belief distribution, runs Monte Carlo tree search over each "determined" world where the opponent's team is assumed known, and aggregates the results. It won the Gen 9 OU division at the NeurIPS 2025 PokéAgent Challenge. Its leaf evaluation is a hand-crafted heuristic: HP ratios, stat boosts, item values. No learning.

**Metamon** is an RL-based agent. It trains a 142M-parameter transformer policy via offline reinforcement learning on ~1M human replays + ~4M self-play battles. It won Gen 1 OU. Its policy directly outputs action distributions given battle state. No search.

The interesting fact: **neither paradigm alone closes the gap to elite humans.** The top human players sit at 93-95 GXE. The best bots, before our work, hovered around 88.

The gap is real. It's not a compute issue. It's a **knowledge representation** issue.

---

## Why imperfect information breaks everything

Here's the thing about Pokémon that makes it fundamentally different from chess or Go: you don't know the full state. Your opponent has five Pokémon you haven't seen yet. Each one could have one of several abilities, items, and movesets. The state space of "what could the opponent have" is enormous.

The standard approach to imperfect-information games in the AI literature is **Perfect Information Monte Carlo (PIMC)**. It works like this:

1. Sample a possible "true" state from your belief distribution (e.g., "the opponent probably has a Choice Scarf on their Garchomp")
2. Run a perfect-information search (MCTS, minimax) on that determined world
3. Repeat for many samples, aggregate the results

PIMC is attractive because it's simple and it reuses all your perfect-information search infrastructure. It's what Foul Play does. And it works — surprisingly well, in fact. Long, Sturtevant, Buro, and Furtak proved in 2010 that PIMC is **near-optimal** in a precise sense: it converges to the optimal strategy as the number of samples grows, under certain conditions.

But there's a catch. Actually, there are two catches.

### Strategy fusion

The first catch is called **strategy fusion**. Here's the problem: in a determined world (where you've guessed the opponent's team), you can compute the optimal move. But you can't actually *commit* to that move, because you don't know which world you're in. If you play move A in world 1 and move B in world 2, the opponent can observe your move and infer information about your belief — which they can then exploit.

In poker terms: if you bluff only when you have a weak hand, the opponent figures it out. You need to bluff with some strong hands too. The determined-world search doesn't know this — it computes the optimal move *assuming you can see the opponent's cards*.

This means PIMC has a systematic bias: it plays too aggressively in positions where the hidden information matters, because it's optimizing as if it knew the truth.

### Non-locality

The second catch is **non-locality**. The expected value of a move depends on the *distribution* of possible worlds, not just on the average outcome across worlds. A move that is good in 90% of worlds but catastrophic in 10% might be worse than a move that is mediocre in all worlds — but PIMC averages them the same way.

These two problems are well-known in the game theory literature (Frank & Basin, 1998; Long et al., 2010). They're the reason why PIMC, despite being "near-optimal," can be exploited by a sound opponent.

For our purposes, they set up the central question: **given that PIMC has these structural flaws, where should we invest our effort to improve it?**

---

## What we tried that failed

### Attempt 1: Replace the leaf evaluation with a learned value network

This is the obvious one. Every survey paper on game AI says the same thing: "replace the hand-crafted eval with a learned value net." AlphaGo did it. AlphaZero did it. Leela Chess Zero did it. It's the consensus recommendation.

We trained a value network to predict win probability from battle state features. We used value-as-classification (win/loss buckets) rather than regression, following the "Stop Regressing" paper (ICML 2024), because classification is more stable and scales better with model size.

It didn't work. Not "it didn't help" — it actively made things worse.

**Why?** The fundamental problem is **distribution mismatch between training and inference**. During training, the value net sees the *true* game state — both teams fully revealed. During search, the leaf evaluation runs on *determined* (guessed) states. The value net was never trained on guessed states; it was trained on truth. So when you give it a leaf in a world where you've assumed the opponent has a Choice Scarf, it evaluates that position as if the Choice Scarf were certain — overconfident, systematically biased.

We call this **C-DEAD**: the learned value is falsified at inference time because the information conditions differ between training and deployment. We verified this five different ways — different architectures, different training targets, different state representations. All of them hit the same ~68% oracle ceiling: the value net could predict outcomes well on the training distribution but was systematically wrong when plugged into search.

This is a direct consequence of strategy fusion. The value net learns `V(true_state)`, but the search needs `E[V(sampled_state)]` weighted by belief probabilities — and these are not the same thing. No amount of architecture improvement fixes this; it's a representation-level problem.

### Attempt 2: Threat matrices and scout values

We tried adding explicit "threat" terms to the evaluation — for each possible opponent Pokémon, how threatening is it to our active Pokémon? We also added "scout" values — how much information do we gain by making a certain move (e.g., using a move that reveals our opponent's ability)?

The threat matrix gave 47.5% in head-to-head testing. The scout values were mathematically inert — they canceled out in the leaf-minus-root computation that MCTS uses to propagate values. We proved this analytically: the threat and scout terms appear in both the root and leaf evaluations with the same sign, so their difference is zero. They contribute nothing to the search's decision.

This was a humbling result. We spent weeks engineering these features, and they were provably useless. The lesson: **understand the math of your search algorithm before adding features to its evaluation.** MCTS doesn't use absolute leaf values; it uses *differences* between leaf and root. Any feature that appears identically in both is invisible to the search.

### Attempt 3: Win-condition matrices

We built a 6×6 matrix: for each of our Pokémon vs each of their Pokémon, who wins the damage race? This is essentially a compressed game-theoretic solve of the endgame. We injected it into the poke-engine's evaluation function.

Result: 34% win rate in H2H testing. Worse than baseline.

**Why?** The win-condition matrix was **redundant** with the existing damage-race computation that Foul Play's heuristic already does implicitly. Adding it double-counted the same information, biasing the evaluation. The heuristic wasn't missing this signal — it was already there, just in a different form.

### Attempt 4: PP-stall arithmetic

Pokémon moves have limited PP (power points). In long games, running out of PP matters. We added arithmetic to track remaining PP and predict which side would run out first.

Result: 50.4% — indistinguishable from noise.

**Why?** In random battles, games rarely go long enough for PP to be the binding constraint. The average game is ~29 turns. PP exhaustion typically matters at turn 50+. The feature was correct but irrelevant to the actual game distribution.

### Attempt 5: Endgame solver

We tried switching from MCTS to exact expectiminimax when the game was nearly over (≤3 Pokémon per side, ≤50 joint actions). The idea: in the endgame, hidden information has mostly been revealed, so the game is closer to perfect information, and exact solving should beat sampling.

Result: depth 3-4 search can't reach terminal states in random battles. The branching factor is too high. We'd need depth 8-10, which is computationally infeasible at 500ms per move.

**Why?** The intuition was right — endgames *are* closer to perfect information. But "closer" isn't "close enough." With 3 Pokémon per side, each with 4 moves + 2 switches, the branching factor is still ~6^4 ≈ 1296 per ply. At depth 4, that's 2.8 billion nodes. Not happening in 500ms.

### Attempt 6: Self-play-only expert iteration

We generated self-play data with our search agent (250ms, 4 worlds — weak), trained a new policy checkpoint on it, and tested on the ladder.

Result: the checkpoint was worse than the base policy it was fine-tuned from.

**Why?** Two reasons. First, the expert was too weak — 250ms with 2-4 worlds is barely better than the raw policy prior, so the "expert" demonstrations were noisy. Second, and more fundamentally, we trained on **self-play only** with no human data. The policy drifted toward the self-play distribution, which is not the same as the human ladder distribution. This is the **population overfitting** problem documented in the AlphaStar paper: a policy that only plays against itself can develop brittle strategies that exploit its own weaknesses but fail against humans.

The Metamon team encountered this exact trap with their TaurosV0 agent: 80-20 head-to-head vs Foul Play, but only 1420 ELO on the human ladder. The policy had learned to exploit Foul Play's specific weaknesses, not to play good Pokémon.

---

## What worked

### The key insight: priors, not leaf values

Here's the thing that took us months to realize: **the binding constraint wasn't leaf evaluation. It was tree exploration.**

MCTS with PUCT (the AlphaZero variant) works like this. For each child node `a` of the current state, the selection score is:

```
PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
```

Where:
- `Q(s, a)` is the average value of taking action `a` (exploitation)
- `P(s, a)` is the **prior probability** of action `a` (exploration guidance)
- `N(s)` is the parent visit count
- `N(s, a)` is the child visit count
- `c_puct` controls the exploration-exploitation balance

The prior `P(s, a)` determines **where the search looks first.** With a uniform prior, MCTS explores all actions equally, wasting time on obviously bad moves. With a good prior, MCTS focuses its budget on the moves that matter — the ones a strong player would actually consider.

This is the insight: **you don't need better leaf values if you can guide the tree to better branches.** The leaf values are wrong anyway (C-DEAD), but if the search visits the right parts of the tree, the *relative* values between good moves are still informative. MCTS is robust to systematic bias in leaf values as long as the bias is consistent across the actions being compared.

### C1: Our-side priors

We run a 142M-parameter Metamon policy (the Kakuna checkpoint) as an HTTP prior server. For every turn, we feed the current battle state into the policy and get back an action probability distribution. This distribution becomes the `P(s, a)` in the PUCT formula.

The policy was trained on ~1M human replays + ~4M self-play battles. It knows what good Pokémon play looks like. It doesn't know the exact optimal move — it's a policy, not a value function — but it knows the *distribution* of reasonable moves given a board state.

Result: **82.8 → 88 GXE.** The stock Foul Play agent with no priors sits at ~75 GXE. Adding our-side priors alone jumped it to 88. That's a 13-point GXE gain from changing **one thing**: where the search looks.

### C2: Opponent-side priors

Here's the game-theoretically interesting part. In PIMC, you sample possible opponent teams and search over them. But the *opponent* is also a rational agent with a policy. If you can model what the opponent is likely to do, you can search more efficiently — you don't need to consider moves the opponent would never make.

We run the same 142M policy from the **opponent's perspective** — we flip the game state (swap sides) and ask: "what would a strong player do if they were in the opponent's position?" This gives us a prior over the opponent's actions, which we inject into the search's model of the opponent.

This is a form of **opponent modeling**, which is well-studied in the game theory literature. The key insight from the theory: in imperfect-information games, modeling your opponent's strategy is not optional — it's the difference between playing the Nash equilibrium and playing a best response to a fictional rational opponent. Real opponents are not perfectly rational; they have tendencies, biases, and habits. A policy trained on human data captures those tendencies.

Result: **88 → 91.4 GXE.** Adding opponent priors on top of our-side priors gave another 3.4 points. We hit #16 on the global leaderboard.

### Expert iteration: the compounding loop

Expert iteration (Anthony et al., 2017) is the algorithm that powers AlphaZero. The idea is simple:

1. Use your current best agent (search + policy) to play games against itself
2. Record the search's move distributions (not just the final moves — the full visit counts)
3. Train the policy to reproduce those distributions
4. The improved policy generates better search, which generates better data, which improves the policy...

Each round, the "expert" (search) is stronger than the "apprentice" (policy). By distilling the expert's knowledge back into the policy, the apprentice gets stronger. Then the apprentice becomes a better prior for the next round's search, making the expert even stronger.

We ran one round of ExIt:
- Generated 12,960 self-play games at 250ms/parallelism-1 (weak search — this was a mistake we'd later correct)
- Parsed into 23,870 POV trajectories
- Fine-tuned Kakuna with 6 epochs, KL-anchored to the frozen base policy (coefficient 0.02)
- Rating-conditioned the training (self-play data labeled as top rating band)
- Mixed 90% self-play / 10% human data (the human anchor is critical — see below)

Result: **91.4 → 92.4 GXE.** One ExIt round, even with a weak expert, improved the policy. The settled result at RD 25 was 92.4-92.7 GXE, peaking at 93.6.

### Why the human anchor matters

This is the part that's easy to get wrong. If you're doing expert iteration, why do you need human data? The search is the expert; the policy should just learn from the expert, right?

No. And the reason is **distribution shift**.

The metric is GXE — expected win rate against the human ladder population. But self-play generates games between two copies of your own agent. The self-play distribution drifts away from the human distribution over time: your agent develops strategies that exploit its own weaknesses, which may not be the same strategies that work against humans.

This is the **TaurosV0 trap**: a policy that wins 80% of games against Foul Play but only achieves 1420 ELO on the human ladder. It learned to exploit Foul Play, not to play good Pokémon.

The fix, validated by the Metamon team and supported by the piKL/Diplodocus literature (NeurIPS 2022), is to **retain human data in every training round**. The human data acts as an anchor — it keeps the policy on the "human strategy manifold," preventing drift. The KL anchor to the frozen base policy serves the same purpose from the regularization side.

We use a 90/10 self-play/human mix. The human data is never removed. This is not a temporary scaffold — it's a permanent component of the training pipeline.

---

## The game theory, stated plainly

Let me put the whole picture together in game-theoretic terms.

Pokémon Showdown random battles are an **imperfect-information sequential game**. The information structure is: both players know the team generator's distribution, but each player's specific team is private information. As the game progresses, information is revealed (Pokémon switch in, items are triggered, moves are used).

The **belief state** at any point is the posterior distribution over the opponent's team given the revealed information. In random battles, this belief can be computed near-exactly because the team generator is public code — you can enumerate all possible teams consistent with the reveals and weight them by generator probability.

**PIMC** approximates the solution to this game by:
1. Sampling teams from the belief distribution
2. Solving each determined (perfect-information) game
3. Aggregating

This is provably near-optimal (Long et al., 2010) but suffers from strategy fusion and non-locality. The key question is: **where does the remaining error come from?**

Our empirical answer: **not from the leaf evaluation, but from the tree exploration.** The hand-crafted heuristic is biased, but its bias is *consistent* — it overvalues the same things in all branches, so the *relative* comparison between moves is still informative. The problem is that MCTS with a uniform prior wastes most of its search budget on moves that are obviously bad, leaving insufficient budget to distinguish between the 2-3 moves that actually matter.

**Policy priors fix this.** A good prior `P(s, a)` concentrates the search budget on the moves that a strong player would consider. The PUCT formula's exploration term `c_puct * P(s, a) * sqrt(N) / (1 + N(s,a))` ensures that high-prior moves are visited first, and the exploitation term `Q(s, a)` ensures that the search eventually corrects the prior if it's wrong.

This is why our approach worked: we didn't try to fix the leaf evaluation (which is fundamentally broken by C-DEAD), we fixed the **exploration policy** (which is fixable because it's a local, per-node decision that doesn't require perfect information).

**Opponent priors** go one step further. They model the opponent's policy, which effectively reduces the branching factor of the search — instead of considering all possible opponent moves, you focus on the ones the opponent is likely to choose. This is a form of **behavioral strategy** modeling: you're not assuming the opponent plays optimally (which would be the Nash equilibrium approach), you're assuming they play *like a human* (which is the population-best-response approach, and is what GXE actually measures).

**Expert iteration** closes the loop. The search agent, armed with policy priors, generates better self-play data. The policy is fine-tuned on this data, becoming a better prior. The next round's search is stronger, generating even better data. This is the AlphaZero recipe, adapted to imperfect-information games by retaining the human anchor and using PIMC instead of perfect-information MCTS.

---

## What we learned

1. **Understand the math before adding features.** Our threat/scout terms were provably inert because they canceled in the leaf-root difference. Weeks of engineering, zero effect.

2. **The binding constraint may not be where the literature says.** Every survey says "replace the leaf eval with a learned value net." In imperfect-information games, this is wrong — the leaf eval is fundamentally broken by C-DEAD, and no architecture fixes that. The real lever is exploration guidance.

3. **Self-play without a human anchor is dangerous.** The metric is against humans. Training only against yourself optimizes the wrong objective. The Metamon team's TaurosV0 is the cautionary tale.

4. **PIMC is more robust than its theoretical flaws suggest.** Strategy fusion and non-locality are real, but in practice, the consistent bias of a hand-crafted eval is less harmful than the systematic mismatch of a learned eval. Relative comparisons survive absolute bias.

5. **Expert iteration compounds, but slowly.** One round gave +1 GXE. The gains come from multiple rounds, each with stronger search and more data. Patience is required.

6. **The expert must be strong.** Our first ExIt round used 250ms/2-4 worlds — barely better than the raw policy. It still helped, but the signal was weak. Stronger search (500ms/16-32 worlds) should produce better training data.

---

## Where we are now

- **Current best:** 92.4-92.7 GXE, peaked at 93.6, ranked ~18th globally
- **Stock Foul Play:** ~75 GXE
- **Improvement:** +17-18 GXE from priors + ExIt
- **Gap to elite humans:** ~2-3 GXE

We're now running a proper evidence-gated protocol for the next ExIt round: strict data provenance, PFSP-lite opponent pools, paired H2H promotion gates, and a 45/45/10 legacy-selfplay/strict/human training mix. The infrastructure is built, smoke-tested, and deployed. The data is accumulating.

The ceiling — whether it's 93, 94, or 95 — depends on whether the ExIt loop compounds beyond one round, and whether we can eventually fix the leaf evaluation problem (perhaps through belief-conditioned value nets that are trained on *determined* states, not true states).

But that's a story for the next blog post.

---

## References

- Anthony, Tian, Barber, *Thinking Fast and Slow with Deep Learning and Tree Search* (NeurIPS 2017)
- Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (2017)
- Vinyals et al., *Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning* (Nature, 2019)
- Long, Sturtevant, Buro, Furtak, *On the Strategy Fusion in Perfect-Information Monte-Carlo Sampling* (AAAI 2010)
- Frank & Basin, *Search in Games with Incomplete Information: A Case Study Using Bridge Card Play* (AIJ, 1998)
- Metamon: *Human-Level Competitive Pokémon via Scalable Offline RL and Transformers* (RLC 2025)
- PokéAgent Challenge: https://pokeagent.github.io/
- piKL / Diplodocus: *Human Population Training for Game Agents* (NeurIPS 2022)
- Foul Play: https://github.com/pmariglia/foul-play
- Oak: *Stockfish for RBY* — https://www.smogon.com/forums/threads/stockfish-for-rby.3770936/

---

*This project uses Foul Play's poke-engine as its rules/search infrastructure. The policy prior system, opponent modeling, expert iteration loop, evaluation harness, and all experimental infrastructure are original work. The 92.4 GXE result was obtained on a fresh account with no prior game history, at rating deviation 25.*
