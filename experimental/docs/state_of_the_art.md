# State Of The Art

## Pokemon Showdown Bots

### Foul Play

Foul Play is the strongest practical public gen9 search baseline in this repository. It uses hidden-state inference plus `poke-engine` MCTS. It is simple but robust.

Unexplored opportunities relative to Foul Play:

- Learned policy/value prior that actually improves search.
- Public-belief state rather than simple determinization.
- Simultaneous-move no-regret root solving.
- Better compute scheduling and search reuse.

### Metamon / PA-Agent / Tauros / Kakuna

Metamon-style offline RL is the strongest learning direction. TaurosV0 empirically crushes Foul Play in Gen1OU H2H in this repo.

Unexplored opportunities:

- Large-scale teacher distillation with a model comparable enough to the teacher.
- Search verifier on top of a strong learned policy.
- Bucket-specific causal disagreement analysis.

### Oak / RBY Stockfish

Oak is conceptually closest to the original plan: fast early-gen engine, no-regret simultaneous-node search, and small learned CPU nets.

Unexplored opportunities:

- Port Oak-like expert iteration into this repo using `libpkmn`.
- Use Exp3/regret matching in a controlled small subgame before applying to full battles.

## General Game-AI Systems

### AlphaZero-Style Expert Iteration

Best for games where self-play search can generate strong targets. Requires large-scale data generation and stable engine/search.

Complexity: high.

Expected payoff: high if the engine/search is correct.

Risk: very high for Gen9; moderate for Gen1.

### Offline RL / Sequence Modeling

Best when large replay datasets exist. Metamon demonstrates feasibility.

Complexity: high.

Expected payoff: high.

Risk: high compute and data engineering requirements.

### Search With Learned Prior

Best immediate hybrid direction. A learned policy narrows or guides search; exact simulator verifies.

Complexity: medium to high.

Expected payoff: medium to high.

Risk: poor priors can hurt search if accepted too aggressively.

### Speculative Search

Best remaining novel systems idea. Use opponent think-time and cheap draft policies to precompute likely next-turn search results, then verify/reject when the actual state arrives.

Complexity: medium.

Expected payoff: unknown but plausible.

Risk: cache hit rate may be low; integration with Foul Play state transitions may be difficult.

## Techniques Not Yet Properly Explored

### Streaming Speculative Search

Description: after sending our move, start background workers that predict likely opponent actions, simulate likely successor states, and pre-run search. When the next `|request|` arrives, accept cached results only if the observed state matches.

Complexity: medium.

Expected payoff: medium.

Risk: state matching and transition generation are hard.

Why promising: it uses otherwise idle opponent think-time and changes compute allocation rather than final selection.

### Public-Belief Randbats Inference With Calibration

Description: formalize randbats hidden-state posterior using Showdown generator code and revealed observations; evaluate calibration with Brier/reliability before H2H.

Complexity: medium.

Expected payoff: medium in randbats.

Risk: previous uncalibrated pool/conditional attempts were neutral.

### Pivotal-Turn CFR Re-Solve

Description: identify high-variance or late-game states and run regret matching/CFR on a small public-belief subgame.

Complexity: high.

Expected payoff: unknown.

Risk: hard to integrate and benchmark.

### Tauros Search Verifier

Description: use TaurosV0 as strategic policy and search only to reject obvious tactical blunders.

Complexity: high due runtime integration.

Expected payoff: plausible in Gen1.

Risk: final agent depends on borrowed Tauros unless distilled later.

## Recommended State-Of-The-Art-Informed Direction

The best MVP is **Streaming Speculative Search for Gen9 Foul Play**, with an initial measurement gate. It is novel enough to be worth testing, uses the current strongest gen9 baseline, and avoids further failed Tauros distillation.
