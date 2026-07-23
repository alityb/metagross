# Literature Review

## Pokemon Battle Agents

### Foul Play

Sources: https://github.com/pmariglia/foul-play, https://pmariglia.github.io/posts/foul-play/

Foul Play is a public search bot using Foul Play Python orchestration plus `poke-engine` Rust MCTS. It is a strong baseline because it combines hidden-set inference, fast simulation, and tactical search. It does not use a learned neural policy/value in stock form.

Relevance: this repo repeatedly failed to beat stock Foul Play in gen9 with small modifications. Foul Play remains the main gen9 baseline.

### Metamon / PA-Agent

Sources: https://github.com/UT-Austin-RPL/metamon, https://metamon.tech, https://arxiv.org/abs/2504.04395

Metamon is an offline RL and imitation learning system using reconstructed human replays and self-play datasets. It includes pretrained policies such as TaurosV0 and Kakuna.

Evidence in this repo: TaurosV0 beat Foul Play 80-20 in Gen1OU at N=100. However, naive distillation into small policies failed.

Relevance: Metamon is the best teacher, not yet a successful final original bot.

### PokéAgent Challenge

Sources: https://pokeagent.github.io/, https://pokeagent.github.io/leaderboard.html, https://arxiv.org/abs/2603.15563

The Challenge provides a useful benchmark landscape: specialist search and RL agents outperform generic LLM agents. Foul Play-style search won Gen9 OU, while PA-Agent/Metamon-style offline RL won Gen1OU.

Relevance: the split supports a hybrid thesis: search is strong tactically, offline RL is strong strategically.

### PokéChamp

Sources: https://github.com/sethkarten/pokechamp, https://arxiv.org/abs/2503.04094

PokéChamp combines LLM reasoning with minimax and opponent modeling. It is relevant as an example of high-level reasoning and opponent modeling, but not the strongest known approach.

Relevance: LLMs are not the core runtime path for this repository.

### Oak / Stockfish For RBY

Sources: https://www.smogon.com/forums/threads/stockfish-for-rby.3770936/, https://github.com/pokemon-labs/oak

Oak combines fast early-gen simulation, ISMCTS, Exp3 at simultaneous nodes, and small learned CPU networks. It is the closest conceptual match to the original thesis in `AGENTS.md`.

Relevance: supports using fast early-gen engines and learned value/policy networks, but the repo’s implemented attempts have not matched Oak-level integration.

## Search And Imperfect Information

### UCT / MCTS

Sources: Kocsis and Szepesvari UCT https://doi.org/10.1007/11871842_29, Browne et al. survey https://doi.org/10.1109/TCIAIG.2012.2186810

MCTS estimates action values via simulations. UCT provides a bandit-inspired exploration/exploitation rule. It works well when simulation is fast and rewards are sparse.

Limitations: vanilla MCTS assumes perfect or fully modeled state and does not solve hidden simultaneous-move games directly.

### Information Set MCTS

Sources: Cowling, Powley, Whitehouse https://doi.org/10.1109/TCIAIG.2012.2200894

ISMCTS searches over information sets instead of hidden concrete states. This directly addresses hidden information such as unrevealed Pokemon sets.

Limitations: belief quality is crucial. Determinization can cause strategy fusion if final actions depend on sampled hidden state.

### Simultaneous-Move MCTS And Exp3

Sources: Lisý et al. https://arxiv.org/abs/1310.8613, Exp3 nonstochastic bandits https://doi.org/10.1137/S0097539701398375

Pokemon moves are selected simultaneously. Hannan-consistent algorithms such as Exp3/regret matching are better theoretical choices for simultaneous nodes than standard UCT.

Repo status: Exp3 was attempted and killed empirically in one setup, but the repository does not contain a mature no-regret simultaneous-root solver.

### CFR And ReBeL

Sources: CFR https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html, CFR+ https://arxiv.org/abs/1407.5042, ReBeL https://arxiv.org/abs/2007.13544

CFR is the standard equilibrium method for imperfect-information games. ReBeL combines RL and search in public-belief states.

Relevance: they provide the correct theoretical frame for belief-conditioned Pokemon search. They are too heavy for an MVP unless restricted to pivotal subgames.

## Learning From Teachers

### Behavior Cloning And DAgger

Sources: DAgger https://proceedings.mlr.press/v15/ross11a.html

Behavior cloning learns from observed teacher actions. DAgger demonstrates covariate shift: a cloned policy visits states not present in the dataset and degrades.

Repo result: BC-style Tauros students failed in H2H despite meaningful offline accuracy. This is consistent with known BC limitations.

### Policy Distillation

Sources: Policy Distillation https://arxiv.org/abs/1511.06295

Policy distillation compresses a teacher policy into a smaller student. It is not novel by itself, but can be powerful with enough data and architecture.

Repo result: small linear/MLP probes from 100 games are insufficient.

### Expert Iteration / AlphaZero / MuZero

Sources: Expert Iteration https://arxiv.org/abs/1705.08439, AlphaZero https://arxiv.org/abs/1712.01815, MuZero https://arxiv.org/abs/1911.08265

Search generates improved policy/value targets; neural networks amortize search. This is the canonical search-plus-learning recipe.

Relevance: a serious future implementation should collect root policies/values at scale and train a model used by search.

## Offline RL

Sources: Offline RL tutorial https://arxiv.org/abs/2005.01643, CQL https://arxiv.org/abs/2006.04779, IQL https://arxiv.org/abs/2110.06169, Decision Transformer https://arxiv.org/abs/2106.01345

Offline RL learns from static datasets but must control distribution shift and overestimation. Metamon is the relevant Pokemon example.

Relevance: if this repo wants a true RL contribution, it likely needs a much larger offline dataset and a stronger policy architecture, not another small final-selector patch.

## Speculative Decoding Applied To Search

Sources: Leviathan et al. https://arxiv.org/abs/2211.17192, Chen et al. https://arxiv.org/abs/2302.01318

Speculative decoding uses a cheap draft model to propose work and an expensive model to verify it. The game-search analogue is a cheap policy proposing future states/actions, with exact search verifying and accepting/rejecting cached work.

Relevance: this is the best remaining algorithmic idea because it changes search allocation and can use opponent think-time.

## Supported Conclusions

- Search alone is strong but plateaus around tuned baselines.
- Offline RL can beat search in some formats, as TaurosV0 demonstrates.
- Naive small distillation is insufficient.
- The most defensible new technique is speculative search scheduling, not another final-move veto.
