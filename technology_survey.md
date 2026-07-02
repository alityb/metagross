# Technology Survey

## Pokemon Showdown

Sources: https://github.com/smogon/pokemon-showdown, https://github.com/smogon/pokemon-showdown/blob/master/PROTOCOL.md, https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md

What it is: the canonical battle server, protocol, and simulator for competitive Pokemon battles.

Why it matters: all public ladder games are played through Showdown, and local H2H uses a local Showdown server.

How it works: websocket messages carry room events, `|request|` JSON asks the client for legal choices, and the bot replies with `/choose ...|rqid`. The simulator resolves battle mechanics and produces protocol messages.

Best practices:

- Use Showdown for live I/O and correctness checks.
- Pin the Showdown commit for reproducible local tests.
- Do not use Showdown JS as the inner-loop MCTS simulator.
- Treat Showdown protocol handling as an integration risk.

Limitations:

- Slow relative to Rust/Zig engines for search.
- Protocol evolves with upstream.
- Public server can lock cloud/proxy IPs.

## poke-env

Sources: https://github.com/hsahovic/poke-env, https://poke-env.readthedocs.io/en/stable/

What it is: a Python interface for creating Showdown-playing agents, challenge/ladder orchestration, and RL-style environment workflows.

Repo use: local smoke tests and simple baselines in `eval/run.py` and `scripts/smoke_poke_env.py`.

Strengths:

- Easy local challenges.
- Built-in baselines such as random and max-damage players.
- Async interface for multiple battles.

Limitations:

- Not a fast simulator.
- State abstractions can lag or differ from upstream mechanics.
- Not suitable for MCTS rollouts.

## Foul Play

Sources: https://github.com/pmariglia/foul-play, https://pmariglia.github.io/posts/foul-play/

What it is: a strong public Pokemon Showdown bot using determinized search over `poke-engine` states.

How it works in this repo:

- Samples possible hidden worlds.
- Converts to `poke-engine.State`.
- Runs MCTS.
- Aggregates root visit counts across determinizations.
- Samples from moves near the top visit share.

Why it matters: it is the strongest practical gen9 baseline in this repo and reportedly won the PokéAgent Gen9 OU track.

Limitations:

- Handcrafted eval.
- Determinization risks strategy fusion.
- No learned policy prior.
- DUCT/visit aggregation is not a proven low-exploitability simultaneous-move solver.
- Dynamax/Z-move support gaps are noted upstream.

## poke-engine

Sources: https://github.com/pmariglia/poke-engine, https://poke-engine.readthedocs.io/en/latest/

What it is: a Rust singles battle/search engine with Python bindings, used by Foul Play.

Repo use:

- Stock Gen9 Foul Play search.
- Patched Gen1 learned-eval/PUCT experiments.
- Foul Play decision logging via `compute_value_features` in patched builds.

Performance characteristics:

- Much faster than Showdown JS for search.
- Native Rust MCTS can run many iterations under 25-100ms budgets.

Limitations:

- Not a perfect simulator.
- Build features are generation-specific.
- Local patched wheels are easy to clobber.

## libpkmn / pkmn engine

Sources: https://github.com/pkmn/engine, https://github.com/pkmn/engine/blob/main/docs/TESTING.md, https://github.com/pkmn/engine/blob/main/docs/PROTOCOL.md

What it is: a high-performance low-level engine, especially mature for early generations.

Why it matters: for Gen1, it is the best long-term exact rollout engine candidate. Upstream documentation reports very large speedups over patched Showdown in supported formats.

Integration pattern:

- Use Showdown for I/O and validation.
- Use `libpkmn` for rollout/search once state conversion is implemented.

Limitations:

- Roadmap prioritizes early gens.
- Modern gen support is not the immediate path.
- Requires custom state translation and validation tests.

## Metamon / AMAGO

Sources: https://github.com/UT-Austin-RPL/metamon, https://metamon.tech, https://arxiv.org/abs/2504.04395, https://github.com/UT-Austin-RPL/amago, https://ut-austin-rpl.github.io/amago

What it is: an RL framework and Pokemon-specific environment/dataset/model stack for offline RL and pretrained Pokemon policies.

Repo use:

- TaurosV0 and Kakuna as teacher policies.
- Challenge mode and public-ladder patches.
- Trace collection through Metamon trajectories.

Strengths:

- Strong pretrained policies.
- Human replay and self-play datasets.
- Rich observation/action/reward abstractions.

Limitations:

- Large dependencies and models.
- CPU inference can be slow.
- Cloud public laddering was blocked by Showdown proxy lock.
- The teacher is borrowed unless distilled or used only for analysis.

## Behavior Cloning And Policy Distillation

Sources: DAgger https://proceedings.mlr.press/v15/ross11a.html, Policy Distillation https://arxiv.org/abs/1511.06295

What it is: supervised imitation of a policy or compression of a stronger teacher into a smaller student.

Repo use:

- Replay value models.
- Tauros action-kind and exact-action probes.
- Standalone distilled Tauros policy.

Lessons from this repo:

- Offline accuracy can improve over majority baselines.
- Naive small students still failed H2H badly.
- Teacher strength does not automatically transfer with small data or weak architecture.

## Offline RL

Sources: Levine et al. offline RL tutorial https://arxiv.org/abs/2005.01643, CQL https://arxiv.org/abs/2006.04779, IQL https://arxiv.org/abs/2110.06169, Decision Transformer https://arxiv.org/abs/2106.01345

What it is: learning policies from static datasets without online exploration.

Why it matters: Metamon/Tauros strength likely comes from large-scale offline RL and human/self-play data.

Limitations:

- Distribution shift.
- Value overestimation on unseen actions.
- Off-policy evaluation is unreliable in high-variance games.

Repo implication: serious distillation likely needs much more data and better architecture than the current small probes.

## MCTS And ISMCTS

Sources: UCT https://doi.org/10.1007/11871842_29, Browne survey https://doi.org/10.1109/TCIAIG.2012.2186810, ISMCTS https://doi.org/10.1109/TCIAIG.2012.2200894

What it is: simulation-based tree search; ISMCTS adapts search to imperfect-information games.

Repo use: Foul Play uses MCTS over sampled determinizations.

Key limitation: naive determinization can leak hidden information and cause strategy fusion.

## Exp3, CFR, ReBeL

Sources: simultaneous MCTS https://arxiv.org/abs/1310.8613, Exp3 https://doi.org/10.1137/S0097539701398375, CFR https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html, ReBeL https://arxiv.org/abs/2007.13544

What these provide:

- Exp3/regret matching: no-regret selection for simultaneous-move nodes.
- CFR: equilibrium solving in imperfect-information games.
- ReBeL: public-belief-state search and learning.

Repo status: original `AGENTS.md` calls for Exp3/ReBeL-style beliefs, but implemented code mostly explored PUCT, value nets, randbats generator sampling, and teacher distillation.

## Speculative Decoding And Speculative Search

Sources: Leviathan et al. https://arxiv.org/abs/2211.17192, Chen et al. https://arxiv.org/abs/2302.01318

What it is: a cheap draft model proposes future work; an expensive verifier accepts or rejects it.

Pokemon analogy:

- Draft policy proposes likely moves, opponent responses, or future states.
- Exact engine/search verifies.
- Cached speculative work is accepted only if the next observed state matches.

Why it matters: this is the most promising novel systems idea remaining because it changes compute allocation, not just final move selection.

## LZ4 And Trace Formats

Sources: LZ4 frame spec https://github.com/lz4/lz4/blob/dev/doc/lz4_Frame_format.md, python-lz4 docs https://python-lz4.readthedocs.io/en/stable/lz4.frame.html

Repo use: Metamon trajectories are `.json.lz4`, read by `analysis/tauros_autopsy.py` and `analysis/export_tauros_policy_dataset.py`.

Limitations: not random-access; schema depends on Metamon version/action space.
