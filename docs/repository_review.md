# Repository Review

## Scope And Method

This review covers the project-owned repository files, tracked experiment artifacts, model checkpoints, and the relevant local third-party checkouts under `external/` that are part of the current working system. Virtual environments and bulk generated data were not treated as source code, but their presence and effects were noted when they affect reproducibility.

Facts are marked when directly supported by repository files. Conclusions are based on recorded experiments. Engineering judgment is called out where the repository does not contain a definitive answer.

## Project Goal

Fact: `AGENTS.md` defines the mission as building the strongest Pokemon Showdown singles agent by ELO/GXE, with target formats in this order: `gen1ou`, `gen9randombattle`, `gen9ou`.

Fact: The original thesis is belief-conditioned expert iteration: fast engine, hidden-information MCTS, Exp3 at simultaneous nodes, learned value/policy, learned beliefs, and bounded ladder-population exploitation.

Fact: `TAUROS_KAKUNA_AUTOPSY_PLAN.md` documents a later pivot: use TaurosV0/Kakuna as teachers, analyze why Tauros beats Foul Play, then build a derived agent.

Conclusion: The repository is a research harness with many falsified branches, not a production bot package.

## Layout

Fact: Important first-party paths are:

- `AGENTS.md`: mission, phased protocol, evaluation rules.
- `SETUP.md`: local setup and pinned versions.
- `eval/run.py`: central H2H and ladder evaluation harness.
- `scripts/run_foul_play.py`: wrapper and monkey patches around external Foul Play.
- `scripts/ladder_runner.py`: live ladder runner for Foul Play variants.
- `scripts/generate_selfplay_data.sh`: local Foul Play self-play data generation.
- `scripts/run_tauros_foulplay_trace.py`: Tauros/Kakuna versus Foul Play trace collection.
- `scripts/run_tauros_distilled_policy.py`: standalone distilled Tauros policy runner.
- `analysis/tauros_autopsy.py`: trace summarization.
- `analysis/export_tauros_policy_dataset.py`: Tauros trajectory to supervised dataset export.
- `train/value_net.py`: replay value model training.
- `train/expert_iter.py`: Gen1 self-play policy/value training from Foul Play decision logs.
- `train/tauros_action_kind.py`: Tauros action-kind/exact-action classifier training.
- `engine/patches/poke-engine-0.0.47-learned-eval.patch`: patched Rust engine for learned eval and PUCT-style priors.
- `nets/checkpoints/`: learned value, policy/value, Tauros probe checkpoints.
- `experiments/`: JSON/CSV results and logs.
- `external/`: ignored local checkouts of Foul Play, Pokemon Showdown, Metamon, caches, and generated data.

Fact: `.gitignore` excludes virtualenvs, `external/`, generated randbats pools, traces, processed data, logs, pids, and caches. This means much critical state is local and not tracked.

## Current Architecture

Current runtime architecture is centered on Foul Play plus wrappers.

Data flow for stock Foul Play:

```text
Showdown websocket
-> external/foul-play battle parser
-> sampled hidden determinizations
-> poke_engine.State conversion
-> poke_engine.monte_carlo_tree_search
-> aggregate root visit policy
-> Showdown /choose command
```

Fact: `external/foul-play/fp/search/main.py` samples `prepare_random_battles` or `prepare_battles`, calls `monte_carlo_tree_search`, aggregates MCTS visits, keeps moves within 75 percent of the top policy share, and samples the final action.

Fact: `scripts/run_foul_play.py` applies runtime patches for local login, no-move fallback, Gen1 unknown move handling, websocket ping handling, rating-line parsing, Foul Play decision logging, randbats belief variants, Tauros gates, and value shields.

Conclusion: The practical architecture is a set of experiment-specific monkey patches around upstream Foul Play rather than a clean first-party agent abstraction.

## Evaluation Pipeline

Fact: `eval/run.py` supports `h2h` and `ladder` modes.

Fact: Agent names currently include `random`, `max_damage`, `foul_play`, `foul_play_learned`, `foul_play_randbats_pool`, `foul_play_randbats_conditional`, `foul_play_tauros_kind`, `foul_play_tauros_action`, and `foul_play_value_shield`.

Fact: H2H supports paired schedules, role splits, void tracking, Wilson confidence intervals, JSON output, and append-only CSV logging.

Fact: `scripts/ladder_runner.py` runs one live ladder game per subprocess, parses winner/ELO/GXE, checkpoints every 20 games, and has a watchdog timeout.

Conclusion: Local H2H infrastructure is strong enough for gates. Public ladder infrastructure exists but has account/IP/reliability constraints and should only be used after H2H gates pass.

## Training Pipeline

Training paths include several independent lines:

- `train/value_net.py`: public replay logistic value model for Gen9 randbats.
- `scripts/generate_selfplay_data.sh` plus `train/expert_iter.py`: Foul Play self-play decision logs and policy/value MLP.
- `analysis/export_tauros_policy_dataset.py` plus `train/tauros_action_kind.py`: Tauros trajectory behavior cloning probes.
- `engine/patches/poke-engine-0.0.47-learned-eval.patch`: Rust-side learned eval and PUCT hooks.

Conclusion: Training is exploratory and fragmented. There is no single promoted training loop.

## Inference Variants And Results

Conclusion supported by experiments: none of the first-party derived agents currently beats stock Foul Play with significance.

Important recorded results:

- Phase 0 gen9 randbats Foul Play self-play scorer gate: N=200, win rate 53.0%, CI [46.1%, 59.8%], gate passed.
- Gen1OU Foul Play scorer gate: N=200 requested, 183 decisive, 48.1%, CI [41.0%, 55.3%], 17 voids.
- Gen9 learned eval versus stock Foul Play: failed in multiple N=20 gates.
- Gen1 learned eval variants versus stock Foul Play: failed badly, including 31.5%, 25.8%, 21.0%, and 0.0% variants.
- Exp3 and early PUCT variants: severe regressions or noisy results.
- Randbats generator-pool belief: N=100 at 25ms, 52-48, CI [42.3%, 61.5%], no promotion.
- Conditional randbats sampler: N=20, 8-12, no promotion.
- TaurosV0 teacher versus Foul Play: N=100, 80-20, CI [71.1%, 86.7%], strong teacher edge.
- Tauros action-kind gate: N=20, 6-14, rollback.
- Tauros exact-action gate: N=20, 6-14, rollback.
- Standalone distilled Tauros linear policy: N=20, 5-15, rollback.
- Gen9 value-veto shield: N=100, 54-46, CI includes 50; shield fired 1/3480 decisions, killed.

## Datasets And Artifacts

Fact: Important local generated datasets include:

- `data/traces/tauros_vs_foulplay/`: raw Tauros/Foul Play trace shards, ignored by git.
- `data/processed/tauros_policy_examples_n100.jsonl`: 6,338 Tauros policy examples, ignored by git.
- `data/randbats_pools/`: static generated Showdown randbats team pools, ignored by git.
- `external/metamon_cache/`: cached teams and pretrained models.
- `experiments/tauros_autopsy/`: committed summaries and selected small smoke/gate artifacts.

Fact: Model checkpoints include Gen9 logistic value, Gen1 BC/selfplay value models, expert-iteration policy/value, Tauros action-kind/exact-action probes, and MLP variants.

## Logging

Fact: Logging is decentralized:

- `METAGROSS_DECISION_LOG` writes Foul Play JSONL rows with features, state string, MCTS visits, winner labels.
- Metamon writes `.json.lz4` trajectories and CSV result logs.
- `eval/run.py` writes JSON summaries and optional experiment CSV rows.
- `ladder_runner.py` writes human-readable and JSON game records into one log file.

Risk: Log formats are useful but not unified. Turn alignment between Tauros and Foul Play remains incomplete.

## Infrastructure

Fact: Setup relies on Python venvs, Rust/maturin for `poke-engine`, Node/npm for Pokemon Showdown, and local ignored external checkouts.

Fact: AWS was used for local H2H/self-play, but public Showdown locked cloud IPs as proxies.

Conclusion: AWS is appropriate for local data generation and H2H, not public laddering.

## Current Bottlenecks

- No promoted agent stronger than stock Foul Play in gen9.
- Strong teacher exists, but naive distillation failed.
- The Foul Play wrapper is heavily monkey-patched and fragile.
- Engine build variants can be accidentally clobbered between Gen1 and Gen9.
- Critical external state is not fully reproducible from tracked files alone.
- Belief experiments lack calibration metrics.
- No unified state/action trace schema across Metamon and Foul Play.

## Technical Debt

- Fragmented experiments without a single current trunk architecture.
- Multiple incompatible model feature schemas.
- No first-party test suite.
- External checkouts and site-packages patches are required for current functionality.
- Secrets/passwords can be passed as process arguments in some paths.
- Several invalid/noisy artifacts remain untracked or unclearly labeled.

## Supported Conclusion

The repository has excellent experimental breadth and useful infrastructure, but it does not yet contain a 2000+ original bot. The best-supported asset is the verified TaurosV0 teacher edge over Foul Play in Gen1OU. The best-supported gen9 baseline remains stock Foul Play.
