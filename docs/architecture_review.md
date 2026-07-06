# Architecture Review

## Strengths

- The repository has a real local H2H harness with paired games, role splits, CIs, void tracking, and JSON outputs.
- Foul Play integration is flexible enough to test many variants quickly.
- Local Showdown and poke-env smoke tests are established.
- Metamon/Tauros trace collection works and produced a strong N=100 teacher result.
- Experiments are often logged with enough metadata to interpret outcomes.

## Weaknesses

### Monkey-Patching Is The Main Architecture

Evidence: `scripts/run_foul_play.py` patches Foul Play protocol handling, websocket behavior, search selection, decision logging, randbats determinization, Tauros gates, and value shields.

Risk: patch ordering is fragile. Combining `METAGROSS_DECISION_LOG` with selector-altering gates can overwrite previous selector patches.

Recommendation: create explicit first-party agent classes or strategy modules instead of stacking runtime monkey patches.

### Engine Builds Are Easy To Clobber

Evidence: Gen1 and Gen9 require different `poke-engine` build features. The repo repeatedly used `.venv-foul-play`, `.venv-exp3`, and rebuilt wheels.

Risk: experiments can silently run the wrong engine build.

Recommendation: maintain separate named venvs or wheel paths per generation, and log engine feature/build hash in every experiment.

### Model Schemas Are Incompatible

Evidence: old Gen1 checkpoints use 12 features, current patched engine uses 14, Gen9 logistic model uses 16, policy-value uses different dimensions.

Risk: runtime crashes or invalid comparisons.

Recommendation: add explicit model schema validation and model cards with feature version.

### External State Is Not Reproducible From Git

Evidence: `external/` is ignored, but patched Metamon/Showdown/Foul Play state is required.

Risk: another engineer cannot reproduce current behavior from the repo alone.

Recommendation: script or patch every external modification and document exact commands.

### Evaluation Artifacts Include Invalid Runs

Evidence: several experiment JSONs have zero decisive games or known scoring failures; old ladder logs contain timeouts and restarts.

Risk: result cherry-picking or misinterpretation.

Recommendation: add an experiment index with status labels: valid, invalid, smoke, killed, promoted.

### No Unified Trace Schema

Evidence: Metamon trajectories are dict/LZ4; Foul Play rows store poke-engine state strings and MCTS visits. Current autopsy explicitly does not align decisions in the same state.

Risk: disagreement analysis remains approximate.

Recommendation: build a common public-observation trace schema before further distillation.

### Insufficient Test Coverage

Evidence: validation is smoke scripts and experiment runs; no first-party test suite was found.

Risk: regressions in wrappers and experiment harnesses.

Recommendation: add unit tests for selector logic, model schema parsing, trace export, and local Showdown H2H smoke.

## Architectural Judgment

The repository is strong as a research notebook in code form. It is weak as a reproducible platform. Before implementing another agent, the team should isolate one MVP path and stop adding more broad experimental hooks to `run_foul_play.py`.
