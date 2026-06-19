# Phase 0 Setup

Format: `gen9randombattle`

This file records the official docs read and the exact commands used to stand up
the Phase 0 infrastructure. Third-party source checkouts and package artifacts
live under `external/` and are intentionally not committed.

## Local Pokemon Showdown Server

Official docs read before install:

- Pokemon Showdown server README: `https://raw.githubusercontent.com/smogon/pokemon-showdown/master/server/README.md`
- poke-env README local server section: `https://raw.githubusercontent.com/hsahovic/poke-env/master/README.md`

Pinned checkout used for verification:

- `smogon/pokemon-showdown` commit `4880d3693580bd33652797cf31179c6fcdf87e50`

Exact commands run:

```bash
mkdir -p external
git clone https://github.com/smogon/pokemon-showdown.git external/pokemon-showdown
npm install --prefix external/pokemon-showdown
cp external/pokemon-showdown/config/config-example.js external/pokemon-showdown/config/config.js
node external/pokemon-showdown/pokemon-showdown start --no-security 8000
```

Reproducible commands from this repo:

```bash
scripts/install_showdown.sh
scripts/start_showdown.sh
node scripts/check_showdown_ws.mjs ws://localhost:8000/showdown/websocket
```

Verification performed:

- The server started on port `8000`.
- A websocket connection to `ws://localhost:8000/showdown/websocket` opened successfully.

## poke-env Match Runner

Official docs read before install:

- poke-env README: `https://raw.githubusercontent.com/hsahovic/poke-env/master/README.md`
- The Read the Docs getting-started page was attempted first, but returned HTTP 429 during setup, so the current GitHub README was used as the official install source.

Pinned package used for verification:

- `poke-env==0.15.0`

Exact commands run:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install poke-env
.venv/bin/python scripts/smoke_poke_env.py
```

Reproducible commands from this repo:

```bash
scripts/install_python_env.sh
.venv/bin/python scripts/smoke_poke_env.py
```

Verification performed:

- One full `gen9randombattle` completed on the local server between poke-env's built-in `RandomPlayer` and `MaxBasePowerPlayer`.
- Smoke-test result: `finished=1`, `random_wins=0`, `max_power_wins=1`.

## Foul Play Baseline

Official docs read before install:

- Foul Play README: `https://raw.githubusercontent.com/pmariglia/foul-play/master/README.md`
- poke-engine README: `https://raw.githubusercontent.com/pmariglia/poke-engine/main/README.md`

Pinned checkouts/packages used for verification:

- `pmariglia/foul-play` commit `e1e2ca650598621e85c3b6ab751c66e625489934`
- `poke-engine==0.0.47`, built from source with Foul Play's pinned requirement: `--config-settings="build-args=--features poke-engine/terastallization --no-default-features"`

Exact commands run:

```bash
git clone https://github.com/pmariglia/foul-play.git external/foul-play
python3.11 -m venv .venv-foul-play
.venv-foul-play/bin/python -m pip install --upgrade pip
.venv-foul-play/bin/python -m pip install -r external/foul-play/requirements.txt
.venv-foul-play/bin/python -m pip download --no-binary :all: --no-deps poke-engine==0.0.47 -d external/source-dist
tar -xzf external/source-dist/poke_engine-0.0.47.tar.gz -C external/source-dist/poke_engine-0.0.47 --strip-components=1
```

Reproducible commands from this repo:

```bash
scripts/install_foul_play.sh
.venv-foul-play/bin/python scripts/run_foul_play.py --help
```

Local verification command shape:

```bash
.venv-foul-play/bin/python scripts/run_foul_play.py \
  --websocket-uri ws://localhost:8000/showdown/websocket \
  --ps-username foulplayphase0d \
  --bot-mode accept_challenge \
  --pokemon-format gen9randombattle \
  --run-count 1 \
  --search-time-ms 25 \
  --search-parallelism 1 \
  --search-threads 1 \
  --log-level INFO
```

Verification performed:

- Stock Foul Play connected to the local server.
- Stock Foul Play accepted a `gen9randombattle` challenge from poke-env `RandomPlayer` and completed the game.
- Smoke-test result: Foul Play won 1 game, and the process logged `Winner: foulplayphase0d`.

Foul Play evaluation location for Phase 1:

- Foul Play calls `poke_engine.monte_carlo_tree_search` from `external/foul-play/fp/search/main.py::get_result_from_mcts`.
- The leaf/position evaluation used by the pinned gen9/terastallization build is `poke-engine==0.0.47`, `src/genx/evaluate.rs::evaluate(state: &State) -> f32`.
- MCTS uses it at `poke-engine==0.0.47`, `src/mcts.rs::Node::rollout` and `src/mcts.rs::perform_mcts` for `root_eval`.
