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
