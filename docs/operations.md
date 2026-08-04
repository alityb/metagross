# Operations

## Launch

```bash
export METAGROSS_SHOWDOWN_PASSWORD='...'
.venv-metamon/bin/python -m srcs.metagross.launch \
  --username YOUR_SHOWDOWN_ACCOUNT \
  --games 200
```

Do not store the password in the repository. The known local credential files
are ignored by `.gitignore`. The launcher passes the password only in
`METAGROSS_SHOWDOWN_PASSWORD`, never in a command-line argument or manifest.

The default remains the accepted r1 profile (`--profile r1`, 200 games). Before
starting either process, the launcher verifies the immutable profile's run name,
checkpoint number, and SHA-256. Run artifacts are written under
`srcs/runtime/ladder-runs/`, including `manifest.json`, `prior.log`, `client.log`,
`decisions.jsonl`, `protocol.jsonl`, `search.jsonl`, and append-only
`ratings.jsonl` samples from the public users API.

## G3/G4 Three-Game Canaries

Place the promoted G4 checkpoint at:

```text
srcs/models/randbats_online_g4_autonomous_freshfix_20260729/ckpts/policy_weights/policy_epoch_1.pt
```

Its required SHA-256 is
`cd21dbb22bfc7a92836c7c75c6287ffc1b79c58e0a2dca3d402b76b001ff7505`.
The G3 canary checkpoint is
`srcs/models/randbats_online_g3_autonomous_freshfix_20260729/ckpts/policy_weights/policy_epoch_1.pt`
with required SHA-256
`0c754bb96953b900e282de91c570aaae5c2c6f002dc2419e149d01132888815c`.

Initial candidate validation requires explicit three-game canaries:

```bash
export METAGROSS_SHOWDOWN_PASSWORD='...'
.venv-metamon/bin/python -m srcs.metagross.launch \
  --username YOUR_SHOWDOWN_ACCOUNT \
  --profile g4 \
  --games 3 \
  --confirm-g4-canary

.venv-metamon/bin/python -m srcs.metagross.launch \
  --username YOUR_SECOND_SHOWDOWN_ACCOUNT \
  --profile g3 \
  --games 3 \
  --confirm-g3-canary
```

Review the run manifest, logs, structured captures, and `ratings.jsonl` after all
three games. Longer G3/G4 runs require the separate
`--confirm-candidate-continuation` acknowledgement and are limited to 100 games
per invocation.

## Continuous G3/G4 Comparison

After both three-game canaries pass, run sequential bounded blocks under the
supervisor:

```bash
export METAGROSS_SHOWDOWN_PASSWORD='...'
.venv-metamon/bin/python -m srcs.metagross.ladder_supervisor \
  --g3-username G3_SHOWDOWN_ACCOUNT \
  --g4-username G4_SHOWDOWN_ACCOUNT \
  --block-games 25
```

The default `--cycles 0` alternates G3 then G4 until signaled. Each child launch
is independently limited to at most 100 games and requires the explicit
candidate-continuation acknowledgement. The supervisor stops on a nonzero child
exit, an incomplete manifest, prior/search errors, invalid choices, login
failures, or timeouts. It never runs both policies concurrently.

State, per-block manifests, logs, and rating snapshots are retained under
`srcs/runtime/ladder-supervisor/`. `SIGINT`, `SIGTERM`, or `SIGHUP` cleanly stops
the active launcher and marks the supervisor state as stopped.

To continue only the accepted public-ladder candidate after retiring G4, omit
the G4 account and select G3-only mode:

```bash
export METAGROSS_SHOWDOWN_PASSWORD='...'
.venv-metamon/bin/python -m srcs.metagross.ladder_supervisor \
  --g3-username G3_SHOWDOWN_ACCOUNT \
  --g3-only \
  --block-games 25
```

Manual rollback means stopping the canary and explicitly starting the pinned r1
profile; there is no silent policy fallback:

```bash
.venv-metamon/bin/python -m srcs.metagross.launch \
  --username YOUR_SHOWDOWN_ACCOUNT \
  --profile r1 \
  --games 200
```

To run sequential bounded r1 blocks until signaled, use r1-only supervisor mode:

```bash
export METAGROSS_SHOWDOWN_PASSWORD='...'
.venv-metamon/bin/python -m srcs.metagross.ladder_supervisor \
  --r1-username YOUR_SHOWDOWN_ACCOUNT \
  --block-games 25
```

Each child still verifies the pinned r1 checkpoint and production engine,
writes an independent manifest, and is bounded by the launcher. The supervisor
starts the next block only after the preceding manifest is complete and its logs
pass the fatal-pattern gate. Any failed block stops the sequence.

## Health

The launcher waits for the prior server before starting Foul Play and continues
checking it throughout the run. During a run:

```bash
curl http://127.0.0.1:8977/health
```

Expected response:

```json
{"ok": true, "sessions": 0}
```

The session count varies while games are active.

## Shutdown

`SIGINT`, `SIGTERM`, and `SIGHUP` stop the rating poller and both child process
groups. Cleanup escalates only if a process does not exit within ten seconds.
Only one launcher may use a normalized Showdown account at a time. By default,
client output may be idle for at most 1,200 seconds and total runtime is bounded
at `max(3600, games * 900)` seconds; both limits are configurable.

## Invariants

- Format is `gen9randombattle`.
- Search budget is 500 ms with parallelism 8.
- Search thread count is exactly 1.
- `c_puct` is 2.0.
- Default policy is `randbats_exit_r1`, epoch 5, with its pinned SHA-256.
- G3 and G4 canary policies are pinned to their documented epoch 1 checkpoints
  and restricted to exactly three games.
- Player root priors are mandatory.

If any invariant changes, the resulting agent is a new candidate, not the
accepted r1 bot.

## Common Failures

`prior server exited`: run `docs/setup.md` again and confirm the checkpoint and
Metamon cache are present.

`required prior fetch failed`: inspect prior-server output. Do not disable
fail-closed behavior for a claimed r1 deployment.

`unexpected keyword s1_priors`: the Foul Play environment contains stock
poke-engine. Rebuild from `srcs/vendor/poke-engine`.

Websocket disconnects during search: confirm `websockets==14.1`; the production
adapter disables keepalive pings because search can block the event loop.
