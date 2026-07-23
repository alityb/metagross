# Operations

## Launch

```bash
export METAGROSS_SHOWDOWN_PASSWORD='...'
.venv-metamon/bin/python -m srcs.metagross.launch \
  --username YOUR_SHOWDOWN_ACCOUNT \
  --games 200
```

Do not store the password in the repository. The known local credential files
are ignored by `.gitignore`.

## Health

The launcher waits for the prior server before starting Foul Play. During a run:

```bash
curl http://127.0.0.1:8977/health
```

Expected response:

```json
{"ok": true, "sessions": 0}
```

The session count varies while games are active.

## Shutdown

Send `SIGINT` with `Ctrl-C` or terminate the launcher. Its cleanup handler
terminates both child processes and escalates only if they do not exit within
ten seconds.

## Invariants

- Format is `gen9randombattle`.
- Search budget is 500 ms with parallelism 8.
- Search thread count is exactly 1.
- `c_puct` is 2.0.
- Policy is `randbats_exit_r1`, epoch 5.
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
