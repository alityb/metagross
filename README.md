# Metagross

Metagross is a Pokemon Showdown `gen9randombattle` agent that combines Foul
Play's determinized search with a fine-tuned 142M-parameter Metamon policy at
the search root.

The accepted r1 deployment reached a settled **92.4-92.7 GXE at RD 25**, with
an observed peak of 93.6, on the `metaexitr1` public-ladder account. It improved
on the project's previous 91.4 GXE result. See
[`results/accepted-r1`](results/accepted-r1/README.md) for the exact claim and
its evidence limitations.

## Accepted Agent

| Component | Frozen value |
|---|---|
| Format | `gen9randombattle` |
| Search | Patched Foul Play / poke-engine |
| Policy | `randbats_exit_r1`, epoch 5 |
| Search budget | 500 ms |
| Parallel worlds | 8 |
| Search threads | 1 |
| PUCT coefficient | 2.0 |
| Runtime | `srcs/metagross/` |

Only this accepted agent is exposed as production code. Training pipelines,
failed candidates, datasets, old evaluation harnesses, and research notes are
archived under [`experimental/`](experimental/README.md).

## Repository

- [`docs/`](docs/README.md): architecture, setup, operation, and provenance.
- [`results/`](results/README.md): curated evidence for the accepted bot.
- [`srcs/`](srcs/README.md): production runtime, patched engine, and manifests.
- [`experimental/`](experimental/README.md): historical research workspace.

## Run

Complete [`docs/setup.md`](docs/setup.md), set the Showdown password without
committing it, and start the frozen deployment:

```bash
export METAGROSS_SHOWDOWN_PASSWORD='...'
.venv-metamon/bin/python -m srcs.metagross.launch \
  --username YOUR_SHOWDOWN_ACCOUNT \
  --games 200
```

The launcher starts the local r1 policy server, waits for its health check, and
then starts Foul Play with the accepted parameters. It fails closed if player
root priors are unavailable.

## Scope

This release is the accepted research artifact, not a hosted service. The
checkpoint is a local 545 MiB artifact and is not committed to Git. Its expected
path and SHA-256 digest are recorded in
[`results/accepted-r1/artifacts.json`](results/accepted-r1/artifacts.json).
