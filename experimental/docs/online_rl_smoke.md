# Staged Online-RL Smoke

This smoke continues the frozen `randbats_exit_r1` epoch-5 policy using fresh
direct-policy Showdown trajectories. It does not modify the accepted engine and
does not run Foul Play during collection.

## Contracts

- Every local population profile pins a checkpoint number and SHA-256.
- A collection shard keeps both policies frozen until all requested battles end.
- Learner roles are balanced between challenger and acceptor.
- Every completed battle must produce exactly one learner trajectory with a
  terminal `WIN` or `LOSS` outcome.
- Training resumes r1 epoch 5 with a fixed mixture: 70% legacy r1 self-play,
  20% fresh current-policy trajectories, and 10% human replays.
- Training uses the terminal-only Metamon `BinaryReward` (its conventional
  `+100/-100` scaling), the `B_klanchor` variant, and refuses empty mixture
  components.
- Modal work runs in the `dnfcubes` workspace and uses the
  `metagross-online-rl` Volume. No Modal Sandbox is required.

## Collection

Start the pinned local Showdown server:

```bash
bash experimental/src/scripts/start_showdown.sh 8000
```

Validate commands without launching actors:

```bash
python3 experimental/src/scripts/online_rl_generate.py \
  --pool experimental/configs/online_rl_smoke_pool.json \
  --schedule experimental/configs/online_rl_smoke_schedule.json \
  --out-dir experimental/runs/online_rl_smoke \
  --dry-run
```

Remove `--dry-run` to collect the configured eight-battle plumbing smoke. The
collector uses `.venv-metamon/bin/python`, writes an atomic `MANIFEST.json`, and
saves only learner-POV trajectories.

## Training

Activate and verify the required Modal workspace:

```bash
modal profile activate dnfcubes
modal profile current
```

Launch the 200-step guarded continuation after the three dataset roots exist:

```bash
modal run experimental/src/scripts/modal_train_online_rl.py \
  --fresh-root experimental/runs/online_rl_smoke_v8_20260728 \
  --legacy-archive-dir /path/to/legacy_shards \
  --human-archive-dir /path/to/human_shards \
  --run-name randbats_online_g1_smoke \
  --steps 200 \
  --batch-size 24
```

The launcher uploads immutable archives and the exact r1 checkpoint to the
Volume before starting one H100. It writes `ONLINE_RL_MANIFEST.json` beside the
resulting checkpoint.

## Autonomous Generations

The resumable controller keeps one local Showdown service alive, collects
current-policy PFSP trajectories, continues from the latest validated snapshot,
runs a separate holdout arena, and atomically advances or rolls back the lineage:

```bash
python3 experimental/src/scripts/online_rl_controller.py \
  --config experimental/configs/online_rl_autonomous_3gen.json \
  --run-dir experimental/runs/online_rl_autonomous_3gen_20260729
```

`STATE.json` is the restart contract. Training trajectories live under each
generation's `collection/`; arena trajectories are under `arena_holdout/` with
`HOLDOUT.json` and must never enter replay. A point estimate below 45% rolls the
lineage back. Safety-passing snapshots may remain in the population, while
accepted-policy promotion requires at least 400 holdout games and a Wilson 95%
lower bound above 50%.
