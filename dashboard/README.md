# Live Ladder Dashboard

Static battle-broadcast UI and Vercel API for `pokemon.amtayeb.dev`.

## Local

From the repository root, publish the active run into the ignored local status
file:

```bash
python3 -m srcs.metagross.dashboard_publisher \
  --run-dir /path/to/ladder/run \
  --output dashboard/public/status.json \
  --interval 2
```

In another terminal:

```bash
npm --prefix dashboard run serve
```

Open `http://localhost:4173`.

## Vercel

Create a Vercel project with `dashboard` as its root directory. Attach an
Upstash Redis integration and configure:

- `DASHBOARD_INGEST_SECRET`
- `KV_REST_API_URL` and `KV_REST_API_TOKEN`, or the equivalent
  `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN`

Point `pokemon.amtayeb.dev` at that project. Then run the publisher with:

```bash
export METAGROSS_DASHBOARD_SECRET='...'
python3 -m srcs.metagross.dashboard_publisher \
  --run-dir /path/to/ladder/run \
  --ingest-url https://pokemon.amtayeb.dev/api/ingest \
  --interval 2
```

The browser polls `/api/status` every three seconds. The publisher posts on
protocol changes and sends a heartbeat at least every 15 seconds. Redis stores
one current sanitized snapshot; deployments are not triggered for battle
updates.

## Discord

Configure these Vercel environment variables:

- `DISCORD_APPLICATION_ID`
- `DISCORD_PUBLIC_KEY`
- `DISCORD_BOT_TOKEN`
- `DISCORD_WEBHOOK_URL`

Set the Discord application's Interactions Endpoint URL to:

```text
https://pokemon.amtayeb.dev/api/discord
```

Discord's verification ping registers the global `/ladder` command. Global
commands can take up to one hour to appear after registration. `/ladder` shows
the live record, Elo, GXE, Glicko estimate/deviation, and frozen r1/G3/G4
comparisons.

The incoming webhook is outbound-only. The ingest function sends deduplicated
alerts when process/network/prior/choice failures appear or when a new loss is
classified as inactivity/auto-forfeit. Rotate webhook URLs as credentials and
never place them in source files.

## Privacy Boundary

Only public Showdown protocol information and aggregate search throughput leave
the runner. The publisher does not include raw requests, own-team details,
unresolved choices, priors, policy observations, sampled opponent worlds,
credentials, PIDs, or local paths.
