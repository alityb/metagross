# Local I/O With Remote Search

Pokemon Showdown explicitly locks the tested Byteful Static Residential ISP
proxy. Public-ladder WebSocket traffic therefore stays on the operator's normal
home connection. Only serialized poke-engine states and policy priors are sent
to the selected remote provider for CPU-intensive search.

## Search Contract

The `r1-p16-cloud-max` variant preserves checkpoint 5, `c_puct=2.0`, one engine
thread, and the 500 ms per-world budget. Parallelism 16 doubles frozen P8's
adaptive coverage:

- 64 worlds at 250 ms early
- 32 worlds at 500 ms later
- One authenticated Modal RPC per decision
- 16 persistent cloud worker processes on 16 physical cores (32 vCPUs) and 16 GiB memory
- Cloud resource identity embedded in each remote engine response
- No local-search fallback

This is a stronger deployment variant and must not be labeled as frozen P8 r1.

## Deployment

Install the pinned Modal client in the Foul Play environment and deploy the
search service:

```bash
.venv-fp-priors/bin/python -m pip install -r srcs/requirements-foul-play.txt
modal deploy srcs/metagross/modal_mcts.py
```

Retrieve and record the deployed native engine hash:

```bash
python3 -c 'import json, modal; print(json.dumps(modal.Function.from_name(
"metagross-mcts-r1-p16", "engine_info").remote(), sort_keys=True))'
```

The launcher refuses remote search unless that exact SHA-256 is supplied.

## Ladder Launch

`METAGROSS_SHOWDOWN_PASSWORD` must be available locally. It is removed from the
prior-server environment and never sent to Modal.

```bash
METAGROSS_SHOWDOWN_PASSWORD="$METAGROSS_PASSWORD" \
  .venv-metamon/bin/python -m srcs.metagross.launch \
  --username ACCOUNT \
  --profile r1 \
  --games 3 \
  --search-parallelism 16 \
  --search-threads 1 \
  --remote-mcts \
  --remote-engine-sha256 REMOTE_NATIVE_SHA256 \
  --output-root srcs/runtime/local-r1-p16-ACCOUNT
```

The Mac must remain powered, awake, and online, but performs no MCTS. It retains
battle parsing, hidden-world sampling, weighted move selection, Showdown I/O,
and full trajectory capture. Any remote error terminates the run rather than
mixing local and cloud search.

## AWS EC2 Over A Private Tunnel

The HTTP provider has no third-party runtime dependency beyond the pinned
`poke_engine`. On the EC2 host, install that engine from this checkout, choose
the actual instance type, and set a high-entropy token:

```bash
export METAGROSS_REMOTE_MCTS_TOKEN="$(openssl rand -hex 32)"
export METAGROSS_AWS_INSTANCE_TYPE=c7a.8xlarge
.venv-fp-priors/bin/python -m srcs.metagross.aws_http_mcts
```

The service binds only to `127.0.0.1:8765` by default and creates 16 persistent
worker processes. Keep the EC2 security group closed to port 8765. From the
ladder machine, forward a private loopback port over SSH:

```bash
ssh -N -L 8765:127.0.0.1:8765 ec2-user@EC2_PUBLIC_HOST
```

Set the same token in a separate local shell and verify the authenticated
engine identity through the tunnel:

```bash
export METAGROSS_REMOTE_MCTS_TOKEN=THE_EC2_TOKEN
curl -H "Authorization: Bearer $METAGROSS_REMOTE_MCTS_TOKEN" \
  http://127.0.0.1:8765/health
```

Record `engine.native_sha256`, then launch the G4 canary. The URL is safe to
record, but the token remains environment-only and is excluded from the prior
server environment, child arguments, and manifest.

```bash
METAGROSS_SHOWDOWN_PASSWORD="$METAGROSS_PASSWORD" \
  .venv-metamon/bin/python -m srcs.metagross.launch \
  --username ACCOUNT \
  --profile g4 \
  --games 3 \
  --confirm-g4-canary \
  --search-parallelism 16 \
  --search-threads 1 \
  --remote-mcts \
  --remote-mcts-transport http \
  --remote-mcts-url http://127.0.0.1:8765/search \
  --remote-mcts-instance-type c7a.8xlarge \
  --remote-engine-sha256 REMOTE_NATIVE_SHA256 \
  --output-root srcs/runtime/aws-g4-p16-ACCOUNT
```

## Admission

Before a larger campaign, require a three-game canary with:

- Completed manifest
- Exact decision/search row join
- 64 early or 32 later worlds per decision
- Matching remote engine hash in every response
- No disconnect, timeout, invalid choice, missing prior, fallback, or process error
- Valid decision, search, protocol, and rating JSONL artifacts
