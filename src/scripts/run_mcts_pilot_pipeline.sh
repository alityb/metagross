#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KEY="/Users/alityb/.ssh/metagross-r2.pem"
RAW_ROOT="$ROOT/experiments/mcts_high_budget_distributed/raw"

workers=(
  "w1 3.81.184.0 /opt/metagross/data/mcts_high_budget_v2_20260715_w1"
  "w2 13.217.116.63 /opt/metagross/data/mcts_high_budget_v2_20260715_w2"
  "w3 100.53.200.213 /opt/metagross/data/mcts_high_budget_v2_20260715_w3"
)

while true; do
  running=0
  for worker in "${workers[@]}"; do
    read -r _ host _ <<<"$worker"
    if ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 "ubuntu@$host" \
      "pgrep -f '[p]fsp_generate.py.*mcts_high_budget_v2' >/dev/null"; then
      running=1
    fi
  done
  [[ "$running" -eq 0 ]] && break
  sleep 120
done

mkdir -p "$RAW_ROOT"
for worker in "${workers[@]}"; do
  read -r name host remote_root <<<"$worker"
  rsync -az -e "ssh -i $KEY -o BatchMode=yes" \
    "ubuntu@$host:$remote_root/raw/" "$RAW_ROOT/$name/"
done

PYTHONPATH="$ROOT/src" "$ROOT/.venv-metamon/bin/python" \
  "$ROOT/src/scripts/finalize_schema_v2_pfsp.py" \
  --raw-root "$RAW_ROOT" \
  --parsed-root "$ROOT/data/mcts_high_budget_distributed_parsed" \
  --learner-only-root "$ROOT/data/mcts_high_budget_distributed_learner_only" \
  --trajectory-index "$ROOT/data/mcts_high_budget_distributed/trajectory_index.jsonl" \
  --sidecar "$ROOT/data/mcts_high_budget_distributed/mcts_policy_targets.jsonl" \
  --pool-path "$ROOT/data/randbats_pools/gen9randombattle_pool_50000.json" \
  --report "$ROOT/experiments/mcts_high_budget_distributed/finalization_report.json"

"$ROOT/.venv-metamon/bin/modal" run "$ROOT/src/scripts/modal_train_mcts_distillation.py" \
  --learner-only-root "$ROOT/data/mcts_high_budget_distributed_learner_only" \
  --sidecar "$ROOT/data/mcts_high_budget_distributed/mcts_policy_targets.jsonl" \
  --human-anchor-root "$ROOT/data/parsed_replays"

"$ROOT/.venv-metamon/bin/modal" volume get metagross-mcts-distillation-pilot \
  /ckpts/mcts_schema_v2_distillation_pilot \
  "$ROOT/src/nets/checkpoints/mcts_schema_v2_distillation_pilot"

METAMON_CACHE_DIR="$ROOT/external/metamon_cache" TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true \
  nohup "$ROOT/.venv-metamon/bin/python" -u "$ROOT/src/scripts/prior_server.py" \
  --local-run-dir "$ROOT/src/nets/checkpoints/mcts_schema_v2_distillation_pilot" \
  --local-run-name mcts_schema_v2_distillation_pilot --checkpoint 1 \
  --port 8988 --username candidate >/dev/null 2>&1 &

sleep 20
PYTHONPATH="$ROOT/src" "$ROOT/.venv-metamon/bin/python" -m eval.run \
  --mode h2h --server local --format gen9randombattle \
  --agent-a foul_play_root_priors_opp --agent-b foul_play_root_priors_opp \
  --agent-a-prior-server-url http://127.0.0.1:8988 \
  --agent-b-prior-server-url http://127.0.0.1:8982 \
  --agent-a-replay-dir "$ROOT/experiments/mcts_schema_v2_promotion/replays" \
  --agent-a-decision-log "$ROOT/experiments/mcts_schema_v2_promotion/agent_a_decisions.jsonl" \
  --agent-a-require-priors --agent-b-require-priors \
  --foul-play-python "$ROOT/.venv-fp-priors/bin/python" \
  --foul-play-search-time-ms 500 --foul-play-search-parallelism 8 \
  --foul-play-search-threads 1 --n-games 500 --paired \
  --json-out "$ROOT/experiments/mcts_schema_v2_promotion/result.json" \
  --log-dir "$ROOT/experiments/mcts_schema_v2_promotion/logs"
