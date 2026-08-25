#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ali/search-policy-round2
PYTHON=/home/ali/metagross-eval/.venv/bin/python
export PYTHONPATH="$ROOT/src"
export METAMON_CACHE_DIR=/home/ali/.cache/metamon
export HF_HOME=/home/ali/.cache/huggingface
export WANDB_MODE=disabled
export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$ROOT/output" "$ROOT/logs"

run_arm() {
  local arm=$1
  local run_name="search_policy_${arm}_round2_seed20260812"
  exec "$PYTHON" "$ROOT/src/scripts/train_search_policy_student.py" \
    --dataset "$ROOT/input/mcts_v3_targets.jsonl" \
    --base-root "$ROOT/accepted" \
    --base-run randbats_exit_r1 \
    --base-checkpoint 5 \
    --base-sha256 c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93 \
    --arm "$arm" \
    --output-root "$ROOT/output" \
    --run-name "$run_name" \
    --steps 10000 \
    --batch-size 64 \
    --eval-batch-size 256 \
    --learning-rate 0.000001 \
    --weight-decay 0.0 \
    --grad-clip 1.0 \
    --eval-interval 500 \
    --early-stop-patience 4 \
    --early-stop-min-delta 0.0001 \
    --early-stop-min-steps 3000 \
    --seed 20260812 \
    --split-seed 20260812
}

if [[ $# -ne 1 || ! $1 =~ ^(action|visits|hybrid)$ ]]; then
  echo "usage: $0 {action|visits|hybrid}" >&2
  exit 2
fi

run_arm "$1"
