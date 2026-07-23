#!/usr/bin/env bash
# Ephemeral BC fine-tune of Kakuna on gen9randombattle, on a remote H200 box.
#
# HARD REQUIREMENT (user): nothing from this project persists on the box.
# Everything lives under one scratch dir (incl. HF + pip caches); the final
# step of `cleanup` removes it. Run `bash h200_finetune_randbats.sh cleanup`
# manually if the run is interrupted.
#
# Usage (run ON the H200 box, after scp'ing the payload):
#   bash h200_finetune_randbats.sh setup     # venv + deps + data unpack
#   bash h200_finetune_randbats.sh train     # the fine-tune (detached-friendly)
#   bash h200_finetune_randbats.sh cleanup   # remove ALL traces
set -euo pipefail

SCRATCH="${METAGROSS_SCRATCH:-/tmp/metagross_scratch}"
RUN_NAME="${RUN_NAME:-randbats_bc_v0}"
EPOCHS="${EPOCHS:-6}"
STEPS="${STEPS:-1000}"
BATCH="${BATCH:-24}"

export METAMON_CACHE_DIR="$SCRATCH/metamon_cache"
export HF_HOME="$SCRATCH/hf_home"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"
export WANDB_MODE=disabled
export XDG_CACHE_HOME="$SCRATCH/xdg_cache"

setup() {
  mkdir -p "$SCRATCH" "$METAMON_CACHE_DIR" "$HF_HOME"
  cd "$SCRATCH"
  python3 -m venv venv
  ./venv/bin/pip install --upgrade pip -q
  # metamon checkout is rsynced from the laptop (includes local patches:
  # format alias, wrappers guard, vanilla-attention gins)
  ./venv/bin/pip install -q -e "$SCRATCH/metamon" torch
  ./venv/bin/pip install -q "amago[flash] @ git+https://github.com/UT-Austin-RPL/amago.git" 2>/dev/null \
    || ./venv/bin/pip install -q "amago @ git+https://github.com/UT-Austin-RPL/amago.git"
  mkdir -p "$SCRATCH/data/parsed_replays"
  tar xzf "$SCRATCH/randbats_parsed_23k.tgz" -C "$SCRATCH/data/parsed_replays"
  # dataset config: 100% custom randbats replays
  cat > "$SCRATCH/randbats_bc.yaml" <<EOF
replay_weight: 0.0
custom_replays:
  - dir: $SCRATCH/data/parsed_replays
    weight: 1.0
formats:
  - gen9randombattle
EOF
  echo "SETUP DONE"
}

train() {
  cd "$SCRATCH"
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
  ./venv/bin/python -m metamon.rl.finetune \
    --run_name "$RUN_NAME" \
    --save_dir "$SCRATCH/ckpts" \
    --base_model Kakuna \
    --dataset_config "$SCRATCH/randbats_bc.yaml" \
    --eval_gens \
    --epochs "$EPOCHS" \
    --steps_per_epoch "$STEPS" \
    --batch_size_per_gpu "$BATCH" \
    --ckpt_interval 1 \
    --dloader_workers 8 2>&1 | tee "$SCRATCH/train.log"
  echo "TRAIN DONE; checkpoints:"
  find "$SCRATCH/ckpts" -name "policy_epoch_*.pt" | sort
}

probe() {
  # STEP 0: throughput probe — short run, reports steps/sec + projected hours
  cd "$SCRATCH"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
  PYTHONPATH="$SCRATCH/repo" ./venv/bin/python "$SCRATCH/repo/scripts/run_finetune_variant.py" \
    --variant base --probe \
    --dataset-config "$SCRATCH/randbats_bc.yaml" \
    --save-dir "$SCRATCH/ckpts" 2>&1 | tee "$SCRATCH/probe.log"
  grep -E "THROUGHPUT|PROJECTION" "$SCRATCH/probe.log"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
}

sweep() {
  # STEP 1: ablation matrix — small-scale screen runs, one variable each.
  # Requires the repo dir (train/ + scripts/) rsynced to $SCRATCH/repo.
  cd "$SCRATCH"
  for VARIANT in base A_rating B_klanchor C_binary D_hlgauss ALL; do
    echo "=== SWEEP VARIANT $VARIANT ==="
    PYTHONPATH="$SCRATCH/repo" ./venv/bin/python "$SCRATCH/repo/scripts/run_finetune_variant.py" \
      --variant "$VARIANT" \
      --dataset-config "$SCRATCH/randbats_bc.yaml" \
      --save-dir "$SCRATCH/ckpts" \
      --epochs "${SWEEP_EPOCHS:-2}" --steps-per-epoch "${SWEEP_STEPS:-500}" \
      --batch-size "$BATCH" 2>&1 | tee "$SCRATCH/sweep_$VARIANT.log" \
      || echo "FATAL variant $VARIANT failed; continuing"
  done
  echo "SWEEP DONE; checkpoints:"
  find "$SCRATCH/ckpts" -name "policy_epoch_*.pt" | sort
}

cleanup() {
  rm -rf "$SCRATCH"
  echo "CLEANUP DONE; residue check:"
  ls -la /tmp | grep -i metagross || echo "  no metagross residue in /tmp"
}

"$@"
