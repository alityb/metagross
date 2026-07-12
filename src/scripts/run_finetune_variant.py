#!/usr/bin/env python3
"""Run one ablation variant of the randbats fine-tune (Phase-1 sweep).

Variants (ONE toggle each, per AGENTS.md §6.8):
  base       — stock metamon finetune.gin
  A_rating   — rating-band conditioning (dataset monkeypatch)
  B_klanchor — frozen-KL anchor agent (light coeff 0.02)
  C_binary   — binary advantage filter instead of IS/exp
  D_hlgauss  — HL-Gauss value classification labels
  ALL        — A + B + D combined (C stays default unless it wins its screen)

Usage (on the GPU box, inside the metamon venv, repo root on PYTHONPATH):
  python scripts/run_finetune_variant.py --variant base \
      --dataset-config <yaml> --save-dir <ckpts> \
      --epochs 5 --steps-per-epoch 1000 --batch-size 24 [--probe]

--probe runs a short throughput probe (200 steps, subsampled) instead of a
full run, and reports steps/sec + projected hours.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARIANTS = {
    "base": {"gin": None, "rating": False, "kl": False, "binary": False},
    "A_rating": {"gin": None, "rating": True, "kl": False, "binary": False},
    "B_klanchor": {"gin": "metagross_B_klanchor.gin", "rating": False, "kl": True, "binary": False},
    "C_binary": {"gin": "metagross_C_binary.gin", "rating": False, "kl": False, "binary": True},
    "D_hlgauss": {"gin": "metagross_D_hlgauss.gin", "rating": False, "kl": False, "binary": False},
    "ALL": {"gin": "metagross_ALL.gin", "rating": True, "kl": True, "binary": False},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--base-model", default="Kakuna")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--dloader-workers", type=int, default=8)
    parser.add_argument("--prev-run-dir", default=None)
    parser.add_argument("--prev-run-name", default=None)
    parser.add_argument("--prev-checkpoint", type=int, default=None)
    parser.add_argument("--probe", action="store_true",
                        help="Short throughput probe: 1 epoch x 200 steps")
    args = parser.parse_args()

    spec = VARIANTS[args.variant]
    os.environ.setdefault("WANDB_MODE", "disabled")

    if spec["rating"]:
        from train.finetune_toggles import install_rating_conditioning
        install_rating_conditioning()
    if spec["kl"]:
        from train.finetune_toggles import install_kl_agent
        install_kl_agent()
    if spec["binary"]:
        from train.finetune_toggles import install_binary_filter
        install_binary_filter()

    # gin files live in the repo (train/gins) and are copied next to metamon's
    # own configs so relative resolution works
    if spec["gin"]:
        import metamon.rl.train as mt
        gin_src = ROOT / "train" / "gins" / spec["gin"]
        gin_dst_dir = Path(mt.__file__).parent / "configs" / "training"
        gin_dst = gin_dst_dir / spec["gin"]
        gin_dst.write_text(gin_src.read_text())
        train_gin = spec["gin"]
    else:
        train_gin = "finetune.gin"

    run_name = f"randbats_{args.variant}"
    epochs = 1 if args.probe else args.epochs
    steps = 200 if args.probe else args.steps_per_epoch

    argv = [
        "--run_name", run_name,
        "--save_dir", args.save_dir,
        "--base_model", args.base_model,
        "--dataset_config", args.dataset_config,
        "--train_gin_config", train_gin,
        "--eval_gens",
        "--epochs", str(epochs),
        "--steps_per_epoch", str(steps),
        "--batch_size_per_gpu", str(args.batch_size),
        "--ckpt_interval", "1",
        "--dloader_workers", str(args.dloader_workers),
    ]
    if args.prev_run_dir is not None:
        if args.prev_run_name is None or args.prev_checkpoint is None:
            parser.error("--prev-run-dir requires --prev-run-name and --prev-checkpoint")
        argv.extend(
            [
                "--prev_run_dir", args.prev_run_dir,
                "--prev_run_name", args.prev_run_name,
                "--prev_checkpoint", str(args.prev_checkpoint),
            ]
        )
    print(f"VARIANT={args.variant} train_gin={train_gin} argv={argv}", flush=True)

    # metamon.rl.finetune's logic lives under `if __name__ == "__main__"`;
    # run it as __main__ with a patched argv (toggles already installed above).
    import runpy

    sys.argv = ["metamon.rl.finetune"] + argv
    t0 = time.monotonic()
    runpy.run_module("metamon.rl.finetune", run_name="__main__")
    dt = time.monotonic() - t0

    total_steps = epochs * steps
    print(f"THROUGHPUT variant={args.variant} wall_s={dt:.1f} "
          f"steps={total_steps} steps_per_sec={total_steps/dt:.3f}", flush=True)
    if args.probe:
        full = args.epochs * args.steps_per_epoch
        print(f"PROJECTION full_run_steps={full} "
              f"projected_hours={full/(total_steps/dt)/3600:.2f}", flush=True)


if __name__ == "__main__":
    main()
