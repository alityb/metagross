#!/usr/bin/env python3
"""Run a pretrained Metamon policy (default Kakuna) in challenge mode on gen9randombattle.

Bypasses `metamon.rl.evaluate`'s CLI (which requires team-based formats) by calling
`pretrained_vs_challenge` directly with `team_set=None` and registering a
FORMAT_ALIASES entry so the agent observes an in-distribution format token
("gen9ou") while actually playing gen9randombattle.

Run inside .venv-metamon with METAMON_CACHE_DIR set, local Showdown on :8000.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="Kakuna")
    parser.add_argument("--username", required=True)
    parser.add_argument("--opponent-username", required=True)
    parser.add_argument("--role", choices=["challenger", "acceptor"], required=True)
    parser.add_argument("--battle-format", default="gen9randombattle")
    parser.add_argument("--alias-to", default="gen9ou",
                        help="Format the agent believes it is playing (obs-space token)")
    parser.add_argument("--total-battles", type=int, default=1)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save-results-to", required=True)
    parser.add_argument("--save-trajectories-to", default=None)
    parser.add_argument("--local-run-dir", default=None,
                        help="Load a LocalFinetunedModel from this --save_dir instead of the HF pretrained")
    parser.add_argument("--local-run-name", default=None)
    parser.add_argument("--local-base-model", default="Kakuna")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")

    from metamon import config as metamon_config

    if args.alias_to:
        metamon_config.FORMAT_ALIASES[args.battle_format.lower()] = args.alias_to
        print(f"KAKUNA_RUNNER format alias registered: "
              f"{args.battle_format.lower()} -> {args.alias_to}", flush=True)

    from metamon.rl.pretrained import get_pretrained_model
    from metamon.rl.evaluate.__main__ import pretrained_vs_challenge

    if args.local_run_dir:
        from metamon.rl.pretrained import LocalFinetunedModel, get_pretrained_model_names
        import metamon.rl.pretrained as _pt

        base_cls = getattr(_pt, args.local_base_model)
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required with --local-run-dir")
        pretrained_model = LocalFinetunedModel(
            base_model=base_cls,
            amago_ckpt_dir=args.local_run_dir,
            model_name=args.local_run_name,
            default_checkpoint=args.checkpoint,
        )
        print(f"KAKUNA_RUNNER loading LOCAL finetuned model run={args.local_run_name} "
              f"dir={args.local_run_dir} checkpoint={args.checkpoint} "
              f"base={args.local_base_model}", flush=True)
    else:
        pretrained_model = get_pretrained_model(args.agent)
        print(f"KAKUNA_RUNNER loading agent={args.agent} "
              f"model_name={pretrained_model.model_name} "
              f"default_checkpoint={pretrained_model.default_checkpoint} "
              f"requested_checkpoint={args.checkpoint}", flush=True)

    results = pretrained_vs_challenge(
        pretrained_model=pretrained_model,
        username=args.username,
        opponent_username=args.opponent_username,
        role=args.role,
        battle_format=args.battle_format,
        team_set=None,
        total_battles=args.total_battles,
        checkpoint=args.checkpoint,
        action_temperature=args.temperature,
        save_results_to=args.save_results_to,
        save_trajectories_to=args.save_trajectories_to,
    )
    print("KAKUNA_RUNNER final results: "
          + json.dumps(results, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    sys.exit(main())
