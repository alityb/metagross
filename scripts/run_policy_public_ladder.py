#!/usr/bin/env python3
"""Ladder a metamon policy (HF pretrained or local fine-tuned) on the PUBLIC
Pokémon Showdown ladder for gen9randombattle.

- Registered account required (username + password via METAGROSSS env or flag).
- Appends per-battle results to metamon's CSV (append-only, crash-safe).
- Polls the PS users API for ELO/GXE/Glicko RD every --poll-seconds, appending
  to ratings.jsonl (the trajectory the gate report needs).

Run in .venv-metamon.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def poll_ratings(username: str, fmt: str, out_path: Path, stop: threading.Event,
                 poll_seconds: float) -> None:
    url = f"https://pokemonshowdown.com/users/{username.lower()}.json"
    while not stop.is_set():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (metagross ladder monitor)"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode())
            rating = (data.get("ratings") or {}).get(fmt) or {}
            line = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "elo": rating.get("elo"),
                "gxe": rating.get("gxe"),
                "rpr": rating.get("rpr"),
                "rprd": rating.get("rprd"),
                "w": rating.get("w"),
                "l": rating.get("l"),
            }
            with out_path.open("a") as f:
                f.write(json.dumps(line) + "\n")
            print(f"RATING {json.dumps(line)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"RATING_POLL_WARN {e!r}", flush=True)
        stop.wait(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="Kakuna")
    parser.add_argument("--local-run-dir", default=None)
    parser.add_argument("--local-run-name", default=None)
    parser.add_argument("--local-base-model", default="Kakuna")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default=os.environ.get("METAGROSS_SHOWDOWN_PASSWORD"))
    parser.add_argument("--battle-format", default="gen9randombattle")
    parser.add_argument("--total-battles", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--poll-seconds", type=float, default=90.0)
    args = parser.parse_args()

    if not args.password:
        print("FATAL: registered-account password required "
              "(--password or METAGROSS_SHOWDOWN_PASSWORD)", flush=True)
        sys.exit(2)

    os.environ.setdefault("WANDB_MODE", "disabled")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from metamon.rl.pretrained import get_pretrained_model
    import metamon.rl.pretrained as _pt
    from metamon.rl.evaluate.__main__ import pretrained_vs_public_ladder

    if args.local_run_dir:
        base_cls = getattr(_pt, args.local_base_model)
        if args.checkpoint is None:
            print("FATAL: --checkpoint required with --local-run-dir", flush=True)
            sys.exit(2)
        model = _pt.LocalFinetunedModel(
            base_model=base_cls,
            amago_ckpt_dir=args.local_run_dir,
            model_name=args.local_run_name,
            default_checkpoint=args.checkpoint,
        )
        print(f"LADDER_RUNNER loading LOCAL model run={args.local_run_name} "
              f"ckpt={args.checkpoint} base={args.local_base_model}", flush=True)
    else:
        model = get_pretrained_model(args.agent)
        print(f"LADDER_RUNNER loading agent={args.agent} "
              f"default_checkpoint={model.default_checkpoint}", flush=True)

    stop = threading.Event()
    poller = threading.Thread(
        target=poll_ratings,
        args=(args.username, args.battle_format, out_dir / "ratings.jsonl", stop,
              args.poll_seconds),
        daemon=True,
    )
    poller.start()

    try:
        results = pretrained_vs_public_ladder(
            pretrained_model=model,
            username=args.username,
            password=args.password,
            battle_format=args.battle_format,
            team_set=None,
            total_battles=args.total_battles,
            checkpoint=args.checkpoint,
            action_temperature=args.temperature,
            save_results_to=str(out_dir),
            save_trajectories_to=str(out_dir / "trajectories"),
        )
        print("LADDER_RUNNER final results: "
              + json.dumps(results, indent=2, sort_keys=True, default=str), flush=True)
    finally:
        stop.set()


if __name__ == "__main__":
    main()
