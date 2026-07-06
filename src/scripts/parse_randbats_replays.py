#!/usr/bin/env python3
"""Parse scraped gen9randombattle replays into metamon BC trajectories.

Idempotent: skips replays whose gameid already has parsed output, so it can be
re-run as the scraper accrues more data. Uses the native RandbatsPoolPredictor
(exact generator prior) for team completion.

Run in .venv-metamon:
  METAMON_CACHE_DIR=external/metamon_cache .venv-metamon/bin/python \
      scripts/parse_randbats_replays.py --workers 8
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_worker_parser = None


def _init_worker(out_dir: str, pool_path: str, seed: int) -> None:
    global _worker_parser
    warnings.filterwarnings("ignore")
    os.environ.setdefault("METAMON_CACHE_DIR", str(ROOT / "external" / "metamon_cache"))
    from belief.randbats_predictor import RandbatsPoolPredictor
    from metamon.backend.replay_parser.parse_replays import ReplayParser

    _worker_parser = ReplayParser(
        replay_output_dir=out_dir,
        team_predictor=RandbatsPoolPredictor(pool_path=pool_path, seed=seed),
    )


def _parse_one(path: str) -> tuple[str, str]:
    global _worker_parser
    try:
        _worker_parser.parse_replay(path)
        return (path, "ok")
    except Exception as e:  # noqa: BLE001 - never kill the pool
        return (path, f"{type(e).__name__}: {str(e)[:100]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", default=str(ROOT / "data" / "replays" / "gen9randombattle"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "parsed_replays" / "gen9randombattle"))
    parser.add_argument("--pool-path", default=str(ROOT / "data" / "randbats_pools" / "gen9randombattle_pool_50000.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done_gameids = set()
    for f in out_dir.glob("*.json.lz4"):
        done_gameids.add(f.name.split("_")[0])

    todo = []
    for path in sorted(glob.glob(os.path.join(args.replay_dir, "*.json"))):
        gameid = os.path.basename(path).replace(".json", "")
        if gameid not in done_gameids:
            todo.append(path)

    print(f"{datetime.now(timezone.utc).isoformat()} replays={len(done_gameids)} done, {len(todo)} to parse", flush=True)
    if not todo:
        return

    ok = 0
    failed = 0
    with mp.Pool(
        args.workers, initializer=_init_worker,
        initargs=(str(out_dir), args.pool_path, args.seed),
    ) as pool:
        for i, (path, status) in enumerate(pool.imap_unordered(_parse_one, todo), 1):
            if status == "ok":
                ok += 1
            else:
                failed += 1
            if i % 250 == 0 or i == len(todo):
                print(f"{datetime.now(timezone.utc).isoformat()} progress {i}/{len(todo)} ok={ok} failed={failed}", flush=True)

    outputs = len(list(out_dir.glob("*.json.lz4")))
    summary = {"parsed_ok": ok, "failed": failed, "total_pov_trajectories": outputs,
               "ts": datetime.now(timezone.utc).isoformat()}
    print(json.dumps(summary), flush=True)
    (out_dir.parent / "parse_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
