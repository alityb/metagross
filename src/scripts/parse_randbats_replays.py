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


def parse_replay_dir(
    replay_dir: Path,
    out_dir: Path,
    pool_path: Path,
    workers: int = 1,
    seed: int = 0,
) -> dict[str, int]:
    """Parse one flat replay directory without replacing existing game outputs.

    ``workers=1`` runs in-process, which lets post-collection finalization stay
    process-free. Higher values retain the standalone parser's pool behavior.
    """
    if not pool_path.is_file():
        raise ValueError(f"RandBats pool is required but missing: {pool_path}")
    if not replay_dir.is_dir():
        raise ValueError(f"replay directory is missing: {replay_dir}")
    if workers <= 0:
        raise ValueError("workers must be positive")

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_by_gameid: dict[str, int] = {}
    for path in out_dir.glob("*.json.lz4"):
        gameid = path.name.split("_", 1)[0]
        outputs_by_gameid[gameid] = outputs_by_gameid.get(gameid, 0) + 1
    incomplete = {gameid: count for gameid, count in outputs_by_gameid.items() if count != 2}
    if incomplete:
        raise ValueError(f"incomplete or ambiguous parsed POVs: {incomplete}")

    raw_by_gameid: dict[str, Path] = {}
    for path in sorted(replay_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError:
            raw = {}
        gameid = str(raw.get("id") or path.stem)
        if gameid in raw_by_gameid:
            raise ValueError(f"duplicate replay identity {gameid}: {raw_by_gameid[gameid]}, {path}")
        raw_by_gameid[gameid] = path
    todo = [path for gameid, path in raw_by_gameid.items() if gameid not in outputs_by_gameid]
    ok = 0
    failed = 0
    if workers == 1:
        _init_worker(str(out_dir), str(pool_path), seed)
        results = (_parse_one(str(path)) for path in todo)
        for _, status in results:
            if status == "ok":
                ok += 1
            else:
                failed += 1
    elif todo:
        with mp.Pool(
            workers,
            initializer=_init_worker,
            initargs=(str(out_dir), str(pool_path), seed),
        ) as pool:
            for _, status in pool.imap_unordered(_parse_one, map(str, todo)):
                if status == "ok":
                    ok += 1
                else:
                    failed += 1
    return {
        "already_parsed": len(outputs_by_gameid),
        "replays_to_parse": len(todo),
        "parsed_ok": ok,
        "failed": failed,
        "total_pov_trajectories": len(list(out_dir.glob("*.json.lz4"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", default=str(ROOT / "data" / "replays" / "gen9randombattle"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "parsed_replays" / "gen9randombattle"))
    parser.add_argument("--pool-path", default=str(ROOT / "data" / "randbats_pools" / "gen9randombattle_pool_50000.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        summary = parse_replay_dir(
            Path(args.replay_dir),
            Path(args.out_dir),
            Path(args.pool_path),
            args.workers,
            args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
    summary["ts"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(summary), flush=True)
    Path(args.out_dir).parent.joinpath("parse_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
