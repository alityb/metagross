#!/usr/bin/env python3
"""Standalone rating poller for a live ladder run (append-only ratings.jsonl)."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--out", required=True)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    args = parser.parse_args()

    url = f"https://pokemonshowdown.com/users/{args.username.lower()}.json"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (metagross ladder monitor)"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode())
            rating = (data.get("ratings") or {}).get(args.format) or {}
            line = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "elo": rating.get("elo"),
                "gxe": rating.get("gxe"),
                "rpr": rating.get("rpr"),
                "rprd": rating.get("rprd"),
                "w": rating.get("w"),
                "l": rating.get("l"),
            }
            with out.open("a") as f:
                f.write(json.dumps(line) + "\n")
            print(f"RATING {json.dumps(line)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"WARN {e!r}", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
