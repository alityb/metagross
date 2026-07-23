#!/usr/bin/env python3
"""Continuously scrape gen9randombattle replays from the public Showdown replay API.

Polite, append-only, crash-tolerant:
- dedupes via file existence (data/replays/<format>/<id>.json)
- rate-limited (default 1 request / 1.5 s)
- pages backward via `before=<uploadtime>` until exhausted, then re-polls
- logs one line per action; FATAL lines on repeated failures, never hangs silently

Usage:
  python scripts/scrape_randbats_replays.py --format gen9randombattle \
      --out-dir data/replays --log-file experiments/scraper_gen9randombattle.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SEARCH_URL = "https://replay.pokemonshowdown.com/search.json?format={fmt}{before}"
REPLAY_URL = "https://replay.pokemonshowdown.com/{rid}.json"
USER_AGENT = "metagross-research-scraper/0.1 (polite; contact: local research project)"


def log(handle, msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def fetch_json(url: str, timeout: float = 30.0):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_with_retries(url: str, log_handle, attempts: int = 4, base_sleep: float = 5.0):
    for attempt in range(1, attempts + 1):
        try:
            return fetch_json(url)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            log(log_handle, f"WARN fetch attempt {attempt}/{attempts} failed url={url} err={exc!r}")
            time.sleep(base_sleep * attempt)
    log(log_handle, f"FATAL giving up on url={url} after {attempts} attempts; continuing with next work item")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--out-dir", default="data/replays")
    parser.add_argument("--log-file", default="experiments/scraper_gen9randombattle.log")
    parser.add_argument("--rate-limit-seconds", type=float, default=1.5)
    parser.add_argument("--repoll-seconds", type=float, default=300.0)
    parser.add_argument("--max-pages-per-sweep", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.format
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")

    log(log_handle, f"START scraper format={args.format} out={out_dir}")
    total_saved = 0
    while True:
        before = None
        pages = 0
        new_this_sweep = 0
        while pages < args.max_pages_per_sweep:
            pages += 1
            before_param = f"&before={before}" if before else ""
            url = SEARCH_URL.format(fmt=args.format, before=before_param)
            listing = fetch_with_retries(url, log_handle)
            time.sleep(args.rate_limit_seconds)
            if not listing:
                break
            # API may return a bare list or {"replays": [...]}
            replays = listing.get("replays", listing) if isinstance(listing, dict) else listing
            if not replays:
                log(log_handle, "INFO listing empty; sweep done")
                break
            oldest = None
            for item in replays:
                rid = item.get("id")
                uploadtime = item.get("uploadtime")
                if uploadtime is not None:
                    oldest = uploadtime if oldest is None else min(oldest, uploadtime)
                if not rid:
                    continue
                target = out_dir / f"{rid}.json"
                if target.exists():
                    continue
                data = fetch_with_retries(REPLAY_URL.format(rid=rid), log_handle)
                time.sleep(args.rate_limit_seconds)
                if data is None:
                    continue
                tmp = target.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(data), encoding="utf-8")
                tmp.rename(target)
                total_saved += 1
                new_this_sweep += 1
                if total_saved % 25 == 0:
                    log(log_handle, f"INFO saved={total_saved} (sweep new={new_this_sweep})")
            # Page backward; a page with zero *new* ids AND deep pagination means we've caught up
            if oldest is None or (before is not None and oldest >= before):
                log(log_handle, "INFO pagination not advancing; sweep done")
                break
            before = oldest
        log(log_handle, f"SWEEP_DONE pages={pages} new={new_this_sweep} total_saved={total_saved}; sleeping {args.repoll_seconds}s")
        time.sleep(args.repoll_seconds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
