from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SEARCH_URL = "https://replay.pokemonshowdown.com/search.json?format={format_id}&page={page}"


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "AlphaPokemonReplayScraper/0.1"})
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
    assert last_error is not None
    raise last_error


def load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            seen.add(str(json.loads(line)["id"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return seen


def scrape(output: Path, target: int, min_rating: int, start_page: int = 1, format_id: str = "gen9randombattle") -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(output)
    written = len(seen)
    page = start_page
    with output.open("a", encoding="utf-8") as handle:
        while written < target:
            data = fetch_json(SEARCH_URL.format(format_id=format_id, page=page))
            time.sleep(0.5)  # Required rate limit. Do not remove.
            if not data:
                break
            added = 0
            for replay in data:
                replay_id = str(replay.get("id") or "")
                rating = replay.get("rating")
                if not replay_id or replay_id in seen or rating is None:
                    continue
                try:
                    rating_value = int(rating)
                except (TypeError, ValueError):
                    continue
                if rating_value < min_rating:
                    continue
                record = {"id": replay_id, "rating": rating_value, "uploadtime": replay.get("uploadtime")}
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                seen.add(replay_id)
                written += 1
                added += 1
                if written >= target:
                    break
            print(json.dumps({"format": format_id, "page": page, "added": added, "total": written}))
            page += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape high-rated Gen9 random battle replay IDs")
    parser.add_argument("--output", default="data/replay_ids.jsonl")
    parser.add_argument("--target", type=int, default=300_000)
    parser.add_argument("--min-rating", type=int, default=1500)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--format", default="gen9randombattle")
    args = parser.parse_args()
    total = scrape(Path(args.output), args.target, args.min_rating, args.start_page, args.format)
    print(json.dumps({"total": total, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
