from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


REPLAY_URL = "https://replay.pokemonshowdown.com/{replay_id}.json"


def iter_ids(path: Path, limit: int | None = None) -> list[str]:
    ids: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ids.append(str(json.loads(line)["id"]))
        except (json.JSONDecodeError, KeyError):
            continue
        if limit is not None and len(ids) >= limit:
            break
    return ids


def download_one(replay_id: str, output_dir: Path) -> tuple[str, bool, str | None]:
    path = output_dir / f"{replay_id}.json"
    if path.exists():
        return replay_id, False, None
    try:
        request = urllib.request.Request(REPLAY_URL.format(replay_id=replay_id), headers={"User-Agent": "AlphaPokemonReplayDownloader/0.1"})
        payload = None
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = response.read().decode("utf-8")
                break
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                time.sleep(2.0 * (attempt + 1))
        if payload is None:
            raise last_error or RuntimeError("download failed")
        path.write_text(payload, encoding="utf-8")
        return replay_id, True, None
    except Exception as exc:  # noqa: BLE001 - downloader logs and continues.
        return replay_id, False, str(exc)
    finally:
        time.sleep(0.5)  # Required per-worker rate limit. Do not remove.


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw PS replay JSON files")
    parser.add_argument("--ids", default="data/replay_ids.jsonl")
    parser.add_argument("--output", default="data/raw_replays")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_ids = iter_ids(Path(args.ids), args.limit)
    completed = 0
    downloaded = 0
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, replay_id, output_dir) for replay_id in replay_ids]
        for future in concurrent.futures.as_completed(futures):
            _replay_id, did_download, error = future.result()
            completed += 1
            downloaded += int(did_download)
            errors += int(error is not None)
            if completed % 1000 == 0 or completed == len(futures):
                print(json.dumps({"completed": completed, "downloaded": downloaded, "errors": errors}))


if __name__ == "__main__":
    main()
