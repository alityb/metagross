#!/usr/bin/env python3
"""Audit a P16 reliability block, then launch a fresh 600-game campaign."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from srcs.metagross.launch import FORMAT, showdown_user_id


ROOT = Path(__file__).resolve().parents[2]
FATAL_MARKERS = (
    "traceback (most recent call last)",
    "connectionclosed",
    "brokenprocesspool",
    "required prior fetch failed",
    "remote mcts failed",
    "invalid choice",
    "timed out",
    "using unguided",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON in {path.name}:{number}") from exc
    return rows


def audit_run(run_dir: Path, expected_games: int, engine_sha256: str) -> dict[str, int]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError(f"reliability manifest is {manifest.get('status')!r}")
    if manifest.get("ladder", {}).get("games") != expected_games:
        raise RuntimeError("reliability manifest game count mismatch")
    search_config = manifest.get("search") or {}
    if search_config.get("execution") != "modal" or search_config.get("parallelism") != 16:
        raise RuntimeError("reliability run did not use P16 Modal search")
    if (search_config.get("modal") or {}).get("engine_sha256") != engine_sha256:
        raise RuntimeError("reliability manifest engine hash mismatch")

    decisions = read_jsonl(run_dir / "decisions.jsonl")
    searches = read_jsonl(run_dir / "search.jsonl")
    protocol = read_jsonl(run_dir / "protocol.jsonl")
    read_jsonl(run_dir / "ratings.jsonl")
    if not decisions or len(decisions) != len(searches):
        raise RuntimeError("decision/search trajectory join mismatch")
    decision_keys = [(row.get("tag"), row.get("decision_idx")) for row in decisions]
    search_keys = [
        ((row.get("context") or {}).get("tag"), (row.get("context") or {}).get("decision_idx"))
        for row in searches
    ]
    if decision_keys != search_keys:
        raise RuntimeError("decision/search trajectory keys do not match")
    for row in searches:
        remote = row.get("remote_search") or {}
        if remote.get("worlds") not in {32, 64}:
            raise RuntimeError("search row has invalid remote world count")
        if (remote.get("engine") or {}).get("native_sha256") != engine_sha256:
            raise RuntimeError("search row engine hash mismatch")
        if len(remote.get("timings") or []) != remote.get("worlds"):
            raise RuntimeError("search row remote timing count mismatch")

    outcomes = sum(
        message.count("|win|") + message.count("|tie|")
        for row in protocol
        for message in [str(row.get("message") or "")]
    )
    if outcomes != expected_games:
        raise RuntimeError(f"expected {expected_games} outcomes, found {outcomes}")
    for name in ("client.log", "prior.log"):
        content = (run_dir / name).read_text(encoding="utf-8", errors="replace").lower()
        marker = next((item for item in FATAL_MARKERS if item in content), None)
        if marker:
            raise RuntimeError(f"{name} contains fatal marker {marker!r}")
    return {"decisions": len(decisions), "outcomes": outcomes}


def fetch_fresh_account(username: str) -> bool:
    request = urllib.request.Request(
        f"https://pokemonshowdown.com/users/{showdown_user_id(username)}.json?ts={time.time_ns()}",
        headers={"User-Agent": "metagross-p16-handoff"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    rating = (payload.get("ratings") or {}).get(FORMAT)
    if rating and int(rating.get("w") or 0) + int(rating.get("l") or 0) > 0:
        raise RuntimeError(f"full-campaign account {username} is not fresh")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--engine-sha256", required=True)
    parser.add_argument("--expected-games", type=int, default=25)
    parser.add_argument("--full-games", type=int, default=600)
    parser.add_argument("--registration-timeout-seconds", type=int, default=24 * 3600)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "srcs" / "runtime" / "local-r1-p16-full",
    )
    parser.add_argument("--state-path", type=Path, required=True)
    args = parser.parse_args()
    if args.wait_pid <= 0 or args.expected_games <= 0 or args.full_games <= 0:
        parser.error("PIDs and game counts must be positive")
    if len(args.engine_sha256) != 64:
        parser.error("engine SHA-256 must contain 64 characters")
    return args


def main() -> int:
    args = parse_args()
    if not os.environ.get("METAGROSS_SHOWDOWN_PASSWORD"):
        raise RuntimeError("METAGROSS_SHOWDOWN_PASSWORD is required")
    state = {
        "schema": 1,
        "status": "waiting_for_reliability_block",
        "created_at": utc_now(),
        "reliability_run": str(args.run_dir.resolve()),
        "full_username": args.username,
        "full_games": args.full_games,
        "engine_sha256": args.engine_sha256,
    }
    args.state_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.state_path, state)

    while True:
        try:
            os.kill(args.wait_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(30)

    try:
        audit = audit_run(args.run_dir.resolve(), args.expected_games, args.engine_sha256)
        state.update({"status": "waiting_for_fresh_account", "audit": audit})
        atomic_json(args.state_path, state)
        deadline = time.monotonic() + args.registration_timeout_seconds
        while not fetch_fresh_account(args.username):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"account {args.username} was not registered in time")
            time.sleep(60)

        command = [
            sys.executable,
            "-m",
            "srcs.metagross.launch",
            "--username",
            args.username,
            "--profile",
            "r1",
            "--games",
            str(args.full_games),
            "--search-parallelism",
            "16",
            "--search-threads",
            "1",
            "--remote-mcts",
            "--remote-engine-sha256",
            args.engine_sha256,
            "--output-root",
            str(args.output_root),
        ]
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        state.update({"status": "full_campaign_running", "started_at": utc_now(), "pid": child.pid})
        atomic_json(args.state_path, state)

        def stop_child(_signum, _frame):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, stop_child)
        return_code = child.wait()
        state.update(
            {
                "status": "completed" if return_code == 0 else "failed",
                "finished_at": utc_now(),
                "return_code": return_code,
            }
        )
        atomic_json(args.state_path, state)
        return return_code
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_json(args.state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
