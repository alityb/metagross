#!/usr/bin/env python3
"""Run bounded public-ladder policy blocks until stopped."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from srcs.metagross.launch import FORMAT, showdown_user_id


ROOT = Path(__file__).resolve().parents[2]
FATAL_LOG_PATTERNS = (
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bERROR\b",
        r"Traceback \(most recent call last\)",
        r"required prior fetch failed",
        r"invalid choice",
        r"login failed",
        r"timed out",
        r"unhandled exception",
    )
)
FATAL_LOG_PATTERNS = tuple(FATAL_LOG_PATTERNS)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-username")
    parser.add_argument("--g3-username")
    parser.add_argument("--g4-username")
    parser.add_argument(
        "--g3-only",
        action="store_true",
        help="run only sequential G3 blocks",
    )
    parser.add_argument("--block-games", type=int, default=25)
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="number of complete profile sequences; 0 runs until signaled",
    )
    parser.add_argument("--pause-seconds", type=float, default=30.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "srcs" / "runtime" / "ladder-supervisor",
    )
    parser.add_argument("--rating-poll-seconds", type=float, default=90.0)
    parser.add_argument("--search-parallelism", type=int, default=8)
    parser.add_argument("--port", type=int, default=8977)
    args = parser.parse_args(argv)
    if not 1 <= args.block_games <= 100:
        parser.error("--block-games must be between 1 and 100")
    if args.cycles < 0:
        parser.error("--cycles must be non-negative")
    if args.pause_seconds < 0:
        parser.error("--pause-seconds must be non-negative")
    if args.rating_poll_seconds <= 0:
        parser.error("--rating-poll-seconds must be positive")
    if args.search_parallelism <= 0:
        parser.error("--search-parallelism must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.r1_username:
        if args.g3_username or args.g4_username or args.g3_only:
            parser.error("--r1-username cannot be combined with G3/G4 options")
        if not showdown_user_id(args.r1_username):
            parser.error("usernames must contain a letter or number")
        return args
    if not args.g3_username:
        parser.error("--g3-username is required unless --r1-username is set")
    if not showdown_user_id(args.g3_username):
        parser.error("usernames must contain a letter or number")
    if not args.g3_only and not args.g4_username:
        parser.error("--g4-username is required unless --g3-only is set")
    if args.g4_username and not showdown_user_id(args.g4_username):
        parser.error("usernames must contain a letter or number")
    if args.g4_username and showdown_user_id(args.g3_username) == showdown_user_id(args.g4_username):
        parser.error("G3 and G4 must use different accounts")
    return args


def configured_profiles(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.r1_username:
        return [("r1", args.r1_username)]
    profiles = [("g3", args.g3_username)]
    if not args.g3_only:
        profiles.append(("g4", args.g4_username))
    return profiles


def make_run_dir(output_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_root.expanduser().resolve() / f"{stamp}-{os.getpid()}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def acquire_supervisor_lock(suffix: str = ""):
    # Lock is scoped by `suffix` (the prior-server port) so a deliberate
    # multi-account pair can run one supervisor per account concurrently,
    # while still refusing a duplicate supervisor on the SAME port.
    lock_dir = ROOT / "srcs" / "runtime" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    handle = (lock_dir / f"public-ladder-supervisor{suffix}.lock").open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"a public-ladder supervisor is already running (lock{suffix})"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def launcher_command(
    profile: str,
    username: str,
    games: int,
    output_root: Path,
    rating_poll_seconds: float,
    search_parallelism: int = 8,
    port: int = 8977,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "srcs.metagross.launch",
        "--username",
        username,
        "--profile",
        profile,
        "--games",
        str(games),
        "--output-root",
        str(output_root),
        "--rating-poll-seconds",
        str(rating_poll_seconds),
        "--search-parallelism",
        str(search_parallelism),
        "--port",
        str(port),
    ]
    if profile in {"g3", "g4"}:
        command.append("--confirm-candidate-continuation")
    return command


def find_fatal_log_line(run_dir: Path) -> str | None:
    for name in ("prior.log", "client.log"):
        path = run_dir / name
        with path.open(encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, start=1):
                if any(pattern.search(line) for pattern in FATAL_LOG_PATTERNS):
                    return f"{name}:{number}: {line.strip()}"
    return None


def fetch_rating(username: str) -> dict:
    url = f"https://pokemonshowdown.com/users/{showdown_user_id(username)}.json"
    request = urllib.request.Request(url, headers={"User-Agent": "metagross-ladder-supervisor"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read())
    rating = (payload.get("ratings") or {}).get(FORMAT) or {}
    return {key: rating.get(key) for key in ("elo", "gxe", "rpr", "rprd", "w", "l")}


def completed_run_dir(block_root: Path, before: set[Path]) -> Path:
    after = {path for path in block_root.iterdir() if path.is_dir()}
    created = after - before
    if len(created) != 1:
        raise RuntimeError(f"expected one launcher run directory, found {len(created)}")
    return created.pop()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.environ.get("METAGROSS_SHOWDOWN_PASSWORD"):
        raise RuntimeError("set METAGROSS_SHOWDOWN_PASSWORD before launching")

    lock = acquire_supervisor_lock(f"-p{args.port}")
    run_dir = make_run_dir(args.output_root)
    block_root = run_dir / "blocks"
    block_root.mkdir()
    history_path = run_dir / "blocks.jsonl"
    state_path = run_dir / "state.json"
    profiles = configured_profiles(args)
    state = {
        "schema": 1,
        "status": "running",
        "started_at": utc_now(),
        "supervisor_pid": os.getpid(),
        "configuration": {
            "block_games": args.block_games,
            "cycles": args.cycles,
            "pause_seconds": args.pause_seconds,
            "profiles": [
                {"profile": profile, "username": username}
                for profile, username in profiles
            ],
        },
        "totals": {
            profile: {"blocks": 0, "requested_games": 0}
            for profile, _username in profiles
        },
        "latest": None,
    }
    atomic_json(state_path, state)

    stopping = False
    child: subprocess.Popen | None = None

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    completed_cycles = 0
    try:
        while not stopping and (args.cycles == 0 or completed_cycles < args.cycles):
            for profile, username in profiles:
                if stopping:
                    break
                before = {path for path in block_root.iterdir() if path.is_dir()}
                command = launcher_command(
                    profile,
                    username,
                    args.block_games,
                    block_root,
                    args.rating_poll_seconds,
                    args.search_parallelism,
                    args.port,
                )
                child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), start_new_session=True)
                return_code = child.wait()
                child = None
                if stopping:
                    break
                child_run = completed_run_dir(block_root, before)
                manifest = json.loads((child_run / "manifest.json").read_text(encoding="utf-8"))
                if return_code != 0 or manifest.get("status") != "completed":
                    raise RuntimeError(
                        f"{profile} block failed: return_code={return_code}, "
                        f"status={manifest.get('status')}"
                    )
                fatal_line = find_fatal_log_line(child_run)
                if fatal_line:
                    raise RuntimeError(f"{profile} block log gate failed: {fatal_line}")
                try:
                    rating = fetch_rating(username)
                except Exception as exc:
                    rating = {"error": f"{type(exc).__name__}: {exc}"}
                record = {
                    "completed_at": utc_now(),
                    "profile": profile,
                    "username": username,
                    "requested_games": args.block_games,
                    "run_dir": str(child_run),
                    "checkpoint_sha256": manifest["checkpoint"]["sha256_verified"],
                    "rating": rating,
                }
                with history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                state["totals"][profile]["blocks"] += 1
                state["totals"][profile]["requested_games"] += args.block_games
                state["latest"] = record
                atomic_json(state_path, state)
                if args.pause_seconds and not stopping:
                    deadline = time.monotonic() + args.pause_seconds
                    while not stopping and time.monotonic() < deadline:
                        time.sleep(min(1.0, deadline - time.monotonic()))
            if not stopping:
                completed_cycles += 1
        state["status"] = "stopped" if stopping else "completed"
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        return 1
    finally:
        state["finished_at"] = utc_now()
        atomic_json(state_path, state)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
