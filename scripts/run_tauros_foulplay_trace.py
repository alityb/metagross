#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_SHOWDOWN_URI = "ws://localhost:8000/showdown/websocket"


def normalize_user_id(username: str) -> str:
    return re.sub(r"[^a-z0-9]", "", username.lower())


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def start_showdown(port: int) -> subprocess.Popen | None:
    sock_check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', %d)); s.close()" % port,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if sock_check.returncode == 0:
        return None

    log_path = ROOT / "external" / "showdown-trace-runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [str(ROOT / "scripts" / "start_showdown.sh"), str(port)],
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    for _ in range(30):
        time.sleep(1)
        sock_check = subprocess.run(
            [
                sys.executable,
                "-c",
                "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', %d)); s.close()" % port,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if sock_check.returncode == 0:
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"Showdown exited early; see {log_path}")
    raise RuntimeError(f"Showdown did not start on port {port}; see {log_path}")


def terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def build_metamon_cmd(args: argparse.Namespace, username: str, opponent: str, out_dir: Path) -> list[str]:
    return [
        str(Path(args.metamon_python)),
        "-m",
        "metamon.rl.evaluate",
        "--eval_type",
        "challenge",
        "--agent",
        args.teacher_agent,
        "--username",
        username,
        "--opponent_username",
        opponent,
        "--role",
        "acceptor",
        "--gens",
        "1",
        "--formats",
        "ou",
        "--total_battles",
        str(args.n_games),
        "--team_set",
        "competitive",
        "--save_results_to",
        str(out_dir / "metamon_results"),
        "--save_trajectories_to",
        str(out_dir / "metamon_trajectories"),
    ]


def build_foul_play_cmd(args: argparse.Namespace, username: str, opponent: str) -> list[str]:
    return [
        str(Path(args.foul_play_python)),
        str(ROOT / "scripts" / "run_foul_play.py"),
        "--websocket-uri",
        args.websocket_uri,
        "--ps-username",
        username,
        "--bot-mode",
        "challenge_user",
        "--user-to-challenge",
        opponent,
        "--pokemon-format",
        "gen1ou",
        "--run-count",
        str(args.n_games),
        "--search-time-ms",
        str(args.foul_play_search_time_ms),
        "--search-parallelism",
        str(args.foul_play_search_parallelism),
        "--search-threads",
        str(args.foul_play_search_threads),
        "--log-level",
        args.foul_play_log_level,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Tauros/Kakuna vs Foul Play Gen1OU traces")
    parser.add_argument("--n-games", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "traces" / "tauros_vs_foulplay"))
    parser.add_argument("--teacher-agent", choices=["TaurosV0", "Kakuna"], default="TaurosV0")
    parser.add_argument("--metamon-python", default=str(ROOT / ".venv-metamon" / "bin" / "python"))
    parser.add_argument("--foul-play-python", default=str(ROOT / ".venv-exp3" / "bin" / "python"))
    parser.add_argument("--websocket-uri", default=LOCAL_SHOWDOWN_URI)
    parser.add_argument("--showdown-port", type=int, default=8000)
    parser.add_argument("--foul-play-search-time-ms", type=int, default=100)
    parser.add_argument("--foul-play-search-parallelism", type=int, default=1)
    parser.add_argument("--foul-play-search-threads", type=int, default=1)
    parser.add_argument("--foul-play-log-level", default="INFO")
    parser.add_argument("--startup-delay-seconds", type=float, default=20.0)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--no-start-showdown", action="store_true")
    args = parser.parse_args()

    if args.n_games <= 0:
        raise ValueError("--n-games must be positive")

    run_id = args.run_id or f"{args.teacher_agent.lower()}_fp_trace_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    teacher_username = normalize_user_id(f"{args.teacher_agent.lower()}trace")[:18]
    fp_username = "fptrace"
    decision_log = out_dir / "foul_play_decisions.jsonl"

    showdown_proc = None if args.no_start_showdown else start_showdown(args.showdown_port)
    metamon_proc = None
    foul_proc = None
    try:
        metamon_env = os.environ.copy()
        metamon_env["METAMON_CACHE_DIR"] = str(ROOT / "external" / "metamon_cache")
        metamon_env.setdefault("PYTHONUNBUFFERED", "1")
        metamon_log = (logs_dir / "metamon_acceptor.log").open("w", encoding="utf-8")
        metamon_proc = subprocess.Popen(
            build_metamon_cmd(args, teacher_username, fp_username, out_dir),
            cwd=str(ROOT),
            env=metamon_env,
            stdout=metamon_log,
            stderr=subprocess.STDOUT,
        )

        time.sleep(args.startup_delay_seconds)
        if metamon_proc.poll() is not None:
            raise RuntimeError(f"Metamon exited before Foul Play started; see {logs_dir / 'metamon_acceptor.log'}")

        foul_env = os.environ.copy()
        foul_env["METAGROSS_DECISION_LOG"] = str(decision_log)
        foul_log = (logs_dir / "foul_play_challenger.log").open("w", encoding="utf-8")
        foul_proc = subprocess.Popen(
            build_foul_play_cmd(args, fp_username, teacher_username),
            cwd=str(ROOT),
            env=foul_env,
            stdout=foul_log,
            stderr=subprocess.STDOUT,
        )

        deadline = time.monotonic() + args.timeout_seconds
        while time.monotonic() < deadline:
            metamon_done = metamon_proc.poll() is not None
            foul_done = foul_proc.poll() is not None
            if metamon_done and foul_done:
                break
            if metamon_done and not foul_done:
                terminate(foul_proc)
                break
            if foul_done and not metamon_done:
                terminate(metamon_proc)
                break
            time.sleep(2)
        else:
            terminate(foul_proc)
            terminate(metamon_proc)
            raise TimeoutError(f"trace run timed out after {args.timeout_seconds}s")
    finally:
        terminate(foul_proc)
        terminate(metamon_proc)
        terminate(showdown_proc)

    trajectory_files = list((out_dir / "metamon_trajectories").glob("**/*.json.lz4"))
    result_files = list((out_dir / "metamon_results").glob("*.csv"))
    summary = {
        "run_id": run_id,
        "teacher_agent": args.teacher_agent,
        "format": "gen1ou",
        "n_games_requested": args.n_games,
        "foul_play_decision_rows": count_lines(decision_log),
        "metamon_trajectory_files": len(trajectory_files),
        "metamon_result_files": [str(path.relative_to(out_dir)) for path in result_files],
        "metamon_returncode": None if metamon_proc is None else metamon_proc.returncode,
        "foul_play_returncode": None if foul_proc is None else foul_proc.returncode,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
