#!/usr/bin/env python3
"""G1 gate: Kakuna zero-shot vs stock Foul Play on gen9randombattle, role-balanced.

Runs two phases (roles swapped halfway) so challenger/acceptor asymmetry cancels:
  Phase A: Kakuna acceptor  <- FP challenger (challenge_user)
  Phase B: Kakuna challenger -> FP acceptor  (accept_challenge)

Robustness: per-phase timeout, stall detection on the results CSV, FATAL logging,
partial scoring from whatever completed. Scores from Kakuna's results CSV
(WIN/LOSS per battle), reports Wilson CI, writes result.json + experiment CSV row.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_WS = "ws://localhost:8000/showdown/websocket"


def log(handle, msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def wilson_ci(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = wins / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def read_results_csv(path: Path) -> list[str]:
    """Return list of results ('WIN'/'LOSS'/...) from a metamon battle log CSV."""
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 4:
                rows.append(row[3].strip().upper())
    return rows


def terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def run_phase(
    args: argparse.Namespace,
    log_handle,
    phase_name: str,
    kakuna_role: str,
    n_games: int,
    out_dir: Path,
) -> list[str]:
    """Run one phase; return per-battle results from Kakuna's perspective."""
    phase_dir = out_dir / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    kak_user = f"kakg1{phase_name[-1].lower()}"
    fp_user = f"fpg1{phase_name[-1].lower()}"
    results_csv = phase_dir / f"battle_log_{kak_user}_gen9randombattle.csv"

    kak_env = os.environ.copy()
    kak_env.update(
        METAMON_CACHE_DIR=str(ROOT / "external" / "metamon_cache"),
        WANDB_MODE="disabled",
        TORCHDYNAMO_DISABLE="1",
        ACCELERATE_USE_CPU="true",
        PYTHONUNBUFFERED="1",
        # cap torch CPU threads so concurrent experiments (G0) aren't starved
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
    )
    kak_log_path = phase_dir / "kakuna.out"
    kak_cmd = [
        str(ROOT / ".venv-metamon" / "bin" / "python"),
        str(ROOT / "scripts" / "run_kakuna_challenge.py"),
        "--agent", args.agent,
        "--username", kak_user,
        "--opponent-username", fp_user,
        "--role", kakuna_role,
        "--battle-format", args.format,
        "--total-battles", str(n_games),
        "--save-results-to", str(phase_dir),
    ]
    fp_mode = "challenge_user" if kakuna_role == "acceptor" else "accept_challenge"
    fp_cmd = [
        str(ROOT / ".venv-foul-play" / "bin" / "python"),
        str(ROOT / "scripts" / "run_foul_play.py"),
        "--websocket-uri", args.websocket_uri,
        "--ps-username", fp_user,
        "--bot-mode", fp_mode,
        "--pokemon-format", args.format,
        "--run-count", str(n_games),
        "--search-time-ms", str(args.foul_play_search_time_ms),
        "--search-parallelism", "1",
        "--search-threads", "1",
        "--log-level", "INFO",
    ]
    if fp_mode == "challenge_user":
        fp_cmd.extend(["--user-to-challenge", kak_user])
    fp_log_path = phase_dir / "foul_play.out"

    def start_kakuna() -> subprocess.Popen:
        kak_log = kak_log_path.open("w", encoding="utf-8")
        log(log_handle, f"{phase_name} starting Kakuna ({kakuna_role}): {' '.join(kak_cmd)}")
        return subprocess.Popen(kak_cmd, cwd=str(ROOT), env=kak_env,
                                stdout=kak_log, stderr=subprocess.STDOUT)

    def start_fp() -> subprocess.Popen:
        fp_log = fp_log_path.open("w", encoding="utf-8")
        log(log_handle, f"{phase_name} starting Foul Play ({fp_mode}): {' '.join(fp_cmd)}")
        return subprocess.Popen(fp_cmd, cwd=str(ROOT), env=os.environ.copy(),
                                stdout=fp_log, stderr=subprocess.STDOUT)

    def wait_for_line(path: Path, needle: str, proc: subprocess.Popen,
                      timeout_minutes: float, what: str) -> bool:
        deadline = time.monotonic() + timeout_minutes * 60
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log(log_handle, f"FATAL {phase_name}: {what} exited while waiting "
                                f"for '{needle}'; see {path}")
                return False
            if path.exists() and needle in path.read_text(errors="ignore"):
                return True
            time.sleep(10)
        log(log_handle, f"FATAL {phase_name}: '{needle}' not seen in {what} log within "
                        f"{timeout_minutes} min")
        return False

    # The ACCEPTOR side must be online before the challenger fires: the
    # challenge is not retried if the target is missing (verified in smoke).
    if kakuna_role == "acceptor":
        kak_proc = start_kakuna()
        if not wait_for_line(kak_log_path, "Checkpoint validated", kak_proc,
                             args.load_timeout_minutes, "Kakuna"):
            terminate(kak_proc)
            return []
        log(log_handle, f"{phase_name} Kakuna checkpoint validated (acceptor up); starting FP challenger")
        time.sleep(10)
        fp_proc = start_fp()
    else:
        fp_proc = start_fp()
        if not wait_for_line(fp_log_path, "Waiting for a", fp_proc, 5.0, "Foul Play"):
            terminate(fp_proc)
            return []
        log(log_handle, f"{phase_name} FP acceptor waiting; starting Kakuna challenger")
        time.sleep(5)
        kak_proc = start_kakuna()

    deadline = time.monotonic() + args.phase_timeout_minutes * 60
    last_count = 0
    last_progress_time = time.monotonic()
    while time.monotonic() < deadline:
        kak_done = kak_proc.poll() is not None
        fp_done = fp_proc.poll() is not None
        results = read_results_csv(results_csv)
        if len(results) > last_count:
            wins = sum(1 for r in results if r == "WIN")
            log(log_handle, f"{phase_name} PROGRESS battles={len(results)} "
                            f"kakuna_wins={wins} last={results[-1]}")
            last_count = len(results)
            last_progress_time = time.monotonic()
        if len(results) >= n_games:
            log(log_handle, f"{phase_name} reached {n_games} battles")
            break
        if kak_done and fp_done:
            log(log_handle, f"{phase_name} both processes exited "
                            f"(kak={kak_proc.returncode} fp={fp_proc.returncode})")
            break
        if kak_done and not fp_done:
            log(log_handle, f"{phase_name} Kakuna exited (rc={kak_proc.returncode}); "
                            f"granting FP 60s grace")
            time.sleep(60)
            break
        if time.monotonic() - last_progress_time > args.stall_minutes * 60:
            log(log_handle, f"FATAL {phase_name}: no new battle for {args.stall_minutes} "
                            f"min at {last_count} battles; killing phase, keeping partial")
            break
        time.sleep(15)
    else:
        log(log_handle, f"FATAL {phase_name}: phase timeout "
                        f"({args.phase_timeout_minutes} min); keeping partial")

    terminate(fp_proc)
    terminate(kak_proc)
    return read_results_csv(results_csv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="Kakuna")
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--foul-play-search-time-ms", type=int, default=100)
    parser.add_argument("--websocket-uri", default=LOCAL_WS)
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "g1_kakuna_vs_fp_n100"))
    parser.add_argument("--load-timeout-minutes", type=float, default=15.0)
    parser.add_argument("--phase-timeout-minutes", type=float, default=420.0)
    parser.add_argument("--stall-minutes", type=float, default=30.0)
    parser.add_argument("--run-id", default="g1_kakuna_zeroshot_vs_stock_fp_n100")
    parser.add_argument("--append-experiment-log",
                        default=str(ROOT / "experiments" / "phase1_eval_gate.csv"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (out_dir / "orchestrator.log").open("a", encoding="utf-8")
    log(log_handle, f"G1 START n_games={args.n_games} fp_ms={args.foul_play_search_time_ms}")

    half = args.n_games // 2
    phase_a = run_phase(args, log_handle, "phaseA", "acceptor", half, out_dir)
    phase_b = run_phase(args, log_handle, "phaseB", "challenger", args.n_games - half, out_dir)

    def score(results: list[str]) -> tuple[int, int]:
        return (sum(1 for r in results if r == "WIN"),
                sum(1 for r in results if r == "LOSS"))

    a_w, a_l = score(phase_a)
    b_w, b_l = score(phase_b)
    wins, losses = a_w + b_w, a_l + b_l
    decisive = wins + losses
    winrate = wins / decisive if decisive else 0.0
    ci_low, ci_high = wilson_ci(wins, decisive)
    other = (len(phase_a) + len(phase_b)) - decisive

    payload = {
        "run_id": args.run_id,
        "agent": args.agent,
        "format": args.format,
        "foul_play_search_time_ms": args.foul_play_search_time_ms,
        "n_games_requested": args.n_games,
        "completed": len(phase_a) + len(phase_b),
        "kakuna_wins": wins,
        "kakuna_losses": losses,
        "ties_or_unknown": other,
        "winrate_kakuna": round(winrate, 4),
        "ci95": [round(ci_low, 4), round(ci_high, 4)],
        "phaseA_kakuna_acceptor": {"wins": a_w, "losses": a_l, "n": len(phase_a)},
        "phaseB_kakuna_challenger": {"wins": b_w, "losses": b_l, "n": len(phase_b)},
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    log(log_handle, "G1 RESULT " + json.dumps(payload, sort_keys=True))

    if args.append_experiment_log:
        row = {
            "run_id": args.run_id,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "phase": "g1",
            "format": args.format,
            "change (ONE var)": "kakuna_zeroshot_vs_stock_foul_play",
            "baseline": "stock_foul_play",
            "N_games": decisive,
            "winrate": f"{winrate:.4f}",
            "CI95": f"[{ci_low:.4f}, {ci_high:.4f}]",
            "ladder_elo": "",
            "gxe": "",
            "belief_brier": "",
            "decision(advance/iterate/rollback)": "record",
            "notes": (
                f"role-balanced local H2H; Kakuna zero-shot gen9randombattle via gen9ou format alias; "
                f"foul_play_search_time_ms={args.foul_play_search_time_ms}; "
                f"phaseA_acceptor={a_w}/{len(phase_a)}; phaseB_challenger={b_w}/{len(phase_b)}; "
                f"ties_or_unknown={other}; checkpoint=kakuna default 34; human decision gate"
            ),
        }
        path = Path(args.append_experiment_log)
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    log(log_handle, "G1 DONE")


if __name__ == "__main__":
    sys.exit(main())
