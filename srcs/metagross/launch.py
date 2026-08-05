#!/usr/bin/env python3
"""Launch a verified production policy profile and public-ladder client."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "gen9randombattle"
SEARCH_TIME_MS = 500
DEFAULT_SEARCH_PARALLELISM = 8
DEFAULT_SEARCH_THREADS = 1
CPU_C_PUCT = 2.0
DEFAULT_REMOTE_MCTS_APP = "metagross-mcts-r1-p16"
DEFAULT_REMOTE_MCTS_FUNCTION = "search_batch"


@dataclass(frozen=True)
class PolicyProfile:
    run_name: str
    checkpoint: int
    sha256: str

    def checkpoint_path(self, checkpoint_root: Path) -> Path:
        return (
            checkpoint_root
            / self.run_name
            / "ckpts"
            / "policy_weights"
            / f"policy_epoch_{self.checkpoint}.pt"
        )


POLICY_PROFILES = MappingProxyType({
    "r1": PolicyProfile(
        run_name="randbats_exit_r1",
        checkpoint=5,
        sha256="c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93",
    ),
    "g4": PolicyProfile(
        run_name="randbats_online_g4_autonomous_freshfix_20260729",
        checkpoint=1,
        sha256="cd21dbb22bfc7a92836c7c75c6287ffc1b79c58e0a2dca3d402b76b001ff7505",
    ),
    "g3": PolicyProfile(
        run_name="randbats_online_g3_autonomous_freshfix_20260729",
        checkpoint=1,
        sha256="0c754bb96953b900e282de91c570aaae5c2c6f002dc2419e149d01132888815c",
    ),
})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def showdown_user_id(username: str) -> str:
    return re.sub(r"[^a-z0-9]", "", username.lower())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(profile: PolicyProfile, checkpoint_root: Path) -> tuple[Path, str]:
    path = profile.checkpoint_path(checkpoint_root).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"policy checkpoint not found: {path}")
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, profile.sha256):
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch for {profile.run_name}@{profile.checkpoint}: "
            f"expected {profile.sha256}, got {actual}"
        )
    return path, actual


def wait_for_health(
    url: str,
    process: subprocess.Popen,
    timeout: int = 240,
    stop: threading.Event | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stop is not None and stop.is_set():
            raise InterruptedError("shutdown requested while waiting for prior server")
        if process.poll() is not None:
            raise RuntimeError(f"prior server exited with code {process.returncode}")
        if prior_is_healthy(url):
            return
        time.sleep(2)
    raise TimeoutError(f"prior server did not become healthy within {timeout}s")


def prior_is_healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=5) as response:
            return json.loads(response.read()).get("ok") is True
    except Exception:
        return False


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def acquire_account_lock(username: str):
    lock_dir = ROOT / "srcs" / "runtime" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"public-ladder-{showdown_user_id(username)}.lock"
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"public-ladder account is already locked: {username}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--profile", choices=sorted(POLICY_PROFILES), default="r1")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument(
        "--confirm-g4-canary",
        action="store_true",
        help="required acknowledgement for the immutable three-game G4 canary",
    )
    parser.add_argument(
        "--confirm-g3-canary",
        action="store_true",
        help="required acknowledgement for the immutable three-game G3 canary",
    )
    parser.add_argument(
        "--confirm-candidate-continuation",
        action="store_true",
        help="allow a verified G3/G4 continuation block of at most 100 games",
    )
    parser.add_argument("--port", type=int, default=8977)
    parser.add_argument(
        "--websocket-uri",
        default="wss://sim3.psim.us/showdown/websocket",
    )
    parser.add_argument(
        "--metamon-python",
        type=Path,
        default=ROOT / ".venv-metamon" / "bin" / "python",
    )
    parser.add_argument(
        "--foul-play-python",
        type=Path,
        default=ROOT / ".venv-fp-priors" / "bin" / "python",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "srcs" / "models",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "srcs" / "runtime" / "ladder-runs",
    )
    parser.add_argument("--health-timeout-seconds", type=int, default=240)
    parser.add_argument("--health-poll-seconds", type=float, default=5.0)
    parser.add_argument("--rating-poll-seconds", type=float, default=90.0)
    parser.add_argument(
        "--search-parallelism", type=int, default=DEFAULT_SEARCH_PARALLELISM
    )
    parser.add_argument("--search-threads", type=int, default=DEFAULT_SEARCH_THREADS)
    parser.add_argument("--remote-mcts", action="store_true")
    parser.add_argument("--remote-mcts-app", default=DEFAULT_REMOTE_MCTS_APP)
    parser.add_argument("--remote-mcts-function", default=DEFAULT_REMOTE_MCTS_FUNCTION)
    parser.add_argument("--remote-engine-sha256")
    parser.add_argument("--stall-timeout-seconds", type=int, default=1200)
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=None,
        help="default is max(3600, games * 900); always bounded",
    )
    args = parser.parse_args(argv)
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.profile in {"g3", "g4"}:
        canary_confirmed = args.games == 3 and getattr(args, f"confirm_{args.profile}_canary")
        continuation_confirmed = args.confirm_candidate_continuation and args.games <= 100
        if not (canary_confirmed or continuation_confirmed):
            parser.error(
                f"{args.profile.upper()} requires its three-game canary acknowledgement or "
                "--confirm-candidate-continuation with at most 100 games"
            )
    for name in (
        "health_timeout_seconds",
        "health_poll_seconds",
        "rating_poll_seconds",
        "stall_timeout_seconds",
        "search_parallelism",
        "search_threads",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_runtime_seconds is None:
        args.max_runtime_seconds = max(3600, args.games * 900)
    elif args.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be positive")
    if args.remote_mcts and not args.remote_engine_sha256:
        parser.error("--remote-mcts requires --remote-engine-sha256")
    if not showdown_user_id(args.username):
        parser.error("--username must contain a letter or number")
    return args


def make_run_dir(output_root: Path, profile_name: str, username: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = output_root.expanduser().resolve() / (
        f"{stamp}-{profile_name}-{showdown_user_id(username)}-{os.getpid()}"
    )
    base.mkdir(parents=True, exist_ok=False)
    return base


def poll_rating(username: str, output: Path, stop: threading.Event, interval: float) -> None:
    url = f"https://pokemonshowdown.com/users/{showdown_user_id(username)}.json"
    while not stop.is_set():
        row = {"timestamp": utc_now()}
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "metagross-production-rating-monitor"}
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read())
            rating = (data.get("ratings") or {}).get(FORMAT) or {}
            row.update({key: rating.get(key) for key in ("elo", "gxe", "rpr", "rprd", "w", "l")})
        except Exception as exc:  # Rating telemetry must not control gameplay.
            row["error"] = f"{type(exc).__name__}: {exc}"
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        stop.wait(interval)


def inspect_foul_play_engine(python: Path) -> dict:
    command = [
        str(python),
        "-c",
        (
            "import json; "
            "from srcs.metagross.run_foul_play import inspect_poke_engine; "
            "print(json.dumps(inspect_poke_engine(), sort_keys=True))"
        ),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"poke-engine provenance check failed: {detail}")
    return json.loads(result.stdout)


def production_environment(source: dict[str, str]) -> dict[str, str]:
    environment = source.copy()
    environment.pop("METAGROSS_VALUE_MODEL", None)
    environment.pop("METAGROSS_LEARNED_VALUE_WEIGHT", None)
    return environment


def terminate_processes(processes: list[subprocess.Popen | None]) -> None:
    live = [process for process in processes if process is not None and process.poll() is None]
    for process in live:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for process in live:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in live:
        if process.poll() is None:
            process.wait()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = POLICY_PROFILES[args.profile]
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    checkpoint_path, checkpoint_sha = verify_checkpoint(profile, checkpoint_root)
    if not os.environ.get("METAGROSS_SHOWDOWN_PASSWORD"):
        raise RuntimeError("set METAGROSS_SHOWDOWN_PASSWORD before launching")
    engine_provenance = inspect_foul_play_engine(args.foul_play_python)

    run_dir = make_run_dir(args.output_root, args.profile, args.username)
    prior_log_path = run_dir / "prior.log"
    client_log_path = run_dir / "client.log"
    decision_dump_path = run_dir / "decisions.jsonl"
    protocol_dump_path = run_dir / "protocol.jsonl"
    search_dump_path = run_dir / "search.jsonl"
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "schema": 1,
        "status": "starting",
        "started_at": utc_now(),
        "launcher_pid": os.getpid(),
        "profile": args.profile,
        "policy": asdict(profile),
        "checkpoint": {"path": str(checkpoint_path), "sha256_verified": checkpoint_sha},
        "poke_engine": engine_provenance,
        "ladder": {"username": args.username, "format": FORMAT, "games": args.games},
        "search": {
            "search_time_ms": SEARCH_TIME_MS,
            "parallelism": args.search_parallelism,
            "threads": args.search_threads,
            "c_puct": CPU_C_PUCT,
            "execution": "modal" if args.remote_mcts else "local",
        },
        "limits": {
            "stall_timeout_seconds": args.stall_timeout_seconds,
            "max_runtime_seconds": args.max_runtime_seconds,
        },
        "outputs": {
            "client_log": str(client_log_path),
            "prior_log": str(prior_log_path),
            "ratings": str(run_dir / "ratings.jsonl"),
            "decisions": str(decision_dump_path),
            "protocol": str(protocol_dump_path),
            "search": str(search_dump_path),
        },
    }
    write_json(manifest_path, manifest)

    prior_url = f"http://127.0.0.1:{args.port}"
    common_env = production_environment(os.environ)
    common_env.update(
        {
            "METAMON_CACHE_DIR": str(ROOT / "srcs" / "runtime" / "metamon-cache"),
            "TORCHDYNAMO_DISABLE": "1",
            "ACCELERATE_USE_CPU": "true",
            "WANDB_MODE": "disabled",
            "METAGROSS_CPUCT": str(CPU_C_PUCT),
        }
    )
    prior_env = common_env.copy()
    prior_env.pop("METAGROSS_SHOWDOWN_PASSWORD", None)
    prior_env.pop("MODAL_TOKEN_ID", None)
    prior_env.pop("MODAL_TOKEN_SECRET", None)
    prior_command = [
        str(args.metamon_python),
        "-u",
        str(ROOT / "srcs" / "metagross" / "prior_server.py"),
        "--local-run-dir",
        str(checkpoint_root),
        "--local-run-name",
        profile.run_name,
        "--checkpoint",
        str(profile.checkpoint),
        "--checkpoint-sha256",
        checkpoint_sha,
        "--port",
        str(args.port),
        "--username",
        args.username,
        "--decision-dump",
        str(decision_dump_path),
    ]
    client_command = [
        str(args.foul_play_python),
        "-u",
        str(ROOT / "srcs" / "metagross" / "run_foul_play.py"),
        "--websocket-uri",
        args.websocket_uri,
        "--ps-username",
        args.username,
        "--bot-mode",
        "search_ladder",
        "--pokemon-format",
        FORMAT,
        "--run-count",
        str(args.games),
        "--search-time-ms",
        str(SEARCH_TIME_MS),
        "--search-parallelism",
        str(args.search_parallelism),
        "--search-threads",
        str(args.search_threads),
        "--log-level",
        "INFO",
        "--save-replay",
        "always",
    ]

    shutdown = threading.Event()
    received_signal: list[int] = []

    def request_shutdown(signum, _frame) -> None:
        received_signal.append(signum)
        shutdown.set()

    previous_handlers = {}
    prior = None
    client = None
    rating_thread = None
    status = "failed"
    detail = None
    started = time.monotonic()
    account_lock = acquire_account_lock(args.username)
    try:
        previous_handlers = {
            signum: signal.signal(signum, request_shutdown)
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        with prior_log_path.open("w", encoding="utf-8") as prior_log, client_log_path.open(
            "w", encoding="utf-8"
        ) as client_log:
            prior = subprocess.Popen(
                prior_command,
                cwd=ROOT,
                env=prior_env,
                stdout=prior_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_for_health(prior_url, prior, args.health_timeout_seconds, shutdown)
            client_env = common_env.copy()
            client_env.update(
                {
                    "FOUL_PLAY_DIR": str(ROOT / "srcs" / "vendor" / "foul-play"),
                    "METAGROSS_PRIOR_SERVER": prior_url,
                    "METAGROSS_CPUCT": "2.0",
                    "METAGROSS_REQUIRE_PRIORS": "1",
                    "METAGROSS_PROTOCOL_DUMP": str(protocol_dump_path),
                    "METAGROSS_SEARCH_DUMP": str(search_dump_path),
                }
            )
            if args.remote_mcts:
                client_env.update(
                    {
                        "METAGROSS_REQUIRE_REMOTE_MCTS": "1",
                        "METAGROSS_REMOTE_MCTS_APP": args.remote_mcts_app,
                        "METAGROSS_REMOTE_MCTS_FUNCTION": args.remote_mcts_function,
                        "METAGROSS_REMOTE_ENGINE_SHA256": args.remote_engine_sha256,
                    }
                )
                manifest["search"]["modal"] = {
                    "app": args.remote_mcts_app,
                    "function": args.remote_mcts_function,
                    "engine_sha256": args.remote_engine_sha256,
                    "schema": 1,
                }
                write_json(manifest_path, manifest)
            client = subprocess.Popen(
                client_command,
                cwd=ROOT,
                env=client_env,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            manifest.update({"status": "running", "prior_pid": prior.pid, "client_pid": client.pid})
            write_json(manifest_path, manifest)
            rating_thread = threading.Thread(
                target=poll_rating,
                args=(args.username, run_dir / "ratings.jsonl", shutdown, args.rating_poll_seconds),
                daemon=True,
            )
            rating_thread.start()
            last_health = 0.0
            while not shutdown.wait(1):
                now = time.monotonic()
                client_code = client.poll()
                if client_code is not None:
                    if client_code != 0:
                        raise RuntimeError(f"ladder client exited with code {client_code}")
                    status = "completed"
                    return_code = 0
                    break
                if prior.poll() is not None:
                    raise RuntimeError(f"prior server exited with code {prior.returncode}")
                if now - last_health >= args.health_poll_seconds:
                    if not prior_is_healthy(prior_url):
                        raise RuntimeError("prior server health check failed during ladder run")
                    last_health = now
                if now - started > args.max_runtime_seconds:
                    raise TimeoutError("maximum ladder runtime exceeded")
                if time.time() - client_log_path.stat().st_mtime > args.stall_timeout_seconds:
                    raise TimeoutError("ladder client output stalled")
            else:
                status = "interrupted"
                return_code = 128 + (received_signal[-1] if received_signal else signal.SIGTERM)
    except Exception as exc:
        if received_signal:
            status = "interrupted"
            return_code = 128 + received_signal[-1]
        else:
            detail = f"{type(exc).__name__}: {exc}"
            raise
    finally:
        shutdown.set()
        terminate_processes([client, prior])
        if rating_thread is not None:
            rating_thread.join(timeout=2)
        manifest.update({"status": status, "finished_at": utc_now()})
        if detail:
            manifest["error"] = detail
        write_json(manifest_path, manifest)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        account_lock.close()
    return return_code


if __name__ == "__main__":
    sys.exit(main())
