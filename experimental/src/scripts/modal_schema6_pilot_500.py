#!/usr/bin/env python3
"""Correctness-gated, resumable schema-6 capture and screening on Modal CPUs."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import modal


SCRIPT_PATH = Path(__file__).resolve()
IS_LOCAL_CHECKOUT = len(SCRIPT_PATH.parents) > 3 and (SCRIPT_PATH.parents[3] / "srcs").is_dir()
ROOT = SCRIPT_PATH.parents[3] if IS_LOCAL_CHECKOUT else Path("/workspace")
EXPERIMENTAL_SRC = ROOT / "experimental" / "src"
if str(EXPERIMENTAL_SRC) not in sys.path:
    # Modal executes this entrypoint from /root even though the project source
    # tree is copied under /workspace.  Keep in-process imports on the same
    # explicit path already used by capture subprocesses.
    sys.path.insert(0, str(EXPERIMENTAL_SRC))
AMAGO = ROOT / ".venv-metamon" / "lib" / "python3.11" / "site-packages" / "amago"
SHOWDOWN_COMMIT = "4880d3693580bd33652797cf31179c6fcdf87e50"
CHECKPOINT_SHA256 = "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"
CAPTURE_ENGINE_SOURCE_SHA256 = "ccc5dd035c25fbf8e6b98301198cdaa35ce1012ae6f45ab81851965b0b05a0a3"
AUDIT_ENGINE_LINUX_SHA256 = "8f69e753b0b2252255b180033d14cabb0bfd6f54e8d37d51a0bbfd52a3eb7ec9"
VOLUME_NAME = "metagross-online-rl"
PERSIST_ROOT = Path("/data/schema6_capture_500")
GAMES_PER_UNIT = 10
PILOT_UNITS = {"peer": 30, "direct_r1": 10, "unguided": 10}
SCALE_5000_UNITS = {"peer": 300, "direct_r1": 100, "unguided": 100}
EXPECTED_UNITS = PILOT_UNITS
MAX_UNITS = SCALE_5000_UNITS
EXPECTED_AGENT_B = {
    "peer": "production_r1_search_first",
    "direct_r1": "direct_r1",
    "unguided": "foul_play",
}
PROFILE_DOMAIN = {"peer": 1, "direct_r1": 2, "unguided": 3}
RANDBATS_POOL = ROOT / "experimental" / "data" / "randbats_pools" / "gen9randombattle_pool_50000.json"
APP = modal.App("metagross-schema6-capture-500")
app = APP
VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


def _build_image() -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("build-essential", "curl", "git", "nodejs", "npm")
        .pip_install(
            "accelerate",
            "datasets",
            "einops",
            "gin-config",
            "gymnasium==0.29.1",
            "huggingface_hub",
            "lz4",
            "maturin>=1.0,<2.0",
            "numpy==2.4.6",
            "pandas",
            "ratarmountcore",
            "rich",
            "scipy",
            "termcolor",
            "torch==2.12.1",
            "tqdm",
            "wandb",
            "poke-env @ git+https://github.com/UT-Austin-RPL/poke-env.git@e1268d270c3f2bd32c7ff5713e01062302020579",
            "amago @ git+https://github.com/UT-Austin-RPL/amago@0974781a9096ff43df1b708312256f96fc2ab127",
        )
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
            "git clone https://github.com/smogon/pokemon-showdown.git /workspace/external/pokemon-showdown",
            f"cd /workspace/external/pokemon-showdown && git checkout {SHOWDOWN_COMMIT}",
            "cd /workspace/external/pokemon-showdown && npm ci && npm run build",
            "mkdir -p /workspace/external/pokemon-showdown/databases /workspace/external/pokemon-showdown/logs/repl /workspace/external/pokemon-showdown/logs/modlog /workspace/external/pokemon-showdown/logs/ladder /workspace/external/metamon_cache",
        )
    )
    if not IS_LOCAL_CHECKOUT:
        return image
    return (
        image.add_local_dir(
            ROOT / "srcs" / "metagross",
            "/workspace/srcs/metagross",
            copy=True,
            ignore=["__pycache__", "*.pyc", "tests"],
        )
        .add_local_dir(
            ROOT / "srcs" / "vendor" / "metamon" / "metamon",
            "/usr/local/lib/python3.11/site-packages/metamon",
            copy=True,
            ignore=["__pycache__", "*.pyc"],
        )
        .add_local_dir(
            AMAGO,
            "/usr/local/lib/python3.11/site-packages/amago",
            copy=True,
            ignore=["__pycache__", "*.pyc"],
        )
        .add_local_dir(
            ROOT / "srcs" / "vendor" / "foul-play",
            "/workspace/srcs/vendor/foul-play",
            copy=True,
            ignore=[".git", "__pycache__", "*.pyc", "external", "tests"],
        )
        .add_local_dir(
            ROOT / "srcs" / "vendor" / "poke-engine",
            "/workspace/srcs/vendor/poke-engine",
            copy=True,
            ignore=[".git", "__pycache__", "*.pyc", "linux_wheels", "release", "target"],
        )
        .add_local_dir(
            ROOT / "experimental" / "engine" / "pe_v3_learned_priors",
            "/workspace/experimental/engine/pe_v3_learned_priors",
            copy=True,
            ignore=[".git", ".cargo", "__pycache__", "*.pyc", "linux_wheels", "release", "target", "tests"],
        )
        .add_local_dir(
            ROOT / "experimental" / "src",
            "/workspace/experimental/src",
            copy=True,
            ignore=["__pycache__", "*.pyc", "tests", "nets", "external", "experiments"],
        )
        .add_local_file(
            RANDBATS_POOL,
            "/workspace/experimental/data/randbats_pools/gen9randombattle_pool_50000.json",
            copy=True,
        )
        .run_commands(
            "python -m venv --system-site-packages /workspace/.venv-foul-play",
            "/workspace/.venv-foul-play/bin/python -m pip install maturin requests==2.33.0 websockets==15.0.1 python-dateutil==2.8.0",
            "python /workspace/experimental/src/scripts/prepare_production_capture_engine.py apply",
            "PATH=/root/.cargo/bin:$PATH CARGO_TARGET_DIR=/tmp/capture-engine /workspace/.venv-foul-play/bin/python -m pip install --no-cache-dir /workspace/srcs/vendor/poke-engine --config-settings='build-args=--no-default-features --features poke-engine/gen9,poke-engine/terastallization'",
            "python /workspace/experimental/src/scripts/prepare_production_capture_engine.py restore",
            "PATH=/root/.cargo/bin:$PATH CARGO_TARGET_DIR=/tmp/audit-engine python -m pip install --no-cache-dir /workspace/experimental/engine/pe_v3_learned_priors --config-settings='build-args=--no-default-features --features poke-engine/gen9,poke-engine/terastallization'",
            "/workspace/.venv-foul-play/bin/python -c \"import inspect,poke_engine; p=list(inspect.signature(poke_engine.monte_carlo_tree_search).parameters); assert 'seed' not in p, p; assert p == ['state','duration_ms','iterations','threads','s1_priors','s2_priors','c_puct'], p\"",
            "PATH=/root/.cargo/bin:$PATH CARGO_TARGET_DIR=/tmp/capture-tests cargo test --manifest-path /workspace/srcs/vendor/poke-engine/Cargo.toml --no-default-features --features gen9,terastallization",
            "PATH=/root/.cargo/bin:$PATH CARGO_TARGET_DIR=/tmp/audit-tests cargo test --manifest-path /workspace/experimental/engine/pe_v3_learned_priors/Cargo.toml --no-default-features --features gen9,terastallization",
            "/workspace/.venv-foul-play/bin/python -c \"import hashlib,pathlib,poke_engine.poke_engine as p; q=pathlib.Path(p.__file__); pathlib.Path('/workspace/capture-engine-linux.sha256').write_text(hashlib.sha256(q.read_bytes()).hexdigest()+'\\n')\"",
            "python -c \"import hashlib,pathlib,poke_engine.poke_engine as p; q=pathlib.Path(p.__file__); pathlib.Path('/workspace/audit-engine-linux.sha256').write_text(hashlib.sha256(q.read_bytes()).hexdigest()+'\\n')\"",
            "rm -rf /tmp/capture-engine /tmp/audit-engine /tmp/capture-tests /tmp/audit-tests",
        )
    )


IMAGE = _build_image()


def _safe_run_id(run_id: str) -> str:
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_id):
        raise ValueError("run_id must contain only letters, numbers, hyphens, and underscores")
    return run_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit_name(profile: str, unit_index: int) -> str:
    return f"{profile}-{unit_index:03d}"


def _unit_destination(run_id: str, profile: str, unit_index: int) -> Path:
    return PERSIST_ROOT / _safe_run_id(run_id) / "units" / _unit_name(profile, unit_index)


def _fingerprint() -> dict[str, str]:
    capture_python = Path("/workspace/.venv-foul-play/bin/python")
    capture_binary = subprocess.check_output(
        [str(capture_python), "-c", "import poke_engine.poke_engine as p; print(p.__file__)"],
        text=True,
    ).strip()
    audit_binary = subprocess.check_output(
        [sys.executable, "-c", "import poke_engine.poke_engine as p; print(p.__file__)"],
        text=True,
    ).strip()
    source_hash = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; from srcs.metagross.mcts_contract import compute_engine_source_sha256; print(compute_engine_source_sha256(Path('/workspace/srcs/vendor/poke-engine')))",
        ],
        cwd="/workspace",
        text=True,
    ).strip()
    audit_source_hash = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; from srcs.metagross.mcts_contract import compute_engine_source_sha256; print(compute_engine_source_sha256(Path('/workspace/experimental/engine/pe_v3_learned_priors')))",
        ],
        cwd="/workspace",
        text=True,
    ).strip()
    showdown = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd="/workspace/external/pokemon-showdown", text=True
    ).strip()
    capture_signature = subprocess.check_output(
        [
            str(capture_python),
            "-c",
            "import inspect,poke_engine; print(inspect.signature(poke_engine.monte_carlo_tree_search))",
        ],
        text=True,
    ).strip()
    checkpoint = Path("/data/accepted/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt")
    result = {
        "checkpoint_sha256": _sha256(checkpoint),
        "capture_engine_source_sha256": source_hash,
        "capture_engine_binary_sha256": _sha256(Path(capture_binary)),
        "capture_engine_image_sha256": Path("/workspace/capture-engine-linux.sha256").read_text().strip(),
        "audit_engine_source_sha256": audit_source_hash,
        "audit_engine_binary_sha256": _sha256(Path(audit_binary)),
        "audit_engine_image_sha256": Path("/workspace/audit-engine-linux.sha256").read_text().strip(),
        "showdown_commit": showdown,
        "capture_mcts_signature": capture_signature,
    }
    if result["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("R1 checkpoint hash mismatch")
    if result["capture_engine_source_sha256"] != CAPTURE_ENGINE_SOURCE_SHA256:
        raise RuntimeError("capture engine source hash mismatch")
    if result["capture_engine_binary_sha256"] != result["capture_engine_image_sha256"]:
        raise RuntimeError("capture engine binary changed after image build")
    if result["audit_engine_binary_sha256"] != result["audit_engine_image_sha256"]:
        raise RuntimeError("audit engine binary changed after image build")
    if result["showdown_commit"] != SHOWDOWN_COMMIT:
        raise RuntimeError("Showdown commit mismatch")
    if "seed" in result["capture_mcts_signature"]:
        raise RuntimeError("experimental capture-engine MCTS ABI detected")
    return result


def _wait_for_port(process: subprocess.Popen[Any], port: int, timeout: float = 240) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service on port {port} exited with {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"service on port {port} did not become ready")


def _stop(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        METAMON_CACHE_DIR="/data/metamon_cache",
        HF_HOME="/data/hf_home",
        WANDB_MODE="disabled",
        TORCHDYNAMO_DISABLE="1",
        ACCELERATE_USE_CPU="true",
        CUDA_VISIBLE_DEVICES="",
        METAGROSS_DUAL_R1_CAPTURE="1",
        # Direct R1 performs a cold 571 MB transformer load before it can
        # accept the challenge.  Keep the already-connected production search
        # client alive during that one-time initialization.
        METAGROSS_WEBSOCKET_RECEIVE_TIMEOUT_SECONDS="600",
        PYTHONPATH="/workspace/experimental/src:/workspace",
    )
    return environment


def _start_prior(side: str, port: int, root: Path) -> subprocess.Popen[Any]:
    command = [
        sys.executable,
        "-u",
        "/workspace/srcs/metagross/prior_server.py",
        "--local-run-dir", "/data/accepted",
        "--local-run-name", "randbats_exit_r1",
        "--local-base-model", "Kakuna",
        "--checkpoint", "5",
        "--checkpoint-sha256", CHECKPOINT_SHA256,
        "--trajectory-mode", "causal-history",
        "--username", f"schema6{side}",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--decision-dump", str(root / f"prior-{side}.jsonl"),
    ]
    return subprocess.Popen(
        command,
        cwd="/workspace",
        env=_environment(),
        stdout=(root / f"prior-{side}.log").open("w"),
        stderr=subprocess.STDOUT,
        text=True,
    )


def _run_checked(command: list[str], root: Path, log_name: str, timeout: int = 3600) -> None:
    with (root / log_name).open("w") as output:
        subprocess.run(
            command,
            check=True,
            cwd="/workspace",
            env=_environment(),
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )


def _audit_unit(root: Path, profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    decisions = [root / "agent-a-decisions.jsonl.dual-r1-roots.jsonl"]
    snapshots = [root / "prior-a.jsonl"]
    if profile == "peer":
        decisions.append(root / "agent-b-decisions.jsonl.dual-r1-roots.jsonl")
        snapshots.append(root / "prior-b.jsonl")
    decision_args = [item for path in decisions for item in ("--decision-log", str(path))]
    snapshot_args = [item for path in snapshots for item in ("--prior-snapshot", str(path))]
    expected_groups = GAMES_PER_UNIT * (2 if profile == "peer" else 1)
    minimum_complete = math.ceil(expected_groups * 0.95)
    capture_path = root / "schema6-capture-audit.json"
    _run_checked(
        [
            sys.executable,
            "/workspace/experimental/src/scripts/audit_schema6_capture.py",
            *decision_args,
            *snapshot_args,
            "--h2h-result", str(root / "result.json"),
            "--minimum-battles", str(minimum_complete),
            "--minimum-capture-rate", "0.95",
            "--output", str(capture_path),
        ],
        root,
        "capture-audit.log",
    )
    bridge_path = root / "schema6-panel-bridge-audit.json"
    _run_checked(
        [
            sys.executable,
            "/workspace/experimental/src/scripts/audit_schema6_panel_bridge.py",
            *decision_args,
            *snapshot_args,
            "--minimum-groups", "1",
            "--output", str(bridge_path),
        ],
        root,
        "panel-bridge-audit.log",
    )
    return json.loads(capture_path.read_text()), json.loads(bridge_path.read_text())


def _validate_game_result(result: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    summary = result.get("summary") or {}
    games = result.get("games")
    if not isinstance(games, list) or len(games) != GAMES_PER_UNIT:
        raise RuntimeError("unit did not produce exactly ten games")
    required_summary = {
        "agent_a": "production_r1_search_first",
        "agent_b": EXPECTED_AGENT_B[profile],
        "completed_games": GAMES_PER_UNIT,
        "completed_pairs": GAMES_PER_UNIT // 2,
        "void_games": 0,
        "void_pairs": 0,
        "paired": True,
        "mirrored_pairs": True,
        "foul_play_search_time_ms": 500,
    }
    if any(summary.get(key) != value for key, value in required_summary.items()):
        raise RuntimeError(f"invalid H2H summary: {summary}")
    pairs: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for game in games:
        if game.get("void") or game.get("error") is not None:
            raise RuntimeError("unit contains a void or errored game")
        pairs[int(game["pair_index"])].append(game)
    if len(pairs) != GAMES_PER_UNIT // 2:
        raise RuntimeError("unit has the wrong mirrored-pair count")
    for pair in pairs.values():
        if sorted(int(game["pair_leg"]) for game in pair) != [1, 2]:
            raise RuntimeError("mirrored pair lacks both legs")
        identities = {
            (game["battle_seed"], game["team_1_sha256"], game["team_2_sha256"])
            for game in pair
        }
        if len(identities) != 1:
            raise RuntimeError("mirrored pair changed seed or teams between legs")
    return games


def _validate_persisted(row: Any, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise RuntimeError("persisted unit report is not an object")
    expected = {
        "schema": "metagross-schema6-modal-unit/v1",
        "run_id": str(payload["run_id"]),
        "profile": str(payload["profile"]),
        "unit_index": int(payload["unit_index"]),
        "games": GAMES_PER_UNIT,
        "mirror_seed": int(payload["mirror_seed"]),
        "production_seed": str(payload["production_seed"]),
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise RuntimeError("persisted unit identity differs from the requested unit")
    if row.get("admitted") is not True:
        raise RuntimeError("persisted unit was not admitted")
    return row


@APP.function(
    image=IMAGE,
    cpu=(16.0, 16.0),
    memory=(24576, 24576),
    timeout=3600,
    max_containers=40,
    retries=2,
    volumes={"/data": VOLUME},
)
def run_unit(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = _safe_run_id(str(payload["run_id"]))
    profile = str(payload["profile"])
    unit_index = int(payload["unit_index"])
    if profile not in MAX_UNITS or not 0 <= unit_index < MAX_UNITS[profile]:
        raise ValueError("invalid profile or unit index")
    destination = _unit_destination(run_id, profile, unit_index)
    report_path = destination / "unit-report.json"
    if report_path.exists():
        return _validate_persisted(json.loads(report_path.read_text()), payload)

    root = Path(tempfile.mkdtemp(prefix=f"schema6-{profile}-{unit_index:03d}-"))
    (root / "logs").mkdir()
    (root / "registrations").mkdir()
    showdown = prior_a = prior_b = None
    showdown_log = (root / "showdown.log").open("w")
    started = time.monotonic()
    try:
        fingerprint = _fingerprint()
        showdown = subprocess.Popen(
            ["node", "pokemon-showdown", "start", "--no-security", "8011"],
            cwd="/workspace/external/pokemon-showdown",
            stdout=showdown_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _wait_for_port(showdown, 8011)
        prior_a = _start_prior("a", 8977, root)
        _wait_for_port(prior_a, 8977)
        if profile == "peer":
            prior_b = _start_prior("b", 8978, root)
            _wait_for_port(prior_b, 8978)

        eval_command = [
            sys.executable,
            "/workspace/experimental/src/eval/run.py",
            "--mode", "h2h",
            "--server", "local",
            "--format", "gen9randombattle",
            "--websocket-uri", "ws://127.0.0.1:8011/showdown/websocket",
            "--paired",
            "--mirrored-pairs",
            "--mirror-seed", str(payload["mirror_seed"]),
            "--showdown-dir", "/workspace/external/pokemon-showdown",
            "--mirrored-team-generator", "/workspace/experimental/src/scripts/generate_mirrored_randbats_pair.cjs",
            "--pair-registration-dir", str(root / "registrations"),
            "--agent-a", "production_r1_search_first",
            "--agent-b", EXPECTED_AGENT_B[profile],
            "--agent-a-prior-server-url", "http://127.0.0.1:8977",
            "--agent-a-require-priors",
            "--agent-a-decision-log", str(root / "agent-a-decisions.jsonl"),
            "--foul-play-python", "/workspace/.venv-foul-play/bin/python",
            "--metamon-python", sys.executable,
            "--direct-r1-checkpoint-root", "/data/accepted",
            "--foul-play-search-time-ms", "500",
            "--foul-play-search-parallelism", "8",
            "--foul-play-search-threads", "1",
            "--cpuct", "2.0",
            "--production-run-seed", str(payload["production_seed"]),
            "--concurrent-games", "1",
            "--fail-fast",
            "--game-timeout-seconds", "900",
            "--n-games", str(GAMES_PER_UNIT),
            "--username-prefix", str(payload["username_prefix"]),
            "--run-id", _unit_name(profile, unit_index),
            "--json-out", str(root / "result.json"),
            "--log-dir", str(root / "logs"),
        ]
        if profile == "peer":
            eval_command.extend(
                [
                    "--agent-b-prior-server-url", "http://127.0.0.1:8978",
                    "--agent-b-require-priors",
                    "--strict-isolated-priors",
                    "--agent-b-decision-log", str(root / "agent-b-decisions.jsonl"),
                ]
            )
        _run_checked(eval_command, root, "eval.log")
        result = json.loads((root / "result.json").read_text())
        games = _validate_game_result(result, profile)
        capture, bridge = _audit_unit(root, profile)
        expected_groups = GAMES_PER_UNIT * (2 if profile == "peer" else 1)
        admitted = bool(
            capture.get("admitted") is True
            and capture.get("groups") == expected_groups
            and capture.get("complete_groups", 0) >= math.ceil(expected_groups * 0.95)
            and capture.get("duplicate_decisions") == 0
            and capture.get("duplicate_snapshots") == 0
            and capture.get("invalid_snapshots") == 0
            and bridge.get("admitted") is True
            and int(bridge.get("candidate_rows", 0)) > 0
        )
        report = {
            "schema": "metagross-schema6-modal-unit/v1",
            "admitted": admitted,
            "run_id": run_id,
            "profile": profile,
            "unit_index": unit_index,
            "games": GAMES_PER_UNIT,
            "mirror_seed": int(payload["mirror_seed"]),
            "production_seed": str(payload["production_seed"]),
            "username_prefix": str(payload["username_prefix"]),
            "elapsed_seconds": time.monotonic() - started,
            "capture": capture,
            "bridge": bridge,
            "runtime_fingerprint": fingerprint,
            "pair_identities": [
                {
                    "battle_seed": game["battle_seed"],
                    "team_1_sha256": game["team_1_sha256"],
                    "team_2_sha256": game["team_2_sha256"],
                }
                for game in games
                if int(game["pair_leg"]) == 1
            ],
        }
        if not admitted:
            raise RuntimeError(f"unit failed schema-6 admission: {json.dumps(report, sort_keys=True)}")
        (root / "unit-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except Exception:
        failed = PERSIST_ROOT / run_id / "failed" / f"{_unit_name(profile, unit_index)}-{uuid.uuid4().hex[:8]}"
        failed.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, failed)
        VOLUME.commit()
        raise
    finally:
        _stop(prior_b)
        _stop(prior_a)
        _stop(showdown)
        showdown_log.close()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.tmp")
    shutil.copytree(root, temporary)
    if destination.exists():
        existing = _validate_persisted(json.loads((destination / "unit-report.json").read_text()), payload)
        shutil.rmtree(temporary)
        return existing
    os.replace(temporary, destination)
    VOLUME.commit()
    return report


def _payloads(
    run_id: str,
    seed: int,
    expected_units: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    expected_units = expected_units or PILOT_UNITS
    payloads = []
    global_index = 0
    for profile, count in expected_units.items():
        for unit_index in range(count):
            identity = f"{run_id}:{profile}:{unit_index}:{seed}"
            production_seed = hashlib.sha256(f"production:{identity}".encode()).hexdigest()
            username_digest = hashlib.sha256(f"username:{identity}".encode()).hexdigest()[:4]
            payloads.append(
                {
                    "run_id": run_id,
                    "profile": profile,
                    "unit_index": unit_index,
                    "mirror_seed": seed + PROFILE_DOMAIN[profile] * 1_000_000 + unit_index,
                    "production_seed": production_seed,
                    "username_prefix": f"m{global_index:03x}{username_digest}"[:8],
                }
            )
            global_index += 1
    return payloads


def _aggregate(
    run_id: str,
    rows: list[dict[str, Any]],
    seed: int,
    expected_units: dict[str, int] | None = None,
    expected_games: int = 500,
    authorization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_units = expected_units or PILOT_UNITS
    rows.sort(key=lambda row: (str(row["profile"]), int(row["unit_index"])))
    expected_keys = {
        (profile, unit_index)
        for profile, count in expected_units.items()
        for unit_index in range(count)
    }
    observed_keys = {(str(row["profile"]), int(row["unit_index"])) for row in rows}
    fingerprints = {json.dumps(row["runtime_fingerprint"], sort_keys=True) for row in rows}
    pair_ids = [
        (
            pair["battle_seed"],
            pair["team_1_sha256"],
            pair["team_2_sha256"],
        )
        for row in rows
        for pair in row["pair_identities"]
    ]
    strata = {}
    for profile, expected_count in expected_units.items():
        selected = [row for row in rows if row["profile"] == profile]
        strata[profile] = {
            "units": len(selected),
            "games": sum(int(row["games"]) for row in selected),
            "capture_groups": sum(int(row["capture"]["groups"]) for row in selected),
            "complete_groups": sum(int(row["capture"]["complete_groups"]) for row in selected),
            "candidate_rows": sum(int(row["bridge"]["candidate_rows"]) for row in selected),
            "expected_units": expected_count,
        }
    total_games = sum(int(row["games"]) for row in rows)
    capture_groups = sum(int(row["capture"]["groups"]) for row in rows)
    complete_groups = sum(int(row["capture"]["complete_groups"]) for row in rows)
    expected_capture_groups = sum(
        count * GAMES_PER_UNIT * (2 if profile == "peer" else 1)
        for profile, count in expected_units.items()
    )
    admitted = bool(
        observed_keys == expected_keys
        and len(rows) == len(expected_keys)
        and all(row.get("admitted") is True for row in rows)
        and total_games == expected_games
        and len(pair_ids) == expected_games // 2
        and len(set(pair_ids)) == expected_games // 2
        and len(fingerprints) == 1
        and capture_groups == expected_capture_groups
        and complete_groups >= math.ceil(capture_groups * 0.95)
    )
    report = {
        "schema": (
            "metagross-schema6-modal-pilot-summary/v1"
            if expected_games == 500
            else "metagross-schema6-modal-capture-summary/v2"
        ),
        "admitted": admitted,
        "stage_games": expected_games,
        "run_id": run_id,
        "seed": seed,
        "completed_units": len(rows),
        "completed_games": total_games,
        "unique_mirrored_pairs": len(set(pair_ids)),
        "capture_groups": capture_groups,
        "complete_groups": complete_groups,
        "capture_rate": complete_groups / capture_groups if capture_groups else 0.0,
        "candidate_rows": sum(int(row["bridge"]["candidate_rows"]) for row in rows),
        "runtime_fingerprint": rows[0]["runtime_fingerprint"] if len(fingerprints) == 1 else None,
        "strata": strata,
        "units": rows,
    }
    if expected_games == 500:
        report.update(
            scale_admitted=False,
            scale_blocker="requires >=50 roots after frozen 20k/50k four-way agreement screening",
        )
    else:
        report.update(
            authorization=authorization,
            model_data_stage_admitted=admitted,
            next_scale_admitted=False,
            next_scale_blocker=(
                "requires physical-battle-grouped out-of-fold outcome-residual improvement "
                "before any 25,000-game stage"
            ),
        )
    return report


@APP.function(image=modal.Image.debian_slim(python_version="3.11"), volumes={"/data": VOLUME})
def persist_summary(run_id: str, report: dict[str, Any]) -> str:
    destination = PERSIST_ROOT / _safe_run_id(run_id) / "summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    VOLUME.commit()
    return str(destination)


def _panel_destination(run_id: str, panel_seed: int) -> Path:
    return PERSIST_ROOT / _safe_run_id(run_id) / "panels" / f"training-seed-{panel_seed}"


def _load_admitted_unit_sources(run_root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    unit_directories = sorted(path for path in (run_root / "units").iterdir() if path.is_dir())
    reports = []
    decisions = []
    priors = []
    for directory in unit_directories:
        report_path = directory / "unit-report.json"
        if not report_path.is_file() or json.loads(report_path.read_text()).get("admitted") is not True:
            raise RuntimeError(f"panel source unit is not admitted: {directory}")
        reports.append(report_path)
        decisions.extend(sorted(directory.glob("agent-*-decisions.jsonl.dual-r1-roots.jsonl")))
        priors.extend(sorted(directory.glob("prior-*.jsonl")))
    if len(reports) != 50 or len(decisions) != 80 or len(priors) != 80:
        raise RuntimeError(
            f"panel source cardinality changed: units={len(reports)} "
            f"decisions={len(decisions)} priors={len(priors)}"
        )
    return reports, decisions, priors


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@APP.function(
    image=IMAGE,
    cpu=(16.0, 16.0),
    memory=(24576, 24576),
    timeout=7200,
    retries=1,
    volumes={"/data": VOLUME},
)
def freeze_training_panel(run_id: str, panel_seed: int) -> dict[str, Any]:
    """Materialize only the deterministic 60% training split from admitted units."""
    run_id = _safe_run_id(run_id)
    run_root = PERSIST_ROOT / run_id
    destination = _panel_destination(run_id, panel_seed)
    existing_report = destination / "training-panel-report.json"
    if existing_report.is_file():
        report = json.loads(existing_report.read_text())
        if (
            report.get("purpose") != "training"
            or report.get("seed") != panel_seed
            or report.get("requested_battles") != "all_eligible"
            or report.get("split_contract", {}).get("selected_split") != "train"
            or report.get("split_contract", {}).get(
                "withheld_history_policy_feature_rows_processed"
            ) != 0
        ):
            raise RuntimeError("persisted training panel identity changed")
        return report

    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("admitted") is not True or summary.get("completed_games") != 500:
        raise RuntimeError("training panel requires the admitted 500-game corpus")
    unit_reports, decision_logs, prior_snapshots = _load_admitted_unit_sources(run_root)

    from scripts.build_causal_action_q_panel import build

    temporary_root = Path(tempfile.mkdtemp(prefix="schema6-training-panel-"))
    panel_path = temporary_root / "training-panel.jsonl"
    report_path = temporary_root / "training-panel-report.json"
    try:
        report = build(
            SimpleNamespace(
                decision_log=decision_logs,
                prior_snapshot=prior_snapshots,
                terminal_trajectories=None,
                history_authority="schema6_snapshot",
                exclude_panel=[],
                pool=Path(
                    "/workspace/experimental/data/randbats_pools/"
                    "gen9randombattle_pool_50000.json"
                ),
                output=panel_path,
                report=report_path,
                purpose="training",
                battles=None,
                all_eligible=True,
                schedules=2,
                worlds=8,
                minimum_history=3,
                minimum_legal_actions=4,
                minimum_entropy=0.45,
                seed=panel_seed,
            )
        )
        report["corpus"] = {
            "run_id": run_id,
            "summary_sha256": _sha256(summary_path),
            "unit_reports": len(unit_reports),
            "decision_logs": len(decision_logs),
            "prior_snapshots": len(prior_snapshots),
            "unit_report_set_sha256": hashlib.sha256(
                "\n".join(_sha256(path) for path in unit_reports).encode("ascii")
            ).hexdigest(),
        }
        _atomic_json(report_path, report)
        temporary_destination = destination.with_name(
            f".{destination.name}-{uuid.uuid4().hex}.tmp"
        )
        temporary_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(temporary_root, temporary_destination)
        if destination.exists():
            shutil.rmtree(temporary_destination)
            return json.loads(existing_report.read_text())
        os.replace(temporary_destination, destination)
        VOLUME.commit()
        return report
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


@APP.function(
    image=IMAGE,
    cpu=2.0,
    memory=4096,
    timeout=1800,
    volumes={"/data": VOLUME},
)
def validate_training_panel(run_id: str, panel_seed: int) -> dict[str, Any]:
    """Independently validate panel identity, split isolation, and world hashes."""
    from scripts.run_public_mcts_leaf_gate import PANEL_SCHEMA
    from train.shallow_search_residual import battle_split

    destination = _panel_destination(_safe_run_id(run_id), panel_seed)
    panel_path = destination / "training-panel.jsonl"
    report_path = destination / "training-panel-report.json"
    report = json.loads(report_path.read_text())
    split_contract = report.get("split_contract", {})
    source_inventory = split_contract.get("source_physical_battles_by_split", {})
    if (
        split_contract.get("selected_split") != "train"
        or split_contract.get("withheld_history_policy_feature_rows_processed") != 0
        or split_contract.get("gate_stage")
        != "identity_before_history_policy_public_state_or_determinization"
        or set(source_inventory) != {"train", "calibration", "test"}
        or sum(int(value) for value in source_inventory.values()) != 500
    ):
        raise RuntimeError("training panel split-isolation contract changed")
    rows = [json.loads(line) for line in panel_path.read_text().splitlines() if line.strip()]
    battle_ids = set()
    root_ids = set()
    world_count = 0
    for row in rows:
        if row.get("schema") != PANEL_SCHEMA or battle_split(row["battle_id"]) != "train":
            raise RuntimeError("training panel contains an invalid schema or non-training battle")
        if row["battle_id"] in battle_ids or row["root_id"] in root_ids:
            raise RuntimeError("training panel contains duplicate battle/root identities")
        battle_ids.add(row["battle_id"])
        root_ids.add(row["root_id"])
        if len(row.get("public_features", [])) != 18:
            raise RuntimeError("training panel public-feature contract changed")
        if row.get("causal_history", {}).get("authority") != "schema6_snapshot":
            raise RuntimeError("training panel contains a non-schema6 history authority")
        schedules = row.get("schedules")
        if not isinstance(schedules, list) or sorted(item.get("schedule_id") for item in schedules) != [0, 1]:
            raise RuntimeError("training panel schedule contract changed")
        for schedule in schedules:
            worlds = schedule.get("worlds")
            if not isinstance(worlds, list) or sorted(item.get("world_index") for item in worlds) != list(range(8)):
                raise RuntimeError("training panel world contract changed")
            if not math.isclose(math.fsum(float(item["weight"]) for item in worlds), 1.0):
                raise RuntimeError("training panel world weights do not sum to one")
            for world in worlds:
                if hashlib.sha256(world["state"].encode()).hexdigest() != world["state_sha256"]:
                    raise RuntimeError("training panel world state hash mismatch")
                world_count += 1
    audit = {
        "schema": "metagross-schema6-training-panel-audit/v1",
        "admitted": True,
        "run_id": run_id,
        "panel_seed": panel_seed,
        "purpose": "training",
        "split": "train",
        "battles": len(rows),
        "schedules": len(rows) * 2,
        "worlds": world_count,
        "unique_battles": len(battle_ids),
        "unique_roots": len(root_ids),
        "panel_sha256": _sha256(panel_path),
        "report_sha256": _sha256(report_path),
        "reported_panel_sha256": report.get("panel_sha256"),
        "confirmation_rows_materialized": 0,
        "calibration_rows_materialized": 0,
        "withheld_history_policy_feature_rows_processed": 0,
        "source_physical_battles_by_split": source_inventory,
    }
    if (
        audit["battles"] < 50
        or audit["panel_sha256"] != audit["reported_panel_sha256"]
        or report.get("battles") != audit["battles"]
        or report.get("schedules") != audit["schedules"]
        or report.get("worlds") != audit["worlds"]
        or report.get("split_contract", {}).get("observed_panel_splits")
        != {"train": audit["battles"]}
    ):
        raise RuntimeError("training panel report does not match independent audit")
    _atomic_json(destination / "training-panel-audit.json", audit)
    VOLUME.commit()
    return audit


def _screen_destination(run_id: str, screen_id: str) -> Path:
    return PERSIST_ROOT / _safe_run_id(run_id) / "screens" / _safe_run_id(screen_id)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@APP.function(
    image=IMAGE,
    cpu=(16.0, 16.0),
    memory=(12288, 12288),
    timeout=3600,
    retries=2,
    volumes={"/data": VOLUME},
)
def run_search_screen_shard(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute frozen 20k and 50k search rows for one training-panel shard."""
    import multiprocessing as mp

    import poke_engine
    from scripts.collect_shallow_search_statistics import _task as shallow_task
    from scripts.run_public_mcts_leaf_gate import _load_panel, _oracle_task
    from train.shallow_search_residual import battle_split

    run_id = _safe_run_id(str(payload["run_id"]))
    screen_id = _safe_run_id(str(payload["screen_id"]))
    panel_seed = int(payload["panel_seed"])
    shard_index = int(payload["shard_index"])
    shard_count = int(payload["shard_count"])
    workers = int(payload.get("workers", 16))
    if not 0 <= shard_index < shard_count or not 1 <= workers <= 16:
        raise ValueError("invalid search-screen shard contract")
    panel_path = _panel_destination(run_id, panel_seed) / "training-panel.jsonl"
    panel, panel_hash = _load_panel(panel_path)
    if panel_hash != str(payload["panel_sha256"]):
        raise RuntimeError("search-screen source panel hash changed")
    if any(battle_split(row["battle_id"]) != "train" for row in panel):
        raise RuntimeError("search-screen source contains a withheld split")
    selected = panel[shard_index::shard_count]
    if not selected:
        raise RuntimeError("search-screen shard has no roots")
    destination = _screen_destination(run_id, screen_id) / "shards" / f"{shard_index:03d}"
    report_path = destination / "shard-report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if (
            report.get("panel_sha256") != panel_hash
            or report.get("shard_index") != shard_index
            or report.get("shard_count") != shard_count
            or report.get("shallow_iterations_per_world") != 20_000
            or report.get("oracle_iterations_per_world") != 50_000
            or _sha256(destination / "shallow-20k.jsonl") != report.get("shallow_sha256")
            or _sha256(destination / "oracle-50k.jsonl") != report.get("oracle_sha256")
        ):
            raise RuntimeError("persisted search-screen shard identity changed")
        return report

    engine_binary = Path(poke_engine.poke_engine.__file__)
    engine_hash = _sha256(engine_binary)
    if engine_hash != AUDIT_ENGINE_LINUX_SHA256:
        raise RuntimeError(f"search-screen engine hash changed: {engine_hash}")
    schedules = []
    for root in selected:
        for schedule in root["schedules"]:
            pair_id = f"{root['root_id']}:{schedule['schedule_id']}"
            schedules.append({
                **schedule,
                "battle_id": root["battle_id"],
                "root_id": root["root_id"],
                "pair_id": pair_id,
            })
    context = mp.get_context("spawn")
    process_count = min(workers, len(schedules))
    with context.Pool(process_count) as pool:
        shallow_rows = list(pool.imap_unordered(
            shallow_task,
            ((schedule, 20_000, 500, 0.72) for schedule in schedules),
        ))
    oracle_tasks = [({**schedule, "panel_sha256": panel_hash}, 50_000) for schedule in schedules]
    with context.Pool(process_count) as pool:
        oracle_rows = list(pool.imap_unordered(_oracle_task, oracle_tasks))
    shallow_rows.sort(key=lambda row: row["pair_id"])
    oracle_rows.sort(key=lambda row: row["pair_id"])
    expected_pairs = {schedule["pair_id"] for schedule in schedules}
    if (
        {row["pair_id"] for row in shallow_rows} != expected_pairs
        or {row["pair_id"] for row in oracle_rows} != expected_pairs
    ):
        raise RuntimeError("search-screen shard lost or duplicated a schedule")

    temporary = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        shallow_path = temporary / "shallow-20k.jsonl"
        oracle_path = temporary / "oracle-50k.jsonl"
        _atomic_jsonl(shallow_path, shallow_rows)
        _atomic_jsonl(oracle_path, oracle_rows)
        report = {
            "schema": "metagross-schema6-search-screen-shard/v1",
            "run_id": run_id,
            "screen_id": screen_id,
            "panel_seed": panel_seed,
            "panel_sha256": panel_hash,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "roots": len(selected),
            "pairs": len(schedules),
            "shallow_iterations_per_world": 20_000,
            "oracle_iterations_per_world": 50_000,
            "worlds_per_schedule": 8,
            "engine_binary_sha256": engine_hash,
            "shallow_sha256": _sha256(shallow_path),
            "oracle_sha256": _sha256(oracle_path),
        }
        _atomic_json(temporary / "shard-report.json", report)
        if destination.exists():
            shutil.rmtree(temporary)
            return json.loads(report_path.read_text())
        os.replace(temporary, destination)
        VOLUME.commit()
        return report
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


@APP.function(
    image=IMAGE,
    cpu=2.0,
    memory=8192,
    timeout=1800,
    volumes={"/data": VOLUME},
)
def merge_search_screen(
    run_id: str,
    panel_seed: int,
    screen_id: str,
    screen_seed: int,
    shard_count: int,
) -> dict[str, Any]:
    """Validate, merge, and apply the frozen four-way agreement gate."""
    from scripts.build_outcome_grounded_panel import build as build_outcome_panel
    from scripts.run_public_mcts_leaf_gate import _load_panel
    from train.shallow_search_residual import battle_split

    run_id = _safe_run_id(run_id)
    screen_id = _safe_run_id(screen_id)
    destination = _screen_destination(run_id, screen_id)
    summary_path = destination / "screen-report.json"
    panel_path = _panel_destination(run_id, panel_seed) / "training-panel.jsonl"
    panel, panel_hash = _load_panel(panel_path)
    if any(battle_split(row["battle_id"]) != "train" for row in panel):
        raise RuntimeError("merge source panel contains a withheld split")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("source_panel_sha256") != panel_hash
            or summary.get("shard_count") != shard_count
            or summary.get("screen_seed") != screen_seed
            or _sha256(destination / "shallow-20k.jsonl") != summary.get("shallow_sha256")
            or _sha256(destination / "oracle-50k.jsonl") != summary.get("oracle_sha256")
            or _sha256(destination / "agreement-panel.jsonl") != summary.get("agreement_panel_sha256")
        ):
            raise RuntimeError("persisted merged search screen changed")
        return summary

    shallow_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    shard_reports = []
    for shard_index in range(shard_count):
        shard = destination / "shards" / f"{shard_index:03d}"
        report = json.loads((shard / "shard-report.json").read_text())
        if (
            report.get("panel_sha256") != panel_hash
            or report.get("shard_index") != shard_index
            or report.get("shard_count") != shard_count
            or report.get("engine_binary_sha256") != AUDIT_ENGINE_LINUX_SHA256
            or _sha256(shard / "shallow-20k.jsonl") != report.get("shallow_sha256")
            or _sha256(shard / "oracle-50k.jsonl") != report.get("oracle_sha256")
        ):
            raise RuntimeError(f"search-screen shard {shard_index} failed merge validation")
        shard_reports.append(report)
        shallow_rows.extend(
            json.loads(line) for line in (shard / "shallow-20k.jsonl").read_text().splitlines() if line
        )
        oracle_rows.extend(
            json.loads(line) for line in (shard / "oracle-50k.jsonl").read_text().splitlines() if line
        )
    expected_pairs = {
        f"{root['root_id']}:{schedule['schedule_id']}"
        for root in panel
        for schedule in root["schedules"]
    }
    if (
        len(shallow_rows) != len(expected_pairs)
        or len(oracle_rows) != len(expected_pairs)
        or {row["pair_id"] for row in shallow_rows} != expected_pairs
        or {row["pair_id"] for row in oracle_rows} != expected_pairs
    ):
        raise RuntimeError("merged search screen is incomplete")
    shallow_rows.sort(key=lambda row: row["pair_id"])
    oracle_rows.sort(key=lambda row: row["pair_id"])

    temporary = destination / f".merge-{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    try:
        shallow_path = temporary / "shallow-20k.jsonl"
        oracle_path = temporary / "oracle-50k.jsonl"
        agreement_path = temporary / "agreement-panel.jsonl"
        agreement_report_path = temporary / "agreement-panel-report.json"
        _atomic_jsonl(shallow_path, shallow_rows)
        _atomic_jsonl(oracle_path, oracle_rows)
        agreement_report = build_outcome_panel(SimpleNamespace(
            panel=panel_path,
            shallow=shallow_path,
            oracle=oracle_path,
            output=agreement_path,
            report=agreement_report_path,
            roots=None,
            seed=screen_seed,
            exclude_panel=[],
            max_candidate_actions=3,
        ))
        eligible_roots = int(agreement_report["eligible_roots"])
        summary = {
            "schema": "metagross-schema6-20k-50k-screen/v1",
            "admitted": True,
            "scale_gate_admitted": eligible_roots >= 50,
            "scale_gate_minimum_roots": 50,
            "run_id": run_id,
            "screen_id": screen_id,
            "panel_seed": panel_seed,
            "screen_seed": screen_seed,
            "source_panel_sha256": panel_hash,
            "source_roots": len(panel),
            "source_schedules": len(expected_pairs),
            "source_worlds": len(expected_pairs) * 8,
            "shard_count": shard_count,
            "shallow_iterations_per_world": 20_000,
            "oracle_iterations_per_world": 50_000,
            "shallow_total_iterations": len(expected_pairs) * 8 * 20_000,
            "oracle_total_iterations": len(expected_pairs) * 8 * 50_000,
            "eligible_roots": eligible_roots,
            "attrition": agreement_report["attrition"],
            "engine_binary_sha256": AUDIT_ENGINE_LINUX_SHA256,
            "shallow_sha256": _sha256(shallow_path),
            "oracle_sha256": _sha256(oracle_path),
            "agreement_panel_sha256": _sha256(agreement_path),
            "agreement_report_sha256": _sha256(agreement_report_path),
            "withheld_roots_processed": 0,
            "confirmation_rows_materialized": 0,
            "calibration_rows_materialized": 0,
            "shards": shard_reports,
        }
        _atomic_json(temporary / "screen-report.json", summary)
        for name in (
            "shallow-20k.jsonl",
            "oracle-50k.jsonl",
            "agreement-panel.jsonl",
            "agreement-panel-report.json",
            "screen-report.json",
        ):
            os.replace(temporary / name, destination / name)
        VOLUME.commit()
        return summary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


@APP.local_entrypoint()
def main(
    run_id: str = "schema6-modal-500-20260814",
    seed: int = 2026081400,
    games: int = 500,
    out: str = "",
    freeze_panel: bool = False,
    panel_seed: int = 20260826,
    panel_out: str = "",
    screen_panel: bool = False,
    screen_id: str = "schema6-train-20k50k-20260826-r1",
    screen_seed: int = 20260827,
    screen_shards: int = 15,
    screen_out: str = "",
    authorization_run_id: str = "schema6-modal-500-20260814-r1",
    authorization_screen_id: str = "schema6-train-20k50k-20260826-r1",
    authorization_panel_seed: int = 20260826,
    authorization_screen_seed: int = 20260827,
    authorization_screen_shards: int = 15,
) -> None:
    _safe_run_id(run_id)
    if freeze_panel and screen_panel:
        raise ValueError("freeze_panel and screen_panel are separate atomic operations")
    if screen_panel:
        _safe_run_id(screen_id)
        if not 1 <= screen_shards <= 30:
            raise ValueError("screen_shards must be between 1 and 30")
        audit = validate_training_panel.remote(run_id, panel_seed)
        payloads = [
            {
                "run_id": run_id,
                "screen_id": screen_id,
                "panel_seed": panel_seed,
                "panel_sha256": audit["panel_sha256"],
                "shard_index": shard_index,
                "shard_count": screen_shards,
                "workers": 16,
            }
            for shard_index in range(screen_shards)
        ]
        print(json.dumps({
            "phase": "20k-50k-screen-fanout",
            "run_id": run_id,
            "screen_id": screen_id,
            "source_roots": audit["battles"],
            "shards": screen_shards,
        }, sort_keys=True), flush=True)
        shard_reports = list(run_search_screen_shard.map(payloads, order_outputs=False))
        if len(shard_reports) != screen_shards:
            raise RuntimeError("search-screen fanout did not return every shard")
        summary = merge_search_screen.remote(
            run_id,
            panel_seed,
            screen_id,
            screen_seed,
            screen_shards,
        )
        if screen_out:
            local_destination = Path(screen_out).expanduser().resolve()
            local_destination.parent.mkdir(parents=True, exist_ok=True)
            local_destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if freeze_panel:
        report = freeze_training_panel.remote(run_id, panel_seed)
        audit = validate_training_panel.remote(run_id, panel_seed)
        payload = {"report": report, "audit": audit}
        if panel_out:
            destination = Path(panel_out).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if games not in {500, 5000}:
        raise ValueError("capture stage must be exactly 500 or 5000 games")
    expected_units = PILOT_UNITS if games == 500 else SCALE_5000_UNITS
    authorization = None
    if games == 5000:
        authorization = merge_search_screen.remote(
            _safe_run_id(authorization_run_id),
            authorization_panel_seed,
            _safe_run_id(authorization_screen_id),
            authorization_screen_seed,
            authorization_screen_shards,
        )
        if (
            authorization.get("admitted") is not True
            or authorization.get("scale_gate_admitted") is not True
            or int(authorization.get("eligible_roots", 0)) < 50
            or authorization.get("withheld_roots_processed") != 0
        ):
            raise RuntimeError("5,000-game stage lacks an admitted training-only search-yield gate")
        authorization = {
            "schema": authorization["schema"],
            "run_id": authorization["run_id"],
            "screen_id": authorization["screen_id"],
            "source_panel_sha256": authorization["source_panel_sha256"],
            "agreement_panel_sha256": authorization["agreement_panel_sha256"],
            "eligible_roots": authorization["eligible_roots"],
            "scale_gate_minimum_roots": authorization["scale_gate_minimum_roots"],
            "withheld_roots_processed": authorization["withheld_roots_processed"],
        }
    payloads = _payloads(run_id, seed, expected_units)
    certification_keys = {("peer", 0), ("direct_r1", 0), ("unguided", 0)}
    certification = [
        payload for payload in payloads
        if (payload["profile"], payload["unit_index"]) in certification_keys
    ]
    print(json.dumps({
        "phase": "certification",
        "run_id": run_id,
        "stage_games": games,
        "authorization": authorization,
        "units": certification,
    }, sort_keys=True), flush=True)
    certified_rows = list(run_unit.map(certification, order_outputs=False))
    certified_fingerprints = {
        json.dumps(row["runtime_fingerprint"], sort_keys=True) for row in certified_rows
    }
    if len(certified_rows) != 3 or len(certified_fingerprints) != 1 or not all(row["admitted"] for row in certified_rows):
        raise RuntimeError("cross-stratum certification failed; full fanout was not launched")
    print(
        json.dumps(
            {
                "phase": "certification-passed",
                "elapsed_seconds": [row["elapsed_seconds"] for row in certified_rows],
                "capture_rates": [row["capture"]["capture_rate"] for row in certified_rows],
                "candidate_rows": [row["bridge"]["candidate_rows"] for row in certified_rows],
                "runtime_fingerprint": certified_rows[0]["runtime_fingerprint"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    remaining = [
        payload for payload in payloads
        if (payload["profile"], payload["unit_index"]) not in certification_keys
    ]
    print(json.dumps({"phase": "full-fanout", "remaining_units": len(remaining)}, sort_keys=True), flush=True)
    rows = certified_rows + list(run_unit.map(remaining, order_outputs=False))
    report = _aggregate(
        run_id,
        rows,
        seed,
        expected_units=expected_units,
        expected_games=games,
        authorization=authorization,
    )
    remote_summary = persist_summary.remote(run_id, report)
    if out:
        destination = Path(out).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**report, "units": f"{len(rows)} unit reports", "remote_summary": remote_summary}, indent=2, sort_keys=True))
    if not report["admitted"]:
        raise RuntimeError(f"{games}-game aggregate failed admission")
