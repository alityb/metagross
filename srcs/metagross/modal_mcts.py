"""Dynamically batch pinned poke-engine MCTS worlds on Modal CPUs."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import modal


APP_NAME = "metagross-mcts-r1-p16"
FUNCTION_NAME = "search_batch"
ENGINE_CONTRACT = "poke-engine-0.0.47-priors-v2"
ENGINE_SOURCE_SHA256 = "90bb051bff687c12bc74e2161b474d9f583b66d7bef73752150c08707a7a7180"
REQUEST_SCHEMA = 1
MAX_BATCH_SIZE = 16
CLOUD_PHYSICAL_CORES = 16.0
CLOUD_MEMORY_MIB = 16384
CLOUD_RESOURCES = {
    "physical_cores": CLOUD_PHYSICAL_CORES,
    "vcpus_equivalent": int(CLOUD_PHYSICAL_CORES * 2),
    "memory_mib": CLOUD_MEMORY_MIB,
    "worker_processes": MAX_BATCH_SIZE,
}
APP = modal.App(APP_NAME)
app = APP
SCRIPT_PATH = Path(__file__).resolve()
IS_LOCAL_CHECKOUT = len(SCRIPT_PATH.parents) > 2 and (
    SCRIPT_PATH.parents[2] / "srcs"
).is_dir()
ROOT = SCRIPT_PATH.parents[2] if IS_LOCAL_CHECKOUT else Path("/workspace")


def _build_image() -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("build-essential", "curl")
        .pip_install("maturin>=1.0,<2.0")
    )
    if not IS_LOCAL_CHECKOUT:
        return image
    return (
        image
        .add_local_dir(
            ROOT / "srcs" / "vendor" / "poke-engine",
            "/opt/poke-engine",
            copy=True,
            ignore=[".git", "__pycache__", "*.pyc", "linux_wheels", "release", "target"],
        )
        .run_commands(
            "PATH=/root/.cargo/bin:$PATH sh -c \"curl --proto '=https' --tlsv1.2 -sSf "
            "https://sh.rustup.rs | sh -s -- -y --profile minimal\"",
            "PATH=/root/.cargo/bin:$PATH CARGO_TARGET_DIR=/tmp/poke-engine "
            "python -m pip install --no-cache-dir /opt/poke-engine "
            "--config-settings='build-args=--no-default-features "
            "--features poke-engine/gen9,poke-engine/terastallization'",
            "rm -rf /tmp/poke-engine",
        )
    )


IMAGE = _build_image()
_ENGINE_IDENTITY: dict[str, object] | None = None
_SEARCH_POOL: ProcessPoolExecutor | None = None


def _engine_identity() -> dict[str, object]:
    global _ENGINE_IDENTITY
    if _ENGINE_IDENTITY is not None:
        return _ENGINE_IDENTITY
    import poke_engine

    native = Path(importlib.import_module("poke_engine.poke_engine").__file__).resolve()
    parameters = list(inspect.signature(poke_engine.monte_carlo_tree_search).parameters)
    required = {"state", "duration_ms", "threads", "s1_priors", "s2_priors", "c_puct"}
    if required - set(parameters) or "seed" in parameters:
        raise RuntimeError("remote poke-engine has an invalid MCTS contract")
    _ENGINE_IDENTITY = {
        "contract": ENGINE_CONTRACT,
        "source_sha256": ENGINE_SOURCE_SHA256,
        "distribution_version": importlib.metadata.version("poke_engine"),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "mcts_parameters": parameters,
        "resources": CLOUD_RESOURCES,
    }
    return _ENGINE_IDENTITY


def _validate_priors(value: object, label: str) -> list[tuple[str, float]] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) > 64:
        raise ValueError(f"{label} must be a bounded list")
    priors: list[tuple[str, float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"{label} entries must contain a move and probability")
        move, probability = row
        probability = float(probability)
        if not isinstance(move, str) or not move or not math.isfinite(probability):
            raise ValueError(f"{label} contains an invalid entry")
        priors.append((move, probability))
    return priors


def _result_payload(result) -> dict[str, object]:
    def side(options) -> list[dict[str, object]]:
        return [
            {
                "move_choice": option.move_choice,
                "total_score": float(option.total_score),
                "visits": int(option.visits),
            }
            for option in options
        ]

    return {
        "side_one": side(result.side_one),
        "side_two": side(result.side_two),
        "total_visits": int(result.total_visits),
    }


def _search_one(request: dict[str, object], batch_size: int) -> dict[str, object]:
    started = time.monotonic()
    request_id = request.get("request_id")
    index = request.get("index")
    base = {
        "schema": REQUEST_SCHEMA,
        "request_id": request_id,
        "index": index,
        "engine": _engine_identity(),
    }
    try:
        if request.get("schema") != REQUEST_SCHEMA:
            raise ValueError("unsupported request schema")
        state_string = request.get("state")
        duration_ms = request.get("duration_ms")
        threads = request.get("threads")
        c_puct = request.get("c_puct")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("invalid request ID")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("invalid world index")
        if not isinstance(state_string, str) or not 0 < len(state_string) <= 1_000_000:
            raise ValueError("invalid state string")
        if duration_ms not in {250, 500}:
            raise ValueError("duration must be 250 or 500 ms")
        if threads != 1:
            raise ValueError("remote search requires one thread")
        if float(c_puct) != 2.0:
            raise ValueError("remote search requires c_puct=2.0")

        import poke_engine

        result = poke_engine.monte_carlo_tree_search(
            poke_engine.State.from_string(state_string),
            duration_ms,
            threads=threads,
            s1_priors=_validate_priors(request.get("s1_priors"), "s1_priors"),
            s2_priors=_validate_priors(request.get("s2_priors"), "s2_priors"),
            c_puct=float(c_puct),
        )
        return {
            **base,
            "ok": True,
            "result": _result_payload(result),
            "timing": {
                "search_ms": round((time.monotonic() - started) * 1000, 3),
                "batch_size": batch_size,
            },
        }
    except Exception as exc:
        return {
            **base,
            "ok": False,
            "error": {"kind": type(exc).__name__},
            "timing": {
                "search_ms": round((time.monotonic() - started) * 1000, 3),
                "batch_size": batch_size,
            },
        }


def _search_pool() -> ProcessPoolExecutor:
    global _SEARCH_POOL
    if _SEARCH_POOL is None:
        _SEARCH_POOL = ProcessPoolExecutor(max_workers=MAX_BATCH_SIZE)
    return _SEARCH_POOL


@APP.function(
    image=IMAGE,
    cpu=(CLOUD_PHYSICAL_CORES, CLOUD_PHYSICAL_CORES),
    memory=(CLOUD_MEMORY_MIB, CLOUD_MEMORY_MIB),
    timeout=120,
    max_containers=1,
    scaledown_window=300,
)
@modal.batched(max_batch_size=MAX_BATCH_SIZE, wait_ms=200)
def search_world(requests: list[dict[str, object]]) -> list[dict[str, object]]:
    batch_size = len(requests)
    futures = [_search_pool().submit(_search_one, request, batch_size) for request in requests]
    return [future.result() for future in futures]


@APP.function(
    image=IMAGE,
    cpu=(CLOUD_PHYSICAL_CORES, CLOUD_PHYSICAL_CORES),
    memory=(CLOUD_MEMORY_MIB, CLOUD_MEMORY_MIB),
    timeout=120,
    max_containers=1,
    scaledown_window=300,
)
def search_batch(requests: list[dict[str, object]]) -> list[dict[str, object]]:
    if not isinstance(requests, list) or not 0 < len(requests) <= 64:
        raise ValueError("search batch must contain between 1 and 64 worlds")
    batch_size = len(requests)
    futures = [_search_pool().submit(_search_one, request, batch_size) for request in requests]
    return [future.result() for future in futures]


@APP.function(image=IMAGE, cpu=1.0, memory=512, timeout=60)
def engine_info() -> dict[str, object]:
    return _engine_identity()
