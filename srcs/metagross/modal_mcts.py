"""Dynamically batch pinned poke-engine MCTS worlds on Modal CPUs."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import modal

from srcs.metagross.mcts_contract import (
    ENGINE_SOURCE_SHA256,
    MODAL_CONTAINER_BATCH_SIZE,
    MODAL_MAX_CONTAINERS,
    MODAL_MAX_WORLD_CONCURRENCY,
    REQUEST_SCHEMA,
    engine_identity,
    holdout_result_payload,
    result_payload,
    validate_priors,
    validate_request,
)


APP_NAME = "metagross-mcts-r1-p16"
FUNCTION_NAME = "search_batch"
MAX_BATCH_SIZE = MODAL_CONTAINER_BATCH_SIZE
MAX_CONTAINERS = MODAL_MAX_CONTAINERS
MAX_WORLD_CONCURRENCY = MODAL_MAX_WORLD_CONCURRENCY
CLOUD_PHYSICAL_CORES = 16.0
CLOUD_MEMORY_MIB = 16384
CLOUD_RESOURCES = {
    "physical_cores": CLOUD_PHYSICAL_CORES,
    "vcpus_equivalent": int(CLOUD_PHYSICAL_CORES * 2),
    "memory_mib": CLOUD_MEMORY_MIB,
    "worker_processes": MAX_BATCH_SIZE,
    "max_containers": MAX_CONTAINERS,
    "max_world_concurrency": MAX_WORLD_CONCURRENCY,
}
APP = modal.App(APP_NAME)
app = APP
SCRIPT_PATH = Path(__file__).resolve()
IS_LOCAL_CHECKOUT = (
    len(SCRIPT_PATH.parents) > 2 and (SCRIPT_PATH.parents[2] / "srcs").is_dir()
)
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
        image.add_local_file(
            ROOT / "srcs" / "metagross" / "__init__.py",
            "/root/srcs/metagross/__init__.py",
            copy=True,
        )
        .add_local_file(
            ROOT / "srcs" / "metagross" / "mcts_contract.py",
            "/root/srcs/metagross/mcts_contract.py",
            copy=True,
        )
        .add_local_dir(
            ROOT / "srcs" / "vendor" / "poke-engine",
            "/opt/poke-engine",
            copy=True,
            ignore=[
                ".git",
                "__pycache__",
                "*.pyc",
                "linux_wheels",
                "release",
                "target",
            ],
        )
        .run_commands(
            "PATH=/root/.cargo/bin:$PATH sh -c \"curl --proto '=https' --tlsv1.2 -sSf "
            'https://sh.rustup.rs | sh -s -- -y --profile minimal"',
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
    _ENGINE_IDENTITY = engine_identity(CLOUD_RESOURCES)
    if _ENGINE_IDENTITY["source_sha256"] != ENGINE_SOURCE_SHA256:
        raise RuntimeError("remote engine source identity mismatch")
    return _ENGINE_IDENTITY


def _validate_priors(value: object, label: str) -> list[tuple[str, float]] | None:
    return validate_priors(value, label)


def _result_payload(result) -> dict[str, object]:
    return result_payload(result)


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
        validated = validate_request(request)

        import poke_engine

        state = poke_engine.State.from_string(validated["state"])
        if validated["operation"] == "search":
            result = poke_engine.monte_carlo_tree_search(
                state,
                validated["duration_ms"],
                threads=validated["threads"],
                s1_priors=validated["s1_priors"],
                s2_priors=validated["s2_priors"],
                c_puct=validated["c_puct"],
            )
            payload = _result_payload(result)
        else:
            result = poke_engine.paired_root_policy_evaluation(
                state,
                validated["baseline_action"],
                validated["candidate_action"],
                validated["rollouts"],
                validated["continuation_iterations"],
                validated["continuation_steps"],
                validated["seed"],
                validated["opponent_priors"],
            )
            payload = holdout_result_payload(
                result,
                expected_pairs=validated["rollouts"],
                maximum_executed=(
                    2
                    * validated["rollouts"]
                    * validated["continuation_iterations"]
                    * validated["continuation_steps"]
                ),
            )
        return {
            **base,
            "ok": True,
            "result": payload,
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
    max_containers=MAX_CONTAINERS,
    scaledown_window=300,
)
@modal.batched(max_batch_size=MAX_BATCH_SIZE, wait_ms=200)
def search_world(requests: list[dict[str, object]]) -> list[dict[str, object]]:
    batch_size = len(requests)
    futures = [
        _search_pool().submit(_search_one, request, batch_size) for request in requests
    ]
    return [future.result() for future in futures]


@APP.function(
    image=IMAGE,
    cpu=(CLOUD_PHYSICAL_CORES, CLOUD_PHYSICAL_CORES),
    memory=(CLOUD_MEMORY_MIB, CLOUD_MEMORY_MIB),
    timeout=120,
    max_containers=MAX_CONTAINERS,
    scaledown_window=300,
)
def search_batch(requests: list[dict[str, object]]) -> list[dict[str, object]]:
    if not isinstance(requests, list) or not 0 < len(requests) <= 64:
        raise ValueError("search batch must contain between 1 and 64 worlds")
    batch_size = len(requests)
    futures = [
        _search_pool().submit(_search_one, request, batch_size) for request in requests
    ]
    return [future.result() for future in futures]


@APP.function(image=IMAGE, cpu=1.0, memory=512, timeout=60)
def engine_info() -> dict[str, object]:
    return _engine_identity()
