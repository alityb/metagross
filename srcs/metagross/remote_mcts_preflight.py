#!/usr/bin/env python3
"""Fail-closed search, holdout, and shared-root smoke for a remote v6 provider."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.request
import uuid

from srcs.metagross.mcts_contract import (
    ENGINE_CONTRACT,
    ENGINE_SOURCE_SHA256,
    REQUEST_SCHEMA,
    validate_holdout_result_payload,
    validate_loopback_search_url,
    validate_request,
    validate_result_payload,
    validate_shared_root_result_payload,
)


MCTS_PARAMETERS = [
    "state",
    "duration_ms",
    "iterations",
    "threads",
    "s1_priors",
    "s2_priors",
    "c_puct",
]
HOLDOUT_PARAMETERS = [
    "state",
    "baseline_action",
    "candidate_action",
    "rollouts",
    "continuation_iterations",
    "continuation_steps",
    "seed",
    "opponent_priors",
]
SHARED_ROOT_PARAMETERS = [
    "states",
    "particle_weights",
    "iterations",
    "continuation_iterations",
    "seed",
    "prior_strength",
    "s1_prior",
    "s2_priors",
]


def synthetic_state() -> str:
    import poke_engine

    def pokemon(identifier: str, types: tuple[str, str], moves: list[str]):
        return poke_engine.Pokemon(
            id=identifier,
            level=100,
            types=types,
            base_types=types,
            hp=100,
            maxhp=100,
            attack=100,
            defense=100,
            special_attack=100,
            special_defense=100,
            speed=100,
            status="none",
            moves=[poke_engine.Move(id=move, pp=32) for move in moves],
        )

    return poke_engine.State(
        side_one=poke_engine.Side(
            pokemon=[
                pokemon(
                    "squirtle",
                    ("water", "typeless"),
                    ["watergun", "tackle", "quickattack", "leer"],
                )
            ]
        ),
        side_two=poke_engine.Side(
            pokemon=[
                pokemon(
                    "charmander",
                    ("fire", "typeless"),
                    ["ember", "tackle", "quickattack", "leer"],
                )
            ]
        ),
        weather="none",
        weather_turns_remaining=-1,
        terrain="none",
        terrain_turns_remaining=-1,
        trick_room=False,
        trick_room_turns_remaining=-1,
    ).to_string()


def build_requests(state: str) -> list[dict[str, object]]:
    requests = [
        {
            "schema": REQUEST_SCHEMA,
            "operation": "search",
            "request_id": uuid.uuid4().hex,
            "index": 0,
            "state": state,
            "duration_ms": 250,
            "threads": 1,
            "s1_priors": [["watergun", 0.75], ["tackle", 0.25]],
            "s2_priors": [["ember", 0.75], ["tackle", 0.25]],
            "c_puct": 2.0,
        },
        {
            "schema": REQUEST_SCHEMA,
            "operation": "search",
            "request_id": uuid.uuid4().hex,
            "index": 1,
            "state": state,
            "duration_ms": 500,
            "threads": 1,
            "s1_priors": [["watergun", 0.75], ["tackle", 0.25]],
            "s2_priors": [["ember", 0.75], ["tackle", 0.25]],
            "c_puct": 2.0,
        },
        {
            "schema": REQUEST_SCHEMA,
            "operation": "paired_holdout",
            "request_id": uuid.uuid4().hex,
            "index": 2,
            "state": state,
            "baseline_action": "watergun",
            "candidate_action": "tackle",
            "rollouts": 2,
            "continuation_iterations": 2,
            "continuation_steps": 1,
            "seed": 7,
            "opponent_priors": [["ember", 0.75], ["tackle", 0.25]],
        },
        {
            "schema": REQUEST_SCHEMA,
            "operation": "shared_root",
            "request_id": uuid.uuid4().hex,
            "index": 3,
            "states": [state, state],
            "particle_weights": [0.5, 0.5],
            "iterations": 100,
            "continuation_iterations": 2,
            "seed": 11,
            "prior_strength": 1.0,
            "s1_prior": [["watergun", 0.75], ["tackle", 0.25]],
            "s2_priors": [
                [["ember", 0.75], ["tackle", 0.25]],
                [["ember", 0.75], ["tackle", 0.25]],
            ],
        },
    ]
    for request in requests:
        validate_request(request)
    return requests


def validate_engine_identity(
    engine: object,
    native_sha256: str,
    transport: str,
    instance_type: str | None = None,
) -> dict[str, object]:
    if not isinstance(engine, dict):
        raise RuntimeError("remote engine identity is not an object")
    expected = {
        "contract": ENGINE_CONTRACT,
        "source_sha256": ENGINE_SOURCE_SHA256,
        "distribution_version": "0.0.47",
        "native_sha256": native_sha256,
        "mcts_parameters": MCTS_PARAMETERS,
        "holdout_parameters": HOLDOUT_PARAMETERS,
        "shared_root_parameters": SHARED_ROOT_PARAMETERS,
    }
    for field, value in expected.items():
        if engine.get(field) != value:
            raise RuntimeError(f"remote engine identity mismatch for {field}")
    resources = engine.get("resources")
    if not isinstance(resources, dict):
        raise RuntimeError("remote engine resource identity is missing")
    if transport == "modal":
        expected_resources = {
            "physical_cores": 16.0,
            "vcpus_equivalent": 32,
            "memory_mib": 16384,
            "worker_processes": 16,
            "max_containers": 4,
            "max_world_concurrency": 64,
        }
        if resources != expected_resources:
            raise RuntimeError("remote Modal resource identity mismatch")
    elif (
        resources.get("provider") != "aws_ec2"
        or resources.get("instance_type") != instance_type
        or resources.get("logical_cpus") != 32
        or resources.get("worker_processes") != 16
    ):
        raise RuntimeError("remote AWS resource identity mismatch")
    return engine


def _http_json(url: str, token: str, payload: object | None = None) -> object:
    request = urllib.request.Request(
        url,
        data=(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
            if payload is not None
            else None
        ),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def run_preflight(
    *,
    transport: str,
    native_sha256: str,
    app: str = "metagross-mcts-r1-p16",
    function: str = "search_batch",
    url: str | None = None,
    instance_type: str | None = None,
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{64}", native_sha256) is None:
        raise ValueError("native SHA-256 must be 64 lowercase hexadecimal characters")
    requests = build_requests(synthetic_state())
    if transport == "modal":
        import modal

        identity = modal.Function.from_name(app, "engine_info").remote()
        responses = modal.Function.from_name(app, function).remote(requests)
    elif transport == "http":
        if url is None:
            raise ValueError("HTTP preflight requires a search URL")
        validate_loopback_search_url(url)
        token = os.environ.get("METAGROSS_REMOTE_MCTS_TOKEN", "")
        if len(token) < 32:
            raise ValueError("HTTP preflight requires the remote bearer token")
        health_url = url.removesuffix("/search") + "/health"
        health = _http_json(health_url, token)
        identity = health.get("engine") if isinstance(health, dict) else None
        responses = _http_json(url, token, requests)
    else:
        raise ValueError("unsupported remote transport")
    engine = validate_engine_identity(
        identity, native_sha256, transport, instance_type
    )
    if not isinstance(responses, list) or len(responses) != len(requests):
        raise RuntimeError("remote preflight returned the wrong response count")
    for request, response in zip(requests, responses, strict=True):
        if (
            not isinstance(response, dict)
            or response.get("schema") != REQUEST_SCHEMA
            or response.get("request_id") != request["request_id"]
            or response.get("index") != request["index"]
            or response.get("ok") is not True
            or response.get("engine") != engine
        ):
            raise RuntimeError("remote preflight response correlation failed")
        if request["operation"] == "search":
            validate_result_payload(response.get("result"))
        elif request["operation"] == "paired_holdout":
            validate_holdout_result_payload(
                response.get("result"),
                expected_pairs=request["rollouts"],
                maximum_executed=(
                    2
                    * request["rollouts"]
                    * request["continuation_iterations"]
                    * request["continuation_steps"]
                ),
            )
        else:
            validate_shared_root_result_payload(
                response.get("result"),
                expected_particles=len(request["states"]),
                expected_iterations=request["iterations"],
                expected_continuation_iterations=request[
                    "continuation_iterations"
                ],
                expected_seed=request["seed"],
                expected_prior_strength=request["prior_strength"],
            )
    return {
        "ok": True,
        "preflight_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "schema": REQUEST_SCHEMA,
        "transport": transport,
        "app": app if transport == "modal" else None,
        "function": function,
        "url": url if transport == "http" else None,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "arguments": list(sys.argv[1:]),
        "environment": dict(sorted(os.environ.items())),
        "engine": engine,
        "operations": list(dict.fromkeys(request["operation"] for request in requests)),
        "search_durations_ms": [
            request["duration_ms"]
            for request in requests
            if request["operation"] == "search"
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("modal", "http"), required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--app", default="metagross-mcts-r1-p16")
    parser.add_argument("--function", default="search_batch")
    parser.add_argument("--url")
    parser.add_argument("--instance-type")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_preflight(
        transport=args.transport,
        native_sha256=args.native_sha256,
        app=args.app,
        function=args.function,
        url=args.url,
        instance_type=args.instance_type,
    )
    if args.output:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(output)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
