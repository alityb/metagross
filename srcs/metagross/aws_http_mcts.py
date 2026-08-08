#!/usr/bin/env python3
"""Serve pinned poke-engine MCTS batches over authenticated loopback HTTP."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from srcs.metagross.mcts_contract import (
    MAX_WIRE_BATCH_SIZE,
    REQUEST_SCHEMA,
    engine_identity,
    holdout_result_payload,
    result_payload,
    validate_request,
)


WORKER_PROCESSES = 16
MAX_REQUEST_BYTES = 70_000_000
MIN_TOKEN_LENGTH = 32


class RequestError(Exception):
    def __init__(self, status: int, kind: str):
        super().__init__(kind)
        self.status = status
        self.kind = kind


def _memory_mib() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**2)
    except (AttributeError, OSError, ValueError):
        return None


def aws_resources(instance_type: str) -> dict[str, object]:
    return {
        "provider": "aws_ec2",
        "instance_type": instance_type,
        "logical_cpus": os.cpu_count(),
        "memory_mib": _memory_mib(),
        "worker_processes": WORKER_PROCESSES,
    }


def authorized(authorization: str | None, token: str) -> bool:
    return authorization is not None and hmac.compare_digest(
        authorization, f"Bearer {token}"
    )


def decode_batch(body: bytes) -> list[dict[str, object]]:
    if not body or len(body) > MAX_REQUEST_BYTES:
        raise RequestError(413, "invalid_body")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RequestError(400, "invalid_json") from exc
    if not isinstance(payload, list) or not 0 < len(payload) <= MAX_WIRE_BATCH_SIZE:
        raise RequestError(400, "invalid_batch")
    return payload


def _search_one(
    request: object,
    batch_size: int,
    identity: dict[str, object],
    submitted: float,
) -> dict[str, object]:
    worker_started = time.monotonic()
    request_id = request.get("request_id") if isinstance(request, dict) else None
    index = request.get("index") if isinstance(request, dict) else None
    base = {
        "schema": REQUEST_SCHEMA,
        "request_id": request_id,
        "index": index,
        "engine": identity,
    }
    try:
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        validated = validate_request(request)
    except Exception as exc:
        validation_finished = time.monotonic()
        search_finished = validation_finished
        payload = {"ok": False, "error": {"kind": type(exc).__name__}}
    else:
        validation_finished = time.monotonic()
        try:
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
                result = result_payload(result)
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
                result = holdout_result_payload(
                    result,
                    expected_pairs=validated["rollouts"],
                    maximum_executed=(
                        2
                        * validated["rollouts"]
                        * validated["continuation_iterations"]
                        * validated["continuation_steps"]
                    ),
                )
            search_finished = time.monotonic()
            payload = {"ok": True, "result": result}
        except Exception as exc:
            search_finished = time.monotonic()
            payload = {"ok": False, "error": {"kind": type(exc).__name__}}
    return {
        **base,
        **payload,
        "timing": {
            "queue_ms": round((worker_started - submitted) * 1000, 3),
            "validation_ms": round((validation_finished - worker_started) * 1000, 3),
            "search_ms": round((search_finished - validation_finished) * 1000, 3),
            "worker_ms": round((search_finished - worker_started) * 1000, 3),
            "batch_size": batch_size,
        },
    }


class MctsService:
    def __init__(
        self,
        token: str,
        instance_type: str,
        pool: ProcessPoolExecutor | None = None,
    ) -> None:
        if len(token) < MIN_TOKEN_LENGTH:
            raise RuntimeError(
                f"METAGROSS_REMOTE_MCTS_TOKEN must contain at least {MIN_TOKEN_LENGTH} characters"
            )
        if not instance_type:
            raise RuntimeError("AWS instance type is required")
        self.token = token
        self.identity = engine_identity(aws_resources(instance_type))
        self.pool = pool or ProcessPoolExecutor(max_workers=WORKER_PROCESSES)

    def search(self, body: bytes) -> list[dict[str, object]]:
        requests = decode_batch(body)
        started = time.monotonic()
        submitted = time.monotonic()
        futures = [
            self.pool.submit(
                _search_one, request, len(requests), self.identity, submitted
            )
            for request in requests
        ]
        responses = [future.result() for future in futures]
        batch_ms = round((time.monotonic() - started) * 1000, 3)
        for response in responses:
            response["timing"]["batch_ms"] = batch_ms
        return responses


class MctsHandler(BaseHTTPRequestHandler):
    server_version = "metagross-mcts"
    sys_version = ""

    @property
    def service(self) -> MctsService:
        return self.server.service

    def _send(self, status: int, payload: object) -> None:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _require_auth(self) -> None:
        if not authorized(self.headers.get("Authorization"), self.service.token):
            raise RequestError(401, "unauthorized")

    def do_GET(self) -> None:
        try:
            self._require_auth()
            if self.path != "/health":
                raise RequestError(404, "not_found")
            self._send(
                200,
                {"ok": True, "schema": REQUEST_SCHEMA, "engine": self.service.identity},
            )
        except RequestError as exc:
            self._send(exc.status, {"ok": False, "error": {"kind": exc.kind}})

    def do_POST(self) -> None:
        try:
            self._require_auth()
            if self.path != "/search":
                raise RequestError(404, "not_found")
            if self.headers.get("Transfer-Encoding"):
                raise RequestError(400, "invalid_body")
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != "application/json":
                raise RequestError(415, "invalid_content_type")
            try:
                content_length = int(self.headers.get("Content-Length", ""))
            except ValueError as exc:
                raise RequestError(411, "invalid_content_length") from exc
            if not 0 < content_length <= MAX_REQUEST_BYTES:
                raise RequestError(413, "invalid_body")
            body = self.rfile.read(content_length)
            if len(body) != content_length:
                raise RequestError(400, "invalid_body")
            self._send(200, self.service.search(body))
        except RequestError as exc:
            self._send(exc.status, {"ok": False, "error": {"kind": exc.kind}})
        except Exception:
            self._send(500, {"ok": False, "error": {"kind": "internal_error"}})

    def log_message(self, _format: str, *_args: object) -> None:
        return


class MctsHttpServer(ThreadingHTTPServer):
    service: MctsService


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--instance-type", default=os.environ.get("METAGROSS_AWS_INSTANCE_TYPE")
    )
    args = parser.parse_args(argv)
    if not 0 < args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not args.instance_type:
        parser.error("--instance-type or METAGROSS_AWS_INSTANCE_TYPE is required")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    token = os.environ.get("METAGROSS_REMOTE_MCTS_TOKEN")
    if not token:
        raise RuntimeError("METAGROSS_REMOTE_MCTS_TOKEN is required")
    service = MctsService(token, args.instance_type)
    server = MctsHttpServer((args.host, args.port), MctsHandler)
    server.service = service
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
