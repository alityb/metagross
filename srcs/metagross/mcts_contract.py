"""Shared wire contract for production remote MCTS providers."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import ipaddress
import math
from pathlib import Path
from urllib.parse import urlparse


ENGINE_CONTRACT = "poke-engine-0.0.47-holdout-v5"
ENGINE_SOURCE_SHA256 = (
    "5621ffdb59e1a460873b715ef0b6cf861aee09b77e1c90a676a68a3f3f0bfe37"
)
REQUEST_SCHEMA = 3
MAX_WIRE_BATCH_SIZE = 64
MODAL_CONTAINER_BATCH_SIZE = 16
MODAL_MAX_CONTAINERS = 4
MODAL_MAX_WORLD_CONCURRENCY = MODAL_CONTAINER_BATCH_SIZE * MODAL_MAX_CONTAINERS
HOLDOUT_RESULT_FIELDS = {
    "pairs",
    "baseline_sum",
    "candidate_sum",
    "delta_sum",
    "delta_squared_sum",
    "catastrophic_count",
    "candidate_catastrophic_count",
    "baseline_catastrophic_count",
    "candidate_catastrophic_severity_sum",
    "baseline_catastrophic_severity_sum",
    "candidate_better_count",
    "baseline_better_count",
    "equal_count",
    "baseline_terminal_count",
    "candidate_terminal_count",
    "baseline_nonterminal_evaluation_delta_sum",
    "candidate_nonterminal_evaluation_delta_sum",
    "baseline_nonterminal_count",
    "candidate_nonterminal_count",
    "continuation_iterations_executed",
}


def compute_engine_source_sha256(root: Path) -> str:
    """Hash the production Rust/Python source contract deterministically."""
    files = {
        root / "Cargo.lock",
        root / "Cargo.toml",
        root / "pyproject.toml",
        root / "poke-engine-py" / "Cargo.toml",
    }
    files.update((root / "src").rglob("*.rs"))
    files.update((root / "poke-engine-py" / "src").rglob("*.rs"))
    for package in (
        root / "python" / "poke_engine",
        root / "poke-engine-py" / "python" / "poke_engine",
    ):
        files.update(
            path for path in package.iterdir() if path.suffix in {".py", ".pyi"}
        )
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def engine_identity(resources: dict[str, object]) -> dict[str, object]:
    import poke_engine

    native = Path(importlib.import_module("poke_engine.poke_engine").__file__).resolve()
    parameters = list(inspect.signature(poke_engine.monte_carlo_tree_search).parameters)
    if parameters != [
        "state",
        "duration_ms",
        "iterations",
        "threads",
        "s1_priors",
        "s2_priors",
        "c_puct",
    ]:
        raise RuntimeError("remote poke-engine has an invalid MCTS contract")
    holdout_parameters = list(
        inspect.signature(poke_engine.paired_root_policy_evaluation).parameters
    )
    if holdout_parameters != [
        "state",
        "baseline_action",
        "candidate_action",
        "rollouts",
        "continuation_iterations",
        "continuation_steps",
        "seed",
        "opponent_priors",
    ]:
        raise RuntimeError("remote poke-engine has an invalid holdout contract")
    return {
        "contract": ENGINE_CONTRACT,
        "source_sha256": ENGINE_SOURCE_SHA256,
        "distribution_version": importlib.metadata.version("poke_engine"),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "mcts_parameters": parameters,
        "holdout_parameters": holdout_parameters,
        "resources": resources,
    }


def validate_priors(value: object, label: str) -> list[tuple[str, float]] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) > 64:
        raise ValueError(f"{label} must be a bounded list")
    priors: list[tuple[str, float]] = []
    moves: set[str] = set()
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"{label} entries must contain a move and probability")
        move, probability = row
        if (
            not isinstance(move, str)
            or not move
            or move in moves
            or isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError(f"{label} contains an invalid entry")
        moves.add(move)
        priors.append((move, float(probability)))
    total = math.fsum(probability for _move, probability in priors)
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"{label} must contain positive finite probability mass")
    return priors


def validate_request(request: dict[str, object]) -> dict[str, object]:
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unsupported request schema")
    request_id = request.get("request_id")
    index = request.get("index")
    state_string = request.get("state")
    operation = request.get("operation")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise ValueError("invalid request ID")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("invalid world index")
    if not isinstance(state_string, str) or not 0 < len(state_string) <= 1_000_000:
        raise ValueError("invalid state string")
    common = {"operation": operation, "state": state_string}
    if operation == "paired_holdout":
        allowed_fields = {
            "schema",
            "request_id",
            "index",
            "operation",
            "state",
            "baseline_action",
            "candidate_action",
            "rollouts",
            "continuation_iterations",
            "continuation_steps",
            "seed",
            "opponent_priors",
        }
        if set(request) - allowed_fields:
            raise ValueError("paired holdout request has unknown fields")
        if "opponent_priors" not in request:
            raise ValueError("paired holdout request requires opponent_priors")
        baseline_action = request.get("baseline_action")
        candidate_action = request.get("candidate_action")
        rollouts = request.get("rollouts")
        continuation_iterations = request.get("continuation_iterations")
        continuation_steps = request.get("continuation_steps")
        seed = request.get("seed")
        if not isinstance(baseline_action, str) or not 0 < len(baseline_action) <= 128:
            raise ValueError("invalid baseline action")
        if (
            not isinstance(candidate_action, str)
            or not 0 < len(candidate_action) <= 128
        ):
            raise ValueError("invalid candidate action")
        for name, value, upper in (
            ("rollouts", rollouts, 100_000),
            ("continuation_iterations", continuation_iterations, 10_000_000),
            ("continuation_steps", continuation_steps, 100),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= upper
            ):
                raise ValueError(f"invalid {name}")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= (1 << 64) - 1
        ):
            raise ValueError("invalid holdout seed")
        if 2 * rollouts * continuation_iterations * continuation_steps > 100_000_000:
            raise ValueError("holdout work exceeds the resource bound")
        return {
            **common,
            "baseline_action": baseline_action,
            "candidate_action": candidate_action,
            "rollouts": rollouts,
            "continuation_iterations": continuation_iterations,
            "continuation_steps": continuation_steps,
            "seed": seed,
            "opponent_priors": validate_priors(
                request["opponent_priors"], "opponent_priors"
            ),
        }
    if operation != "search":
        raise ValueError("unsupported operation")
    allowed_fields = {
        "schema",
        "request_id",
        "index",
        "operation",
        "state",
        "duration_ms",
        "threads",
        "s1_priors",
        "s2_priors",
        "c_puct",
    }
    if set(request) - allowed_fields:
        raise ValueError("search request has unknown fields")
    duration_ms = request.get("duration_ms")
    threads = request.get("threads")
    c_puct = request.get("c_puct")
    if duration_ms not in {250, 500}:
        raise ValueError("duration must be 250 or 500 ms")
    if threads != 1:
        raise ValueError("remote search requires one thread")
    if float(c_puct) != 2.0:
        raise ValueError("remote search requires c_puct=2.0")
    return {
        **common,
        "duration_ms": duration_ms,
        "threads": threads,
        "s1_priors": validate_priors(request.get("s1_priors"), "s1_priors"),
        "s2_priors": validate_priors(request.get("s2_priors"), "s2_priors"),
        "c_puct": float(c_puct),
    }


def holdout_result_payload(
    result,
    *,
    expected_pairs: int | None = None,
    maximum_executed: int | None = None,
) -> dict[str, object]:
    payload = {name: getattr(result, name) for name in HOLDOUT_RESULT_FIELDS}
    return validate_holdout_result_payload(
        payload,
        expected_pairs=expected_pairs,
        maximum_executed=maximum_executed,
    )


def validate_holdout_result_payload(
    value: object,
    *,
    expected_pairs: int | None = None,
    maximum_executed: int | None = None,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != HOLDOUT_RESULT_FIELDS:
        raise ValueError("holdout result has invalid fields")
    pairs = value["pairs"]
    if (
        isinstance(pairs, bool)
        or not isinstance(pairs, int)
        or not 1 <= pairs <= 100_000
    ):
        raise ValueError("holdout result has invalid pair count")
    if expected_pairs is not None and pairs != expected_pairs:
        raise ValueError("holdout result pair count differs from the request")
    count_fields = HOLDOUT_RESULT_FIELDS - {
        "baseline_sum",
        "candidate_sum",
        "delta_sum",
        "delta_squared_sum",
        "candidate_catastrophic_severity_sum",
        "baseline_catastrophic_severity_sum",
        "baseline_nonterminal_evaluation_delta_sum",
        "candidate_nonterminal_evaluation_delta_sum",
        "continuation_iterations_executed",
    }
    for name in count_fields:
        field = value[name]
        if (
            isinstance(field, bool)
            or not isinstance(field, int)
            or not 0 <= field <= pairs
        ):
            raise ValueError(f"holdout result has invalid {name}")
    executed = value["continuation_iterations_executed"]
    if isinstance(executed, bool) or not isinstance(executed, int) or executed < 0:
        raise ValueError("holdout result has invalid executed iterations")
    execution_bound = 100_000_000 if maximum_executed is None else maximum_executed
    if executed > execution_bound:
        raise ValueError("holdout result exceeds its iteration bound")
    numeric = {}
    for name in (
        "baseline_sum",
        "candidate_sum",
        "delta_sum",
        "delta_squared_sum",
        "candidate_catastrophic_severity_sum",
        "baseline_catastrophic_severity_sum",
        "baseline_nonterminal_evaluation_delta_sum",
        "candidate_nonterminal_evaluation_delta_sum",
    ):
        field = value[name]
        if (
            isinstance(field, bool)
            or not isinstance(field, (int, float))
            or not math.isfinite(field)
        ):
            raise ValueError(f"holdout result has invalid {name}")
        numeric[name] = float(field)
    tolerance = 1e-8 * max(1, pairs)
    if not -tolerance <= numeric["baseline_sum"] <= pairs + tolerance:
        raise ValueError("holdout baseline sum is out of range")
    if not -tolerance <= numeric["candidate_sum"] <= pairs + tolerance:
        raise ValueError("holdout candidate sum is out of range")
    if not -pairs - tolerance <= numeric["delta_sum"] <= pairs + tolerance:
        raise ValueError("holdout delta sum is out of range")
    if not -tolerance <= numeric["delta_squared_sum"] <= pairs + tolerance:
        raise ValueError("holdout squared delta sum is out of range")
    if (
        numeric["delta_squared_sum"] + tolerance
        < numeric["delta_sum"] * numeric["delta_sum"] / pairs
    ):
        raise ValueError("holdout delta moments are inconsistent")
    if (
        abs(numeric["candidate_sum"] - numeric["baseline_sum"] - numeric["delta_sum"])
        > tolerance
    ):
        raise ValueError("holdout sums are inconsistent")
    if (
        value["candidate_better_count"]
        + value["baseline_better_count"]
        + value["equal_count"]
        != pairs
    ):
        raise ValueError("holdout comparison counts are inconsistent")
    if value["catastrophic_count"] != value["candidate_catastrophic_count"]:
        raise ValueError("holdout catastrophic alias is inconsistent")
    if value["candidate_catastrophic_count"] > value["baseline_better_count"]:
        raise ValueError("holdout candidate catastrophe count is inconsistent")
    if value["baseline_catastrophic_count"] > value["candidate_better_count"]:
        raise ValueError("holdout baseline catastrophe count is inconsistent")
    for arm in ("baseline", "candidate"):
        terminal_count = value[f"{arm}_terminal_count"]
        nonterminal_count = value[f"{arm}_nonterminal_count"]
        if terminal_count + nonterminal_count != pairs:
            raise ValueError(f"holdout {arm} terminal partition is inconsistent")
        count = value[f"{arm}_catastrophic_count"]
        severity = numeric[f"{arm}_catastrophic_severity_sum"]
        if not 0.5 * count - tolerance <= severity <= count + tolerance:
            raise ValueError(f"holdout {arm} catastrophe severity is out of range")
    return {**value, **numeric}


def result_payload(result) -> dict[str, object]:
    def side(options, label: str) -> tuple[list[dict[str, object]], int]:
        payload = []
        moves: set[str] = set()
        visit_sum = 0
        for option in options:
            move = option.move_choice
            score = option.total_score
            visits = option.visits
            if not isinstance(move, str) or not move or move in moves:
                raise ValueError(f"MCTS {label} contains an invalid action")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise ValueError(f"MCTS {label} contains an invalid score")
            if isinstance(visits, bool) or not isinstance(visits, int) or visits < 0:
                raise ValueError(f"MCTS {label} contains invalid visits")
            if not 0 <= score <= visits:
                raise ValueError(f"MCTS {label} contains a score outside its visits")
            moves.add(move)
            visit_sum += visits
            payload.append(
                {
                    "move_choice": move,
                    "total_score": float(score),
                    "visits": visits,
                }
            )
        return payload, visit_sum

    total_visits = result.total_visits
    if (
        isinstance(total_visits, bool)
        or not isinstance(total_visits, int)
        or total_visits < 0
    ):
        raise ValueError("MCTS result contains invalid total visits")
    side_one, side_one_visits = side(result.side_one, "side_one")
    side_two, side_two_visits = side(result.side_two, "side_two")
    if side_one_visits != side_two_visits or side_one_visits > total_visits:
        raise ValueError("MCTS result contains inconsistent visit totals")

    return {
        "side_one": side_one,
        "side_two": side_two,
        "total_visits": total_visits,
    }


def validate_result_payload(value: object) -> dict[str, object]:
    """Validate an untrusted wire result using the producer's invariants."""
    if not isinstance(value, dict):
        raise ValueError("MCTS result must be an object")

    def side(name: str) -> tuple[list[dict[str, object]], int]:
        rows = value.get(name)
        if (
            not isinstance(rows, list)
            or len(rows) > 64
            or (name == "side_one" and not rows)
        ):
            raise ValueError(f"MCTS {name} is invalid")
        payload = []
        moves = set()
        visit_sum = 0
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"MCTS {name} contains an invalid option")
            move = row.get("move_choice")
            score = row.get("total_score")
            visits = row.get("visits")
            if (
                not isinstance(move, str)
                or not 0 < len(move) <= 128
                or move in moves
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or isinstance(visits, bool)
                or not isinstance(visits, int)
                or visits < 0
                or not 0 <= score <= visits
            ):
                raise ValueError(f"MCTS {name} contains an invalid option")
            moves.add(move)
            visit_sum += visits
            payload.append(
                {"move_choice": move, "total_score": float(score), "visits": visits}
            )
        return payload, visit_sum

    total_visits = value.get("total_visits")
    if (
        isinstance(total_visits, bool)
        or not isinstance(total_visits, int)
        or total_visits < 0
    ):
        raise ValueError("MCTS result contains invalid total visits")
    side_one, side_one_visits = side("side_one")
    side_two, side_two_visits = side("side_two")
    if side_one_visits != side_two_visits or side_one_visits > total_visits:
        raise ValueError("MCTS result contains inconsistent visit totals")
    return {
        "side_one": side_one,
        "side_two": side_two,
        "total_visits": total_visits,
    }


def validate_loopback_search_url(value: str) -> str:
    parsed = urlparse(value)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        parsed.port
    except ValueError as exc:
        raise ValueError("remote HTTP MCTS URL must use a loopback IP address") from exc
    if (
        parsed.scheme != "http"
        or not address.is_loopback
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/search"
    ):
        raise ValueError(
            "remote HTTP MCTS URL must be a loopback http://.../search URL"
        )
    return value
