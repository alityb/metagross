#!/usr/bin/env python3
"""Fail-closed orchestration for preregistered offline promotion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from srcs.metagross import shadow_replay
from srcs.metagross.mcts_contract import ENGINE_CONTRACT, ENGINE_SOURCE_SHA256


SCHEMA_VERSION = 1
HASH_KEYS = {"path", "sha256"}
REGISTERED_IDENTITY_KEYS = {"id", "sha256"}
PAIR_MATCH_FIELDS = (
    "game_index",
    "pair_id",
    "battle_seed",
    "team_1_sha256",
    "team_2_sha256",
)


class ValidationError(ValueError):
    """Raised when a preregistration or evidence artifact is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be an object")
    return value


def _require_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise ValidationError(f"{label} keys differ: missing={missing}, extra={extra}")


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def _probability(value: object, label: str, *, inclusive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be numeric")
    number = float(value)
    lower_ok = number >= 0 if inclusive else number > 0
    upper_ok = number <= 1 if inclusive else number < 1
    if not math.isfinite(number) or not lower_ok or not upper_ok:
        raise ValidationError(f"{label} must be between zero and one")
    return number


def _validate_hashed_path(value: object, label: str) -> None:
    artifact = _require_object(value, label)
    _require_keys(artifact, HASH_KEYS, label)
    if not isinstance(artifact["path"], str) or not artifact["path"]:
        raise ValidationError(f"{label}.path must be a nonempty string")
    if not _is_sha256(artifact["sha256"]):
        raise ValidationError(f"{label}.sha256 must be a lowercase SHA-256 digest")


def _validate_registered_identity(value: object, label: str) -> None:
    identity = _require_object(value, label)
    _require_keys(identity, REGISTERED_IDENTITY_KEYS, label)
    if not isinstance(identity["id"], str) or not identity["id"]:
        raise ValidationError(f"{label}.id must be a nonempty string")
    if not _is_sha256(identity["sha256"]):
        raise ValidationError(f"{label}.sha256 must be a lowercase SHA-256 digest")


def validate_preregistration(value: object) -> dict[str, Any]:
    prereg = _require_object(value, "preregistration")
    top_keys = {
        "schema_version",
        "candidate",
        "baseline",
        "captures",
        "counterbalanced_ab",
        "pathology",
        "shadow_replay",
        "canary_audit",
        "confidence",
        "gates",
    }
    _require_keys(prereg, top_keys, "preregistration")
    if prereg["schema_version"] != SCHEMA_VERSION:
        raise ValidationError("unsupported preregistration schema_version")

    for name in ("candidate", "baseline"):
        identity = _require_object(prereg[name], name)
        _require_keys(identity, {"id", "path", "sha256"}, name)
        if not isinstance(identity["id"], str) or not identity["id"]:
            raise ValidationError(f"{name}.id must be a nonempty string")
        _validate_hashed_path(
            {"path": identity["path"], "sha256": identity["sha256"]}, name
        )
    if prereg["candidate"]["id"] == prereg["baseline"]["id"]:
        raise ValidationError("candidate and baseline IDs must differ")
    if prereg["candidate"]["sha256"] == prereg["baseline"]["sha256"]:
        raise ValidationError("candidate and baseline hashes must differ")

    captures = _require_object(prereg["captures"], "captures")
    if not captures:
        raise ValidationError("captures must not be empty")
    for capture_id, capture in captures.items():
        if not isinstance(capture_id, str) or not capture_id:
            raise ValidationError("capture IDs must be nonempty strings")
        capture = _require_object(capture, f"captures.{capture_id}")
        _require_keys(capture, {"path", "digest"}, f"captures.{capture_id}")
        if not isinstance(capture["path"], str) or not capture["path"]:
            raise ValidationError(f"captures.{capture_id}.path is invalid")
        if not _is_sha256(capture["digest"]):
            raise ValidationError(f"captures.{capture_id}.digest is invalid")

    ab = _require_object(prereg["counterbalanced_ab"], "counterbalanced_ab")
    _require_keys(
        ab,
        {"candidate_as_agent_a", "candidate_as_agent_b", "games_per_orientation"},
        "counterbalanced_ab",
    )
    for orientation in ("candidate_as_agent_a", "candidate_as_agent_b"):
        artifact = _require_object(ab[orientation], f"counterbalanced_ab.{orientation}")
        _require_keys(
            artifact,
            {"path", "sha256", "games", "artifacts"},
            f"counterbalanced_ab.{orientation}",
        )
        _validate_hashed_path(
            {"path": artifact["path"], "sha256": artifact["sha256"]},
            f"counterbalanced_ab.{orientation}",
        )
        _positive_int(artifact["games"], f"counterbalanced_ab.{orientation}.games")
        identities = _require_object(
            artifact["artifacts"],
            f"counterbalanced_ab.{orientation}.artifacts",
        )
        _require_keys(
            identities,
            {"candidate", "baseline"},
            f"counterbalanced_ab.{orientation}.artifacts",
        )
        for name in ("candidate", "baseline"):
            _validate_registered_identity(
                identities[name],
                f"counterbalanced_ab.{orientation}.artifacts.{name}",
            )
    games_per_orientation = _positive_int(
        ab["games_per_orientation"], "counterbalanced_ab.games_per_orientation"
    )
    if games_per_orientation % 2:
        raise ValidationError("games_per_orientation must be even")
    if any(
        ab[name]["games"] != games_per_orientation
        for name in ab
        if name != "games_per_orientation"
    ):
        raise ValidationError(
            "orientation game counts must equal games_per_orientation"
        )

    pathology = _require_object(prereg["pathology"], "pathology")
    _require_keys(pathology, {"command", "cwd", "expected_tests"}, "pathology")
    command = pathology["command"]
    expected_tests = pathology["expected_tests"]
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise ValidationError("pathology.command must be a nonempty string array")
    if not isinstance(pathology["cwd"], str) or not pathology["cwd"]:
        raise ValidationError("pathology.cwd must be a nonempty string")
    if (
        not isinstance(expected_tests, list)
        or not expected_tests
        or not all(isinstance(test, str) and test for test in expected_tests)
    ):
        raise ValidationError("pathology.expected_tests must be nonempty")

    shadow = _require_object(prereg["shadow_replay"], "shadow_replay")
    _require_keys(shadow, {"report", "capture_id", "expected_counts"}, "shadow_replay")
    _validate_hashed_path(shadow["report"], "shadow_replay.report")
    if (
        not isinstance(shadow["capture_id"], str)
        or shadow["capture_id"] not in captures
    ):
        raise ValidationError("shadow_replay.capture_id is not registered")
    expected_counts = _require_object(
        shadow["expected_counts"], "shadow_replay.expected_counts"
    )
    if not expected_counts:
        raise ValidationError("shadow_replay.expected_counts must not be empty")
    for name, count in expected_counts.items():
        if not isinstance(name, str) or not name:
            raise ValidationError("shadow replay count names must be nonempty")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValidationError(f"shadow replay count {name} must be nonnegative")

    canary = _require_object(prereg["canary_audit"], "canary_audit")
    _require_keys(canary, {"report", "capture_id"}, "canary_audit")
    _validate_hashed_path(canary["report"], "canary_audit.report")
    if (
        not isinstance(canary["capture_id"], str)
        or canary["capture_id"] not in captures
    ):
        raise ValidationError("canary_audit.capture_id is not registered")

    confidence = _require_object(prereg["confidence"], "confidence")
    _require_keys(
        confidence,
        {"method", "alpha", "seed", "bootstrap_repeats"},
        "confidence",
    )
    if confidence["method"] not in {"exact_sign", "cluster_bootstrap"}:
        raise ValidationError("confidence.method is unsupported")
    _probability(confidence["alpha"], "confidence.alpha", inclusive=False)
    if isinstance(confidence["seed"], bool) or not isinstance(confidence["seed"], int):
        raise ValidationError("confidence.seed must be an integer")
    _positive_int(confidence["bootstrap_repeats"], "confidence.bootstrap_repeats")

    gates = _require_object(prereg["gates"], "gates")
    gate_keys = {
        "minimum_total_games",
        "minimum_candidate_win_rate",
        "maximum_model_slot_gap",
        "minimum_lower_confidence_bound",
        "maximum_p_value",
        "minimum_normal_combat_games",
    }
    _require_keys(gates, gate_keys, "gates")
    _positive_int(gates["minimum_total_games"], "gates.minimum_total_games")
    _positive_int(
        gates["minimum_normal_combat_games"],
        "gates.minimum_normal_combat_games",
    )
    for name in gate_keys - {"minimum_total_games", "minimum_normal_combat_games"}:
        _probability(gates[name], f"gates.{name}")
    return prereg


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_games(
    path: Path,
    registered_artifacts: dict[str, dict[str, str]],
    label: str,
) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{label} artifact envelope is invalid JSON") from exc
    envelope = _require_object(payload, f"{label} artifact envelope")
    artifacts = _require_object(
        envelope.get("artifacts"), f"{label} artifact envelope artifacts"
    )
    if artifacts != registered_artifacts:
        raise ValidationError(f"{label} artifact envelope identity mismatch")
    payload = envelope.get("games")
    if not isinstance(payload, list):
        raise ValidationError(f"{path} does not contain a games array")
    games: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValidationError(f"{path} contains a non-object game row")
        nested = row.get("games")
        if isinstance(nested, list):
            if not all(isinstance(game, dict) for game in nested):
                raise ValidationError(f"{path} contains invalid nested games")
            games.extend(nested)
        else:
            games.append(row)
    return games


def _validated_pairs(
    games: list[dict[str, Any]], label: str
) -> dict[int, dict[int, dict[str, Any]]]:
    pairs: defaultdict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for game in games:
        pair_index = game.get("pair_index")
        pair_leg = game.get("pair_leg")
        if (
            isinstance(pair_index, bool)
            or not isinstance(pair_index, int)
            or pair_index <= 0
        ):
            raise ValidationError(f"{label} has an invalid pair_index")
        if pair_leg not in {1, 2}:
            raise ValidationError(f"{label} has an invalid pair_leg")
        if pair_leg in pairs[pair_index]:
            raise ValidationError(
                f"{label} duplicates pair {pair_index} leg {pair_leg}"
            )
        if game.get("void") is not False or game.get("error") is not None:
            raise ValidationError(f"{label} contains a void or error game")
        if game.get("winner") not in {"agent_a", "agent_b"}:
            raise ValidationError(f"{label} contains a tie or unknown winner")
        if game.get("forfeit") is True or game.get("end_reason") == "forfeit":
            raise ValidationError(f"{label} contains a forfeit game")
        pairs[pair_index][pair_leg] = game
    if not pairs or sorted(pairs) != list(range(1, len(pairs) + 1)):
        raise ValidationError(f"{label} has a non-contiguous pair schedule")
    for pair_index, legs in pairs.items():
        if set(legs) != {1, 2}:
            raise ValidationError(f"{label} pair {pair_index} is incomplete")
        pair_ids = {game.get("pair_id") for game in legs.values()}
        if len(pair_ids) != 1 or None in pair_ids:
            raise ValidationError(
                f"{label} pair {pair_index} has inconsistent pair IDs"
            )
    return dict(pairs)


def _cluster_scores(
    as_a: list[dict[str, Any]], as_b: list[dict[str, Any]]
) -> tuple[list[float], dict[str, int]]:
    first = _validated_pairs(as_a, "candidate_as_agent_a")
    second = _validated_pairs(as_b, "candidate_as_agent_b")
    if set(first) != set(second):
        raise ValidationError("model-slot orientations do not cover the same pairs")
    scores = []
    wins_as_a = 0
    wins_as_b = 0
    for pair_index in sorted(first):
        cluster_wins = 0
        for leg in (1, 2):
            left = first[pair_index][leg]
            right = second[pair_index][leg]
            if any(left.get(field) != right.get(field) for field in PAIR_MATCH_FIELDS):
                raise ValidationError(
                    f"model-slot orientations mismatch at pair {pair_index} leg {leg}"
                )
            left_win = left["winner"] == "agent_a"
            right_win = right["winner"] == "agent_b"
            wins_as_a += left_win
            wins_as_b += right_win
            cluster_wins += left_win + right_win
        scores.append(cluster_wins / 4)
    return scores, {
        "candidate_as_agent_a": wins_as_a,
        "candidate_as_agent_b": wins_as_b,
    }


def _binomial_survival(successes: int, trials: int, probability: float) -> float:
    return sum(
        math.comb(trials, count)
        * probability**count
        * (1 - probability) ** (trials - count)
        for count in range(successes, trials + 1)
    )


def _exact_lower_bound(successes: int, trials: int, alpha: float) -> float:
    if successes == 0 or trials == 0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(80):
        midpoint = (low + high) / 2
        if _binomial_survival(successes, trials, midpoint) > alpha:
            high = midpoint
        else:
            low = midpoint
    return low


def matched_pair_inference(
    cluster_scores: list[float], confidence: dict[str, Any]
) -> dict[str, Any]:
    if not cluster_scores:
        raise ValidationError("inference requires at least one complete pair cluster")
    alpha = float(confidence["alpha"])
    positive = sum(score > 0.5 for score in cluster_scores)
    negative = sum(score < 0.5 for score in cluster_scores)
    neutral = len(cluster_scores) - positive - negative
    informative = positive + negative
    sign_p_value = (
        _binomial_survival(positive, informative, 0.5) if informative else 1.0
    )
    result: dict[str, Any] = {
        "method": confidence["method"],
        "alpha": alpha,
        "clusters": len(cluster_scores),
        "positive_clusters": positive,
        "negative_clusters": negative,
        "neutral_clusters": neutral,
        "candidate_win_rate": sum(cluster_scores) / len(cluster_scores),
        "exact_one_sided_sign_p_value": sign_p_value,
        "seed": confidence["seed"],
    }
    if confidence["method"] == "exact_sign":
        result["lower_confidence_bound"] = _exact_lower_bound(
            positive, informative, alpha
        )
        result["lower_bound_estimand"] = "probability_of_positive_pair_cluster"
        result["p_value"] = sign_p_value
        result["bootstrap_repeats"] = 0
        return result

    repeats = confidence["bootstrap_repeats"]
    generator = random.Random(confidence["seed"])
    sample_size = len(cluster_scores)
    means = sorted(
        sum(generator.choice(cluster_scores) for _ in range(sample_size)) / sample_size
        for _ in range(repeats)
    )
    index = max(0, math.floor(alpha * (repeats + 1)) - 1)
    result["lower_confidence_bound"] = means[index]
    result["lower_bound_estimand"] = "candidate_game_win_rate"
    result["p_value"] = sign_p_value
    result["bootstrap_repeats"] = repeats
    return result


def _artifact_check(root: Path, artifact: dict[str, str], label: str) -> Path:
    path = _resolve(root, artifact["path"])
    if not path.is_file():
        raise ValidationError(f"{label} is missing: {path}")
    actual = _sha256(path)
    if actual != artifact["sha256"]:
        raise ValidationError(f"{label} SHA-256 mismatch")
    return path


def _read_report(root: Path, artifact: dict[str, str], label: str) -> dict[str, Any]:
    path = _artifact_check(root, artifact, label)
    try:
        return _require_object(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{label} is invalid JSON") from exc


def _canonical_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n"


def _capture_candidate_sha256(manifest: object) -> str:
    manifest = _require_object(manifest, "capture manifest")
    registered: list[tuple[str, object]] = []
    policy = manifest.get("policy")
    if isinstance(policy, dict) and "sha256" in policy:
        registered.append(("policy.sha256", policy["sha256"]))
    checkpoint = manifest.get("checkpoint")
    if isinstance(checkpoint, dict) and "sha256_verified" in checkpoint:
        registered.append(
            ("checkpoint.sha256_verified", checkpoint["sha256_verified"])
        )
    if not registered:
        raise ValidationError(
            "capture manifest has no policy.sha256 or checkpoint.sha256_verified"
        )
    for label, value in registered:
        if not _is_sha256(value):
            raise ValidationError(f"capture manifest {label} is invalid")
    hashes = {value for _label, value in registered}
    if len(hashes) != 1:
        raise ValidationError("capture manifest candidate SHA-256 fields disagree")
    return registered[0][1]


def _validate_ab_identities(
    artifact: dict[str, Any], prereg: dict[str, Any], orientation: str
) -> None:
    for name in ("candidate", "baseline"):
        expected = {"id": prereg[name]["id"], "sha256": prereg[name]["sha256"]}
        if artifact["artifacts"][name] != expected:
            raise ValidationError(
                f"{orientation} {name} artifact identity does not match registration"
            )


def run_validation(
    preregistration_path: Path,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    preregistration_path = preregistration_path.resolve()
    root = preregistration_path.parent
    prereg = validate_preregistration(
        json.loads(preregistration_path.read_text(encoding="utf-8"))
    )
    if command_runner is None:
        command_runner = subprocess.run
    failures: list[str] = []
    checks: dict[str, bool] = {}

    def check(name: str, operation: Callable[[], None]) -> None:
        try:
            operation()
        except (
            Exception
        ) as exc:  # Each unexpected artifact/API failure closes the gate.
            checks[name] = False
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
        else:
            checks[name] = True

    for name in ("candidate", "baseline"):
        check(
            f"{name}_identity",
            lambda name=name: _artifact_check(root, prereg[name], name),
        )

    for capture_id, capture in prereg["captures"].items():

        def validate_capture(
            capture_id: str = capture_id, capture: dict = capture
        ) -> None:
            path = _resolve(root, capture["path"])
            _protocol, _searches, metadata = shadow_replay.load_capture(path)
            if metadata.get("capture_digest") != capture["digest"]:
                raise ValidationError("capture digest mismatch")
            candidate_sha256 = _capture_candidate_sha256(metadata.get("manifest"))
            if candidate_sha256 != prereg["candidate"]["sha256"]:
                raise ValidationError(
                    "capture manifest candidate SHA-256 does not match registration"
                )

        check(f"capture_{capture_id}", validate_capture)

    ab = prereg["counterbalanced_ab"]
    orientation_games: dict[str, list[dict[str, Any]]] = {}
    for orientation in ("candidate_as_agent_a", "candidate_as_agent_b"):

        def validate_orientation(orientation: str = orientation) -> None:
            artifact = ab[orientation]
            _validate_ab_identities(artifact, prereg, orientation)
            path = _artifact_check(root, artifact, orientation)
            games = _load_games(path, artifact["artifacts"], orientation)
            if len(games) != artifact["games"]:
                raise ValidationError(
                    f"expected {artifact['games']} games, found {len(games)}"
                )
            orientation_games[orientation] = games

        check(orientation, validate_orientation)

    inference: dict[str, Any] | None = None
    slot_wins: dict[str, int] | None = None

    def infer() -> None:
        nonlocal inference, slot_wins
        scores, slot_wins = _cluster_scores(
            orientation_games["candidate_as_agent_a"],
            orientation_games["candidate_as_agent_b"],
        )
        inference = matched_pair_inference(scores, prereg["confidence"])

    check("matched_pair_inference", infer)

    pathology_result: dict[str, Any] = {}

    def pathology() -> None:
        nonlocal pathology_result
        specification = prereg["pathology"]
        completed = command_runner(
            specification["command"],
            cwd=_resolve(root, specification["cwd"]),
            capture_output=True,
            text=True,
            check=False,
        )
        pathology_result = {
            "command": specification["command"],
            "returncode": completed.returncode,
            "demonstrated_tests": [],
        }
        if completed.returncode != 0:
            raise ValidationError(
                f"pathology fixture command returned {completed.returncode}"
            )
        stdout = completed.stdout if isinstance(completed.stdout, str) else ""
        stderr = completed.stderr if isinstance(completed.stderr, str) else ""
        demonstrated = [
            marker
            for marker in specification["expected_tests"]
            if marker in stdout or marker in stderr
        ]
        pathology_result["demonstrated_tests"] = demonstrated
        missing = [
            marker
            for marker in specification["expected_tests"]
            if marker not in demonstrated
        ]
        if missing:
            raise ValidationError(
                f"pathology fixture output omitted expected tests: {missing}"
            )

    check("pathology_fixtures", pathology)

    shadow_result: dict[str, Any] = {}

    def shadow_report() -> None:
        nonlocal shadow_result
        specification = prereg["shadow_replay"]
        report = _read_report(root, specification["report"], "shadow replay report")
        capture_id = specification["capture_id"]
        expected_digest = prereg["captures"][capture_id]["digest"]
        actual_digest = report.get("capture", {}).get("capture_digest")
        if actual_digest != expected_digest:
            raise ValidationError("shadow replay capture digest mismatch")
        if report.get("mode") not in {"captured_v5_holdout", "remote_v5_holdout"}:
            raise ValidationError("shadow replay did not execute a v5 holdout mode")
        engine = _require_object(report.get("engine"), "shadow replay engine")
        if (
            engine.get("contract") != ENGINE_CONTRACT
            or engine.get("source_sha256") != ENGINE_SOURCE_SHA256
            or not _is_sha256(engine.get("native_sha256"))
        ):
            raise ValidationError("shadow replay engine identity mismatch")
        counts = _require_object(report.get("counts"), "shadow replay counts")
        expected_counts = specification["expected_counts"]
        mismatches = {
            name: {"expected": expected, "actual": counts.get(name)}
            for name, expected in expected_counts.items()
            if counts.get(name) != expected
        }
        if mismatches:
            raise ValidationError(f"shadow replay count mismatch: {mismatches}")
        if counts.get("holdout_failures") != 0:
            raise ValidationError("shadow replay has holdout failures")
        evaluated = counts.get("evaluated_disagreements")
        if isinstance(evaluated, bool) or not isinstance(evaluated, int) or evaluated <= 0:
            raise ValidationError("shadow replay evaluated no holdout disagreements")
        shadow_result = {
            "capture_id": capture_id,
            "capture_digest": actual_digest,
            "counts": {name: counts[name] for name in sorted(expected_counts)},
        }

    check("shadow_replay_report", shadow_report)

    canary_result: dict[str, Any] = {}

    def canary_report() -> None:
        nonlocal canary_result
        specification = prereg["canary_audit"]
        report = _read_report(root, specification["report"], "canary audit report")
        capture_id = specification["capture_id"]
        expected_digest = prereg["captures"][capture_id]["digest"]
        if report.get("capture", {}).get("digest") != expected_digest:
            raise ValidationError("canary capture digest mismatch")
        if report.get("identity", {}).get("policy_sha256") != prereg["candidate"][
            "sha256"
        ]:
            raise ValidationError("canary policy SHA-256 does not match candidate")
        integrity = _require_object(report.get("integrity"), "canary integrity")
        if integrity.get("passed") is not True or integrity.get("failures") != []:
            raise ValidationError("canary integrity did not pass cleanly")
        result = _require_object(report.get("result"), "canary result")
        normal_wins = result.get("normal_wins")
        normal_losses = result.get("normal_losses")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (normal_wins, normal_losses)
        ):
            raise ValidationError("canary normal-combat counts are invalid")
        normal_games = normal_wins + normal_losses
        minimum = prereg["gates"]["minimum_normal_combat_games"]
        if normal_games < minimum:
            raise ValidationError(
                f"normal-combat games {normal_games} are below minimum {minimum}"
            )
        canary_result = {
            "capture_id": capture_id,
            "normal_wins": normal_wins,
            "normal_losses": normal_losses,
            "normal_combat_games": normal_games,
            "integrity_passed": True,
        }

    check("canary_audit_report", canary_report)

    gate_results: dict[str, dict[str, Any]] = {}
    ab_summary: dict[str, Any] = {
        "games_per_orientation": ab["games_per_orientation"],
        "registered_artifacts": {
            orientation: {
                "sha256": ab[orientation]["sha256"],
                "games": ab[orientation]["games"],
                "artifacts": ab[orientation]["artifacts"],
            }
            for orientation in ("candidate_as_agent_a", "candidate_as_agent_b")
        },
    }
    if inference is not None and slot_wins is not None:
        games_per_orientation = ab["games_per_orientation"]
        total_games = games_per_orientation * 2
        slot_gap = abs(
            slot_wins["candidate_as_agent_a"] / games_per_orientation
            - slot_wins["candidate_as_agent_b"] / games_per_orientation
        )
        ab_summary.update(
            {
                "total_games": total_games,
                "candidate_wins": sum(slot_wins.values()),
                "candidate_win_rate": inference["candidate_win_rate"],
                "model_slot_gap": slot_gap,
            }
        )
        observed = {
            "minimum_total_games": total_games,
            "minimum_candidate_win_rate": inference["candidate_win_rate"],
            "maximum_model_slot_gap": slot_gap,
            "minimum_lower_confidence_bound": inference["lower_confidence_bound"],
            "maximum_p_value": inference["p_value"],
        }
        for name, value in observed.items():
            threshold = prereg["gates"][name]
            passed = (
                value >= threshold
                if name.startswith("minimum_")
                else value <= threshold
            )
            gate_results[name] = {
                "passed": passed,
                "observed": value,
                "threshold": threshold,
            }
            if not passed:
                failures.append(
                    f"gate {name} failed: observed={value}, threshold={threshold}"
                )
    else:
        failures.append("statistical gates unavailable because inference failed")

    normal_gate_passed = checks.get("canary_audit_report", False)
    gate_results["minimum_normal_combat_games"] = {
        "passed": normal_gate_passed,
        "observed": canary_result.get("normal_combat_games"),
        "threshold": prereg["gates"]["minimum_normal_combat_games"],
    }
    passed = (
        not failures
        and all(checks.values())
        and all(gate["passed"] for gate in gate_results.values())
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "preregistered_offline_validation",
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": _sha256(preregistration_path),
        },
        "identities": {
            name: {"id": prereg[name]["id"], "sha256": prereg[name]["sha256"]}
            for name in ("candidate", "baseline")
        },
        "captures": {
            capture_id: {
                "digest": capture["digest"],
                "passed": checks.get(f"capture_{capture_id}", False),
            }
            for capture_id, capture in sorted(prereg["captures"].items())
        },
        "counterbalanced_ab": ab_summary,
        "checks": dict(sorted(checks.items())),
        "pathology": pathology_result,
        "shadow_replay": shadow_result,
        "canary_audit": canary_result,
        "inference": inference,
        "model_slot_wins": slot_wins,
        "gates": gate_results,
        "failures": failures,
        "passed": passed,
    }


def generate_template() -> dict[str, Any]:
    zeros = "0" * 64
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate": {"id": "candidate-id", "path": "candidate.bin", "sha256": zeros},
        "baseline": {"id": "baseline-id", "path": "baseline.bin", "sha256": "1" * 64},
        "captures": {
            "offline": {"path": "capture", "digest": zeros},
            "canary": {"path": "canary-capture", "digest": "2" * 64},
        },
        "counterbalanced_ab": {
            "candidate_as_agent_a": {
                "path": "candidate-as-a.json",
                "sha256": zeros,
                "games": 100,
                "artifacts": {
                    "candidate": {"id": "candidate-id", "sha256": zeros},
                    "baseline": {"id": "baseline-id", "sha256": "1" * 64},
                },
            },
            "candidate_as_agent_b": {
                "path": "candidate-as-b.json",
                "sha256": zeros,
                "games": 100,
                "artifacts": {
                    "candidate": {"id": "candidate-id", "sha256": zeros},
                    "baseline": {"id": "baseline-id", "sha256": "1" * 64},
                },
            },
            "games_per_orientation": 100,
        },
        "pathology": {
            "command": [
                "python3",
                "-m",
                "unittest",
                "-v",
                "srcs.metagross.tests.test_launch.LaunchTest.test_canary_turn_5_allows_wish_without_a_pending_wish",
                "srcs.metagross.tests.test_launch.LaunchTest.test_canary_turn_6_rejects_wish_while_one_is_pending",
                "srcs.metagross.tests.test_launch.LaunchTest.test_canary_turn_26_rejects_encore_after_the_target_switched",
                "srcs.metagross.tests.test_launch.LaunchTest.test_canary_turn_46_rejects_a_move_blocked_by_revealed_bulletproof",
                "srcs.metagross.tests.test_launch.LaunchTest.test_no_progress_cycle_detector_covers_periods_one_two_and_three",
                "srcs.metagross.tests.test_launch.LaunchTest.test_semantic_no_progress_blocks_consecutive_protect",
                "srcs.metagross.tests.test_launch.LaunchTest.test_semantic_no_progress_breaks_switch_carousel",
                "srcs.metagross.tests.test_launch.LaunchTest.test_terminal_pivot_without_reserve_uses_nonpivot_action",
            ],
            "cwd": ".",
            "expected_tests": [
                "test_canary_turn_5_allows_wish_without_a_pending_wish",
                "test_canary_turn_6_rejects_wish_while_one_is_pending",
                "test_canary_turn_26_rejects_encore_after_the_target_switched",
                "test_canary_turn_46_rejects_a_move_blocked_by_revealed_bulletproof",
                "test_no_progress_cycle_detector_covers_periods_one_two_and_three",
                "test_semantic_no_progress_blocks_consecutive_protect",
                "test_semantic_no_progress_breaks_switch_carousel",
                "test_terminal_pivot_without_reserve_uses_nonpivot_action",
            ],
        },
        "shadow_replay": {
            "report": {"path": "shadow-report.json", "sha256": zeros},
            "capture_id": "offline",
            "expected_counts": {"decisions": 1, "holdout_failures": 0},
        },
        "canary_audit": {
            "report": {"path": "canary-report.json", "sha256": zeros},
            "capture_id": "canary",
        },
        "confidence": {
            "method": "exact_sign",
            "alpha": 0.05,
            "seed": 0,
            "bootstrap_repeats": 10000,
        },
        "gates": {
            "minimum_total_games": 200,
            "minimum_candidate_win_rate": 0.5,
            "maximum_model_slot_gap": 0.1,
            "minimum_lower_confidence_bound": 0.5,
            "maximum_p_value": 0.05,
            "minimum_normal_combat_games": 3,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--generate-template", type=Path)
    args = parser.parse_args(argv)
    if args.generate_template:
        if args.preregistration or args.output:
            parser.error(
                "--generate-template cannot be combined with validation options"
            )
    elif not args.preregistration or not args.output:
        parser.error("--preregistration and --output are required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.generate_template:
        args.generate_template.write_text(
            json.dumps(generate_template(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    try:
        summary = run_validation(args.preregistration)
    except Exception as exc:  # Malformed or inaccessible preregistrations fail closed.
        summary = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "preregistered_offline_validation",
            "passed": False,
            "failures": [f"preregistration: {type(exc).__name__}: {exc}"],
        }
        status = 2
    else:
        status = 0 if summary["passed"] else 1
    args.output.write_text(_canonical_json(summary), encoding="utf-8")
    print(_canonical_json(summary), end="")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
