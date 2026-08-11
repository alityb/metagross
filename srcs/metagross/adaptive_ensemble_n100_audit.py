#!/usr/bin/env python3
"""Fail-closed aggregate audit for the preregistered 50-pair local screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from srcs.metagross import adaptive_ensemble_smoke_audit, paired_h2h_inference, shadow_replay
from srcs.metagross.h2h_audit import _read_jsonl, _sha256


def _nearest_rank(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires observations")
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def _opponent_support(battle: object) -> tuple[bool, set[str]]:
    opponent = getattr(battle, "opponent", None)
    active = getattr(opponent, "active", None)
    reserve = list(getattr(opponent, "reserve", ()) or ())
    moves = list(getattr(active, "moves", ()) or ()) if active is not None else []
    complete = active is not None and len(moves) == 4 and len(reserve) == 5
    if not complete:
        return False, set()
    normalize = lambda value: "".join(  # noqa: E731
        character for character in str(value or "").lower() if character.isalnum()
    )
    move_names = {normalize(getattr(move, "name", move)) for move in moves}
    switch_names = {
        normalize(getattr(pokemon, "name", ""))
        for pokemon in reserve
        if float(getattr(pokemon, "hp", 0) or 0) > 0
    }
    actions = set(move_names)
    actions.update(f"switch {name}" for name in switch_names)
    if not any(
        bool(getattr(pokemon, "terastallized", False))
        for pokemon in [active, *reserve]
    ):
        actions.update(f"{name}-tera" for name in move_names)
    return True, actions


def _load_prior_rows(paths: list[Path], tags: set[str]) -> dict[str, dict[tuple[str, int], dict]]:
    rows: dict[str, dict[tuple[str, int], dict]] = {"candidate": {}, "comparator": {}}
    for path in paths:
        for row in _read_jsonl(path):
            if row.get("tag") not in tags:
                continue
            mode = {"agent_a": "candidate", "agent_b": "comparator"}.get(row.get("namespace"))
            if mode is None:
                raise ValueError(f"unexpected prior namespace in {path}")
            key = shadow_replay._decision_key(row)
            if key in rows[mode]:
                raise ValueError(f"duplicate {mode} prior key: {key}")
            rows[mode][key] = row
    return rows


def _text_log_clean(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    return not re.search(
        r"Traceback \(most recent call last\)|(^|\n).*\bWARN(?:ING)?\b|HTTP Error|\bERROR\b|\bException:\s|\bError:\s",
        text,
        re.IGNORECASE,
    )


def audit(
    preregistration_path: Path,
    authorization_path: Path,
    audit_protocol_path: Path,
    results_path: Path,
    pairs_path: Path,
    log_dir: Path,
    prior_decision_paths: list[Path],
    prior_server_logs: list[Path],
    base_audit_path: Path,
    runtime_manifest_path: Path,
    python_runtime_manifest_path: Path,
    runtime_preflight_path: Path,
    remote_preflight_path: Path,
    showdown_launch_path: Path,
    showdown_log_path: Path,
) -> dict[str, object]:
    prereg = json.loads(preregistration_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    frozen = json.loads(audit_protocol_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    base_audit = json.loads(base_audit_path.read_text(encoding="utf-8"))
    base_protocol_path = Path(base_audit.get("inputs", {}).get("audit_protocol", {}).get("path", ""))
    if not base_protocol_path.is_absolute():
        base_protocol_path = (base_audit_path.parent / base_protocol_path).resolve()
    try:
        recomputed_base_audit = adaptive_ensemble_smoke_audit.audit(
            results_path,
            log_dir,
            prior_decision_paths,
            preregistration_path,
            pairs_path,
            base_protocol_path,
        )
    except Exception as exc:  # noqa: BLE001
        recomputed_base_audit = None
        failures = [f"base audit recomputation failed: {type(exc).__name__}: {exc}"]
    else:
        failures = []
    runtime_preflight = json.loads(runtime_preflight_path.read_text(encoding="utf-8"))
    remote_preflight = json.loads(remote_preflight_path.read_text(encoding="utf-8"))
    showdown_launch = json.loads(showdown_launch_path.read_text(encoding="utf-8"))
    tags = {game.get("battle_tag") for game in results.get("games", [])}

    input_paths = {
        "preregistration": preregistration_path,
        "authorization": authorization_path,
        "results": results_path,
        "pairs": pairs_path,
        "base_audit": base_audit_path,
        "base_audit_protocol": base_protocol_path,
        "runtime_manifest": runtime_manifest_path,
        "python_runtime_manifest": python_runtime_manifest_path,
        "runtime_preflight": runtime_preflight_path,
        "remote_preflight": remote_preflight_path,
        "showdown_launch": showdown_launch_path,
        "showdown_log": showdown_log_path,
    }
    expected_inputs = frozen.get("inputs", {})
    if (
        frozen.get("status") != "frozen_before_audit"
        or frozen.get("audit_source_sha256") != _sha256(Path(__file__).resolve())
        or any(expected_inputs.get(name) != _sha256(path) for name, path in input_paths.items())
        or expected_inputs.get("prior_decisions") != [_sha256(path) for path in prior_decision_paths]
        or expected_inputs.get("prior_server_logs") != [_sha256(path) for path in prior_server_logs]
        or expected_inputs.get("log_files") != {
            str(path.relative_to(log_dir)): _sha256(path)
            for path in sorted(item for item in log_dir.rglob("*") if item.is_file())
        }
    ):
        failures.append("aggregate audit differs from its frozen protocol")

    schedule = prereg.get("schedule", {})
    if (
        prereg.get("status") != "frozen_before_games"
        or prereg.get("scope", {}).get("games") != 100
        or prereg.get("scope", {}).get("mirrored_pairs") != 50
        or schedule.get("bootstrap_resamples") != 100_000
    ):
        failures.append("execution preregistration is not the fixed N=100 protocol")

    games = results.get("games", [])
    summary = results.get("summary", {})
    game_indices = {game.get("game_index") for game in games}
    pair_ids = [pair.get("pair_id") for pair in pairs.get("pairs", [])]
    completion = {
        "exact_game_objects": len(games) == 100,
        "exact_game_indices": game_indices == set(range(1, 101)),
        "unique_battle_tags": len(tags) == 100 and None not in tags,
        "summary_complete": summary.get("completed_games") == summary.get("decisive_games") == 100,
        "zero_voids": summary.get("void_games") == summary.get("void_pairs") == 0,
        "zero_ties_unknown": summary.get("ties_or_unknown") == 0,
        "all_games_decisive": all(
            game.get("winner") in {"agent_a", "agent_b"}
            and game.get("void") is False
            and game.get("error") is None
            for game in games
        ),
        "exact_pair_manifest": len(pair_ids) == len(set(pair_ids)) == 50,
        "exact_pair_indices": {pair.get("pair_index") for pair in pairs.get("pairs", [])} == set(range(1, 51)),
    }
    expected_eval_argv = prereg.get("execution", {}).get("evaluation_argv", [])
    execution_identity = results.get("execution_identity", {})
    execution_gates = {
        "arguments_exact": execution_identity.get("arguments") == expected_eval_argv[3:],
        "python_exact": execution_identity.get("python_executable")
        == str((Path(__file__).resolve().parents[2] / expected_eval_argv[0]).resolve()),
        "source_exact": execution_identity.get("source_sha256")
        == prereg.get("source_identity", {}).get("eval_run.py"),
        "pair_config_exact": execution_identity.get("config_sha256") == pairs.get("config_sha256"),
        "environment_exact": execution_identity.get("environment")
        == prereg.get("execution", {}).get("environment"),
    }

    candidate_p1 = [game for game in games if game.get("challenger") == "agent_a"]
    candidate_p2 = [game for game in games if game.get("acceptor") == "agent_a"]
    p1_rate = sum(game.get("winner") == "agent_a" for game in candidate_p1) / len(candidate_p1) if candidate_p1 else math.nan
    p2_rate = sum(game.get("winner") == "agent_a" for game in candidate_p2) / len(candidate_p2) if candidate_p2 else math.nan
    role_gap = abs(p1_rate - p2_rate)
    roles = {
        "candidate_p1_games": len(candidate_p1),
        "candidate_p2_games": len(candidate_p2),
        "candidate_p1_win_rate": p1_rate,
        "candidate_p2_win_rate": p2_rate,
        "absolute_role_gap": role_gap,
    }
    role_gates = {
        "exact_role_balance": len(candidate_p1) == len(candidate_p2) == 50,
        "role_gap_within_limit": math.isfinite(role_gap) and role_gap <= 0.20,
    }

    prior_rows = _load_prior_rows(prior_decision_paths, tags)
    player_legal_by_key: dict[str, dict[tuple[str, int], set[str]]] = {
        "candidate": {},
        "comparator": {},
    }
    for mode, rows in prior_rows.items():
        for key, row in rows.items():
            table = row.get("name_table") or {}
            illegal = row.get("illegal_actions") or []
            if (
                len(illegal) != 13
                or any(not isinstance(value, bool) for value in illegal)
                or any(not isinstance(index, int) or not 0 <= index < 13 for index in table.values())
            ):
                failures.append(f"{mode}/{key}: malformed player legal support")
                continue
            player_legal_by_key[mode][key] = {
                name for name, index in table.items() if illegal[index] is False
            }
    search_time_by_mode: dict[str, dict[tuple[str, int], tuple[int, int]]] = {
        "candidate": {},
        "comparator": {},
    }
    for search_path in sorted(log_dir.glob("*.search.jsonl")):
        rows = [
            row for row in _read_jsonl(search_path)
            if (row.get("context") or {}).get("tag") in tags
        ]
        if not rows:
            continue
        operations = {(row.get("remote_search") or {}).get("operation") for row in rows}
        mode = "candidate" if operations == {"independent_ensemble"} else "comparator" if operations == {None} else None
        if mode is None:
            continue
        for row in rows:
            key = shadow_replay._search_key(row)
            search_time_by_mode[mode][key] = (
                int(row.get("time_ns")),
                int(row["context"].get("battle_turn")),
            )
    telemetry = {
        mode: {
            "decisions": 0,
            "eligible": 0,
            "eligible_with_prior": 0,
            "ineligible_with_prior": 0,
            "support_disagreements": 0,
            "illegal_actions": 0,
            "nonnormalized": 0,
            "errors": 0,
            "cross_side_legal_mismatches": 0,
        }
        for mode in ("candidate", "comparator")
    }
    rpc_values = {"candidate": [], "comparator": []}
    request_counts = {"candidate": 0, "comparator": 0}
    search_cpu_ms = {"candidate": 0.0, "comparator": 0.0}
    observed_search_keys = {"candidate": set(), "comparator": set()}

    for search_path in sorted(log_dir.glob("*.search.jsonl")):
        search_rows = [
            row for row in _read_jsonl(search_path)
            if (row.get("context") or {}).get("tag") in tags
        ]
        if not search_rows:
            continue
        operations = {(row.get("remote_search") or {}).get("operation") for row in search_rows}
        mode = "candidate" if operations == {"independent_ensemble"} else "comparator" if operations == {None} else None
        if mode is None:
            failures.append(f"{search_path.name}: invalid operation mixture")
            continue
        by_key = {shadow_replay._search_key(row): row for row in search_rows}
        username = search_path.name.removesuffix(".search.jsonl")
        protocol_path = log_dir / f"{username}.protocol.jsonl"
        reconstructed = shadow_replay.reconstruct_battles(_read_jsonl(protocol_path), by_key, username)
        for key, row in by_key.items():
            observed_search_keys[mode].add(key)
            prior = prior_rows[mode].get(key)
            if prior is None:
                failures.append(f"{mode}/{key}: missing prior telemetry")
                continue
            context = row["context"]
            if (
                prior.get("schema") != 4
                or prior.get("rqid") != context.get("rqid")
                or prior.get("battle_turn") != context.get("battle_turn")
                or prior.get("username") != username
            ):
                failures.append(f"{mode}/{key}: prior context mismatch")
            evidence = prior.get("opponent_prior") or {}
            independently_eligible, expected_actions = _opponent_support(reconstructed[key])
            reported_eligible = evidence.get("support_complete") is True
            raw = evidence.get("raw_priors") or {}
            telemetry[mode]["decisions"] += 1
            telemetry[mode]["eligible"] += int(independently_eligible)
            telemetry[mode]["eligible_with_prior"] += int(independently_eligible and bool(raw))
            telemetry[mode]["ineligible_with_prior"] += int(not independently_eligible and bool(raw))
            telemetry[mode]["support_disagreements"] += int(reported_eligible != independently_eligible)
            telemetry[mode]["errors"] += int(evidence.get("status") == "error")
            if raw:
                values = list(raw.values())
                valid_mass = all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) >= 0
                    for value in values
                )
                mass = math.fsum(float(value) for value in values) if valid_mass else math.nan
                telemetry[mode]["nonnormalized"] += int(not valid_mass or not math.isclose(mass, 1.0, abs_tol=1e-6))
                legal_indices = {
                    index
                    for index, illegal in enumerate(evidence.get("illegal_actions") or [])
                    if illegal is False
                }
                table = evidence.get("name_table") or {}
                server_legal_actions = {
                    name for name, index in table.items() if index in legal_indices
                }
                opposite = "comparator" if mode == "candidate" else "candidate"
                current_time = search_time_by_mode[mode].get(key, (None, None))[0]
                cross_candidates = [
                    (abs(other_time - current_time), other_key)
                    for other_key, (other_time, other_turn) in search_time_by_mode[opposite].items()
                    if current_time is not None
                    and other_key[0] == key[0]
                    and other_turn == prior.get("battle_turn")
                ]
                cross_candidates.sort()
                cross_key = cross_candidates[0][1] if cross_candidates else None
                cross_side_actions = player_legal_by_key[opposite].get(cross_key) if cross_key else None
                cross_time_valid = bool(cross_candidates) and cross_candidates[0][0] <= 30_000_000_000
                telemetry[mode]["illegal_actions"] += len(set(raw) - expected_actions)
                telemetry[mode]["illegal_actions"] += len(set(raw) ^ server_legal_actions)
                telemetry[mode]["cross_side_legal_mismatches"] += int(
                    not cross_time_valid or cross_side_actions is None or set(raw) != cross_side_actions
                )
                used = dict(row.get("opponent_priors") or [])
                normalized = {name: float(value) / mass for name, value in raw.items()} if valid_mass and mass > 0 else {}
                if used.keys() != normalized.keys() or any(
                    not math.isclose(float(used[name]), value, abs_tol=1e-7)
                    for name, value in normalized.items()
                ):
                    failures.append(f"{mode}/{key}: used opponent prior mismatch")
            elif row.get("opponent_priors") is not None:
                failures.append(f"{mode}/{key}: used prior lacks server evidence")
            remote = row.get("remote_search") or {}
            rpc = remote.get("rpc_ms")
            if isinstance(rpc, (int, float)) and not isinstance(rpc, bool) and math.isfinite(float(rpc)) and rpc >= 0:
                rpc_values[mode].append(float(rpc))
            else:
                failures.append(f"{mode}/{key}: invalid rpc_ms")
            request_counts[mode] += len(remote.get("request_ids") or [])
            search_cpu_ms[mode] += math.fsum(float(timing["search_ms"]) for timing in remote.get("timings") or [])

    for mode in ("candidate", "comparator"):
        if observed_search_keys[mode] != set(prior_rows[mode]):
            failures.append(f"{mode}: aggregate prior/search key mismatch")
    prior_gates = {
        f"{mode}_{gate}": value
        for mode in ("candidate", "comparator")
        for gate, value in {
            "has_eligible_decisions": telemetry[mode]["eligible"] > 0,
            "complete_coverage": telemetry[mode]["eligible"] == telemetry[mode]["eligible_with_prior"],
            "none_when_ineligible": telemetry[mode]["ineligible_with_prior"] == 0,
            "independent_support_matches": telemetry[mode]["support_disagreements"] == 0,
            "all_actions_legal": telemetry[mode]["illegal_actions"] == 0,
            "all_mass_normalized": telemetry[mode]["nonnormalized"] == 0,
            "zero_errors": telemetry[mode]["errors"] == 0,
            "cross_side_exact_legal_support": telemetry[mode]["cross_side_legal_mismatches"] == 0,
        }.items()
    }

    latency = {
        mode: {
            "count": len(rpc_values[mode]),
            "p50_ms": _nearest_rank(rpc_values[mode], 0.50) if rpc_values[mode] else None,
            "p95_ms": _nearest_rank(rpc_values[mode], 0.95) if rpc_values[mode] else None,
            "max_ms": max(rpc_values[mode]) if rpc_values[mode] else None,
            "remote_requests": request_counts[mode],
            "search_cpu_ms": search_cpu_ms[mode],
        }
        for mode in ("candidate", "comparator")
    }
    candidate_latency = latency["candidate"]
    latency_gates = {
        "candidate_has_latency": candidate_latency["count"] > 0,
        "candidate_p95_within_limit": candidate_latency["p95_ms"] is not None and candidate_latency["p95_ms"] <= 1500,
        "candidate_max_within_limit": candidate_latency["max_ms"] is not None and candidate_latency["max_ms"] <= 2500,
    }

    scores = paired_h2h_inference.pair_scores(
        results,
        prereg["scope"]["candidate"],
        prereg["scope"]["comparator"],
    )
    lower, upper = paired_h2h_inference.bootstrap_interval(
        scores,
        resamples=schedule["bootstrap_resamples"],
        seed=schedule["inference_seed"],
    )
    mean = math.fsum(scores) / len(scores)
    inference = {
        "complete_pairs": len(scores),
        "pair_score_mean": mean,
        "pair_sweeps_candidate": sum(score == 1 for score in scores),
        "pair_splits": sum(score == 0.5 for score in scores),
        "pair_sweeps_comparator": sum(score == 0 for score in scores),
        "bootstrap": {
            "resamples": schedule["bootstrap_resamples"],
            "seed": schedule["inference_seed"],
            "ci95_low": lower,
            "ci95_high": upper,
        },
    }
    inference_gates = {
        "exact_50_pairs": len(scores) == 50,
        "pair_mean_above_half": mean > 0.50,
        "bootstrap_lower_above_half": lower > 0.50,
    }

    preflight_gates = {
        "execution_authorization_exact": (
            authorization.get("status") == "authorized"
            and authorization.get("preregistration_sha256") == _sha256(preregistration_path)
            and authorization.get("local_n100_authorized") is True
            and authorization.get("public_ladder_authorized") is False
        ),
        "base_integrity_passed": (
            base_audit.get("mode") == "adaptive_independent_ensemble_local_smoke_audit"
            and base_audit.get("audit_source_sha256")
            == prereg.get("source_identity", {}).get("adaptive_smoke_audit.py")
            and base_audit.get("inputs", {}).get("results", {}).get("sha256") == _sha256(results_path)
            and base_audit.get("inputs", {}).get("pairs", {}).get("sha256") == _sha256(pairs_path)
            and base_audit.get("inputs", {}).get("protocol", {}).get("sha256") == _sha256(preregistration_path)
            and [item.get("sha256") for item in base_audit.get("inputs", {}).get("prior_decisions", [])]
            == [_sha256(path) for path in prior_decision_paths]
            and all(
                _sha256(log_dir / f"{process['username']}.search.jsonl") == process.get("search_sha256")
                and _sha256(log_dir / f"{process['username']}.protocol.jsonl") == process.get("protocol_sha256")
                for process in base_audit.get("processes", [])
            )
            and base_audit.get("gate", {}).get("passed") is True
            and recomputed_base_audit == base_audit
        ),
        "runtime_preflight_passed": (
            runtime_preflight.get("mode") == "adaptive_ensemble_local_runtime_preflight"
            and runtime_preflight.get("preflight_source_sha256")
            == prereg.get("source_identity", {}).get("local_preflight.py")
            and runtime_preflight.get("preregistration_sha256") == _sha256(preregistration_path)
            and runtime_preflight.get("passed") is True
            and all(runtime_preflight.get("gates", {}).values())
        ),
        "runtime_manifest_matches": runtime_preflight.get("runtime_manifest_sha256") == _sha256(runtime_manifest_path),
        "python_runtime_manifest_matches": runtime_preflight.get("python_runtime_manifest_sha256") == _sha256(python_runtime_manifest_path),
        "remote_preflight_passed": (
            remote_preflight.get("ok") is True
            and remote_preflight.get("preflight_source_sha256")
            == prereg.get("source_identity", {}).get("remote_preflight.py")
            and remote_preflight.get("transport") == "modal"
            and remote_preflight.get("app") == prereg.get("remote_worker", {}).get("app")
            and remote_preflight.get("function") == prereg.get("remote_worker", {}).get("function")
            and remote_preflight.get("url") is None
            and remote_preflight.get("arguments")
            == prereg.get("execution", {}).get("remote_preflight_argv", [])[3:]
            and remote_preflight.get("python_executable_sha256")
            == json.loads(python_runtime_manifest_path.read_text(encoding="utf-8"))["python"]["executable_sha256"]
            and remote_preflight.get("environment")
            == prereg.get("execution", {}).get("remote_preflight_environment")
        ),
        "remote_identity_matches": remote_preflight.get("engine") == frozen.get("expected_engine_identity"),
        "prior_server_logs_clean": all(_text_log_clean(path) for path in prior_server_logs),
        "agent_logs_clean": all(_text_log_clean(path) for path in log_dir.glob("*.log")),
        "showdown_launch_attested": (
            showdown_launch.get("mode") == "verified_showdown_runtime_launch"
            and showdown_launch.get("supervisor_source_sha256")
            == prereg.get("source_identity", {}).get("showdown_supervisor.py")
            and showdown_launch.get("runtime_manifest_sha256") == _sha256(runtime_manifest_path)
            and showdown_launch.get("argv") == prereg.get("execution", {}).get("showdown", {}).get("argv")
            and showdown_launch.get("cwd")
            == str((Path(__file__).resolve().parents[2] / prereg["execution"]["showdown"]["cwd"]).resolve())
            and showdown_launch.get("ready") is True
            and isinstance(showdown_launch.get("pid"), int)
            and showdown_launch.get("node_executable_sha256")
            == json.loads(runtime_manifest_path.read_text(encoding="utf-8"))["node"]["executable_sha256"]
            and showdown_launch.get("server_log") == str(showdown_log_path)
            and showdown_launch.get("pair_directory")
            == str((Path(__file__).resolve().parents[2] / prereg["artifacts"]["pair_directory"]).resolve())
            and showdown_launch.get("environment")
            == prereg.get("execution", {}).get("showdown_environment")
        ),
        "showdown_log_clean": _text_log_clean(showdown_log_path),
    }
    gates = {
        "completion": completion,
        "execution": execution_gates,
        "roles": role_gates,
        "opponent_priors": prior_gates,
        "latency": latency_gates,
        "inference": inference_gates,
        "preflights": preflight_gates,
    }
    mandatory = [value for group in gates.values() for value in group.values()]
    passed = not failures and all(mandatory)
    return {
        "schema_version": 1,
        "mode": "adaptive_independent_ensemble_n100_protocol_audit",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in input_paths.items()
        }
        | {
            "audit_protocol": {"path": str(audit_protocol_path), "sha256": _sha256(audit_protocol_path)},
            "prior_decisions": [{"path": str(path), "sha256": _sha256(path)} for path in prior_decision_paths],
            "prior_server_logs": [{"path": str(path), "sha256": _sha256(path)} for path in prior_server_logs],
        },
        "roles": roles,
        "opponent_priors": telemetry,
        "latency": latency,
        "inference": inference,
        "gates": gates,
        "failures": failures,
        "passed": passed,
        "authorization": {
            "bounded_public_canary_eligible_for_independent_review": passed,
            "public_ladder_authorized": False,
            "unrestricted_ladder_authorized": False,
            "strength_claim_authorized": False,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--audit-protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--prior-decisions", type=Path, action="append", required=True)
    parser.add_argument("--prior-server-log", type=Path, action="append", required=True)
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-preflight", type=Path, required=True)
    parser.add_argument("--python-runtime-manifest", type=Path, required=True)
    parser.add_argument("--remote-preflight", type=Path, required=True)
    parser.add_argument("--showdown-launch", type=Path, required=True)
    parser.add_argument("--showdown-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    result = audit(
        args.preregistration.expanduser().resolve(),
        args.authorization.expanduser().resolve(),
        args.audit_protocol.expanduser().resolve(),
        args.results.expanduser().resolve(),
        args.pairs.expanduser().resolve(),
        args.log_dir.expanduser().resolve(),
        [path.expanduser().resolve() for path in args.prior_decisions],
        [path.expanduser().resolve() for path in args.prior_server_log],
        args.base_audit.expanduser().resolve(),
        args.runtime_manifest.expanduser().resolve(),
        args.python_runtime_manifest.expanduser().resolve(),
        args.runtime_preflight.expanduser().resolve(),
        args.remote_preflight.expanduser().resolve(),
        args.showdown_launch.expanduser().resolve(),
        args.showdown_log.expanduser().resolve(),
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "failures": result["failures"]}, sort_keys=True))


if __name__ == "__main__":
    main()
