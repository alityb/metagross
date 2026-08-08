#!/usr/bin/env python3
"""Audit a completed bounded ladder canary from its immutable artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

from srcs.metagross import run_foul_play, shadow_replay
from srcs.metagross.mcts_contract import (
    ENGINE_CONTRACT,
    ENGINE_SOURCE_SHA256,
    REQUEST_SCHEMA,
)


SAFETY_REASON_PREFIXES = (
    "guaranteed_noop_",
    "repeated_",
    "semantic_no_progress_",
    "terminal_",
    "unqualified_copied_",
    "unqualified_stale_",
)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"calls": 0, "minimum": None, "maximum": None, "mean": None}
    return {
        "calls": len(values),
        "minimum": round(min(values), 3),
        "maximum": round(max(values), 3),
        "mean": round(statistics.fmean(values), 3),
    }


def _same_number(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)
    return left == right


def _recompute_certificate(certificate: dict, decision_index: int) -> dict:
    results = []
    weights = []
    for world in certificate.get("worlds", []):
        pairs = world["pairs"]
        delta_sum = world["delta_mean"] * pairs
        delta_squared_sum = (
            world["within_variance"] + world["delta_mean"] ** 2
        ) * pairs
        results.append(
            {
                "pairs": pairs,
                "baseline_sum": (pairs - delta_sum) / 2,
                "candidate_sum": (pairs + delta_sum) / 2,
                "delta_sum": delta_sum,
                "delta_squared_sum": delta_squared_sum,
                "catastrophic_count": world["catastrophic_count"],
                "candidate_better_count": world["candidate_better_count"],
                "baseline_better_count": world["baseline_better_count"],
                "equal_count": world["equal_count"],
                "baseline_terminal_count": 0,
                "candidate_terminal_count": 0,
                "continuation_iterations_executed": world[
                    "continuation_iterations_executed"
                ],
            }
        )
        weights.append(world["weight"])
    return run_foul_play.independent_holdout_certificate(
        results,
        weights,
        certificate["candidate"],
        certificate["baseline"],
        decision_index,
    )


def _validate_certificate(
    certificate: object,
    decision_index: int,
    label: str,
    failures: list[str],
) -> None:
    if not isinstance(certificate, dict):
        failures.append(f"{label}: certificate is not an object")
        return
    try:
        recomputed = _recompute_certificate(certificate, decision_index)
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(
            f"{label}: certificate cannot be recomputed ({type(exc).__name__})"
        )
        return
    fields = (
        "candidate",
        "baseline",
        "evidence_kind",
        "fresh_worlds",
        "effective_worlds",
        "pairs",
        "posterior_delta",
        "standard_error",
        "alpha",
        "z_value",
        "paired_lower_confidence_bound",
        "positive_world_weight",
        "positive_worlds",
        "sign_p_value",
        "catastrophic_rate",
        "candidate_better_rate",
        "baseline_better_rate",
        "coverage",
        "complete",
        "qualified",
        "checks",
    )
    mismatches = [
        field
        for field in fields
        if not _same_number(certificate.get(field), recomputed.get(field))
    ]
    if mismatches:
        failures.append(f"{label}: recomputed fields differ: {', '.join(mismatches)}")


def _protocol_games(protocol: list[dict], username: str) -> dict[str, dict]:
    games: dict[str, dict] = {}
    for row in protocol:
        if row.get("direction") != "received":
            continue
        message = row.get("message")
        if not isinstance(message, str):
            continue
        tag, lines = shadow_replay._protocol_lines(message)
        if tag is None:
            continue
        game = games.setdefault(
            tag,
            {
                "tag": tag,
                "players": {},
                "winner": None,
                "inactivity": False,
                "forfeit": False,
            },
        )
        for line in lines:
            parts = line.split("|")
            if len(parts) >= 4 and parts[1] == "player":
                if parts[3]:
                    game["players"][parts[2]] = parts[3]
            elif len(parts) >= 3 and parts[1] == "win":
                game["winner"] = parts[2]
            elif "lost due to inactivity" in line:
                game["inactivity"] = True
            elif len(parts) >= 3 and parts[1] == "forfeit":
                game["forfeit"] = True
    for game in games.values():
        opponents = [
            player
            for player in game["players"].values()
            if player.lower() != username.lower()
        ]
        game["opponent"] = opponents[0] if len(opponents) == 1 else None
        winner = game["winner"]
        game["result"] = (
            "win"
            if isinstance(winner, str) and winner.lower() == username.lower()
            else "loss"
            if isinstance(winner, str)
            else "unknown"
        )
        game["end_reason"] = (
            "inactivity"
            if game["inactivity"]
            else "forfeit"
            if game["forfeit"]
            else "normal"
            if game["result"] != "unknown"
            else "interrupted"
        )
    return games


def _rating_summary(capture: Path, username: str) -> dict:
    client_log = capture / "client.log"
    if not client_log.is_file():
        return {"events": [], "start": None, "end": None, "change": None}
    pattern = re.compile(
        rf"{re.escape(username)}'s rating: (\d+) &rarr; <strong>(\d+)</strong>"
        r"<br />\(([+-]\d+) for (winning|losing)\)"
    )
    events = [
        {
            "before": int(match.group(1)),
            "after": int(match.group(2)),
            "delta": int(match.group(3)),
            "result": match.group(4),
        }
        for match in pattern.finditer(client_log.read_text(encoding="utf-8"))
    ]
    return {
        "events": events,
        "start": events[0]["before"] if events else None,
        "end": events[-1]["after"] if events else None,
        "change": events[-1]["after"] - events[0]["before"] if events else None,
    }


def _sent_commands(protocol: list[dict]) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    for row in protocol:
        if row.get("direction") != "sent":
            continue
        room = row.get("room")
        messages = row.get("messages")
        if not isinstance(room, str) or not isinstance(messages, list) or not messages:
            continue
        command = messages[0]
        if isinstance(command, str) and (
            command.startswith("/choose move ")
            or command.startswith("/switch ")
            or command == "/choose default"
        ):
            commands.setdefault(room, []).append(command)
    return commands


def _command_action(command: str, battle: object) -> str:
    if command == "/choose default":
        return "no move"
    if command.startswith("/choose move "):
        parts = command.removeprefix("/choose move ").split()
        suffix = "-tera" if parts[-1:] == ["terastallize"] else ""
        if suffix:
            parts.pop()
        if not parts:
            raise ValueError("move command has no move")
        return _norm("".join(parts)) + suffix
    if command.startswith("/switch "):
        slot = int(command.removeprefix("/switch ")) - 1
        pokemon = battle.request_json["side"]["pokemon"]
        if not 0 <= slot < len(pokemon):
            raise ValueError("switch command slot is outside the request team")
        species = pokemon[slot]["details"].split(",", 1)[0]
        return f"switch {_norm(species)}"
    raise ValueError("unsupported outbound command")


def audit_capture(
    capture: Path,
    *,
    reconstruct: bool = True,
    allow_interrupted: bool = False,
) -> dict:
    protocol, searches, metadata = shadow_replay.load_capture(capture)
    manifest = metadata["manifest"]
    ledger_by_key = metadata.pop("holdout_ledger_rows") or {}
    username = manifest.get("ladder", {}).get("username")
    if not isinstance(username, str) or not username:
        raise ValueError("manifest has no ladder username")
    if reconstruct:
        reconstructed = shadow_replay.reconstruct_battles(protocol, searches, username)
    else:
        reconstructed = searches

    failures: list[str] = []
    if not reconstruct:
        failures.append("protocol reconstruction was explicitly skipped")
    expected_games = manifest.get("ladder", {}).get("games")
    status = manifest.get("status")
    interrupted = allow_interrupted and status == "interrupted"
    if status != "completed" and not interrupted:
        failures.append("manifest status is not completed")
    if set(reconstructed) != set(searches):
        failures.append(
            "protocol reconstruction does not exactly cover captured decisions"
        )

    sent_matches = None
    if reconstruct:
        sent_matches = 0
        sent_commands = _sent_commands(protocol)
        for tag in metadata["battle_tags"]:
            keys = sorted(key for key in searches if key[0] == tag)
            commands = sent_commands.get(tag, [])
            if len(commands) != len(keys):
                failures.append(
                    f"{tag}: {len(commands)} outbound actions for {len(keys)} decisions"
                )
                continue
            for key, command in zip(keys, commands, strict=True):
                try:
                    sent_action = _command_action(command, reconstructed[key])
                except (KeyError, TypeError, ValueError) as exc:
                    failures.append(
                        f"{tag} decision {key[1]}: invalid outbound command "
                        f"({type(exc).__name__})"
                    )
                    continue
                selected_action = searches[key].get("choice")
                if sent_action != selected_action:
                    failures.append(
                        f"{tag} decision {key[1]}: selected {selected_action} "
                        f"but sent {sent_action}"
                    )
                    continue
                sent_matches += 1

    games = _protocol_games(protocol, username)
    incomplete_games = [game for game in games.values() if game["result"] == "unknown"]
    completed_games = [game for game in games.values() if game["result"] != "unknown"]
    if interrupted:
        if len(games) > expected_games:
            failures.append(f"target was {expected_games} games but found {len(games)}")
        if len(incomplete_games) > 1:
            failures.append("interrupted capture contains multiple incomplete games")
    else:
        if len(games) != expected_games:
            failures.append(f"expected {expected_games} games but found {len(games)}")
        if incomplete_games:
            failures.append("at least one game has no terminal outcome")
    if any(game["opponent"] is None for game in games.values()):
        failures.append("at least one game has invalid player metadata")

    remote_config = (
        manifest.get("search", {}).get("modal", {})
        or manifest.get("search", {}).get("http", {})
    )
    expected_native = remote_config.get("engine_sha256")
    if remote_config.get("schema") == REQUEST_SCHEMA:
        expected_contract = remote_config.get("contract")
        expected_source = remote_config.get("source_sha256")
        if expected_contract != ENGINE_CONTRACT or expected_source != ENGINE_SOURCE_SHA256:
            failures.append("manifest v5 engine identity differs from the auditor")
    else:
        first_remote = next(iter(searches.values())).get("remote_search") or {}
        first_engine = first_remote.get("engine") or {}
        expected_contract = first_engine.get("contract")
        expected_source = first_engine.get("source_sha256")
    expected_horizons = list(
        (manifest.get("holdout") or {}).get(
            "continuation_horizons", run_foul_play.HOLDOUT_CONTINUATION_HORIZONS
        )
    )
    search_latencies: list[float] = []
    holdout_latencies: list[float] = []
    holdout_world_calls = 0
    remote_worlds = 0
    holdout_rows = []
    admission_rows = []
    safety_rows = []
    reasons: Counter[str] = Counter()
    by_game: dict[str, list[dict]] = {tag: [] for tag in games}

    for key in sorted(searches):
        tag, decision_index = key
        row = searches[key]
        override = row.get("choice_override")
        label = f"{tag} decision {decision_index}"
        if not isinstance(override, dict):
            failures.append(f"{label}: missing choice override telemetry")
            continue
        reason = override.get("reason")
        reasons[str(reason)] += 1
        request_actions = override.get("request_actions")
        if (
            not isinstance(request_actions, list)
            or row.get("choice") not in request_actions
        ):
            failures.append(f"{label}: final choice is absent from request actions")
        if override.get("missing_request_actions"):
            failures.append(f"{label}: search omitted legal request actions")

        remote = row.get("remote_search")
        if not isinstance(remote, dict):
            failures.append(f"{label}: missing remote-search telemetry")
            continue
        engine = remote.get("engine")
        if not isinstance(engine, dict):
            failures.append(f"{label}: missing remote engine identity")
        else:
            expected_identity = {
                "contract": expected_contract,
                "source_sha256": expected_source,
                "native_sha256": expected_native,
            }
            for field, expected in expected_identity.items():
                if engine.get(field) != expected:
                    failures.append(f"{label}: remote engine {field} mismatch")
        rpc_ms = remote.get("rpc_ms")
        if isinstance(rpc_ms, (int, float)) and not isinstance(rpc_ms, bool):
            search_latencies.append(float(rpc_ms))
        else:
            failures.append(f"{label}: invalid search latency")
        worlds = remote.get("worlds")
        if isinstance(worlds, int) and not isinstance(worlds, bool) and worlds > 0:
            remote_worlds += worlds
        else:
            failures.append(f"{label}: invalid remote world count")

        holdout = override.get("holdout")
        holdout_calls = remote.get("holdout", [])
        if key in ledger_by_key:
            ledger = ledger_by_key[key]
            run_seed = (manifest.get("rng") or {}).get("run_seed")
            selection = ledger.get("selection_cohort") or {}
            expected_selection_seed = shadow_replay.derive_seed(
                run_seed, "selection-worlds", tag, decision_index, 0
            )
            expected_selection_ids = [
                shadow_replay.deterministic_request_id(
                    run_seed,
                    tag,
                    decision_index,
                    index,
                    channel="selection-search-request",
                )
                for index in range(worlds if isinstance(worlds, int) else 0)
            ]
            if (
                selection.get("sampling_seed") != expected_selection_seed
                or selection.get("request_ids") != expected_selection_ids
            ):
                failures.append(f"{label}: selection derivation differs")
            if reconstruct and isinstance(worlds, int) and worlds > 0:
                selection_states, selection_weights = shadow_replay._fresh_worlds(
                    reconstructed[key], worlds, expected_selection_seed
                )
                if (
                    selection.get("state_hashes")
                    != [shadow_replay.state_sha256(state) for state in selection_states]
                    or selection.get("weights") != selection_weights
                ):
                    failures.append(f"{label}: reconstructed selection cohort differs")
            panel = ledger.get("certification")
            if panel is None:
                if holdout is not None or holdout_calls:
                    failures.append(f"{label}: empty v5 panel has holdout evidence")
            else:
                try:
                    evidence_by_action = shadow_replay._recompute_captured_v5_panel(panel)
                except (KeyError, TypeError, ValueError) as exc:
                    failures.append(
                        f"{label}: v5 panel cannot be recomputed ({type(exc).__name__})"
                    )
                    evidence_by_action = {}
                expected_calls = []
                for panel_row in panel.get("candidate_panel", []):
                    combined = evidence_by_action.get(panel_row.get("action"), {})
                    expected_calls.extend(
                        (panel_row.get("rank"), panel_row.get("action"), horizon)
                        for horizon in combined.get("executed_horizons", [])
                    )
                observed_calls = [
                    (
                        call.get("candidate_rank"),
                        call.get("candidate"),
                        call.get("continuation_steps"),
                    )
                    for call in holdout_calls
                ] if isinstance(holdout_calls, list) else []
                if observed_calls != expected_calls:
                    failures.append(f"{label}: v5 adaptive holdout calls differ")
                cohort = panel.get("certification_cohort") or {}
                expected_certification_seed = shadow_replay.derive_seed(
                    run_seed, "certification-worlds", tag, decision_index, 0
                )
                expected_tape_seeds = [
                    shadow_replay.derive_seed(
                        run_seed, "holdout-tape", tag, decision_index, index
                    )
                    for index in range(len(cohort.get("state_hashes") or ()))
                ]
                if (
                    cohort.get("sampling_seed") != expected_certification_seed
                    or cohort.get("tape_seeds") != expected_tape_seeds
                    or panel.get("opponent_uniform_mix")
                    != (manifest.get("holdout") or {}).get("opponent_uniform_mix")
                ):
                    failures.append(f"{label}: v5 certification derivation differs")
                if reconstruct:
                    certification_states, certification_weights = (
                        shadow_replay._fresh_worlds(
                            reconstructed[key],
                            len(cohort.get("state_hashes") or ()),
                            expected_certification_seed,
                        )
                    )
                    if (
                        cohort.get("state_hashes")
                        != [
                            shadow_replay.state_sha256(state)
                            for state in certification_states
                        ]
                        or cohort.get("weights") != certification_weights
                    ):
                        failures.append(
                            f"{label}: reconstructed certification cohort differs"
                        )
                for call in holdout_calls if isinstance(holdout_calls, list) else ():
                    expected_request_ids = [
                        shadow_replay.deterministic_request_id(
                            run_seed,
                            tag,
                            decision_index,
                            f"{call.get('candidate_rank')}:"
                            f"{call.get('continuation_steps')}:{index}",
                            channel="certification-request",
                        )
                        for index in range(len(cohort.get("state_hashes") or ()))
                    ]
                    if (
                        call.get("seeds") != cohort.get("tape_seeds")
                        or call.get("state_hashes") != cohort.get("state_hashes")
                        or call.get("worlds") != len(cohort.get("state_hashes") or ())
                        or call.get("request_ids") != expected_request_ids
                        or call.get("opponent_priors")
                        != (
                            [list(prior) for prior in panel.get("opponent_priors")]
                            if panel.get("opponent_priors")
                            else None
                        )
                    ):
                        failures.append(f"{label}: v5 holdout cohort differs")
                    else:
                        holdout_world_calls += call["worlds"]
                    latency = call.get("rpc_ms")
                    if isinstance(latency, (int, float)) and not isinstance(
                        latency, bool
                    ):
                        holdout_latencies.append(float(latency))
                    else:
                        failures.append(f"{label}: invalid holdout latency")
                holdout_rows.append(row)
        elif holdout is None:
            if holdout_calls:
                failures.append(f"{label}: holdout timings exist without a certificate")
        elif not isinstance(holdout, dict):
            failures.append(f"{label}: holdout certificate is not an object")
        else:
            holdout_rows.append(row)
            candidate = override.get("raw_choice")
            baseline = override.get("baseline")
            if (
                holdout.get("candidate") != candidate
                or holdout.get("baseline") != baseline
            ):
                failures.append(f"{label}: holdout evaluated different frozen actions")
            certificates = holdout.get("certificates")
            if not isinstance(certificates, dict) or sorted(certificates) != [
                str(horizon) for horizon in expected_horizons
            ]:
                failures.append(f"{label}: holdout does not cover required horizons")
                certificates = {}
            for horizon in expected_horizons:
                _validate_certificate(
                    certificates.get(str(horizon)),
                    decision_index,
                    f"{label} horizon {horizon}",
                    failures,
                )
            if certificates:
                try:
                    combined = run_foul_play.combined_holdout_certificate(
                        {int(horizon): value for horizon, value in certificates.items()}
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    failures.append(
                        f"{label}: combined certificate is invalid ({type(exc).__name__})"
                    )
                else:
                    for field in (
                        "candidate",
                        "baseline",
                        "evidence_kind",
                        "horizons",
                        "complete",
                        "coverage",
                        "pairs",
                        "posterior_delta",
                        "paired_lower_confidence_bound",
                        "catastrophic_rate",
                        "qualified",
                    ):
                        if not _same_number(holdout.get(field), combined.get(field)):
                            failures.append(f"{label}: combined {field} differs")
            if not isinstance(holdout_calls, list) or len(holdout_calls) != len(
                expected_horizons
            ):
                failures.append(f"{label}: invalid holdout timing coverage")
            else:
                steps = [call.get("continuation_steps") for call in holdout_calls]
                seeds = [call.get("seeds") for call in holdout_calls]
                if steps != expected_horizons:
                    failures.append(f"{label}: holdout timing horizons differ")
                if not seeds or any(seed_list != seeds[0] for seed_list in seeds[1:]):
                    failures.append(f"{label}: horizons did not share the same seeds")
                for call in holdout_calls:
                    call_worlds = call.get("worlds")
                    call_seeds = call.get("seeds")
                    if not isinstance(call_worlds, int) or not isinstance(
                        call_seeds, list
                    ):
                        failures.append(f"{label}: invalid holdout call metadata")
                        continue
                    if call_worlds != len(call_seeds):
                        failures.append(
                            f"{label}: holdout seeds do not cover every world"
                        )
                    holdout_world_calls += call_worlds
                    latency = call.get("rpc_ms")
                    if isinstance(latency, (int, float)) and not isinstance(
                        latency, bool
                    ):
                        holdout_latencies.append(float(latency))
                    else:
                        failures.append(f"{label}: invalid holdout latency")

        admitted = override.get("search_override_admitted") is True
        if admitted:
            admission_rows.append(row)
            if not isinstance(holdout, dict) or holdout.get("qualified") is not True:
                failures.append(f"{label}: search override lacked a qualified holdout")
            if reason != "independent_holdout_qualified_search_override":
                failures.append(f"{label}: admitted override has the wrong reason")
        if isinstance(reason, str) and reason.startswith(SAFETY_REASON_PREFIXES):
            safety_rows.append(row)
        by_game.setdefault(tag, []).append(row)

    game_rows = []
    for tag, game in sorted(games.items()):
        rows = by_game.get(tag, [])
        decisions = [row["context"].get("battle_turn") for row in rows]
        game_rows.append(
            {
                **game,
                "decisions": len(rows),
                "last_decision_turn": max(decisions) if decisions else None,
                "holdouts": sum(
                    isinstance(row.get("choice_override", {}).get("holdout"), dict)
                    for row in rows
                ),
                "admissions": [
                    {
                        "decision_idx": row["context"]["decision_idx"],
                        "battle_turn": row["context"]["battle_turn"],
                        "baseline": row["choice_override"]["baseline"],
                        "candidate": row["choice_override"]["raw_choice"],
                    }
                    for row in rows
                    if row.get("choice_override", {}).get("search_override_admitted")
                    is True
                ],
            }
        )

    rating = _rating_summary(capture, username)
    if len(rating["events"]) != len(completed_games):
        failures.append("rating event count does not match completed game count")
    completed_game_rows = [game for game in game_rows if game["result"] != "unknown"]
    wins = sum(game["result"] == "win" for game in completed_game_rows)
    losses = sum(game["result"] == "loss" for game in completed_game_rows)
    inactivity_wins = sum(
        game["result"] == "win" and game["inactivity"] for game in completed_game_rows
    )
    forfeit_wins = sum(
        game["result"] == "win" and game["forfeit"] for game in completed_game_rows
    )
    normal_wins = sum(
        game["result"] == "win" and game["end_reason"] == "normal"
        for game in completed_game_rows
    )
    normal_losses = sum(
        game["result"] == "loss" and game["end_reason"] == "normal"
        for game in completed_game_rows
    )
    performance_conclusion = (
        f"failed early ladder gate: normal-combat record {normal_wins}-{normal_losses}"
        if interrupted
        else (
            "inconclusive: the sample is only three games and one win was by inactivity"
            if expected_games == 3 and inactivity_wins
            else "inconclusive: bounded canary samples are not a promotion test"
        )
    )
    return {
        "schema": 1,
        "capture": {
            "path": str(capture),
            "digest": metadata["capture_digest"],
            "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"),
            "status": manifest.get("status"),
            "target_games": expected_games,
        },
        "identity": {
            "username": username,
            "policy_sha256": manifest.get("policy", {}).get("sha256"),
            "engine_contract": ENGINE_CONTRACT,
            "engine_source_sha256": ENGINE_SOURCE_SHA256,
            "engine_native_sha256": expected_native,
        },
        "integrity": {
            "passed": not failures,
            "failures": failures,
            "protocol_reconstructed_decisions": len(reconstructed) if reconstruct else 0,
            "selected_actions_matching_outbound_commands": sent_matches or 0,
            "capture_decisions": len(searches),
            "mask_fallbacks": sum(
                bool(row.get("mask_fallback"))
                for row in metadata["decision_rows"].values()
            ),
        },
        "result": {
            "observed_games": len(game_rows),
            "completed_games": len(completed_game_rows),
            "incomplete_games": len(incomplete_games),
            "wins": wins,
            "losses": losses,
            "inactivity_wins": inactivity_wins,
            "forfeit_wins": forfeit_wins,
            "normal_wins": normal_wins,
            "normal_losses": normal_losses,
            "rating": rating,
        },
        "search": {
            "decisions": len(searches),
            "policy_search_agreements": sum(
                row.get("choice_override", {}).get("raw_choice")
                == row.get("choice_override", {}).get("baseline")
                for row in searches.values()
            ),
            "holdouts": len(holdout_rows),
            "dual_horizon_admissions": len(admission_rows),
            "safety_corrections": len(safety_rows),
            "reasons": dict(sorted(reasons.items())),
            "remote_worlds": remote_worlds,
            "holdout_world_calls": holdout_world_calls,
            "search_latency_ms": _latency_summary(search_latencies),
            "holdout_latency_ms": _latency_summary(holdout_latencies),
        },
        "games": game_rows,
        "verdict": {
            "operational_and_safety": "pass" if not failures else "fail",
            "performance": performance_conclusion,
            "automatic_continuation": False,
        },
    }


def render_markdown(report: dict) -> str:
    result = report["result"]
    search = report["search"]
    integrity = report["integrity"]
    rating = result["rating"]
    lines = [
        "# Bounded Ladder Artifact Audit",
        "",
        f"- Artifact integrity: **{'PASS' if integrity['passed'] else 'FAIL'}**",
        f"- Ladder result: **{result['wins']}-{result['losses']}** "
        f"({result['normal_wins']}-{result['normal_losses']} normal combat; "
        f"{result['inactivity_wins']} inactivity, {result['forfeit_wins']} forfeit wins)",
        f"- Completion: **{result['completed_games']} / "
        f"{report['capture']['target_games']}** target games; "
        f"{result['incomplete_games']} interrupted game",
        f"- Rating: **{rating['start']} -> {rating['end']} ({rating['change']:+d})**"
        if rating["change"] is not None
        else "- Rating: unavailable",
        f"- Decisions reconstructed: **{integrity['protocol_reconstructed_decisions']} / "
        f"{integrity['capture_decisions']}**",
        f"- Selected actions matching outbound commands: "
        f"**{integrity['selected_actions_matching_outbound_commands']} / "
        f"{integrity['capture_decisions']}**",
        f"- Independent dual-horizon holdouts: **{search['holdouts']}**",
        f"- Dual-horizon admissions: **{search['dual_horizon_admissions']}**",
        f"- Deterministic safety corrections: **{search['safety_corrections']}**",
        "- Automatic continuation: **disabled**",
        "",
        "## Per Game",
        "",
    ]
    for game in report["games"]:
        suffix = f" ({game['end_reason']})" if game["end_reason"] != "normal" else ""
        lines.append(
            f"- `{game['tag']}`: {game['result']} vs `{game['opponent']}`{suffix}; "
            f"{game['decisions']} decisions, {game['holdouts']} holdouts, "
            f"{len(game['admissions'])} admissions."
        )
        for admission in game["admissions"]:
            lines.append(
                f"  - T{admission['battle_turn']}: `{admission['candidate']}` over "
                f"`{admission['baseline']}`."
            )
    lines.extend(
        [
            "",
            "## Infrastructure",
            "",
            f"- Search RPC latency: mean {search['search_latency_ms']['mean']} ms, "
            f"max {search['search_latency_ms']['maximum']} ms across "
            f"{search['search_latency_ms']['calls']} calls.",
            f"- Holdout RPC latency: mean {search['holdout_latency_ms']['mean']} ms, "
            f"max {search['holdout_latency_ms']['maximum']} ms across "
            f"{search['holdout_latency_ms']['calls']} calls.",
            f"- Selection reasons: `{json.dumps(search['reasons'], sort_keys=True)}`.",
            "",
            "## Verdict",
            "",
            (
                "The bounded run passed artifact, remote-engine, legality, and dual-horizon "
                "integrity checks."
                if integrity["passed"]
                else "The bounded run failed one or more integrity checks."
            )
            + f" Performance verdict: {report['verdict']['performance']}. "
            + "This audit does not justify automatic continuation or promotion.",
        ]
    )
    if integrity["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in integrity["failures"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument(
        "--allow-interrupted",
        action="store_true",
        help="audit a deliberately interrupted partial run without requiring its target count",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = audit_capture(
        args.capture.resolve(),
        reconstruct=not args.skip_reconstruction,
        allow_interrupted=args.allow_interrupted,
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "AUDIT.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"integrity": report["integrity"]["passed"], **report["result"]}))
    print(output)
    if not report["integrity"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
