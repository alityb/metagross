#!/usr/bin/env python3
"""Outcome-blind role audit, then one-time scoring of immutable Cycle40 bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from experimental.src.scripts.monitor_cycle19_operational_smoke import (
    rows,
    to_id,
)
from experimental.src.scripts.monitor_cycle21_registered_form_smoke import player_roles
from experimental.src.scripts.summarize_cycle19_h2h import (
    engine_provenance,
    validate_candidate_file,
    wilson,
)
from experimental.src.scripts.summarize_cycle33_h2h import registration_audit
from experimental.src.scripts.verify_cycle33_h2h_freeze import verify as verify_cycle40

ROOT = Path(__file__).resolve().parents[3]
C40 = ROOT / "experimental/runs/search_native_v2_cycle40_integrated_h2h_20260816"
RUN = ROOT / "experimental/runs/search_native_v2_cycle41_blind_scorer_repair_20260816"
CONTROLLER = "metagross-cycle19-equal8192-production-selector/v1"
MOVE_SCHEMA = "metagross-causal-move-conversion-receipt/v1"
ABILITY_SCHEMA = "metagross-certified-ability-installation/v2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def search_identity(row: dict) -> tuple[str, int, int, str]:
    context = row.get("context") or {}
    tag = context.get("tag")
    index = context.get("decision_idx")
    root = hashlib.sha256(f"terminal-mcts-live\0{tag}\0{index}".encode()).hexdigest()
    return tag, context.get("rqid"), index, root


def receipt_key(row: dict) -> tuple:
    context = row.get("execution_context") or {}
    return (
        context.get("phase"),
        row.get("observer_role"),
        context.get("battle_tag"),
        context.get("rqid"),
        context.get("decision_index"),
        context.get("root_id"),
    )


def username(path: Path, suffix: str) -> str:
    return path.name.removesuffix(suffix)


def public_role(protocol_path: Path, external_username: str) -> str:
    mapping = player_roles(rows(protocol_path))
    from experimental.src.scripts.monitor_cycle19_operational_smoke import to_id

    role = mapping.get(to_id(external_username))
    if role not in {"p1", "p2"}:
        raise RuntimeError(f"missing public player-role mapping: {external_username}")
    return role


def validate_move_receipt_envelope(row: dict) -> None:
    context = row.get("execution_context") or {}
    if row.get("schema") != MOVE_SCHEMA:
        raise RuntimeError("wrong move receipt schema")
    if row.get("observer_role") not in {"p1", "p2"} or row.get("swap") is not False:
        raise RuntimeError("invalid public role or engine orientation")
    if (context.get("phase"), context.get("cohort")) not in {
        ("production_control", "adaptive_root_search"),
        ("equal8192_candidate", "fixed_two_by_eight"),
    }:
        raise RuntimeError("unknown receipt cohort")
    nested = row.get("move_receipt") or {}
    if (
        nested.get("protocol_sha256") != row.get("protocol_sha256")
        or nested.get("battle_tag") != context.get("battle_tag")
        or not isinstance(nested.get("moves"), list)
        or not isinstance(nested.get("derived_executions"), list)
    ):
        raise RuntimeError("move receipt nested identity changed")


def load_move_receipts(run: Path) -> dict[tuple, list[dict]]:
    indexed: dict[tuple, list[dict]] = {}
    files = sorted((run / "move-receipts").glob("*.jsonl"))
    if len(files) != 40:
        raise RuntimeError(f"expected 40 move receipt files, found {len(files)}")
    for path in files:
        file_rows = rows(path)
        if not file_rows:
            raise RuntimeError(f"empty move receipt file: {path}")
        for row in file_rows:
            validate_move_receipt_envelope(row)
            nested = row["move_receipt"]
            for move in nested["moves"]:
                if (
                    move.get("disable_authority") not in {"causal_disable", "world_mechanical_disable"}
                    or isinstance(move.get("current_pp"), bool)
                    or not isinstance(move.get("current_pp"), int)
                    or isinstance(move.get("max_pp"), bool)
                    or not isinstance(move.get("max_pp"), int)
                    or not 0 <= move["current_pp"] <= move["max_pp"]
                    or not isinstance(move.get("world_disabled"), bool)
                ):
                    raise RuntimeError("invalid PP/disable receipt")
                if move["disable_authority"] == "causal_disable" and move["world_disabled"] is not True:
                    raise RuntimeError("causal disable absent from world")
            indexed.setdefault(receipt_key(row), []).append(row)
    return indexed


def pop_cohort(
    indexed: dict[tuple, list[dict]],
    phase: str,
    role: str,
    base: tuple[str, int, int, str],
    decision_time_ns: int,
) -> dict:
    key = (phase, role, *base)
    cohort = indexed.pop(key, [])
    if not cohort:
        raise RuntimeError(f"missing {phase} cohort for authenticated {role}")
    if any(int(row.get("receipt_time_ns", 0)) > decision_time_ns for row in cohort):
        raise RuntimeError("receipt occurred after decision completion")
    protocols = {row.get("protocol_sha256") for row in cohort}
    if len(protocols) != 1:
        raise RuntimeError("cohort spans causal prefixes")
    contexts = [row["execution_context"] for row in cohort]
    if phase == "production_control":
        declared = {row.get("declared_world_count") for row in contexts}
        if len(declared) != 1 or next(iter(declared)) not in {16, 32}:
            raise RuntimeError("production declaration invalid")
        count = int(next(iter(declared)))
        if len(cohort) != count or {row.get("conversion_index") for row in contexts} != set(range(count)):
            raise RuntimeError("production receipt cohort incomplete")
    else:
        count = 16
        cells = {(row.get("schedule_index"), row.get("world_index")) for row in contexts}
        if (
            len(cohort) != 16
            or cells != {(schedule, world) for schedule in range(2) for world in range(8)}
            or {row.get("conversion_index") for row in contexts} != set(range(16))
            or any(row.get("declared_world_count") != 16 for row in contexts)
        ):
            raise RuntimeError("candidate receipt cohort incomplete")
    return {"count": count, "protocol_sha256": next(iter(protocols))}


def validate_public_execution(search_path: Path) -> None:
    protocol_path = search_path.with_name(
        search_path.name.removesuffix(".search.jsonl") + ".protocol.jsonl"
    )
    protocol = rows(protocol_path)
    sent = [
        row for row in protocol
        if row.get("direction") == "sent"
        and isinstance(row.get("messages"), list)
        and row["messages"]
        and str(row["messages"][0]).startswith(("/choose move ", "/switch "))
    ]
    search_rows = rows(search_path)
    if len(sent) != len(search_rows):
        raise RuntimeError("candidate public-execution decision count changed")
    role = public_role(protocol_path, username(search_path, ".search.jsonl"))
    for search_row, sent_row in zip(search_rows, sent, strict=True):
        teacher = ((search_row.get("choice_override") or {}).get("terminal_mcts_teacher") or {})
        selected = str(teacher.get("selected_action") or "")
        if public_action_adjudication(protocol, int(sent_row["time_ns"]), role, selected) is None:
            raise RuntimeError("candidate selected action lacks public adjudication")


def public_action_adjudication(
    protocol: list[dict], after_ns: int, role: str, selected: str
) -> str | None:
    if role not in {"p1", "p2"}:
        raise RuntimeError("public adjudication requires an authenticated role")
    target = to_id(selected.removeprefix("switch ").removesuffix("-tera"))
    for row in protocol:
        if int(row.get("time_ns", 0)) <= after_ns or row.get("direction") not in {
            "received", "reconnect_received"
        }:
            continue
        stop_after_message = False
        for line in str(row.get("message", "")).splitlines():
            parts = line.split("|")
            actor = parts[2] if len(parts) >= 3 else ""
            own_active = actor.startswith(f"{role}a:")
            if selected.startswith("switch "):
                if (
                    len(parts) >= 4
                    and parts[1] in {"switch", "drag", "replace"}
                    and own_active
                    and to_id(parts[3].split(",", 1)[0]) == target
                ):
                    return "executed_switch"
            elif len(parts) >= 4 and parts[1] == "move" and own_active and to_id(parts[3]) == target:
                return "executed_move"
            elif not selected.startswith("switch ") and len(parts) >= 3 and parts[1] == "cant" and own_active:
                return "public_cant"
            elif not selected.startswith("switch ") and len(parts) >= 3 and parts[1] == "faint" and own_active:
                return "fainted_before_action"
            elif (
                not selected.startswith("switch ")
                and len(parts) >= 4
                and parts[1] == "-activate"
                and own_active
                and to_id(parts[3]) == "confusion"
            ):
                return "confusion_self_hit"
            elif line.startswith("|request|"):
                try:
                    request = json.loads(line.removeprefix("|request|"))
                except json.JSONDecodeError as exc:
                    raise RuntimeError("public adjudication saw malformed request") from exc
                if request.get("wait") is not True:
                    stop_after_message = True
        if stop_after_message:
            return None
    return None


def validate_ability_receipts(run: Path, move_evidence: dict[tuple, tuple[str, str]]) -> dict:
    count = installations = 0
    for path in sorted((run / "ability-receipts").glob("*.jsonl")):
        for row in rows(path):
            count += 1
            if (
                row.get("schema") != ABILITY_SCHEMA
                or row.get("observer_role") not in {"p1", "p2"}
                or row.get("swap") is not False
                or not isinstance(row.get("installations"), list)
                or not row["installations"]
            ):
                raise RuntimeError("invalid ability receipt envelope")
            key = receipt_key(row)
            expected = move_evidence.get(key)
            if expected != (row.get("observer_role"), row.get("protocol_sha256")):
                raise RuntimeError("ability receipt lacks matching move cohort/role/protocol")
            for installation in row["installations"]:
                installations += 1
                if (
                    installation.get("authority") not in {
                        "explicit_public_event", "rule_implied_switch_reactivation"
                    }
                    or not isinstance(installation.get("exact_public_species"), str)
                    or not installation["exact_public_species"]
                    or not isinstance(installation.get("installed_current_ability"), str)
                    or not installation["installed_current_ability"]
                    or isinstance(installation.get("slot"), bool)
                    or not isinstance(installation.get("slot"), int)
                    or not 0 <= installation["slot"] < 6
                    or not isinstance(installation.get("update_base"), bool)
                ):
                    raise RuntimeError("invalid certified ability installation")
    return {"receipt_rows": count, "installations": installations}


def preflight() -> dict:
    manifest = verify_cycle40(C40 / "H2H_PREMEASUREMENT_MANIFEST.json")
    pair_path = C40 / "h2h-result.json.pairs.json"
    protocol_paths = sorted((C40 / "h2h-logs").glob("*.protocol.jsonl"))
    search_paths = sorted((C40 / "h2h-logs").glob("*.search.jsonl"))
    logs = sorted((C40 / "h2h-logs").glob("*.log"))
    if not (len(protocol_paths) == len(search_paths) == len(logs) == 40):
        raise RuntimeError("Cycle40 spawned artifact denominator changed")
    registration = registration_audit(C40, pair_path, protocol_paths)
    witness = json.loads((C40 / "REGISTRATION_CONSUMPTION.json").read_text())
    registered = {row["username"]: row["side"] for row in witness["registrations"]}
    if len(registered) != 40:
        raise RuntimeError("registration username denominator changed")
    for log in logs:
        engine_provenance(log, manifest["engine"]["native_sha256"])
    for path in protocol_paths:
        if any(row.get("direction") in {"send_failure", "send_rejected", "reconnect"} for row in rows(path)):
            raise RuntimeError("protocol operational failure")

    candidate_paths = []
    comparator_paths = []
    role_by_search: dict[Path, str] = {}
    for path in search_paths:
        external = username(path, ".search.jsonl")
        if external not in registered:
            raise RuntimeError("search stream lacks registration witness")
        protocol = path.with_name(external + ".protocol.jsonl")
        role = public_role(protocol, external)
        if role != registered[external]:
            raise RuntimeError("registration role disagrees with public player line")
        role_by_search[path] = role
        has_candidate = any(
            ((row.get("choice_override") or {}).get("terminal_mcts_teacher") or {}).get("controller_schema") == CONTROLLER
            for row in rows(path)
        )
        (candidate_paths if has_candidate else comparator_paths).append(path)
    if len(candidate_paths) != 20 or len(comparator_paths) != 20:
        raise RuntimeError("candidate/comparator assignment changed")
    if {role: sum(role_by_search[path] == role for path in candidate_paths) for role in ("p1", "p2")} != {"p1": 10, "p2": 10}:
        raise RuntimeError("candidate public role balance changed")

    indexed = load_move_receipts(C40)
    move_evidence = {
        key: (key[1], rows_for_key[0].get("protocol_sha256"))
        for key, rows_for_key in indexed.items()
    }
    decisions = overrides = production_count = candidate_count = 0
    latencies: list[float] = []
    for path in search_paths:
        role = role_by_search[path]
        is_candidate = path in candidate_paths
        if is_candidate:
            count, observed, changed = validate_candidate_file(path)
            validate_public_execution(path)
            decisions += count
            overrides += changed
            latencies.extend(observed)
        elif any((row.get("choice_override") or {}).get("terminal_mcts_teacher") for row in rows(path)):
            raise RuntimeError("candidate controller leaked into comparator")
        for row in rows(path):
            base = search_identity(row)
            production = pop_cohort(indexed, "production_control", role, base, int(row["time_ns"]))
            production_count += production["count"]
            if is_candidate:
                candidate = pop_cohort(indexed, "equal8192_candidate", role, base, int(row["time_ns"]))
                candidate_count += candidate["count"]
                if candidate["protocol_sha256"] != production["protocol_sha256"]:
                    raise RuntimeError("candidate and production causal prefixes differ")
    if indexed:
        raise RuntimeError(f"unjoined move receipt cohorts remain: {len(indexed)}")
    ability = validate_ability_receipts(C40, move_evidence)
    return {
        "schema": "metagross-cycle41-outcome-blind-role-audit/v1",
        "status": "pass",
        "outcome_or_win_fields_read": False,
        "result_bytes_sha256": sha(C40 / "h2h-result.json"),
        "games_expected_but_outcomes_unopened": 20,
        "mirrored_pairs_expected": 10,
        "candidate_streams": 20,
        "comparator_streams": 20,
        "candidate_public_roles": {"p1": 10, "p2": 10},
        "public_role_authority": "registration_witness_plus_exact_public_player_line",
        "engine_orientation": "swap_false_local_observer_is_engine_side_one_independent_of_public_role",
        "candidate_decisions": decisions,
        "candidate_overrides": overrides,
        "candidate_mean_latency_ms": statistics.fmean(latencies),
        "production_conversion_receipts": production_count,
        "candidate_conversion_receipts": candidate_count,
        "ability_receipts": ability,
        "registration": registration,
        "unjoined_receipt_cohorts": 0,
        "semantic_operational_failures": 0,
        "sealed93_rows_read": 0,
    }


def verify_cycle41_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "metagross-cycle41-preoutcome-manifest/v1":
        raise RuntimeError("wrong Cycle41 manifest")
    for row in payload["files"]:
        source = Path(row["path"])
        if not source.is_file() or sha(source) != row["sha256"]:
            raise RuntimeError(f"Cycle41 frozen file mismatch: {source}")
    return payload


def score(manifest_path: Path) -> dict:
    manifest = verify_cycle41_manifest(manifest_path)
    audit = json.loads((RUN / "PREOUTCOME_AUDIT.json").read_text())
    if audit.get("status") != "pass" or audit.get("outcome_or_win_fields_read") is not False:
        raise RuntimeError("Cycle41 outcome-blind preflight was not admitted")
    result_path = C40 / "h2h-result.json"
    if sha(result_path) != manifest["immutable_cycle40_result_sha256"]:
        raise RuntimeError("Cycle40 result bytes changed before score opening")

    # This is the first outcome-field read in Cycle41.
    payload = json.loads(result_path.read_text())
    games = payload.get("games") or []
    summary = payload.get("summary") or {}
    if (
        len(games) != 20
        or summary.get("completed_games") != 20
        or summary.get("void_games") != 0
        or summary.get("decisive_games") != 20
        or any(game.get("void") or game.get("error") or game.get("winner") not in {"agent_a", "agent_b"} for game in games)
    ):
        raise RuntimeError("Cycle41 outcome denominator or integrity invalid")
    if (
        sum(game.get("challenger") == "agent_a" for game in games) != 10
        or sum(game.get("acceptor") == "agent_a" for game in games) != 10
    ):
        raise RuntimeError("candidate challenger/acceptor role balance invalid")
    by_pair: dict[int, list[dict]] = {}
    for game in games:
        by_pair.setdefault(int(game["pair_index"]), []).append(game)
    if set(by_pair) != set(range(1, 11)):
        raise RuntimeError("mirrored pair indices incomplete")
    pair_results = []
    for index, group in sorted(by_pair.items()):
        if (
            len(group) != 2
            or {row["pair_leg"] for row in group} != {1, 2}
            or len({row["pair_id"] for row in group}) != 1
            or len({tuple(sorted((row["team_1_sha256"], row["team_2_sha256"]))) for row in group}) != 1
            or len({(row["agent_a_team_sha256"], row["agent_b_team_sha256"]) for row in group}) != 2
        ):
            raise RuntimeError(f"pair {index} mirror integrity invalid")
        ordered = sorted(group, key=lambda row: row["pair_leg"])
        pair_results.append({
            "pair_index": index,
            "pair_id": ordered[0]["pair_id"],
            "leg_1_winner": ordered[0]["winner"],
            "leg_2_winner": ordered[1]["winner"],
            "candidate_wins": sum(row["winner"] == "agent_a" for row in group),
        })
    wins = int(summary.get("agent_a_wins"))
    losses = int(summary.get("agent_a_losses"))
    if wins + losses != 20:
        raise RuntimeError("candidate win/loss accounting incomplete")
    low, high = wilson(wins, 20)
    return {
        "schema": "metagross-cycle41-blind-scorer-result/v1",
        "status": "pass" if wins >= 13 else "fail",
        "games": 20,
        "mirrored_pairs": 10,
        "candidate_wins": wins,
        "candidate_losses": losses,
        "candidate_win_rate": wins / 20,
        "wilson95": [low, high],
        "pair_results": pair_results,
        "candidate_public_roles": audit["candidate_public_roles"],
        "candidate_decisions": audit["candidate_decisions"],
        "candidate_overrides": audit["candidate_overrides"],
        "production_conversion_receipts": audit["production_conversion_receipts"],
        "candidate_conversion_receipts": audit["candidate_conversion_receipts"],
        "ability_receipts": audit["ability_receipts"],
        "semantic_operational_integrity_failures": 0,
        "result_sha256": sha(result_path),
        "postrun_manifest_integrity": "pass",
        "gate": "continue_only_if_at_least_13_wins",
        "strength_claim_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "score"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.phase == "preflight":
        result = preflight()
    else:
        if args.manifest is None:
            raise RuntimeError("score phase requires the frozen Cycle41 manifest")
        result = score(args.manifest.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
