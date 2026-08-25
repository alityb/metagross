#!/usr/bin/env python3
"""Audit shared-root RM+ versus independent-MCTS local H2H artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from srcs.metagross import canary_audit, shadow_replay
from srcs.metagross.mcts_contract import (
    validate_result_payload,
    validate_shared_root_result_payload,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def _pair_failures(games: list[dict]) -> list[str]:
    failures = []
    grouped: dict[str, list[dict]] = {}
    for game in games:
        pair_id = game.get("pair_id")
        if not pair_id:
            failures.append(f"game {game.get('game_index')}: missing pair_id")
            continue
        grouped.setdefault(pair_id, []).append(game)
    for pair_id, pair in grouped.items():
        if len(pair) != 2 or {game.get("pair_leg") for game in pair} != {1, 2}:
            failures.append(f"pair {pair_id}: does not contain exactly legs 1 and 2")
            continue
        first, second = sorted(pair, key=lambda game: game["pair_leg"])
        invariant_fields = ("battle_seed", "team_1_sha256", "team_2_sha256")
        if any(first.get(field) != second.get(field) for field in invariant_fields):
            failures.append(f"pair {pair_id}: seed or team identity changed between legs")
        if (
            first.get("agent_a_team_sha256") != second.get("agent_b_team_sha256")
            or first.get("agent_b_team_sha256") != second.get("agent_a_team_sha256")
        ):
            failures.append(f"pair {pair_id}: team assignments were not swapped")
    return failures


def _provenance(log_path: Path) -> dict | None:
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("POKE_ENGINE_PROVENANCE "):
                return json.loads(line.removeprefix("POKE_ENGINE_PROVENANCE "))
    return None


def audit(
    results_path: Path,
    log_dir: Path,
    candidate_prior_path: Path,
    comparator_prior_path: Path,
    expected_games: int = 4,
) -> dict:
    result = json.loads(results_path.read_text(encoding="utf-8"))
    summary = result["summary"]
    games = result["games"]
    expected_tags = {game["battle_tag"] for game in games}
    prior_by_mode = {
        "shared_rm_plus": {
            shadow_replay._decision_key(row): row
            for row in _read_jsonl(candidate_prior_path)
            if row.get("tag") in expected_tags
        },
        "independent_mcts": {
            shadow_replay._decision_key(row): row
            for row in _read_jsonl(comparator_prior_path)
            if row.get("tag") in expected_tags
        },
    }
    failures = _pair_failures(games)
    process_rows = []
    search_keys = {mode: set() for mode in prior_by_mode}
    candidate_latencies = []
    exploitabilities = []
    nash_convs = []
    payoff_cells = []
    provenance_identities = []
    selection_counts = {
        "shared_rm_plus": {"search_selection": 0, "deterministic_correction": 0},
        "independent_mcts": {"search_selection": 0, "deterministic_correction": 0},
    }
    advisory_counts = {
        "shared_rm_plus": {"shadow_risks": 0, "missing_request_actions": 0},
        "independent_mcts": {"shadow_risks": 0, "missing_request_actions": 0},
    }

    for search_path in sorted(log_dir.glob("*.search.jsonl")):
        rows = [
            row
            for row in _read_jsonl(search_path)
            if (row.get("context") or {}).get("tag") in expected_tags
        ]
        if not rows:
            continue
        username = search_path.name.removesuffix(".search.jsonl")
        protocol_path = log_dir / f"{username}.protocol.jsonl"
        log_path = log_dir / f"{username}.log"
        if not protocol_path.is_file() or not log_path.is_file():
            failures.append(f"{username}: missing protocol or process log")
            continue
        candidate_shape = [isinstance(row.get("shared_root"), dict) for row in rows]
        comparator_shape = [isinstance(row.get("samples"), list) for row in rows]
        if all(candidate_shape) and not any(comparator_shape):
            mode = "shared_rm_plus"
        elif all(comparator_shape) and not any(candidate_shape):
            mode = "independent_mcts"
        else:
            failures.append(f"{username}: mixed or unrecognized root-search telemetry")
            continue

        by_key = {shadow_replay._search_key(row): row for row in rows}
        if len(by_key) != len(rows):
            failures.append(f"{username}: duplicate search decision key")
        search_keys[mode].update(by_key)
        malformed = 0
        policy_failures = 0
        for row in rows:
            override = row.get("choice_override") or {}
            if not row.get("player_priors"):
                policy_failures += 1
            if override.get("blocked_safeguard") is not None:
                policy_failures += 1
            advisory_counts[mode]["shadow_risks"] += len(override.get("shadow_risks") or ())
            advisory_counts[mode]["missing_request_actions"] += len(
                override.get("missing_request_actions") or ()
            )
            selection_class = override.get("selection_class")
            if selection_class not in {"search_selection", "deterministic_correction"}:
                policy_failures += 1
            else:
                selection_counts[mode][selection_class] += 1
            try:
                if mode == "shared_rm_plus":
                    validate_shared_root_result_payload(row["shared_root"])
                    diagnostics = override["solver_diagnostics"]
                    if (
                        override.get("root_search_mode") != "shared_rm_plus"
                        or override.get("search_mass_kind") != "shared_policy_probability"
                        or diagnostics.get("solver_contract") != "weighted-shared-rm-plus-v1"
                        or diagnostics.get("iterations") != 10_000
                        or diagnostics.get("continuation_iterations") != 8
                        or diagnostics.get("prior_strength") != 1.0
                    ):
                        policy_failures += 1
                    remote = row.get("remote_search") or {}
                    if remote.get("operation") != "shared_root" or remote.get("transport") != "local":
                        policy_failures += 1
                    candidate_latencies.append(float(remote["rpc_ms"]))
                    exploitabilities.append(float(diagnostics["exploitability"]))
                    nash_convs.append(float(diagnostics["nash_conv"]))
                    payoff_cells.append(int(diagnostics["payoff_cells"]))
                else:
                    if override.get("search_mass_kind") != "weighted_visits":
                        policy_failures += 1
                    for sample in row["samples"]:
                        validate_result_payload(sample.get("result"))
            except (KeyError, TypeError, ValueError):
                malformed += 1
        if malformed:
            failures.append(f"{username}: {malformed} malformed search results")
        if policy_failures:
            failures.append(f"{username}: {policy_failures} policy-integrity failures")

        protocol = _read_jsonl(protocol_path)
        reconstructed = shadow_replay.reconstruct_battles(protocol, by_key, username)
        commands = canary_audit._sent_commands(protocol)
        outbound_matches = 0
        for tag in {key[0] for key in by_key}:
            keys = sorted(key for key in by_key if key[0] == tag)
            sent = commands.get(tag, [])
            if len(keys) != len(sent):
                failures.append(f"{username}/{tag}: {len(sent)} commands for {len(keys)} decisions")
                continue
            for key, command in zip(keys, sent, strict=True):
                if canary_audit._command_action(command, reconstructed[key]) == by_key[key].get("choice"):
                    outbound_matches += 1
                else:
                    failures.append(f"{username}/{tag}/{key[1]}: selected-command mismatch")

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "Traceback (most recent call last)" in log_text or "ERROR " in log_text:
            failures.append(f"{username}: process log contains an error")
        identity = _provenance(log_path)
        if identity is None:
            failures.append(f"{username}: missing poke-engine provenance")
        else:
            provenance_identities.append(identity)
        process_rows.append(
            {
                "username": username,
                "mode": mode,
                "battle_tags": sorted({key[0] for key in by_key}),
                "decisions": len(rows),
                "outbound_matches": outbound_matches,
                "search_path": str(search_path),
                "search_sha256": _sha256(search_path),
                "protocol_path": str(protocol_path),
                "protocol_sha256": _sha256(protocol_path),
            }
        )

    for mode, keys in search_keys.items():
        priors = prior_by_mode[mode]
        if keys != set(priors):
            failures.append(f"{mode}: prior/search key mismatch search={len(keys)} prior={len(priors)}")
        fallback_count = sum(bool(row.get("mask_fallback")) for row in priors.values())
        if fallback_count:
            failures.append(f"{mode}: {fallback_count} prior legality-mask fallbacks")

    tag_modes = {
        tag: {row["mode"] for row in process_rows if tag in row["battle_tags"]}
        for tag in expected_tags
    }
    if any(modes != {"shared_rm_plus", "independent_mcts"} for modes in tag_modes.values()):
        failures.append("at least one battle lacks exactly one process per root-search mode")
    native_hashes = {identity.get("native_sha256") for identity in provenance_identities}
    versions = {identity.get("distribution_version") for identity in provenance_identities}
    if len(provenance_identities) != 2 * expected_games or len(native_hashes) != 1 or versions != {"0.0.47"}:
        failures.append("poke-engine provenance differs across controller processes")

    counts = {
        "games": len(games),
        "completed_games": summary["completed_games"],
        "completed_pairs": summary["completed_pairs"],
        "void_games": summary["void_games"],
        "void_pairs": summary["void_pairs"],
        "processes": len(process_rows),
        "candidate_decisions": len(search_keys["shared_rm_plus"]),
        "comparator_decisions": len(search_keys["independent_mcts"]),
        "prior_decisions": sum(len(rows) for rows in prior_by_mode.values()),
        "outbound_matches": sum(row["outbound_matches"] for row in process_rows),
        "failures": len(failures),
    }
    total_decisions = counts["candidate_decisions"] + counts["comparator_decisions"]
    gate = {
        "all_requested_games_complete": counts["games"] == counts["completed_games"] == expected_games,
        "all_requested_mirrored_pairs_complete": counts["completed_pairs"] == expected_games // 2,
        "zero_voids": counts["void_games"] == counts["void_pairs"] == 0,
        "both_root_solvers_active_each_game": len(process_rows) == 2 * expected_games
        and all(modes == {"shared_rm_plus", "independent_mcts"} for modes in tag_modes.values()),
        "priors_active_and_exactly_joined": total_decisions == counts["prior_decisions"],
        "outbound_commands_exactly_joined": counts["outbound_matches"] == total_decisions,
        "zero_failures": not failures,
    }
    gate["passed"] = all(gate.values())
    return {
        "schema_version": 1,
        "mode": "shared_root_rm_plus_local_h2h_audit",
        "inputs": {
            "results": {"path": str(results_path), "sha256": _sha256(results_path)},
            "candidate_prior_decisions": {"path": str(candidate_prior_path), "sha256": _sha256(candidate_prior_path)},
            "comparator_prior_decisions": {"path": str(comparator_prior_path), "sha256": _sha256(comparator_prior_path)},
        },
        "counts": counts,
        "outcome": {
            "candidate_wins": summary["agent_a_wins"],
            "candidate_losses": summary["agent_a_losses"],
            "pair_score_mean": summary["pair_score_mean"],
            "claim_eligible": False,
        },
        "candidate_diagnostics": {
            "selection_counts": selection_counts["shared_rm_plus"],
            "advisory_counts": advisory_counts["shared_rm_plus"],
            "latency_ms": {
                "p50": _percentile(candidate_latencies, 0.50),
                "p95": _percentile(candidate_latencies, 0.95),
                "max": max(candidate_latencies, default=None),
            },
            "max_exploitability": max(exploitabilities, default=None),
            "max_nash_conv": max(nash_convs, default=None),
            "max_payoff_cells": max(payoff_cells, default=None),
        },
        "comparator_diagnostics": {
            "selection_counts": selection_counts["independent_mcts"],
            "advisory_counts": advisory_counts["independent_mcts"],
        },
        "engine_provenance": {
            "distribution_versions": sorted(versions),
            "native_sha256": sorted(native_hashes),
        },
        "processes": process_rows,
        "failures": failures,
        "gate": gate,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--candidate-prior-decisions", type=Path, required=True)
    parser.add_argument("--comparator-prior-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = audit(
        args.results.resolve(),
        args.log_dir.resolve(),
        args.candidate_prior_decisions.resolve(),
        args.comparator_prior_decisions.resolve(),
        args.expected_games,
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], sort_keys=True))
    print(output)
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
