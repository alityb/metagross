#!/usr/bin/env python3
"""Audit local production-controller H2H execution artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from srcs.metagross import canary_audit, shadow_replay
from srcs.metagross.mcts_contract import validate_result_payload


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


def audit(
    results_path: Path,
    log_dir: Path,
    search_first_decisions_path: Path,
    certified_decisions_path: Path,
    expected_games: int = 4,
) -> dict:
    result = json.loads(results_path.read_text(encoding="utf-8"))
    summary = result["summary"]
    expected_tags = {game["battle_tag"] for game in result["games"]}
    prior_by_mode = {
        "search_first": {
            shadow_replay._decision_key(row): row
            for row in _read_jsonl(search_first_decisions_path)
            if row.get("tag") in expected_tags
        },
        "certified": {
            shadow_replay._decision_key(row): row
            for row in _read_jsonl(certified_decisions_path)
            if row.get("tag") in expected_tags
        },
    }
    failures = []
    process_rows = []
    all_search_keys = {"search_first": set(), "certified": set()}

    for search_path in sorted(log_dir.glob("*.search.jsonl")):
        search_rows = _read_jsonl(search_path)
        if not search_rows:
            continue
        search_by_key = {
            shadow_replay._search_key(row): row
            for row in search_rows
            if (row.get("context") or {}).get("tag") in expected_tags
        }
        if not search_by_key:
            continue
        username = search_path.name.removesuffix(".search.jsonl")
        protocol_path = log_dir / f"{username}.protocol.jsonl"
        if not protocol_path.is_file():
            failures.append(f"{username}: missing protocol dump")
            continue
        protocol = _read_jsonl(protocol_path)
        modes = {
            (row.get("choice_override") or {}).get("controller_mode")
            for row in search_by_key.values()
        }
        if len(modes) != 1 or next(iter(modes)) not in prior_by_mode:
            failures.append(f"{username}: invalid or mixed controller modes")
            continue
        mode = next(iter(modes))
        all_search_keys[mode].update(search_by_key)
        malformed_results = 0
        empty_priors = 0
        for row in search_by_key.values():
            if not row.get("player_priors"):
                empty_priors += 1
            for sample in row.get("samples") or ():
                try:
                    validate_result_payload(sample.get("result"))
                except ValueError:
                    malformed_results += 1
        if empty_priors:
            failures.append(f"{username}: {empty_priors} decisions lack player priors")
        if malformed_results:
            failures.append(f"{username}: {malformed_results} malformed search results")

        reconstructed = shadow_replay.reconstruct_battles(
            protocol, search_by_key, username
        )
        commands = canary_audit._sent_commands(protocol)
        outbound_matches = 0
        for tag in {key[0] for key in search_by_key}:
            keys = sorted(key for key in search_by_key if key[0] == tag)
            sent = commands.get(tag, [])
            if len(keys) != len(sent):
                failures.append(
                    f"{username}/{tag}: {len(sent)} commands for {len(keys)} decisions"
                )
                continue
            for key, command in zip(keys, sent, strict=True):
                action = canary_audit._command_action(command, reconstructed[key])
                if action != search_by_key[key].get("choice"):
                    failures.append(
                        f"{username}/{tag}/{key[1]}: selected-command mismatch"
                    )
                else:
                    outbound_matches += 1
        process_rows.append(
            {
                "username": username,
                "mode": mode,
                "battle_tags": sorted({key[0] for key in search_by_key}),
                "decisions": len(search_by_key),
                "outbound_matches": outbound_matches,
                "search_path": str(search_path),
                "search_sha256": _sha256(search_path),
                "protocol_path": str(protocol_path),
                "protocol_sha256": _sha256(protocol_path),
            }
        )

    for mode, keys in all_search_keys.items():
        prior_keys = set(prior_by_mode[mode])
        if keys != prior_keys:
            failures.append(
                f"{mode}: prior/search key mismatch search={len(keys)} prior={len(prior_keys)}"
            )
    tag_modes = {
        tag: {
            row["mode"]
            for row in process_rows
            if tag in row["battle_tags"]
        }
        for tag in expected_tags
    }
    if any(modes != {"search_first", "certified"} for modes in tag_modes.values()):
        failures.append("at least one battle lacks exactly one process per controller mode")

    counts = {
        "games": len(result["games"]),
        "completed_games": summary["completed_games"],
        "completed_pairs": summary["completed_pairs"],
        "void_games": summary["void_games"],
        "void_pairs": summary["void_pairs"],
        "processes": len(process_rows),
        "search_first_decisions": len(all_search_keys["search_first"]),
        "certified_decisions": len(all_search_keys["certified"]),
        "search_decisions": sum(len(keys) for keys in all_search_keys.values()),
        "prior_decisions": sum(len(rows) for rows in prior_by_mode.values()),
        "outbound_matches": sum(row["outbound_matches"] for row in process_rows),
        "failures": len(failures),
    }
    gate = {
        "all_requested_games_complete": counts["games"]
        == counts["completed_games"]
        == expected_games,
        "all_requested_mirrored_pairs_complete": counts["completed_pairs"]
        == expected_games // 2,
        "zero_voids": counts["void_games"] == counts["void_pairs"] == 0,
        "both_controllers_active_each_game": len(process_rows) == 2 * expected_games
        and all(modes == {"search_first", "certified"} for modes in tag_modes.values()),
        "priors_active_and_exactly_joined": counts["search_decisions"]
        == counts["prior_decisions"],
        "search_active": counts["search_decisions"] > 0,
        "outbound_commands_exactly_joined": counts["outbound_matches"]
        == counts["search_decisions"],
        "zero_failures": not failures,
    }
    gate["passed"] = all(gate.values())
    return {
        "schema_version": 1,
        "mode": "production_controller_local_h2h_audit",
        "inputs": {
            "results": {"path": str(results_path), "sha256": _sha256(results_path)},
            "search_first_prior_decisions": {
                "path": str(search_first_decisions_path),
                "sha256": _sha256(search_first_decisions_path),
            },
            "certified_prior_decisions": {
                "path": str(certified_decisions_path),
                "sha256": _sha256(certified_decisions_path),
            },
        },
        "counts": counts,
        "processes": process_rows,
        "failures": failures,
        "gate": gate,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--search-first-decisions", type=Path, required=True)
    parser.add_argument("--certified-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = audit(
        args.results.resolve(),
        args.log_dir.resolve(),
        args.search_first_decisions.resolve(),
        args.certified_decisions.resolve(),
        args.expected_games,
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["counts"], sort_keys=True))
    print(output)
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
