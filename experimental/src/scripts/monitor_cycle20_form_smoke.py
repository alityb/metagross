#!/usr/bin/env python3
"""Cycle20 live smoke: Cycle19 operational checks plus causal form lineage."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experimental.src.scripts.monitor_cycle19_operational_smoke import (
    rows,
    to_id,
    try_validate as try_validate_cycle19,
)
from srcs.metagross.causal_reveal_ledger import freeze_ledger


def observer_role(protocol: list[dict], username: str) -> str:
    for row in protocol:
        if row.get("direction") not in {"received", "reconnect_received"}:
            continue
        for line in str(row.get("message", "")).splitlines():
            parts = line.split("|")
            if len(parts) >= 4 and parts[1] == "player" and to_id(parts[3]) == to_id(username):
                if parts[2] in {"p1", "p2"}:
                    return parts[2]
    raise RuntimeError("candidate role is absent from public protocol")


def public_prefix(protocol: list[dict], before_ns: int) -> list[str]:
    lines = []
    for row in protocol:
        if row.get("direction") not in {"received", "reconnect_received"}:
            continue
        if int(row.get("time_ns", 0)) > before_ns:
            continue
        lines.extend(
            line for line in str(row.get("message", "")).splitlines()
            if line.startswith("|") and not line.startswith("|request|")
        )
    return lines


def validate_form_transition(protocol: list[dict], username: str, before_ns: int) -> dict:
    prefix = public_prefix(protocol, before_ns)
    role = observer_role(protocol, username)
    opponent = "p2" if role == "p1" else "p1"
    shift = [
        index for index, line in enumerate(prefix)
        if line.startswith(f"|-activate|{opponent}a:") and "|ability: Tera Shift" in line
    ]
    form = [
        index for index, line in enumerate(prefix)
        if line.startswith(f"|detailschange|{opponent}a:")
        and "|Terapagos-Terastal," in line
    ]
    if not shift or not form or min(form) <= min(shift):
        raise RuntimeError("candidate did not observe ordered Terapagos form transition")
    ledger = freeze_ledger("battle-cycle20-live-smoke", role, prefix)
    fact = next((row for row in ledger.facts if row.species == "terapagos"), None)
    if fact is None:
        raise RuntimeError("live v2 ledger omitted Terapagos")
    history = [(event.ability, event.authority) for event in fact.ability_history]
    expected = [
        ("terashift", "explicit_public_event"),
        ("terashell", "rule_implied_form_transition"),
    ]
    if (
        fact.exact_public_species != "terapagosterastal"
        or fact.current_ability != "terashell"
        or history[-2:] != expected
    ):
        raise RuntimeError("live v2 ledger reconstructed the wrong ability lineage")
    return {
        "observer_role": role,
        "opponent_role": opponent,
        "exact_public_species": fact.exact_public_species,
        "current_ability": fact.current_ability,
        "ability_history_tail": history[-2:],
        "protocol_sha256": ledger.protocol_sha256,
        "shift_event_index": min(shift),
        "detailschange_event_index": min(form),
    }


def try_validate(run: Path, expected_engine_sha: str) -> dict | None:
    base = try_validate_cycle19(run, expected_engine_sha)
    if base is None:
        return None
    search_path = Path(base["candidate_search_path"])
    search_rows = rows(search_path)
    candidate = next(
        row for row in search_rows
        if ((row.get("choice_override") or {}).get("terminal_mcts_teacher") or {}).get(
            "controller_schema"
        ) == "metagross-cycle19-equal8192-production-selector/v1"
    )
    protocol_path = Path(base["candidate_protocol_path"])
    protocol = rows(protocol_path)
    lineage = validate_form_transition(
        protocol, search_path.name.removesuffix(".search.jsonl"),
        int(candidate.get("time_ns", 0)),
    )
    return {
        **base,
        "schema": "metagross-cycle20-form-transition-smoke/v1",
        "causal_ability_lineage": lineage,
        "cycle19_semantics_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-engine-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        result = try_validate(args.run.resolve(), args.expected_engine_sha256.lower())
        if result is not None:
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps(result, sort_keys=True))
            return
        time.sleep(0.25)
    raise TimeoutError("Cycle20 form-transition operational smoke did not pass in time")


if __name__ == "__main__":
    main()
