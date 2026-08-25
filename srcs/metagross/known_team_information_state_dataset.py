#!/usr/bin/env python3
"""Export leakage-safe information-state action-value targets.

Inputs contain only the observer-visible protocol prefix and the observer's
request JSON.  Targets are belief-averaged root Q advantages from the immutable
Phase-2 world bank.  Serialized sampled worlds, candidate identities, and truth
teams are never copied into output records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from srcs.metagross.known_team_belief_eval import _pristine_candidates, _reconstruct_view
from srcs.metagross.known_team_decision_v2 import (
    canonical_json,
    root_beliefs,
    sha256_bytes,
    sha256_path,
)
from srcs.metagross.known_team_decision_v2_phase2 import (
    WORLD_BANK_SCHEMA,
    aggregate_advantages,
    aggregate_visit_policy,
    normalized_weights,
)


SCHEMA = "metagross-information-state-q-dataset/v1"
RECORD_SCHEMA = "metagross-information-state-q-record/v1"
FORBIDDEN_OUTPUT_KEYS = {
    "state",
    "sampled_state",
    "selected_candidates",
    "truth",
    "truth_team",
    "teams",
}


def _sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("ascii"))


def _request_for_decision(view: Mapping[str, Any], decision_idx: int) -> dict[str, Any]:
    decision = view["decisions"][decision_idx]
    count = int(decision["chunk_count"])
    prefix = list(view["chunks"][:count])
    for chunk in reversed(prefix):
        for line in reversed(chunk.splitlines()):
            if line.startswith("|request|"):
                payload = json.loads(line[len("|request|") :])
                if not isinstance(payload, dict):
                    raise ValueError("request payload is not an object")
                return payload
    raise ValueError("decision prefix contains no request")


def _public_protocol_prefix(view: Mapping[str, Any], decision_idx: int) -> list[str]:
    count = int(view["decisions"][decision_idx]["chunk_count"])
    result: list[str] = []
    for chunk in view["chunks"][:count]:
        public_lines = [line for line in chunk.splitlines() if not line.startswith("|request|")]
        result.append("\n".join(public_lines))
    return result


def _draw_searches(root: Mapping[str, Any], repeat_key: str) -> list[Mapping[str, Any]]:
    searches = root[repeat_key]
    return [searches[draw["state_sha256"]]["actions"] for draw in root["draws"]]


def _average_mapping(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    if set(left) != set(right):
        raise ValueError("independent repeat action supports differ")
    return {action: (float(left[action]) + float(right[action])) / 2 for action in sorted(left)}


def _record(
    root: Mapping[str, Any], battle: Mapping[str, Any], pristine: Mapping[str, tuple[Any, ...]]
) -> dict[str, Any]:
    observer = str(root["observer"])
    decision_idx = int(root["decision_idx"])
    view = battle["views"][observer]
    request = _request_for_decision(view, decision_idx)
    protocol = _public_protocol_prefix(view, decision_idx)
    reconstructed = _reconstruct_view(battle, observer)[decision_idx]
    current_belief, _history, _tv, _affected = root_beliefs(reconstructed, pristine)
    weights = normalized_weights(root["draws"], 0.0)
    searches0 = _draw_searches(root, "searches")
    searches1 = _draw_searches(root, "repeat_searches")
    policy0 = aggregate_visit_policy(searches0, weights)
    policy1 = aggregate_visit_policy(searches1, weights)
    baseline = sorted(policy0, key=lambda action: (-policy0[action], action))[0]
    advantages0 = aggregate_advantages(searches0, weights, baseline)
    advantages1 = aggregate_advantages(searches1, weights, baseline)
    record: dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "identity": {
            "battle_id": root["battle_id"],
            "corpus_uid": root["corpus_uid"],
            "observer": observer,
            "decision_idx": decision_idx,
        },
        "split_group": str(root["battle_id"]),
        "panel": root["panel"],
        "turn": int(root["turn"]),
        "information_state": {
            "public_protocol_chunks": protocol,
            "player_request": request,
            "current_opponent_set_posterior": current_belief,
        },
        "legal_actions": list(root["legal_actions"]),
        "target": {
            "belief": "current_strict_foul_play_alpha_0",
            "world_draw_count": len(root["draws"]),
            "world_iterations_per_repeat": 20_000,
            "independent_repeats": 2,
            "baseline_action": baseline,
            "visit_policy_repeat_0": policy0,
            "visit_policy_repeat_1": policy1,
            "visit_policy_mean": _average_mapping(policy0, policy1),
            "q_advantage_repeat_0": advantages0,
            "q_advantage_repeat_1": advantages1,
            "q_advantage_mean": _average_mapping(advantages0, advantages1),
        },
        "provenance": {
            "oracle": False,
            "known_team_teacher_used": False,
            "sampled_hidden_world_in_input": False,
            "sampled_hidden_world_serialized_in_record": False,
        },
    }
    record["record_sha256"] = _sha256_json(record)
    validate_record(record)
    return record


def validate_record(record: Mapping[str, Any]) -> None:
    if record.get("schema") != RECORD_SCHEMA:
        raise ValueError("invalid information-state record schema")
    unhashed = dict(record)
    claimed = unhashed.pop("record_sha256", None)
    if claimed != _sha256_json(unhashed):
        raise ValueError("information-state record hash does not match")
    serialized = canonical_json(record)
    parsed = json.loads(serialized)
    stack: list[Any] = [parsed]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if FORBIDDEN_OUTPUT_KEYS & set(value):
                raise ValueError("information-state record contains a forbidden key")
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    target = record.get("target") or {}
    legal = set(record.get("legal_actions") or [])
    if not legal or set(target.get("q_advantage_mean") or {}) != legal:
        raise ValueError("target support does not match legal actions")
    provenance = record.get("provenance") or {}
    if provenance != {
        "oracle": False,
        "known_team_teacher_used": False,
        "sampled_hidden_world_in_input": False,
        "sampled_hidden_world_serialized_in_record": False,
    }:
        raise ValueError("information-state provenance is unsafe")


def export_dataset(corpus_path: Path, bank_path: Path, output_path: Path) -> dict[str, Any]:
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("schema") != WORLD_BANK_SCHEMA or not bank.get("complete"):
        raise ValueError("world bank is invalid or incomplete")
    battles: dict[str, dict[str, Any]] = {}
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            battle = json.loads(line)
            battles[battle["battle_id"]] = battle
    records = []
    pristine = _pristine_candidates()
    for root in bank["roots"]:
        if root["battle_id"] not in battles:
            raise ValueError("world-bank battle is absent from corpus")
        records.append(_record(root, battles[root["battle_id"]], pristine))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
    return {
        "schema": SCHEMA,
        "status": "development_only_untrained_targets",
        "records": len(records),
        "battle_groups": len({record["split_group"] for record in records}),
        "panels": {
            panel: sum(record["panel"] == panel for record in records)
            for panel in sorted({record["panel"] for record in records})
        },
        "input": {
            "corpus_sha256": sha256_path(corpus_path),
            "world_bank_sha256": sha256_path(bank_path),
        },
        "output": {"path": str(output_path), "sha256": sha256_path(output_path)},
        "constraints": {
            "battle_group_split_required": True,
            "sampled_hidden_states_exported": False,
            "known_team_teacher_targets_exported": False,
            "public_ladder_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--world-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_dataset(args.corpus, args.world_bank, args.output)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
