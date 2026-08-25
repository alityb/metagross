#!/usr/bin/env python3
"""Select label-blind TRAIN-only natural Pressure PP roots for Cycle 39."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from experimental.src.scripts import audit_cycle13_train_rehydration as c13
from experimental.src.scripts import cycle12_replay_audit as v12


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "srcs/vendor/foul-play"))
from data import all_move_json  # noqa: E402

RANK_DOMAIN = "cycle39-natural-pressure-pp-v1"
CATEGORIES = ("self", "foe", "spread", "mustpressure")
PER_ROLE_CATEGORY = 2


def norm(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


def rank(*parts: object) -> str:
    return hashlib.sha256(
        "\0".join((RANK_DOMAIN, *(str(part) for part in parts))).encode("utf-8")
    ).hexdigest()


def pressure_events(log: str) -> list[dict[str, Any]]:
    pressure_active = {"p1": False, "p2": False}
    occurrences: dict[str, int] = {}
    result = []
    for index, line in enumerate(log.splitlines()):
        parts = line.split("|")
        tag = parts[1] if len(parts) > 1 else ""
        if tag in {"switch", "drag"} and len(parts) >= 4:
            pressure_active[parts[2][:2]] = False
        elif tag == "-ability" and len(parts) >= 4 and norm(parts[3]) == "pressure":
            pressure_active[parts[2][:2]] = True
        if tag != "move" or len(parts) < 5:
            continue
        actor_role = parts[2][:2]
        observer_role = "p2" if actor_role == "p1" else "p1"
        if not pressure_active[observer_role]:
            continue
        move_id = norm(parts[3])
        move = all_move_json.get(move_id, {})
        target = move.get("target", "")
        mustpressure = bool(move.get("flags", {}).get("mustpressure"))
        if mustpressure:
            category = "mustpressure"
        elif target in {"all", "allAdjacent", "allAdjacentFoes"}:
            category = "spread"
        elif target in {"normal", "any", "adjacentFoe", "randomNormal"} \
                and parts[4].startswith(observer_role):
            category = "foe"
        elif target in {"self", "allySide", "allyTeam", "adjacentAlly", "allies"}:
            category = "self"
        else:
            continue
        occurrences[line] = occurrences.get(line, 0) + 1
        result.append({
            "line": line,
            "line_occurrence": occurrences[line],
            "line_index": index,
            "observer_role": observer_role,
            "actor_role": actor_role,
            "move": move_id,
            "target_semantics": target,
            "mustpressure": mustpressure,
            "category": category,
        })
    return result


def prefix_contains(prefix: list[str], line: str, occurrence: int) -> bool:
    return sum(value == line for value in prefix) >= occurrence


def private_active_is_pressure(request: dict[str, Any]) -> bool:
    rows = [
        row for row in request.get("side", {}).get("pokemon", ())
        if row.get("active") is True
    ]
    return len(rows) == 1 and norm(
        rows[0].get("ability") or rows[0].get("baseAbility") or ""
    ) == "pressure"


def rematerialize(source: dict[str, Any], event: dict[str, Any], worktree: str):
    with tempfile.TemporaryDirectory(prefix="cycle39-select-") as temporary:
        output = Path(temporary)
        subprocess.run([
            "node", str(ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs"),
            "--showdown", worktree, "--input", source["raw_path"],
            "--out-dir", str(output),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        raw = json.loads(Path(source["raw_path"]).read_text())
        public = json.loads((output / "public.json").read_text())
        role = event["observer_role"]
        pov = json.loads((output / f"{role}.json").read_text())
        materialized = v12.materialize_role(
            battle_id=source["battle_id"], role=role,
            public_capture=public, pov_capture=pov, inputlog=raw["inputlog"],
            showdown_commit=source["showdown_commit"],
        )
    for state in materialized["states"]:
        if (
            state.get("actionable")
            and not state["pp_disable_sidecar"].get("revival_prompt")
            and private_active_is_pressure(state["private_request"])
            and prefix_contains(
                state["public_prefix"], event["line"], event["line_occurrence"]
            )
        ):
            compact = c13._compact_state(materialized, pov, state["request_index"])
            return state, compact
    raise RuntimeError("Pressure event lacks a following ordinary same-active request")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--worktrees", type=Path, required=True)
    parser.add_argument("--exclude-selection", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.master_index, "rt", encoding="utf-8") as handle:
        train = {
            row["battle_id"]: row
            for line in handle if line.strip()
            if (row := json.loads(line))["split"] == "train"
        }
    corpus = {
        row["battle_id"]: row
        for row in (json.loads(line) for line in args.corpus.read_text().splitlines() if line)
    }
    worktrees = json.loads(args.worktrees.read_text())
    excluded_clusters = set()
    for selection in args.exclude_selection:
        excluded_clusters.update(
            json.loads(line)["dependency_cluster_id"]
            for line in selection.read_text().splitlines() if line
        )
    inventory = []
    category_counts: dict[str, int] = {}
    for battle_id, battle in train.items():
        source = corpus.get(battle_id)
        if (
            source is None or not source.get("commit_present")
            or battle["dependency_cluster_id"] in excluded_clusters
        ):
            continue
        raw = json.loads(Path(source["raw_path"]).read_text())
        for event in pressure_events(raw["log"]):
            key = f"{event['observer_role']}:{event['category']}"
            category_counts[key] = category_counts.get(key, 0) + 1
            inventory.append({
                **event,
                "battle_id": battle_id,
                "dependency_cluster_id": battle["dependency_cluster_id"],
                "source": source,
                "rank_sha256": rank(
                    event["observer_role"], event["category"],
                    battle["dependency_cluster_id"], event["line_index"],
                ),
            })
    selected = []
    used = set()
    failures: dict[str, int] = {}
    for role in ("p1", "p2"):
        for category in CATEGORIES:
            admitted = 0
            candidates = sorted((
                row for row in inventory
                if row["observer_role"] == role and row["category"] == category
            ), key=lambda row: row["rank_sha256"])
            for candidate in candidates:
                if candidate["battle_id"] in used:
                    continue
                try:
                    state, compact = rematerialize(
                        candidate["source"], candidate,
                        worktrees[candidate["source"]["showdown_commit"]],
                    )
                except Exception as exc:
                    key = f"{type(exc).__name__}:{exc}"
                    failures[key] = failures.get(key, 0) + 1
                    continue
                candidate["state"], candidate["compact"] = state, compact
                selected.append(candidate)
                used.add(candidate["battle_id"])
                admitted += 1
                if admitted == PER_ROLE_CATEGORY:
                    break
            if admitted != PER_ROLE_CATEGORY:
                raise RuntimeError(f"cannot fill Pressure panel:{role}:{category}")
    rows = []
    for candidate in selected:
        source, state, compact = (
            candidate["source"], candidate["state"], candidate["compact"]
        )
        rows.append({
            "schema": "metagross-cycle39-pressure-panel/v1",
            "split": "train",
            "battle_id": candidate["battle_id"],
            "dependency_cluster_id": candidate["dependency_cluster_id"],
            "role": candidate["observer_role"],
            "actor_role": candidate["actor_role"],
            "category": candidate["category"],
            "move": candidate["move"],
            "target_semantics": candidate["target_semantics"],
            "mustpressure": candidate["mustpressure"],
            "pressure_event_line_index": candidate["line_index"],
            "rank_sha256": candidate["rank_sha256"],
            "raw_path": source["raw_path"], "raw_sha256": source["raw_sha256"],
            "showdown_commit": source["showdown_commit"],
            "request_index": state["request_index"],
            "command_input_index": compact["command_input_index"],
            "public_event_index": compact["public_event_index"],
            "model_information_fingerprint_sha256": compact[
                "model_information_fingerprint_sha256"
            ],
            "private_request_sha256": compact["private_request_sha256"],
            "causal_prefix_sha256": compact["causal_prefix_sha256"],
            "legal_action_contract_sha256": compact["legal_action_contract_sha256"],
            "typed_reveal_ledger_sha256": compact["typed_reveal_ledger_sha256"],
            "pp_disable_sidecar_sha256": compact["pp_disable_sidecar_sha256"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ))
    args.report.write_text(json.dumps({
        "schema": "metagross-cycle39-pressure-selection-report/v1",
        "inventory_events_by_role_category": category_counts,
        "selected_roots": len(rows),
        "selected_by_role_category": {
            f"{role}:{category}": sum(
                row["role"] == role and row["category"] == category for row in rows
            )
            for role in ("p1", "p2") for category in CATEGORIES
        },
        "unique_battles": len({row["battle_id"] for row in rows}),
        "unique_dependency_clusters": len({row["dependency_cluster_id"] for row in rows}),
        "excluded_cycle37_dependency_clusters": len(excluded_clusters),
        "selection_failures": failures,
        "teacher_values_opened": 0,
        "outcomes_used": False,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
