#!/usr/bin/env python3
"""Select TRAIN-only causal-Disable transitions from the frozen Cycle 12 corpus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from experimental.src.scripts import cycle12_replay_audit as v12
from experimental.src.scripts import audit_cycle13_train_rehydration as c13
from srcs.metagross.causal_reveal_ledger import freeze_ledger


ROOT = Path(__file__).resolve().parents[3]
RANK_DOMAIN = "cycle34-causal-disable-transition-v1"
CATEGORIES = ("force_faint_carry", "explicit_end", "own_switch")
PER_ROLE_CATEGORY = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank(*parts: object) -> str:
    return hashlib.sha256(
        "\0".join((RANK_DOMAIN, *(str(part) for part in parts))).encode("utf-8")
    ).hexdigest()


def public_events(log: str) -> list[dict[str, Any]]:
    lines = log.splitlines()
    events = []
    for index, line in enumerate(lines):
        if not (
            line.startswith("|-start|")
            and "|Disable|" in line
            and "[from] ability: Cursed Body" in line
        ):
            continue
        parts = line.split("|")
        disabled_role = parts[2][:2]
        observer_role = "p2" if disabled_role == "p1" else "p1"
        disabled_species = parts[2].split(":", 1)[1].strip()
        move = parts[4]
        clear_kind = "none"
        clear_index = None
        for later in range(index + 1, len(lines)):
            value = lines[later]
            if value.startswith(f"|-end|{disabled_role}a:") and "|Disable" in value:
                clear_kind, clear_index = "explicit_end", later
                break
            if value.startswith(f"|switch|{disabled_role}a:") or value.startswith(
                f"|drag|{disabled_role}a:"
            ):
                clear_kind, clear_index = "own_switch", later
                break
        near = lines[index : min(len(lines), index + 14)]
        force_faint = (
            any(value.startswith(f"|faint|{observer_role}a:") for value in near)
            and "|upkeep" in near
            and any(value.startswith(f"|switch|{observer_role}a:") for value in near)
        )
        categories = []
        if force_faint:
            categories.append("force_faint_carry")
        if clear_kind in {"explicit_end", "own_switch"}:
            categories.append(clear_kind)
        for category in categories:
            events.append({
                "category": category,
                "start_line_index": index,
                "clear_line_index": clear_index,
                "disabled_role": disabled_role,
                "observer_role": observer_role,
                "disabled_species": disabled_species,
                "move": move,
            })
    return events


def latest_disable(
    prefix: list[str], observer_role: str, species: str, move: str
) -> bool | None:
    ledger = freeze_ledger("selection-probe", observer_role, prefix)
    wanted_move = "".join(ch for ch in move.lower() if ch.isalnum())
    matches = [fact for fact in ledger.facts if any(
        event.move == wanted_move for event in fact.disable_history
    )]
    if len(matches) != 1:
        return None
    state = None
    for event in matches[0].disable_history:
        if event.move == wanted_move:
            state = event.disabled
    return state


def rematerialize(source: dict[str, Any], event: dict[str, Any], worktree: str) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="cycle34-select-") as temporary:
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
    states = materialized["states"]
    active = []
    cleared = []
    for state in states:
        try:
            value = latest_disable(
                state["public_prefix"], role,
                event["disabled_species"], event["move"]
            )
        except Exception:
            continue
        if value is True:
            active.append(state)
        elif value is False:
            cleared.append(state)
    if not active:
        raise RuntimeError("natural transition lacks an active causal-disable request")
    if event["category"] == "force_faint_carry":
        forced = [state for state in active if state["private_request"].get("forceSwitch")]
        if not forced:
            raise RuntimeError("force/faint candidate lacks a forceSwitch carry request")
        chosen = [forced[0]]
    else:
        actionable = [state for state in active if state.get("actionable")]
        cleared_actionable = [state for state in cleared if state.get("actionable")]
        if not actionable or not cleared_actionable:
            raise RuntimeError("clear transition lacks active/cleared request states")
        chosen = [actionable[-1], cleared_actionable[0]]
    for state in chosen:
        state["_compact"] = c13._compact_state(
            materialized, pov, state["request_index"]
        )
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-index", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--worktrees", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.master_index, "rt", encoding="utf-8") as handle:
        master = [json.loads(line) for line in handle if line.strip()]
    train = {row["battle_id"]: row for row in master if row["split"] == "train"}
    corpus = {
        row["battle_id"]: row
        for row in (json.loads(line) for line in args.corpus.read_text().splitlines() if line)
    }
    worktrees = json.loads(args.worktrees.read_text())
    inventory = []
    for battle_id, battle in train.items():
        source = corpus.get(battle_id)
        if source is None or not source.get("commit_present"):
            continue
        raw = json.loads(Path(source["raw_path"]).read_text())
        for event in public_events(raw["log"]):
            inventory.append({
                **event,
                "battle_id": battle_id,
                "dependency_cluster_id": battle["dependency_cluster_id"],
                "rank_sha256": rank(
                    event["category"], event["disabled_role"],
                    battle["dependency_cluster_id"], event["start_line_index"],
                ),
                "source": source,
                "battle_index": battle,
            })
    selected_events = []
    used_battles = set()
    selection_failures: dict[str, int] = {}
    for category in CATEGORIES:
        for disabled_role in ("p1", "p2"):
            candidates = sorted(
                (row for row in inventory if row["category"] == category and row["disabled_role"] == disabled_role),
                key=lambda row: row["rank_sha256"],
            )
            for candidate in candidates:
                if candidate["battle_id"] in used_battles:
                    continue
                source = candidate["source"]
                try:
                    candidate["_states"] = rematerialize(
                        source, candidate, worktrees[source["showdown_commit"]]
                    )
                except Exception as exc:
                    key = type(exc).__name__ + ":" + str(exc)
                    selection_failures[key] = selection_failures.get(key, 0) + 1
                    continue
                selected_events.append(candidate)
                used_battles.add(candidate["battle_id"])
                if sum(
                    row["category"] == category and row["disabled_role"] == disabled_role
                    for row in selected_events
                ) == PER_ROLE_CATEGORY:
                    break
    if len(selected_events) != len(CATEGORIES) * 2 * PER_ROLE_CATEGORY:
        raise RuntimeError("natural TRAIN-only targeted inventory is too small")
    rows = []
    for event in selected_events:
        source = event["source"]
        states = event["_states"]
        for transition_index, state in enumerate(states):
            compact = state["_compact"]
            rows.append({
                "schema": "metagross-cycle34-disable-panel/v1",
                "split": "train",
                "battle_id": event["battle_id"],
                "dependency_cluster_id": event["dependency_cluster_id"],
                "category": event["category"],
                "disabled_role": event["disabled_role"],
                "observer_role": event["observer_role"],
                "disabled_species": event["disabled_species"],
                "move": event["move"],
                "transition_state": (
                    "active" if event["category"] == "force_faint_carry" or transition_index == 0
                    else "cleared"
                ),
                "rank_sha256": event["rank_sha256"],
                "raw_path": source["raw_path"],
                "raw_sha256": source["raw_sha256"],
                "showdown_commit": source["showdown_commit"],
                "role": event["observer_role"],
                "request_index": state["request_index"],
                "command_input_index": compact["command_input_index"],
                "public_event_index": compact["public_event_index"],
                "model_information_fingerprint_sha256": compact["model_information_fingerprint_sha256"],
                "private_request_sha256": compact["private_request_sha256"],
                "causal_prefix_sha256": compact["causal_prefix_sha256"],
                "legal_action_contract_sha256": compact["legal_action_contract_sha256"],
                "typed_reveal_ledger_sha256": compact["typed_reveal_ledger_sha256"],
                "pp_disable_sidecar_sha256": compact["pp_disable_sidecar_sha256"],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report = {
        "schema": "metagross-cycle34-disable-selection-report/v1",
        "train_battles": len(train),
        "natural_events": len(inventory),
        "natural_battles": len({row["battle_id"] for row in inventory}),
        "selected_battles": len(selected_events),
        "selected_states": len(rows),
        "label_blind_selection_failures": selection_failures,
        "by_category_role": {
            f"{category}:{role}": sum(
                row["category"] == category and row["disabled_role"] == role
                for row in selected_events
            )
            for category in CATEGORIES for role in ("p1", "p2")
        },
        "validation_test_rows_opened": 0,
        "sealed93_rows_read": 0,
        "teacher_values_opened": 0,
        "master_index_sha256": sha256(args.master_index),
        "corpus_sha256": sha256(args.corpus),
        "selection_sha256": sha256(args.output),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
