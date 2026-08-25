#!/usr/bin/env python3
"""Select label-blind TRAIN form-change/switch-reactivation roots for Cycle 36."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from experimental.src.scripts import audit_cycle13_train_rehydration as c13
from experimental.src.scripts import cycle12_replay_audit as v12
from srcs.metagross.causal_reveal_ledger import canonical_species, freeze_ledger, norm


ROOT = Path(__file__).resolve().parents[3]
RANK_DOMAIN = "cycle36-form-switch-reactivation-v1"
FAMILY_TARGETS = {"morpeko": 3, "minior": 2, "terapagos": 3, "ogerpon": 2}


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


def family(species: str) -> str | None:
    if species.startswith("ogerpon"):
        return "ogerpon"
    return species if species in {"morpeko", "minior", "terapagos"} else None


def public_events(log: str) -> list[dict[str, Any]]:
    active: dict[str, str] = {"p1": "", "p2": ""}
    changed: dict[str, set[str]] = {"p1": set(), "p2": set()}
    away: dict[str, set[str]] = {"p1": set(), "p2": set()}
    occurrences: dict[str, int] = {}
    result = []
    for index, line in enumerate(log.splitlines()):
        parts = line.split("|")
        tag = parts[1] if len(parts) > 1 else ""
        if tag in {"switch", "drag"} and len(parts) >= 4:
            side = parts[2][:2]
            exact = norm(parts[3].split(",", 1)[0])
            species = canonical_species(exact)
            old = active.get(side, "")
            if old and old != species and old in changed[side]:
                away[side].add(old)
            occurrences[line] = occurrences.get(line, 0) + 1
            category = family(species)
            if category and species in changed[side] and species in away[side]:
                result.append({
                    "family": category,
                    "changed_species": species,
                    "exact_return_species": exact,
                    "changed_role": side,
                    "observer_role": "p2" if side == "p1" else "p1",
                    "reactivation_tag": tag,
                    "reactivation_line": line,
                    "reactivation_line_occurrence": occurrences[line],
                    "reactivation_line_index": index,
                })
            active[side] = species
        elif tag in {"-formechange", "detailschange"} and len(parts) >= 4:
            side = parts[2][:2]
            exact = norm(parts[3].split(",", 1)[0])
            species = canonical_species(exact)
            if active.get(side) == species:
                changed[side].add(species)
    return result


def rematerialize(
    source: dict[str, Any], event: dict[str, Any], worktree: str
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cycle36-select-") as temporary:
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
        if not state.get("actionable"):
            continue
        prefix = state["public_prefix"]
        ledger = freeze_ledger("cycle36-selection", role, prefix)
        matches = [
            fact for fact in ledger.facts
            if fact.species == event["changed_species"]
            and fact.exact_public_species == event["exact_return_species"]
            and fact.current_ability_authority == "rule_implied_switch_reactivation"
            and fact.ability_history
            and fact.ability_history[-1].protocol_tag == event["reactivation_tag"]
        ]
        if len(matches) != 1:
            continue
        compact = c13._compact_state(materialized, pov, state["request_index"])
        return {"state": state, "compact": compact, "ability": matches[0].current_ability}
    raise RuntimeError("no actionable state after exact causal reactivation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-index", type=Path, required=True)
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
                "source": source,
                "rank_sha256": rank(
                    event["family"], event["changed_role"],
                    event["reactivation_tag"], battle["dependency_cluster_id"],
                    event["reactivation_line_index"],
                ),
            })
    selected = []
    used = set()
    failures: dict[str, int] = {}

    def admit(candidates: list[dict[str, Any]], wanted: int, label: str) -> None:
        before = len(selected)
        for candidate in sorted(candidates, key=lambda row: row["rank_sha256"]):
            if candidate["battle_id"] in used:
                continue
            source = candidate["source"]
            try:
                materialized = rematerialize(
                    source, candidate, worktrees[source["showdown_commit"]]
                )
            except Exception as exc:
                key = f"{type(exc).__name__}:{exc}"
                failures[key] = failures.get(key, 0) + 1
                continue
            candidate["_materialized"] = materialized
            selected.append(candidate)
            used.add(candidate["battle_id"])
            if len(selected) - before == wanted:
                return
        raise RuntimeError(
            f"natural TRAIN inventory cannot fill target:{label}:{wanted}:"
            f"candidates={len(candidates)}:admitted={len(selected) - before}:"
            f"failures={failures}"
        )

    for category, wanted in FAMILY_TARGETS.items():
        for changed_role in ("p1", "p2"):
            role_wanted = 0 if category == "ogerpon" else wanted
            if role_wanted == 0:
                continue
            admit([
                row for row in inventory
                if row["family"] == category
                and row["changed_role"] == changed_role
                and row["reactivation_tag"] == "switch"
            ], role_wanted, f"{category}:{changed_role}:switch")
    for changed_role in ("p1", "p2"):
        admit([
            row for row in inventory
            if row["changed_role"] == changed_role and row["reactivation_tag"] == "drag"
        ], 1, f"any:{changed_role}:drag")

    rows = []
    for candidate in selected:
        source = candidate["source"]
        state = candidate["_materialized"]["state"]
        compact = candidate["_materialized"]["compact"]
        rows.append({
            "schema": "metagross-cycle36-reactivation-panel/v1",
            "split": "train",
            "battle_id": candidate["battle_id"],
            "dependency_cluster_id": candidate["dependency_cluster_id"],
            "family": candidate["family"],
            "changed_role": candidate["changed_role"],
            "observer_role": candidate["observer_role"],
            "reactivation_tag": candidate["reactivation_tag"],
            "changed_species": candidate["changed_species"],
            "exact_return_species": candidate["exact_return_species"],
            "certified_reactivation_ability": candidate["_materialized"]["ability"],
            "rank_sha256": candidate["rank_sha256"],
            "raw_path": source["raw_path"],
            "raw_sha256": source["raw_sha256"],
            "showdown_commit": source["showdown_commit"],
            "role": candidate["observer_role"],
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
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report = {
        "schema": "metagross-cycle36-reactivation-selection-report/v1",
        "train_battles": len(train),
        "natural_events": len(inventory),
        "natural_battles": len({row["battle_id"] for row in inventory}),
        "selected_battles": len(rows),
        "selected_states": len(rows),
        "by_family_role_tag": {
            f"{category}:{role}:{tag}": sum(
                row["family"] == category
                and row["changed_role"] == role
                and row["reactivation_tag"] == tag
                for row in rows
            )
            for category in FAMILY_TARGETS
            for role in ("p1", "p2")
            for tag in ("switch", "drag")
        },
        "label_blind_selection_failures": failures,
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
