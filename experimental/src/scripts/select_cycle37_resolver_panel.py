#!/usr/bin/env python3
"""Select label-blind TRAIN own request/public battle-form mismatch roots."""

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
from srcs.metagross.causal_reveal_ledger import canonical_species, norm


ROOT = Path(__file__).resolve().parents[3]
RANK_DOMAIN = "cycle37-own-private-active-resolver-v1"
PER_ROLE = 12


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


def has_battle_form_event(log: str) -> bool:
    return "|-formechange|" in log or "|detailschange|" in log


def public_active_exact(prefix: list[str], role: str) -> str:
    exact = ""
    for line in prefix:
        parts = line.split("|")
        tag = parts[1] if len(parts) > 1 else ""
        if tag in {"switch", "drag", "replace", "detailschange"} and len(parts) >= 4:
            if parts[2][:2] == role:
                exact = norm(parts[3].split(",", 1)[0])
        elif tag == "-formechange" and len(parts) >= 4 and parts[2][:2] == role:
            exact = norm(parts[3].split(",", 1)[0])
    return exact


def mismatch_rows(
    source: dict[str, Any], battle: dict[str, Any], worktree: str
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="cycle37-select-") as temporary:
        output = Path(temporary)
        subprocess.run([
            "node", str(ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs"),
            "--showdown", worktree, "--input", source["raw_path"],
            "--out-dir", str(output),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        raw = json.loads(Path(source["raw_path"]).read_text())
        public = json.loads((output / "public.json").read_text())
        results = []
        for role in ("p1", "p2"):
            pov = json.loads((output / f"{role}.json").read_text())
            materialized = v12.materialize_role(
                battle_id=source["battle_id"], role=role,
                public_capture=public, pov_capture=pov, inputlog=raw["inputlog"],
                showdown_commit=source["showdown_commit"],
            )
            for state in materialized["states"]:
                if not state.get("actionable"):
                    continue
                active_rows = [
                    row for row in state["private_request"]["side"]["pokemon"]
                    if row.get("active") is True
                ]
                if len(active_rows) != 1:
                    continue
                request_exact = norm(active_rows[0]["details"].split(",", 1)[0])
                public_exact = public_active_exact(state["public_prefix"], role)
                if (
                    not public_exact or public_exact == request_exact
                    or canonical_species(public_exact) != canonical_species(request_exact)
                ):
                    continue
                compact = c13._compact_state(materialized, pov, state["request_index"])
                results.append({
                    "role": role,
                    "state": state,
                    "compact": compact,
                    "public_exact_active": public_exact,
                    "request_exact_active": request_exact,
                    "canonical_active": canonical_species(request_exact),
                })
                break
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--worktrees", type=Path, required=True)
    parser.add_argument("--cycle36-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.master_index, "rt", encoding="utf-8") as handle:
        master = [json.loads(line) for line in handle if line.strip()]
    train = {row["battle_id"]: row for row in master if row["split"] == "train"}
    excluded_clusters = {
        json.loads(line)["dependency_cluster_id"]
        for line in args.cycle36_selection.read_text().splitlines() if line
    }
    corpus = {
        row["battle_id"]: row
        for row in (json.loads(line) for line in args.corpus.read_text().splitlines() if line)
    }
    worktrees = json.loads(args.worktrees.read_text())
    candidates = []
    for battle_id, battle in train.items():
        source = corpus.get(battle_id)
        if (
            source is None or not source.get("commit_present")
            or battle["dependency_cluster_id"] in excluded_clusters
        ):
            continue
        raw = json.loads(Path(source["raw_path"]).read_text())
        if has_battle_form_event(raw["log"]):
            candidates.append({
                "battle": battle,
                "source": source,
                "rank_sha256": rank(battle["dependency_cluster_id"]),
            })
    found = []
    failures: dict[str, int] = {}
    for candidate in sorted(candidates, key=lambda row: row["rank_sha256"]):
        source, battle = candidate["source"], candidate["battle"]
        try:
            rows = mismatch_rows(source, battle, worktrees[source["showdown_commit"]])
        except Exception as exc:
            key = f"{type(exc).__name__}:{exc}"
            failures[key] = failures.get(key, 0) + 1
            continue
        for row in rows:
            if sum(value["role"] == row["role"] for value in found) >= PER_ROLE:
                continue
            found.append({**candidate, **row})
        if all(sum(row["role"] == role for row in found) == PER_ROLE for role in ("p1", "p2")):
            break
    if any(sum(row["role"] == role for row in found) != PER_ROLE for role in ("p1", "p2")):
        raise RuntimeError("natural TRAIN resolver panel cannot fill 12 roots per role")
    if len({row["battle"]["dependency_cluster_id"] for row in found}) != len(found):
        raise RuntimeError("resolver panel dependency clusters are not unique")
    output_rows = []
    for row in found:
        source, battle = row["source"], row["battle"]
        state, compact = row["state"], row["compact"]
        output_rows.append({
            "schema": "metagross-cycle37-resolver-panel/v1",
            "split": "train",
            "battle_id": source["battle_id"],
            "dependency_cluster_id": battle["dependency_cluster_id"],
            "rank_sha256": row["rank_sha256"],
            "raw_path": source["raw_path"],
            "raw_sha256": source["raw_sha256"],
            "showdown_commit": source["showdown_commit"],
            "role": row["role"],
            "request_index": state["request_index"],
            "public_exact_active": row["public_exact_active"],
            "request_exact_active": row["request_exact_active"],
            "canonical_active": row["canonical_active"],
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
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows)
    )
    report = {
        "schema": "metagross-cycle37-resolver-selection-report/v1",
        "train_battles": len(train),
        "candidate_form_event_battles": len(candidates),
        "screened_battles": sum(1 for _ in sorted(candidates, key=lambda row: row["rank_sha256"])
                                if _["rank_sha256"] <= candidate["rank_sha256"]),
        "selected_battles": len(output_rows),
        "selected_states": len(output_rows),
        "selected_by_role": {
            role: sum(row["role"] == role for row in output_rows) for role in ("p1", "p2")
        },
        "selected_by_canonical": dict(sorted(__import__("collections").Counter(
            row["canonical_active"] for row in output_rows
        ).items())),
        "label_blind_selection_failures": failures,
        "excluded_cycle36_clusters": len(excluded_clusters),
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
