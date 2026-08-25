#!/usr/bin/env python3
"""Extract per-decision belief-evaluation records from pinned TRAIN battles.

One extraction pass feeds three consumers (Belief-v2 program):
  (a) TSSR evaluation records (replayable public streams per POV),
  (b) Component-1 evidence-table counting,
  (c) later 1b/1c training data.

Truth labels (each side's actual generated team, parsed from the pinned
inputlog) are written to a SEPARATE labels file. Evaluation joins on
battle_id; agents/models never read the labels file. Public prefixes are
delta-encoded per state to keep records compact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

from experimental.src.scripts import cycle12_replay_audit as v12

ROOT = Path(__file__).resolve().parents[3]
HARNESS = ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs"


class ExtractError(RuntimeError):
    pass


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def true_teams_from_first_requests(roles: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Each side's actual generated team, from that side's own first private
    request (the pinned sim regenerates teams from the inputlog seeds; the
    first request carries the full side). Used as evaluation labels only."""
    teams: dict[str, list[dict]] = {}
    for role in ("p1", "p2"):
        states = roles.get(role) or []
        if not states:
            raise ExtractError(f"missing POV stream for {role}")
        side = (states[0]["private_request"].get("side") or {})
        pokemon = side.get("pokemon")
        if not isinstance(pokemon, list) or len(pokemon) != 6:
            raise ExtractError(f"first request lacks a full team for {role}")
        teams[role] = pokemon
    return teams


def delta_encoded_states(states: list[dict]) -> list[dict]:
    """Per-state record with delta-encoded public prefix (reconstruct by
    cumulative concatenation)."""
    rows = []
    previous: list[str] = []
    for index, state in enumerate(states):
        current = list(state["public_prefix"])
        if current[: len(previous)] != previous:
            raise ExtractError("public prefix regressed during extraction")
        rows.append({
            "request_index": index,
            "actionable": bool(state.get("actionable")),
            "prefix_delta": current[len(previous):],
            "private_request": state["private_request"],
            "chosen_action": state.get("chosen_action"),
        })
        previous = current
    return rows


def extract_battle(row: dict, corpus: dict, worktrees: dict, index_root: Path,
                   tmp_root: Path) -> tuple[dict, dict]:
    with gzip.open(index_root / row["relative_index"], "rt", encoding="ascii") as handle:
        battle = json.load(handle)
    source = corpus[battle["raw_sha256"]]
    raw_path = Path(source["raw_path"])
    if sha256_path(raw_path) != battle["raw_sha256"]:
        raise ExtractError("raw replay hash changed")
    raw = json.loads(raw_path.read_text())
    temp = Path(tempfile.mkdtemp(prefix="extract-", dir=tmp_root))
    try:
        out = temp / "capture"
        subprocess.run([
            "node", str(HARNESS), "--showdown", worktrees[battle["showdown_commit"]],
            "--input", str(raw_path), "--out-dir", str(out),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        public = json.loads((out / "public.json").read_text())
        roles = {}
        for role in ("p1", "p2"):
            pov = json.loads((out / f"{role}.json").read_text())
            derived = v12.materialize_role(
                battle_id=battle["battle_id"], role=role,
                public_capture=public, pov_capture=pov,
                inputlog=raw["inputlog"], showdown_commit=battle["showdown_commit"],
            )
            roles[role] = delta_encoded_states(derived["states"])
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    record = {
        "schema": "metagross-belief-eval-record/v1",
        "battle_id": battle["battle_id"],
        "dependency_cluster_id": battle["dependency_cluster_id"],
        "split": battle["split"],
        "raw_sha256": battle["raw_sha256"],
        "inputlog_sha256": battle["inputlog_sha256"],
        "showdown_commit": battle["showdown_commit"],
        "roles": roles,
    }
    labels = {
        "schema": "metagross-belief-eval-labels/v1",
        "battle_id": battle["battle_id"],
        "raw_sha256": battle["raw_sha256"],
        "true_teams": true_teams_from_first_requests(roles),
        "labels_are_evaluation_only": True,
    }
    return record, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--worktrees", type=Path, required=True)
    parser.add_argument("--output-records", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    corpus = {
        row["raw_sha256"]: row
        for row in map(json.loads, args.corpus.read_text().splitlines())
    }
    worktrees = json.loads(args.worktrees.read_text())
    rows = [json.loads(line) for line in args.selection.read_text().splitlines() if line]
    rows = rows[args.offset: (args.offset + args.limit) if args.limit else None]
    tmp_root = args.tmp_root or Path(tempfile.gettempdir())
    tmp_root.mkdir(parents=True, exist_ok=True)
    args.output_records.parent.mkdir(parents=True, exist_ok=True)

    done = failed = 0
    with args.output_records.open("x", encoding="utf-8") as records_handle, \
            args.output_labels.open("x", encoding="utf-8") as labels_handle:
        for row in rows:
            try:
                record, labels = extract_battle(
                    row, corpus, worktrees, args.index_root, tmp_root)
            except Exception as exc:
                failed += 1
                records_handle.write(json.dumps({
                    "schema": "metagross-belief-eval-record/v1",
                    "battle_id": row.get("battle_id"),
                    "status": "failed", "failure_class": type(exc).__name__,
                }, sort_keys=True, separators=(",", ":")) + "\n")
                continue
            done += 1
            records_handle.write(json.dumps(record, sort_keys=True,
                                            separators=(",", ":")) + "\n")
            labels_handle.write(json.dumps(labels, sort_keys=True,
                                           separators=(",", ":")) + "\n")
            if done % 25 == 0:
                print(json.dumps({"done": done, "failed": failed}), flush=True)
    print(json.dumps({"done": done, "failed": failed,
                      "records_sha256": sha256_path(args.output_records),
                      "labels_sha256": sha256_path(args.output_labels)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
