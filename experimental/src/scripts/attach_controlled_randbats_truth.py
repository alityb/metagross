#!/usr/bin/env python3
"""Attach private controlled-replay Randbats truth labels to public JSONL rows.

Private simulator manifests are used only while this script runs.  They are
never copied into the emitted public state or candidate records.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from belief.action_conditioned_randbats import Candidate, CandidateValidationError, load_generator_pool_active_candidates


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _messages(raw: Mapping[str, Any]) -> list[list[str]]:
    log = raw.get("log")
    if not isinstance(log, str):
        raise ValueError("replay has no string log")
    return [[part.strip() for part in line.split("|")[1:]] for line in log.splitlines() if line.replace("|", "").strip()]


def _players_by_side(raw: Mapping[str, Any]) -> dict[str, str]:
    players: dict[str, str] = {}
    for message in _messages(raw):
        if len(message) >= 3 and message[0] == "player" and message[1] in {"p1", "p2"}:
            players[message[1]] = message[2]
    if set(players) != {"p1", "p2"} or not all(players.values()):
        raise ValueError("replay lacks p1/p2 player metadata")
    raw_players = raw.get("players")
    if isinstance(raw_players, list) and all(isinstance(player, str) for player in raw_players):
        if set(raw_players) != set(players.values()):
            raise ValueError("replay players metadata disagrees with protocol players")
    return players


def _active_species(row: Mapping[str, Any], side: str) -> str:
    public_state = row.get("public_state")
    if not isinstance(public_state, Mapping):
        raise ValueError("row has no public_state")
    prefix = public_state.get("protocol_prefix")
    if not isinstance(prefix, list):
        raise ValueError("row public_state has no protocol_prefix")
    active: str | None = None
    for message in prefix:
        if not isinstance(message, list) or len(message) < 3 or message[0] not in {"switch", "drag", "replace"}:
            continue
        if isinstance(message[1], str) and message[1].startswith(f"{side}a:"):
            # The species before the comma is public in a switch protocol line.
            active = message[2].split(",", 1)[0]
    if not active:
        raise ValueError("cannot identify acting active Pokemon from protocol prefix")
    return _norm(active)


def _set_species(set_: Mapping[str, Any]) -> str:
    return _norm(set_.get("speciesId", set_.get("species", set_.get("name", ""))))


def _moves(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list) or len(value) != 4 or not all(isinstance(move, str) and move for move in value):
        return None
    normalized = tuple(sorted(_norm(move) for move in value))
    return normalized if len(set(normalized)) == 4 else None


def _evs(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, int] = {}
    for stat, amount in value.items():
        if isinstance(amount, bool) or not isinstance(amount, int):
            return None
        normalized[_norm(stat)] = amount
    return normalized


def _set_matches_candidate(set_: Mapping[str, Any], candidate: Candidate) -> bool:
    data = candidate.public_data or {}
    # Prefer speciesId on each record: Showdown uses it to distinguish forms.
    if _set_species(set_) != _norm(data.get("speciesId", data.get("species", ""))):
        return False
    if set_.get("level") != data.get("level"):
        return False
    if _moves(set_.get("moves")) != _moves(data.get("moves")):
        return False
    for key in ("ability", "item", "teraType"):
        if _norm(set_.get(key, "")) != _norm(data.get(key, "")) or not set_.get(key) or not data.get(key):
            return False
    return _evs(set_.get("evs")) == _evs(data.get("evs")) and _evs(set_.get("evs")) is not None


def _read_manifest_rows(manifest_dir: Path) -> list[Mapping[str, Any]]:
    paths = sorted(manifest_dir.rglob("*.jsonl")) + sorted(manifest_dir.rglob("*.json"))
    if not paths:
        raise ValueError(f"no manifest files found in {manifest_dir}")
    rows: list[Mapping[str, Any]] = []
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8")
            decoded = json.loads(content) if path.suffix == ".json" else [json.loads(line) for line in content.splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read manifest {path}: {exc}") from exc
        decoded_rows = decoded if isinstance(decoded, list) else [decoded]
        if not all(isinstance(row, Mapping) for row in decoded_rows):
            raise ValueError(f"manifest {path} contains a non-object row")
        rows.extend(decoded_rows)
    return rows


def _manifest_captures(rows: list[Mapping[str, Any]]) -> dict[frozenset[str], list[dict[str, Mapping[str, Any]]]]:
    by_capture: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        capture_id, side, player, team = row.get("capture_id"), row.get("side"), row.get("player"), row.get("team")
        if not isinstance(capture_id, str) or not isinstance(side, str) or not isinstance(player, str) or not isinstance(team, list):
            continue
        by_capture[capture_id].append(row)
    indexed: dict[frozenset[str], list[dict[str, Mapping[str, Any]]]] = defaultdict(list)
    for capture_rows in by_capture.values():
        sides = {row["side"]: row for row in capture_rows}
        if set(sides) != {"p1", "p2"} or len(capture_rows) != 2:
            continue
        pair = frozenset(str(row["player"]) for row in capture_rows)
        if len(pair) == 2:
            indexed[pair].append(sides)
    return indexed


def _replay_index(replay_dir: Path) -> dict[str, list[Mapping[str, Any]]]:
    paths = sorted(replay_dir.rglob("*.json"))
    if not paths:
        raise ValueError(f"no replay files found in {replay_dir}")
    indexed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read replay {path}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError(f"replay {path} is not an object")
        for replay_id in {path.stem, raw.get("id")}:
            if isinstance(replay_id, str) and replay_id:
                indexed[replay_id].append(raw)
    return indexed


def attach(rows_path: Path, replay_dir: Path, manifest_dir: Path, pool_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    pool = load_generator_pool_active_candidates(pool_path)
    manifests = _manifest_captures(_read_manifest_rows(manifest_dir))
    replays = _replay_index(replay_dir)
    report: dict[str, Any] = {"input_rows": 0, "output_rows": 0, "rejected_rows": 0, "rejection_reasons": {}}
    output_rows: list[dict[str, Any]] = []
    try:
        lines = rows_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read rows: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        report["input_rows"] += 1
        try:
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError("row is not an object")
            replay_id = row.get("replay_id")
            if not isinstance(replay_id, str) or len(replays.get(replay_id, [])) != 1:
                raise ValueError("replay_id_not_unique")
            raw = replays[replay_id][0]
            players = _players_by_side(raw)
            acting_side = row.get("public_state", {}).get("acting_side") if isinstance(row.get("public_state"), Mapping) else None
            if acting_side not in players:
                raise ValueError("invalid_acting_side")
            captures = manifests.get(frozenset(players.values()), [])
            if len(captures) != 1:
                raise ValueError("manifest_capture_pair_not_unique")
            manifest = captures[0][acting_side]
            if manifest["player"] != players[acting_side]:
                raise ValueError("manifest_acting_player_mismatch")
            active_species = _active_species(row, acting_side)
            active_sets = [set_ for set_ in manifest["team"] if isinstance(set_, Mapping) and _set_species(set_) == active_species]
            if len(active_sets) != 1:
                raise ValueError("active_manifest_set_not_unique")
            matches = [candidate for candidate in pool if _set_matches_candidate(active_sets[0], candidate)]
            if len(matches) != 1:
                raise ValueError("pool_candidate_match_not_unique")
            candidate_ids = {candidate.get("candidate_id") for candidate in row.get("active_candidates", []) if isinstance(candidate, Mapping)}
            if matches[0].candidate_id not in candidate_ids:
                raise ValueError("label_not_in_active_candidates")
            labeled = dict(row)
            labeled["label"] = matches[0].candidate_id
            output_rows.append(labeled)
        except (json.JSONDecodeError, ValueError, TypeError, KeyError, CandidateValidationError) as exc:
            reason = str(exc) or type(exc).__name__
            report["rejected_rows"] += 1
            report["rejection_reasons"][reason] = report["rejection_reasons"].get(reason, 0) + 1
    report["output_rows"] = len(output_rows)
    report["label_coverage"] = len(output_rows) / report["input_rows"] if report["input_rows"] else 0.0
    report["output"] = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if not output_rows:
        raise ValueError("no labeled rows output")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(attach(args.rows, args.replay_dir, args.manifest_dir, args.pool, args.output, args.report), sort_keys=True))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
