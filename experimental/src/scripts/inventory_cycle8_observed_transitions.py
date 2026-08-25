#!/usr/bin/env python3
"""Read-only Cycle 8 inventory of already-opened observed-transition sources.

The inventory deliberately does not read any search-teacher value, visit, Q, or
label field.  It inventories source identity, sequential-state counts, causal
capture capabilities, terminal joins, and the largest observable dependency
cluster.  The sealed confirmation split is not an input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import lz4.frame


ROOT = Path(__file__).resolve().parents[3]
PHYSICAL_ID = re.compile(r"(?:battle-)?gen9randombattle-([0-9]+)")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(paths: Iterable[Path], *, anchor: Path = ROOT) -> tuple[str, int, int]:
    """Hash relative path, size, and content hash for a selected immutable tree."""
    digest = hashlib.sha256()
    files = sorted({path.resolve() for path in paths if path.is_file()})
    byte_count = 0
    for path in files:
        try:
            rel = path.relative_to(anchor.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        size = path.stat().st_size
        byte_count += size
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        digest.update(file_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest(), len(files), byte_count


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_index(index: Path) -> list[str]:
    with index.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if rows and rows[0] == ["filename"]:
        rows = rows[1:]
    return [row[0].strip() for row in rows if row and row[0].strip()]


def physical_id(name: str) -> str | None:
    match = PHYSICAL_ID.search(name)
    if match is None:
        return None
    numeric = match.group(1)
    # Public human replay IDs are globally assigned.  Generated self-play uses
    # many local Showdown servers whose numeric tags collide, so bind the tag
    # to the unordered player pair when the filename exposes it.
    if "battle-gen9randombattle-" not in name or "_Unrated_" not in name:
        return numeric
    tail = name.split("_Unrated_", 1)[1]
    players = re.match(r"(.+)_vs_(.+)_\d{2}-\d{2}-\d{4}(?:-\d{2}:\d{2}:\d{2})?_(?:WIN|LOSS)\.json\.lz4$", tail)
    if players is None:
        return numeric
    return numeric + "|" + "|".join(sorted((players.group(1), players.group(2))))


def trajectory_inventory(directory: Path, *, index: Path | None = None) -> dict[str, Any]:
    names = read_index(index) if index is not None else [path.relative_to(directory).as_posix() for path in directory.rglob("*.lz4")]
    paths = [(directory / name) if index is not None else (ROOT / name) for name in names]
    # An index is authoritative, but missing payloads count as unavailable.
    existing = [path for path in paths if path.exists()]
    state_rows = 0
    malformed = 0
    terminal_named = 0
    physical: set[str] = set()
    for name, path in zip(names, paths):
        pid = physical_id(name)
        if pid is not None:
            physical.add(pid)
        terminal_named += int("_WIN.json.lz4" in name or "_LOSS.json.lz4" in name)
        if not path.exists():
            continue
        try:
            with lz4.frame.open(path, "rb") as handle:
                record = json.load(handle)
            states = record.get("states")
            actions = record.get("actions")
            if not isinstance(states, list) or not isinstance(actions, list) or len(states) != len(actions):
                malformed += 1
                continue
            state_rows += len(states)
        except Exception:
            malformed += 1
    digest, file_count, byte_count = tree_sha256([*(existing), *([index] if index else [])])
    return {
        "artifact_tree_sha256": digest,
        "artifact_files": file_count,
        "artifact_bytes": byte_count,
        "trajectory_records": len(names),
        "payloads_present": len(existing),
        "physical_battle_tokens": len(physical),
        "sequential_decision_states": state_rows,
        "malformed_or_unreadable": malformed,
        "terminal_outcome_in_filename": terminal_named,
    }


def raw_human_replay_inventory(directory: Path, showdown_repo: Path) -> dict[str, Any]:
    paths = sorted(directory.glob("*.json"))
    versions: Counter[str] = Counter()
    parsed: list[dict[str, Any]] = []
    all_choice_commands = 0
    for path in paths:
        record = json.loads(path.read_text())
        inputlog = record.get("inputlog") or ""
        public_log = record.get("log") or ""
        version: str | None = None
        start: dict[str, Any] | None = None
        player_records: list[dict[str, Any]] = []
        p1_choices = 0
        p2_choices = 0
        for line in inputlog.splitlines():
            if line.startswith(">version "):
                version = line.split(" ", 1)[1].strip()
            elif line.startswith(">start "):
                try:
                    start = json.loads(line.split(" ", 1)[1])
                except json.JSONDecodeError:
                    start = None
            elif line.startswith(">player "):
                try:
                    player_records.append(json.loads(line.split(" ", 2)[2]))
                except (json.JSONDecodeError, IndexError):
                    pass
            elif line.startswith(">p1 "):
                p1_choices += 1
            elif line.startswith(">p2 "):
                p2_choices += 1
        if version:
            versions[version] += 1
        terminal = any(
            line.startswith("|win|") or line.startswith("|tie|")
            for line in public_log.splitlines()
        )
        start_seed = start.get("seed") if isinstance(start, dict) else None
        complete = bool(
            inputlog
            and public_log
            and version
            and start_seed
            and len(player_records) == 2
            and terminal
        )
        choices = p1_choices + p2_choices
        all_choice_commands += choices
        parsed.append({
            "battle_id": str(record.get("id")),
            "version": version,
            "start_seed": str(start_seed) if start_seed else None,
            "complete": complete,
            "both_choice_streams": bool(p1_choices and p2_choices),
            "choice_commands": choices,
        })

    commit_available: dict[str, bool] = {}
    for commit in versions:
        result = subprocess.run(
            ["git", "-C", str(showdown_repo), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        commit_available[commit] = result.returncode == 0
    eligible = [row for row in parsed if row["complete"] and commit_available.get(str(row["version"]), False)]
    missing_commit = [row for row in parsed if row["complete"] and not commit_available.get(str(row["version"]), False)]
    digest, file_count, byte_count = tree_sha256(paths)
    return {
        "root": rel(directory),
        "artifact_tree_sha256": digest,
        "artifact_files": file_count,
        "artifact_bytes": byte_count,
        "raw_replays": len(paths),
        "causally_complete_terminal_replays": sum(row["complete"] for row in parsed),
        "locally_pinned_replayable_battles": len(eligible),
        "locally_pinned_both_choice_streams": sum(row["both_choice_streams"] for row in eligible),
        "locally_pinned_choice_commands": sum(row["choice_commands"] for row in eligible),
        "all_choice_commands": all_choice_commands,
        "missing_commit_battles": len(missing_commit),
        "missing_commit_choice_commands": sum(row["choice_commands"] for row in missing_commit),
        "unique_battle_ids": len({row["battle_id"] for row in eligible}),
        "unique_start_seed_clusters": len({row["start_seed"] for row in eligible}),
        "showdown_versions": [
            {"commit": commit, "battles": count, "present_locally": commit_available[commit]}
            for commit, count in versions.most_common()
        ],
        "largest_dependency_cluster": "start_seed_before_labels; later strengthened to generated unordered team-pair hash after deterministic replay",
        "capabilities": {
            "exact_own_private_request_stored": False,
            "exact_own_private_request_deterministically_rematerializable": True,
            "exact_own_team_deterministically_rematerializable": True,
            "causal_protocol_prefix_stored": True,
            "exact_engine_mechanical_state_deterministically_rematerializable": True,
            "terminal_outcome": True,
            "upgrade_without_hidden_leakage": "conditional_on_preregistered_byte-exact_dual-view_replay_audit; each actor receives only its own request plus public protocol",
        },
    }


def schema6_smoke_inventory(run_dirs: list[Path]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    total_games = total_pairs = total_rows = materialized_games = 0
    selected_files: list[Path] = []
    for directory in run_dirs:
        progress = directory / "result.json.progress.json"
        if not progress.exists():
            continue
        games = [
            game for game in json.loads(progress.read_text()).get("games", [])
            if not game.get("void") and game.get("winner")
        ]
        completed_tags = {str(game["battle_tag"]) for game in games}
        pairs = {str(game.get("pair_id")) for game in games if game.get("pair_id")}
        root_files = sorted(directory.glob("agent-*-decisions.jsonl.dual-r1-roots.jsonl"))
        rows = 0
        root_tags: set[str] = set()
        for path in root_files:
            with path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows += 1
                    root_tags.add(str(row.get("identity", {}).get("battle_tag")))
        logs = sorted((directory / "logs").glob("*.protocol.jsonl")) + sorted((directory / "logs").glob("*.search.jsonl"))
        selected_files.extend([progress, directory / "result.json.pairs.json", *root_files, *logs])
        total_games += len(completed_tags)
        total_pairs += len(pairs)
        total_rows += rows
        materialized_games += len(completed_tags & root_tags)
        runs.append({
            "run": rel(directory),
            "completed_games": len(completed_tags),
            "pair_clusters": len(pairs),
            "materialized_exact_root_games": len(completed_tags & root_tags),
            "materialized_sequential_states": rows,
            "complete_private_and_protocol_logs_but_root_export_pending": len(completed_tags - root_tags),
        })
    digest, file_count, byte_count = tree_sha256(selected_files)
    return {
        "artifact_tree_sha256": digest,
        "artifact_files": file_count,
        "artifact_bytes": byte_count,
        "runs": runs,
        "completed_games": total_games,
        "pair_clusters": total_pairs,
        "materialized_exact_root_games": materialized_games,
        "materialized_sequential_states": total_rows,
        "root_export_pending_games": total_games - materialized_games,
        "largest_dependency_cluster": "pair_id",
        "capabilities": {
            "exact_private_protocol_capture": True,
            "exact_engine_root_materialized": f"{materialized_games}/{total_games} games",
            "terminal_outcome": True,
            "upgrade_without_hidden_leakage": "materialized rows ready; remaining complete capture logs require separately audited exporter",
        },
    }


def public_replay_identity(record: dict[str, Any]) -> str:
    payload = {
        "format": record.get("format"),
        "id": record.get("id"),
        "log": record.get("log"),
        "players": record.get("players"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def mcts_archive_inventory(root: Path, excluded_source_battles: set[tuple[str, str]]) -> dict[str, Any]:
    decisions = sorted(root.glob("**/agent_a_decisions.jsonl"))
    manifests = sorted(root.glob("w*/MANIFEST.json"))
    result_paths = sorted(root.glob("**/result.json"))
    replay_paths = sorted(root.glob("**/replays/*.json"))
    battle_rows: Counter[tuple[str, str]] = Counter()
    namespaces: set[str] = set()
    state_rows = 0
    state_missing = 0
    for path in decisions:
        shard = rel(path.parent)
        namespaces.add(shard)
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("record_type") != "decision":
                    continue
                # Do not inspect visits, values, targets, or labels.
                tag = str(row.get("battle_tag"))
                battle_rows[(shard, tag)] += 1
                state_rows += 1
                state_missing += int(not isinstance(row.get("state"), str) or not row.get("state"))

    result_by_shard: dict[tuple[str, str], dict[str, Any]] = {}
    for path in result_paths:
        shard = rel(path.parent)
        record = json.loads(path.read_text())
        for game in record.get("games", []):
            result_by_shard[(shard, str(game.get("battle_tag")))] = game

    replay_by_shard: dict[tuple[str, str], str] = {}
    replay_hashes: set[str] = set()
    for path in replay_paths:
        shard = rel(path.parent.parent)
        record = json.loads(path.read_text())
        tag = str(record.get("id") or path.stem)
        identity = public_replay_identity(record)
        replay_by_shard[(shard, tag)] = identity
        replay_hashes.add(identity)

    completed: set[tuple[str, str]] = set()
    for key in battle_rows:
        game = result_by_shard.get(key)
        if game is not None and not game.get("void") and game.get("winner") in {"agent_a", "agent_b"}:
            completed.add(key)
    replay_joined = set(battle_rows) & set(replay_by_shard)
    full_join = completed & replay_joined
    excluded = set(battle_rows) & excluded_source_battles
    eligible = full_join - excluded
    eligible_rows = sum(battle_rows[key] for key in eligible)
    digest, file_count, byte_count = tree_sha256([*decisions, *manifests, *result_paths, *replay_paths])
    return {
        "root": rel(root),
        "artifact_tree_sha256": digest,
        "artifact_files": file_count,
        "artifact_bytes": byte_count,
        "decision_files": len(decisions),
        "sequential_decision_states": state_rows,
        "decision_battles_source_scoped": len(battle_rows),
        "exact_engine_state_missing": state_missing,
        "terminal_completed_battles": len(completed),
        "public_replay_battles": len(replay_joined),
        "fully_joined_battles": len(full_join),
        "canonical_public_replay_identities": len(replay_hashes),
        "excluded_opened_dev_battles": len(excluded),
        "eligible_fully_joined_battles": len(eligible),
        "eligible_sequential_decision_states": eligible_rows,
        "largest_known_dependency_clusters": {
            "kind": "source_generation_worker_shard",
            "count": len(namespaces),
            "note": "per-game seed, mirrored-pair, and team identities were not stored; shard is the conservative observable seed namespace",
        },
        "capabilities": {
            "exact_own_private_request_stored": False,
            "exact_own_private_team_in_engine_state": True,
            "causal_protocol_prefix_stored": False,
            "causal_public_replay_available": bool(replay_paths),
            "exact_engine_mechanical_state": state_missing == 0,
            "terminal_outcome": bool(completed),
            "upgrade_without_hidden_leakage": "conditional_public_replay_alignment_and_opponent_masking_gate_required",
        },
        "_public_replay_identity_set": replay_hashes,
    }


def schema6_inventory(base: Path) -> dict[str, Any]:
    progress = json.loads((base / "result.json.progress.json").read_text())
    completed_games = [game for game in progress.get("games", []) if not game.get("void") and game.get("winner")]
    tags = {str(game["battle_tag"]) for game in completed_games}
    pair_ids = {str(game["pair_id"]) for game in completed_games}
    seeds = {str(game["battle_seed"]) for game in completed_games}
    teams = {
        str(game[field])
        for game in completed_games
        for field in ("team_1_sha256", "team_2_sha256")
    }
    roots = sorted(base.glob("agent-*-decisions.jsonl.dual-r1-roots.jsonl"))
    rows = 0
    eligible_rows = 0
    all_tags: set[str] = set()
    missing_contract = 0
    for path in roots:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                tag = str(row.get("identity", {}).get("battle_tag"))
                all_tags.add(tag)
                snapshot = row.get("r1_policy_snapshot", {})
                info = snapshot.get("player_information_state", {})
                complete = (
                    isinstance(row.get("state"), str)
                    and bool(row.get("state"))
                    and isinstance(snapshot.get("protocol_prefix"), list)
                    and isinstance(info.get("private_request"), dict)
                    and isinstance(info.get("player_team"), list)
                )
                missing_contract += int(not complete)
                eligible_rows += int(tag in tags and complete)
    files = [*roots, base / "result.json.progress.json", base / "result.json.pairs.json"]
    digest, file_count, byte_count = tree_sha256(files)
    return {
        "root": rel(base),
        "artifact_tree_sha256": digest,
        "artifact_files": file_count,
        "artifact_bytes": byte_count,
        "completed_battles": len(tags),
        "mirrored_pair_clusters": len(pair_ids),
        "team_clusters": len(teams),
        "battle_seed_clusters": len(seeds),
        "root_battles_all": len(all_tags),
        "incomplete_smoke_battles_excluded": len(all_tags - tags),
        "sequential_decision_states_all": rows,
        "eligible_sequential_decision_states": eligible_rows,
        "contract_rows_missing": missing_contract,
        "largest_dependency_cluster": "pair_id",
        "capabilities": {
            "exact_own_private_request_stored": True,
            "exact_own_private_team_stored": True,
            "causal_protocol_prefix_stored": True,
            "exact_engine_mechanical_state": True,
            "terminal_outcome": True,
            "upgrade_without_hidden_leakage": "ready_subject_to_cycle8_fingerprint_and_split_audit",
        },
    }


def exclusion_inventory() -> tuple[dict[str, Any], set[tuple[str, str]]]:
    registry = ROOT / "experimental/runs/terminal_mcts_direct_controller_20260815/exclusions.json"
    record = json.loads(registry.read_text())
    battle_ids = set(map(str, record.get("battle_ids", [])))
    panels = [
        ROOT / "experimental/runs/outcome_residual_scale_20260814/panel-950.jsonl",
        ROOT / "experimental/runs/causal_action_q_local_20260814/training-panel.jsonl",
    ]
    mapped: set[tuple[str, str]] = set()
    mapped_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    for panel in panels:
        with panel.open() as handle:
            for line in handle:
                row = json.loads(line)
                if str(row.get("battle_id")) not in battle_ids:
                    continue
                source = Path(str(row["source_path"])).resolve().parent
                key = (source.relative_to(ROOT.resolve()).as_posix(), str(row["battle_tag"]))
                mapped.add(key)
                mapped_ids.add(str(row["battle_id"]))
                source_counts["mcts_v3_final" if "mcts_v3_final_snapshot" in key[0] else "other"] += 1
    return ({
        "registry": rel(registry),
        "registry_sha256": file_sha256(registry),
        "opened_development_battles": len(battle_ids),
        "mapped_source_scoped_battles": len(mapped),
        "unmapped_battle_ids": len(battle_ids - mapped_ids),
        "source_counts": dict(sorted(source_counts.items())),
        "sealed_confirmation": {
            "identity_rows_read": 0,
            "materialized": False,
            "rule": "excluded by distinct unopened collection scope; no sealed panel path was read or hashed",
        },
    }, mapped)


def old_visit_inventory(path: Path) -> dict[str, Any]:
    # Only identity/schema keys are read.  Visit distributions, policy values,
    # selected actions, and terminal labels are deliberately unopened.
    rows = 0
    tags: set[str] = set()
    users: set[str] = set()
    schemas: set[int] = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            tags.add(str(row.get("battle_tag")))
            users.add(str(row.get("username")))
            if isinstance(row.get("schema"), int):
                schemas.add(row["schema"])
    return {
        "path": rel(path),
        "sha256": file_sha256(path),
        "rows": rows,
        "battle_tag_tokens": len(tags),
        "username_tokens": len(users),
        "schema_versions": sorted(schemas),
        "teacher_value_fields_opened": False,
        "use": "identity-linked historical production-search control only; never a Cycle8 stability label",
    }


def online_rl_inventory(root: Path, *, include_holdout: bool = False) -> dict[str, Any]:
    trajectories = sorted(root.glob("**/learner_trajectories/**/*.lz4"))
    if not include_holdout:
        trajectories = [path for path in trajectories if "arena_" not in path.as_posix()]
    state_rows = 0
    malformed = 0
    physical: set[str] = set()
    for path in trajectories:
        pid = physical_id(path.name)
        if pid:
            physical.add(pid)
        try:
            with lz4.frame.open(path, "rb") as handle:
                row = json.load(handle)
            states, actions = row.get("states"), row.get("actions")
            if not isinstance(states, list) or not isinstance(actions, list) or len(states) != len(actions):
                malformed += 1
            else:
                state_rows += len(states)
        except Exception:
            malformed += 1
    ledgers = [path for path in root.glob("**/BATTLE_LEDGER.jsonl") if include_holdout or "arena_" not in path.as_posix()]
    files = [*trajectories, *ledgers, *root.glob("**/MANIFEST.json")]
    digest, file_count, byte_count = tree_sha256(files)
    return {
        "root": rel(root),
        "artifact_tree_sha256": digest,
        "artifact_files": file_count,
        "artifact_bytes": byte_count,
        "learner_trajectories": len(trajectories),
        "physical_battle_tokens": len(physical),
        "sequential_decision_states": state_rows,
        "malformed_or_unreadable": malformed,
        "battle_ledgers": len(ledgers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    exclusions, excluded_source_battles = exclusion_inventory()
    schema6 = schema6_inventory(ROOT / "experimental/runs/schema6_local_5000_20260814_r1/peer")
    schema6_smokes = schema6_smoke_inventory([
        ROOT / "experimental/runs/schema6_fresh_capture_direct_r1_smoke_20260814",
        ROOT / "experimental/runs/schema6_fresh_capture_release_smoke_20260814",
        ROOT / "experimental/runs/schema6_fresh_capture_smoke_v2_20260814",
        ROOT / "experimental/runs/schema6_fresh_capture_smoke_v4_20260814",
        ROOT / "experimental/runs/schema6_fresh_capture_unguided_smoke_20260814",
    ])
    raw_human = raw_human_replay_inventory(
        ROOT / "experimental/data/replays/gen9randombattle",
        ROOT / "external/pokemon-showdown",
    )
    raw_human_external = raw_human_replay_inventory(
        ROOT / "experimental/external/replays/gen9randombattle",
        ROOT / "external/pokemon-showdown",
    )
    primary_ids = {
        json.loads(path.read_text()).get("id")
        for path in (ROOT / "experimental/data/replays/gen9randombattle").glob("*.json")
    }
    external_ids = {
        json.loads(path.read_text()).get("id")
        for path in (ROOT / "experimental/external/replays/gen9randombattle").glob("*.json")
    }

    mcts_roots = {
        "mcts_v3_final_raw": ROOT / "experimental/runs/mcts_v3_final_snapshot/raw",
        "mcts_v3_round2_raw": ROOT / "experimental/runs/mcts_v3_round2_snapshot/raw",
        "mcts_v3_partial_raw": ROOT / "experimental/runs/mcts_v3_partial_snapshot/raw",
    }
    mcts = {name: mcts_archive_inventory(path, excluded_source_battles) for name, path in mcts_roots.items()}
    replay_sets = {name: record.pop("_public_replay_identity_set") for name, record in mcts.items()}
    mcts_overlap = {
        "final_round2": len(replay_sets["mcts_v3_final_raw"] & replay_sets["mcts_v3_round2_raw"]),
        "final_partial": len(replay_sets["mcts_v3_final_raw"] & replay_sets["mcts_v3_partial_raw"]),
        "round2_partial": len(replay_sets["mcts_v3_round2_raw"] & replay_sets["mcts_v3_partial_raw"]),
        "union": len(set().union(*replay_sets.values())),
        "dedup_rule": "canonical source-pinned public replay identity; raw battle_tag is forbidden because server namespaces collide",
        "preferred_lineage": "final supersedes partial exact copies; round2 is independent despite reused battle tags",
    }

    trajectory_sources = {}
    for name, directory in {
        "local_human_randbats": ROOT / "experimental/data/parsed_replays",
        "legacy_selfplay_indexed": ROOT / "experimental/data/selfplay_parsed_indexed",
        "strict_round2": ROOT / "experimental/data/strict_round2_parsed",
        "round2_pre_strict": ROOT / "experimental/data/selfplay_round2_parsed",
        "gate1_pfsp_all_pov": ROOT / "experimental/data/gate1_pfsp_indexed",
        "gate1_pfsp_learner_only": ROOT / "experimental/data/gate1_pfsp_learner_only",
    }.items():
        record = trajectory_inventory(directory, index=directory / "index.csv")
        record.update({
            "root": rel(directory),
            "largest_dependency_cluster": "physical_battle_token_from_filename",
            "capabilities": {
                "exact_own_private_request_stored": False,
                "exact_own_team": "simplified_parser_snapshot_only",
                "causal_protocol_prefix_stored": False,
                "exact_engine_mechanical_state": False,
                "terminal_outcome": True,
                "upgrade_without_hidden_leakage": "not_currently; requires separately admitted public-event rehydrator and must not join opposite POV hidden fields",
            },
        })
        trajectory_sources[name] = record

    online_rl = {
        "league_g5_collection": online_rl_inventory(
            ROOT / "experimental/runs/online_rl_g5_league_5k_20260730/generation_005/collection"
        ),
        "online_rl_generations_2_to_4": online_rl_inventory(
            ROOT / "experimental/runs/online_rl_autonomous_3gen_20260729"
        ),
        "online_rl_g1_vs_r1": online_rl_inventory(
            ROOT / "experimental/runs/online_rl_g1_vs_r1_n200_20260729"
        ),
    }
    for record in online_rl.values():
        record.update({
            "largest_dependency_cluster": "generation_chunk_or_battle_when_ledger_available",
            "capabilities": {
                "exact_own_private_request_stored": False,
                "exact_own_team": "simplified_learner_trajectory_only",
                "causal_protocol_prefix_stored": False,
                "exact_engine_mechanical_state": False,
                "terminal_outcome": True,
                "upgrade_without_hidden_leakage": "not_from_local_payload_alone",
            },
        })

    visit_paths = {
        "mcts_v3_final_175k": ROOT / "experimental/data/mcts_v3_final/mcts_v3_targets.jsonl",
        "mcts_v3_round2_179k": ROOT / "experimental/data/mcts_v3_round2/mcts_v3_targets.jsonl",
        "mcts_v3_partial_135k": ROOT / "experimental/data/mcts_v3_partial_snapshot/mcts_v3_targets.jsonl",
    }
    old_visits = {name: old_visit_inventory(path) for name, path in visit_paths.items()}

    modal_summary = ROOT / "experimental/runs/schema6_modal_500_20260814_r1/summary.json"
    modal_training_report = ROOT / "experimental/runs/schema6_modal_500_20260814_r1/training_panel_artifacts_v2/training-panel-report.json"
    modal_schema6 = {
        "local_summary_path": rel(modal_summary),
        "local_summary_sha256": file_sha256(modal_summary),
        "local_training_report_path": rel(modal_training_report),
        "local_training_report_sha256": file_sha256(modal_training_report),
        "completed_remote_games": 500,
        "mirrored_pair_clusters": 250,
        "local_derived_roots": 240,
        "raw_exact_capture_local": False,
        "raw_exact_capture_recoverable": "remote Modal volume only; retrieval and hash audit required before eligibility",
        "sealed_confirmation_rows_read": 0,
    }

    registry_path = ROOT / "experimental/configs/data_sources_v1.json"
    registry = json.loads(registry_path.read_text())
    metamon = []
    for source in registry["sources"]:
        if not str(source.get("id", "")).startswith("metamon_"):
            continue
        metamon.append({
            "id": source["id"],
            "location": source["location"],
            "revision": source.get("revision"),
            "selected_artifact_hashes": [item.get("sha256") for item in source.get("selected_artifacts", [])],
            "status": source.get("status"),
            "local_payload_present": False,
            "format_matched": "gen9randombattle" in source.get("formats", []),
            "capabilities": {
                "exact_own_private_request_stored": False,
                "causal_protocol_prefix_stored": False,
                "exact_engine_mechanical_state": False,
                "terminal_outcome": True,
                "upgrade_without_hidden_leakage": "no; cross-format representation/terminal anchor only under current contract",
            },
        })

    report = {
        "schema": "metagross-cycle8-observed-transition-inventory/v1",
        "created_at": "2026-08-15",
        "mode": "opened-development-read-only-no-teacher-values",
        "sealed_confirmation_rows_read": 0,
        "exclusions": exclusions,
        "sources": {
            "schema6_preserved_42": schema6,
            "schema6_completed_smokes_10": schema6_smokes,
            "schema6_modal_500_remote_recoverable": modal_schema6,
            "raw_human_deterministic_replay": {
                "primary": raw_human,
                "external": raw_human_external,
                "cross_source_battle_id_overlap": len(primary_ids & external_ids),
                "combined_locally_pinned_replayable_battles": (
                    raw_human["locally_pinned_replayable_battles"]
                    + raw_human_external["locally_pinned_replayable_battles"]
                ),
                "combined_observed_choice_commands": (
                    raw_human["locally_pinned_choice_commands"]
                    + raw_human_external["locally_pinned_choice_commands"]
                ),
            },
            "mcts_exact_state_archives": mcts,
            "parsed_human_selfplay_pfsp": trajectory_sources,
            "online_rl_league": online_rl,
            "metamon_remote_pinned": metamon,
            "old_visit_controls": old_visits,
        },
        "deduplication": mcts_overlap,
        "admission_summary": {
            "immediately_exact_materialized_battles": (
                schema6["completed_battles"] + schema6_smokes["materialized_exact_root_games"]
            ),
            "immediately_exact_pair_clusters": (
                schema6["mirrored_pair_clusters"]
                + schema6_smokes["materialized_exact_root_games"] // 2
            ),
            "immediately_exact_sequential_states": (
                schema6["eligible_sequential_decision_states"]
                + schema6_smokes["materialized_sequential_states"]
            ),
            "human_deterministically_replayable_battles_pending_gate": (
                raw_human["locally_pinned_replayable_battles"]
                + raw_human_external["locally_pinned_replayable_battles"]
            ),
            "human_observed_choice_commands_pending_gate": (
                raw_human["locally_pinned_choice_commands"]
                + raw_human_external["locally_pinned_choice_commands"]
            ),
            "conditionally_upgradeable_mcts_battles": (
                mcts["mcts_v3_final_raw"]["eligible_fully_joined_battles"]
                + mcts["mcts_v3_round2_raw"]["eligible_fully_joined_battles"]
            ),
            "conditionally_upgradeable_mcts_states": (
                mcts["mcts_v3_final_raw"]["eligible_sequential_decision_states"]
                + mcts["mcts_v3_round2_raw"]["eligible_sequential_decision_states"]
            ),
            "partial_archive_authorized_additional_battles": 0,
            "teacher_values_opened": 0,
            "training_targets_collected": 0,
            "main_blocker": "human inputlogs require a preregistered pinned-commit byte-exact dual-view replay audit; MCTS archives require public-replay alignment and opponent-hidden sanitization",
        },
        "global_rules": [
            "Never deduplicate or exclude across source archives by battle_tag alone.",
            "Never join the opposite POV to fill hidden opponent fields.",
            "Split by the largest known dependency cluster before teacher labels.",
            "Partial MCTS archive is a lineage duplicate of final and contributes no new examples.",
            "Evaluation/H2H/ladder artifacts and the sealed 93-battle split remain excluded.",
        ],
        "registry": {"path": rel(registry_path), "sha256": file_sha256(registry_path)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["admission_summary"], sort_keys=True))


if __name__ == "__main__":
    main()
