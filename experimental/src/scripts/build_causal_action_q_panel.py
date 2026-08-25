#!/usr/bin/env python3
"""Freeze information-sensitive, full-history-compatible counterfactual roots."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from belief.public_reveal_mask import (
    ReplayRevealFacts,
    from_replay_facts,
    information_fractions,
    replay_reveal_snapshots,
)
from belief.randbats_determinize import RandbatsDeterminizer
from scripts.collect_gen9_mcts_leaf_samples import determinize_leaf_state
from scripts.run_public_mcts_leaf_gate import PANEL_SCHEMA
from train.shallow_search_residual import battle_split


REPORT_SCHEMA = "metagross-causal-action-q-panel-report/v1"
R1_MAX_SEQUENCE_LENGTH = 128
PURPOSE_SPLITS = {
    "training": "train",
    "calibration": "calibration",
    "evaluation": "test",
}


def split_for_purpose(purpose: str) -> str:
    try:
        return PURPOSE_SPLITS[purpose]
    except KeyError as exc:
        raise ValueError(f"unsupported panel purpose: {purpose}") from exc


def battle_is_in_purpose(battle_id: str, purpose: str) -> bool:
    return battle_split(battle_id) == split_for_purpose(purpose)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower().removeprefix("battle-"))


def read_rows(path: Path):
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def terminal_lengths(path: Path, groups_by_battle: dict[str, set[tuple[str, str]]]) -> dict[tuple[str, str], int]:
    if path.name.endswith(".lz4"):
        import lz4.frame

        opened = lz4.frame.open(path, "rt", encoding="utf-8")
    else:
        opened = path.open(encoding="utf-8")
    matched: dict[tuple[str, str], int] = {}
    with opened as handle:
        for line in handle:
            row = json.loads(line)
            battle = norm(row.get("battle_tag"))
            candidates = groups_by_battle.get(battle, set())
            if not candidates:
                continue
            pov = norm(row.get("pov"))
            possible = [group for group in candidates if pov.endswith(group[1]) or group[1].endswith(pov)]
            if len(possible) == 1:
                matched[possible[0]] = len(row.get("text_tokens", []))
    return matched


def schema6_history_valid(row: dict[str, Any]) -> bool:
    decision = row.get("decision_idx")
    trajectory = row.get("trajectory")
    observation_rows = trajectory.get("observation_rows") if isinstance(trajectory, dict) else None
    length = min(decision + 1, R1_MAX_SEQUENCE_LENGTH) if isinstance(decision, int) else -1
    first_time = decision + 1 - length if isinstance(decision, int) else -1
    return bool(
        row.get("schema") == 6
        and isinstance(decision, int)
        and decision >= 0
        and row.get("mask_fallback") is False
        and isinstance(row.get("player_information_state"), dict)
        and isinstance(row.get("player_observation_history"), dict)
        and isinstance(trajectory, dict)
        and trajectory.get("mode") == "causal-history"
        and trajectory.get("observations") == length
        and trajectory.get("transitions") == length - 1
        and trajectory.get("inference_length") == length
        and isinstance(trajectory.get("action_receipts"), list)
        and len(trajectory["action_receipts"]) == length - 1
        and isinstance(trajectory.get("rl2"), list)
        and len(trajectory["rl2"]) == length
        and trajectory.get("time_indices") == list(range(first_time, decision + 1))
        and isinstance(observation_rows, dict)
        and all(
            isinstance(observation_rows.get(name), list)
            and len(observation_rows[name]) == length
            and observation_rows[name][-1] == row.get(name)
            for name in ("text_tokens", "numbers", "illegal_actions")
        )
    )


def schema6_public_facts(row: dict[str, Any]) -> ReplayRevealFacts:
    if not schema6_history_valid(row):
        raise ValueError("schema-6 snapshot has no valid causal public information")
    information = row["player_information_state"]
    opponent = information.get("opponent_public_team")
    if not isinstance(opponent, list):
        raise ValueError("schema-6 snapshot lacks an opponent public team")
    species: set[str] = set()
    moves: dict[str, set[str]] = collections.defaultdict(set)
    items: set[str] = set()
    abilities: set[str] = set()
    for record in opponent:
        pokemon = record.get("pokemon") if isinstance(record, dict) else None
        if not isinstance(pokemon, dict):
            raise ValueError("schema-6 public team record is malformed")
        name = norm(pokemon.get("name"))
        if not name:
            continue
        species.add(name)
        for move in pokemon.get("moves", []):
            move_name = norm(move.get("name") if isinstance(move, dict) else move)
            if move_name not in {"", "nomove", "none"}:
                moves[name].add(move_name)
        if norm(pokemon.get("item")) not in {"", "none", "unknownitem"}:
            items.add(name)
        if norm(pokemon.get("ability")) not in {"", "none", "unknownability"}:
            abilities.add(name)
    return ReplayRevealFacts(
        species=frozenset(species),
        moves=tuple((name, tuple(sorted(values))) for name, values in sorted(moves.items())),
        items=frozenset(items),
        abilities=frozenset(abilities),
    )


def as_paths(value: Path | list[Path]) -> list[Path]:
    return value if isinstance(value, list) else [value]


def snapshot_metrics(
    paths: Path | list[Path],
    allowed_groups_by_scope: dict[str, set[tuple[str, str]]] | None = None,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Read policy/history features only for battle identities admitted by the split.

    Snapshot files contain several physical battles.  Their identity fields must
    be decoded to route a row, but withheld rows are discarded before any policy,
    history, or public-information feature is inspected.
    """
    metrics = {}
    for path in as_paths(paths):
        source_hash = sha256(path)
        allowed_groups = (
            allowed_groups_by_scope.get(str(path.parent.resolve()), set())
            if allowed_groups_by_scope is not None
            else None
        )
        for line_number, row in read_rows(path):
            group = (norm(row.get("tag")), norm(row.get("username")))
            if allowed_groups is not None and group not in allowed_groups:
                continue
            decision = row.get("decision_idx")
            probabilities = row.get("probs")
            illegal = row.get("illegal_actions")
            if (
                not isinstance(decision, int)
                or not isinstance(probabilities, list)
                or len(probabilities) != 13
                or not isinstance(illegal, list)
                or len(illegal) != 13
            ):
                continue
            legal_values = [
                float(value)
                for value, blocked in zip(probabilities, illegal, strict=True)
                if not blocked
            ]
            total = math.fsum(legal_values)
            if not legal_values or not math.isfinite(total) or total <= 0:
                continue
            normalized = [max(0.0, value / total) for value in legal_values]
            entropy = (
                -math.fsum(
                    value * math.log(max(value, 1e-12)) for value in normalized
                ) / math.log(len(normalized))
                if len(normalized) > 1
                else 0.0
            )
            ordered = sorted(normalized, reverse=True)
            key = (norm(row.get("tag")), norm(row.get("username")), decision)
            if key in metrics:
                raise ValueError(f"duplicate schema-6 snapshot identity: {key}")
            metrics[key] = {
                "legal_actions": len(normalized),
                "policy_entropy": entropy,
                "policy_top_probability": ordered[0],
                "policy_top_gap": ordered[0] - ordered[1] if len(ordered) > 1 else 1.0,
                "schema": row.get("schema"),
                "schema6_history_valid": schema6_history_valid(row),
                "snapshot_source_path": str(path.resolve()),
                "snapshot_source_sha256": source_hash,
                "snapshot_source_line": line_number,
                "schema6_public_facts": (
                    schema6_public_facts(row) if schema6_history_valid(row) else None
                ),
            }
    return metrics


def canonical_decision(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("record_type") == "decision" and row.get("fixed_side") == "side_one":
        return row
    if row.get("schema") != "metagross-causal-dual-r1-root/v1":
        return None
    identity = row.get("identity")
    if not isinstance(identity, dict):
        return None
    return {
        "battle_tag": identity.get("battle_tag"),
        "username": identity.get("username"),
        "prior_decision_idx": identity.get("decision_idx"),
        "turn": identity.get("battle_turn", 0),
        "state": row.get("state"),
    }


def replay_index(paths: list[Path]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for decision_path in paths:
        for replay_path in sorted((decision_path.parent / "replays").glob("*.json")):
            payload = json.loads(replay_path.read_text(encoding="utf-8"))
            key = norm(payload.get("id") or replay_path.stem)
            if key in indexed:
                raise ValueError(f"duplicate replay identity: {key}")
            indexed[key] = payload
    return indexed


def excluded_battles(paths: list[Path]) -> set[str]:
    return {
        str(row["battle_id"])
        for path in paths
        for _, row in read_rows(path)
        if isinstance(row.get("battle_id"), str)
    }


def source_groups(paths: list[Path]) -> tuple[dict[str, set[tuple[str, str]]], dict[Path, str]]:
    groups_by_battle: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    hashes = {path: sha256(path) for path in paths}
    for path in paths:
        for _, row in read_rows(path):
            row = canonical_decision(row)
            if row is None:
                continue
            group = (norm(row.get("battle_tag")), norm(row.get("username")))
            if all(group):
                groups_by_battle[group[0]].add(group)
    return groups_by_battle, hashes


def canonical_observers(
    groups_by_battle: dict[Any, set[tuple[str, str]]], seed: int
) -> dict[Any, tuple[str, str]]:
    """Freeze one POV per physical battle without looking at actions/outcomes."""
    return {
        battle: min(groups, key=lambda group: stable_hash([seed, battle, group[1]]))
        for battle, groups in groups_by_battle.items()
        if groups
    }


def scoped_source_groups(
    paths: list[Path],
) -> tuple[
    dict[tuple[str, str], set[tuple[str, str]]],
    dict[str, str],
]:
    """Separate battle-id domains for independently restarted collections."""
    groups: dict[tuple[str, str], set[tuple[str, str]]] = collections.defaultdict(set)
    hashes_by_scope: dict[str, list[str]] = collections.defaultdict(list)
    for path in paths:
        scope = str(path.parent.resolve())
        hashes_by_scope[scope].append(sha256(path))
        for _, raw_row in read_rows(path):
            row = canonical_decision(raw_row)
            if row is None:
                continue
            battle = norm(row.get("battle_tag"))
            group = (battle, norm(row.get("username")))
            if all(group):
                groups[(scope, battle)].add(group)
    collection_hashes = {
        scope: stable_hash(sorted(hashes)) for scope, hashes in hashes_by_scope.items()
    }
    return groups, collection_hashes


def choose_candidates(
    args: argparse.Namespace, engine: Any
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    groups_by_battle, source_hashes = source_groups(args.decision_log)
    scoped_groups, collection_hashes = scoped_source_groups(args.decision_log)
    observers = canonical_observers(scoped_groups, args.seed)
    selected_split = split_for_purpose(args.purpose)
    source_split_inventory: collections.Counter[str] = collections.Counter()
    allowed_groups_by_scope: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    for (scope, battle), groups in scoped_groups.items():
        battle_id = stable_hash({
            "collection_sha256": collection_hashes[scope],
            "battle_tag": battle,
        })
        observed_split = battle_split(battle_id)
        source_split_inventory[observed_split] += 1
        if observed_split == selected_split:
            allowed_groups_by_scope[scope].update(groups)
    if args.history_authority == "terminal_archive":
        if args.terminal_trajectories is None:
            raise ValueError("terminal_archive history authority requires --terminal-trajectories")
        lengths = terminal_lengths(args.terminal_trajectories, groups_by_battle)
    else:
        lengths = {}
    snapshots = snapshot_metrics(args.prior_snapshot, allowed_groups_by_scope)
    replays = replay_index(args.decision_log) if args.history_authority == "terminal_archive" else {}
    excluded = excluded_battles(args.exclude_panel)
    facts_cache: dict[tuple[str, str], dict[int, Any]] = {}
    chosen: dict[tuple[str, str], list[tuple[float, str, dict[str, Any]]]] = {}
    failures: collections.Counter[str] = collections.Counter()
    for path in args.decision_log:
        source_hash = source_hashes[path]
        source_scope = str(path.parent.resolve())
        for line_number, row in read_rows(path):
            row = canonical_decision(row)
            if row is None:
                continue
            battle, username = norm(row.get("battle_tag")), norm(row.get("username"))
            group = (battle, username)
            battle_id = stable_hash({
                "collection_sha256": collection_hashes[source_scope],
                "battle_tag": battle,
            })
            # This identity-only gate must precede observer selection, history
            # joins, policy statistics, public-state parsing, and determinization.
            if battle_split(battle_id) != selected_split:
                failures["purpose_split_excluded"] += 1
                continue
            if (
                args.history_authority == "schema6_snapshot"
                and observers.get((source_scope, battle)) != group
            ):
                failures["noncanonical_observer"] += 1
                continue
            decision = row.get("prior_decision_idx")
            if not isinstance(decision, int) or decision < args.minimum_history:
                failures["history_too_short"] += 1
                continue
            metric = snapshots.get((battle, username, decision))
            if metric is None:
                failures["no_exact_snapshot"] += 1
                continue
            if args.history_authority == "terminal_archive":
                if decision >= lengths.get(group, 0):
                    failures["no_exact_terminal_trajectory"] += 1
                    continue
            elif not metric["schema6_history_valid"]:
                failures["invalid_schema6_causal_history"] += 1
                continue
            if int(metric["legal_actions"]) < args.minimum_legal_actions or float(metric["policy_entropy"]) < args.minimum_entropy:
                failures["uninformative_policy_root"] += 1
                continue
            state_text = row.get("state")
            if not isinstance(state_text, str):
                failures["missing_state"] += 1
                continue
            if battle_id in excluded:
                failures["excluded_battle"] += 1
                continue
            turn = int(row.get("turn", 0))
            if args.history_authority == "schema6_snapshot":
                facts = metric["schema6_public_facts"]
            else:
                replay = replays.get(battle)
                if replay is None:
                    failures["missing_replay"] += 1
                    continue
                cache_key = (battle, username)
                if cache_key not in facts_cache:
                    facts_cache[cache_key] = replay_reveal_snapshots(
                        replay.get("log", ""), str(row.get("username"))
                    )
                facts = facts_cache[cache_key].get(turn)
                if facts is None:
                    failures["missing_turn_snapshot"] += 1
                    continue
            try:
                public_state = engine.State.from_string(state_text)
                bits = from_replay_facts(public_state, facts)
                fractions = information_fractions(bits)
            except Exception:
                failures["invalid_public_state"] += 1
                continue
            if fractions[1] <= 0 or fractions[0] <= 0:
                failures["no_relevant_public_information"] += 1
                continue
            information_mix = math.fsum(4.0 * value * (1.0 - value) for value in fractions) / 4.0
            score = 0.55 * float(metric["policy_entropy"]) + 0.35 * information_mix + 0.10 * min(1.0, decision / 12.0)
            tie = stable_hash([args.seed, source_hash, line_number])
            candidate = {
                "source_path": str(path.resolve()),
                "source_file_sha256": source_hash,
                "source_line": line_number,
                "battle_tag": str(row.get("battle_tag")),
                "username": str(row.get("username")),
                "battle_id": battle_id,
                "state": state_text,
                "turn": turn,
                "decision_idx": decision,
                "facts": facts,
                "selection": {
                    "legal_actions": metric["legal_actions"],
                    "policy_entropy": metric["policy_entropy"],
                    "policy_top_probability": metric["policy_top_probability"],
                    "policy_top_gap": metric["policy_top_gap"],
                    "information_mix": information_mix,
                    "public_reveal_fractions": list(fractions),
                    "score": score,
                },
                "causal_history": {
                    "authority": args.history_authority,
                    "snapshot_schema": metric["schema"],
                    "snapshot_source_path": metric["snapshot_source_path"],
                    "snapshot_source_sha256": metric["snapshot_source_sha256"],
                    "snapshot_source_line": metric["snapshot_source_line"],
                },
            }
            options = chosen.setdefault(group, [])
            options.append((score, tie, candidate))
            options.sort(key=lambda value: (value[0], value[1]), reverse=True)
            del options[8:]
    candidates = []
    for options in chosen.values():
        for rank, (_, _, candidate) in enumerate(options):
            candidates.append({**candidate, "candidate_rank": rank})
    candidates.sort(
        key=lambda row: (
            stable_hash([args.seed, row["battle_id"]]),
            int(row["candidate_rank"]),
        )
    )
    if args.battles is not None and len(chosen) < args.battles:
        raise ValueError(f"only {len(chosen)} eligible battles for {args.battles}; failures={dict(failures)}")
    split_audit = {
        "selected_split": selected_split,
        "source_physical_battles_by_split": dict(source_split_inventory),
        "selected_source_physical_battles": source_split_inventory[selected_split],
        "withheld_history_policy_feature_rows_processed": 0,
        "gate_stage": "identity_before_history_policy_public_state_or_determinization",
    }
    return candidates, dict(failures), split_audit


def build(args: argparse.Namespace) -> dict[str, Any]:
    import poke_engine

    if (args.battles is not None and args.battles < 50) or args.schedules != 2 or args.worlds != 8:
        raise ValueError("panel contract requires >=50 battles, two schedules, and eight worlds")
    candidates, failures, split_audit = choose_candidates(args, poke_engine)
    determinizer = RandbatsDeterminizer(args.pool, seed=args.seed)
    rows = []
    used_battles: set[str] = set()
    for candidate in candidates:
        if args.battles is not None and len(rows) >= args.battles:
            break
        if candidate["battle_id"] in used_battles:
            continue
        public_state = poke_engine.State.from_string(candidate["state"])
        root_bits = from_replay_facts(public_state, candidate["facts"])
        public_state = public_state.with_side_one_public_reveals(root_bits)
        # The shared panel loader retains its historical 18-value shape check.
        # For this panel the values are the first 18 causal resource features:
        # 16 public battle/resource terms plus species and move reveal fractions.
        public_features = list(poke_engine.compute_resource_features(public_state)[:18])
        root_id = stable_hash({
            "battle_id": candidate["battle_id"],
            "source_line": candidate["source_line"],
            "public_state_sha256": hashlib.sha256(candidate["state"].encode()).hexdigest(),
        })
        schedules = []
        expected_fractions = None
        valid = True
        for schedule_id in range(args.schedules):
            schedule_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{root_id}:{schedule_id}".encode()).digest()[:8], "big")
            determinizer.reseed(schedule_seed)
            worlds = []
            for world_index in range(args.worlds):
                completed = determinize_leaf_state(
                    poke_engine.State.from_string(candidate["state"]),
                    poke_engine,
                    determinizer=determinizer,
                )
                if completed is None:
                    valid = False
                    break
                bits = from_replay_facts(completed, candidate["facts"])
                fractions = information_fractions(bits)
                if expected_fractions is None:
                    expected_fractions = fractions
                elif any(abs(left - right) > 1e-12 for left, right in zip(fractions, expected_fractions, strict=True)):
                    raise ValueError("public reveal fractions changed across hidden completions")
                completed = completed.with_side_one_public_reveals(bits)
                if list(poke_engine.compute_resource_features(completed)[:18]) != public_features:
                    raise ValueError("public feature contract changed under determinization")
                state_text = completed.to_string()
                worlds.append({
                    "world_index": world_index,
                    "weight": 1.0 / args.worlds,
                    "state_sha256": hashlib.sha256(state_text.encode()).hexdigest(),
                    "state": state_text,
                })
            if not valid:
                break
            schedules.append({"schedule_id": schedule_id, "seed": schedule_seed, "worlds": worlds})
        if not valid:
            failures["determinization_failed"] = failures.get("determinization_failed", 0) + 1
            continue
        rows.append({
            "schema": PANEL_SCHEMA,
            "battle_id": candidate["battle_id"],
            "root_id": root_id,
            "source_file_sha256": candidate["source_file_sha256"],
            "source_path": candidate["source_path"],
            "source_line": candidate["source_line"],
            "battle_tag": candidate["battle_tag"],
            "username": candidate["username"],
            "decision_idx": candidate["decision_idx"],
            "battle_turn": candidate["turn"],
            "public_state_sha256": hashlib.sha256(candidate["state"].encode()).hexdigest(),
            "public_features": public_features,
            "public_reveal_fractions": list(expected_fractions or (0.0, 0.0, 0.0, 0.0)),
            "selection": candidate["selection"],
            "causal_history": candidate["causal_history"],
            "schedules": schedules,
        })
        used_battles.add(candidate["battle_id"])
    if args.battles is None and len(rows) < 50:
        raise ValueError(f"only built {len(rows)} all-eligible roots; failures={failures}")
    if args.battles is not None and len(rows) != args.battles:
        raise ValueError(f"only built {len(rows)} of {args.battles}; failures={failures}")
    observed_splits = collections.Counter(battle_split(row["battle_id"]) for row in rows)
    expected_split = split_for_purpose(args.purpose)
    if observed_splits != {expected_split: len(rows)}:
        raise RuntimeError(
            f"panel crossed its purpose split: expected={expected_split} observed={dict(observed_splits)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "schema": REPORT_SCHEMA,
        "purpose": args.purpose,
        "split_contract": {
            "function": "train.shallow_search_residual.battle_split",
            "seed": 20260817,
            "buckets": {"training": "0-59", "calibration": "60-79", "evaluation": "80-99"},
            "selected_split": expected_split,
            "observed_panel_splits": dict(observed_splits),
            **split_audit,
        },
        "requested_battles": args.battles if args.battles is not None else "all_eligible",
        "feature_contract": "metagross-causal-resource-prefix-18/v1",
        "battles": len(rows),
        "schedules": len(rows) * args.schedules,
        "worlds": len(rows) * args.schedules * args.worlds,
        "selection": {
            "minimum_history": args.minimum_history,
            "minimum_legal_actions": args.minimum_legal_actions,
            "minimum_entropy": args.minimum_entropy,
            "mean_history": math.fsum(row["decision_idx"] + 1 for row in rows) / len(rows),
            "mean_policy_entropy": math.fsum(row["selection"]["policy_entropy"] for row in rows) / len(rows),
            "mean_reveal_fractions": [
                math.fsum(row["public_reveal_fractions"][index] for row in rows) / len(rows)
                for index in range(4)
            ],
        },
        "failures": failures,
        "seed": args.seed,
        "history_authority": args.history_authority,
        "panel_sha256": sha256(args.output),
        "randbats_pool_sha256": sha256(args.pool),
        "prior_snapshot_sha256": [sha256(path) for path in as_paths(args.prior_snapshot)],
        "terminal_trajectories_sha256": (
            sha256(args.terminal_trajectories)
            if args.terminal_trajectories is not None
            else None
        ),
        "excluded_panel_sha256": [sha256(path) for path in args.exclude_panel],
        "source_file_sha256": [sha256(path) for path in args.decision_log],
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-log", type=Path, action="append", required=True)
    parser.add_argument("--prior-snapshot", type=Path, action="append", required=True)
    parser.add_argument("--terminal-trajectories", type=Path)
    parser.add_argument(
        "--history-authority",
        choices=("terminal_archive", "schema6_snapshot"),
        default="terminal_archive",
    )
    parser.add_argument("--exclude-panel", type=Path, action="append", default=[])
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--purpose",
        choices=("training", "calibration", "evaluation"),
        required=True,
    )
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--battles", type=int)
    size.add_argument(
        "--all-eligible",
        action="store_true",
        help="Freeze one determinizable root for every eligible battle in the selected split.",
    )
    parser.add_argument("--schedules", type=int, default=2)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--minimum-history", type=int, default=3)
    parser.add_argument("--minimum-legal-actions", type=int, default=4)
    parser.add_argument("--minimum-entropy", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    print(json.dumps(build(args), sort_keys=True))


if __name__ == "__main__":
    main()
