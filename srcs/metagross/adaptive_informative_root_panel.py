#!/usr/bin/env python3
"""Build a battle-balanced panel screened only by baseline search ambiguity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from experimental.src.scripts import teacher_root_bundle
from srcs.metagross import shadow_replay
from srcs.metagross.adaptive_ensemble_reuse_panel import _legacy_capture
from srcs.metagross.h2h_audit import _read_jsonl, _sha256
from srcs.metagross.world_provenance import derive_seed, state_sha256


def _canonical(value: object) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def _selection_score(seed: str, identity: dict[str, object]) -> str:
    return hashlib.sha256(
        _canonical(["baseline-multiaction-panel-v1", seed, identity]).encode("ascii")
    ).hexdigest()


def baseline_distribution(search: dict) -> dict[str, float]:
    remote = search.get("remote_search") or {}
    worlds = remote.get("worlds")
    samples = search.get("samples")
    if (
        isinstance(worlds, bool)
        or not isinstance(worlds, int)
        or worlds <= 0
        or not isinstance(samples, list)
        or len(samples) < worlds
    ):
        raise ValueError("baseline screening samples are incomplete")
    masses: dict[str, float] = {}
    for expected_index, sample in enumerate(samples[:worlds]):
        if sample.get("index") != expected_index:
            raise ValueError("baseline screening sample order is invalid")
        weight = float(sample.get("sample_chance", math.nan))
        result = sample.get("result") or {}
        total_visits = result.get("total_visits")
        options = result.get("side_one")
        if (
            not math.isfinite(weight)
            or weight < 0
            or isinstance(total_visits, bool)
            or not isinstance(total_visits, int)
            or total_visits <= 0
            or not isinstance(options, list)
            or not options
        ):
            raise ValueError("baseline screening sample is invalid")
        for option in options:
            action = option.get("move_choice")
            visits = option.get("visits")
            if (
                not isinstance(action, str)
                or not action
                or isinstance(visits, bool)
                or not isinstance(visits, int)
                or visits < 0
            ):
                raise ValueError("baseline screening action is invalid")
            masses[action] = masses.get(action, 0.0) + weight * visits / total_visits
    if not masses:
        raise ValueError("baseline screening distribution is empty")
    threshold = max(masses.values()) * 0.75
    retained = {action: mass for action, mass in masses.items() if mass >= threshold}
    total = math.fsum(retained.values())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("baseline retained distribution has no positive mass")
    return {action: retained[action] / total for action in sorted(retained)}


def support_metrics(states: list[str], weights: list[float]) -> dict[str, float | int]:
    if len(states) != len(weights) or not states:
        raise ValueError("fresh schedule state/weight shape mismatch")
    grouped: dict[str, float] = {}
    for state, weight in zip(states, weights, strict=True):
        value = float(weight)
        if not math.isfinite(value) or value < 0:
            raise ValueError("fresh schedule has an invalid weight")
        digest = state_sha256(state)
        grouped[digest] = grouped.get(digest, 0.0) + value
    total = math.fsum(grouped.values())
    if total <= 0:
        raise ValueError("fresh schedule has no positive weight")
    normalized = [value / total for value in grouped.values()]
    return {
        "nominal_worlds": len(states),
        "unique_worlds": len(grouped),
        "unique_fraction": len(grouped) / len(states),
        "state_ess": 1.0 / math.fsum(value * value for value in normalized),
    }


def write_source_manifest(log_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for search_path in sorted(log_dir.glob("*.search.jsonl")):
        operations = {
            (row.get("remote_search") or {}).get("operation")
            for row in _read_jsonl(search_path)
        }
        if operations != {"independent_ensemble"}:
            continue
        protocol_path = search_path.with_name(
            search_path.name.removesuffix(".search.jsonl") + ".protocol.jsonl"
        )
        if not protocol_path.is_file():
            raise FileNotFoundError(protocol_path)
        rows.append(
            {
                "username": search_path.name.removesuffix(".search.jsonl"),
                "search": {"name": search_path.name, "sha256": _sha256(search_path)},
                "protocol": {
                    "name": protocol_path.name,
                    "sha256": _sha256(protocol_path),
                },
            }
        )
    if not rows:
        raise ValueError("no independent-ensemble process files found")
    output.write_text(
        json.dumps({"schema_version": 1, "processes": rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _validate_source_manifest(log_dir: Path, manifest: dict) -> list[dict]:
    rows = manifest.get("processes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source manifest has no processes")
    for row in rows:
        for field in ("search", "protocol"):
            entry = row.get(field) or {}
            path = log_dir / entry.get("name", "")
            if not path.is_file() or _sha256(path) != entry.get("sha256"):
                raise ValueError(f"source process identity mismatch: {path.name}")
    return rows


def screen(
    *,
    results_path: Path,
    log_dir: Path,
    prior_path: Path,
    execution_protocol_path: Path,
    source_manifest_path: Path,
    screen_protocol_path: Path,
) -> dict[str, object]:
    protocol = json.loads(screen_protocol_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    configuration = protocol.get("configuration")
    if (
        protocol.get("status") != "frozen_before_screening"
        or protocol.get("runner_sha256") != _sha256(Path(__file__).resolve())
        or protocol.get("inputs")
        != {
            "results_sha256": _sha256(results_path),
            "prior_sha256": _sha256(prior_path),
            "execution_protocol_sha256": _sha256(execution_protocol_path),
            "source_manifest_sha256": _sha256(source_manifest_path),
        }
        or configuration
        != {
            "namespace": "agent_a",
            "selection_seed": "9" * 64,
            "selection_world_channel": "selection-worlds",
            "minimum_retained_actions": 2,
            "one_root_per_battle": True,
        }
    ):
        raise ValueError("informative-root screening differs from its frozen protocol")
    root = Path(__file__).resolve().parents[2]
    dependencies = {
        "shadow_replay.py": root / "srcs" / "metagross" / "shadow_replay.py",
        "world_provenance.py": root / "srcs" / "metagross" / "world_provenance.py",
    }
    if any(
        protocol.get("dependency_sha256", {}).get(name) != _sha256(path)
        for name, path in dependencies.items()
    ):
        raise ValueError("informative-root screening dependency mismatch")
    processes = _validate_source_manifest(log_dir, source_manifest)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    tags = {game["battle_tag"] for game in results["games"]}
    prior_rows = {
        (row["namespace"], row["tag"], row["decision_idx"]): row
        for row in _read_jsonl(prior_path)
        if row.get("namespace") == configuration["namespace"]
    }
    execution = json.loads(execution_protocol_path.read_text(encoding="utf-8"))
    run_seed = execution["schedule"]["production_run_seed"]
    eligible: dict[str, list[dict]] = {tag: [] for tag in tags}
    flow = {"search_rows": 0, "prior_joined": 0, "cohort_reconstructed": 0, "multi_action": 0}
    for process in processes:
        username = process["username"]
        search_path = log_dir / process["search"]["name"]
        protocol_path = log_dir / process["protocol"]["name"]
        search_rows = _read_jsonl(search_path)
        search_by_key = {shadow_replay._search_key(row): row for row in search_rows}
        reconstructed = shadow_replay.reconstruct_battles(
            _read_jsonl(protocol_path), search_by_key, username
        )
        for key, search_row in search_by_key.items():
            flow["search_rows"] += 1
            tag, decision_idx = key
            context = search_row.get("context") or {}
            prior = prior_rows.get((configuration["namespace"], tag, decision_idx))
            if (
                tag not in tags
                or prior is None
                or prior.get("mask_fallback") is not False
                or prior.get("username") != username
                or prior.get("rqid") != context.get("rqid")
                or prior.get("battle_turn") != context.get("battle_turn")
            ):
                continue
            flow["prior_joined"] += 1
            remote = search_row.get("remote_search") or {}
            world_count = remote.get("worlds")
            seed = derive_seed(
                run_seed,
                configuration["selection_world_channel"],
                tag,
                decision_idx,
                0,
            )
            states, weights = shadow_replay._fresh_worlds(
                reconstructed[key], world_count, seed
            )
            hashes = [state_sha256(state) for state in states]
            if hashes != remote.get("state_hashes"):
                continue
            flow["cohort_reconstructed"] += 1
            distribution = baseline_distribution(search_row)
            if len(distribution) < configuration["minimum_retained_actions"]:
                continue
            flow["multi_action"] += 1
            identity = {
                "battle_tag": tag,
                "username": username,
                "namespace": configuration["namespace"],
                "decision_idx": decision_idx,
                "battle_turn": prior["battle_turn"],
                "rqid": prior["rqid"],
            }
            eligible[tag].append(
                {
                    "identity": identity,
                    "selection_score": _selection_score(
                        configuration["selection_seed"], identity
                    ),
                    "source": {
                        "search_file": search_path.name,
                        "protocol_file": protocol_path.name,
                    },
                    "baseline_distribution": distribution,
                    "screening_world_count": world_count,
                    "screening_seed": seed,
                    "screening_state_hashes": hashes,
                    "screening_weights": weights,
                    "recorded_player_priors": search_row["player_priors"],
                    "recorded_opponent_priors": search_row.get("opponent_priors") or [],
                }
            )
    if any(not rows for rows in eligible.values()):
        missing = sorted(tag for tag, rows in eligible.items() if not rows)
        raise ValueError(f"completed battles have no informative roots: {missing}")
    return {
        "schema_version": 1,
        "mode": "baseline_multiaction_root_screen",
        "protocol": {"path": str(screen_protocol_path), "sha256": _sha256(screen_protocol_path)},
        "source_manifest_sha256": _sha256(source_manifest_path),
        "flow": flow,
        "counts": {
            "battles": len(tags),
            "eligible_roots": sum(len(rows) for rows in eligible.values()),
        },
        "battles": [
            {
                "battle_tag": tag,
                "eligible": sorted(
                    eligible[tag], key=lambda row: (row["selection_score"], _canonical(row["identity"]))
                ),
            }
            for tag in sorted(tags)
        ],
        "prohibited_inputs_used": [],
        "treatment_execution_authorized": False,
        "new_games_authorized": False,
        "public_ladder_authorized": False,
    }


def materialize(
    *,
    log_dir: Path,
    source_manifest_path: Path,
    screen_path: Path,
    panel_protocol_path: Path,
) -> tuple[list[dict], dict[str, object]]:
    protocol = json.loads(panel_protocol_path.read_text(encoding="utf-8"))
    screen_report = json.loads(screen_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    configuration = protocol.get("configuration")
    if (
        protocol.get("status") != "frozen_before_panel_generation"
        or protocol.get("runner_sha256") != _sha256(Path(__file__).resolve())
        or protocol.get("inputs")
        != {
            "source_manifest_sha256": _sha256(source_manifest_path),
            "screen_sha256": _sha256(screen_path),
        }
        or configuration
        != {
            "schedule_world_counts": [16, 32, 16, 32],
            "schedule_base_seed": 2026081301,
            "minimum_unique_fraction": 0.75,
            "minimum_ess_fraction": 0.60,
            "require_disjoint_screen_and_schedules": True,
            "require_disjoint_schedules": True,
            "c_puct": 2.0,
        }
    ):
        raise ValueError("informative-root panel differs from its frozen protocol")
    processes = {
        row["username"]: row
        for row in _validate_source_manifest(log_dir, source_manifest)
    }
    reconstructed_cache: dict[str, dict] = {}
    captures = []
    selections = []
    rejected_support = 0
    manifest_sha256 = _sha256(panel_protocol_path)
    for battle_row in screen_report["battles"]:
        selected = None
        for candidate in battle_row["eligible"]:
            username = candidate["identity"]["username"]
            process = processes[username]
            if username not in reconstructed_cache:
                search_rows = _read_jsonl(log_dir / process["search"]["name"])
                search_by_key = {
                    shadow_replay._search_key(row): row for row in search_rows
                }
                reconstructed_cache[username] = shadow_replay.reconstruct_battles(
                    _read_jsonl(log_dir / process["protocol"]["name"]),
                    search_by_key,
                    username,
                )
            identity = candidate["identity"]
            key = (identity["battle_tag"], identity["decision_idx"])
            battle = reconstructed_cache[username][key]
            schedules = []
            generation_seeds = []
            metrics = []
            used_hashes = set(candidate["screening_state_hashes"])
            valid = True
            for schedule_id, world_count in enumerate(
                configuration["schedule_world_counts"]
            ):
                seed = teacher_root_bundle.derive_schedule_seed(
                    configuration["schedule_base_seed"], identity, schedule_id
                )
                states, weights = shadow_replay._fresh_worlds(battle, world_count, seed)
                hashes = {state_sha256(state) for state in states}
                support = support_metrics(states, weights)
                if (
                    hashes & used_hashes
                    or support["unique_fraction"] < configuration["minimum_unique_fraction"]
                    or support["state_ess"]
                    < configuration["minimum_ess_fraction"] * world_count
                ):
                    valid = False
                    break
                used_hashes.update(hashes)
                schedules.append(list(zip(states, weights, strict=True)))
                generation_seeds.append(seed)
                metrics.append(support)
            if not valid:
                rejected_support += 1
                continue
            selected = (candidate, schedules, generation_seeds, metrics)
            break
        if selected is None:
            raise ValueError(
                f"battle has no informative root with adequate fresh support: {battle_row['battle_tag']}"
            )
        candidate, schedules, generation_seeds, metrics = selected
        identity = candidate["identity"]
        capture = _legacy_capture(
            identity=identity,
            player_priors=candidate["recorded_player_priors"],
            opponent_priors=candidate["recorded_opponent_priors"],
            schedules=schedules,
            generation_seeds=generation_seeds,
            schedule_base_seed=configuration["schedule_base_seed"],
            manifest_sha256=manifest_sha256,
            sampling={
                "source": "n100_v7_baseline_multiaction_screen_v1",
                "stratum": {"battle_tag": identity["battle_tag"]},
                "population_count": len(battle_row["eligible"]),
                "selected_count": 1,
                "inclusion_probability": 1.0 / len(battle_row["eligible"]),
                "poststratification_weight": 1.0,
                "selection_score": candidate["selection_score"],
                "screening_state_hashes": candidate["screening_state_hashes"],
                "baseline_distribution": candidate["baseline_distribution"],
                "support_metrics": metrics,
            },
        )
        captures.append(capture)
        selections.append(
            {
                "battle_tag": identity["battle_tag"],
                "eligible_roots": len(battle_row["eligible"]),
                "selected_identity": identity,
                "selection_score": candidate["selection_score"],
                "schedule_support": metrics,
            }
        )
    summary = {
        "schema_version": 1,
        "mode": "baseline_multiaction_battle_balanced_panel",
        "protocol": {"path": str(panel_protocol_path), "sha256": manifest_sha256},
        "inputs": {"screen_sha256": _sha256(screen_path)},
        "counts": {
            "battles": len(captures),
            "selected_roots": len(captures),
            "schedules": sum(len(row["schedules"]) for row in captures),
            "worlds": sum(
                len(schedule["worlds"])
                for row in captures
                for schedule in row["schedules"]
            ),
            "support_rejections_before_selection": rejected_support,
        },
        "selections": selections,
        "teacher_replay_authorized": len(captures) == 100,
        "new_games_authorized": False,
        "public_ladder_authorized": False,
    }
    return captures, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--log-dir", type=Path, required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)
    screen_parser = subparsers.add_parser("screen")
    screen_parser.add_argument("--results", type=Path, required=True)
    screen_parser.add_argument("--log-dir", type=Path, required=True)
    screen_parser.add_argument("--prior", type=Path, required=True)
    screen_parser.add_argument("--execution-protocol", type=Path, required=True)
    screen_parser.add_argument("--source-manifest", type=Path, required=True)
    screen_parser.add_argument("--screen-protocol", type=Path, required=True)
    screen_parser.add_argument("--output", type=Path, required=True)
    panel_parser = subparsers.add_parser("materialize")
    panel_parser.add_argument("--log-dir", type=Path, required=True)
    panel_parser.add_argument("--source-manifest", type=Path, required=True)
    panel_parser.add_argument("--screen", type=Path, required=True)
    panel_parser.add_argument("--panel-protocol", type=Path, required=True)
    panel_parser.add_argument("--output", type=Path, required=True)
    panel_parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("output", "summary"):
        path = getattr(args, name, None)
        if path is not None and path.exists():
            raise FileExistsError(path)
    if args.command == "manifest":
        write_source_manifest(args.log_dir, args.output)
    elif args.command == "screen":
        report = screen(
            results_path=args.results,
            log_dir=args.log_dir,
            prior_path=args.prior,
            execution_protocol_path=args.execution_protocol,
            source_manifest_path=args.source_manifest,
            screen_protocol_path=args.screen_protocol,
        )
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report["counts"], sort_keys=True))
    else:
        captures, summary = materialize(
            log_dir=args.log_dir,
            source_manifest_path=args.source_manifest,
            screen_path=args.screen,
            panel_protocol_path=args.panel_protocol,
        )
        args.output.write_text(
            "".join(_canonical(capture) + "\n" for capture in captures),
            encoding="ascii",
        )
        summary["output"] = {"path": str(args.output), "sha256": _sha256(args.output)}
        args.summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
