#!/usr/bin/env python3
"""Build a fresh-world, battle-balanced reuse panel from the local smoke logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from experimental.src.scripts import teacher_root_bundle
from srcs.metagross import shadow_replay
from srcs.metagross.h2h_audit import _read_jsonl, _sha256
from srcs.metagross.world_provenance import derive_seed, state_sha256


def _canonical(value: object) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def _selection_score(seed: str, identity: dict[str, object]) -> str:
    return hashlib.sha256(
        _canonical({"selection_seed": seed, "identity": identity}).encode("ascii")
    ).hexdigest()


def _legacy_capture(
    *,
    identity: dict[str, object],
    player_priors: list,
    opponent_priors: list,
    schedules: list[list[tuple[str, float]]],
    generation_seeds: list[int],
    schedule_base_seed: int,
    manifest_sha256: str,
    sampling: dict[str, object],
) -> dict[str, object]:
    captured_schedules = []
    for schedule_id, (worlds, generation_seed) in enumerate(
        zip(schedules, generation_seeds, strict=True)
    ):
        captured_worlds = [
            {
                "world_index": index,
                "sample_weight": float(weight),
                "state_sha256": state_sha256(state),
                "sampled_state": state,
            }
            for index, (state, weight) in enumerate(worlds)
        ]
        captured_schedules.append(
            {
                "schedule_id": schedule_id,
                "sampling_seed": (
                    None
                    if schedule_id == 0
                    else teacher_root_bundle.derive_schedule_seed(
                        schedule_base_seed, identity, schedule_id
                    )
                ),
                "generation_seed": generation_seed,
                "world_count": len(captured_worlds),
                "world_weight_sum": math.fsum(
                    world["sample_weight"] for world in captured_worlds
                ),
                "worlds": captured_worlds,
            }
        )
    capture = {
        "schema_version": teacher_root_bundle.LEGACY_CAPTURE_SCHEMA_VERSION,
        "record_type": "teacher_root_capture",
        "identity": identity,
        "configuration": {
            "schedule_count": len(captured_schedules),
            "schedule_base_seed": schedule_base_seed,
            "input_manifest_sha256": manifest_sha256,
            "c_puct": 2.0,
        },
        "behavior_schedule_id": 0,
        "recorded_player_priors": player_priors,
        "recorded_opponent_priors": opponent_priors,
        "schedules": captured_schedules,
        "sampling": sampling,
    }
    capture["capture_sha256"] = hashlib.sha256(
        _canonical(capture).encode("ascii")
    ).hexdigest()
    teacher_root_bundle.validate_root_capture(capture)
    return capture


def build_panel(
    *,
    results_path: Path,
    log_dir: Path,
    prior_decisions_path: Path,
    execution_protocol_path: Path,
    smoke_audit_path: Path,
    panel_protocol_path: Path,
) -> tuple[list[dict], dict[str, object]]:
    panel_protocol = json.loads(panel_protocol_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    execution_protocol = json.loads(
        execution_protocol_path.read_text(encoding="utf-8")
    )
    smoke_audit = json.loads(smoke_audit_path.read_text(encoding="utf-8"))
    inputs = panel_protocol.get("inputs", {})
    if (
        panel_protocol.get("status") != "frozen_before_panel_generation"
        or panel_protocol.get("runner_sha256") != _sha256(Path(__file__).resolve())
        or inputs.get("results_sha256") != _sha256(results_path)
        or inputs.get("prior_decisions_sha256") != _sha256(prior_decisions_path)
        or inputs.get("execution_protocol_sha256") != _sha256(execution_protocol_path)
        or inputs.get("smoke_audit_sha256") != _sha256(smoke_audit_path)
        or smoke_audit.get("gate", {}).get("passed") is not True
        or panel_protocol.get("configuration")
        != {
            "roots_per_battle": 1,
            "schedule_count": 4,
            "schedule_world_counts": [16, 32, 16, 32],
            "schedule_base_seed": 2026081101,
            "selection_seed": "8" * 64,
            "selection_world_channel": "selection-worlds",
            "fresh_world_contract": "teacher_root_bundle.derive_schedule_seed",
            "c_puct": 2.0,
        }
    ):
        raise ValueError("reuse panel differs from its frozen protocol")
    root = Path(__file__).resolve().parents[2]
    dependencies = {
        "shadow_replay.py": root / "srcs" / "metagross" / "shadow_replay.py",
        "teacher_root_bundle.py": root / "experimental" / "src" / "scripts" / "teacher_root_bundle.py",
        "world_provenance.py": root / "srcs" / "metagross" / "world_provenance.py",
        "run_foul_play.py": root / "srcs" / "metagross" / "run_foul_play.py",
        "random_battles.py": root / "srcs" / "vendor" / "foul-play" / "fp" / "search" / "random_battles.py",
        "poke_engine_helpers.py": root / "srcs" / "vendor" / "foul-play" / "fp" / "search" / "poke_engine_helpers.py",
    }
    if any(
        panel_protocol.get("dependency_sha256", {}).get(name) != _sha256(path)
        for name, path in dependencies.items()
    ):
        raise ValueError("reuse panel dependency identity mismatch")
    process_files = inputs.get("process_files")
    if not isinstance(process_files, dict) or not process_files:
        raise ValueError("reuse panel protocol has no process-file manifest")
    for name, digest in process_files.items():
        path = log_dir / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"reuse panel process-file identity mismatch: {name}")

    tags = {game["battle_tag"] for game in results["games"]}
    if tags != set(panel_protocol.get("selection", {}).get("accepted_battles", [])):
        raise ValueError("reuse panel accepted battle set mismatch")
    prior_rows = {
        (row["namespace"], row["tag"], row["decision_idx"]): row
        for row in _read_jsonl(prior_decisions_path)
        if row.get("tag") in tags
    }
    candidates: dict[str, list[dict]] = {tag: [] for tag in tags}
    for search_path in sorted(
        log_dir / name
        for name in process_files
        if name.endswith(".search.jsonl")
    ):
        search_rows = [
            row
            for row in _read_jsonl(search_path)
            if (row.get("context") or {}).get("tag") in tags
        ]
        if not search_rows:
            continue
        operations = {
            (row.get("remote_search") or {}).get("operation") for row in search_rows
        }
        namespace = "agent_a" if operations == {"independent_ensemble"} else "agent_b"
        username = search_path.name.removesuffix(".search.jsonl")
        protocol_path = log_dir / f"{username}.protocol.jsonl"
        search_by_key = {
            shadow_replay._search_key(row): row for row in search_rows
        }
        reconstructed = shadow_replay.reconstruct_battles(
            _read_jsonl(protocol_path), search_by_key, username
        )
        for key, search in search_by_key.items():
            tag, decision_idx = key
            prior = prior_rows.get((namespace, tag, decision_idx))
            if prior is None or prior.get("mask_fallback") is not False:
                continue
            remote = search.get("remote_search") or {}
            world_count = remote.get("worlds")
            original_seed = derive_seed(
                execution_protocol["schedule"]["production_run_seed"],
                panel_protocol["configuration"]["selection_world_channel"],
                tag,
                decision_idx,
                0,
            )
            states, weights = shadow_replay._fresh_worlds(
                reconstructed[key], world_count, original_seed
            )
            if [state_sha256(state) for state in states] != remote.get("state_hashes"):
                raise ValueError(f"original selection cohort did not reconstruct: {key}")
            identity = {
                "battle_tag": tag,
                "username": username,
                "namespace": namespace,
                "decision_idx": decision_idx,
                "battle_turn": prior["battle_turn"],
            }
            candidates[tag].append(
                {
                    "identity": identity,
                    "score": _selection_score(
                        panel_protocol["configuration"]["selection_seed"], identity
                    ),
                    "battle": reconstructed[key],
                    "search": search,
                    "prior": prior,
                    "original_world_count": world_count,
                    "original_sampling_seed": original_seed,
                    "original_state_hashes": [state_sha256(state) for state in states],
                    "original_weights": weights,
                }
            )

    if any(not rows for rows in candidates.values()):
        raise ValueError("at least one completed battle has no eligible reuse root")
    manifest_sha256 = _sha256(panel_protocol_path)
    schedule_base_seed = panel_protocol["configuration"]["schedule_base_seed"]
    captures = []
    selections = []
    for tag in sorted(candidates):
        eligible = sorted(candidates[tag], key=lambda row: row["score"])
        selected = eligible[0]
        identity = selected["identity"]
        schedule_world_counts = panel_protocol["configuration"][
            "schedule_world_counts"
        ]
        schedules = []
        generation_seeds = []
        for schedule_id, world_count in enumerate(schedule_world_counts):
            generation_seed = teacher_root_bundle.derive_schedule_seed(
                schedule_base_seed, identity, schedule_id
            )
            states, weights = shadow_replay._fresh_worlds(
                selected["battle"], world_count, generation_seed
            )
            schedules.append(list(zip(states, weights, strict=True)))
            generation_seeds.append(generation_seed)
        capture = _legacy_capture(
            identity=identity,
            player_priors=selected["search"]["player_priors"],
            opponent_priors=selected["search"].get("opponent_priors") or [],
            schedules=schedules,
            generation_seeds=generation_seeds,
            schedule_base_seed=schedule_base_seed,
            manifest_sha256=manifest_sha256,
            sampling={
                "source": "adaptive_local_smoke_reuse_panel_v1",
                "stratum": {"battle_tag": tag},
                "population_count": len(eligible),
                "selected_count": 1,
                "inclusion_probability": 1.0 / len(eligible),
                "poststratification_weight": 1.0,
                "source_smoke_audit_sha256": _sha256(smoke_audit_path),
                "selection_score": selected["score"],
                "original_sampling_seed": selected["original_sampling_seed"],
                "original_state_hashes": selected["original_state_hashes"],
            },
        )
        captures.append(capture)
        selections.append(
            {
                "battle_tag": tag,
                "eligible_roots": len(eligible),
                "selected_identity": identity,
                "selection_score": selected["score"],
                "world_counts": schedule_world_counts,
                "fresh_schedule_hashes": [
                    hashlib.sha256(
                        "\n".join(
                            world["state_sha256"] for world in schedule["worlds"]
                        ).encode("ascii")
                    ).hexdigest()
                    for schedule in capture["schedules"]
                ],
            }
        )
    summary = {
        "schema_version": 1,
        "mode": "adaptive_ensemble_battle_balanced_reuse_panel",
        "protocol": {
            "path": str(panel_protocol_path),
            "sha256": manifest_sha256,
        },
        "counts": {
            "battles": len(tags),
            "selected_roots": len(captures),
            "schedules": sum(len(capture["schedules"]) for capture in captures),
            "worlds": sum(
                len(schedule["worlds"])
                for capture in captures
                for schedule in capture["schedules"]
            ),
        },
        "selections": selections,
        "limitations": [
            "These battles were already used for execution-smoke evidence.",
            "Fresh schedules are held out from prior teacher treatments, not from architecture development.",
            "Four roots from two mirrored matchup blocks cannot establish strength.",
        ],
        "teacher_replay_authorized": len(captures) == len(tags) == 4,
        "new_games_authorized": False,
        "public_ladder_authorized": False,
    }
    return captures, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--prior-decisions", type=Path, required=True)
    parser.add_argument("--execution-protocol", type=Path, required=True)
    parser.add_argument("--smoke-audit", type=Path, required=True)
    parser.add_argument("--panel-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    if output.exists() or summary_path.exists():
        raise FileExistsError(output if output.exists() else summary_path)
    captures, summary = build_panel(
        results_path=args.results.expanduser().resolve(),
        log_dir=args.log_dir.expanduser().resolve(),
        prior_decisions_path=args.prior_decisions.expanduser().resolve(),
        execution_protocol_path=args.execution_protocol.expanduser().resolve(),
        smoke_audit_path=args.smoke_audit.expanduser().resolve(),
        panel_protocol_path=args.panel_protocol.expanduser().resolve(),
    )
    output.write_text(
        "".join(_canonical(capture) + "\n" for capture in captures),
        encoding="ascii",
    )
    summary["output"] = {"path": str(output), "sha256": _sha256(output)}
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["counts"], sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
