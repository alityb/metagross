#!/usr/bin/env python3
"""Audit an adaptive independent-ensemble local execution smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from srcs.metagross import canary_audit, shadow_replay
from srcs.metagross.h2h_audit import _read_jsonl, _sha256
from srcs.metagross.mcts_contract import MAX_WIRE_BATCH_SIZE, validate_result_payload
from srcs.metagross.world_provenance import deterministic_request_id


def audit(
    results_path: Path,
    log_dir: Path,
    prior_decisions_path: Path | list[Path],
    protocol_path: Path,
    pairs_path: Path,
    audit_protocol_path: Path,
) -> dict[str, object]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    summary = results["summary"]
    tags = {game["battle_tag"] for game in results["games"]}
    remote_identity = protocol["remote_worker"]
    source_identity = protocol["source_identity"]
    root = Path(__file__).resolve().parents[2]
    failures = []
    pair_manifest = json.loads(pairs_path.read_text(encoding="utf-8"))
    audit_protocol = json.loads(audit_protocol_path.read_text(encoding="utf-8"))
    prior_paths = (
        prior_decisions_path
        if isinstance(prior_decisions_path, list)
        else [prior_decisions_path]
    )
    expected_prior_hashes = audit_protocol.get("inputs", {}).get(
        "prior_decisions_sha256s"
    )
    if expected_prior_hashes is None and len(prior_paths) == 1:
        expected_prior_hashes = [
            audit_protocol.get("inputs", {}).get("prior_decisions_sha256")
        ]
    if (
        audit_protocol.get("status") != "frozen_before_audit"
        or audit_protocol.get("audit_source_sha256")
        != _sha256(Path(__file__).resolve())
        or audit_protocol.get("inputs", {}).get("results_sha256")
        != _sha256(results_path)
        or audit_protocol.get("inputs", {}).get("pairs_sha256")
        != _sha256(pairs_path)
        or expected_prior_hashes != [_sha256(path) for path in prior_paths]
        or audit_protocol.get("inputs", {}).get("execution_protocol_sha256")
        != _sha256(protocol_path)
    ):
        failures.append("audit differs from its frozen protocol")
    expected_engine = audit_protocol.get("expected_engine_identity")
    if pair_manifest.get("showdown_commit") != audit_protocol.get("inputs", {}).get(
        "showdown_commit"
    ):
        failures.append("Showdown commit mismatch")

    expected_sources = {
        "eval_run.py": root / "experimental" / "src" / "eval" / "run.py",
        "run_foul_play.py": root / "srcs" / "metagross" / "run_foul_play.py",
        "modal_mcts.py": root / "srcs" / "metagross" / "modal_mcts.py",
    }
    for name, path in expected_sources.items():
        if source_identity.get(name) != _sha256(path):
            failures.append(f"source identity mismatch: {name}")
    known_artifacts = {
        "offline_protocol_v2": root / "experimental" / "runs" / "search_native_stage2_20260809" / "adaptive-independent-ensemble-protocol-v2.json",
        "offline_result_v2": root / "experimental" / "runs" / "search_native_stage2_20260809" / "adaptive-independent-ensemble-v2.json",
        "latency_protocol_v2": root / "experimental" / "runs" / "search_native_stage2_20260809" / "adaptive-independent-ensemble-latency-protocol-v2.json",
        "latency_result_v2": root / "experimental" / "runs" / "search_native_stage2_20260809" / "adaptive-independent-ensemble-latency-v2.json",
        "policy_checkpoint": root / "srcs" / "models" / "randbats_exit_r1" / "ckpts" / "policy_weights" / "policy_epoch_5.pt",
        "team_generator": root / "experimental" / "src" / "scripts" / "generate_mirrored_randbats_pair.cjs",
        "execution_smoke_audit": root / "experimental" / "runs" / "search_native_stage2_20260809" / "adaptive-independent-ensemble-local-smoke-audit-v5.json",
        "reuse_replay_audit": root / "experimental" / "runs" / "search_native_stage2_20260809" / "adaptive-independent-ensemble-reuse-audit-v1.json",
    }
    for name, digest in source_identity.items():
        path = known_artifacts.get(name)
        if path is not None and digest != _sha256(path):
            failures.append(f"artifact identity mismatch: {name}")

    pairs_by_id = {pair["pair_id"]: pair for pair in pair_manifest["pairs"]}
    games_by_pair: dict[str, list[dict]] = {}
    for game in results["games"]:
        games_by_pair.setdefault(game["pair_id"], []).append(game)
    for pair_id, games in games_by_pair.items():
        pair = pairs_by_id.get(pair_id)
        if pair is None or len(games) != 2:
            failures.append(f"invalid mirrored pair membership: {pair_id}")
            continue
        games.sort(key=lambda game: game["pair_leg"])
        first, second = games
        packed_hashes = {
            index: hashlib.sha256(pair[f"team_{index}_packed"].encode()).hexdigest()
            for index in (1, 2)
        }
        if (
            {first["pair_leg"], second["pair_leg"]} != {1, 2}
            or not first["battle_seed"] == second["battle_seed"] == pair["battle_seed"]
            or packed_hashes[1] != pair["team_1_sha256"]
            or packed_hashes[2] != pair["team_2_sha256"]
            or first["team_1_sha256"] != second["team_1_sha256"]
            or first["team_2_sha256"] != second["team_2_sha256"]
            or first["team_1_sha256"] != pair["team_1_sha256"]
            or first["team_2_sha256"] != pair["team_2_sha256"]
            or first["agent_a_team_sha256"] != second["agent_b_team_sha256"]
            or first["agent_b_team_sha256"] != second["agent_a_team_sha256"]
        ):
            failures.append(f"mirrored pair integrity mismatch: {pair_id}")

    prior_by_mode = {
        "candidate": {},
        "baseline": {},
    }
    for prior_path in prior_paths:
        for row in _read_jsonl(prior_path):
            if row.get("tag") not in tags:
                continue
            mode = {"agent_a": "candidate", "agent_b": "baseline"}.get(
                row.get("namespace")
            )
            if mode is None:
                failures.append("unexpected prior namespace")
                continue
            key = shadow_replay._decision_key(row)
            if key in prior_by_mode[mode]:
                failures.append(f"duplicate {mode} prior key: {key}")
            prior_by_mode[mode][key] = row

    process_rows = []
    search_keys = {"candidate": set(), "baseline": set()}
    adaptive_shapes: dict[tuple[int, int, int], int] = {}
    engine_mismatches = 0
    canonical_engine = None
    malformed_samples = 0
    outbound_matches = 0
    all_request_ids = set()
    request_id_count = 0

    for search_path in sorted(log_dir.glob("*.search.jsonl")):
        rows = [
            row
            for row in _read_jsonl(search_path)
            if (row.get("context") or {}).get("tag") in tags
        ]
        if not rows:
            continue
        operations = {
            (row.get("remote_search") or {}).get("operation") for row in rows
        }
        if operations == {"independent_ensemble"}:
            mode = "candidate"
        elif operations == {None}:
            mode = "baseline"
        else:
            failures.append(f"{search_path.name}: mixed or invalid operations")
            continue
        by_key = {}
        for row in rows:
            key = shadow_replay._search_key(row)
            if key in by_key:
                failures.append(f"{search_path.name}: duplicate search key {key}")
            by_key[key] = row
            if not row.get("player_priors"):
                failures.append(f"{search_path.name}: decision lacks player priors")
            prior = prior_by_mode[mode].get(key)
            if prior is None:
                failures.append(f"{search_path.name}/{key}: missing prior row")
            else:
                expected_priors = sorted(
                    [
                        action,
                        0.0
                        if prior["illegal_actions"][index]
                        else float(prior["probs"][index]),
                    ]
                    for action, index in prior["name_table"].items()
                )
                actual_priors = row.get("player_priors") or []
                if (
                    len(actual_priors) != len(expected_priors)
                    or dict(actual_priors) != dict(expected_priors)
                    or prior.get("mask_fallback") is not False
                ):
                    failures.append(f"{search_path.name}/{key}: prior value mismatch")
            for sample in row.get("samples") or ():
                try:
                    validate_result_payload(sample.get("result"))
                except ValueError:
                    malformed_samples += 1
            remote = row["remote_search"]
            worlds = remote.get("worlds")
            request_ids = remote.get("request_ids") or []
            state_hashes = remote.get("state_hashes") or []
            timings = remote.get("timings") or []
            samples = row.get("samples") or []
            if mode == "candidate":
                searches = remote.get("searches")
                request_channel = "selection-ensemble-request"
            else:
                searches = worlds
                request_channel = "selection-search-request"
            context = row["context"]
            expected_ids = [
                deterministic_request_id(
                    protocol["schedule"]["production_run_seed"],
                    context["tag"],
                    context["decision_idx"],
                    index,
                    channel=request_channel,
                )
                for index in range(searches)
            ]
            sample_weights = [float(sample["sample_chance"]) for sample in samples]
            if (
                request_ids != expected_ids
                or any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in state_hashes)
                or len(state_hashes) != worlds
                or len(samples) != searches
                or [sample.get("index") for sample in samples] != list(range(searches))
                or any(not math.isfinite(weight) or weight < 0 for weight in sample_weights)
                or not math.isclose(math.fsum(sample_weights), 1.0, abs_tol=1e-12)
                or any(
                    not isinstance(timing.get("batch_size"), int)
                    or timing["batch_size"] != 16
                    or not math.isfinite(float(timing.get("search_ms", math.nan)))
                    or float(timing["search_ms"]) < 0
                    for timing in timings
                )
                or len(timings) != searches
            ):
                failures.append(f"{search_path.name}/{key}: request evidence mismatch")
            request_id_count += len(request_ids)
            all_request_ids.update(request_ids)
            engine = remote.get("engine") or {}
            if engine != expected_engine:
                engine_mismatches += 1
            if canonical_engine is None:
                canonical_engine = engine
            elif engine != canonical_engine:
                engine_mismatches += 1
            if mode != "candidate":
                continue
            effective = remote.get("effective_repeat_count")
            expected = min(3, MAX_WIRE_BATCH_SIZE // worlds) if worlds else None
            if (
                effective != expected
                or remote.get("repeat_count") != effective
                or remote.get("maximum_repeat_count") != 3
                or remote.get("wire_batch_limit") != MAX_WIRE_BATCH_SIZE
                or searches != worlds * effective
                or searches > MAX_WIRE_BATCH_SIZE
                or len(request_ids) != searches
                or len(set(request_ids)) != searches
                or len(state_hashes) != worlds
                or len(timings) != searches
            ):
                failures.append(f"{search_path.name}/{key}: adaptive contract mismatch")
            else:
                shape = (worlds, effective, searches)
                adaptive_shapes[shape] = adaptive_shapes.get(shape, 0) + 1
            engine = remote.get("engine") or {}
            if (
                engine.get("contract") != remote_identity["contract"]
                or engine.get("source_sha256") != remote_identity["source_sha256"]
                or engine.get("native_sha256") != remote_identity["native_sha256"]
                or engine.get("resources", {}).get("max_world_concurrency")
                != MAX_WIRE_BATCH_SIZE
            ):
                engine_mismatches += 1

        search_keys[mode].update(by_key)
        username = search_path.name.removesuffix(".search.jsonl")
        protocol_dump = log_dir / f"{username}.protocol.jsonl"
        if not protocol_dump.is_file():
            failures.append(f"{username}: missing protocol dump")
            continue
        protocol_rows = _read_jsonl(protocol_dump)
        player_sides = {
            match.group(1)
            for row in protocol_rows
            if row.get("direction") == "received"
            for match in re.finditer(
                rf"\|player\|(p[12])\|{re.escape(username)}\|",
                row.get("message", ""),
            )
        }
        tag = next(iter({key[0] for key in by_key}))
        game = next(game for game in results["games"] if game["battle_tag"] == tag)
        expected_agent = "agent_a" if mode == "candidate" else "agent_b"
        expected_role = (
            "challenger" if game["challenger"] == expected_agent else "acceptor"
        )
        actual_role = (
            "challenger"
            if player_sides == {"p1"}
            else "acceptor" if player_sides == {"p2"} else None
        )
        if actual_role != expected_role:
            failures.append(f"{username}/{tag}: process-agent identity mismatch")
        reconstructed = shadow_replay.reconstruct_battles(
            protocol_rows, by_key, username
        )
        commands = canary_audit._sent_commands(protocol_rows)
        process_matches = 0
        for tag in {key[0] for key in by_key}:
            keys = sorted(key for key in by_key if key[0] == tag)
            sent = commands.get(tag, [])
            if len(keys) != len(sent):
                failures.append(
                    f"{username}/{tag}: {len(sent)} commands for {len(keys)} decisions"
                )
                continue
            for key, command in zip(keys, sent, strict=True):
                action = canary_audit._command_action(command, reconstructed[key])
                if action != by_key[key].get("choice"):
                    failures.append(f"{username}/{tag}/{key[1]}: command mismatch")
                else:
                    process_matches += 1
                    outbound_matches += 1
        process_rows.append(
            {
                "username": username,
                "mode": mode,
                "tags": sorted({key[0] for key in by_key}),
                "decisions": len(by_key),
                "outbound_matches": process_matches,
                "search_sha256": _sha256(search_path),
                "protocol_sha256": _sha256(protocol_dump),
            }
        )

    for mode in ("candidate", "baseline"):
        if search_keys[mode] != set(prior_by_mode[mode]):
            failures.append(
                f"{mode}: prior/search mismatch "
                f"{len(prior_by_mode[mode])}/{len(search_keys[mode])}"
            )
    tag_modes = {
        tag: {row["mode"] for row in process_rows if tag in row["tags"]}
        for tag in tags
    }
    if any(modes != {"candidate", "baseline"} for modes in tag_modes.values()):
        failures.append("not every battle has one candidate and one baseline process")
    for tag in tags:
        for mode in ("candidate", "baseline"):
            if sum(
                row["mode"] == mode and tag in row["tags"] for row in process_rows
            ) != 1:
                failures.append(f"{tag}: expected exactly one {mode} process")
    if malformed_samples:
        failures.append(f"{malformed_samples} malformed search samples")
    if engine_mismatches:
        failures.append(f"{engine_mismatches} remote engine mismatches")
    if request_id_count != len(all_request_ids):
        failures.append(
            f"request IDs are not globally unique: {len(all_request_ids)}/{request_id_count}"
        )

    counts = {
        "games": len(results["games"]),
        "completed_games": summary["completed_games"],
        "completed_pairs": summary["completed_pairs"],
        "void_games": summary["void_games"],
        "candidate_decisions": len(search_keys["candidate"]),
        "baseline_decisions": len(search_keys["baseline"]),
        "outbound_matches": outbound_matches,
        "processes": len(process_rows),
        "failures": len(failures),
        "request_ids": request_id_count,
        "globally_unique_request_ids": len(all_request_ids),
        "candidate_wins": summary["agent_a_wins"],
        "candidate_losses": summary["agent_a_losses"],
        "pair_score_mean": summary["pair_score_mean"],
        "pair_sweeps_candidate": summary["pair_sweeps_a"],
        "pair_splits": summary["pair_splits"],
        "pair_sweeps_baseline": summary["pair_sweeps_b"],
    }
    expected_games = protocol["schedule"]["games"]
    expected_pairs = protocol["schedule"]["mirrored_pairs"]
    gate = {
        "all_games_complete": counts["games"]
        == counts["completed_games"]
        == expected_games,
        "all_pairs_complete": counts["completed_pairs"] == expected_pairs,
        "zero_voids": counts["void_games"] == summary["void_pairs"] == 0,
        "candidate_search_active": counts["candidate_decisions"] > 0,
        "adaptive_contract_exact": bool(adaptive_shapes) and engine_mismatches == 0,
        "priors_exactly_joined": all(
            search_keys[mode] == set(prior_by_mode[mode])
            for mode in ("candidate", "baseline")
        ),
        "commands_exactly_joined": outbound_matches
        == counts["candidate_decisions"] + counts["baseline_decisions"],
        "zero_failures": not failures,
        "developmental_pair_score_above_half": counts["pair_score_mean"] > 0.5,
    }
    gate["passed"] = all(gate.values())
    return {
        "schema_version": 1,
        "mode": "adaptive_independent_ensemble_local_smoke_audit",
        "inputs": {
            "results": {"path": str(results_path), "sha256": _sha256(results_path)},
            "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
            "prior_decisions": [
                {"path": str(path), "sha256": _sha256(path)} for path in prior_paths
            ],
            "pairs": {"path": str(pairs_path), "sha256": _sha256(pairs_path)},
            "audit_protocol": {
                "path": str(audit_protocol_path),
                "sha256": _sha256(audit_protocol_path),
            },
        },
        "audit_source_sha256": _sha256(Path(__file__).resolve()),
        "engine_identity_sha256": hashlib.sha256(
            json.dumps(canonical_engine, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "counts": counts,
        "adaptive_shapes": [
            {
                "worlds": shape[0],
                "effective_repeats": shape[1],
                "searches": shape[2],
                "decisions": count,
            }
            for shape, count in sorted(adaptive_shapes.items())
        ],
        "processes": process_rows,
        "failures": failures,
        "gate": gate,
        "authorization": {
            "execution_smoke_passed": gate["passed"],
            "developmental_screen_passed": gate["passed"],
            "controlled_local_screen_authorized": False,
            "strength_claim_authorized": False,
            "larger_local_sample_authorized": False,
            "public_ladder_authorized": False,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--prior-decisions", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--audit-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit(
        args.results.expanduser().resolve(),
        args.log_dir.expanduser().resolve(),
        [path.expanduser().resolve() for path in args.prior_decisions],
        args.protocol.expanduser().resolve(),
        args.pairs.expanduser().resolve(),
        args.audit_protocol.expanduser().resolve(),
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "gate": report["gate"]}, sort_keys=True))
    print(output)
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
