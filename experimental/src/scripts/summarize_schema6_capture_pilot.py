#!/usr/bin/env python3
"""Fail-closed aggregate audit for frozen local schema-6 capture stages."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


PILOT_EXPECTED = {
    "peer": (300, "production_r1_search_first", 2),
    "direct_r1": (100, "direct_r1", 1),
    "unguided": (100, "foul_play", 1),
}
SCALE_5000_EXPECTED = {
    "peer": (3000, "production_r1_search_first", 2),
    "direct_r1": (1000, "direct_r1", 1),
    "unguided": (1000, "foul_play", 1),
}
R1_CHECKPOINT_SHA256 = "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"
CAPTURE_ENGINE_SOURCE_SHA256 = "ccc5dd035c25fbf8e6b98301198cdaa35ce1012ae6f45ab81851965b0b05a0a3"
CAPTURE_ENGINE_BINARY_SHA256 = "3910185bb7f5e5f0283781b0b2292664f4c980126f320325143ba5970d4aba35"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-games", type=int, choices=(500, 5000), default=500)
    parser.add_argument("--authorization-report", type=Path)
    args = parser.parse_args()
    expected = PILOT_EXPECTED if args.stage_games == 500 else SCALE_5000_EXPECTED
    strict_scale_contract = args.stage_games == 5000
    authorization = None
    if args.stage_games == 5000:
        if args.authorization_report is None:
            parser.error("5,000-game stage requires --authorization-report")
        authorization = json.loads(args.authorization_report.read_text())
        if (
            authorization.get("schema") != "metagross-schema6-20k-50k-screen/v1"
            or authorization.get("admitted") is not True
            or authorization.get("scale_gate_admitted") is not True
            or int(authorization.get("eligible_roots", 0)) < 50
            or authorization.get("withheld_roots_processed") != 0
        ):
            raise ValueError("5,000-game authorization report did not pass the frozen gate")
    strata = {}
    total_games = total_groups = total_complete = total_candidates = 0
    pair_identities = []
    admitted = True
    for profile, (expected_games, expected_agent_b, group_multiplier) in expected.items():
        directory = args.root / profile
        result = json.loads((directory / "result.json").read_text())
        capture = json.loads((directory / "schema6-capture-audit.json").read_text())
        bridge = json.loads((directory / "schema6-panel-bridge-audit.json").read_text())
        summary = result.get("summary") or {}
        games = result.get("games")
        completed_games = int(summary.get("completed_games", 0))
        expected_groups = expected_games * group_multiplier
        minimum_complete = (expected_groups * 95 + 99) // 100
        pairs = {}
        if isinstance(games, list):
            for game in games:
                pairs.setdefault(int(game["pair_index"]), []).append(game)
        valid_pairs = len(pairs) == expected_games // 2
        for pair in pairs.values():
            legs = sorted(int(game["pair_leg"]) for game in pair)
            identities = {
                (game["battle_seed"], game["team_1_sha256"], game["team_2_sha256"])
                for game in pair
            }
            if legs != [1, 2] or len(identities) != 1:
                valid_pairs = False
                continue
            pair_identities.append(next(iter(identities)))
        stratum_admitted = bool(
            completed_games == expected_games
            and (
                not strict_scale_contract
                or (isinstance(games, list) and len(games) == expected_games and valid_pairs)
            )
            and summary.get("agent_a") == "production_r1_search_first"
            and summary.get("agent_b") == expected_agent_b
            and int(summary.get("void_games", -1)) == 0
            and capture.get("admitted") is True
            and int(capture.get("groups", -1)) == expected_groups
            and int(capture.get("complete_groups", 0)) >= minimum_complete
            and float(capture.get("capture_rate", 0.0)) >= 0.95
            and (
                not strict_scale_contract
                or (
                    int(capture.get("duplicate_decisions", -1)) == 0
                    and int(capture.get("duplicate_snapshots", -1)) == 0
                    and int(capture.get("invalid_snapshots", -1)) == 0
                )
            )
            and bridge.get("admitted") is True
            and int(bridge.get("candidate_rows", 0)) > 0
        )
        admitted = admitted and stratum_admitted
        total_games += completed_games
        total_groups += int(capture.get("groups", 0))
        total_complete += int(capture.get("complete_groups", 0))
        total_candidates += int(bridge.get("candidate_rows", 0))
        strata[profile] = {
            "admitted": stratum_admitted,
            "completed_games": completed_games,
            "capture_groups": capture.get("groups"),
            "complete_groups": capture.get("complete_groups"),
            "candidate_rows": bridge.get("candidate_rows"),
            "mirrored_pairs": len(pairs),
        }
    unique_pairs = len(set(pair_identities))
    expected_pairs = args.stage_games // 2
    admitted = bool(admitted and total_games == args.stage_games)
    if strict_scale_contract:
        admitted = bool(
            admitted
            and len(pair_identities) == expected_pairs
            and unique_pairs == expected_pairs
        )
    report = {
        "schema": (
            "metagross-schema6-capture-pilot-summary/v1"
            if args.stage_games == 500
            else "metagross-schema6-local-capture-summary/v2"
        ),
        "admitted": admitted,
        "stage_games": args.stage_games,
        "completed_games": total_games,
        "unique_mirrored_pairs": unique_pairs,
        "capture_groups": total_groups,
        "complete_groups": total_complete,
        "capture_rate": total_complete / total_groups if total_groups else 0.0,
        "candidate_rows": total_candidates,
        "pinned_runtime": {
            "r1_checkpoint_sha256": R1_CHECKPOINT_SHA256,
            "capture_engine_source_sha256": CAPTURE_ENGINE_SOURCE_SHA256,
            "capture_engine_binary_sha256": CAPTURE_ENGINE_BINARY_SHA256,
        },
        "strata": strata,
    }
    if args.stage_games == 500:
        report.update(
            scale_admitted=False,
            scale_blocker="requires >=50 roots after frozen 20k/50k four-way agreement screening",
        )
    else:
        report.update(
            authorization={
                "screen_id": authorization["screen_id"],
                "source_panel_sha256": authorization["source_panel_sha256"],
                "agreement_panel_sha256": authorization["agreement_panel_sha256"],
                "eligible_roots": authorization["eligible_roots"],
                "withheld_roots_processed": authorization["withheld_roots_processed"],
            },
            model_data_stage_admitted=admitted,
            next_scale_admitted=False,
            next_scale_blocker=(
                "requires physical-battle-grouped out-of-fold outcome-residual improvement "
                "before any 25,000-game stage"
            ),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", dir=args.output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, args.output)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["admitted"] else 2)


if __name__ == "__main__":
    main()
