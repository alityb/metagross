#!/usr/bin/env python3
"""Compare certified and search-first controllers on captured decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from srcs.metagross import run_foul_play, shadow_replay


def _request_valid(action: str, request_actions: list[str]) -> bool:
    return action in request_actions or (not request_actions and action == "no move")


def replay_capture(capture: Path) -> tuple[dict, list[dict]]:
    protocol, searches, metadata = shadow_replay.load_capture(capture)
    decisions = metadata.pop("decision_rows")
    ledgers = metadata.pop("holdout_ledger_rows") or {}
    username = metadata["manifest"]["ladder"]["username"]
    battles = shadow_replay.reconstruct_battles(protocol, searches, username)
    certified_histories: dict = {}
    search_first_histories: dict = {}
    rows = []

    for key in sorted(searches):
        tag, decision_index = key
        battle = battles[key]
        decision = decisions[key]
        search = searches[key]
        results = shadow_replay._mcts_results(search)
        priors = shadow_replay._priors(decision)
        panel = (ledgers.get(key) or {}).get("certification")
        evidence = (
            shadow_replay._recompute_captured_v5_panel(panel)
            if isinstance(panel, dict)
            else None
        )
        certified_choice, certified = run_foul_play.select_final_choice(
            battle,
            results,
            priors,
            certified_histories,
            independent_evidence=evidence,
        )
        search_choice, search_first = run_foul_play.select_search_first_choice(
            battle,
            results,
            priors,
            search_first_histories,
            independent_evidence=evidence,
        )
        request_actions = search_first["request_actions"]
        changed = certified_choice != search_choice
        disagreement = certified["baseline"] != certified["raw_choice"]
        deterministic = (
            certified["selection_class"] == "deterministic_correction"
            or search_first["selection_class"] == "deterministic_correction"
        )
        rows.append(
            {
                "capture": str(capture),
                "tag": tag,
                "decision_idx": decision_index,
                "battle_turn": decision.get("battle_turn"),
                "recorded_choice": search.get("choice"),
                "policy_choice": certified["baseline"],
                "search_choice": certified["raw_choice"],
                "policy_search_disagreement": disagreement,
                "certified_choice": certified_choice,
                "certified_reason": certified["reason"],
                "search_first_choice": search_choice,
                "search_first_reason": search_first["reason"],
                "search_first_selection_class": search_first["selection_class"],
                "changed": changed,
                "change_explained": not changed or disagreement or deterministic,
                "certified_matches_recorded": certified_choice == search.get("choice"),
                "certified_request_valid": _request_valid(
                    certified_choice, request_actions
                ),
                "search_first_request_valid": _request_valid(
                    search_choice, request_actions
                ),
                "request_actions": request_actions,
                "certified_blocked_safeguard": certified["blocked_safeguard"],
                "search_first_blocked_safeguard": search_first["blocked_safeguard"],
                "search_first_shadow_risks": search_first["shadow_risks"],
            }
        )

    counts = {
        "decisions": len(rows),
        "decision_search_joins": len(rows),
        "canonical_action_mappings": sum(
            row["certified_request_valid"] + row["search_first_request_valid"]
            for row in rows
        ),
        "expected_canonical_action_mappings": 2 * len(rows),
        "policy_search_agreements": sum(
            not row["policy_search_disagreement"] for row in rows
        ),
        "policy_search_disagreements": sum(
            row["policy_search_disagreement"] for row in rows
        ),
        "actions_changed": sum(row["changed"] for row in rows),
        "unchanged_policy_search_agreements": sum(
            not row["policy_search_disagreement"] and not row["changed"] for row in rows
        ),
        "deterministic_safeguard_activations": sum(
            row["search_first_selection_class"] == "deterministic_correction"
            for row in rows
        ),
        "illegal_certified_outputs": sum(
            not row["certified_request_valid"] for row in rows
        ),
        "illegal_search_first_outputs": sum(
            not row["search_first_request_valid"] for row in rows
        ),
        "unexplained_changes": sum(not row["change_explained"] for row in rows),
        "certified_recorded_mismatches": sum(
            not row["certified_matches_recorded"] for row in rows
        ),
    }
    report = {
        "schema_version": 1,
        "mode": "certified_vs_search_first_offline_replay",
        "capture": metadata,
        "counts": counts,
        "deltas": [row for row in rows if row["changed"]],
        "gate": {
            "complete_decision_search_joins": counts["decision_search_joins"]
            == counts["decisions"],
            "complete_canonical_action_mappings": counts[
                "canonical_action_mappings"
            ]
            == counts["expected_canonical_action_mappings"],
            "zero_illegal_outputs": counts["illegal_certified_outputs"] == 0
            and counts["illegal_search_first_outputs"] == 0,
            "zero_unexplained_changes": counts["unexplained_changes"] == 0,
            "certified_controller_reproduces_capture": counts[
                "certified_recorded_mismatches"
            ]
            == 0,
        },
    }
    report["gate"]["passed"] = all(report["gate"].values())
    return report, rows


def aggregate(reports: list[dict]) -> dict:
    count_names = reports[0]["counts"] if reports else {}
    counts = {
        name: sum(report["counts"][name] for report in reports)
        for name in count_names
    }
    gate = {
        "all_capture_gates_passed": all(report["gate"]["passed"] for report in reports),
        "decisions": counts.get("decisions", 0),
        "expected_decisions": 664,
        "complete_664_decision_corpus": counts.get("decisions") == 664,
        "zero_illegal_outputs": counts.get("illegal_certified_outputs") == 0
        and counts.get("illegal_search_first_outputs") == 0,
        "zero_unexplained_changes": counts.get("unexplained_changes") == 0,
        "certified_controller_reproduces_captures": counts.get(
            "certified_recorded_mismatches"
        )
        == 0,
    }
    gate["passed"] = all(
        value for name, value in gate.items() if name not in {"decisions", "expected_decisions"}
    )
    return {
        "schema_version": 1,
        "mode": "certified_vs_search_first_offline_replay_aggregate",
        "captures": [report["capture"] for report in reports],
        "counts": counts,
        "gate": gate,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    rows = []
    for capture in args.capture:
        report, capture_rows = replay_capture(capture.expanduser().resolve())
        reports.append(report)
        rows.extend(capture_rows)
    summary = aggregate(reports)
    (output / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "captures.jsonl").open("w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(report, separators=(",", ":"), sort_keys=True) + "\n")
    with (output / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    print(json.dumps(summary["counts"], sort_keys=True))
    print(output)
    if not summary["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
