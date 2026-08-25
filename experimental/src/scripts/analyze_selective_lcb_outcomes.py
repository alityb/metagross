#!/usr/bin/env python3
"""Descriptive LCB/outcome analysis for a selective shared-root H2H run."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


class AnalysisError(ValueError):
    """Raised when live-run artifacts cannot support a strict analysis."""


LOG_GAME_RE = re.compile(r"[xy](\d{3})[0-9a-f]*$")
TELEMETRY_MARKER = "SELECTIVE_SHARED "
LCB_BINS = (
    ("0_to_0.01", 0.01),
    ("0.01_to_0.025", 0.025),
    ("0.025_to_0.05", 0.05),
    ("above_0.05", math.inf),
)


def _load_completed_games(progress_path: Path) -> dict[int, dict[str, Any]]:
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read progress snapshot: {exc}") from exc
    entries = progress.get("games")
    if not isinstance(entries, list):
        raise AnalysisError("progress snapshot games must be a list")
    games: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise AnalysisError("progress game entries must be objects")
        game = entry.get("game_index")
        winner = entry.get("winner")
        if isinstance(game, bool) or not isinstance(game, int) or game <= 0:
            raise AnalysisError("game_index must be a positive integer")
        if game in games:
            raise AnalysisError(f"duplicate completed game index {game}")
        if entry.get("void") is not False or winner not in {"agent_a", "agent_b"}:
            raise AnalysisError(f"game {game} is not a completed decisive game")
        games[game] = entry
    if not games:
        raise AnalysisError("progress snapshot contains no completed games")
    return games


def _read_events(
    log_dir: Path, completed_games: set[int]
) -> tuple[dict[int, list[dict[str, Any]]], set[int], int]:
    events_by_game = {game: [] for game in completed_games}
    games_with_candidate_telemetry: set[int] = set()
    all_records = 0
    for path in sorted(log_dir.glob("*.log")):
        match = LOG_GAME_RE.search(path.stem)
        if match is None:
            continue
        game = int(match.group(1))
        if game not in completed_games:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            raise AnalysisError(f"cannot read {path}: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            if TELEMETRY_MARKER not in line:
                continue
            try:
                record = json.loads(line.split(TELEMETRY_MARKER, 1)[1])
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"invalid telemetry in {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise AnalysisError(f"telemetry in {path}:{line_number} must be an object")
            games_with_candidate_telemetry.add(game)
            all_records += 1
            if not record.get("triggered"):
                continue
            if record.get("paired_evaluation_complete") is not True:
                raise AnalysisError(f"incomplete paired evaluation in {path}:{line_number}")
            lcb = record.get("paired_lcb")
            if lcb is not None:
                if isinstance(lcb, bool) or not isinstance(lcb, (int, float)):
                    raise AnalysisError(f"nonnumeric paired_lcb in {path}:{line_number}")
                lcb = float(lcb)
                if not math.isfinite(lcb):
                    raise AnalysisError(f"nonfinite paired_lcb in {path}:{line_number}")
                record["paired_lcb"] = lcb
            if not isinstance(record.get("overridden"), bool):
                raise AnalysisError(f"invalid overridden flag in {path}:{line_number}")
            events_by_game[game].append(record)
    missing = sorted(completed_games - games_with_candidate_telemetry)
    if missing:
        raise AnalysisError(f"completed games lack candidate telemetry: {missing}")
    return events_by_game, games_with_candidate_telemetry, all_records


def _wilson95(wins: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = wins / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [round(center - radius, 6), round(center + radius, 6)]


def _game_summary(game_ids: Iterable[int], games: dict[int, dict[str, Any]]) -> dict[str, Any]:
    unique_ids = sorted(set(game_ids))
    wins = sum(games[game]["winner"] == "agent_a" for game in unique_ids)
    total = len(unique_ids)
    return {
        "games": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(wins / total, 6) if total else None,
        "wilson95": _wilson95(wins, total),
    }


def _lcb_bin(value: float) -> str:
    for name, upper_bound in LCB_BINS:
        if value <= upper_bound:
            return name
    raise AssertionError("unreachable LCB bin")


def analyze(run_dir: Path) -> dict[str, Any]:
    progress_path = run_dir / "result.json.progress.json"
    games = _load_completed_games(progress_path)
    events_by_game, telemetry_games, all_records = _read_events(
        run_dir / "logs", set(games)
    )

    classes: dict[str, list[int]] = {
        "actual_override": [],
        "positive_lcb_retained_baseline": [],
        "triggered_without_positive_lcb": [],
        "no_trigger": [],
    }
    bin_events: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in LCB_BINS}
    triggered_events = 0
    positive_lcb_events = 0
    override_events = 0
    unavailable_lcb_events = 0

    for game, events in events_by_game.items():
        triggered_events += len(events)
        positive = [event for event in events if (event.get("paired_lcb") or 0.0) > 0.0]
        overrides = [event for event in events if event["overridden"]]
        positive_lcb_events += len(positive)
        override_events += len(overrides)
        unavailable_lcb_events += sum(event.get("paired_lcb") is None for event in events)
        if overrides:
            classes["actual_override"].append(game)
        elif positive:
            classes["positive_lcb_retained_baseline"].append(game)
        elif events:
            classes["triggered_without_positive_lcb"].append(game)
        else:
            classes["no_trigger"].append(game)
        for event in overrides:
            lcb = event.get("paired_lcb")
            if lcb is None or lcb <= 0.0:
                raise AnalysisError(f"game {game} has an override without positive LCB")
            bin_events[_lcb_bin(lcb)].append({"game": game, "lcb": lcb})

    return {
        "schema_version": 1,
        "claim_status": "descriptive_only_no_same_state_counterfactual",
        "run_dir": str(run_dir.resolve()),
        "completed_games": len(games),
        "max_completed_game_index": max(games),
        "candidate_wins": sum(game["winner"] == "agent_a" for game in games.values()),
        "candidate_losses": sum(game["winner"] == "agent_b" for game in games.values()),
        "telemetry": {
            "games": len(telemetry_games),
            "all_decision_records": all_records,
            "triggered_events": triggered_events,
            "positive_lcb_events": positive_lcb_events,
            "unavailable_lcb_events": unavailable_lcb_events,
            "override_events": override_events,
        },
        "exclusive_game_classes": {
            name: _game_summary(game_ids, games) for name, game_ids in classes.items()
        },
        "override_lcb_bins": {
            name: {
                "events": len(events),
                "distinct_game_outcomes": _game_summary(
                    (event["game"] for event in events), games
                ),
            }
            for name, events in bin_events.items()
        },
        "limitations": [
            "Game outcome is downstream of the analyzed decision and all later decisions.",
            "Overrides are selected in high-disagreement states, so game classes are confounded.",
            "Multiple override events can occur in one game; LCB-bin game sets can overlap.",
            "The run did not execute baseline and override from the same private root and chance tape.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    classes = report["exclusive_game_classes"]
    bins = report["override_lcb_bins"]
    lines = [
        "# Selective Shared-Root LCB Outcome Snapshot",
        "",
        f"Scope: {report['completed_games']} completed decisive games through game "
        f"{report['max_completed_game_index']}, candidate "
        f"{report['candidate_wins']}-{report['candidate_losses']}.",
        "",
        "This report is descriptive only. Positive LCB is paired-search evidence for the shared "
        "policy over the baseline action; it is not a same-state estimate of the executed "
        "mixture action's effect on the battle winner.",
        "",
        "## Exclusive Game Classes",
        "",
        "| Class | Record | Win rate | Wilson 95% interval |",
        "|---|---:|---:|---:|",
    ]
    for name, values in classes.items():
        interval = values["wilson95"]
        interval_text = "n/a" if interval is None else f"{interval[0]:.1%}-{interval[1]:.1%}"
        rate = "n/a" if values["win_rate"] is None else f"{values['win_rate']:.1%}"
        lines.append(
            f"| `{name}` | {values['wins']}-{values['losses']} | {rate} | {interval_text} |"
        )
    lines.extend(
        [
            "",
            "## Actual Overrides By LCB",
            "",
            "| LCB bin | Events | Distinct-game record | Win rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, values in bins.items():
        outcomes = values["distinct_game_outcomes"]
        rate = "n/a" if outcomes["win_rate"] is None else f"{outcomes['win_rate']:.1%}"
        lines.append(
            f"| `{name}` | {values['events']} | {outcomes['wins']}-{outcomes['losses']} | {rate} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()
    report = analyze(args.run_dir)
    if args.json_out:
        _atomic_write(args.json_out, json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.markdown_out:
        _atomic_write(args.markdown_out, render_markdown(report))
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
