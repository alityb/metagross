#!/usr/bin/env python3
"""Freeze Cycle 11's exact-standard-format corpus without opening labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from experimental.src.scripts.select_cycle8_replay_panel import sha256


DETERMINISM_DOMAIN = "cycle11-determinism-20260815"


def format_id(raw_path: str) -> str:
    raw = json.loads(Path(raw_path).read_text())
    inputlog = raw.get("inputlog")
    if not isinstance(inputlog, str):
        return "<missing-inputlog>"
    starts = []
    for line in inputlog.splitlines():
        if line.startswith(">start "):
            payload = json.loads(line.split(" ", 1)[1])
            starts.append(payload.get("formatid"))
    if len(starts) != 1 or not isinstance(starts[0], str):
        return "<invalid-start-format>"
    return starts[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle10-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spot-output", type=Path, required=True)
    parser.add_argument("--exclusions-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    original = [json.loads(line) for line in args.cycle10_corpus.read_text().splitlines() if line]
    included, excluded = [], []
    for row in original:
        fmt = format_id(row["raw_path"])
        if row["commit_present"] and fmt != "gen9randombattle":
            excluded.append({
                "source": row["source"],
                "raw_sha256": row["raw_sha256"],
                "showdown_commit": row["showdown_commit"],
                "formatid_sha256": hashlib.sha256(fmt.encode("utf-8")).hexdigest(),
            })
        else:
            if not row["commit_present"] and fmt != "gen9randombattle":
                raise ValueError("negative control is not exact gen9randombattle")
            included.append(row)
    positives = [row for row in included if row["commit_present"]]
    negatives = [row for row in included if not row["commit_present"]]
    if len(original) != 20633 or len(positives) != 20560 or len(negatives) != 4 or len(excluded) != 69:
        raise ValueError(
            f"unexpected denominator original={len(original)} positive={len(positives)} "
            f"negative={len(negatives)} excluded={len(excluded)}"
        )
    counts = Counter(row["showdown_commit"] for row in positives)
    majors = sorted(commit for commit, count in counts.items() if count >= 100)
    rares = [row for row in positives if row["showdown_commit"] not in majors]
    if len(majors) != 5:
        raise ValueError("expected five major exact-format commits")
    for row in positives:
        row["determinism_rank_sha256"] = hashlib.sha256(
            "\0".join((
                DETERMINISM_DOMAIN, row["source"], row["battle_id"], row["raw_sha256"],
            )).encode("utf-8")
        ).hexdigest()
    spot = list(rares)
    for commit in majors:
        spot.extend(sorted(
            (row for row in positives if row["showdown_commit"] == commit),
            key=lambda row: row["determinism_rank_sha256"],
        )[:50])
    spot.sort(key=lambda row: (row["showdown_commit"], row["determinism_rank_sha256"]))
    if len(spot) != 256 or len({row["battle_id"] for row in spot}) != 256:
        raise ValueError("unexpected exact-format determinism panel")
    included.sort(key=lambda row: (row["source"], row["battle_id"], row["raw_sha256"]))
    excluded.sort(key=lambda row: (row["source"], row["raw_sha256"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in included))
    args.spot_output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in spot))
    args.exclusions_output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in excluded))
    report = {
        "schema": "metagross-cycle11-corpus-manifest/v1",
        "source_cycle10_corpus_sha256": sha256(args.cycle10_corpus),
        "rows": len(included), "positive": len(positives),
        "negative_controls": len(negatives), "custom_rule_exclusions": len(excluded),
        "positive_by_commit": dict(sorted(counts.items())),
        "excluded_by_format_sha256": dict(sorted(Counter(
            row["formatid_sha256"] for row in excluded
        ).items())),
        "determinism_rows": len(spot),
        "corpus_sha256": sha256(args.output),
        "determinism_sha256": sha256(args.spot_output),
        "exclusions_sha256": sha256(args.exclusions_output),
        "teacher_choice_or_terminal_value_used_for_selection": False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

