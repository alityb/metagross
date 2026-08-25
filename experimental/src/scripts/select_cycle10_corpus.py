#!/usr/bin/env python3
"""Build the label-blind complete Cycle 10 human replay corpus manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from experimental.src.scripts.select_cycle8_replay_panel import metadata, sha256


DETERMINISM_DOMAIN = "cycle10-determinism-20260815"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--showdown-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spot-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cache: dict[str, bool] = {}
    rows = []
    for source, root in (("primary", args.primary), ("external", args.external)):
        for path in sorted(root.glob("*.json")):
            row = metadata(path, source, args.showdown_repo, cache)
            if row is not None:
                row["raw_relative_path"] = str(path.relative_to(root))
                rows.append(row)
    rows.sort(key=lambda row: (row["source"], row["battle_id"], row["raw_sha256"]))
    positives = [row for row in rows if row["commit_present"]]
    negatives = [row for row in rows if not row["commit_present"]]
    if len(rows) != 20633 or len(positives) != 20629 or len(negatives) != 4:
        raise ValueError(
            f"unexpected corpus cardinality rows={len(rows)} positives={len(positives)} negatives={len(negatives)}"
        )
    if len({row["battle_id"] for row in rows}) != len(rows):
        raise ValueError("battle IDs overlap across raw sources")
    if {row["showdown_commit"] for row in negatives} != {
        "16af15f45d6fa53367a8503a9003f022b639a793",
        "48fe7f089d532c1f99fd4071a92d6dbfb4b0f145",
    }:
        raise ValueError("negative-control commits changed")
    counts = Counter(row["showdown_commit"] for row in positives)
    majors = sorted(commit for commit, count in counts.items() if count >= 100)
    rares = [row for row in positives if row["showdown_commit"] not in majors]
    if len(majors) != 5 or len(rares) != 6:
        raise ValueError("expected five major commits and six positive rare rows")
    for row in positives:
        row["determinism_rank_sha256"] = hashlib.sha256(
            "\0".join((
                DETERMINISM_DOMAIN, row["source"], row["battle_id"], row["raw_sha256"],
            )).encode()
        ).hexdigest()
    spot = list(rares)
    for commit in majors:
        cohort = sorted(
            (row for row in positives if row["showdown_commit"] == commit),
            key=lambda row: row["determinism_rank_sha256"],
        )[:50]
        if len(cohort) != 50:
            raise ValueError(f"major commit {commit} lacks 50 spot rows")
        spot.extend(cohort)
    spot.sort(key=lambda row: (row["showdown_commit"], row["determinism_rank_sha256"]))
    if len(spot) != 256 or len({row["battle_id"] for row in spot}) != 256:
        raise ValueError("determinism panel cardinality changed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    args.spot_output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in spot))
    report = {
        "schema": "metagross-cycle10-corpus-manifest/v1",
        "rows": len(rows), "positive": len(positives), "negative": len(negatives),
        "sources": dict(sorted(Counter(row["source"] for row in rows).items())),
        "positive_by_commit": dict(sorted(counts.items())),
        "major_commits": majors,
        "rare_positive_rows": len(rares),
        "determinism_rows": len(spot),
        "corpus_sha256": sha256(args.output),
        "determinism_sha256": sha256(args.spot_output),
        "teacher_choice_or_terminal_value_used_for_selection": False,
        "terminal_presence_required": True,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
