#!/usr/bin/env python3
"""Count Cycle 12 server-transport grammars label-blind before measurement."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from experimental.src.scripts import cycle12_replay_audit as v12
from experimental.src.scripts.select_cycle8_replay_panel import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines() if line]
    rows = [row for row in rows if row["commit_present"]]
    counts: Counter[str] = Counter()
    by_commit: Counter[tuple[str, str]] = Counter()
    authenticated = Counter()
    for row in rows:
        raw = json.loads(Path(row["raw_path"]).read_text())
        lines = raw["log"].splitlines()
        names = v12.authenticated_player_names(lines)
        for line in lines:
            category = None
            if v12.INVITE_RE.fullmatch(line): category = "invite_response"
            elif v12.HIDELINES_UNLINK_RE.fullmatch(line): category = "hidelines_unlink"
            elif v12.SIMPLE_FORFEIT_RE.fullmatch(line): category = "simple_forfeit"
            elif line == v12.MODERATED_CHAT_LINE: category = "moderated_chat_banner"
            elif v12.LOOKUP_ERROR_RE.fullmatch(line): category = "lookup_error"
            if category:
                counts[category] += 1
                by_commit[(category, row["showdown_commit"])] += 1
                authenticated[category] += int(v12._server_transport(line, names))
    report = {
        "schema": "metagross-cycle12-transport-inventory/v1",
        "corpus_sha256": sha256(args.corpus),
        "positive_rows_scanned": len(rows),
        "occurrences": dict(sorted(counts.items())),
        "authenticated_or_exact_occurrences": dict(sorted(authenticated.items())),
        "by_commit": {
            f"{category}:{commit}": count
            for (category, commit), count in sorted(by_commit.items())
        },
        "terminal_winner_or_teacher_fields_used": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

