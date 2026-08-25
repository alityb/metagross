#!/usr/bin/env python3
"""Select the frozen label-blind 128-battle Cycle 8 replay panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


DOMAIN = "cycle8-remat-20260815"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata(
    path: Path,
    source: str,
    showdown_repo: Path,
    commit_cache: dict[str, bool],
) -> dict[str, Any] | None:
    record = json.loads(path.read_text())
    version = None
    start_seed = None
    player_records = 0
    inputlog = record.get("inputlog") or ""
    for line in inputlog.splitlines():
        if line.startswith(">version "):
            version = line.split(" ", 1)[1].strip()
        elif line.startswith(">start "):
            start = json.loads(line.split(" ", 1)[1])
            start_seed = start.get("seed")
        elif line.startswith(">player "):
            player_records += 1
    terminal = any(
        line.startswith("|win|") or line.startswith("|tie|")
        for line in (record.get("log") or "").splitlines()
    )
    if not version or not start_seed or not record.get("id") or player_records != 2 or not terminal:
        return None
    file_hash = sha256(path)
    rank = hashlib.sha256(
        "\0".join((DOMAIN, source, str(record["id"]), file_hash)).encode()
    ).hexdigest()
    if version not in commit_cache:
        commit_cache[version] = subprocess.run(
            ["git", "-C", str(showdown_repo), "cat-file", "-e", f"{version}^{{commit}}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    commit_present = commit_cache[version]
    return {
        "source": source,
        "battle_id": str(record["id"]),
        "raw_path": str(path.resolve()),
        "raw_sha256": file_hash,
        "public_log_sha256": hashlib.sha256((record.get("log") or "").encode()).hexdigest(),
        "inputlog_sha256": hashlib.sha256(inputlog.encode()).hexdigest(),
        "start_seed_sha256": hashlib.sha256(str(start_seed).encode()).hexdigest(),
        "showdown_commit": version,
        "commit_present": commit_present,
        "rank_sha256": rank,
    }


def select(primary: list[dict[str, Any]], external: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row["showdown_commit"] for row in primary)
    major = [commit for commit, _ in counts.most_common(4)]
    rare = [row for row in primary if row["showdown_commit"] not in major]
    if len(rare) != 10:
        raise ValueError(f"expected 10 rare-commit rows, found {len(rare)}")
    selected: list[dict[str, Any]] = []
    for row in sorted(rare, key=lambda item: item["rank_sha256"]):
        selected.append({**row, "stratum": "primary_rare_all"})
    for commit in major:
        cohort = sorted(
            (row for row in primary if row["showdown_commit"] == commit),
            key=lambda item: item["rank_sha256"],
        )[:22]
        if len(cohort) != 22:
            raise ValueError(f"major commit {commit} has only {len(cohort)} rows")
        selected.extend({**row, "stratum": f"primary_major_{commit}"} for row in cohort)
    cohort = sorted(external, key=lambda item: item["rank_sha256"])[:30]
    if len(cohort) != 30:
        raise ValueError(f"external source has only {len(cohort)} rows")
    selected.extend({**row, "stratum": "external_30"} for row in cohort)
    selected.sort(key=lambda row: (row["stratum"], row["rank_sha256"]))
    if len(selected) != 128:
        raise ValueError(f"expected 128 rows, found {len(selected)}")
    if sum(row["commit_present"] for row in selected) != 124:
        raise ValueError("expected 124 positive and four unavailable-commit controls")
    for key in ("battle_id", "raw_sha256", "start_seed_sha256"):
        if len({(row["source"], row[key]) for row in selected}) != 128:
            raise ValueError(f"duplicate source-scoped {key}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--showdown-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    commit_cache: dict[str, bool] = {}
    primary_candidates = [
        metadata(path, "primary", args.showdown_repo, commit_cache)
        for path in sorted(args.primary.glob("*.json"))
    ]
    external_candidates = [
        metadata(path, "external", args.showdown_repo, commit_cache)
        for path in sorted(args.external.glob("*.json"))
    ]
    primary = [row for row in primary_candidates if row is not None]
    external = [row for row in external_candidates if row is not None]
    rows = select(primary, external)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report = {
        "schema": "metagross-cycle8-replay-panel/v1",
        "selection_domain": DOMAIN,
        "rows": len(rows),
        "positive": sum(row["commit_present"] for row in rows),
        "negative_controls": sum(not row["commit_present"] for row in rows),
        "strata": dict(sorted(Counter(row["stratum"] for row in rows).items())),
        "commits": dict(sorted(Counter(row["showdown_commit"] for row in rows).items())),
        "panel_sha256": sha256(args.output),
        "teacher_or_outcome_fields_read": False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
