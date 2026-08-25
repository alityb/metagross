#!/usr/bin/env python3
"""Build Cycle40's label-blind registry of every discoverable prior live identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle40_integrated_h2h_20260816"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def argv_value(argv: list[object], flag: str) -> str | None:
    for index, value in enumerate(argv[:-1]):
        if value == flag and isinstance(argv[index + 1], str):
            return argv[index + 1]
    return None


def main() -> None:
    output = RUN / "PRIOR_IDENTITY_REGISTRY_V2.json"
    if output.exists():
        raise FileExistsError(output)
    pair_sources = []
    pairs: set[str] = set()
    teams: set[str] = set()
    pair_ids: set[str] = set()
    battle_seeds: set[str] = set()
    def mechanical_rows(value: object):
        if isinstance(value, dict):
            if any(key in value for key in ("team_1_sha256", "team_2_sha256", "pair_id", "battle_seed")):
                yield value
            for child in value.values():
                yield from mechanical_rows(child)
        elif isinstance(value, list):
            for child in value:
                yield from mechanical_rows(child)

    for path in sorted((ROOT / "experimental/runs").rglob("*.json")):
        if RUN in path.parents:
            continue
        if "sealed" in str(path).lower() and "93" in str(path):
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        accepted = 0
        for row in mechanical_rows(payload):
            left, right = row.get("team_1_sha256"), row.get("team_2_sha256")
            if isinstance(left, str) and isinstance(right, str):
                pairs.add(canonical(sorted((left, right))))
                teams.update((left, right))
                accepted += 1
            if isinstance(row.get("pair_id"), str):
                pair_ids.add(row["pair_id"])
            if row.get("battle_seed") is not None:
                battle_seeds.add(canonical(row["battle_seed"]))
        if accepted or any(
            isinstance(row.get(key), (str, list))
            for row in mechanical_rows(payload)
            for key in ("pair_id", "battle_seed")
        ):
            pair_sources.append({"path": str(path.resolve()), "sha256": sha(path), "rows": accepted})

    mirror_seeds: set[str] = set()
    production_seeds: set[str] = set()
    run_ids: set[str] = set()
    username_prefixes: set[str] = set()
    config_sources = []
    for path in sorted((ROOT / "experimental/runs").rglob("*ARGV.json")):
        if RUN in path.parents:
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        argv = payload.get("argv")
        if not isinstance(argv, list):
            continue
        for flag, sink in (
            ("--mirror-seed", mirror_seeds),
            ("--production-run-seed", production_seeds),
            ("--run-id", run_ids),
            ("--username-prefix", username_prefixes),
        ):
            value = argv_value(argv, flag)
            if value is not None:
                sink.add(value)
        config_sources.append({"path": str(path.resolve()), "sha256": sha(path)})

    usernames: set[str] = set()
    username_sources = []
    for path in sorted((ROOT / "experimental/runs").rglob("REGISTRATION_CONSUMPTION.json")):
        if RUN in path.parents:
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        found = 0
        for row in payload.get("registrations", []):
            if isinstance(row, dict) and isinstance(row.get("username"), str):
                usernames.add(row["username"])
                found += 1
        username_sources.append({"path": str(path.resolve()), "sha256": sha(path), "rows": found})
    for path in (ROOT / "experimental/runs").rglob("*.protocol.jsonl"):
        if RUN not in path.parents:
            usernames.add(path.name.removesuffix(".protocol.jsonl"))

    payload = {
        "schema": "metagross-cycle40-prior-identity-registry/v1",
        "status": "frozen_before_cycle40_pair_generation",
        "pair_sources": pair_sources,
        "config_sources": config_sources,
        "username_sources": username_sources,
        "unordered_team_pairs": sorted(pairs),
        "individual_team_sha256": sorted(teams),
        "pair_ids": sorted(pair_ids),
        "battle_seeds": sorted(battle_seeds),
        "mirror_seeds": sorted(mirror_seeds),
        "production_run_seeds": sorted(production_seeds),
        "run_ids": sorted(run_ids),
        "username_prefixes": sorted(username_prefixes),
        "usernames": sorted(usernames),
        "counts": {
            "pair_sources": len(pair_sources),
            "unordered_team_pairs": len(pairs),
            "individual_teams": len(teams),
            "pair_ids": len(pair_ids),
            "battle_seeds": len(battle_seeds),
            "config_sources": len(config_sources),
            "usernames": len(usernames),
        },
        "labels_or_outcomes_read": False,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
