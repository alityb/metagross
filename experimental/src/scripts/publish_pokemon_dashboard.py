#!/usr/bin/env python3
"""Publish a sanitized live snapshot for pokemon.amtayeb.dev.

Set either:
  METAGROSS_DASHBOARD_FILE=/path/to/personal-website/public/pokemon-status.json
or:
  METAGROSS_DASHBOARD_INGEST_URL=https://pokemon.amtayeb.dev/api/pokemon/ingest
  METAGROSS_DASHBOARD_SECRET=...

This intentionally never reads or sends ladder passwords.
"""
from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
USERNAME = os.environ.get("METAGROSS_DASHBOARD_USERNAME", "zukofan839")
LOG = Path(os.environ.get("METAGROSS_DASHBOARD_LOG", ROOT / "experiments" / "exit_r2_ladder" / "logs" / "zukofan_839.log"))


def showdown() -> dict:
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = ssl.create_default_context()
    request = urllib.request.Request(
        f"https://pokemonshowdown.com/users/{USERNAME}.json",
        headers={"User-Agent": "metagross-telemetry/1.0"},
    )
    with urllib.request.urlopen(request, timeout=10, context=context) as response:
        data = json.load(response)
    rating = (data.get("ratings") or {}).get("gen9randombattle") or {}
    return {
        "username": data.get("userid", USERNAME),
        "rating": {
            "elo": rating.get("elo"), "gxe": rating.get("gxe"), "rd": rating.get("rprd"),
            "wins": rating.get("w"), "losses": rating.get("l"),
        },
    }


def battle() -> dict:
    if not LOG.exists():
        return {}
    lines = LOG.read_text(errors="ignore").splitlines()
    out: dict[str, object] = {}
    for line in lines[-500:]:
        init = re.search(r"Initialized (battle-[^ ]+) against: (.+)$", line)
        if init:
            out["tag"], out["opponent"] = init.groups()
        turn = re.search(r"Turn: (\d+)", line)
        if turn:
            out["turn"] = int(turn.group(1))
        choice = re.search(r"Choice: (.+)$", line)
        if choice:
            out["choice"] = choice.group(1)
    return out


def main() -> None:
    # Public rating is fetched by the dashboard itself. Publish only private,
    # sanitized live telemetry here so a stale local snapshot cannot overwrite
    # current Showdown numbers.
    snapshot = {
        "battle": battle(),
        "system": {"ladderRunning": bool(battle()), "priorHealthy": True},
        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output = os.environ.get("METAGROSS_DASHBOARD_FILE")
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2) + "\n")
        print(path)
        return
    url = os.environ["METAGROSS_DASHBOARD_INGEST_URL"]
    secret = os.environ["METAGROSS_DASHBOARD_SECRET"]
    request = urllib.request.Request(url, data=json.dumps(snapshot).encode(), method="POST", headers={"Content-Type": "application/json", "Authorization": f"Bearer {secret}"})
    with urllib.request.urlopen(request, timeout=10) as response:
        print(response.status)


if __name__ == "__main__":
    main()
