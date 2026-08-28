#!/usr/bin/env python3
"""Opponent-strength-stratified win rates from ladder protocol logs.

Tests the sign-flip mechanism directly: if interventions trade decisiveness
for robustness, the flattened arm's deficit should concentrate against
WEAK opponents while holding or gaining against STRONG ones.

Walks protocol.jsonl files under each run root, reconstructs per battle:
opponent Elo at match start (|player| line) and the winner (|win| line),
then reports win rate by opponent-Elo bucket per run.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

PLAYER_RE = re.compile(r"\|player\|(p[12])\|([^|\n]+)\|[^|\n]*\|?(\d+)?")
WIN_RE = re.compile(r"\|win\|([^\n|]+)")
ROOM_RE = re.compile(r">(battle-gen9randombattle-\S+)")
BUCKETS = [(0, 1199), (1200, 1399), (1400, 1599), (1600, 9999)]


def bucket(elo: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= elo <= hi:
            return f"{lo}-{hi if hi < 9999 else '+'}"
    return "?"


def wilson(w: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def analyze(root: Path, me: str) -> dict:
    battles: dict[str, dict] = defaultdict(dict)
    for pf in root.rglob("protocol.jsonl"):
        for line in pf.open():
            try:
                msg = json.loads(line).get("message", "")
            except Exception:
                continue
            room_m = ROOM_RE.search(msg)
            if not room_m:
                continue
            room = room_m.group(1)
            b = battles[room]
            for slot, name, rating in PLAYER_RE.findall(msg):
                name = name.strip()
                if name.lower() != me.lower() and rating:
                    # first-seen rating = at match start
                    b.setdefault("opp", name)
                    b.setdefault("opp_elo", int(rating))
            w = WIN_RE.search(msg)
            if w:
                b["winner"] = w.group(1).strip()
    per_bucket = defaultdict(lambda: [0, 0])
    used = 0
    for room, b in battles.items():
        if "winner" not in b or "opp_elo" not in b:
            continue
        used += 1
        k = bucket(b["opp_elo"])
        per_bucket[k][1] += 1
        if b["winner"].lower() == me.lower():
            per_bucket[k][0] += 1
    out = {"run": str(root), "me": me, "battles_used": used, "buckets": {}}
    for lo, hi in BUCKETS:
        k = f"{lo}-{hi if hi < 9999 else '+'}"
        w, n = per_bucket.get(k, (0, 0))
        lo_ci, hi_ci = wilson(w, n)
        out["buckets"][k] = {
            "wins": w, "games": n,
            "winrate": round(w / n, 3) if n else None,
            "ci95": [round(lo_ci, 3), round(hi_ci, 3)],
        }
    return out


def main() -> None:
    runs = [
        ("flattened", Path("experimental/runs/ladder_flattened_20260826"),
         "roguefan55"),
        ("plain-causal", Path("experimental/runs/ladder_local_pair_20260822/causal_seq"),
         "roguefan31"),
        ("stateless", Path("experimental/runs/ladder_local_pair_20260822/stateless_seq"),
         "tophfan32"),
    ]
    reports = {}
    for label, root, me in runs:
        if root.exists():
            reports[label] = analyze(root, me)
    print(json.dumps(reports, indent=1))
    out = Path("experimental/runs/ladder_flattened_20260826/strength_stratified.json")
    out.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
