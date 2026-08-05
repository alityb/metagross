#!/usr/bin/env python3
"""Publish a sanitized live public-ladder snapshot for the web dashboard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORMAT = "gen9randombattle"
BATTLE_ID = re.compile(r"battle-gen9randombattle-\d+")
ISSUE_PATTERNS = (
    ("network", "connectionclosed", "Showdown connection closed unexpectedly"),
    ("prior", "required prior fetch failed", "Required policy priors failed"),
    ("choice", "invalid choice", "Showdown rejected a selected action"),
    ("login", "login failed", "Showdown login failed"),
    ("runtime", "maximum ladder runtime exceeded", "Ladder runtime limit was reached"),
    ("exception", "traceback (most recent call last)", "The ladder process raised an exception"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_username(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def parse_condition(condition: str) -> tuple[float, str | None, bool]:
    fields = condition.split()
    hp = fields[0] if fields else "0"
    fainted = "fnt" in fields
    status = next((field for field in fields[1:] if field not in {"fnt"}), None)
    if "/" in hp:
        current, maximum = hp.split("/", 1)
        try:
            percent = 100 * float(current) / max(float(maximum), 1)
        except ValueError:
            percent = 0
    else:
        try:
            percent = float(hp)
        except ValueError:
            percent = 0
    return max(0, min(100, percent)), status, fainted or percent <= 0


def parse_species(details: str) -> str:
    return details.split(",", 1)[0].strip() or "Unknown"


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_last_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.stat().st_size:
        return None
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        buffer = bytearray()
        while position > 0:
            position -= 1
            handle.seek(position)
            byte = handle.read(1)
            if byte == b"\n" and buffer:
                break
            buffer[:0] = byte
    try:
        return json.loads(buffer.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def public_pokemon(species: str, condition: str) -> dict[str, Any]:
    hp, status, fainted = parse_condition(condition)
    return {
        "species": species,
        "hpPercent": round(hp, 1),
        "status": status,
        "fainted": fainted,
    }


def battle_state() -> dict[str, Any]:
    return {
        "players": {},
        "active": {},
        "teams": {"p1": {}, "p2": {}},
        "turn": 0,
        "events": [],
        "winner": None,
        "reason": "normal",
        "lastActionBySide": {},
        "lastTimeNs": 0,
    }


def add_event(battle: dict[str, Any], kind: str, label: str, side: str | None) -> None:
    battle["events"].append({"kind": kind, "label": label, "side": side})
    battle["events"] = battle["events"][-12:]


def update_pokemon(
    battle: dict[str, Any], side: str, ident: str, species: str | None, condition: str | None
) -> dict[str, Any]:
    team = battle["teams"][side]
    key = normalize_username(ident) or normalize_username(species or "unknown")
    existing = team.get(key) or {
        "species": species or ident,
        "hpPercent": 100.0,
        "status": None,
        "fainted": False,
    }
    if species:
        existing["species"] = species
    if condition:
        existing.update(public_pokemon(existing["species"], condition))
    team[key] = existing
    return existing


def reduce_protocol(path: Path, username: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    battles: dict[str, dict[str, Any]] = {}
    user_id = normalize_username(username)
    for row in read_jsonl(path) or ():
        if row.get("direction") != "received":
            continue
        message = str(row.get("message", ""))
        match = BATTLE_ID.search(message)
        if not match:
            continue
        tag = match.group(0)
        battle = battles.setdefault(tag, battle_state())
        battle["lastTimeNs"] = max(battle["lastTimeNs"], int(row.get("time_ns") or 0))
        for line in message.splitlines()[1:]:
            parts = line.split("|")
            event = parts[1] if len(parts) > 1 else ""
            if event == "player" and len(parts) >= 4:
                if parts[3]:
                    battle["players"][parts[2]] = parts[3]
            elif event == "turn" and len(parts) >= 3:
                battle["turn"] = int(parts[2])
                add_event(battle, "turn", f"Turn {parts[2]}", None)
            elif event in {"switch", "drag", "replace"} and len(parts) >= 5:
                side = parts[2][:2]
                ident = parts[2].split(":", 1)[-1].strip()
                species = parse_species(parts[3])
                pokemon = update_pokemon(battle, side, ident, species, parts[4])
                battle["active"][side] = pokemon
                add_event(battle, "switch", f"{species} entered the battle", side)
            elif event in {"-damage", "-heal"} and len(parts) >= 4:
                side = parts[2][:2]
                ident = parts[2].split(":", 1)[-1].strip()
                pokemon = update_pokemon(battle, side, ident, None, parts[3])
                if battle["active"].get(side):
                    battle["active"][side] = pokemon
            elif event == "-status" and len(parts) >= 4:
                side = parts[2][:2]
                ident = parts[2].split(":", 1)[-1].strip()
                pokemon = update_pokemon(battle, side, ident, None, None)
                pokemon["status"] = parts[3]
            elif event == "-curestatus" and len(parts) >= 3:
                side = parts[2][:2]
                ident = parts[2].split(":", 1)[-1].strip()
                update_pokemon(battle, side, ident, None, None)["status"] = None
            elif event == "faint" and len(parts) >= 3:
                side = parts[2][:2]
                ident = parts[2].split(":", 1)[-1].strip()
                pokemon = update_pokemon(battle, side, ident, None, "0 fnt")
                add_event(battle, "faint", f"{pokemon['species']} fainted", side)
            elif event == "move" and len(parts) >= 4:
                side = parts[2][:2]
                actor = parts[2].split(":", 1)[-1].strip()
                label = f"{actor} used {parts[3]}"
                battle["lastActionBySide"][side] = label
                add_event(battle, "move", label, side)
            elif event == "win" and len(parts) >= 3:
                battle["winner"] = parts[2]
            elif event == "tie":
                battle["winner"] = ""
            if "lost due to inactivity" in line.lower():
                battle["reason"] = "inactivity"
            elif "forfeited" in line.lower():
                battle["reason"] = "forfeit"

    ordered = sorted(battles.items(), key=lambda item: item[1]["lastTimeNs"])
    recent: list[dict[str, Any]] = []
    current: tuple[str, dict[str, Any]] | None = None
    for tag, battle in ordered:
        if battle["winner"] is None:
            current = (tag, battle)
            continue
        us_side = next(
            (side for side, name in battle["players"].items() if normalize_username(name) == user_id),
            None,
        )
        opponent = next(
            (name for side, name in battle["players"].items() if side != us_side), "Unknown"
        )
        result = "tie" if battle["winner"] == "" else (
            "win" if normalize_username(battle["winner"]) == user_id else "loss"
        )
        recent.append({
            "id": tag,
            "opponent": opponent,
            "result": result,
            "turn": battle["turn"],
            "reason": battle["reason"],
        })

    if not current:
        return None, list(reversed(recent[-8:]))
    tag, battle = current
    us_side = next(
        (side for side, name in battle["players"].items() if normalize_username(name) == user_id),
        None,
    )
    if us_side not in {"p1", "p2"}:
        return None, list(reversed(recent[-8:]))
    opponent_side = "p2" if us_side == "p1" else "p1"
    opponent = battle["players"].get(opponent_side, "Unknown")
    mapped_events = [
        {
            **event,
            "side": "us" if event["side"] == us_side else (
                "opponent" if event["side"] == opponent_side else None
            ),
        }
        for event in battle["events"]
    ]
    return {
        "id": tag,
        "opponent": opponent,
        "turn": battle["turn"],
        "state": "active",
        "us": {
            "active": battle["active"].get(us_side),
            "revealed": list(battle["teams"][us_side].values()),
        },
        "opponentSide": {
            "active": battle["active"].get(opponent_side),
            "revealed": list(battle["teams"][opponent_side].values()),
        },
        "lastAction": battle["lastActionBySide"].get(us_side),
        "events": mapped_events,
    }, list(reversed(recent[-8:]))


def fetch_rating(username: str) -> dict[str, Any] | None:
    request = urllib.request.Request(
        f"https://pokemonshowdown.com/users/{normalize_username(username)}.json?ts={time.time_ns()}",
        headers={"User-Agent": "metagross-dashboard/1.0", "Cache-Control": "no-cache"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10, context=ssl.create_default_context()) as response:
            payload = json.load(response)
    except Exception:
        return None
    rating = (payload.get("ratings") or {}).get(FORMAT) or {}
    return {
        "elo": rating.get("elo"),
        "gxe": rating.get("gxe"),
        "glicko": rating.get("rpr"),
        "glickoDeviation": rating.get("rprd"),
        "rd": rating.get("rprd"),
        "wins": rating.get("w"),
        "losses": rating.get("l"),
    }


def process_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def prior_healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return bool(json.load(response).get("ok"))
    except Exception:
        return False


def latest_issue(run_dir: Path) -> dict[str, str] | None:
    for name in ("client.log", "prior.log"):
        path = run_dir / name
        if not path.exists():
            continue
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 128 * 1024))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            lower = line.lower()
            for category, pattern, message in ISSUE_PATTERNS:
                if pattern in lower:
                    digest = hashlib.sha256(f"{name}:{line}".encode()).hexdigest()[:20]
                    return {"id": digest, "category": category, "message": message}
    return None


def search_summary(path: Path, battle_id: str | None) -> dict[str, int] | None:
    row = read_last_json(path)
    if not row or not battle_id:
        return None
    tag = BATTLE_ID.search(str((row.get("context") or {}).get("tag", "")))
    if not tag or tag.group(0) != battle_id:
        return None
    samples = row.get("samples") or []
    visits = sum(int((sample.get("result") or {}).get("total_visits") or 0) for sample in samples)
    worlds = len(samples)
    return {
        "worlds": worlds,
        "totalVisits": visits,
        "averageVisits": round(visits / worlds) if worlds else 0,
    }


def build_snapshot(run_dir: Path, rating: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    username = str(manifest["ladder"]["username"])
    battle, recent = reduce_protocol(run_dir / "protocol.jsonl", username)
    if battle:
        battle["search"] = search_summary(run_dir / "search.jsonl", battle["id"])
    protocol_path = run_dir / "protocol.jsonl"
    protocol_age = time.time() - protocol_path.stat().st_mtime if protocol_path.exists() else 10**9
    launcher = process_alive(manifest.get("launcher_pid"))
    client = process_alive(manifest.get("client_pid"))
    prior = process_alive(manifest.get("prior_pid")) and prior_healthy(8977)
    issue = latest_issue(run_dir)
    telemetry_fresh = protocol_age < 300
    if not launcher:
        issue = issue or {
            "id": "launcher-offline",
            "category": "process",
            "message": "The ladder launcher is no longer running",
        }
    elif not client:
        issue = issue or {
            "id": "client-offline",
            "category": "process",
            "message": "The Showdown client is no longer running",
        }
    elif not prior:
        issue = issue or {
            "id": "prior-offline",
            "category": "prior",
            "message": "The policy prior server is unhealthy",
        }
    healthy = launcher and client and prior and issue is None and telemetry_fresh
    search = manifest.get("search") or {}
    return {
        "schema": 1,
        "sequence": protocol_path.stat().st_mtime_ns if protocol_path.exists() else time.time_ns(),
        "updatedAt": utc_now(),
        "run": {
            "username": username,
            "profile": manifest.get("profile"),
            "format": manifest["ladder"].get("format"),
            "status": manifest.get("status"),
            "requestedGames": manifest["ladder"].get("games"),
            "search": {
                "timeMs": search.get("search_time_ms"),
                "parallelism": search.get("parallelism"),
                "threads": search.get("threads"),
                "cPuct": search.get("c_puct"),
            },
        },
        "rating": rating or fetch_rating(username),
        "health": {
            "overall": "healthy" if healthy else ("degraded" if launcher else "offline"),
            "client": client,
            "prior": prior,
            "telemetryFresh": telemetry_fresh,
            "issue": issue,
        },
        "battle": battle,
        "recentBattles": recent,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def post_snapshot(url: str, secret: str, snapshot: dict[str, Any]) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(snapshot, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {secret}",
            "Content-Type": "application/json",
            "User-Agent": "metagross-dashboard-publisher/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.status not in {200, 201, 202, 204}:
            raise RuntimeError(f"dashboard ingest returned {response.status}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ingest-url", default=os.environ.get("METAGROSS_DASHBOARD_INGEST_URL"))
    parser.add_argument("--interval", type=float, default=0)
    args = parser.parse_args(argv)
    if not args.output and not args.ingest_url:
        parser.error("set --output or --ingest-url")
    if args.interval < 0:
        parser.error("--interval must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    secret = os.environ.get("METAGROSS_DASHBOARD_SECRET")
    if args.ingest_url and not secret:
        raise RuntimeError("set METAGROSS_DASHBOARD_SECRET for remote ingest")
    rating: dict[str, Any] | None = None
    rating_checked = 0.0
    while True:
        now = time.monotonic()
        if now - rating_checked >= 30:
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            rating = fetch_rating(str(manifest["ladder"]["username"])) or rating
            rating_checked = now
        snapshot = build_snapshot(run_dir, rating)
        if args.output:
            atomic_json(args.output.expanduser().resolve(), snapshot)
        if args.ingest_url:
            try:
                post_snapshot(args.ingest_url, secret or "", snapshot)
            except (urllib.error.URLError, RuntimeError) as exc:
                print(f"dashboard publish warning: {exc}", flush=True)
        if not args.interval:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
