from __future__ import annotations

import hashlib
import json
import threading
import time

from experimental.src.scripts.monitor_cycle21_registered_form_smoke import packed_roster
from experimental.src.scripts.watch_cycle22_registrations import watch


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_packed_roster_preserves_order_and_canonicalizes_public_battle_form() -> None:
    assert packed_roster("Nick|Terapagos-Terastal|||||||]Pikachu||||||||") == [
        "terapagos", "pikachu"
    ]


def test_watcher_attests_two_exact_registrations_and_consumption(tmp_path) -> None:
    pair_dir = tmp_path / "pairs"
    pair_dir.mkdir()
    team1 = "Pikachu||||||||"
    team2 = "Terapagos||ChestoBerry|TeraShift|||||"
    pair = {
        "pair_id": "pair21", "battle_seed": "1,2,3,4",
        "team_1_sha256": digest(team1), "team_2_sha256": digest(team2),
        "team_1_packed": team1, "team_2_packed": team2,
    }
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps({"pairs": [pair]}))
    shared = {
        "schema_version": 1, "pair_id": "pair21", "leg": 1,
        "format": "gen9randombattle", "battle_seed": "1,2,3,4",
        "team_1_sha256": digest(team1), "team_2_sha256": digest(team2),
    }

    def producer() -> None:
        time.sleep(0.05)
        for username, team, role in (
            ("c22smkx001abcd", team1, "p1"),
            ("c22smky001abcd", team2, "p2"),
        ):
            (pair_dir / f"{username}.json").write_text(json.dumps({
                **shared, "packed_team": team,
                "assigned_team_sha256": digest(team),
            }))
        time.sleep(0.5)
        for path in pair_dir.glob("*.json"):
            path.unlink()

    thread = threading.Thread(target=producer)
    thread.start()
    result = watch(pair_dir, manifest, 3)
    thread.join()
    assert result["registrations_observed"] == 2
    assert result["registrations_consumed"] == 2
    assert [row["orientation"] for row in result["registrations"]] == ["p1", "p2"]
