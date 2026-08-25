import hashlib

import pytest

from experimental.src.scripts.watch_cycle23_registrations import validate_registration


def test_cycle23_registration_binds_team_seed_and_orientation() -> None:
    team1, team2 = "Pikachu||||tackle", "Terapagos||||terastarstorm"
    pair = {
        "pair_id": "pair23", "battle_seed": "1,2,3,4",
        "team_1_packed": team1, "team_2_packed": team2,
        "team_1_sha256": hashlib.sha256(team1.encode()).hexdigest(),
        "team_2_sha256": hashlib.sha256(team2.encode()).hexdigest(),
    }
    payload = {
        "schema_version": 1, "pair_id": "pair23", "leg": 1,
        "format": "gen9randombattle", "battle_seed": "1,2,3,4",
        "team_1_sha256": pair["team_1_sha256"],
        "team_2_sha256": pair["team_2_sha256"],
        "assigned_team_sha256": pair["team_2_sha256"], "packed_team": team2,
    }
    assert validate_registration(payload, pair, "c23smky0000001") == "p2"
    payload["battle_seed"] = "4,3,2,1"
    with pytest.raises(RuntimeError, match="identity"):
        validate_registration(payload, pair, "c23smky0000001")
