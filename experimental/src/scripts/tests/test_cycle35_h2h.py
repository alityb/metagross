from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experimental.src.scripts.cycle33_canonical_h2h import identity
from experimental.src.scripts.watch_cycle35_registrations import validate


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "experimental/runs/search_native_v2_cycle35_fresh_h2h_20260815"


def test_prepare_and_live_config_identity_match_frozen_pair() -> None:
    canonical = RUN / "CANONICAL_H2H_ARGV.json"
    pair_path = RUN / "h2h-result.json.pairs.json"
    pair_sha = hashlib.sha256(pair_path.read_bytes()).hexdigest()
    payload = json.loads(pair_path.read_text())
    assert identity(canonical, "prepare") == identity(canonical, "live", pair_sha)
    assert identity(canonical, "prepare") == payload["config_sha256"]


def test_fresh_pair_manifest_has_ten_unique_unordered_teams() -> None:
    pairs = json.loads((RUN / "h2h-result.json.pairs.json").read_text())["pairs"]
    unordered = {
        tuple(sorted((row["team_1_sha256"], row["team_2_sha256"]))) for row in pairs
    }
    assert len(pairs) == len(unordered) == 10


@pytest.mark.parametrize("suffix,side", [("x1234abc", "p1"), ("y1234abc", "p2")])
def test_cycle35_registration_identity_and_packed_team(suffix: str, side: str) -> None:
    pair = json.loads((RUN / "h2h-result.json.pairs.json").read_text())["pairs"][0]
    payload = {
        "schema_version": 1,
        "pair_id": pair["pair_id"],
        "format": "gen9randombattle",
        "battle_seed": pair["battle_seed"],
        "team_1_sha256": pair["team_1_sha256"],
        "team_2_sha256": pair["team_2_sha256"],
        "leg": 1,
        "assigned_team_sha256": pair["team_1_sha256"],
        "packed_team": pair["team_1_packed"],
    }
    result = validate(payload, pair, "c35h2h" + suffix)
    assert result["side"] == side


def test_cycle35_registration_rejects_retired_cycle33_identity() -> None:
    pair = json.loads((RUN / "h2h-result.json.pairs.json").read_text())["pairs"][0]
    with pytest.raises(RuntimeError, match="unexpected Cycle35"):
        validate({}, pair, "c33h2hx1234abc")
