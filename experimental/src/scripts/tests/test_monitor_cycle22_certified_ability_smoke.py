import json

import pytest

from experimental.src.scripts.monitor_cycle22_certified_ability_smoke import (
    validate_ability_receipts,
)


def test_receipt_requires_exact_current_and_base_installation(tmp_path) -> None:
    directory = tmp_path / "ability-receipts"
    directory.mkdir()
    path = directory / "agenta-123.jsonl"
    payload = {
        "schema": "metagross-certified-ability-installation/v1",
        "battle_tag": "battle-test", "observer_role": "p1",
        "protocol_sha256": "a" * 64, "swap": False,
        "installations": [{
            "authority": "rule_implied_form_transition",
            "exact_public_species": "terapagosterastal",
            "installed_base_ability": "terashell",
            "installed_current_ability": "terashell",
            "slot": 0, "update_base": True,
        }],
    }
    path.write_text(json.dumps(payload) + "\n")
    assert validate_ability_receipts(tmp_path)["terapagos_installation_receipts"] == 1
    payload["installations"][0]["installed_base_ability"] = "terashift"
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(RuntimeError, match="wrong Terapagos"):
        validate_ability_receipts(tmp_path)
