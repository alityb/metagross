import json

import pytest

from experimental.src.scripts.monitor_cycle23_first_decision_smoke import (
    first_request_window,
    validate_bound_receipts,
)


def receipt(protocol_hash: str, slot: int = 1) -> dict:
    return {
        "schema": "metagross-certified-ability-installation/v1",
        "battle_tag": "battle-test", "observer_role": "p1",
        "protocol_sha256": protocol_hash, "swap": False,
        "installations": [{
            "authority": "rule_implied_form_transition",
            "exact_public_species": "terapagosterastal",
            "installed_base_ability": "terashell",
            "installed_current_ability": "terashell",
            "slot": slot, "update_base": True,
        }],
    }


def protocol(rqid: int = 7) -> list[dict]:
    return [
        {"direction": "received", "time_ns": 10,
         "message": "|player|p1|c23smkx0000001|102|"},
        {"direction": "received", "time_ns": 20,
         "message": "|request|" + json.dumps({"rqid": rqid, "active": [{}]})},
        {"direction": "sent", "time_ns": 40,
         "messages": ["/choose move terastarstorm", str(rqid)]},
        {"direction": "received", "time_ns": 50,
         "message": "|move|p1a: Terapagos|Tera Starstorm|p2a: Target"},
    ]


def test_accepts_unique_exact_form_in_nonzero_slot(tmp_path) -> None:
    directory = tmp_path / "ability-receipts"
    directory.mkdir()
    path = directory / "agenta-123.jsonl"
    rows = [receipt("a" * 64, slot=1) for _ in range(16)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = validate_bound_receipts(tmp_path, "a" * 64)
    assert result["receipt_rows"] == 16
    assert result["slot_values"] == [1]


def test_rejects_receipt_from_later_causal_root(tmp_path) -> None:
    directory = tmp_path / "ability-receipts"
    directory.mkdir()
    path = directory / "agenta-123.jsonl"
    rows = [receipt("a" * 64) for _ in range(16)] + [receipt("b" * 64, slot=0)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="escaped the first causal root"):
        validate_bound_receipts(tmp_path, "a" * 64)


def test_request_decision_send_execution_window_and_rqid() -> None:
    result = first_request_window(
        protocol(), decision_ns=30, username="c23smkx0000001", selected="terastarstorm"
    )
    assert result == {
        "rqid": "7", "request_time_ns": 20, "decision_time_ns": 30,
        "send_time_ns": 40, "public_execution_time_ns": 50,
    }
    bad = protocol(rqid=8)
    bad[2]["messages"][1] = "7"
    with pytest.raises(RuntimeError, match="rqid"):
        first_request_window(bad, 30, "c23smkx0000001", "terastarstorm")


def test_decision_must_be_inside_exact_window() -> None:
    with pytest.raises(RuntimeError, match="outside"):
        first_request_window(protocol(), 45, "c23smkx0000001", "terastarstorm")
