from __future__ import annotations

import unittest
import json
from pathlib import Path

from experimental.src.scripts.cycle8_replay_audit import (
    ReplayAuditError,
    _command_action,
    _request_support,
    canonical_public_lines,
    materialize_role,
    model_information_fingerprint,
)


def request(role: str = "p1") -> dict:
    return {
        "active": [{
            "moves": [
                {"id": "thunderbolt", "pp": 23, "maxpp": 24, "disabled": False},
                {"id": "protect", "pp": 15, "maxpp": 16, "disabled": True},
            ],
            "canTerastallize": "Electric",
        }],
        "side": {
            "id": role,
            "pokemon": [
                {"active": True, "condition": "200/200", "details": "Zapdos, L80"},
                {"active": False, "condition": "180/200", "details": "Moltres, L80"},
            ],
        },
    }


class Cycle8ReplayAuditTest(unittest.TestCase):
    def test_frozen_panel_is_label_blind_and_has_fixed_strata(self) -> None:
        root = Path(__file__).resolve().parents[4]
        panel = root / "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/replay-panel-128.jsonl"
        rows = [json.loads(line) for line in panel.read_text().splitlines()]
        self.assertEqual(len(rows), 128)
        self.assertEqual(sum(row["commit_present"] for row in rows), 124)
        self.assertEqual(sum(not row["commit_present"] for row in rows), 4)
        self.assertEqual(len({(row["source"], row["battle_id"]) for row in rows}), 128)
        forbidden = {"winner", "outcome", "log", "inputlog", "actions", "rating"}
        self.assertFalse(any(forbidden.intersection(row) for row in rows))

    def test_transport_and_hp_normalization_is_narrow(self) -> None:
        lines = [
            "|player|p1|Alice|1|1500",
            "|player|p2|Bob|2|1500",
            "|t:|999",
            "|switch|p1a: X|X, L80|199/200",
            "|-damage|p2a: Y|1/3 tox",
            "|-message|Alice forfeited.",
            "|win|Bob",
            "|raw|Alice's rating: 1500 &rarr; <strong>1490</strong>",
            "|player|p1|RematchUser",
        ]
        self.assertEqual(
            canonical_public_lines(lines, inputlog=">forcelose p1"),
            [
                "|player|p1|Alice|1|1500",
                "|player|p2|Bob|2|1500",
                "|switch|p1a: X|X, L80|99/100",
                "|-damage|p2a: Y|34/100 tox",
                "|win|Bob",
            ],
        )

    def test_exact_request_defines_action_table_and_command(self) -> None:
        actions, table, sidecar = _request_support(request())
        self.assertEqual(actions, {
            "thunderbolt", "thunderbolt-tera", "switch moltres",
        })
        self.assertEqual(table, {
            "thunderbolt": 0, "switch moltres": 4, "thunderbolt-tera": 9,
        })
        self.assertEqual(_command_action("move 1 terastallize", request()), "thunderbolt-tera")
        self.assertEqual(_command_action("switch 2", request()), "switch moltres")
        self.assertEqual(sidecar["moves"][1]["disabled"], True)

    def test_null_or_reused_request_and_opposite_pov_fail_closed(self) -> None:
        public = {
            "public_chunks": [{"data": "|player|p1|Alice\n|player|p2|Bob\n|switch|p1a: Zapdos|Zapdos, L80|100/100\n|switch|p2a: Moltres|Moltres, L80|100/100"}],
        }
        base = {
            "role": "p1", "errors": [], "sideupdate_chunks": [],
            "requests": [{"request": request(), "public_chunk_count": 1}],
            "commands": [{"command": "move thunderbolt", "preceding_request_index": None}],
        }
        with self.assertRaisesRegex(ReplayAuditError, "null preceding"):
            materialize_role(
                battle_id="battle-test", role="p1", public_capture=public,
                pov_capture=base, inputlog="",
            )
        base["commands"] = [
            {"command": "move thunderbolt", "preceding_request_index": 0},
            {"command": "move thunderbolt", "preceding_request_index": 0},
        ]
        with self.assertRaisesRegex(ReplayAuditError, "reused"):
            materialize_role(
                battle_id="battle-test", role="p1", public_capture=public,
                pov_capture=base, inputlog="",
            )
        base["commands"] = []
        base["requests"][0]["request"] = request("p2")
        with self.assertRaisesRegex(ReplayAuditError, "Opposite|opposite"):
            materialize_role(
                battle_id="battle-test", role="p1", public_capture=public,
                pov_capture=base, inputlog="",
            )

    def test_team_preview_is_not_silently_mapped(self) -> None:
        with self.assertRaisesRegex(ReplayAuditError, "team-preview"):
            _command_action("team 1, 2, 3, 4, 5, 6", request())

    def test_wait_request_is_nonactionable(self) -> None:
        public = {
            "public_chunks": [{"data": "|player|p1|Alice\n|player|p2|Bob\n|switch|p1a: Zapdos|Zapdos, L80|100/100\n|switch|p2a: Moltres|Moltres, L80|100/100"}],
        }
        waiting = request()
        waiting["wait"] = True
        output = materialize_role(
            battle_id="battle-test", role="p1", public_capture=public,
            pov_capture={
                "role": "p1", "errors": [], "sideupdate_chunks": [],
                "requests": [{"request": waiting, "public_chunk_count": 1}],
                "commands": [],
            },
            inputlog="",
        )
        state = output["states"][0]
        self.assertFalse(state["actionable"])
        self.assertEqual(state["legal_actions"], [])
        self.assertEqual(state["action_table"], {})

    def test_model_fingerprint_excludes_identity_but_binds_mechanics(self) -> None:
        ledger = {
            "schema": "causal-reveal-ledger/v1", "battle_tag": "battle-a",
            "observer_role": "p1", "opponent_role": "p2",
            "opponent_active_species": "moltres", "facts": [],
            "protocol_sha256": "a" * 64,
            "pp_disable_contract": "existing-sampled-move-only;missing-fails-closed",
        }
        first_request = request()
        first_request["side"]["name"] = "Alice"
        first = model_information_fingerprint(
            role="p1", public_event_index=3,
            public_prefix=["|player|p1|Alice|avatar|1500", "|player|p2|Bob|avatar|1600", "|turn|1"],
            private_request=first_request, ledger_payload=ledger,
        )
        second_request = request()
        second_request["side"]["name"] = "CompletelyDifferent"
        identity_changed = model_information_fingerprint(
            role="p1", public_event_index=3,
            public_prefix=["|player|p1|CompletelyDifferent|x|9999", "|player|p2|Other|y|1", "|turn|1"],
            private_request=second_request,
            ledger_payload={**ledger, "battle_tag": "battle-other", "protocol_sha256": "b" * 64},
        )
        self.assertEqual(first, identity_changed)
        mechanic_changed = request()
        mechanic_changed["side"]["name"] = "Alice"
        mechanic_changed["active"][0]["moves"][0]["pp"] = 22
        self.assertNotEqual(first, model_information_fingerprint(
            role="p1", public_event_index=3,
            public_prefix=["|player|p1|Alice|avatar|1500", "|player|p2|Bob|avatar|1600", "|turn|1"],
            private_request=mechanic_changed, ledger_payload=ledger,
        ))
        self.assertNotEqual(first, model_information_fingerprint(
            role="p1", public_event_index=4,
            public_prefix=["|player|p1|Alice|avatar|1500", "|player|p2|Bob|avatar|1600", "|turn|1", "|move|p2a: Moltres|Flamethrower|p1a: Zapdos"],
            private_request=first_request, ledger_payload=ledger,
        ))


if __name__ == "__main__":
    unittest.main()
