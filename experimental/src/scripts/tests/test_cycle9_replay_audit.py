from __future__ import annotations

import unittest

from experimental.src.scripts.cycle9_replay_audit import (
    HISTORICAL_PROTOCOL_COMMIT,
    ReplayAuditError,
    _request_support,
    canonical_public_lines,
    materialize_role,
)


OTHER_COMMIT = "f8ac14003a5f27e1bdc8d8c59608a773c1cb96e5"


def revival_request() -> dict:
    return {
        "forceSwitch": [True],
        "active": [{"moves": [{"id": "revivalblessing", "pp": 1, "maxpp": 1, "disabled": False}]}],
        "side": {
            "id": "p1", "name": "Player",
            "pokemon": [
                {"active": True, "reviving": True, "condition": "100/100", "details": "Rabsca, L90"},
                {"active": False, "reviving": False, "condition": "0 fnt", "details": "Persian-Alola, L86"},
                {"active": False, "reviving": False, "condition": "50/100", "details": "Chesnaught, L84"},
                {"active": False, "reviving": False, "condition": "0 fnt", "details": "Moltres, L80"},
            ],
        },
    }


class Cycle9ReplayRepairTest(unittest.TestCase):
    def test_server_transport_repairs_preserve_mechanics(self) -> None:
        lines = [
            "|player|p1|Alice|avatar|1500",
            "|player|p2|Bob|avatar|1600",
            "|badge|p1|silver|gen9randombattle|30-10",
            "|raw|<div class=\"broadcast-blue\"><strong>This battle is required to be public due to a player having a name starting with 'medal'.</div>",
            "|move|p1a: Zapdos|Thunderbolt|p2a: Moltres",
            "|-damage|p2a: Moltres|50/100",
            "|player|p1|",
            "|-message|Alice lost due to inactivity.",
            "|-message|A real battle mechanic annotation",
            "|win|Bob",
        ]
        output = canonical_public_lines(
            lines, inputlog=">forcelose p1", showdown_commit=OTHER_COMMIT,
        )
        self.assertEqual(output, [
            "|player|p1|Alice|avatar|1500",
            "|player|p2|Bob|avatar|1600",
            "|move|p1a: Zapdos|Thunderbolt|p2a: Moltres",
            "|-damage|p2a: Moltres|50/100",
            "|-message|A real battle mechanic annotation",
            "|win|Bob",
        ])
        self.assertTrue(any(line.startswith("|move|") for line in output))
        self.assertTrue(any(line.startswith("|-damage|") for line in output))
        self.assertEqual(output[-1], "|win|Bob")

    def test_unrelated_raw_message_is_preserved(self) -> None:
        line = "|raw|A mechanics-relevant custom message"
        self.assertEqual(canonical_public_lines(
            [line], inputlog="", showdown_commit=OTHER_COMMIT,
        ), [line])

    def test_inactivity_message_requires_matching_forcelose(self) -> None:
        lines = [
            "|player|p1|Alice", "|player|p2|Bob",
            "|-message|Alice lost due to inactivity.", "|win|Bob",
        ]
        output = canonical_public_lines(
            lines, inputlog=">forcelose p2", showdown_commit=OTHER_COMMIT,
        )
        self.assertIn("|-message|Alice lost due to inactivity.", output)

    def test_historical_aliases_are_exactly_version_scoped(self) -> None:
        lines = [
            "|-damage|p1a: X|0 fnt",
            "|-weather|Rain Dance|[upkeep]",
            "|faint|p1a: X",
        ]
        self.assertEqual(canonical_public_lines(
            lines, inputlog="", showdown_commit=HISTORICAL_PROTOCOL_COMMIT,
        ), [
            "|-damage|p1a: X|0",
            "|-weather|RainDance|[upkeep]",
            "|faint|p1a: X",
        ])
        self.assertEqual(canonical_public_lines(
            lines, inputlog="", showdown_commit=OTHER_COMMIT,
        ), lines)
        unrelated = ["|-weather|Sunny Day", "|-damage|p1a: X|1 fnt"]
        self.assertEqual(canonical_public_lines(
            unrelated, inputlog="", showdown_commit=HISTORICAL_PROTOCOL_COMMIT,
        ), unrelated)

    def test_revival_prompt_maps_only_fainted_targets(self) -> None:
        actions, table, sidecar, semantics = _request_support(revival_request())
        self.assertEqual(actions, {"switch persianalola", "switch moltres"})
        self.assertEqual(table, {"switch moltres": 4, "switch persianalola": 5})
        self.assertEqual(set(semantics.values()), {"revival_target"})
        self.assertTrue(sidecar["revival_prompt"])
        self.assertNotIn("switch chesnaught", actions)
        self.assertFalse(any(action.endswith("-tera") for action in actions))

    def test_revival_prompt_must_be_forced_and_have_target(self) -> None:
        request = revival_request()
        request["forceSwitch"] = [False]
        with self.assertRaisesRegex(ReplayAuditError, "not forced"):
            _request_support(request)
        request = revival_request()
        for row in request["side"]["pokemon"]:
            if not row["active"]:
                row["condition"] = "50/100"
        with self.assertRaisesRegex(ReplayAuditError, "target count"):
            _request_support(request)

    def test_ordinary_request_retains_switch_semantics(self) -> None:
        request = revival_request()
        request["side"]["pokemon"][0]["reviving"] = False
        request["forceSwitch"] = [True]
        actions, _, sidecar, semantics = _request_support(request)
        self.assertIn("switch chesnaught", actions)
        self.assertFalse(sidecar["revival_prompt"])
        self.assertEqual(set(semantics.values()), {"switch"})

    def test_command_boundary_includes_only_prior_public_chunks(self) -> None:
        request = {
            "active": [{"moves": [{"id": "thunderbolt", "pp": 24, "maxpp": 24, "disabled": False}]}],
            "side": {
                "id": "p1", "name": "Alice",
                "pokemon": [
                    {"active": True, "condition": "100/100", "details": "Zapdos, L80"},
                    {"active": False, "condition": "100/100", "details": "Moltres, L80"},
                ],
            },
        }
        public = {"public_chunks": [
            {"input_index": 2, "data": "|gametype|singles"},
            {"input_index": 3, "data": "|player|p1|Alice"},
            {"input_index": 4, "data": "|player|p2|Bob\n|switch|p1a: Zapdos|Zapdos, L80|100/100\n|switch|p2a: Moltres|Moltres, L80|100/100\n|turn|1"},
            {"input_index": 5, "data": "|move|p2a: Moltres|Flamethrower|p1a: Zapdos"},
            {"input_index": 6, "data": "|move|p1a: Zapdos|Thunderbolt|p2a: Moltres"},
        ]}
        output = materialize_role(
            battle_id="battle-test", role="p1", public_capture=public,
            pov_capture={
                "role": "p1", "errors": [], "sideupdate_chunks": [],
                "requests": [{"request": request, "public_chunk_count": 2}],
                "commands": [{
                    "command": "move thunderbolt", "input_index": 6,
                    "preceding_request_index": 0,
                }],
            },
            inputlog="", showdown_commit=OTHER_COMMIT,
        )
        prefix = output["states"][0]["public_prefix"]
        self.assertIn("|switch|p2a: Moltres|Moltres, L80|100/100", prefix)
        self.assertIn("|move|p2a: Moltres|Flamethrower|p1a: Zapdos", prefix)
        self.assertNotIn("|move|p1a: Zapdos|Thunderbolt|p2a: Moltres", prefix)
        self.assertIn(
            "flamethrower",
            output["states"][0]["typed_reveal_ledger"]["facts"][0]["moves"],
        )
        without_precommand_move = {
            "public_chunks": [chunk for chunk in public["public_chunks"] if chunk["input_index"] != 5]
        }
        baseline = materialize_role(
            battle_id="battle-test", role="p1", public_capture=without_precommand_move,
            pov_capture={
                "role": "p1", "errors": [], "sideupdate_chunks": [],
                "requests": [{"request": request, "public_chunk_count": 2}],
                "commands": [{
                    "command": "move thunderbolt", "input_index": 6,
                    "preceding_request_index": 0,
                }],
            },
            inputlog="", showdown_commit=OTHER_COMMIT,
        )
        self.assertNotEqual(
            output["states"][0]["model_information_fingerprint_sha256"],
            baseline["states"][0]["model_information_fingerprint_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
