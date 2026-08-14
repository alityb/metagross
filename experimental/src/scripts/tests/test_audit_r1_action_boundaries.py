from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experimental.src.scripts.audit_r1_action_boundaries import (
    ActionBoundaryAuditError,
    action_index,
    audit_protocols,
)


def request(rqid: int = 2) -> dict:
    return {
        "rqid": rqid,
        "active": [{
            "canTerastallize": "Normal",
            "trapped": False,
            "moves": [
                {"id": "protect", "pp": 16, "disabled": False},
                {"id": "tackle", "pp": 35, "disabled": False},
            ],
        }],
        "side": {"pokemon": [
            {"active": True, "condition": "100/100", "details": "Lead, L80"},
            {"active": False, "condition": "100/100", "details": "Weezing-Galar, L84"},
        ]},
    }


class ActionBoundaryAuditTests(unittest.TestCase):
    def write_protocol(self, rows: list[dict]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "protocol.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_private_request_mapping_covers_tera_and_switch(self):
        req = request()
        self.assertEqual(action_index(req, "tackle-tera"), 10)
        self.assertEqual(action_index(req, "switch weezinggalar"), 4)

    def test_protocol_audit_joins_choice_to_request_not_public_event(self):
        req = request()
        path = self.write_protocol([
            {
                "direction": "received",
                "message": ">battle-test\n|request|" + json.dumps(req),
            },
            {
                "direction": "sent",
                "room": "battle-test",
                "messages": ["/choose move tackle terastallize", "2"],
            },
            {
                "direction": "received",
                "message": ">battle-test\n|move|p1a: Lead|Protect|p2a: Target",
            },
        ])

        report = audit_protocols([path])

        self.assertEqual(report["counts"]["unique_mapped_choices"], 1)
        self.assertEqual(report["counts"]["public_action_events_ignored_as_labels"], 1)
        self.assertEqual(report["action_index_histogram"]["10"], 1)

    def test_protocol_audit_fails_without_correlated_request(self):
        path = self.write_protocol([{
            "direction": "sent",
            "room": "battle-test",
            "messages": ["/choose move tackle", "2"],
        }])
        with self.assertRaisesRegex(ActionBoundaryAuditError, "no correlated"):
            audit_protocols([path])


if __name__ == "__main__":
    unittest.main()
