from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from srcs.metagross import dashboard_publisher


class DashboardPublisherTest(unittest.TestCase):
    def test_parse_condition(self):
        self.assertEqual(dashboard_publisher.parse_condition("172/200 brn"), (86.0, "brn", False))
        self.assertEqual(dashboard_publisher.parse_condition("0 fnt"), (0, None, True))

    def test_protocol_reducer_publishes_only_public_battle_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            protocol = Path(temporary) / "protocol.jsonl"
            messages = [
                ">battle-gen9randombattle-1\n|player|p1|opponent|1|1200\n|player|p2|bot|2|1200\n|start\n|switch|p1a: Ivy|Ogerpon-Cornerstone, L80|100/100\n|switch|p2a: Duck|Golduck, L80|240/240\n|turn|1",
                ">battle-gen9randombattle-1\n|request|{\"side\":{\"pokemon\":[{\"ident\":\"p2: Secret\",\"details\":\"Arceus\"}]}}",
                ">battle-gen9randombattle-1\n|move|p2a: Duck|Surf|p1a: Ivy\n|-damage|p1a: Ivy|42/100\n|turn|2",
            ]
            protocol.write_text(
                "".join(
                    json.dumps({"direction": "received", "message": message, "time_ns": index}) + "\n"
                    for index, message in enumerate(messages, start=1)
                ),
                encoding="utf-8",
            )
            battle, recent = dashboard_publisher.reduce_protocol(protocol, "bot")
            self.assertEqual(recent, [])
            self.assertEqual(battle["opponent"], "opponent")
            self.assertEqual(battle["turn"], 2)
            self.assertEqual(battle["us"]["active"]["species"], "Golduck")
            self.assertEqual(battle["opponentSide"]["active"]["hpPercent"], 42)
            self.assertNotIn("Secret", json.dumps(battle))
            self.assertNotIn("Arceus", json.dumps(battle))

    def test_snapshot_omits_paths_pids_and_private_search_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = {
                "status": "running",
                "profile": "r1",
                "launcher_pid": 1,
                "client_pid": 2,
                "prior_pid": 3,
                "ladder": {"username": "bot", "format": "gen9randombattle", "games": 600},
                "search": {"search_time_ms": 500, "parallelism": 8, "threads": 1, "c_puct": 2.0},
                "checkpoint": {"path": "/private/checkpoint.pt"},
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run_dir / "protocol.jsonl").write_text("", encoding="utf-8")
            with mock.patch.object(dashboard_publisher, "process_alive", return_value=True), mock.patch.object(
                dashboard_publisher, "prior_healthy", return_value=True
            ):
                snapshot = dashboard_publisher.build_snapshot(
                    run_dir,
                    {
                        "elo": 1500,
                        "gxe": 75,
                        "glicko": 1700,
                        "glickoDeviation": 50,
                        "rd": 50,
                        "wins": 10,
                        "losses": 2,
                    },
                )
            encoded = json.dumps(snapshot)
            self.assertNotIn("checkpoint.pt", encoded)
            self.assertNotIn("launcher_pid", encoded)
            self.assertNotIn("player_priors", encoded)
            self.assertEqual(snapshot["run"]["search"]["parallelism"], 8)
            self.assertEqual(snapshot["rating"]["glicko"], 1700)

    def test_latest_issue_sanitizes_network_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "client.log").write_text(
                "websockets.exceptions.ConnectionClosedError: private detail\n",
                encoding="utf-8",
            )
            issue = dashboard_publisher.latest_issue(run_dir)
            self.assertEqual(issue["category"], "network")
            self.assertEqual(issue["message"], "Showdown connection closed unexpectedly")
            self.assertNotIn("private", json.dumps(issue))


if __name__ == "__main__":
    unittest.main()
