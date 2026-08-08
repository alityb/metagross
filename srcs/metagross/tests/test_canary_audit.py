from __future__ import annotations

import unittest
from types import SimpleNamespace

from srcs.metagross import canary_audit, run_foul_play


class CanaryAuditTest(unittest.TestCase):
    def test_failed_markdown_does_not_claim_integrity_passed(self):
        report = {
            "result": {
                "wins": 0,
                "losses": 1,
                "normal_wins": 0,
                "normal_losses": 1,
                "inactivity_wins": 0,
                "forfeit_wins": 0,
                "completed_games": 1,
                "incomplete_games": 0,
                "rating": {"start": None, "end": None, "change": None},
            },
            "search": {
                "holdouts": 0,
                "dual_horizon_admissions": 0,
                "safety_corrections": 0,
                "search_latency_ms": {"mean": None, "maximum": None, "calls": 0},
                "holdout_latency_ms": {"mean": None, "maximum": None, "calls": 0},
                "reasons": {},
            },
            "integrity": {
                "passed": False,
                "failures": ["cohort mismatch"],
                "protocol_reconstructed_decisions": 1,
                "selected_actions_matching_outbound_commands": 1,
                "capture_decisions": 1,
            },
            "capture": {"target_games": 1},
            "games": [],
            "verdict": {"performance": "inconclusive"},
        }

        markdown = canary_audit.render_markdown(report)

        self.assertIn("failed one or more integrity checks", markdown)
        self.assertNotIn("passed artifact", markdown)

    def test_recomputes_stored_certificate_aggregates(self):
        raw = {
            "pairs": 64,
            "baseline_sum": 25.6,
            "candidate_sum": 38.4,
            "delta_sum": 12.8,
            "delta_squared_sum": 2.56,
            "catastrophic_count": 0,
            "candidate_better_count": 64,
            "baseline_better_count": 0,
            "equal_count": 0,
            "baseline_terminal_count": 0,
            "candidate_terminal_count": 0,
            "continuation_iterations_executed": 8192,
        }
        certificate = run_foul_play.independent_holdout_certificate(
            [raw] * 32,
            [1 / 32] * 32,
            "candidate",
            "baseline",
            3,
        )
        failures = []

        canary_audit._validate_certificate(certificate, 3, "test", failures)

        self.assertEqual(failures, [])

    def test_protocol_summary_marks_inactivity_win(self):
        protocol = [
            {
                "direction": "received",
                "message": ">battle-test\n|player|p1|bot|1|1000\n"
                "|player|p2|opponent|1|1000\n"
                "|-message|opponent lost due to inactivity.\n|win|bot",
            }
        ]

        games = canary_audit._protocol_games(protocol, "bot")

        self.assertEqual(games["battle-test"]["result"], "win")
        self.assertEqual(games["battle-test"]["opponent"], "opponent")
        self.assertTrue(games["battle-test"]["inactivity"])
        self.assertEqual(games["battle-test"]["end_reason"], "inactivity")

    def test_protocol_summary_marks_forfeit_and_interrupted_games(self):
        protocol = [
            {
                "direction": "received",
                "message": ">battle-forfeit\n|player|p1|bot|1|1000\n"
                "|player|p2|opponent|1|1000\n|forfeit|opponent\n|win|bot",
            },
            {
                "direction": "received",
                "message": ">battle-interrupted\n|player|p1|bot|1|1000\n"
                "|player|p2|other|1|1000\n|turn|2",
            },
            {
                "direction": "received",
                "message": ">battle-forfeit\n|player|p2|",
            },
        ]

        games = canary_audit._protocol_games(protocol, "bot")

        self.assertEqual(games["battle-forfeit"]["end_reason"], "forfeit")
        self.assertEqual(games["battle-forfeit"]["opponent"], "opponent")
        self.assertEqual(games["battle-interrupted"]["result"], "unknown")
        self.assertEqual(games["battle-interrupted"]["end_reason"], "interrupted")

    def test_protocol_summary_ignores_rule_text_that_mentions_forfeit(self):
        protocol = [
            {
                "direction": "received",
                "message": ">battle-normal\n|player|p1|bot|1|1000\n"
                "|player|p2|opponent|1|1000\n"
                "|raw|Timer rules mention that inactivity may forfeit a game.\n|win|bot",
            }
        ]

        games = canary_audit._protocol_games(protocol, "bot")

        self.assertEqual(games["battle-normal"]["end_reason"], "normal")

    def test_outbound_command_resolves_move_tera_and_switch_slot(self):
        battle = SimpleNamespace(
            request_json={
                "side": {
                    "pokemon": [
                        {"details": "Lead, L80"},
                        {"details": "Minior-Blue, L88"},
                    ]
                }
            }
        )

        self.assertEqual(
            canary_audit._command_action("/choose move bugbuzz terastallize", battle),
            "bugbuzz-tera",
        )
        self.assertEqual(
            canary_audit._command_action("/switch 2", battle), "switch miniorblue"
        )
        self.assertEqual(
            canary_audit._command_action("/choose default", battle), "no move"
        )


if __name__ == "__main__":
    unittest.main()
