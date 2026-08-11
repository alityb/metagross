from __future__ import annotations

import hashlib
import random
import unittest

from srcs.metagross import run_foul_play, shadow_replay


class ShadowReplayTest(unittest.TestCase):
    @staticmethod
    def v5_result(pairs=64):
        return {
            "pairs": pairs,
            "baseline_sum": 0.4 * pairs,
            "candidate_sum": 0.6 * pairs,
            "delta_sum": 0.2 * pairs,
            "delta_squared_sum": 0.04 * pairs,
            "catastrophic_count": 0,
            "candidate_catastrophic_count": 0,
            "baseline_catastrophic_count": 0,
            "candidate_catastrophic_severity_sum": 0.0,
            "baseline_catastrophic_severity_sum": 0.0,
            "candidate_better_count": pairs,
            "baseline_better_count": 0,
            "equal_count": 0,
            "baseline_terminal_count": 0,
            "candidate_terminal_count": 0,
            "baseline_nonterminal_evaluation_delta_sum": 0.0,
            "candidate_nonterminal_evaluation_delta_sum": 0.2 * pairs,
            "baseline_nonterminal_count": pairs,
            "candidate_nonterminal_count": pairs,
            "continuation_iterations_executed": 2 * pairs * 64,
        }

    def test_canary_capture_has_exact_complete_joins(self):
        protocol, searches, metadata = shadow_replay.load_capture(
            shadow_replay.DEFAULT_CAPTURE
        )
        decisions = metadata["decision_rows"]

        self.assertEqual(len(protocol), 387)
        self.assertEqual(len(decisions), 77)
        self.assertEqual(set(decisions), set(searches))
        self.assertEqual(len(metadata["battle_tags"]), 3)
        self.assertEqual(
            metadata["capture_digest"],
            "1769ea5485f57061a31ba01c918839a2c0d3f5c9ca544cf82473318c1390fb06",
        )

    def test_terminal_message_ignores_bundled_request(self):
        protocol, searches, metadata = shadow_replay.load_capture(
            shadow_replay.DEFAULT_CAPTURE
        )
        request_message = next(
            row["message"]
            for row in reversed(protocol)
            if row.get("direction") == "received"
            and "\n|request|" in row.get("message", "")
        )
        room, request_line = request_message.split("\n", 1)
        request_line = next(
            line for line in request_line.splitlines() if line.startswith("|request|")
        )
        terminal_protocol = protocol + [
            {
                "direction": "received",
                "message": f"{room}\n{request_line}\n|win|winner",
            }
        ]

        battles = shadow_replay.reconstruct_battles(
            terminal_protocol,
            searches,
            metadata["manifest"]["ladder"]["username"],
        )

        self.assertEqual(set(battles), set(searches))

    def test_replay_seeds_exclude_action_identity(self):
        arguments = ("digest", "fresh-worlds", "battle", 3)
        self.assertEqual(
            shadow_replay._seed(*arguments), shadow_replay._seed(*arguments)
        )
        self.assertNotEqual(
            shadow_replay._seed(*arguments),
            shadow_replay._seed("digest", "holdout-tape", "battle", 3),
        )
        self.assertNotEqual(
            shadow_replay._seed(*arguments), shadow_replay._seed(*arguments, 1)
        )

    def test_fresh_worlds_are_isolated_from_process_global_random(self):
        protocol, searches, metadata = shadow_replay.load_capture(
            shadow_replay.DEFAULT_CAPTURE
        )
        battles = shadow_replay.reconstruct_battles(
            protocol, searches, metadata["manifest"]["ladder"]["username"]
        )
        battle = battles[sorted(battles)[0]]
        random.seed(9876)
        before = random.getstate()

        first = shadow_replay._fresh_worlds(battle, 2, 1234)
        random.random()
        second = shadow_replay._fresh_worlds(battle, 2, 1234)

        self.assertEqual(first, second)
        random.setstate(before)
        shadow_replay._fresh_worlds(battle, 2, 1234)
        self.assertEqual(random.getstate(), before)

    def test_fresh_worlds_use_decision_time_revealed_set_snapshot(self):
        protocol, searches, metadata = shadow_replay.load_capture(
            shadow_replay.DEFAULT_CAPTURE
        )
        battles = shadow_replay.reconstruct_battles(
            protocol, searches, metadata["manifest"]["ladder"]["username"]
        )
        battle = battles[sorted(battles)[0]]
        from data.pkmn_sets import RandomBattleTeamDatasets

        active_name = battle.opponent.active.name
        original = RandomBattleTeamDatasets.pkmn_sets[active_name]
        expected = shadow_replay._fresh_worlds(battle, 2, 4321)
        try:
            RandomBattleTeamDatasets.pkmn_sets[active_name] = []
            actual = shadow_replay._fresh_worlds(battle, 2, 4321)
        finally:
            RandomBattleTeamDatasets.pkmn_sets[active_name] = original

        self.assertEqual(actual, expected)

    def test_horizon_comparison_keeps_only_stable_causal_admissions(self):
        engine = {
            "contract": "v4",
            "source_sha256": "source",
            "native_sha256": "native",
        }
        common = {"capture": {"capture_digest": "capture"}, "engine": engine}
        first = {
            **common,
            "evaluation": {"continuation_steps": 1},
            "holdout_required_admitted": [
                {"tag": "battle", "decision_idx": 1, "candidate": "stable"},
                {"tag": "battle", "decision_idx": 2, "candidate": "shallow"},
            ],
        }
        second = {
            **common,
            "evaluation": {"continuation_steps": 2},
            "holdout_required_admitted": [
                {"tag": "battle", "decision_idx": 1, "candidate": "stable"},
                {"tag": "battle", "decision_idx": 3, "candidate": "deep"},
            ],
        }

        comparison = shadow_replay.compare_horizon_reports(first, second)

        self.assertEqual(comparison["counts"]["stable_required_admissions"], 1)
        self.assertEqual(
            comparison["stable_required_admitted"][0]["candidate"], "stable"
        )

    def test_replay_modes_reject_conflicting_offline_sources(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            shadow_replay.run_replay(
                shadow_replay.DEFAULT_CAPTURE,
                dry_run=True,
                captured_holdout=True,
            )
        with self.assertRaisesRegex(ValueError, "live remote replay"):
            shadow_replay.run_replay(
                shadow_replay.DEFAULT_CAPTURE,
                dry_run=True,
                recursive_shadow=True,
            )

    def test_recursive_shadow_remote_call_uses_isolated_channels(self):
        captured = []

        class Function:
            @staticmethod
            def remote(requests):
                captured.extend(requests)
                return [
                    {
                        "schema": run_foul_play.REMOTE_MCTS_SCHEMA,
                        "request_id": request["request_id"],
                        "index": request["index"],
                        "ok": True,
                        "engine": {
                            "contract": shadow_replay.ENGINE_CONTRACT,
                            "source_sha256": shadow_replay.ENGINE_SOURCE_SHA256,
                            "native_sha256": "a" * 64,
                        },
                        "result": self.v5_result(),
                    }
                    for request in requests
                ]

        results, _elapsed = shadow_replay._remote_holdout(
            ["state"],
            "recover",
            "earthquake",
            "capture",
            "battle-test",
            4,
            Function(),
            "a" * 64,
            2,
            run_seed="04" * 32,
            candidate_rank=1,
            request_channel="recursive-shadow-request",
            tape_channel="recursive-shadow-tape",
        )

        self.assertEqual(results, [self.v5_result()])
        self.assertEqual(
            captured[0]["request_id"],
            shadow_replay.deterministic_request_id(
                "04" * 32,
                "battle-test",
                4,
                "1:2:0",
                channel="recursive-shadow-request",
            ),
        )
        self.assertEqual(
            captured[0]["seed"],
            shadow_replay.derive_seed(
                "04" * 32,
                "recursive-shadow-tape",
                "battle-test",
                4,
                0,
            ),
        )

    def test_v5_panel_recomputes_raw_aggregates_and_rejects_tampering(self):
        hashes = [
            hashlib.sha256(f"state-{index}".encode()).hexdigest()
            for index in range(16)
        ]
        look = run_foul_play.robust_holdout_certificate(
            [self.v5_result() for _ in hashes],
            [1.0] * len(hashes),
            hashes,
            hashes,
            "candidate",
            "baseline",
            0,
            1,
            0,
        )
        second = run_foul_play.robust_holdout_certificate(
            [self.v5_result() for _ in hashes],
            [1.0] * len(hashes),
            hashes,
            hashes,
            "candidate",
            "baseline",
            0,
            1,
            1,
        )
        combined = run_foul_play.combined_robust_holdout_certificate(
            {1: look, 2: second}
        )
        panel = {
            "complete": True,
            "alpha_sequence_index": 0,
            "candidate_panel": [{"rank": 1, "action": "candidate"}],
            "certification_cohort": {
                "state_hashes": hashes,
                "cluster_hashes": hashes,
                "weights": [1.0] * len(hashes),
            },
            "certificates_by_action": {"candidate": combined},
            "qualified_actions": ["candidate"],
        }

        self.assertEqual(
            shadow_replay._recompute_captured_v5_panel(panel),
            {"candidate": combined},
        )
        look["raw_results"][0]["candidate_sum"] -= 1.0
        with self.assertRaises(ValueError):
            shadow_replay._recompute_captured_v5_panel(panel)


if __name__ == "__main__":
    unittest.main()
