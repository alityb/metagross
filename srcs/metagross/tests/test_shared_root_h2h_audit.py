import unittest

from srcs.metagross.shared_root_h2h_audit import _pair_failures, _percentile


class SharedRootH2HAuditTest(unittest.TestCase):
    def test_nearest_rank_percentile(self):
        self.assertIsNone(_percentile([], 0.95))
        self.assertEqual(_percentile([4.0, 1.0, 3.0, 2.0], 0.50), 2.0)
        self.assertEqual(_percentile([4.0, 1.0, 3.0, 2.0], 0.95), 4.0)

    def test_pair_integrity_requires_exact_swapped_legs(self):
        first = {
            "game_index": 1,
            "pair_id": "pair-a",
            "pair_leg": 1,
            "battle_seed": "1,2,3,4",
            "team_1_sha256": "team-1",
            "team_2_sha256": "team-2",
            "agent_a_team_sha256": "team-1",
            "agent_b_team_sha256": "team-2",
        }
        second = {
            **first,
            "game_index": 2,
            "pair_leg": 2,
            "agent_a_team_sha256": "team-2",
            "agent_b_team_sha256": "team-1",
        }
        self.assertEqual(_pair_failures([first, second]), [])

        second["agent_a_team_sha256"] = "team-1"
        self.assertEqual(
            _pair_failures([first, second]),
            ["pair pair-a: team assignments were not swapped"],
        )


if __name__ == "__main__":
    unittest.main()
