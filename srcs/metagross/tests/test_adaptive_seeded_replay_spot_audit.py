import unittest

from srcs.metagross.adaptive_seeded_replay_spot_audit import _rank


class AdaptiveSeededReplaySpotAuditTest(unittest.TestCase):
    def test_rank_is_deterministic_and_coordinate_sensitive(self):
        identity = {"battle_tag": "battle-test", "decision_idx": 1}
        first = _rank("a" * 64, identity, 0, 0, "S-B", 0)
        self.assertEqual(first, _rank("a" * 64, identity, 0, 0, "S-B", 0))
        self.assertNotEqual(first, _rank("a" * 64, identity, 0, 0, "S-B", 1))


if __name__ == "__main__":
    unittest.main()
