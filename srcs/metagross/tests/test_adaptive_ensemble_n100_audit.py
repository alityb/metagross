import unittest
from types import SimpleNamespace

from srcs.metagross.adaptive_ensemble_n100_audit import (
    _nearest_rank,
    _opponent_support,
)


class AdaptiveEnsembleN100AuditTest(unittest.TestCase):
    def test_nearest_rank_percentile(self):
        self.assertEqual(_nearest_rank([4.0, 1.0, 3.0, 2.0], 0.50), 2.0)
        self.assertEqual(_nearest_rank([4.0, 1.0, 3.0, 2.0], 0.95), 4.0)

    def test_independent_opponent_support_requires_full_public_roster(self):
        active = SimpleNamespace(
            moves=[SimpleNamespace(name=name) for name in ("Body Press", "Rest", "Moonblast", "Iron Defense")],
            terastallized=False,
        )
        reserve = [
            SimpleNamespace(name=f"Reserve {index}", hp=1 if index < 4 else 0)
            for index in range(5)
        ]
        battle = SimpleNamespace(
            opponent=SimpleNamespace(active=active, reserve=reserve, tera_used=False)
        )
        complete, actions = _opponent_support(battle)
        self.assertTrue(complete)
        self.assertIn("bodypress", actions)
        self.assertIn("bodypress-tera", actions)
        self.assertIn("switch reserve0", actions)
        self.assertNotIn("switch reserve4", actions)
        battle.opponent.reserve.pop()
        self.assertEqual(_opponent_support(battle), (False, set()))


if __name__ == "__main__":
    unittest.main()
