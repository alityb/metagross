from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.prior_server import session_key  # noqa: E402


class PriorServerSessionKeyTests(unittest.TestCase):
    def test_empty_namespace_preserves_legacy_tag(self):
        self.assertEqual(session_key("", "battle-gen9randombattle-1"), "battle-gen9randombattle-1")

    def test_namespace_isolates_identical_battle_tags(self):
        tag = "battle-gen9randombattle-1"
        self.assertNotEqual(session_key("worker-1", tag), session_key("worker-2", tag))

    def test_raw_tag_is_unmodified_after_key_creation(self):
        tag = "battle-gen9randombattle-1"
        key = session_key("worker-1", tag)
        self.assertTrue(key.endswith(tag))
        self.assertIn("\0", key)


if __name__ == "__main__":
    unittest.main()
