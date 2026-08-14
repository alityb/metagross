from __future__ import annotations

import unittest

from experimental.src.scripts import replay_r1_history_canary


class HistoryReplayCanaryTests(unittest.TestCase):
    def test_module_imports_without_starting_server(self):
        self.assertTrue(callable(replay_r1_history_canary.replay_protocol))


if __name__ == "__main__":
    unittest.main()
