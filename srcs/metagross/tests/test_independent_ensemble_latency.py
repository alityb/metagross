import unittest

from srcs.metagross.independent_ensemble_latency import DURATION_MS, _requests


class IndependentEnsembleLatencyTest(unittest.TestCase):
    def test_requests_repeat_states_with_unique_indices(self):
        requests = _requests(["a", "b"], 3)
        self.assertEqual(len(requests), 6)
        self.assertEqual([row["index"] for row in requests], list(range(6)))
        self.assertEqual([row["state"] for row in requests], ["a", "b"] * 3)
        self.assertEqual({row["duration_ms"] for row in requests}, {500})
        self.assertEqual(DURATION_MS, 500)
        early = _requests(["a"] * 32, 2, 250)
        self.assertEqual(len(early), 64)
        self.assertEqual({row["duration_ms"] for row in early}, {250})


if __name__ == "__main__":
    unittest.main()
