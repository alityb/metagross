from __future__ import annotations

import unittest

from srcs.metagross import modal_mcts


class ModalMctsTest(unittest.TestCase):
    def test_cloud_resource_contract(self):
        self.assertEqual(modal_mcts.APP_NAME, "metagross-mcts-r1-p16")
        self.assertEqual(modal_mcts.FUNCTION_NAME, "search_batch")
        self.assertEqual(modal_mcts.MAX_BATCH_SIZE, 16)
        self.assertEqual(modal_mcts.CLOUD_PHYSICAL_CORES, 16.0)
        self.assertEqual(modal_mcts.CLOUD_MEMORY_MIB, 16384)
        self.assertEqual(
            modal_mcts.CLOUD_RESOURCES,
            {
                "physical_cores": 16.0,
                "vcpus_equivalent": 32,
                "memory_mib": 16384,
                "worker_processes": 16,
            },
        )
        self.assertEqual(modal_mcts.REQUEST_SCHEMA, 1)
        self.assertEqual(len(modal_mcts.ENGINE_SOURCE_SHA256), 64)

    def test_prior_validation_preserves_values(self):
        self.assertEqual(
            modal_mcts._validate_priors([["tackle", 0.75], ["protect", 0.25]], "priors"),
            [("tackle", 0.75), ("protect", 0.25)],
        )

    def test_prior_validation_rejects_invalid_values(self):
        for value in ([["tackle"]], [["", 1.0]], [["tackle", float("nan")]]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                modal_mcts._validate_priors(value, "priors")


if __name__ == "__main__":
    unittest.main()
