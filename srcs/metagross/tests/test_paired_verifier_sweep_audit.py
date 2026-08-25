import unittest

from srcs.metagross.paired_verifier_sweep_audit import _teacher_mass


class PairedVerifierSweepAuditTest(unittest.TestCase):
    def test_teacher_mass_averages_repeats_and_missing_actions(self):
        schedule = {
            "aggregate_treatments": {
                "S-4B": [
                    {"side_one_policy": [{"action": "a", "probability": 0.6}]},
                    {"side_one_policy": [{"action": "a", "probability": 0.4}]},
                ]
            }
        }
        self.assertEqual(_teacher_mass(schedule, "a"), 0.5)
        self.assertEqual(_teacher_mass(schedule, "missing"), 0.0)


if __name__ == "__main__":
    unittest.main()
