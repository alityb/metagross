import hashlib
import json
import unittest

from srcs.metagross.shared_root_prior_enrichment import (
    _condition_prior,
    _validate_source_join,
    validate_report,
)


def _hash_record(record, field):
    record[field] = hashlib.sha256(
        json.dumps(record, separators=(",", ":"), sort_keys=True).encode("ascii")
    ).hexdigest()
    return record


class SharedRootPriorEnrichmentTest(unittest.TestCase):
    def test_condition_prior_uses_only_legal_mass(self):
        effective, matched = _condition_prior(
            [("protect", 0.2), ("switch foo", 0.3), ("unknown", 0.5)],
            ["protect", "switch foo", "tackle"],
        )
        self.assertAlmostEqual(matched, 0.5)
        self.assertEqual(
            effective,
            [["protect", 0.4], ["switch foo", 0.6], ["tackle", 0.0]],
        )

    def test_condition_prior_reports_no_overlap(self):
        self.assertEqual(_condition_prior([("unknown", 1.0)], ["protect"]), (None, 0.0))

    def test_source_join_binds_hash_identity_schedule_and_world(self):
        identity = {"battle_tag": "battle-1", "decision_idx": 0}
        capture = _hash_record(
            {
                "record_type": "teacher_root_capture",
                "identity": identity,
                "schedules": [
                    {
                        "schedule_id": 0,
                        "worlds": [
                            {
                                "world_index": 0,
                                "sampled_state": "state",
                                "state_sha256": "state-hash",
                                "sample_weight": 1.0,
                            }
                        ],
                    }
                ],
            },
            "capture_sha256",
        )
        evaluation = _hash_record(
            {
                "record_type": "teacher_root_evaluation",
                "identity": identity,
                "configuration": {"source_capture_sha256": capture["capture_sha256"]},
                "schedules": [
                    {
                        "schedule_id": 0,
                        "worlds": [
                            {
                                "world_index": 0,
                                "sampled_state": "state",
                                "state_sha256": "state-hash",
                                "sample_weight": 1.0,
                            }
                        ],
                    }
                ],
            },
            "evaluation_sha256",
        )
        _validate_source_join(evaluation, capture)
        evaluation["schedules"][0]["worlds"][0]["sample_weight"] = 0.5
        with self.assertRaisesRegex(ValueError, "content hash"):
            _validate_source_join(evaluation, capture)

    def test_report_validator_rejects_invalid_contract_before_loading_sources(self):
        with self.assertRaisesRegex(ValueError, "invalid contract"):
            validate_report({}, None, None)


if __name__ == "__main__":
    unittest.main()
