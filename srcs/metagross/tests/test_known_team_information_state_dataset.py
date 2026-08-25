import unittest

from srcs.metagross.known_team_information_state_dataset import validate_record


def record():
    row = {
        "schema": "metagross-information-state-q-record/v1",
        "identity": {"battle_id": "b", "corpus_uid": "c", "observer": "p1", "decision_idx": 0},
        "split_group": "b",
        "panel": "representative",
        "turn": 1,
        "information_state": {"public_protocol_chunks": ["|turn|1"], "player_request": {}},
        "legal_actions": ["a"],
        "target": {"q_advantage_mean": {"a": 0.0}},
        "provenance": {
            "oracle": False,
            "known_team_teacher_used": False,
            "sampled_hidden_world_in_input": False,
            "sampled_hidden_world_serialized_in_record": False,
        },
    }
    from srcs.metagross.known_team_information_state_dataset import _sha256_json
    row["record_sha256"] = _sha256_json(row)
    return row


class KnownTeamInformationStateDatasetTest(unittest.TestCase):
    def test_valid_minimal_record(self):
        validate_record(record())

    def test_forbidden_hidden_state_key_rejects(self):
        row = record()
        row["information_state"]["sampled_state"] = "secret"
        row.pop("record_sha256")
        from srcs.metagross.known_team_information_state_dataset import _sha256_json
        row["record_sha256"] = _sha256_json(row)
        with self.assertRaises(ValueError):
            validate_record(row)

    def test_truth_provenance_rejects(self):
        row = record()
        row["provenance"]["known_team_teacher_used"] = True
        row.pop("record_sha256")
        from srcs.metagross.known_team_information_state_dataset import _sha256_json
        row["record_sha256"] = _sha256_json(row)
        with self.assertRaises(ValueError):
            validate_record(row)


if __name__ == "__main__":
    unittest.main()
