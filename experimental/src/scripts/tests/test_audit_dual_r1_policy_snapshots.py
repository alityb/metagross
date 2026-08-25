from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.audit_dual_r1_policy_snapshots import (  # noqa: E402
    DualSnapshotAuditError,
    audit_files,
    audit_snapshots,
    normalize_battle_tag,
    normalize_username,
    validate_report,
    write_report,
)


HASHES = ["a" * 64, "b" * 64]


def snapshot(
    role: str,
    username: str,
    opponent: str,
    *,
    decision_idx: int = 0,
    turn: int = 1,
    tag: str = "battle-gen9randombattle-private",
) -> dict:
    return {
        "schema": 3,
        "namespace": "private-worker",
        "tag": tag,
        "username": username,
        "opponent_username": opponent,
        "player_role": role,
        "decision_idx": decision_idx,
        "battle_turn": turn,
        "probs": [1.0] + [0.0] * 12,
        "illegal_actions": [False] + [True] * 12,
        "mask_fallback": False,
        "name_table": {"private-action": 0},
        "protocol_prefix": [
            "|turn|1",
            f"|request|private-{username}",
            "|move|p1a: Private|Tackle|p2a: Private",
        ],
        "player_information_state": {
            "schema_version": 1,
            "private_request": {"side": {"id": role}, "active": [{}]},
        },
    }


def pair() -> list[dict]:
    return [
        snapshot("p1", "Alice Smith", "BOB-JONES"),
        snapshot("p2", "Bob Jones", "alice_smith"),
    ]


def audit(rows) -> dict:
    return audit_snapshots(rows, input_file_sha256=HASHES)


def rehash(report: dict) -> None:
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    report["report_sha256"] = hashlib.sha256(
        json.dumps(
            unhashed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


class DualR1PolicySnapshotAuditTests(unittest.TestCase):
    def test_valid_pair_is_eligible_and_request_lines_are_private(self):
        report = audit(pair())
        self.assertEqual(report["status"], "eligible")
        self.assertEqual(report["counts"]["valid_pairs"], 1)
        self.assertEqual(report["counts"]["certified_one_sided_boundaries"], 0)
        self.assertEqual(sum(report["failures"].values()), 0)
        self.assertFalse(report["r1_continuation_value_allowed"])
        self.assertEqual(report["claim"], "proves_capture_joinability_only")
        serialized = json.dumps(report)
        for private in ("Alice", "Bob", "private-worker", "private-action", "|request|"):
            self.assertNotIn(private, serialized)

    def test_wrapped_snapshot_and_normalization_join(self):
        rows = pair()
        rows[1]["tag"] = "gen9randombattle-private"
        report = audit([{"r1_policy_snapshot": row} for row in rows])
        self.assertEqual(report["status"], "eligible")
        self.assertEqual(normalize_battle_tag("battle-ABC-1"), "abc-1")
        self.assertEqual(normalize_username(" Alice_Smith "), "alicesmith")

    def test_unpaired_boundary(self):
        report = audit([pair()[0]])
        self.assertEqual(report["failures"]["unpaired_boundary"], 1)
        self.assertEqual(report["status"], "ineligible")

    def test_duplicate_role(self):
        rows = pair()
        rows[1]["player_role"] = "p1"
        rows[1]["player_information_state"]["private_request"]["side"]["id"] = "p1"
        report = audit(rows)
        self.assertEqual(report["failures"]["duplicate_role"], 1)

    def test_nonreciprocal_identity(self):
        rows = pair()
        rows[1]["opponent_username"] = "someone-else"
        report = audit(rows)
        self.assertEqual(report["failures"]["nonreciprocal_identity"], 1)

    def test_public_prefix_mismatch(self):
        rows = pair()
        rows[1]["protocol_prefix"][0] = "|turn|2"
        report = audit(rows)
        self.assertEqual(report["failures"]["public_prefix_mismatch"], 1)

    def test_pov_hp_and_transport_noise_are_canonicalized(self):
        rows = pair()
        rows[0]["protocol_prefix"] = [
            "|init|battle",
            "|j|Alice Smith",
            "|switch|p1a: Private|Private, L80|149/301",
            "|-damage|p1a: Private|146/301",
            "|-heal|p1a: Private|300/301",
            "|drag|p2a: Private|Private, L80|300/301",
            "|turn|1",
        ]
        rows[1]["protocol_prefix"] = [
            "|init|battle",
            "|title|private",
            "|switch|p1a: Private|Private, L80|50/100",
            "|-damage|p1a: Private|49/100",
            "|-heal|p1a: Private|99/100",
            "|drag|p2a: Private|Private, L80|99/100",
            "|inactive|private timer text",
            "|turn|1",
        ]
        self.assertEqual(audit(rows)["status"], "eligible")

    def test_forced_switch_singleton_is_certified(self):
        rows = pair()
        forced = snapshot("p1", "Alice Smith", "BOB-JONES", decision_idx=1)
        forced["protocol_prefix"].insert(-1, "|faint|p1a: Private")
        forced["player_information_state"]["private_request"] = {
            "side": {"id": "p1"},
            "forceSwitch": [True],
        }
        report = audit(rows + [forced])
        self.assertEqual(report["status"], "eligible")
        self.assertEqual(report["counts"]["valid_pairs"], 1)
        self.assertEqual(report["counts"]["certified_one_sided_boundaries"], 1)

    def test_invalid_snapshot_and_private_request_role_check(self):
        rows = pair()
        rows[0]["player_information_state"]["private_request"]["side"]["id"] = "p2"
        report = audit(rows)
        self.assertEqual(report["failures"]["invalid_snapshot"], 1)
        self.assertEqual(report["failures"]["unpaired_boundary"], 1)
        self.assertEqual(report["counts"]["input_rows"], 2)

    def test_policy_shape_is_fail_closed(self):
        for mutation in ("mass", "illegal_mass", "fallback", "nan"):
            with self.subTest(mutation=mutation):
                rows = pair()
                if mutation == "mass":
                    rows[0]["probs"][0] = 0.5
                elif mutation == "illegal_mass":
                    rows[0]["probs"] = [0.5, 0.5] + [0.0] * 11
                elif mutation == "fallback":
                    rows[0]["mask_fallback"] = True
                else:
                    rows[0]["probs"][0] = float("nan")
                self.assertEqual(audit(rows)["failures"]["invalid_snapshot"], 1)

    def test_duplicate_identity_rejects_every_colliding_row(self):
        rows = pair()
        duplicate = copy.deepcopy(rows[0])
        duplicate["username"] = "alice_smith"
        rows.append(duplicate)
        report = audit(rows)
        self.assertEqual(report["failures"]["duplicate_identity"], 2)
        self.assertEqual(report["failures"]["unpaired_boundary"], 1)
        self.assertEqual(report["counts"]["candidate_rows"], 1)

    def test_more_than_two_rows_is_unpaired(self):
        rows = pair()
        rows.append(
            snapshot(
                "p1", "Third Player", "Bob Jones", decision_idx=4
            )
        )
        report = audit(rows)
        self.assertEqual(report["failures"]["unpaired_boundary"], 1)

    def test_duplicate_input_artifact_is_rejected(self):
        with self.assertRaisesRegex(DualSnapshotAuditError, "distinct"):
            audit_snapshots(pair(), input_file_sha256=["a" * 64, "a" * 64])

    def test_report_hash_count_conservation_and_privacy_validation(self):
        report = audit(pair())
        validate_report(report)
        unhashed = dict(report)
        claimed = unhashed.pop("report_sha256")
        self.assertEqual(
            claimed,
            hashlib.sha256(
                json.dumps(
                    unhashed,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
            ).hexdigest(),
        )

        damaged = copy.deepcopy(report)
        damaged["counts"]["input_rows"] += 1
        rehash(damaged)
        with self.assertRaisesRegex(DualSnapshotAuditError, "not conserved"):
            validate_report(damaged)

        private = copy.deepcopy(report)
        private["privacy"]["username"] = "private-user"
        rehash(private)
        with self.assertRaises(DualSnapshotAuditError):
            validate_report(private)

    def test_audit_files_counts_invalid_json_and_hashes_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            first_payload = (json.dumps(pair()[0]) + "\n").encode()
            second_payload = (json.dumps(pair()[1]) + "\nnot-json\n").encode()
            first.write_bytes(first_payload)
            second.write_bytes(second_payload)
            report = audit_files([first, second])
        self.assertEqual(report["failures"]["invalid_snapshot"], 1)
        self.assertEqual(report["counts"]["valid_pairs"], 1)
        self.assertEqual(
            report["inputs"]["file_sha256"],
            [hashlib.sha256(first_payload).hexdigest(), hashlib.sha256(second_payload).hexdigest()],
        )

    def test_write_is_exclusive_atomic_mode_0600_and_force_replaces(self):
        report = audit(pair())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_report(report, path)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded, report)
            validate_report(loaded)
            with self.assertRaisesRegex(DualSnapshotAuditError, "already exists"):
                write_report(report, path)
            write_report(report, path, force=True)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
