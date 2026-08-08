from __future__ import annotations

import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from srcs.metagross import launch, world_provenance


class WorldProvenanceTest(unittest.TestCase):
    RUN_SEED = "01" * 32

    def test_seed_derivation_is_deterministic_and_channel_separated(self):
        first = world_provenance.derive_seed(
            self.RUN_SEED, "world-sampling", "battle-gen9-1", 7, "candidate"
        )
        self.assertEqual(
            first,
            world_provenance.derive_seed(
                self.RUN_SEED, "world-sampling", "battle-gen9-1", 7, "candidate"
            ),
        )
        self.assertNotEqual(
            first,
            world_provenance.derive_seed(
                self.RUN_SEED, "holdout", "battle-gen9-1", 7, "candidate"
            ),
        )

    def test_seeded_global_random_restores_the_exact_state(self):
        random.seed(9876)
        before = random.getstate()
        with world_provenance.seeded_global_random(1234):
            sampled = [random.random() for _ in range(3)]
        self.assertEqual(random.getstate(), before)
        with world_provenance.seeded_global_random(1234):
            self.assertEqual([random.random() for _ in range(3)], sampled)

    def test_semantic_digest_ignores_relocation_timing_and_process_metadata(self):
        first = {
            "path": "/first/run",
            "started_at": "2026-08-06T00:00:00Z",
            "client_pid": 12,
            "timing": {"latency_ms": 1.2},
            "decision": {"tag": "battle-1", "choice": "recover"},
        }
        second = {
            "path": "/relocated/run",
            "started_at": "2027-01-01T00:00:00Z",
            "client_pid": 999,
            "timing": {"latency_ms": 800.0},
            "decision": {"choice": "recover", "tag": "battle-1"},
        }
        self.assertEqual(
            world_provenance.semantic_capture_digest(first),
            world_provenance.semantic_capture_digest(second),
        )
        second["decision"]["choice"] = "tackle"
        self.assertNotEqual(
            world_provenance.semantic_capture_digest(first),
            world_provenance.semantic_capture_digest(second),
        )

    def test_exact_utf8_state_hash_detects_tampering(self):
        state = "pikachu\ntera=électric"
        expected = hashlib.sha256(state.encode("utf-8")).hexdigest()
        self.assertEqual(world_provenance.state_sha256(state), expected)
        self.assertTrue(world_provenance.verify_state_sha256(state, expected))
        self.assertFalse(world_provenance.verify_state_sha256(state + " ", expected))

    def test_request_ids_are_deterministic_and_context_specific(self):
        request_id = world_provenance.deterministic_request_id(
            self.RUN_SEED, "battle-1", 3, 2
        )
        self.assertEqual(len(request_id), 32)
        self.assertEqual(
            request_id,
            world_provenance.deterministic_request_id(self.RUN_SEED, "battle-1", 3, 2),
        )
        self.assertNotEqual(
            request_id,
            world_provenance.deterministic_request_id(self.RUN_SEED, "battle-1", 4, 2),
        )

    def test_canonical_jsonl_has_identical_bytes_and_hex_floats(self):
        first = {"z": [0.5, -0.0], "a": "é"}
        second = {"a": "é", "z": [0.5, -0.0]}
        expected = (
            b'{"a":"\xc3\xa9","z":[{"$float":"0x1.0000000000000p-1"},'
            b'{"$float":"-0x0.0p+0"}]}\n'
        )
        self.assertEqual(world_provenance.canonical_jsonl_line(first), expected)
        self.assertEqual(
            world_provenance.canonical_jsonl_line(first),
            world_provenance.canonical_jsonl_line(second),
        )
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                world_provenance.canonical_jsonl_line({"value": value})

    def test_ledger_reader_roundtrips_canonical_rows_and_nested_floats(self):
        rows = [
            {"name": "first", "value": 0.5},
            {"nested": {"values": [-0.0, 1.25]}, "present": True},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            for row in rows:
                world_provenance.append_ledger_row(ledger, row)

            restored = world_provenance.read_ledger(ledger)

        self.assertEqual(restored, rows)
        self.assertEqual(restored[1]["nested"]["values"][0].hex(), "-0x0.0p+0")

    def test_ledger_reader_preserves_float_hex_strings_without_type_collision(self):
        lookalikes = [
            "0x1.0000000000000p+0",
            "0x1.0000000000000p0",
            "0x1.000000000000p+0",
            "0X1.0000000000000p+0",
            "0x1.0000000000000p+00",
            "0x1.0000000000000p+0 trailing",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            world_provenance.append_ledger_row(ledger, {"strings": lookalikes})
            restored = world_provenance.read_ledger(ledger)

        self.assertEqual(restored, [{"strings": lookalikes}])

    def test_ledger_reader_rejects_tampered_or_noncanonical_rows(self):
        invalid_ledgers = {
            "blank line": b"\n",
            "trailing blank line": b'{"a":1}\n\n',
            "malformed JSON": b'{"a":}\n',
            "unsorted keys": b'{"b":1,"a":2}\n',
            "spacing": b'{"a": 1}\n',
            "decimal float": b'{"a":1.0}\n',
            "negative zero integer": b'{"a":-0}\n',
            "escaped ordinary character": b'{"a":"\\u0062"}\n',
            "non-finite value": b'{"a":NaN}\n',
            "missing final newline": b'{"a":1}',
            "top-level array": b'[]\n',
            "invalid float tag": b'{"$float":"ordinary"}\n',
        }
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            for name, content in invalid_ledgers.items():
                with self.subTest(name=name):
                    ledger.write_bytes(content)
                    with self.assertRaises(ValueError):
                        world_provenance.read_ledger(ledger)

    def test_source_and_dataset_hashes_change_with_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "module.py"
            dataset = root / "gen9randombattle.json"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            dataset.write_text('{"pikachu": 1}\n', encoding="utf-8")
            source_before = world_provenance.source_tree_sha256(root)
            dataset_before = world_provenance.randbats_dataset_provenance(dataset)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(
                source_before, world_provenance.source_tree_sha256(root)
            )
            dataset.write_text('{"pikachu": 2}\n', encoding="utf-8")
            self.assertNotEqual(
                dataset_before,
                world_provenance.randbats_dataset_provenance(dataset),
            )

    def test_launcher_manifest_includes_runtime_provenance(self):
        fields = {
            "rng": {
                "scheme": world_provenance.RNG_SCHEME,
                "run_seed": self.RUN_SEED,
            },
            "foul_play": {"commit": "a" * 40, "source_sha256": "b" * 64},
            "metamon": {"commit": "d" * 40, "source_sha256": "e" * 64},
            "metagross": {"commit": "f" * 40, "source_sha256": "1" * 64},
            "randbats_dataset": {"sha256": "c" * 64, "bytes": 10},
            "python_version": "3.11.0",
            "holdout": {"continuation_horizons": [1, 2], "thresholds": {}},
        }
        with mock.patch.object(
            world_provenance, "manifest_provenance", return_value=fields
        ) as provenance:
            manifest_fields = launch.build_world_manifest(self.RUN_SEED)
        self.assertEqual(manifest_fields, fields)
        self.assertEqual(provenance.call_args.args[0], self.RUN_SEED)
        self.assertNotIn("password", json.dumps(manifest_fields).lower())


if __name__ == "__main__":
    unittest.main()
