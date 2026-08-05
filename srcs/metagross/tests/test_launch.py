from __future__ import annotations

import hashlib
import contextlib
import io
import inspect
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

from srcs.metagross import launch
from srcs.metagross import prior_server
from srcs.metagross import run_foul_play


class LaunchTest(unittest.TestCase):
    def test_r1_remains_default(self):
        args = launch.parse_args(["--username", "bot"])
        self.assertEqual(args.profile, "r1")
        self.assertEqual(args.games, 200)
        self.assertEqual(args.search_parallelism, launch.DEFAULT_SEARCH_PARALLELISM)
        self.assertEqual(args.search_threads, launch.DEFAULT_SEARCH_THREADS)
        self.assertEqual(launch.SEARCH_TIME_MS, 500)
        self.assertEqual(launch.CPU_C_PUCT, 2.0)

    def test_search_parallelism_is_configurable_and_positive(self):
        args = launch.parse_args(
            ["--username", "bot", "--search-parallelism", "5", "--search-threads", "1"]
        )
        self.assertEqual(args.search_parallelism, 5)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launch.parse_args(["--username", "bot", "--search-parallelism", "0"])

    def test_remote_mcts_requires_pinned_engine_hash(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launch.parse_args(["--username", "bot", "--remote-mcts"])
        args = launch.parse_args(
            [
                "--username",
                "bot",
                "--remote-mcts",
                "--remote-engine-sha256",
                "a" * 64,
            ]
        )
        self.assertTrue(args.remote_mcts)
        self.assertEqual(args.remote_mcts_app, launch.DEFAULT_REMOTE_MCTS_APP)
        self.assertEqual(args.remote_mcts_function, launch.DEFAULT_REMOTE_MCTS_FUNCTION)

    def test_g4_requires_explicit_three_game_canary(self):
        for extra in ([], ["--games", "3"], ["--confirm-g4-canary"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    launch.parse_args(["--username", "bot", "--profile", "g4", *extra])
        args = launch.parse_args(
            [
                "--username",
                "bot",
                "--profile",
                "g4",
                "--games",
                "3",
                "--confirm-g4-canary",
            ]
        )
        self.assertEqual(args.games, 3)

    def test_g3_requires_explicit_three_game_canary(self):
        for extra in ([], ["--games", "3"], ["--confirm-g3-canary"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    launch.parse_args(["--username", "bot", "--profile", "g3", *extra])
        args = launch.parse_args(
            [
                "--username",
                "bot",
                "--profile",
                "g3",
                "--games",
                "3",
                "--confirm-g3-canary",
            ]
        )
        self.assertEqual(args.games, 3)

    def test_candidate_continuation_is_explicit_and_bounded(self):
        for profile in ("g3", "g4"):
            args = launch.parse_args(
                [
                    "--username",
                    "bot",
                    "--profile",
                    profile,
                    "--games",
                    "25",
                    "--confirm-candidate-continuation",
                ]
            )
            self.assertEqual(args.games, 25)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launch.parse_args(
                    [
                        "--username",
                        "bot",
                        "--profile",
                        profile,
                        "--games",
                        "101",
                        "--confirm-candidate-continuation",
                    ]
                )

    def test_checkpoint_verification_accepts_only_pinned_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            content = b"policy"
            profile = launch.PolicyProfile("run", 7, hashlib.sha256(content).hexdigest())
            path = profile.checkpoint_path(root)
            path.parent.mkdir(parents=True)
            path.write_bytes(content)
            verified_path, actual = launch.verify_checkpoint(profile, root)
            self.assertEqual(verified_path, path.resolve())
            self.assertEqual(actual, profile.sha256)
            path.write_bytes(b"other")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                launch.verify_checkpoint(profile, root)

    def test_prior_server_verifies_exact_checkpoint_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "r1" / "ckpts" / "policy_weights" / "policy_epoch_5.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"r1")
            expected = hashlib.sha256(b"r1").hexdigest()
            verified_path, actual = prior_server.verify_local_checkpoint(
                str(root), "r1", 5, expected
            )
            self.assertEqual(verified_path, path.resolve())
            self.assertEqual(actual, expected)
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                prior_server.verify_local_checkpoint(str(root), "r1", 5, "0" * 64)

    def test_profile_records_are_immutable(self):
        with self.assertRaises(Exception):
            launch.POLICY_PROFILES["g4"].checkpoint = 2
        with self.assertRaises(TypeError):
            launch.POLICY_PROFILES["other"] = launch.POLICY_PROFILES["r1"]

    def test_password_is_not_added_to_child_arguments(self):
        production_source = inspect.getsource(launch) + inspect.getsource(run_foul_play.main)
        self.assertNotIn("--ps-password", production_source)
        self.assertNotIn("sys.argv.extend", production_source)

    def test_experimental_engine_provenance_is_rejected(self):
        provenance = {
            "source_path": "/repo/experimental/engine/pe_v3_learned_priors",
            "editable": True,
            "mcts_parameters": [
                "state",
                "duration_ms",
                "threads",
                "s1_priors",
                "s2_priors",
                "c_puct",
                "seed",
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "source mismatch"):
            run_foul_play.validate_poke_engine_provenance(
                provenance, Path("/repo/srcs/vendor/poke-engine")
            )

    def test_production_environment_removes_experimental_value_model(self):
        source = {
            "KEEP": "yes",
            "METAGROSS_VALUE_MODEL": "model.pt",
            "METAGROSS_LEARNED_VALUE_WEIGHT": "0.5",
        }
        self.assertEqual(launch.production_environment(source), {"KEEP": "yes"})

    def test_mcts_result_payload_preserves_training_targets(self):
        result = SimpleNamespace(
            side_one=[SimpleNamespace(move_choice="tackle", total_score=3.5, visits=7)],
            side_two=[SimpleNamespace(move_choice="protect", total_score=-1, visits=2)],
            total_visits=9,
        )
        payload = run_foul_play._mcts_result_payload(result)
        self.assertEqual(payload["total_visits"], 9)
        self.assertEqual(
            payload["side_one"],
            [{"move_choice": "tackle", "total_score": 3.5, "visits": 7}],
        )

    def test_mcts_result_payload_round_trip(self):
        engine = SimpleNamespace(MctsSideResult=SimpleNamespace, MctsResult=SimpleNamespace)
        payload = {
            "side_one": [{"move_choice": "tackle", "total_score": 3.5, "visits": 7}],
            "side_two": [{"move_choice": "protect", "total_score": -1.0, "visits": 2}],
            "total_visits": 9,
        }
        result = run_foul_play._mcts_result_from_payload(payload, engine)
        self.assertEqual(run_foul_play._mcts_result_payload(result), payload)

    def test_mcts_result_rejects_invalid_numeric_values(self):
        engine = SimpleNamespace(MctsSideResult=SimpleNamespace, MctsResult=SimpleNamespace)
        base = {
            "side_one": [{"move_choice": "tackle", "total_score": 1.0, "visits": 1}],
            "side_two": [],
            "total_visits": 1,
        }
        for field, value in (("visits", -1), ("visits", True), ("total_score", float("nan"))):
            payload = {**base, "side_one": [{**base["side_one"][0], field: value}]}
            with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                run_foul_play._mcts_result_from_payload(payload, engine)

    def test_remote_response_requires_correlation_and_engine_identity(self):
        response = {
            "schema": 1,
            "request_id": "request",
            "index": 3,
            "ok": True,
            "engine": {
                "contract": run_foul_play.REMOTE_ENGINE_CONTRACT,
                "native_sha256": "a" * 64,
            },
            "result": {},
        }
        with mock.patch.dict(
            "os.environ", {"METAGROSS_REMOTE_ENGINE_SHA256": "a" * 64}, clear=False
        ):
            self.assertIs(
                run_foul_play._validate_remote_response(response, "request", 3), response
            )
            with self.assertRaisesRegex(RuntimeError, "correlation"):
                run_foul_play._validate_remote_response(response, "other", 3)
            response["engine"]["native_sha256"] = "b" * 64
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                run_foul_play._validate_remote_response(response, "request", 3)


if __name__ == "__main__":
    unittest.main()
