from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from srcs.metagross import aws_http_mcts
from srcs.metagross.mcts_contract import REQUEST_SCHEMA, validate_request
from srcs.metagross.tests.shared_root_fixture import native_shared_root_result


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
INSTALLER = REPOSITORY_ROOT / "infra" / "aws_mcts" / "install.sh"
PROBE = REPOSITORY_ROOT / "infra" / "aws_mcts" / "probe.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("aws_mcts_probe", PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load AWS MCTS probe")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shared_root_result():
    return native_shared_root_result()


class AwsHttpMctsTest(unittest.TestCase):
    def test_bearer_auth_is_required_and_exact(self):
        self.assertTrue(aws_http_mcts.authorized("Bearer secret", "secret"))
        self.assertFalse(aws_http_mcts.authorized(None, "secret"))
        self.assertFalse(aws_http_mcts.authorized("Bearer other", "secret"))
        self.assertFalse(aws_http_mcts.authorized("secret", "secret"))

    def test_body_and_batch_are_bounded_json(self):
        self.assertEqual(aws_http_mcts.decode_batch(b"[{}]"), [{}])
        for body in (b"", b"not-json", b"{}", b"[]"):
            with self.subTest(body=body), self.assertRaises(aws_http_mcts.RequestError):
                aws_http_mcts.decode_batch(body)
        oversized_batch = json.dumps([{}] * 65).encode()
        with self.assertRaises(aws_http_mcts.RequestError):
            aws_http_mcts.decode_batch(oversized_batch)
        with mock.patch.object(aws_http_mcts, "MAX_REQUEST_BYTES", 3):
            with self.assertRaises(aws_http_mcts.RequestError):
                aws_http_mcts.decode_batch(b"[{}]")

    def test_invalid_world_schema_fails_without_exception_detail(self):
        response = aws_http_mcts._search_one(
            {"schema": 99, "request_id": "request", "index": 0},
            1,
            {"contract": "test"},
            time.monotonic(),
        )
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"], {"kind": "ValueError"})
        self.assertNotIn("result", response)
        self.assertEqual(response["timing"]["batch_size"], 1)

    def test_worker_dispatches_bounded_holdout_without_per_rollout_data(self):
        aggregate = SimpleNamespace(
            pairs=2,
            baseline_sum=0.5,
            candidate_sum=1.0,
            delta_sum=0.5,
            delta_squared_sum=0.25,
            catastrophic_count=0,
            candidate_catastrophic_count=0,
            baseline_catastrophic_count=1,
            candidate_catastrophic_severity_sum=0.0,
            baseline_catastrophic_severity_sum=0.5,
            candidate_better_count=1,
            baseline_better_count=0,
            equal_count=1,
            baseline_terminal_count=0,
            candidate_terminal_count=0,
            baseline_nonterminal_evaluation_delta_sum=1.0,
            candidate_nonterminal_evaluation_delta_sum=-1.0,
            baseline_nonterminal_count=2,
            candidate_nonterminal_count=2,
            continuation_iterations_executed=8,
        )
        engine = SimpleNamespace(
            State=SimpleNamespace(from_string=lambda value: value),
            paired_root_policy_evaluation=mock.Mock(return_value=aggregate),
        )
        request = {
            "schema": aws_http_mcts.REQUEST_SCHEMA,
            "operation": "paired_holdout",
            "request_id": "request",
            "index": 0,
            "state": "state",
            "baseline_action": "tackle",
            "candidate_action": "protect",
            "rollouts": 2,
            "continuation_iterations": 2,
            "continuation_steps": 1,
            "seed": 7,
            "opponent_priors": [["protect", 1.0]],
        }
        with mock.patch.dict("sys.modules", {"poke_engine": engine}):
            response = aws_http_mcts._search_one(
                request, 1, {"contract": "test"}, time.monotonic()
            )
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["pairs"], 2)
        self.assertEqual(len(response["result"]), 20)
        self.assertNotIn("outcomes", response["result"])
        engine.paired_root_policy_evaluation.assert_called_once_with(
            "state", "tackle", "protect", 2, 2, 1, 7, [("protect", 1.0)]
        )

    def test_worker_dispatches_shared_root_as_one_particle_cohort(self):
        engine = SimpleNamespace(
            State=SimpleNamespace(from_string=lambda value: f"parsed:{value}"),
            shared_information_set_root_search=mock.Mock(
                return_value=shared_root_result()
            ),
        )
        request = {
            "schema": aws_http_mcts.REQUEST_SCHEMA,
            "operation": "shared_root",
            "request_id": "shared",
            "index": 0,
            "states": ["a", "b"],
            "particle_weights": [0.25, 0.75],
            "iterations": 100,
            "continuation_iterations": 8,
            "seed": 7,
            "prior_strength": 1.0,
            "s1_prior": [["tackle", 1.0]],
            "s2_priors": [None, [["protect", 1.0]]],
        }
        with mock.patch.dict("sys.modules", {"poke_engine": engine}):
            response = aws_http_mcts._search_one(
                request, 1, {"contract": "test"}, time.monotonic()
            )

        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["diagnostics"]["input_particle_count"], 2)
        engine.shared_information_set_root_search.assert_called_once_with(
            ["parsed:a", "parsed:b"],
            [0.25, 0.75],
            100,
            8,
            7,
            1.0,
            [("tackle", 1.0)],
            [None, [("protect", 1.0)]],
        )

    def test_aws_resource_identity_is_explicit(self):
        resources = aws_http_mcts.aws_resources("c7a.8xlarge")
        self.assertEqual(resources["provider"], "aws_ec2")
        self.assertEqual(resources["instance_type"], "c7a.8xlarge")
        self.assertEqual(resources["worker_processes"], 16)

    def test_service_rejects_short_token(self):
        with self.assertRaisesRegex(RuntimeError, "at least 32"):
            aws_http_mcts.MctsService("short", "c7a.8xlarge", pool=mock.Mock())


class AwsInstallerProbeTest(unittest.TestCase):
    def test_installer_uses_runtime_files_from_repository(self):
        source = INSTALLER.read_text()
        for name in ("__init__.py", "aws_http_mcts.py", "mcts_contract.py"):
            self.assertIn(f'"${{source_repo}}/srcs/metagross/{name}"', source)
        self.assertNotIn('"${script_dir}/aws_http_mcts.py"', source)
        self.assertNotIn('"${script_dir}/mcts_contract.py"', source)
        self.assertNotIn('"${script_dir}/poke_engine.whl"', source)

    def test_installer_fails_preflight_for_missing_repository(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_repository = Path(temporary_directory) / "missing"
            env = {
                **os.environ,
                "METAGROSS_REPO_SOURCE": str(missing_repository),
            }
            result = subprocess.run(
                ["bash", str(INSTALLER)],
                capture_output=True,
                check=False,
                env=env,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"repository input not found at {missing_repository}", result.stderr)

    def test_installer_fails_preflight_for_missing_wheel(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            for relative in (
                "srcs/metagross/__init__.py",
                "srcs/metagross/aws_http_mcts.py",
                "srcs/metagross/mcts_contract.py",
            ):
                path = repository / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            missing_wheel = repository / "missing.whl"
            env = {
                **os.environ,
                "METAGROSS_REPO_SOURCE": str(repository),
                "METAGROSS_POKE_ENGINE_WHEEL": str(missing_wheel),
            }
            result = subprocess.run(
                ["bash", str(INSTALLER)],
                capture_output=True,
                check=False,
                env=env,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"wheel input not found at {missing_wheel}", result.stderr)

    def test_probe_builds_current_valid_search_requests(self):
        requests = load_probe().build_requests("state", worlds=2, duration_ms=250)
        expected_fields = {
            "schema",
            "operation",
            "request_id",
            "index",
            "state",
            "duration_ms",
            "threads",
            "s1_priors",
            "s2_priors",
            "c_puct",
        }
        self.assertEqual(len(requests), 2)
        for index, request in enumerate(requests):
            self.assertEqual(set(request), expected_fields)
            self.assertEqual(request["schema"], REQUEST_SCHEMA)
            self.assertEqual(request["operation"], "search")
            self.assertEqual(request["index"], index)
            self.assertEqual(validate_request(request)["operation"], "search")


if __name__ == "__main__":
    unittest.main()
