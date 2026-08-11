from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import poke_env.player  # noqa: F401
except ModuleNotFoundError:
    poke_env = ModuleType("poke_env")
    player = ModuleType("poke_env.player")
    ps_client = ModuleType("poke_env.ps_client")
    server_configuration = ModuleType("poke_env.ps_client.server_configuration")
    player.MaxBasePowerPlayer = type("MaxBasePowerPlayer", (), {})
    player.Player = type("Player", (), {})
    player.RandomPlayer = type("RandomPlayer", (), {})
    ps_client.AccountConfiguration = type("AccountConfiguration", (), {})
    server_configuration.ServerConfiguration = type("ServerConfiguration", (), {})
    sys.modules.setdefault("poke_env", poke_env)
    sys.modules.setdefault("poke_env.player", player)
    sys.modules.setdefault("poke_env.ps_client", ps_client)
    sys.modules.setdefault("poke_env.ps_client.server_configuration", server_configuration)

from eval.run import (  # noqa: E402
    GameResult,
    PairPlan,
    apply_pair_metadata,
    await_operational_gate,
    foul_play_command,
    foul_play_env,
    load_resume_results,
    parse_args,
    progress_snapshot_path,
    resume_config_sha256,
    side_schedule,
    write_pair_registrations,
    write_json,
)


def args() -> SimpleNamespace:
    return SimpleNamespace(
        randbats_belief_pool="/tmp/pool.json",
        randbats_conditional_script="/tmp/conditional.js",
        randbats_conditional_samples=8,
        randbats_conditional_max_teams=100,
        randbats_conditional_max_ms=1000,
        randbats_conditional_timeout_seconds=5,
        format="gen9randombattle",
        prior_server_url="http://localhost:8001",
        cpuct=2.0,
        tauros_kind_model="/tmp/tauros.json",
        tauros_kind_threshold=0.5,
        tauros_kind_min_policy_frac=0.75,
        tauros_kind_allowed_kinds="switch",
        value_shield_margin=0.15,
        value_shield_min_support=0.1,
        value_shield_close_policy_frac=0.75,
        value_shield_log=None,
        learned_value_model=None,
        agent_a_teacher_root_bundle=None,
        agent_b_teacher_root_bundle=None,
        teacher_determinization_schedules=1,
        teacher_determinization_seed=0,
        teacher_manifest_sha256=None,
    )


class SelectiveSharedRootAgentTests(unittest.TestCase):
    def test_production_controller_agents_use_one_harness_with_isolated_modes(self):
        parsed = parse_args(
            [
                "--server",
                "local",
                "--websocket-uri",
                "ws://127.0.0.1:8000/showdown/websocket",
                "--agent-a",
                "production_r1_search_first",
                "--agent-b",
                "production_r1_certified",
                "--agent-a-prior-server-url",
                "http://127.0.0.1:8977",
                "--agent-b-prior-server-url",
                "http://127.0.0.1:8978",
                "--agent-a-require-priors",
                "--agent-b-require-priors",
                "--strict-isolated-priors",
                "--production-run-seed",
                "11" * 32,
            ]
        )
        server = SimpleNamespace(
            websocket_url="ws://127.0.0.1:8000/showdown/websocket"
        )
        with patch.dict(
            os.environ,
            {
                "METAGROSS_CONTROLLER_MODE": "leaked",
                "METAGROSS_REQUIRE_REMOTE_MCTS": "1",
            },
            clear=True,
        ):
            candidate = foul_play_env(
                parsed, "production_r1_search_first", slot="agent_a"
            )
            comparator = foul_play_env(
                parsed, "production_r1_certified", slot="agent_b"
            )
        command = foul_play_command(
            parsed, server, "candidate", "challenge_user", "comparator", "agent_a"
        )

        self.assertIn("srcs.metagross.run_foul_play", command)
        self.assertEqual(candidate["METAGROSS_CONTROLLER_MODE"], "search_first")
        self.assertEqual(comparator["METAGROSS_CONTROLLER_MODE"], "certified")
        self.assertEqual(candidate["METAGROSS_ALLOW_INSECURE_LOOPBACK"], "1")
        self.assertEqual(candidate["METAGROSS_WEBSOCKET_KEEPALIVE"], "1")
        self.assertEqual(
            candidate["METAGROSS_WEBSOCKET_RECEIVE_TIMEOUT_SECONDS"], "120"
        )
        self.assertEqual(candidate["METAGROSS_WEBSOCKET_MAX_RECONNECTS"], "1")
        self.assertEqual(candidate["METAGROSS_REMOTE_MCTS_TIMEOUT_SECONDS"], "10")
        self.assertNotIn("METAGROSS_REQUIRE_REMOTE_MCTS", candidate)
        self.assertNotEqual(
            candidate["METAGROSS_PRIOR_SERVER"], comparator["METAGROSS_PRIOR_SERVER"]
        )
        with patch.dict(
            os.environ,
            {"METAGROSS_WEBSOCKET_RECEIVE_TIMEOUT_SECONDS": "1"},
            clear=True,
        ):
            fault_candidate = foul_play_env(
                parsed, "production_r1_search_first", slot="agent_a"
            )
        self.assertEqual(
            fault_candidate["METAGROSS_WEBSOCKET_RECEIVE_TIMEOUT_SECONDS"], "1"
        )

    def test_production_remote_smoke_has_frozen_rng_and_engine_identity(self):
        parsed = parse_args(
            [
                "--server",
                "local",
                "--websocket-uri",
                "ws://127.0.0.1:8000/showdown/websocket",
                "--agent-a",
                "production_r1_search_first",
                "--production-remote-mcts",
                "--production-run-seed",
                "11" * 32,
            ]
        )

        environment = foul_play_env(
            parsed, "production_r1_search_first", slot="agent_a"
        )

        self.assertEqual(environment["METAGROSS_REQUIRE_REMOTE_MCTS"], "1")
        self.assertEqual(environment["METAGROSS_RUN_SEED"], "11" * 32)
        self.assertEqual(
            environment["METAGROSS_REMOTE_ENGINE_SHA256"],
            parsed.production_remote_engine_sha256,
        )

    def test_production_shared_root_is_one_isolated_root_solver_variable(self):
        parsed = parse_args(
            [
                "--server",
                "local",
                "--websocket-uri",
                "ws://127.0.0.1:8000/showdown/websocket",
                "--agent-a",
                "production_r1_shared_rm_plus",
                "--agent-b",
                "production_r1_search_first",
                "--shared-root-iterations",
                "10000",
                "--shared-root-continuation-iterations",
                "8",
                "--shared-root-prior-strength",
                "1.0",
                "--production-run-seed",
                "22" * 32,
            ]
        )

        candidate = foul_play_env(
            parsed, "production_r1_shared_rm_plus", slot="agent_a"
        )
        baseline = foul_play_env(
            parsed, "production_r1_search_first", slot="agent_b"
        )

        self.assertEqual(candidate["METAGROSS_CONTROLLER_MODE"], "search_first")
        self.assertEqual(candidate["METAGROSS_ROOT_SEARCH_MODE"], "shared_rm_plus")
        self.assertEqual(candidate["METAGROSS_ALLOW_EXPERIMENTAL_SHARED_ROOT"], "1")
        self.assertEqual(candidate["METAGROSS_SHARED_ROOT_ITERATIONS"], "10000")
        self.assertEqual(
            candidate["METAGROSS_SHARED_ROOT_CONTINUATION_ITERATIONS"], "8"
        )
        self.assertEqual(candidate["METAGROSS_SHARED_ROOT_PRIOR_STRENGTH"], "1.0")
        self.assertEqual(candidate["METAGROSS_RUN_SEED"], "22" * 32)
        self.assertEqual(baseline["METAGROSS_RUN_SEED"], "22" * 32)
        self.assertEqual(baseline["METAGROSS_CONTROLLER_MODE"], "search_first")
        self.assertEqual(baseline["METAGROSS_ROOT_SEARCH_MODE"], "independent_mcts")
        self.assertNotIn("METAGROSS_ALLOW_EXPERIMENTAL_SHARED_ROOT", baseline)

    def test_production_independent_ensemble_is_one_isolated_repeat_variable(self):
        parsed = parse_args(
            [
                "--server",
                "local",
                "--websocket-uri",
                "ws://127.0.0.1:8000/showdown/websocket",
                "--agent-a",
                "production_r1_independent_ensemble",
                "--agent-b",
                "production_r1_search_first",
                "--production-remote-mcts",
                "--production-run-seed",
                "33" * 32,
            ]
        )

        candidate = foul_play_env(
            parsed, "production_r1_independent_ensemble", slot="agent_a"
        )
        baseline = foul_play_env(
            parsed, "production_r1_search_first", slot="agent_b"
        )

        self.assertEqual(candidate["METAGROSS_CONTROLLER_MODE"], "search_first")
        self.assertEqual(candidate["METAGROSS_ROOT_SEARCH_MODE"], "independent_ensemble")
        self.assertEqual(candidate["METAGROSS_ALLOW_EXPERIMENTAL_ENSEMBLE"], "1")
        self.assertEqual(candidate["METAGROSS_INDEPENDENT_ENSEMBLE_REPEATS"], "3")
        self.assertEqual(candidate["METAGROSS_REQUIRE_REMOTE_MCTS"], "1")
        self.assertEqual(candidate["METAGROSS_PRIOR_NAMESPACE"], "agent_a")
        self.assertEqual(baseline["METAGROSS_ROOT_SEARCH_MODE"], "independent_mcts")
        self.assertEqual(baseline["METAGROSS_PRIOR_NAMESPACE"], "agent_b")
        self.assertNotIn("METAGROSS_ALLOW_EXPERIMENTAL_ENSEMBLE", baseline)

    def test_default_audit_and_agent_isolation(self):
        with patch.dict(os.environ, {}, clear=True):
            selective = foul_play_env(args(), "foul_play_selective_shared_root_opp")
            baseline = foul_play_env(args(), "foul_play_action_belief_root_priors_opp")

        self.assertEqual(selective["METAGROSS_SELECTIVE_SHARED_ROOT_MODE"], "audit")
        self.assertEqual(selective["METAGROSS_SHARED_ROOT_SEARCH"], "1")
        self.assertEqual(selective["METAGROSS_ACTION_CONDITIONED_BELIEF"], "1")
        self.assertEqual(
            selective["METAGROSS_REQUIRE_SELECTIVE_PAIRED_EVALUATION"], "1"
        )
        self.assertIn("METAGROSS_PRIOR_SERVER", selective)
        self.assertNotIn("METAGROSS_SELECTIVE_SHARED_ROOT_MODE", baseline)
        self.assertNotIn("METAGROSS_SHARED_ROOT_SEARCH", baseline)
        self.assertNotIn("METAGROSS_REQUIRE_SELECTIVE_PAIRED_EVALUATION", baseline)

    def test_override_is_selective_agent_only(self):
        with patch.dict(
            os.environ, {"METAGROSS_SELECTIVE_SHARED_ROOT_MODE": "override"}, clear=True
        ):
            selective = foul_play_env(args(), "foul_play_selective_shared_root_opp")
            shared_only = foul_play_env(args(), "foul_play_shared_root_action_belief_opp")

        self.assertEqual(selective["METAGROSS_SELECTIVE_SHARED_ROOT_MODE"], "override")
        self.assertNotIn("METAGROSS_SELECTIVE_SHARED_ROOT_MODE", shared_only)
        self.assertEqual(shared_only["METAGROSS_SHARED_ROOT_SEARCH"], "1")

    def test_moves_only_belief_variant_is_slot_local(self):
        values = args()
        candidate = foul_play_env(
            values,
            "foul_play_action_belief_moves_only_root_priors_opp",
            slot="agent_a",
        )
        baseline = foul_play_env(
            values,
            "foul_play_action_belief_root_priors_opp",
            slot="agent_b",
        )
        self.assertEqual(candidate["METAGROSS_ACTION_EVIDENCE_MOVES_ONLY"], "1")
        self.assertNotIn("METAGROSS_ACTION_EVIDENCE_MOVES_ONLY", baseline)

    def test_teacher_bundle_is_slot_local_and_clears_parent_environment(self):
        values = args()
        values.agent_a_teacher_root_bundle = "/tmp/teacher-a.jsonl"
        values.teacher_determinization_schedules = 3
        values.teacher_determinization_seed = 17
        values.teacher_manifest_sha256 = "a" * 64
        with patch.dict(
            os.environ,
            {"METAGROSS_TEACHER_ROOT_BUNDLE": "/tmp/leaked.jsonl"},
            clear=True,
        ):
            agent_a = foul_play_env(
                values, "foul_play_root_priors_opp", slot="agent_a"
            )
            agent_b = foul_play_env(
                values, "foul_play_root_priors_opp", slot="agent_b"
            )

        self.assertEqual(
            agent_a["METAGROSS_TEACHER_ROOT_BUNDLE"],
            str(Path("/tmp/teacher-a.jsonl").resolve()),
        )
        self.assertEqual(agent_a["METAGROSS_TEACHER_DETERMINIZATION_SCHEDULES"], "3")
        self.assertEqual(agent_a["METAGROSS_TEACHER_DETERMINIZATION_SEED"], "17")
        self.assertEqual(agent_a["METAGROSS_TEACHER_MANIFEST_SHA256"], "a" * 64)
        self.assertNotIn("METAGROSS_TEACHER_ROOT_BUNDLE", agent_b)
        self.assertNotIn("METAGROSS_TEACHER_DETERMINIZATION_SCHEDULES", agent_b)

    def test_teacher_bundle_cli_requires_manifest_and_strict_priors(self):
        with self.assertRaisesRegex(ValueError, "teacher-manifest-sha256"):
            parse_args(["--agent-a-teacher-root-bundle", "/tmp/a.jsonl"])
        with self.assertRaisesRegex(ValueError, "requires --agent-a-require-priors"):
            parse_args(
                [
                    "--agent-a-teacher-root-bundle",
                    "/tmp/a.jsonl",
                    "--teacher-manifest-sha256",
                    "a" * 64,
                ]
            )
        parsed = parse_args(
            [
                "--agent-a-teacher-root-bundle",
                "/tmp/a.jsonl",
                "--agent-a",
                "foul_play_root_priors_opp",
                "--teacher-manifest-sha256",
                "a" * 64,
                "--teacher-determinization-schedules",
                "3",
                "--agent-a-require-priors",
            ]
        )
        self.assertEqual(parsed.teacher_determinization_schedules, 3)

    def test_strict_isolated_priors_require_distinct_fail_closed_servers(self):
        base = [
            "--strict-isolated-priors",
            "--agent-a",
            "foul_play_selective_shared_root_opp",
            "--agent-b",
            "foul_play_action_belief_root_priors_opp",
            "--agent-a-require-priors",
            "--agent-b-require-priors",
            "--agent-a-prior-server-url",
            "http://127.0.0.1:8977",
            "--agent-b-prior-server-url",
            "http://127.0.0.1:8978",
        ]
        parsed = parse_args(base)
        self.assertTrue(parsed.strict_isolated_priors)
        with self.assertRaisesRegex(ValueError, "distinct prior server URLs"):
            parse_args(base[:-1] + ["http://127.0.0.1:8977/"])

    def test_resume_snapshot_is_config_bound_and_schedule_checked(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "result.json")
            parsed = parse_args(
                ["--n-games", "2", "--json-out", output, "--resume"]
            )
            game = GameResult(
                1,
                parsed.agent_a,
                parsed.agent_b,
                "agent_a",
                "agent_b",
                "agent_a",
                "winner",
                "battle-test-1",
            )
            write_json(
                str(progress_snapshot_path(parsed)),
                {
                    "schema_version": 1,
                    "config_sha256": resume_config_sha256(parsed),
                    "games": [game.__dict__],
                },
            )
            loaded = load_resume_results(parsed, side_schedule(2, True))
            self.assertEqual(loaded, [game])
            self.assertEqual(progress_snapshot_path(parsed).stat().st_mode & 0o777, 0o600)

            changed = parse_args(
                ["--n-games", "4", "--json-out", output, "--resume"]
            )
            with self.assertRaisesRegex(ValueError, "configuration does not match"):
                load_resume_results(changed, side_schedule(4, True))

    def test_mirrored_pair_registration_and_atomic_resume_contract(self):
        import hashlib
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = [
                "--mirrored-pairs",
                "--n-games",
                "2",
                "--agent-a",
                "foul_play",
                "--agent-b",
                "foul_play",
                "--json-out",
                str(root / "result.json"),
                "--log-dir",
                str(root / "logs"),
                "--pair-registration-dir",
                str(root / "registrations"),
                "--fail-fast",
            ]
            parsed = parse_args(argv)
            team_1 = "team-one"
            team_2 = "team-two"
            plan = PairPlan(
                pair_id="pair-1",
                pair_index=1,
                battle_seed="1,2,3,4",
                team_1_seed="5,6,7,8",
                team_2_seed="9,10,11,12",
                team_1_packed=team_1,
                team_2_packed=team_2,
                team_1_sha256=hashlib.sha256(team_1.encode()).hexdigest(),
                team_2_sha256=hashlib.sha256(team_2.encode()).hexdigest(),
            )
            write_pair_registrations(parsed, plan, 1, "MirrorX001", "MirrorY001")
            left = json.loads((root / "registrations" / "mirrorx001.json").read_text())
            right = json.loads((root / "registrations" / "mirrory001.json").read_text())
            self.assertEqual(left["assigned_team_sha256"], plan.team_1_sha256)
            self.assertEqual(right["assigned_team_sha256"], plan.team_2_sha256)
            self.assertEqual(left["battle_seed"], right["battle_seed"])

            games = []
            for index, challenger, acceptor in (
                (1, "agent_a", "agent_b"),
                (2, "agent_b", "agent_a"),
            ):
                game = GameResult(
                    index,
                    parsed.agent_a,
                    parsed.agent_b,
                    challenger,
                    acceptor,
                    "agent_a",
                    "winner",
                    f"battle-{index}",
                )
                apply_pair_metadata(game, plan, index, challenger)
                games.append(game)
            write_json(
                str(progress_snapshot_path(parsed)),
                {
                    "schema_version": 2,
                    "config_sha256": resume_config_sha256(parsed),
                    "games": [game.__dict__ for game in games],
                },
            )
            resumed = parse_args(argv + ["--resume"])
            self.assertEqual(
                load_resume_results(resumed, side_schedule(2, True), [plan]), games
            )

            write_json(
                str(progress_snapshot_path(parsed)),
                {
                    "schema_version": 2,
                    "config_sha256": resume_config_sha256(parsed),
                    "games": [games[0].__dict__],
                },
            )
            with self.assertRaisesRegex(ValueError, "complete pairs"):
                load_resume_results(resumed, side_schedule(2, True), [plan])

    def test_mirrored_pairs_require_fail_closed_local_foul_play(self):
        base = [
            "--mirrored-pairs",
            "--n-games",
            "2",
            "--agent-a",
            "foul_play",
            "--agent-b",
            "foul_play",
            "--json-out",
            "/tmp/result.json",
            "--log-dir",
            "/tmp/logs",
            "--pair-registration-dir",
            "/tmp/registrations",
        ]
        with self.assertRaisesRegex(ValueError, "requires --fail-fast"):
            parse_args(base)
        self.assertTrue(parse_args(base + ["--fail-fast"]).mirrored_pairs)

    def test_operational_gate_requires_fresh_mirrored_configuration(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = [
                "--mirrored-pairs", "--n-games", "4",
                "--agent-a", "foul_play", "--agent-b", "foul_play",
                "--json-out", str(root / "result.json"),
                "--log-dir", str(root / "logs"),
                "--pair-registration-dir", str(root / "registrations"),
                "--fail-fast",
                "--operational-gate-after-pairs", "1",
                "--operational-gate-request", str(root / "gate-request.json"),
                "--operational-gate-approval", str(root / "gate-approval.json"),
                "--operational-gate-review", str(root / "gate-review.json"),
                "--operational-gate-prior-decisions", str(root / "candidate.jsonl"),
                "--operational-gate-prior-decisions", str(root / "comparator.jsonl"),
                "--operational-gate-showdown-launch", str(root / "showdown-launch.json"),
                "--operational-gate-token", "frozen-token",
            ]
            parsed = parse_args(base)
            self.assertEqual(parsed.operational_gate_after_pairs, 1)
            (root / "gate-approval.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "gate path must be fresh"):
                parse_args(base)

    def test_operational_gate_requires_and_binds_complete_evidence(self):
        import asyncio
        import hashlib
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            registrations = root / "registrations"
            logs.mkdir()
            registrations.mkdir()
            for index in range(4):
                tag = f"battle-{1 if index < 2 else 2}"
                (logs / f"p{index}.protocol.jsonl").write_text(
                    json.dumps({"direction": "received", "message": f">{tag}\n|turn|1"}) + "\n",
                    encoding="utf-8",
                )
                (logs / f"p{index}.search.jsonl").write_text(
                    json.dumps({"context": {"tag": tag}}) + "\n", encoding="utf-8"
                )
            prior_paths = [root / "candidate.jsonl", root / "comparator.jsonl"]
            for path in prior_paths:
                path.write_text(
                    "\n".join(json.dumps({"schema": 4, "tag": f"battle-{index}"}) for index in (1, 2)) + "\n",
                    encoding="utf-8",
                )
            showdown_launch = root / "showdown-launch.json"
            showdown_launch.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
            gate_args = SimpleNamespace(
                operational_gate_request=str(root / "request.json"),
                operational_gate_approval=str(root / "approval.json"),
                operational_gate_review=str(root / "review.json"),
                operational_gate_prior_decisions=[str(path) for path in prior_paths],
                operational_gate_showdown_launch=str(showdown_launch),
                operational_gate_token="frozen-token",
                operational_gate_timeout_seconds=2.0,
                pair_registration_dir=str(registrations),
                log_dir=str(logs),
                json_out=str(root / "result.json"),
                agent_a_prior_server_url="http://127.0.0.1:8977",
                agent_b_prior_server_url="http://127.0.0.1:8978",
            )
            write_json(
                str(progress_snapshot_path(gate_args)),
                {"schema_version": 2, "games": []},
            )
            games = [
                GameResult(
                    index, "a", "b", "a", "b", "agent_a", "winner",
                    f"battle-{index}", pair_id="pair-1", pair_index=1, pair_leg=index,
                )
                for index in (1, 2)
            ]

            async def approve():
                request_path = Path(gate_args.operational_gate_request)
                while not request_path.exists():
                    await asyncio.sleep(0.01)
                request_sha = hashlib.sha256(request_path.read_bytes()).hexdigest()
                review_path = Path(gate_args.operational_gate_review)
                review_path.write_text(json.dumps({
                    "status": "authorized",
                    "request_sha256": request_sha,
                    "public_ladder_authorized": False,
                    "checks_passed": [
                        "two_decisive_mirrored_games",
                        "two_unique_battle_tags",
                        "registrations_consumed",
                        "protocol_semantics",
                        "search_semantics",
                        "schema4_prior_telemetry",
                        "processes_healthy",
                        "no_public_ladder",
                    ],
                }), encoding="utf-8")
                review_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
                Path(gate_args.operational_gate_approval).write_text(json.dumps({
                    "approved": True,
                    "request_sha256": request_sha,
                    "review_sha256": review_sha,
                    "token": gate_args.operational_gate_token,
                }), encoding="utf-8")

            async def exercise():
                await asyncio.gather(await_operational_gate(gate_args, games), approve())

            response = SimpleNamespace(read=lambda: b'{"ok":true,"identity":{"pid":1}}')
            with patch("eval.run.urllib.request.urlopen", return_value=response):
                asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
