from __future__ import annotations

import hashlib
import json
import stat
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.experiment_manifest import build_experiment_manifest  # noqa: E402
from scripts.collect_independent_action_values import (  # noqa: E402
    IndependentActionValueError,
    _canonical_json,
    collect_file,
    counter_uniform,
    parse_args,
    validate_action_value_record,
)
from scripts.teacher_root_bundle import (  # noqa: E402
    RootCaptureConfig,
    build_root_capture,
)


GIT = {"commit": "a" * 40, "dirty": True, "dirty_diff_sha256": "b" * 64}
HOST = {
    "hostname": "test-host",
    "platform": "test-platform",
    "system": "TestOS",
    "release": "1",
    "machine": "test-machine",
    "python_implementation": "CPython",
    "python_version": "3.test",
    "python_executable": "/test/python",
}


def frozen_manifest() -> dict:
    return build_experiment_manifest(
        experiment_id="independent-values-test",
        run_id="run-1",
        model_configuration={"policy": "uniform"},
        engine_configuration={"name": "poke-engine"},
        search_configuration={"rollouts": 3},
        belief_configuration={"worlds": 1},
        random_seeds={"collector": 17},
        resources={"workers": 1},
        metrics=["independent-action-values"],
        gates=["development-only"],
        sample_plan={"roots": 1},
        argv=["collect_independent_action_values.py"],
        environment_keys=[],
        environ={},
        git_identity=GIT,
        host_identity=HOST,
        created_at_utc="2026-07-24T00:00:00Z",
    )


def source_capture(manifest_sha256: str, *, schedules: int = 1) -> dict:
    return build_root_capture(
        identity={
            "namespace": "private-worker",
            "battle_tag": "private-battle",
            "username": "private-player",
            "decision_idx": 3,
            "battle_turn": 7,
        },
        player_priors=[("attack-a", 0.5), ("attack-b", 0.5)],
        opponent_priors=[("opp-0", 0.5), ("opp-1", 0.5)],
        r1_policy_snapshot={
            "schema": 3, "tag": "private-battle", "namespace": "private-worker",
            "username": "private-player", "decision_idx": 3, "battle_turn": 7,
            "text_tokens": [1], "numbers": [0.0],
            "illegal_actions": [False, False] + [True] * 11,
            "mask_fallback": False, "mask_fallback_error": None,
            "name_table": {"attack-a": 0, "attack-b": 1},
            "probs": [0.5, 0.5] + [0.0] * 11,
            "protocol_prefix": ["|request|{}"],
            "player_information_state": {"schema_version": 1, "universal_state": {}, "player_team": [], "opponent_public_team": []},
            "player_observation_history": {"any_opponent_asleep": False, "any_opponent_frozen": False, "revealed_opponents": ["opponent"]},
            "continuation_observation_history": {"any_opponent_asleep": False, "any_opponent_frozen": False, "revealed_opponents": ["opponent", "player"]},
        },
        schedules=[[(f"private-state-{index}", 1.0)] for index in range(schedules)],
        config=RootCaptureConfig(Path("unused"), schedules, 29, manifest_sha256),
    )


def panel_capture(capture: dict) -> dict:
    selected = dict(capture)
    selected.pop("capture_sha256")
    selected["sampling"] = {
        "source_capture_sha256": capture["capture_sha256"],
        "stratum": {"legal_action_count": 2},
        "population_count": 4,
        "selected_count": 1,
        "inclusion_probability": 0.25,
        "poststratification_weight": 4.0,
    }
    selected["capture_sha256"] = hashlib.sha256(
        _canonical_json(selected).encode("ascii")
    ).hexdigest()
    return selected


class FakeState:
    def __init__(
        self,
        source: str,
        *,
        decision: int = 0,
        candidate: str | None = None,
        side_one_hp: int = 1,
        side_two_hp: int = 1,
    ):
        self.source = source
        self.decision = decision
        self.candidate = candidate
        self.side_one = SimpleNamespace(pokemon=[SimpleNamespace(hp=side_one_hp)])
        self.side_two = SimpleNamespace(pokemon=[SimpleNamespace(hp=side_two_hp)])

    @classmethod
    def from_string(cls, value: str):
        return cls(value)


class FakeEngine:
    State = FakeState

    def __init__(self, *, terminal_after: int = 2):
        self.terminal_after = terminal_after
        self.steps: list[tuple[str, int, str, str, float]] = []

    @staticmethod
    def root_options(state: FakeState):
        if state.decision == 0:
            return ["attack-b", "attack-a"], ["opp-0", "opp-1"]
        return ["continue-0", "continue-1"], ["opp-0", "opp-1"]

    def step_with_uniform(
        self,
        state: FakeState,
        side_one_action: str,
        side_two_action: str,
        uniform: float,
    ):
        self.steps.append(
            (state.source, state.decision, side_one_action, side_two_action, uniform)
        )
        candidate = side_one_action if state.decision == 0 else state.candidate
        next_decision = state.decision + 1
        terminal = next_decision >= self.terminal_after
        side_one_hp = 0 if terminal and candidate == "attack-b" else 1
        side_two_hp = 0 if terminal and candidate == "attack-a" else 1
        return (
            FakeState(
                state.source,
                decision=next_decision,
                candidate=candidate,
                side_one_hp=side_one_hp,
                side_two_hp=side_two_hp,
            ),
            0,
            1.0,
        )

    @staticmethod
    def terminal_value(state: FakeState):
        side_one_alive = any(mon.hp > 0 for mon in state.side_one.pokemon)
        side_two_alive = any(mon.hp > 0 for mon in state.side_two.pokemon)
        if side_one_alive and not side_two_alive:
            return 1.0
        if side_two_alive and not side_one_alive:
            return -1.0
        return 0.0


def write_inputs(directory: Path, *, selected: bool = False, schedules: int = 1):
    manifest = frozen_manifest()
    capture = source_capture(manifest["manifest_sha256"], schedules=schedules)
    if selected:
        capture = panel_capture(capture)
    manifest_path = directory / "manifest.json"
    input_path = directory / "captures.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    input_path.write_text(json.dumps(capture) + "\n", encoding="utf-8")
    return manifest, capture, input_path, manifest_path


class IndependentActionValueTests(unittest.TestCase):
    def test_every_legal_action_uses_identical_candidate_independent_tape(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, _, input_path, manifest_path = write_inputs(directory)
            output_path = directory / "values.jsonl"
            engine = FakeEngine()
            collect_file(
                input_path,
                manifest_path,
                output_path,
                rollouts=3,
                max_decisions=2,
                base_seed=17,
                engine=engine,
            )
            record = json.loads(output_path.read_text(encoding="ascii"))

        self.assertEqual([result["action"] for result in record["actions"]], ["attack-a", "attack-b"])
        self.assertEqual([result["q"] for result in record["actions"]], [1.0, 0.0])
        self.assertEqual([result["sample_count"] for result in record["actions"]], [3, 3])
        calls_per_action = 3 * 2
        attack_a = engine.steps[:calls_per_action]
        attack_b = engine.steps[calls_per_action:]
        self.assertEqual(len(attack_a), len(attack_b))
        for left, right in zip(attack_a, attack_b):
            self.assertEqual(left[0:2], right[0:2])
            self.assertEqual(left[3:], right[3:])
            if left[1] > 0:
                self.assertEqual(left[2], right[2])
        self.assertTrue(all(left[2] == "attack-a" for left in attack_a[::2]))
        self.assertTrue(all(right[2] == "attack-b" for right in attack_b[::2]))

    def test_counter_uniform_does_not_accept_or_depend_on_candidate_action(self):
        fields = {
            "base_seed": 17,
            "source_capture_identity": {"battle_tag": "b", "decision_idx": 1},
            "source_capture_sha256": "a" * 64,
            "schedule_id": 2,
            "world_index": 3,
            "rollout": 4,
            "decision": 5,
            "channel": "chance",
        }
        self.assertEqual(counter_uniform(**fields), counter_uniform(**fields))
        with self.assertRaises(TypeError):
            counter_uniform(**fields, candidate_action="attack-a")

    def test_equal_schedule_aggregation_and_panel_linkage(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, capture, input_path, manifest_path = write_inputs(
                directory, selected=True, schedules=2
            )
            output_path = directory / "values.jsonl"
            collect_file(
                input_path,
                manifest_path,
                output_path,
                rollouts=2,
                max_decisions=2,
                base_seed=5,
                engine=FakeEngine(),
            )
            record = json.loads(output_path.read_text(encoding="ascii"))
        self.assertEqual(record["schedule_count"], 2)
        self.assertEqual(record["world_count"], 2)
        self.assertEqual([result["sample_count"] for result in record["actions"]], [4, 4])
        self.assertEqual(
            record["source_linkage"]["panel_source_capture_sha256"],
            capture["sampling"]["source_capture_sha256"],
        )
        self.assertEqual(record["configuration"]["schedule_aggregation"], "equal")

    def test_output_is_deterministic_private_hashed_mode_0600_and_exclusive(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest, capture, input_path, manifest_path = write_inputs(directory)
            first = directory / "first.jsonl"
            second = directory / "second.jsonl"
            kwargs = {
                "rollouts": 2,
                "max_decisions": 2,
                "base_seed": 31,
            }
            first_summary = collect_file(
                input_path, manifest_path, first, engine=FakeEngine(), **kwargs
            )
            collect_file(input_path, manifest_path, second, engine=FakeEngine(), **kwargs)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o600)
            with self.assertRaisesRegex(IndependentActionValueError, "already exists"):
                collect_file(input_path, manifest_path, first, engine=FakeEngine(), **kwargs)

            record = json.loads(first.read_text(encoding="ascii"))
            validate_action_value_record(record)
            unhashed = dict(record)
            claimed = unhashed.pop("record_sha256")
            self.assertEqual(
                claimed,
                hashlib.sha256(_canonical_json(unhashed).encode("ascii")).hexdigest(),
            )
            self.assertEqual(first_summary["output_sha256"], hashlib.sha256(first.read_bytes()).hexdigest())
            self.assertEqual(record["source_linkage"]["capture_sha256"], capture["capture_sha256"])
            self.assertEqual(
                record["source_linkage"]["input_manifest_sha256"], manifest["manifest_sha256"]
            )
            serialized = first.read_text(encoding="ascii")
            self.assertNotIn("private-state", serialized)
            self.assertNotIn("sampled_state", serialized)
            self.assertFalse(record["oracle"])
            self.assertFalse(record["r1_continuation_value"])
            self.assertIn("not r1-continuation value", record["estimand"])
            self.assertEqual(record["opponent_policy_id"], "uniform_legal_v1")
            self.assertEqual(record["continuation_policy_id"], "uniform_legal_v1")
            self.assertFalse(record["common_tape"]["candidate_player_action_in_counter"])

    def test_horizon_is_fail_closed_and_installs_no_partial_output(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, _, input_path, manifest_path = write_inputs(directory)
            output_path = directory / "values.jsonl"
            with self.assertRaisesRegex(IndependentActionValueError, "while nonterminal"):
                collect_file(
                    input_path,
                    manifest_path,
                    output_path,
                    rollouts=1,
                    max_decisions=2,
                    base_seed=0,
                    engine=FakeEngine(terminal_after=3),
                )
            self.assertFalse(output_path.exists())

    def test_manifest_and_capture_linkage_are_validated(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest, _, input_path, manifest_path = write_inputs(directory)
            tampered_manifest = dict(manifest)
            tampered_manifest["run_id"] = "tampered"
            manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
            with self.assertRaisesRegex(IndependentActionValueError, "invalid frozen input manifest"):
                collect_file(
                    input_path,
                    manifest_path,
                    directory / "bad-manifest.jsonl",
                    rollouts=1,
                    max_decisions=2,
                    base_seed=0,
                    engine=FakeEngine(),
                )

            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            wrong_capture = source_capture("f" * 64)
            input_path.write_text(json.dumps(wrong_capture) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(IndependentActionValueError, "does not link"):
                collect_file(
                    input_path,
                    manifest_path,
                    directory / "bad-link.jsonl",
                    rollouts=1,
                    max_decisions=2,
                    base_seed=0,
                    engine=FakeEngine(),
                )

    def test_cli_contract(self):
        args = parse_args(
            [
                "--input",
                "captures.jsonl",
                "--input-manifest",
                "manifest.json",
                "--output",
                "values.jsonl",
                "--rollouts",
                "8",
                "--max-decisions",
                "200",
                "--seed",
                "19",
                "--max-records",
                "3",
                "--force",
            ]
        )
        self.assertEqual(args.rollouts, 8)
        self.assertEqual(args.max_decisions, 200)
        self.assertEqual(args.seed, 19)
        self.assertEqual(args.max_records, 3)
        self.assertTrue(args.force)


if __name__ == "__main__":
    unittest.main()
