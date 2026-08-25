from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.r1_public_events import R1_SEMANTIC_CONTRACT  # noqa: E402
from scripts.r1_sequential_policy_coverage_probe import (  # noqa: E402
    SequentialCoverageProbeError,
    counter_tape_uniform,
    player_tracker_snapshot,
    probe_captures,
    validate_report,
    write_report,
)
from scripts.teacher_root_bundle import RootCaptureConfig, build_root_capture  # noqa: E402


class StateBinding:
    registry = {}

    @classmethod
    def from_string(cls, value):
        return copy.deepcopy(cls.registry[value])


class Engine:
    State = StateBinding

    def r1_semantic_contract(self):
        return R1_SEMANTIC_CONTRACT

    def root_options(self, *, state):
        return state.first, state.second


class Tracker:
    seen = []

    def __init__(self):
        self.terminal = False
        self.applied = []

    def fork(self):
        return copy.deepcopy(self)

    def public_opponent_registry(self):
        return {}

    def apply_basic_move_class(self, item):
        self.applied.append(item.source_world_indices)
        self.seen.append(tuple(self.applied))
        self.terminal = bool(item.next_states[0].terminal)
        return {
            "illegal_actions": [False, False] + [True] * 11,
            "name_table": {"tackle": 0, "other": 1},
            "terminal": self.terminal,
        }


def state(label, *, first=("tackle",), second=("tackle",), terminal=False):
    return SimpleNamespace(label=label, first=list(first), second=list(second), terminal=terminal)


def snapshot(probs=(1.0, 0.0)):
    full_probs = list(probs) + [0.0] * (13 - len(probs))
    return {
        "schema": 3,
        "tag": "private-battle-sentinel",
        "namespace": "private-worker",
        "username": "private-user",
        "decision_idx": 0,
        "battle_turn": 1,
        "text_tokens": [1],
        "numbers": [0.5],
        "illegal_actions": [False, False] + [True] * 11,
        "mask_fallback": False,
        "mask_fallback_error": None,
        "name_table": {"tackle": 0, "other": 1},
        "probs": full_probs,
        "protocol_prefix": ["|request|private-protocol-sentinel"],
        "player_information_state": {"schema_version": 1, "universal_state": {}, "player_team": [], "opponent_public_team": []},
        "player_observation_history": {"any_opponent_asleep": False, "any_opponent_frozen": False, "revealed_opponents": []},
        "continuation_observation_history": {"any_opponent_asleep": False, "any_opponent_frozen": False, "revealed_opponents": []},
    }


def capture(worlds, *, probs=(1.0, 0.0)):
    identity = {"namespace": "private-worker", "battle_tag": "private-battle-sentinel", "username": "private-user", "decision_idx": 0, "battle_turn": 1}
    return build_root_capture(
        identity=identity,
        player_priors=[("tackle", float(probs[0])), ("other", float(probs[1]))],
        opponent_priors=[("tackle", 1.0)],
        r1_policy_snapshot=snapshot(probs),
        schedules=[worlds],
        config=RootCaptureConfig(Path("private"), 1, 9, "a" * 64),
    )


def projection(states):
    groups = {}
    for index, source in enumerate(states):
        groups.setdefault(source.label, []).append(index)
    classes = []
    for label, indices in groups.items():
        next_states = tuple(state(label + "-next") for _ in indices)
        classes.append(SimpleNamespace(source_world_indices=tuple(indices), next_states=next_states))
    return SimpleNamespace(observation_classes=tuple(classes))


def run(source_capture, *, horizon=1, infer=None):
    return probe_captures(
        [source_capture],
        engine=Engine(),
        tracker_factory=lambda _: Tracker(),
        policy_infer=infer or (lambda _: [1.0, 0.0] + [0.0] * 11),
        horizon=horizon,
        rollouts=1,
        base_seed=17,
        capture_file_sha256="b" * 64,
    )


class SequentialCoverageProbeTests(unittest.TestCase):
    def setUp(self):
        Tracker.seen = []
        StateBinding.registry = {
            "a": state("a"),
            "b": state("b"),
            "zero-bad": state("zero", first=("other",)),
        }

    def test_counter_tape_is_deterministic_and_channel_separated(self):
        first = counter_tape_uniform(1, "a" * 64, 2, 3, 4, "player")
        self.assertEqual(first, counter_tape_uniform(1, "a" * 64, 2, 3, 4, "player"))
        self.assertNotEqual(first, counter_tape_uniform(1, "a" * 64, 2, 3, 4, "chance"))

    def test_player_tracker_snapshot_removes_legacy_opponent_history(self):
        source = snapshot()
        source["player_observation_history"]["revealed_opponents"] = ["public"]
        source["continuation_observation_history"]["revealed_opponents"] = [
            "public",
            "private-contamination",
        ]
        cleaned = player_tracker_snapshot(source)
        self.assertEqual(
            cleaned["continuation_observation_history"],
            source["player_observation_history"],
        )
        self.assertNotEqual(
            cleaned["continuation_observation_history"],
            source["continuation_observation_history"],
        )
        self.assertIn(
            "private-contamination",
            source["continuation_observation_history"]["revealed_opponents"],
        )

    def test_policy_mapping_sampling_and_class_lineage_preserve_mass(self):
        source = capture([("a", 0.25), ("b", 0.75)])
        trackers = []

        def factory(_):
            item = Tracker()
            trackers.append(item)
            return item

        with patch("scripts.r1_sequential_policy_coverage_probe.project_information_set_observations", side_effect=lambda _engine, states, *_args, **_kwargs: projection(states)):
            report = probe_captures(
                [source], engine=Engine(), tracker_factory=factory,
                policy_infer=lambda _: [1.0, 0.0] + [0.0] * 11, horizon=1, rollouts=1,
                base_seed=17, capture_file_sha256="b" * 64,
            )
        self.assertEqual(report["depths"][0]["certified_continuation"]["count"], 2)
        self.assertAlmostEqual(report["depths"][0]["certified_continuation"]["mass"], 1.0)
        self.assertAlmostEqual(report["depths"][1]["horizon_censored"]["mass"], 1.0)
        self.assertEqual(trackers[0].applied, [])

    def test_policy_index_maps_to_canonical_engine_action(self):
        StateBinding.registry["both"] = state(
            "both", first=("tackle", "other"), second=("tackle",)
        )
        selected = []

        def projected(_engine, states, _tracker, player_action, opponent_action, *_args, **_kwargs):
            selected.append((player_action, opponent_action))
            return projection(states)

        with patch(
            "scripts.r1_sequential_policy_coverage_probe.project_information_set_observations",
            side_effect=projected,
        ):
            run(
                capture([("both", 1.0)], probs=(0.0, 1.0)),
                infer=lambda _: [0.0, 1.0] + [0.0] * 11,
            )
        self.assertEqual(selected, [("other", "tackle")])

    def test_complete_set_rejects_zero_weight_world(self):
        source = capture([("a", 1.0), ("zero-bad", 0.0)])
        report = run(source)
        failure = report["depths"][0]["fixed_failures"]["COMMON_LEGAL_SUPPORT_REJECTED"]
        self.assertEqual(failure["count"], 1)
        self.assertEqual(failure["mass"], 1.0)

    def test_positive_policy_support_must_map_to_common_engine_action(self):
        source = capture([("a", 1.0)], probs=(0.0, 1.0))
        report = run(source, infer=lambda _: [0.0, 1.0] + [0.0] * 11)
        self.assertEqual(report["depths"][0]["fixed_failures"]["POLICY_SUPPORT_REJECTED"]["mass"], 1.0)

    def test_tracker_forks_are_isolated(self):
        source = capture([("a", 0.5), ("b", 0.5)])
        with patch("scripts.r1_sequential_policy_coverage_probe.project_information_set_observations", side_effect=lambda _engine, states, *_args, **_kwargs: projection(states)):
            run(source)
        self.assertEqual(Tracker.seen, [((0,),), ((1,),)])

    def test_mass_conservation_validation_and_report_hash(self):
        report = run(capture([("a", 1.0)]))
        validate_report(report)
        unhashed = dict(report)
        claimed = unhashed.pop("report_sha256")
        actual = hashlib.sha256(json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
        self.assertEqual(claimed, actual)
        damaged = copy.deepcopy(report)
        damaged["depths"][0]["entering"]["mass"] = 2.0
        damaged_unhashed = dict(damaged)
        damaged_unhashed.pop("report_sha256")
        damaged["report_sha256"] = hashlib.sha256(json.dumps(damaged_unhashed, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
        with self.assertRaisesRegex(SequentialCoverageProbeError, "mass is not conserved"):
            validate_report(damaged)

    def test_privacy_sentinels_are_absent(self):
        serialized = json.dumps(run(capture([("a", 1.0)])), sort_keys=True)
        for sentinel in ("private-battle-sentinel", "private-user", "private-protocol-sentinel", "tackle", '"probs"'):
            self.assertNotIn(sentinel, serialized)

    def test_write_is_mode_0600_and_exclusive(self):
        report = run(capture([("a", 1.0)]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_report(report, path)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(SequentialCoverageProbeError, "already exists"):
                write_report(report, path)

    def test_invalid_input_creates_no_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with self.assertRaises(SequentialCoverageProbeError):
                report = run(capture([("a", 1.0)]), infer=lambda _: [0.0, 1.0] + [0.0] * 11)
                write_report(report, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
