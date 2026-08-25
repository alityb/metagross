from __future__ import annotations

import copy
import hashlib
import json
import stat
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.r1_dual_tracker_parity_probe import (  # noqa: E402
    DualTrackerProbeError,
    canonical_public_events,
    counter_tape_uniform,
    fuse_actual_state,
    join_captures,
    probe_joined_captures,
    validate_report,
    write_report,
)
from scripts.r1_public_events import R1_SEMANTIC_CONTRACT  # noqa: E402
from scripts.teacher_root_bundle import RootCaptureConfig, build_root_capture  # noqa: E402


@dataclass
class Event:
    actor: str
    kind: str = "hp"
    hp_fraction: float = 1.0
    fainted: bool = False


class State:
    registry = {}

    def __init__(self, side_one=None, side_two=None, weather="none", weather_turns_remaining=0,
                 terrain="none", terrain_turns_remaining=0, trick_room=False,
                 trick_room_turns_remaining=0, team_preview=False, s1_threat=0.0,
                 s2_threat=0.0, scout_value=0.0, threat_matrix=None,
                 wincon_matrix=None):
        self.side_one = copy.deepcopy(side_one)
        self.side_two = copy.deepcopy(side_two)
        self.weather = weather
        self.weather_turns_remaining = weather_turns_remaining
        self.terrain = terrain
        self.terrain_turns_remaining = terrain_turns_remaining
        self.trick_room = trick_room
        self.trick_room_turns_remaining = trick_room_turns_remaining
        self.team_preview = team_preview
        self.s1_threat = s1_threat
        self.s2_threat = s2_threat
        self.scout_value = scout_value
        self.threat_matrix = list(threat_matrix or [0.0] * 36)
        self.wincon_matrix = list(wincon_matrix or [0.0] * 36)
        self.terminal = False
        self.winner = 0

    @classmethod
    def from_string(cls, value):
        return copy.deepcopy(cls.registry[value])

    def to_string(self):
        def side(value):
            return {"label": value.label, "moves": list(value.moves)}
        return json.dumps({
            "s1": side(self.side_one), "s2": side(self.side_two),
            "g": [self.weather, self.weather_turns_remaining, self.terrain,
                  self.terrain_turns_remaining, self.trick_room,
                  self.trick_room_turns_remaining, self.team_preview],
            "terminal": self.terminal, "winner": self.winner,
        }, sort_keys=True)


def side(label, moves=("tackle",)):
    return SimpleNamespace(label=label, moves=list(moves))


def world(local, opponent, **globals_):
    return State(side(local), side(opponent), **globals_)


class Engine:
    State = State

    def __init__(self):
        self.options_override = None

    @staticmethod
    def r1_semantic_contract():
        return R1_SEMANTIC_CONTRACT

    def root_options(self, *, state):
        if self.options_override is not None:
            return self.options_override(state)
        return list(state.side_one.moves), list(state.side_two.moves)

    @staticmethod
    def terminal_value(state):
        return state.winner if state.terminal else 0


def snapshot(role, user, opponent, *, probs=None, turn=1, prefix=None):
    probs = list(probs or [1.0] + [0.0] * 12)
    return {
        "schema": 3, "tag": "battle-private-sentinel", "namespace": "worker",
        "username": user, "opponent_username": opponent, "player_role": role,
        "decision_idx": turn, "battle_turn": turn, "text_tokens": [1], "numbers": [0.5],
        "illegal_actions": [False] + [True] * 12, "mask_fallback": False,
        "mask_fallback_error": None, "name_table": {"tackle": 0}, "probs": probs,
        "protocol_prefix": prefix or ["|turn|1", "|request|private-request-sentinel"],
        "player_information_state": {
            "schema_version": 1, "universal_state": {}, "player_team": [],
            "opponent_public_team": [], "private_request": {"side": {"id": role}},
        },
        "player_observation_history": {
            "any_opponent_asleep": False, "any_opponent_frozen": False,
            "revealed_opponents": [],
        },
        "continuation_observation_history": {
            "any_opponent_asleep": False, "any_opponent_frozen": False,
            "revealed_opponents": ["contamination-sentinel"],
        },
    }


def capture(role, strings, *, user=None, opponent=None, turn=1, forced=False):
    user = user or ("Alice" if role == "p1" else "Bob")
    opponent = opponent or ("Bob" if role == "p1" else "Alice")
    snap = snapshot(role, user, opponent, turn=turn)
    if forced:
        snap["player_information_state"]["private_request"]["forceSwitch"] = [True]
    identity = {
        "namespace": "worker", "battle_tag": "battle-private-sentinel", "username": user,
        "decision_idx": turn, "battle_turn": turn,
    }
    return build_root_capture(
        identity=identity, player_priors=[("tackle", 1.0)], opponent_priors=[("tackle", 1.0)],
        r1_policy_snapshot=snap, schedules=[strings],
        config=RootCaptureConfig(Path("unused"), 1, 9, "a" * 64),
    )


def projection(next_state, observer, *, events=None, lineage=(0,)):
    translated = events or (
        Event("self" if observer == "SideOne" else "opponent", hp_fraction=0.5),
        Event("opponent" if observer == "SideOne" else "self", hp_fraction=0.5),
    )
    item = SimpleNamespace(
        events=translated, own_delta=SimpleNamespace(), legal_actions=("tackle",),
        next_states=(copy.deepcopy(next_state),), source_world_indices=lineage,
    )
    return SimpleNamespace(observation_classes=(item,))


class Tracker:
    forks = []

    def __init__(self, role, *, reject=False, bad_legal=False, bad_policy=False):
        self.role = role
        self.reject = reject
        self.bad_legal = bad_legal
        self.bad_policy = bad_policy
        self.state = SimpleNamespace(battle_won=False, battle_lost=False)
        self.history = []

    def public_opponent_registry(self):
        return {}

    def fork(self):
        result = copy.deepcopy(self)
        Tracker.forks.append(result)
        return result

    def apply_basic_move_class(self, item):
        if self.reject:
            raise ValueError
        self.history.append(self.role)
        state = item.next_states[0]
        self.state.battle_won = state.terminal and state.winner == (1 if self.role == "p1" else -1)
        self.state.battle_lost = state.terminal and not self.state.battle_won
        return {
            "illegal_actions": ([True] * 13 if state.terminal else
                                ([False, False] + [True] * 11 if self.bad_legal else [False] + [True] * 12)),
            "name_table": {"tackle": 0, **({"other": 1} if self.bad_legal else {})},
            "terminal": state.terminal, "bad_policy": self.bad_policy,
        }


HASHES = {
    "p1_capture_file_sha256": "b" * 64, "p2_capture_file_sha256": "c" * 64,
    "source_manifest_sha256": "d" * 64, "source_manifest_file_sha256": "e" * 64,
    "analysis_manifest_sha256": "f" * 64, "analysis_manifest_file_sha256": "1" * 64,
    "engine_binding_sha256": "2" * 64, "checkpoint_sha256": "3" * 64,
    "probe_script_sha256": "4" * 64, "public_events_module_sha256": "5" * 64,
}


class DualTrackerParityProbeTests(unittest.TestCase):
    def setUp(self):
        State.registry = {
            "p1-a": world("p1", "hidden-a"), "p1-b": world("p1", "hidden-b"),
            "p2-a": world("p2", "hidden-c"), "p2-b": world("p2", "hidden-d"),
            "p1-zero-bad": world("changed", "hidden-z"),
            "p2-weather": world("p2", "hidden", weather="rain", weather_turns_remaining=3),
        }
        Tracker.forks = []

    def run_probe(self, *, first=None, second=None, engine=None, factory=None,
                  infer=None, project=None, rollouts=2):
        first = first or [capture("p1", [("p1-a", 1.0), ("p1-b", 0.0)])]
        second = second or [capture("p2", [("p2-a", 1.0), ("p2-b", 0.0)])]
        engine = engine or Engine()
        factory = factory or (lambda snap: Tracker(snap["player_role"]))
        infer = infer or (lambda obs: ([0.0, 1.0] + [0.0] * 11) if obs.get("bad_policy") else [1.0] + [0.0] * 12)
        project = project or (lambda _e, states, _a, _b, _u, observer_side, **_kw: projection(states[0], observer_side))
        return probe_joined_captures(
            first, second, engine=engine, tracker_factory=factory, policy_infer=infer,
            rollouts=rollouts, base_seed=17, input_hashes=HASHES, projection_fn=project,
        )

    def test_valid_asymmetric_fusion_and_no_opponent_pairing(self):
        fused = fuse_actual_state(capture("p1", [("p1-a", 1.0), ("p1-b", 0.0)]),
                                  capture("p2", [("p2-a", 1.0), ("p2-b", 0.0)]), Engine())
        self.assertEqual((fused.side_one.label, fused.side_two.label), ("p1", "p2"))
        self.assertNotIn("hidden", fused.to_string())
        self.assertEqual(fused.threat_matrix, [0.0] * 36)

    def test_side_invariance_includes_zero_weight_worlds(self):
        with self.assertRaisesRegex(DualTrackerProbeError, "ROOT_FUSION_REJECTED"):
            fuse_actual_state(capture("p1", [("p1-a", 1.0), ("p1-zero-bad", 0.0)]),
                              capture("p2", [("p2-a", 1.0)]), Engine())

    def test_public_global_mismatch_rejected(self):
        with self.assertRaisesRegex(DualTrackerProbeError, "ROOT_FUSION_REJECTED"):
            fuse_actual_state(capture("p1", [("p1-a", 1.0)]),
                              capture("p2", [("p2-weather", 1.0)]), Engine())

    def test_join_orients_roles_and_excludes_forced_singleton(self):
        forced = capture("p1", [("p1-a", 1.0)], turn=2, forced=True)
        pairs, excluded = join_captures(
            [capture("p2", [("p2-a", 1.0)]), forced],
            [capture("p1", [("p1-a", 1.0)])],
        )
        self.assertEqual(pairs[0][0]["r1_policy_snapshot"]["player_role"], "p1")
        self.assertEqual(excluded, 1)

    def test_forced_and_ordinary_boundaries_may_share_turn(self):
        forced = capture("p1", [("p1-a", 1.0)], turn=2, forced=True)
        forced["identity"]["battle_turn"] = 1
        forced["r1_policy_snapshot"]["battle_turn"] = 1
        forced["r1_policy_snapshot"]["protocol_prefix"] = [
            "|faint|p1a: Private",
            "|turn|1",
            "|request|private",
        ]
        unhashed = {key: value for key, value in forced.items() if key != "capture_sha256"}
        forced["capture_sha256"] = hashlib.sha256(
            json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        pairs, excluded = join_captures(
            [capture("p1", [("p1-a", 1.0)]), forced],
            [capture("p2", [("p2-a", 1.0)])],
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(excluded, 1)

    def test_unexplained_and_nonreciprocal_joins_fail_closed(self):
        with self.assertRaises(DualTrackerProbeError):
            join_captures([capture("p1", [("p1-a", 1.0)])], [capture("p2", [("p2-a", 1.0)], user="Eve")])
        with self.assertRaises(DualTrackerProbeError):
            join_captures([capture("p1", [("p1-a", 1.0)])], [capture("p2", [("p2-a", 1.0)], turn=2)])

    def test_counter_tape_golden_values_and_channels(self):
        self.assertEqual(counter_tape_uniform(17, 3, 5, "p1"), 0.5057180395055361)
        self.assertEqual(counter_tape_uniform(17, 3, 5, "p2"), 0.1904892691937754)
        self.assertEqual(counter_tape_uniform(17, 3, 5, "chance"), 0.09456603827890298)

    def test_action_mapping_side_masks(self):
        for missing, expected in (({0}, "P1_ONLY"), ({1}, "P2_ONLY"), ({0, 1}, "BOTH")):
            engine = Engine()
            calls = {0: 0}
            def options(state):
                calls[0] += 1
                if calls[0] <= 2:
                    return ["tackle"], ["tackle"]
                return ([] if 0 in missing else ["tackle"], [] if 1 in missing else ["tackle"])
            engine.options_override = options
            report = self.run_probe(engine=engine, rollouts=1)
            self.assertEqual(report["outcomes"][f"ACTION_MAPPING_REJECTED_{expected}"]["count"], 1)

    def test_projection_side_masks(self):
        for failed, expected in (({"SideOne"}, "P1_ONLY"), ({"SideTwo"}, "P2_ONLY"), ({"SideOne", "SideTwo"}, "BOTH")):
            def project(_e, states, _a, _b, _u, observer_side, **_kw):
                if observer_side in failed:
                    raise ValueError
                return projection(states[0], observer_side)
            report = self.run_probe(project=project, rollouts=1)
            self.assertEqual(report["outcomes"][f"PROJECTION_REJECTED_{expected}"]["count"], 1)

    def test_public_event_canonicalization_exact_display_and_99_cap(self):
        p1 = SimpleNamespace(observation_classes=(SimpleNamespace(events=(Event("self", hp_fraction=0.991), Event("opponent", hp_fraction=0.5))),))
        p2 = SimpleNamespace(observation_classes=(SimpleNamespace(events=(Event("opponent", hp_fraction=0.991), Event("self", hp_fraction=0.5))),))
        self.assertEqual(canonical_public_events(p1, "p1"), canonical_public_events(p2, "p2"))
        self.assertIn('"hp_fraction":0.99', "".join(canonical_public_events(p1, "p1")))
        fainted = SimpleNamespace(observation_classes=(SimpleNamespace(events=(Event("self", hp_fraction=0.01, fainted=True),)),))
        self.assertIn('"hp_fraction":0.0', canonical_public_events(fainted, "p1")[0])

    def test_lineage_corruption(self):
        report = self.run_probe(project=lambda _e, states, _a, _b, _u, observer_side, **_kw: projection(states[0], observer_side, lineage=(1,)), rollouts=1)
        self.assertEqual(report["outcomes"]["PROJECTION_LINEAGE_REJECTED"]["count"], 1)

    def test_independent_next_state_mismatch(self):
        def project(_e, states, _a, _b, _u, observer_side, **_kw):
            target = copy.deepcopy(states[0])
            if observer_side == "SideTwo":
                target.weather = "sun"
            return projection(target, observer_side)
        report = self.run_probe(project=project, rollouts=1)
        self.assertEqual(report["outcomes"]["PUBLIC_OUTCOME_MISMATCH"]["count"], 1)

    def test_tracker_fork_isolation_and_side_masks(self):
        for failed, expected in (({"p1"}, "P1_ONLY"), ({"p2"}, "P2_ONLY"), ({"p1", "p2"}, "BOTH")):
            report = self.run_probe(factory=lambda snap: Tracker(snap["player_role"], reject=snap["player_role"] in failed), rollouts=1)
            self.assertEqual(report["outcomes"][f"TRACKER_REJECTED_{expected}"]["count"], 1)
            self.assertTrue(all(len(tracker.history) == 1 for tracker in Tracker.forks if not tracker.reject))

    def test_next_legality_side_masks(self):
        for failed, expected in (({"p1"}, "P1_ONLY"), ({"p2"}, "P2_ONLY"), ({"p1", "p2"}, "BOTH")):
            report = self.run_probe(factory=lambda snap: Tracker(snap["player_role"], bad_legal=snap["player_role"] in failed), rollouts=1)
            self.assertEqual(report["outcomes"][f"NEXT_LEGALITY_MISMATCH_{expected}"]["count"], 1)

    def test_next_policy_validity_side_masks(self):
        for failed, expected in (({"p1"}, "P1_ONLY"), ({"p2"}, "P2_ONLY"), ({"p1", "p2"}, "BOTH")):
            report = self.run_probe(factory=lambda snap: Tracker(snap["player_role"], bad_policy=snap["player_role"] in failed), rollouts=1)
            self.assertEqual(report["outcomes"][f"NEXT_POLICY_INVALID_{expected}"]["count"], 1)

    def test_terminal_agreement_and_inversion(self):
        def project(_e, states, _a, _b, _u, observer_side, **_kw):
            target = copy.deepcopy(states[0]); target.terminal = True; target.winner = 1
            return projection(target, observer_side)
        report = self.run_probe(project=project, rollouts=1)
        self.assertEqual(report["outcomes"]["certified_terminal"]["count"], 1)
        report = self.run_probe(project=project, factory=lambda snap: Tracker("p1"), rollouts=1)
        self.assertEqual(report["outcomes"]["TERMINAL_DISAGREEMENT"]["count"], 1)

    def test_root_tracker_and_policy_side_masks(self):
        for failed, expected in (({"p1"}, "P1_ONLY"), ({"p2"}, "P2_ONLY"), ({"p1", "p2"}, "BOTH")):
            def factory(snap):
                if snap["player_role"] in failed:
                    raise ValueError
                return Tracker(snap["player_role"])
            report = self.run_probe(factory=factory, rollouts=2)
            self.assertEqual(report["outcomes"][f"ROOT_TRACKER_REJECTED_{expected}"], {"count": 2, "mass": 1.0})
            infer = lambda obs: ([0.0, 1.0] + [0.0] * 11 if obs.get("player_role") in failed else [1.0] + [0.0] * 12)
            report = self.run_probe(infer=infer, rollouts=2)
            self.assertEqual(report["outcomes"][f"ROOT_POLICY_PARITY_REJECTED_{expected}"], {"count": 2, "mass": 1.0})

    def test_count_mass_conservation_self_hash_and_tamper_rejection(self):
        report = self.run_probe(rollouts=2)
        validate_report(report)
        unhashed = dict(report); claimed = unhashed.pop("report_sha256")
        self.assertEqual(claimed, hashlib.sha256(json.dumps(unhashed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest())
        damaged = copy.deepcopy(report); damaged["outcomes"]["certified_nonterminal"]["mass"] = 0.5
        unhashed = dict(damaged); unhashed.pop("report_sha256"); damaged["report_sha256"] = hashlib.sha256(json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with self.assertRaisesRegex(DualTrackerProbeError, "conserved"):
            validate_report(damaged)

        for field, value, message in (
            (("claim_status",), "strength_estimate", "schema"),
            (("configuration", "p1_policy"), "unfrozen", "configuration"),
            (("common_tape", "algorithm"), "mutable_rng", "common tape"),
        ):
            with self.subTest(field=field):
                altered = copy.deepcopy(report)
                target = altered
                for key in field[:-1]:
                    target = target[key]
                target[field[-1]] = value
                unhashed = dict(altered)
                unhashed.pop("report_sha256")
                altered["report_sha256"] = hashlib.sha256(
                    json.dumps(
                        unhashed, sort_keys=True, separators=(",", ":")
                    ).encode("ascii")
                ).hexdigest()
                with self.assertRaisesRegex(DualTrackerProbeError, message):
                    validate_report(altered)

        altered = copy.deepcopy(report)
        altered["outcomes"]["certified_nonterminal"] = {
            "count": 1,
            "mass": 1.0,
        }
        altered["outcomes"]["ROOT_FUSION_REJECTED"] = {
            "count": 1,
            "mass": 0.0,
        }
        unhashed = dict(altered)
        unhashed.pop("report_sha256")
        altered["report_sha256"] = hashlib.sha256(
            json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).hexdigest()
        with self.assertRaisesRegex(DualTrackerProbeError, "disagree"):
            validate_report(altered)

    def test_privacy_sentinels_absent(self):
        serialized = json.dumps(self.run_probe(rollouts=1), sort_keys=True)
        for sentinel in ("private-sentinel", "Alice", "Bob", "tackle", "contamination-sentinel", '"events"'):
            self.assertNotIn(sentinel, serialized)

    def test_mode_0600_exclusive_force_and_invalid_no_partial_output(self):
        report = self.run_probe(rollouts=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_report(report, path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            validate_report(json.loads(path.read_text()))
            with self.assertRaisesRegex(DualTrackerProbeError, "already exists"):
                write_report(report, path)
            write_report(report, path, force=True)
            invalid = copy.deepcopy(report); invalid["report_sha256"] = "0" * 64
            absent = Path(directory) / "absent.json"
            with self.assertRaises(DualTrackerProbeError):
                write_report(invalid, absent)
            self.assertFalse(absent.exists())

    def test_performance_smoke_32_roots_two_rollouts(self):
        first, second = [], []
        for turn in range(32):
            first.append(capture("p1", [("p1-a", 1.0)], turn=turn + 1))
            second.append(capture("p2", [("p2-a", 1.0)], turn=turn + 1))
        started = time.perf_counter()
        report = self.run_probe(first=first, second=second, rollouts=2)
        self.assertLess(time.perf_counter() - started, 30.0)
        self.assertEqual(report["counts"]["total_trials"], 64)


if __name__ == "__main__":
    unittest.main()
