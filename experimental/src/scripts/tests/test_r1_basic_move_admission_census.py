from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.r1_basic_move_admission_census import (  # noqa: E402
    AdmissionCensusError,
    analyze_captures,
    write_report,
)
from scripts.r1_public_events import (  # noqa: E402
    R1_SEMANTIC_CONTRACT,
    SIDE_CONDITION_FIELDS,
    SIDE_FIELDS,
    VOLATILE_DURATION_FIELDS,
)
from scripts.teacher_root_bundle import (  # noqa: E402
    RootCaptureConfig,
    build_root_capture,
)


def _identity(index=0):
    return {
        "namespace": "private-worker",
        "battle_tag": f"private-battle-{index}",
        "username": "private-player",
        "decision_idx": index,
        "battle_turn": index + 1,
    }


def _snapshot(identity):
    return {
        "schema": 3,
        "tag": identity["battle_tag"],
        "namespace": identity["namespace"],
        "username": identity["username"],
        "decision_idx": identity["decision_idx"],
        "battle_turn": identity["battle_turn"],
        "text_tokens": [1],
        "numbers": [0.5],
        "illegal_actions": [False] + [True] * 12,
        "mask_fallback": False,
        "mask_fallback_error": None,
        "name_table": {"tackle": 0},
        "probs": [1.0] + [0.0] * 12,
        "protocol_prefix": ["|request|{}"],
        "player_information_state": {
            "schema_version": 1,
            "universal_state": {},
            "player_team": [],
            "opponent_public_team": [],
        },
        "player_observation_history": {
            "any_opponent_asleep": False,
            "any_opponent_frozen": False,
            "revealed_opponents": ["private-opponent"],
        },
        "continuation_observation_history": {
            "any_opponent_asleep": False,
            "any_opponent_frozen": False,
            "revealed_opponents": ["private-opponent"],
        },
    }


def _side(*, item="none", ability="none"):
    pokemon = SimpleNamespace(
        id="private-species",
        hp=100,
        maxhp=100,
        status="none",
        item=item,
        ability=ability,
        terastallized=False,
        types=("normal", "typeless"),
        base_types=("normal", "typeless"),
        moves=[SimpleNamespace(id="tackle", pp=35, disabled=False)],
    )
    values = {field: 0 for field in SIDE_FIELDS}
    values.update(
        switch_out_move_second_saved_move="NONE",
        pokemon=[pokemon],
        active_index="0",
        side_conditions=SimpleNamespace(
            **{field: 0 for field in SIDE_CONDITION_FIELDS}
        ),
        volatile_status_durations=SimpleNamespace(
            **{field: 0 for field in VOLATILE_DURATION_FIELDS}
        ),
    )
    return SimpleNamespace(**values)


def _state(*, item="none", ability="none"):
    return SimpleNamespace(
        team_preview=False,
        weather="NONE",
        terrain="NONE",
        trick_room=False,
        side_one=_side(),
        side_two=_side(item=item, ability=ability),
        side_one_actions=["tackle"],
        side_two_actions=["tackle"],
    )


class FakeStateBinding:
    registry = {}

    @classmethod
    def from_string(cls, value):
        return cls.registry[value]


class FakeEngine:
    State = FakeStateBinding

    def __init__(self, *, contract=R1_SEMANTIC_CONTRACT, fail=False, unaccounted=()):
        self.contract = contract
        self.fail = fail
        self.unaccounted = list(unaccounted)

    def r1_semantic_contract(self):
        return self.contract

    def root_options(self, *, state):
        return state.side_one_actions, state.side_two_actions

    def step_with_uniform_r1_semantic(self, state, _side_one, _side_two, _uniform):
        if self.fail:
            raise RuntimeError("private-hidden-exception-sentinel")
        return SimpleNamespace(
            state=state,
            events=[],
            unaccounted_instruction_kinds=self.unaccounted,
        )


def _capture(worlds):
    root_identity = _identity()
    return build_root_capture(
        identity=root_identity,
        player_priors=[("tackle", 1.0)],
        opponent_priors=[("tackle", 1.0)],
        r1_policy_snapshot=_snapshot(root_identity),
        schedules=[worlds],
        config=RootCaptureConfig(Path("private.jsonl"), 1, 7, "a" * 64),
    )


def _analyze(capture, engine):
    return analyze_captures(
        [capture],
        engine=engine,
        uniforms=(0.25, 0.75),
        capture_file_sha256="b" * 64,
        source_manifest_sha256="a" * 64,
        source_manifest_file_sha256="c" * 64,
        analysis_manifest_sha256="1" * 64,
        analysis_manifest_file_sha256="2" * 64,
        engine_binding_sha256="d" * 64,
        census_script_sha256="e" * 64,
        public_events_module_sha256="f" * 64,
    )


class R1BasicMoveAdmissionCensusTests(unittest.TestCase):
    def setUp(self):
        FakeStateBinding.registry = {
            "private-blocked-world": _state(
                item="private-hidden-item-sentinel",
                ability="private-hidden-ability-sentinel",
            ),
            "private-clean-world": _state(),
        }

    def test_hidden_items_and_abilities_are_safe_when_trace_is_fully_accounted(self):
        report = _analyze(
            _capture(
                [
                    ("private-blocked-world", 0.25),
                    ("private-clean-world", 0.75),
                ]
            ),
            FakeEngine(),
        )

        blockers = report["world_trials"]["overlapping_blockers"]
        outcomes = report["world_trials"]["certificate_outcomes"]
        self.assertEqual(blockers["ACTIVE_ITEM"]["weighted_rate"], 0.0)
        self.assertEqual(blockers["ACTIVE_ABILITY"]["weighted_rate"], 0.0)
        self.assertEqual(
            report["world_trials"]["eligibility"]["ELIGIBLE"]["weighted_rate"],
            1.0,
        )
        self.assertEqual(outcomes["STRUCTURAL_REJECTION"]["weighted_rate"], 0.0)
        self.assertEqual(outcomes["ADMITTED"]["weighted_rate"], 1.0)
        self.assertEqual(
            report["strict_schedule_trials"]["certificate_outcomes"][
                "ADMITTED"
            ]["weighted_rate"],
            1.0,
        )
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("private-hidden", serialized)
        self.assertNotIn("private-battle", serialized)
        self.assertEqual(report, _analyze(_capture([
            ("private-blocked-world", 0.25),
            ("private-clean-world", 0.75),
        ]), FakeEngine()))

    def test_unaccounted_hidden_mechanic_is_rejected(self):
        report = _analyze(
            _capture([("private-blocked-world", 1.0)]),
            FakeEngine(unaccounted=("ToggleChoiceLock",)),
        )
        outcomes = report["world_trials"]["certificate_outcomes"]
        self.assertEqual(outcomes["UNACCOUNTED_INSTRUCTION"]["weighted_rate"], 1.0)

    def test_hidden_engine_exception_is_reduced_to_fixed_code(self):
        report = _analyze(
            _capture([("private-clean-world", 1.0)]), FakeEngine(fail=True)
        )
        outcomes = report["world_trials"]["certificate_outcomes"]
        self.assertEqual(outcomes["ENGINE_OR_BINDING_ERROR"]["weighted_rate"], 1.0)
        self.assertNotIn("private-hidden-exception-sentinel", json.dumps(report))

    def test_contract_mismatch_fails_before_report(self):
        with self.assertRaisesRegex(AdmissionCensusError, "contract does not match"):
            _analyze(
                _capture([("private-clean-world", 1.0)]),
                FakeEngine(contract="wrong-contract"),
            )

    def test_private_report_write_is_exclusive_and_mode_0600(self):
        report = _analyze(
            _capture([("private-clean-world", 1.0)]), FakeEngine()
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            write_report(report, path)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(path.read_text()), report)
            with self.assertRaisesRegex(AdmissionCensusError, "already exists"):
                write_report(report, path)


if __name__ == "__main__":
    unittest.main()
