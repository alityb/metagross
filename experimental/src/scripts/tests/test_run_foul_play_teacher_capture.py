from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_foul_play  # noqa: E402


class RunFoulPlayTeacherCaptureTests(unittest.TestCase):
    def test_shadow_schedules_do_not_change_behavior_rng_or_selector_input(self):
        events = []

        def sampler(_battle, num_battles):
            return [
                (SimpleNamespace(state=f"state-{random.random()}-{index}"), 1 / num_battles)
                for index in range(num_battles)
            ]

        def converter(battle):
            return SimpleNamespace(to_string=lambda: battle.state)

        def selector(results):
            events.append(("selector", results))
            return random.random()

        random.seed(123)
        baseline_schedule = sampler(object(), 2)
        baseline_results = [(object(), 0.5, 0), (object(), 0.5, 1)]
        baseline_choice = selector(baseline_results)
        baseline_next = random.random()
        events.clear()

        search_main = ModuleType("fp.search.main")
        search_main.prepare_random_battles = sampler
        search_main.battle_to_poke_engine_state = converter
        search_main.select_move_from_mcts_results = selector
        fp = ModuleType("fp")
        search = ModuleType("fp.search")
        fp.search = search
        search.main = search_main
        run_foul_play._PRIOR_STATE.update(
            {
                "priors": [("move-a", 1.0)],
                "opp_priors": [("opp-a", 1.0)],
                "namespace": "",
                "battle_tag": "battle-1",
                "username": "learner",
                "prior_decision_idx": 0,
                "prior_battle_turn": 1,
                "r1_policy_snapshot": {
                    "schema": 3, "tag": "battle-1", "namespace": "", "username": "learner",
                    "decision_idx": 0, "battle_turn": 1, "text_tokens": [1], "numbers": [0.0],
                    "illegal_actions": [False] + [True] * 12, "mask_fallback": False,
                    "mask_fallback_error": None, "name_table": {"move-a": 0},
                    "probs": [1.0] + [0.0] * 12, "protocol_prefix": ["|request|{}"],
                    "player_information_state": {"schema_version": 1, "universal_state": {}, "player_team": [], "opponent_public_team": []},
                    "player_observation_history": {"any_opponent_asleep": False, "any_opponent_frozen": False, "revealed_opponents": ["opponent"]},
                    "continuation_observation_history": {"any_opponent_asleep": False, "any_opponent_frozen": False, "revealed_opponents": ["opponent", "player"]},
                },
            }
        )
        environment = {
            "METAGROSS_TEACHER_ROOT_BUNDLE": "/tmp/capture.jsonl",
            "METAGROSS_TEACHER_DETERMINIZATION_SCHEDULES": "3",
            "METAGROSS_TEACHER_DETERMINIZATION_SEED": "17",
            "METAGROSS_TEACHER_MANIFEST_SHA256": "a" * 64,
        }
        appended = []
        with patch.dict(
            sys.modules,
            {"fp": fp, "fp.search": search, "fp.search.main": search_main},
        ), patch.dict("os.environ", environment, clear=True), patch(
            "teacher_root_bundle.append_root_capture",
            side_effect=lambda path, capture: (
                events.append(("append", capture)),
                appended.append((path, capture)),
            ),
        ):
            run_foul_play.patch_teacher_root_bundle_capture()
            random.seed(123)
            behavior_schedule = search_main.prepare_random_battles(object(), 2)
            results = [(object(), 0.5, 0), (object(), 0.5, 1)]
            choice = search_main.select_move_from_mcts_results(results)
            next_random = random.random()

        self.assertEqual(
            [battle.state for battle, _ in behavior_schedule],
            [battle.state for battle, _ in baseline_schedule],
        )
        self.assertEqual(choice, baseline_choice)
        self.assertEqual(next_random, baseline_next)
        self.assertIs(events[0][1], results)
        self.assertEqual(events[0][0], "selector")
        self.assertEqual(events[1][0], "append")
        capture = appended[0][1]
        self.assertEqual(len(capture["schedules"]), 3)
        self.assertEqual(capture["behavior_schedule_id"], 0)
        self.assertEqual(capture["r1_policy_snapshot"], run_foul_play._PRIOR_STATE["r1_policy_snapshot"])
        self.assertNotIn("treatments", capture)


if __name__ == "__main__":
    unittest.main()
