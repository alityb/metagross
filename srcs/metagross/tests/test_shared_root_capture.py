from __future__ import annotations

import copy
import unittest

from srcs.metagross import run_foul_play
from srcs.metagross.mcts_contract import (
    ENGINE_CONTRACT,
    ENGINE_SOURCE_SHA256,
    shared_root_result_payload,
)
from srcs.metagross.mcts_contract import _fnv1a64_digest
from srcs.metagross.shared_root_capture import canonical_sha256, validate_search_row
from srcs.metagross.tests.shared_root_fixture import native_shared_root_result


class SharedRootCaptureTest(unittest.TestCase):
    def setUp(self):
        self.previous_prior_state = dict(run_foul_play._PRIOR_STATE)
        run_foul_play._PRIOR_STATE.update(
            {"priors": None, "opp_priors": None, "remote_search": None}
        )

    def tearDown(self):
        run_foul_play._PRIOR_STATE.clear()
        run_foul_play._PRIOR_STATE.update(self.previous_prior_state)

    def capture_row(self, *, aliased_switch: bool = False):
        result = shared_root_result_payload(native_shared_root_result())
        request_action = "tackle"
        if aliased_switch:
            native_action = "switch mr-mime"
            request_action = "switch mrmime"
            result["policy"][0]["action"] = native_action
            result["replay_capture"]["own_action_support"] = [native_action]
            result["diagnostics"]["action_support_digest"] = _fnv1a64_digest(
                [native_action.encode()]
            )
        remote_search = {
            "sampling_seed": 13,
            "action_seed": 17,
            "request_ids": ["request-1"],
            "engine": {
                "contract": ENGINE_CONTRACT,
                "source_sha256": ENGINE_SOURCE_SHA256,
                "native_sha256": "b" * 64,
                "distribution_version": "0.0.47",
            },
        }
        envelope = run_foul_play.build_shared_root_replay_envelope(
            states=["state", "state"],
            source_weights=[1.0, 3.0],
            normalized_weights=[0.25, 0.75],
            iterations=100,
            continuation_iterations=8,
            solver_seed=7,
            action_seed=17,
            result=result,
            remote_search=remote_search,
            request_actions={request_action},
        )
        sampled, draw = run_foul_play._sample_shared_root_action(
            [request_action], {request_action: 1.0}, 17
        )
        return {
            "schema": 4,
            "shared_root": result,
            "shared_root_replay": envelope,
            "choice": sampled,
            "choice_override": {
                "sampled_action": sampled,
                "mixed_strategy_draw": draw,
            },
            "player_priors": None,
            "opponent_priors": None,
            "remote_search": remote_search,
        }

    def test_capture_validates_source_mapping_and_sampling(self):
        row = self.capture_row()
        summary = validate_search_row(row)
        self.assertEqual(summary["particles"], 2)
        self.assertEqual(summary["canonical_particles"], 1)
        self.assertEqual(summary["payoff_cells"], 1)
        self.assertEqual(summary["sampled_action"], "tackle")
        self.assertFalse(summary["exact_replay"])

    def test_capture_replays_request_authorized_switch_alias(self):
        summary = validate_search_row(self.capture_row(aliased_switch=True))
        self.assertEqual(summary["sampled_action"], "switch mrmime")

    def test_capture_tampering_fails_closed(self):
        for mutation in (
            "source",
            "matrix",
            "draw",
            "nan_draw",
            "prior",
            "engine",
            "missing_remote_engine",
            "fabricated_alias",
            "hash",
        ):
            with self.subTest(mutation=mutation):
                row = self.capture_row()
                if mutation == "source":
                    row["shared_root_replay"]["source_particles"][0][
                        "serialized_state"
                    ] = "changed"
                elif mutation == "matrix":
                    row["shared_root"]["replay_capture"]["canonical_particles"][0][
                        "payoff_matrix"
                    ][0][0] = 0.75
                elif mutation == "draw":
                    row["choice_override"]["mixed_strategy_draw"] = 0.0
                elif mutation == "nan_draw":
                    row["choice_override"]["mixed_strategy_draw"] = float("nan")
                elif mutation == "prior":
                    envelope = row["shared_root_replay"]
                    envelope["solver"]["s1_prior"] = [["tackle", 1.0]]
                    unsigned = {**envelope}
                    unsigned.pop("capture_sha256")
                    envelope["capture_sha256"] = canonical_sha256(unsigned)
                elif mutation == "engine":
                    envelope = row["shared_root_replay"]
                    envelope["engine"]["source_sha256"] = "0" * 64
                    unsigned = {**envelope}
                    unsigned.pop("capture_sha256")
                    envelope["capture_sha256"] = canonical_sha256(unsigned)
                elif mutation == "missing_remote_engine":
                    row["remote_search"].pop("engine")
                elif mutation == "fabricated_alias":
                    row = self.capture_row(aliased_switch=True)
                    envelope = row["shared_root_replay"]
                    envelope["request_action_support"].append("tackle")
                    envelope["request_action_support"].sort()
                    envelope["action_aliases"][0]["request_action"] = "tackle"
                    row["choice_override"]["sampled_action"] = "tackle"
                    unsigned = {**envelope}
                    unsigned.pop("capture_sha256")
                    envelope["capture_sha256"] = canonical_sha256(unsigned)
                else:
                    row["shared_root_replay"]["capture_sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    validate_search_row(copy.deepcopy(row))


if __name__ == "__main__":
    unittest.main()
