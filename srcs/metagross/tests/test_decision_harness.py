from __future__ import annotations

import unittest

from srcs.metagross.decision_harness import (
    RecursiveShadowPlan,
    execute_recursive_shadow,
    plan_recursive_shadow,
)


def certificate(**overrides):
    checks = {
        "candidate_catastrophe_rate": True,
        "symmetric_catastrophe_rate_gap": True,
        "symmetric_catastrophe_severity": True,
        "lower_tail_cvar": True,
    }
    checks.update(overrides)
    return {"complete": True, "checks": checks}


class DecisionHarnessTest(unittest.TestCase):
    def test_policy_search_agreement_allocates_no_recursive_work(self):
        plan = plan_recursive_shadow(
            {"baseline": "recover", "raw_choice": "recover"}, {}
        )

        artifact = execute_recursive_shadow(
            plan,
            "recover",
            lambda _plan: self.fail("agreement must not allocate shadow work"),
        )

        self.assertFalse(plan.triggered)
        self.assertEqual(artifact["stop_reason"], "policy_search_agreement")
        self.assertEqual(artifact["nodes"], [])
        self.assertTrue(artifact["played_action_unchanged"])

    def test_disagreement_allocates_one_shadow_horizon_without_changing_action(self):
        plan = plan_recursive_shadow(
            {"baseline": "recover", "raw_choice": "earthquake"},
            {"earthquake": {1: certificate()}},
        )
        calls = []

        artifact = execute_recursive_shadow(
            plan,
            "recover",
            lambda value: calls.append(value) or {"certificate": {"qualified": True}},
        )

        self.assertEqual(plan.operation, "horizon")
        self.assertEqual(len(calls), 1)
        self.assertEqual(artifact["played_action"], "recover")
        self.assertTrue(artifact["played_action_unchanged"])
        self.assertFalse(artifact["admission_eligible"])
        self.assertEqual(artifact["nodes"][0]["depth"], 1)

    def test_unstable_catastrophic_tail_allocates_one_world_cohort(self):
        plan = plan_recursive_shadow(
            {"baseline": "recover", "raw_choice": "earthquake"},
            {
                "earthquake": {
                    1: certificate(),
                    2: certificate(lower_tail_cvar=False),
                }
            },
        )

        artifact = execute_recursive_shadow(
            plan, "recover", lambda _plan: {"worlds": 16}
        )

        self.assertEqual(plan.operation, "worlds")
        self.assertEqual(plan.reason, "catastrophic_tail_unstable")
        self.assertEqual(artifact["nodes"][0]["worlds"], 16)
        self.assertEqual(artifact["limits"]["max_remote_batches"], 1)

    def test_shadow_failure_cannot_replace_the_frozen_action(self):
        plan = RecursiveShadowPlan(
            True, "horizon", "policy_search_disagreement", "earthquake", depth=1
        )

        def fail(_plan):
            raise RuntimeError("remote unavailable")

        artifact = execute_recursive_shadow(plan, "recover", fail)

        self.assertFalse(artifact["complete"])
        self.assertEqual(artifact["stop_reason"], "shadow_allocation_failed")
        self.assertEqual(artifact["played_action"], "recover")
        self.assertTrue(artifact["played_action_unchanged"])
        self.assertNotIn("recommended_action", artifact)


if __name__ == "__main__":
    unittest.main()
