from __future__ import annotations

from srcs.metagross.run_foul_play import automatic_request_action


def test_automatic_request_action_accepts_only_sole_forced_commands() -> None:
    assert automatic_request_action({"actions": ["struggle"]}) == "struggle"
    assert automatic_request_action({"actions": ("recharge",)}) == "recharge"
    assert automatic_request_action({"actions": ["struggle", "switch blissey"]}) is None
    assert automatic_request_action({"actions": ["tackle"]}) is None
    assert automatic_request_action(None) is None

