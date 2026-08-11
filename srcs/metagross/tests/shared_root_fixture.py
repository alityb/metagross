from __future__ import annotations

import struct
from types import SimpleNamespace

from srcs.metagross.mcts_contract import _fnv1a64_digest


def native_shared_root_result(
    *, input_weights: tuple[float, ...] = (0.25, 0.75)
) -> SimpleNamespace:
    state = "state"
    payoff = 0.5
    continuation = SimpleNamespace(
        seed=11,
        requested_iterations=8,
        executed_iterations=8,
        visits=8,
        total_score=4.0,
        total_score_bits=struct.unpack("<I", struct.pack("<f", 4.0))[0],
        payoff=payoff,
        payoff_bits=struct.unpack("<Q", struct.pack("<d", payoff))[0],
    )
    none_digest = _fnv1a64_digest([b"none"])
    diagnostics = SimpleNamespace(
        solver_contract="weighted-shared-rm-plus-v1",
        iterations=100,
        continuation_iterations=8,
        seed=7,
        prior_strength=1.0,
        expected_value=payoff,
        player_best_response_value=payoff,
        opponent_best_response_value=payoff,
        player_best_response_gain=0.0,
        opponent_best_response_gain=0.0,
        nash_conv=0.0,
        exploitability=0.0,
        player_regret_bound=0.0,
        opponent_regret_bound=0.0,
        total_regret_bound=0.0,
        payoff_cells=1,
        total_forced_continuation_iterations=8,
        input_particle_count=len(input_weights),
        positive_particle_count=len(input_weights),
        canonical_particle_count=1,
        normalized_weight_sum=1.0,
        action_support_digest=_fnv1a64_digest([b"tackle"]),
        particle_digest=_fnv1a64_digest([state.encode(), struct.pack("<d", 1.0)]),
        payoff_digest=_fnv1a64_digest([struct.pack("<d", payoff)]),
        player_prior_digest=none_digest,
        opponent_prior_digest=none_digest,
    )
    capture = SimpleNamespace(
        schema_version=1,
        solver_contract="weighted-shared-rm-plus-v1",
        configuration=SimpleNamespace(
            iterations=100,
            continuation_iterations=8,
            seed=7,
            prior_strength=1.0,
        ),
        own_action_support=["tackle"],
        normalized_player_prior=None,
        canonical_particles=[
            SimpleNamespace(
                canonical_index=0,
                state=state,
                normalized_weight=1.0,
                source_particles=[
                    SimpleNamespace(input_index=index, input_weight=weight)
                    for index, weight in enumerate(input_weights)
                ],
                opponent_action_support=["protect"],
                normalized_opponent_prior=None,
                payoff_matrix=[[payoff]],
                continuations=[[continuation]],
                opponent_policy=[1.0],
            )
        ],
    )
    return SimpleNamespace(
        policy=[
            SimpleNamespace(
                action="tackle", probability=1.0, counterfactual_value=payoff
            )
        ],
        opponent_policies=[[('protect', 1.0)]],
        diagnostics=diagnostics,
        replay_capture=capture,
    )
