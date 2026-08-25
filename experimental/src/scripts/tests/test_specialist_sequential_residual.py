from __future__ import annotations

import numpy as np
import torch

from scripts.evaluate_sequential_search_residual import TinyInteractionHead, projection
from scripts.evaluate_specialist_action_residual import BoostedStumps, action_family, action_subfamily
from train.action_semantic_residual import SEMANTIC_FEATURE_NAMES


def semantic_row(**values):
    vector = [0.0] * len(SEMANTIC_FEATURE_NAMES)
    index = {name: offset for offset, name in enumerate(SEMANTIC_FEATURE_NAMES)}
    for name, value in values.items():
        vector[index[name]] = value
    return {"semantic_features": vector}


def test_direct_attack_precedes_damage_tempo_fallback():
    row = semantic_row(
        action_is_attack=1.0,
        baseline_is_attack=1.0,
        relative_damage_tempo_delta_mean=0.5,
    )
    assert action_family(row) == "direct_attack"
    assert action_subfamily(row) == "direct_move_choice"


def test_switch_and_status_directional_subfamilies():
    switch = semantic_row(action_is_switch=1.0, baseline_is_attack=1.0)
    status = semantic_row(action_is_setup=1.0, baseline_is_attack=1.0)
    assert action_family(switch) == "switch_option"
    assert action_subfamily(switch) == "attack_or_status_to_switch"
    assert action_family(status) == "status_tempo"
    assert action_subfamily(status) == "attack_to_status"


def test_boosted_stump_learns_nonlinear_partition():
    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.where(x[:, 0] <= 9, -1.0, 1.0)
    prediction = BoostedStumps(16, 0.1).fit(x, y).predict(x)
    assert prediction[:10].max() < 0
    assert prediction[10:].min() > 0


def test_sequential_projection_and_capacity_are_frozen():
    assert np.array_equal(projection(16, 123), projection(16, 123))
    assert not np.array_equal(projection(16, 123), projection(16, 124))
    model = TinyInteractionHead(20 + 32, 8)
    assert sum(parameter.numel() for parameter in model.parameters()) == 433
    assert model(torch.zeros(3, 52)).shape == (3,)
