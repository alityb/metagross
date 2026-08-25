from pathlib import Path

import pytest

from train.action_semantic_residual import (
    ABSOLUTE_SEMANTIC_FEATURE_NAMES, SEMANTIC_FEATURE_NAMES, assert_confirmation_disjoint,
    residualize_semantics, summarize_semantics,
)


def test_confirmation_guard_rejects_root_or_battle_overlap() -> None:
    freeze = {"root_ids": ["r1"], "battle_ids": ["b1"]}
    assert_confirmation_disjoint([{"root_id": "r2", "battle_id": "b2"}], freeze)
    with pytest.raises(ValueError, match="overlaps development freeze"):
        assert_confirmation_disjoint([{"root_id": "r1", "battle_id": "b2"}], freeze)
    with pytest.raises(ValueError, match="overlaps development freeze"):
        assert_confirmation_disjoint([{"root_id": "r2", "battle_id": "b1"}], freeze)


def test_semantics_serializes_only_aggregates() -> None:
    rows = [[float(index + probe) / 100 for index in range(16)] for probe in range(8)]
    features = summarize_semantics("switch foo", rows, [0.1] * 8, [0.2] * 8)
    assert len(features) == len(ABSOLUTE_SEMANTIC_FEATURE_NAMES)
    assert features[2] == 1.0
    assert all(not isinstance(value, list) for value in features)
    residual = residualize_semantics(features, [0.0] * len(features))
    assert len(residual) == len(SEMANTIC_FEATURE_NAMES)
