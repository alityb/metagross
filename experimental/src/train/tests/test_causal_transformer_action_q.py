from __future__ import annotations

import torch
import pytest

from train.causal_transformer_action_q import (
    CausalResidualQHead,
    battle_split,
    corrected_logits,
    ranking_metrics,
    residual_q_loss,
)


def test_battle_split_is_deterministic_and_battle_grouped() -> None:
    battle_ids = [f"battle-{index}" for index in range(500)]
    first = {battle_id: battle_split(battle_id) for battle_id in battle_ids}
    second = {battle_id: battle_split(battle_id) for battle_id in reversed(battle_ids)}

    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}
    assert all(battle_split(battle_id) == split for battle_id, split in first.items())


def test_zero_residual_reproduces_base_distribution_on_support() -> None:
    base = torch.tensor([[0.1, 0.2, 0.3, 0.4] + [0.0] * 9])
    support = torch.tensor([[True, True, False, True] + [False] * 9])
    correction = torch.zeros_like(base)

    probabilities = torch.softmax(corrected_logits(base, correction, support), dim=1)
    expected = base * support
    expected /= expected.sum(dim=1, keepdim=True)

    assert torch.allclose(probabilities, expected)
    assert probabilities[0, 2] == 0


def test_teacher_aligned_residual_has_lower_loss() -> None:
    base = torch.full((1, 13), 1.0 / 13.0)
    support = torch.ones((1, 13), dtype=torch.bool)
    teacher_q = torch.zeros((1, 13))
    teacher_q[0, 5] = 1.0
    neutral = torch.zeros_like(base)
    aligned = torch.zeros_like(base)
    aligned[0, 5] = 4.0

    assert residual_q_loss(aligned, base, teacher_q, support) < residual_q_loss(
        neutral, base, teacher_q, support
    )


def test_ranking_metrics_respect_support_and_historical_action() -> None:
    probabilities = torch.zeros((2, 13))
    teacher_q = torch.zeros((2, 13))
    support = torch.zeros((2, 13), dtype=torch.bool)
    support[:, :3] = True
    probabilities[0, 1] = 1.0
    probabilities[1, 0] = 1.0
    teacher_q[0, 1] = 0.8
    teacher_q[1, 2] = 0.6
    historical = torch.tensor([1, 2])

    metrics = ranking_metrics(probabilities, teacher_q, support, historical)

    assert metrics["top1_agreement"] == 0.5
    assert metrics["mean_oracle_regret"] == pytest.approx(0.3)
    assert metrics["historical_top1_agreement"] == 1.0
    assert metrics["historical_mean_oracle_regret"] == 0.0


def test_residual_head_parameter_contract() -> None:
    head = CausalResidualQHead()
    assert sum(parameter.numel() for parameter in head.parameters()) == 11_713
    assert head(torch.zeros((3, 900))).shape == (3, 13)
