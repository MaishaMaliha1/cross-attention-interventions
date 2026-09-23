import torch

from cross_attention_interventions.math_utils import (
    contribution_norms,
    kl_sensitivity,
    projected_value_norms,
)


def test_projected_value_norms_matches_direct_computation():
    values = torch.tensor([[[[3.0, 4.0], [1.0, 0.0]], [[2.0, 0.0], [0.0, 2.0]]]])
    weight = torch.eye(4)
    got = projected_value_norms(values, weight)
    expected = torch.tensor([[[5.0, 1.0], [2.0, 2.0]]])
    assert torch.allclose(got, expected)


def test_contribution_norms_multiplies_attention_by_projected_magnitude():
    values = torch.tensor([[[[3.0, 4.0], [1.0, 0.0]]]])
    weight = torch.eye(2)
    attention = torch.tensor([[[[0.25, 0.75]]]])
    got = contribution_norms(attention, values, weight)
    assert torch.allclose(got, torch.tensor([[[[1.25, 0.75]]]]))


def test_kl_is_zero_for_identical_distributions():
    x = torch.tensor([[1.0, 3.0, 2.0]])
    assert torch.allclose(kl_sensitivity(x, x), torch.zeros(1, dtype=torch.float64), atol=1e-12)


def test_kl_increases_for_changed_distribution():
    p = torch.tensor([[8.0, 1.0, 1.0]])
    q = torch.tensor([[1.0, 8.0, 1.0]])
    assert kl_sensitivity(p, q).item() > 0.5
