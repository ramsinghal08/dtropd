"""Robust aggregation operators (SPEC Sec. 2.6)."""

from __future__ import annotations

import pytest
import torch

from dtropd.robust import (
    aggregate,
    geometric_median,
    krum,
    trimmed_mean,
    weighted_sum,
)


def test_weighted_sum():
    g = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(weighted_sum(g), torch.tensor([4.0, 6.0]))
    assert torch.allclose(
        weighted_sum(g, torch.tensor([0.5, 0.0])), torch.tensor([0.5, 1.0])
    )


def test_trimmed_mean_ignores_outliers():
    honest = torch.ones(9, 3)
    grads = torch.cat([honest, torch.full((1, 3), 1e6)])
    assert torch.allclose(trimmed_mean(grads, 0.2), torch.ones(3), atol=1e-6)
    assert trimmed_mean(grads, 0.0).max() > 1e4  # plain mean is destroyed


def test_geometric_median_is_robust():
    grads = torch.cat([torch.ones(7, 2), torch.full((2, 2), 500.0)])
    gm = geometric_median(grads)
    assert torch.allclose(gm, torch.ones(2), atol=1e-2)


def test_geometric_median_matches_mean_for_symmetric_input():
    grads = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    assert torch.allclose(geometric_median(grads), torch.zeros(2), atol=1e-4)


def test_krum_selects_the_honest_cluster():
    torch.manual_seed(0)
    honest = torch.randn(8, 4) * 0.01 + 1.0
    byz = torch.full((2, 4), -50.0)
    grads = torch.cat([honest, byz])
    selected = krum(grads, n_byzantine=2)
    assert torch.allclose(selected, torch.ones(4), atol=0.1)


def test_krum_falls_back_when_too_few_vectors():
    grads = torch.tensor([[1.0], [2.0]])
    out = krum(grads, n_byzantine=1)
    assert out.shape == (1,)


def test_aggregate_dispatch_and_errors():
    grads = torch.ones(5, 3)
    for method in ("sum", "trimmed_mean", "geometric_median", "krum"):
        out = aggregate(grads, method)
        assert out.shape == (3,)
        assert torch.allclose(out, torch.full((3,), 5.0), atol=1e-3)
    with pytest.raises(ValueError):
        aggregate(grads, "nope")
    with pytest.raises(ValueError):
        weighted_sum(torch.ones(3))
