"""Continuous attenuation w_ij: w(0) = 1, monotone decreasing, C^1 at D = delta.

SPEC Sec. 2.3::

    w_ij = exp( -gamma * [ softplus(D_ij - delta_i) - softplus(-delta_i) ] )

Phase 3 acceptance: "test_attenuation.py confirms w(0)=1 and derivative
continuity at D = delta".
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dtropd.algorithms import trust_radius, trust_weight

DELTAS = [0.01, 0.05, 0.1, 0.5, 1.0]
GAMMAS = [0.5, 1.0, 2.0, 5.0]


# --------------------------------------------------------------------------- #
# w(0) = 1  (invariant I7)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("delta", DELTAS)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_w_is_one_at_zero_divergence(delta, gamma):
    """An identical neighbour must be fully trusted, for every (delta, gamma)."""
    w = trust_weight(torch.tensor(0.0, dtype=torch.float64), delta, gamma)
    assert float(w) == pytest.approx(1.0, abs=1e-12)


def test_w_normalisation_term_is_not_omitted():
    """Dropping ``- softplus(-delta)`` breaks w(0) = 1 -- that is why it is there."""
    delta, gamma = 0.1, 2.0
    correct = float(trust_weight(torch.tensor(0.0, dtype=torch.float64), delta, gamma))

    d = torch.tensor(0.0, dtype=torch.float64)
    unnormalised = float(torch.exp(-gamma * torch.nn.functional.softplus(d - delta)))

    assert correct == pytest.approx(1.0, abs=1e-12)
    assert unnormalised < 0.9  # ~exp(-2 * softplus(-0.1)) = 0.176
    assert abs(unnormalised - 1.0) > 0.1


# --------------------------------------------------------------------------- #
# Monotonicity and range
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("delta", DELTAS)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_w_is_monotone_decreasing_in_divergence(delta, gamma):
    """More behavioural divergence must never buy more trust."""
    d = torch.linspace(0.0, 5.0, 2001, dtype=torch.float64)
    w = trust_weight(d, delta, gamma)
    diffs = w[1:] - w[:-1]
    assert torch.all(diffs <= 1e-12), f"w increased somewhere: max diff {diffs.max()}"
    assert torch.all(w <= 1.0 + 1e-12)
    assert torch.all(w > 0.0)


def test_w_decays_towards_zero_for_large_divergence():
    w = trust_weight(torch.tensor([0.0, 1.0, 10.0, 100.0], dtype=torch.float64), 0.1, 2.0)
    assert float(w[0]) == pytest.approx(1.0)
    assert float(w[-1]) < 1e-80


def test_gamma_zero_disables_attenuation():
    """gamma = 0 means every neighbour is trusted equally -- the NaiveDistill limit."""
    d = torch.tensor([0.0, 1.0, 50.0], dtype=torch.float64)
    assert torch.allclose(trust_weight(d, 0.1, 0.0), torch.ones(3, dtype=torch.float64))


# --------------------------------------------------------------------------- #
# C^1 at the radius boundary
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("delta", DELTAS)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_w_is_c1_at_the_radius_boundary(delta, gamma):
    """Finite-difference derivative is continuous across D = delta."""
    h = 1e-6
    left = [delta - 2 * h, delta - h, delta]
    right = [delta, delta + h, delta + 2 * h]

    def fd(points):
        w = trust_weight(torch.tensor(points, dtype=torch.float64), delta, gamma)
        return (float(w[-1]) - float(w[0])) / (points[-1] - points[0])

    d_left, d_right = fd(left), fd(right)
    assert d_left < 0 and d_right < 0
    assert abs(d_left - d_right) < 1e-5 * max(1.0, abs(d_left)), (
        f"derivative jumps at D=delta: {d_left} vs {d_right}"
    )

    # Analytic check: d/dD w = -gamma * sigmoid(D - delta) * w, so at D = delta
    # the slope is exactly -gamma/2 * w(delta).
    w_at = float(trust_weight(torch.tensor(delta, dtype=torch.float64), delta, gamma))
    assert d_left == pytest.approx(-0.5 * gamma * w_at, rel=1e-3)


def test_w_is_smooth_via_autograd():
    """The second derivative exists too: no kink anywhere, including at delta."""
    delta, gamma = 0.1, 2.0
    d = torch.tensor(delta, dtype=torch.float64, requires_grad=True)
    w = trust_weight(d, delta, gamma)
    (g1,) = torch.autograd.grad(w, d, create_graph=True)
    (g2,) = torch.autograd.grad(g1, d)
    assert torch.isfinite(g1) and torch.isfinite(g2)
    assert float(g1.detach()) == pytest.approx(-0.5 * gamma * float(w.detach()), rel=1e-9)


# --------------------------------------------------------------------------- #
# The H1 ablation arms must be measurably worse-behaved
# --------------------------------------------------------------------------- #


def test_hard_cutoff_is_discontinuous_at_the_boundary():
    delta = 0.1
    below = trust_weight(torch.tensor(delta - 1e-9, dtype=torch.float64), delta, 2.0, "hard_cutoff")
    above = trust_weight(torch.tensor(delta + 1e-9, dtype=torch.float64), delta, 2.0, "hard_cutoff")
    assert float(below) == 1.0 and float(above) == 0.0


def test_hard_switch_is_c0_but_not_c1():
    """The kink at D = delta is the whole point of the H1 contrast."""
    delta, gamma, h = 0.1, 2.0, 1e-6

    def fd(a, b):
        wa = float(trust_weight(torch.tensor(a, dtype=torch.float64), delta, gamma, "hard_switch"))
        wb = float(trust_weight(torch.tensor(b, dtype=torch.float64), delta, gamma, "hard_switch"))
        return (wb - wa) / (b - a)

    left = fd(delta - 2 * h, delta - h)
    right = fd(delta + h, delta + 2 * h)
    assert left == pytest.approx(0.0, abs=1e-9)  # flat inside the radius
    assert right == pytest.approx(-gamma, rel=1e-3)  # slope -gamma outside
    assert abs(left - right) > 0.1  # derivative jumps

    # Still continuous in value, and still normalised at zero divergence.
    assert float(
        trust_weight(torch.tensor(0.0, dtype=torch.float64), delta, gamma, "hard_switch")
    ) == pytest.approx(1.0)


def test_unknown_attenuation_mode_raises():
    with pytest.raises(ValueError):
        trust_weight(torch.tensor(0.1), 0.1, 1.0, "nope")


# --------------------------------------------------------------------------- #
# Trust radius
# --------------------------------------------------------------------------- #


def test_trust_radius_contracts_with_nonstationarity():
    """delta = delta_base * exp(-kappa * Delta): monotone, equal to base at 0."""
    base, kappa = 0.1, 100.0
    assert trust_radius(base, kappa, 0.0) == pytest.approx(base)
    values = [trust_radius(base, kappa, x) for x in np.linspace(0, 0.05, 50)]
    assert values == sorted(values, reverse=True)
    assert trust_radius(base, kappa, 0.02) == pytest.approx(base * math.exp(-2.0))


def test_trust_radius_dynamic_range_over_the_spec_kappa_eta_band():
    """kappa * eta in [1, 3] gives the radius real dynamic range (SPEC Sec. 2.2)."""
    base, eta = 0.1, 0.02
    for product in (1.0, 2.0, 3.0):
        kappa = product / eta
        # Moving by a full eta of nonstationarity must shrink the radius by
        # a factor of e^-product, i.e. between ~2.7x and ~20x.
        ratio = trust_radius(base, kappa, eta) / base
        assert ratio == pytest.approx(math.exp(-product))
        assert 0.04 < ratio < 0.4
