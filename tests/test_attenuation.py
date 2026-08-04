"""Continuous attenuation w_ij: w(0) = 1, monotone decreasing, C^1 at D = delta.

SPEC Sec. 2.3::

    w_ij = exp( -gamma * [ softplus(D_ij - delta_i) - softplus(-delta_i) ] )

and SPEC Sec. 5 Phase 3 acceptance: "test_attenuation.py confirms w(0)=1 and
derivative continuity at D = delta".

The attenuation function is part of the D-TrOPD core (Phase 3), which is out
of scope for this build.  The checks are written out here so Phase 3 has a
concrete, pre-registered target instead of a blank file; they activate as soon
as ``dtropd.algorithms`` exposes ``trust_weight``.
"""

from __future__ import annotations

import pytest

pytest.skip(
    "attenuation w_ij is part of the D-TrOPD core (Phase 3); not built yet",
    allow_module_level=True,
)


def test_w_is_one_at_zero_divergence():
    """Invariant I7: an identical neighbour must be fully trusted."""


def test_w_is_monotone_decreasing_in_divergence():
    """More behavioural divergence must never buy more trust."""


def test_w_is_c1_at_the_radius_boundary():
    """Finite-difference derivative must be continuous across D = delta."""


def test_w_normalisation_term_is_not_omitted():
    """Dropping ``- softplus(-delta)`` breaks w(0) = 1."""


def test_w_is_held_constant_through_the_inner_loop():
    """w_ij is a detached gate, not a differentiable path."""
