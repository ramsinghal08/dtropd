"""Closed-form Gaussian KL utilities -- the single source of truth.

SPEC Sec. 2.5: policies are tanh-squashed diagonal Gaussians.  KL is invariant
under the shared tanh diffeomorphism, so it is computed analytically on the
**pre-squash latent** ``(mu, log_sigma)``.  It is never estimated by Monte
Carlo sampling anywhere in the training path.

Invariant I2 concerns the *social* (distillation) KL: there is exactly one
direction in this codebase and every social caller passes the **student**
first, i.e. ``KL(pi_student || pi_teacher)`` -- reverse KL.  There is no
``direction`` argument, no ``forward=`` flag and no branch anywhere.

The helper below is a plain, direction-agnostic ``KL(p || q)``; the invariant
is enforced at the call sites (and by ``tests/test_invariants.py``).  The one
non-social caller is the PPO early stop, which SPEC Sec. 2.4 defines
explicitly as ``KL(pi^(t) || pi_theta)`` -- old policy first.  That is a
different quantity from the social divergence and is not a direction switch.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "gaussian_kl",
    "mean_gaussian_kl",
    "gaussian_entropy",
]


def gaussian_kl(
    mu_p: Tensor,
    log_std_p: Tensor,
    mu_q: Tensor,
    log_std_q: Tensor,
) -> Tensor:
    """``KL( N(mu_p, sigma_p^2) || N(mu_q, sigma_q^2) )`` summed over action dims.

    Per dimension (SPEC Sec. 2.5)::

        log(sigma_q / sigma_p) + (sigma_p^2 + (mu_p - mu_q)^2) / (2 sigma_q^2) - 0.5

    Parameters have shape ``(..., action_dim)``; the result has shape ``(...)``.

    The value is identical for the tanh-squashed distributions because the two
    policies share the same diffeomorphism, so the Jacobian terms cancel.
    """
    if not (mu_p.shape == log_std_p.shape == mu_q.shape == log_std_q.shape):
        raise ValueError(
            "gaussian_kl expects matching shapes, got "
            f"{tuple(mu_p.shape)}, {tuple(log_std_p.shape)}, "
            f"{tuple(mu_q.shape)}, {tuple(log_std_q.shape)}"
        )
    var_p = torch.exp(2.0 * log_std_p)
    inv_var_q = torch.exp(-2.0 * log_std_q)
    per_dim = (
        (log_std_q - log_std_p)
        + 0.5 * (var_p + (mu_p - mu_q) ** 2) * inv_var_q
        - 0.5
    )
    return per_dim.sum(dim=-1)


def mean_gaussian_kl(
    mu_p: Tensor,
    log_std_p: Tensor,
    mu_q: Tensor,
    log_std_q: Tensor,
) -> Tensor:
    """``E_s[ KL(p(.|s) || q(.|s)) ]`` -- the state-average of :func:`gaussian_kl`.

    The expectation is over whatever states were used to produce the batched
    parameters.  Callers are responsible for passing the agent's **own** rollout
    states (invariant I3).
    """
    return gaussian_kl(mu_p, log_std_p, mu_q, log_std_q).mean()


def gaussian_entropy(log_std: Tensor) -> Tensor:
    """Differential entropy of the *latent* diagonal Gaussian, summed over dims.

    This is the entropy of the pre-squash Gaussian, not of the tanh-squashed
    action distribution (which additionally carries an ``E[log|det J|]`` term).
    It is used as the PPO entropy bonus and as the quantity tracked for H2.
    """
    const = 0.5 * torch.log(torch.tensor(2.0 * torch.pi * torch.e, dtype=log_std.dtype))
    return (log_std + const).sum(dim=-1)
