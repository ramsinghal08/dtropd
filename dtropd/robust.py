"""Byzantine-robust aggregation operators (SPEC Sec. 2.6).

Pure tensor functions over a stack of per-neighbour gradient vectors of shape
``(n_neighbours, dim)``.  They are self-contained and dependency-free; the
D-TrOPD update wires them in behind ``DistillConfig.aggregation`` (default
``"sum"``, i.e. the plain weighted sum of SPEC Sec. 2.3).

Provided here in Phase 0 because they are part of the declared repository
structure and are pure numerics with no coupling to the algorithm; they are
*exercised* by the Phase 7 robustness experiments.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["weighted_sum", "trimmed_mean", "geometric_median", "krum", "aggregate"]


def _check(grads: Tensor) -> None:
    if grads.dim() != 2:
        raise ValueError(f"expected a (n, dim) stack of gradients, got {tuple(grads.shape)}")
    if grads.shape[0] == 0:
        raise ValueError("cannot aggregate an empty gradient stack")


def weighted_sum(grads: Tensor, weights: Tensor | None = None) -> Tensor:
    """``sum_j w_j g_j`` -- the non-robust default of SPEC Sec. 2.3."""
    _check(grads)
    if weights is None:
        return grads.sum(dim=0)
    return (grads * weights.reshape(-1, 1)).sum(dim=0)


def trimmed_mean(grads: Tensor, trim_fraction: float = 0.2) -> Tensor:
    """Coordinate-wise trimmed mean.

    Drops ``floor(trim_fraction * n)`` values from each end of every coordinate
    independently, then averages the rest.
    """
    _check(grads)
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must lie in [0, 0.5)")
    n = grads.shape[0]
    k = int(trim_fraction * n)
    if k == 0:
        return grads.mean(dim=0)
    if n - 2 * k <= 0:
        return grads.median(dim=0).values
    sorted_grads, _ = torch.sort(grads, dim=0)
    return sorted_grads[k : n - k].mean(dim=0)


def geometric_median(
    grads: Tensor,
    max_iter: int = 100,
    tol: float = 1e-7,
    eps: float = 1e-8,
) -> Tensor:
    """Weiszfeld iteration for the L2 geometric median."""
    _check(grads)
    z = grads.mean(dim=0)
    for _ in range(max_iter):
        dist = torch.norm(grads - z, dim=1).clamp_min(eps)
        w = 1.0 / dist
        z_new = (grads * w.reshape(-1, 1)).sum(dim=0) / w.sum()
        if torch.norm(z_new - z) <= tol * max(1.0, float(torch.norm(z))):
            z = z_new
            break
        z = z_new
    return z


def krum(grads: Tensor, n_byzantine: int = 1, multi: int = 1) -> Tensor:
    """(Multi-)Krum: pick the vector(s) closest to their nearest neighbours.

    Each candidate scores the sum of squared distances to its
    ``n - n_byzantine - 2`` nearest other candidates; the ``multi`` lowest
    scorers are averaged.
    """
    _check(grads)
    n = grads.shape[0]
    if n_byzantine < 0:
        raise ValueError("n_byzantine must be >= 0")
    if multi < 1 or multi > n:
        raise ValueError("multi must lie in [1, n]")
    k = n - n_byzantine - 2
    if k < 1:
        # Not enough vectors for the Krum guarantee; fall back to the median,
        # which is still a breakdown-point-1/2 estimator.
        return grads.median(dim=0).values
    dists = torch.cdist(grads, grads) ** 2
    dists.fill_diagonal_(float("inf"))
    nearest, _ = torch.sort(dists, dim=1)
    scores = nearest[:, :k].sum(dim=1)
    idx = torch.argsort(scores)[:multi]
    return grads[idx].mean(dim=0)


def aggregate(
    grads: Tensor,
    method: str,
    *,
    weights: Tensor | None = None,
    trim_fraction: float = 0.2,
    n_byzantine: int = 1,
) -> Tensor:
    """Dispatch on ``DistillConfig.aggregation``."""
    if method == "sum":
        return weighted_sum(grads, weights)
    if method == "trimmed_mean":
        return trimmed_mean(grads, trim_fraction) * grads.shape[0]
    if method == "geometric_median":
        return geometric_median(grads) * grads.shape[0]
    if method == "krum":
        return krum(grads, n_byzantine) * grads.shape[0]
    raise ValueError(f"unknown aggregation method {method!r}")
