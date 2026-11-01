"""Measurement helpers for instrumentation.

SPEC Sec. 5's acceptance bar is that a *single* training run's metrics
file supports every H1-H7 measurement without re-running. That makes the
question "what does each hypothesis actually need?" a code question, so the
answers live here rather than being scattered through the update loop:

* :class:`SignFlipTracker` -- H1: per-edge social-gradient sign flips per epoch,
  bucketed by ``|D_ij - delta_i|``. A "flip" is a negative cosine between an
  edge's gradient this epoch and last.
* :func:`quantiles` -- SPEC Sec. 11: log distributions, not means. The
  ``delta_base`` failure was invisible in the mean and obvious in the quantiles.
* :func:`graph_distances` -- H5: hop distance on the communication graph, for
  the decay probe.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "quantiles",
    "SignFlipTracker",
    "graph_distances",
    "bucket_label",
]


def quantiles(
    values: Sequence[float] | np.ndarray,
    qs: Sequence[float],
    prefix: str = "",
) -> dict[str, float]:
    """``{prefix}p05``-style quantile dict, plus mean/min/max/count."""
    arr = np.asarray(list(values), dtype=np.float64).ravel()
    if arr.size == 0:
        return {f"{prefix}count": 0}
    out = {f"{prefix}p{int(round(q * 100)):02d}": float(np.quantile(arr, q)) for q in qs}
    out[f"{prefix}mean"] = float(arr.mean())
    out[f"{prefix}min"] = float(arr.min())
    out[f"{prefix}max"] = float(arr.max())
    out[f"{prefix}count"] = int(arr.size)
    return out


def bucket_label(gap: float, edges: Sequence[float]) -> str:
    """Label the bucket of ``|D_ij - delta_i|`` that ``gap`` falls into."""
    for lo, hi in zip(edges, edges[1:]):
        if lo <= gap < hi:
            return f"[{lo:g},{hi:g})"
    if gap < edges[0]:
        return f"[<{edges[0]:g})"
    return f"[>={edges[-1]:g})"


class SignFlipTracker:
    """H1's measurement: per-edge social-gradient sign flips per epoch.

    SPEC Sec. 6 H1: "per-edge social-gradient sign flips per epoch, bucketed by
    ``|D_ij - delta_i|``. *Confirmed if* the switched variant spikes near the
    boundary and the attenuated one doesn't."

    A flip is recorded when the cosine similarity between an edge's social
    gradient this epoch and the same edge's gradient last epoch is negative --
    the gradient reversed direction, which is what chattering looks like from
    the inside. Bucketing by distance to the radius boundary is what makes the
    hypothesis falsifiable: a smooth gate should show no excess of flips near
    ``D_ij = delta_i``, a switched one should.
    """

    def __init__(self, buckets: Sequence[float]):
        self.buckets = list(buckets)
        self._previous: dict[tuple[int, int], np.ndarray] = {}
        self.reset_epoch()

    def reset_epoch(self) -> None:
        self.flips: dict[str, int] = {}
        self.totals: dict[str, int] = {}
        self.cosines: list[float] = []
        self.gaps: list[float] = []

    def observe(
        self, agent: int, neighbor: int, gradient: Tensor, gap: float
    ) -> bool | None:
        """Record one edge's gradient. Returns the flip flag, or None on epoch 0."""
        key = (agent, neighbor)
        flat = gradient.detach().reshape(-1).cpu().numpy().astype(np.float64)
        previous = self._previous.get(key)
        self._previous[key] = flat

        if previous is None:
            return None
        denom = float(np.linalg.norm(previous) * np.linalg.norm(flat))
        if denom <= 0.0:
            return None
        cosine = float(np.dot(previous, flat) / denom)
        label = bucket_label(gap, self.buckets)
        flipped = cosine < 0.0
        self.totals[label] = self.totals.get(label, 0) + 1
        self.flips[label] = self.flips.get(label, 0) + int(flipped)
        self.cosines.append(cosine)
        self.gaps.append(gap)
        return flipped

    def summary(self, qs: Sequence[float]) -> dict[str, Any]:
        total = sum(self.totals.values())
        flips = sum(self.flips.values())
        out: dict[str, Any] = {
            "h1/sign_flips_total": flips,
            "h1/edges_compared": total,
            "h1/sign_flip_rate": (flips / total) if total else float("nan"),
            "h1/sign_flips_by_bucket": {
                label: {
                    "flips": self.flips.get(label, 0),
                    "edges": count,
                    "rate": self.flips.get(label, 0) / count if count else float("nan"),
                }
                for label, count in sorted(self.totals.items())
            },
            "h1/bucket_edges": list(self.buckets),
        }
        out.update(quantiles(self.cosines, qs, "h1/grad_cosine_"))
        out.update(quantiles(self.gaps, qs, "h1/boundary_gap_"))
        return out


def graph_distances(neighbors: Sequence[Sequence[int]], source: int) -> list[int]:
    """Hop distance from ``source`` to every agent (-1 if unreachable)."""
    n = len(neighbors)
    dist = [-1] * n
    dist[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for nxt in neighbors[node]:
            if dist[nxt] == -1:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist


def pairwise_policy_dispersion(
    actors: Sequence[Any], probe_obs: Tensor
) -> np.ndarray:
    """All ordered-by-index pairwise ``E_s[KL(pi_i || pi_j)]`` on a probe batch.

    D9's dispersion measure. Deliberately computed the same way for *every*
    algorithm -- including IPPO, which exchanges nothing -- so the consensus
    pull of the distillation arms has an uncontaminated control to be compared
    against. The probe batch is shared, so this is not the own-rollout ``D_ij``
    of invariant I3 and is never used as a trust signal; it is a diagnostic.
    """
    from .kl import gaussian_kl

    with torch.no_grad():
        params = [actor(probe_obs) for actor in actors]
        out = []
        for i in range(len(actors)):
            for j in range(len(actors)):
                if i >= j:
                    continue
                mu_i, ls_i = params[i]
                mu_j, ls_j = params[j]
                out.append(float(gaussian_kl(mu_i, ls_i, mu_j, ls_j).mean()))
    return np.asarray(out, dtype=np.float64)
