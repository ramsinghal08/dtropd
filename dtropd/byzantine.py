"""Byzantine adversaries (SPEC Sec. 5 Phase 7, H6).

:mod:`dtropd.robust` holds the *defences* -- trimmed mean, geometric median,
Krum. This module holds the *attacks* they are defended against, kept separate
so neither is written with the other's convenience in mind.

H6 predicts that "robust aggregation degrades gracefully to `f ≈ 0.4` while
plain-sum collapses early". That prediction is only worth testing against an
adversary that is actually trying, which is why the interesting mode here is
``adaptive``.

The threat model
----------------

Byzantine agents are **omniscient**: they see the honest gradients of the round
they are attacking and may construct their message from them. This is the
standard strong model, and anything weaker makes a defence look better than it
is -- a blind adversary is defeated by the *variance* of its own attack rather
than by the defence.

Why ``adaptive`` is a different thing from ``gaussian``
--------------------------------------------------------

``sign_flip`` and ``gaussian`` are loud. They move the aggregate a long way, but
they also place the Byzantine vectors far outside the honest cloud, where
coordinate-wise trimming and Krum's distance score find them easily. A defence
that stops them has proved very little.

``adaptive`` is the attack from Baruch et al., *"A Little Is Enough"* (2019):
perturb each coordinate by a *fraction of the honest standard deviation*, in a
consistent direction. Each Byzantine vector then sits comfortably inside the
honest range -- within the untrimmed middle, and close to the honest cloud by
Euclidean distance -- so the defences see nothing anomalous, while the shared
sign of the perturbation means the contributions accumulate rather than cancel.
The attack trades magnitude for survivability, and against an averaging defence
that is the correct trade.

This is why :data:`StressConfig.byzantine_scale` is measured in honest standard
deviations for this mode, and why a scale above 1 earns a config warning: past
that point the attack has left the honest range and is merely a noisier
``gaussian``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .utils import derive_seed

__all__ = [
    "byzantine_agents",
    "apply_attack",
    "ATTACKS",
]

ATTACKS = ("none", "sign_flip", "gaussian", "adaptive")


def byzantine_agents(n_agents: int, n_byzantine: int, seed: int) -> list[int]:
    """Pick the Byzantine set deterministically from the run seed.

    Fixed for the whole run (an agent does not become honest between epochs) but
    varying across seeds, so a result cannot be an artifact of one unlucky
    assignment -- with paired seeds, every algorithm in a cell faces the *same*
    Byzantine set.
    """
    if n_byzantine <= 0:
        return []
    if n_byzantine >= n_agents:
        raise ValueError(
            f"{n_byzantine} Byzantine agents of {n_agents} leaves no honest majority"
        )
    rng = np.random.default_rng(derive_seed(seed, "byzantine"))
    return sorted(rng.choice(n_agents, size=n_byzantine, replace=False).tolist())


def apply_attack(
    per_edge: Tensor,
    byzantine_rows: list[int],
    mode: str,
    scale: float,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Replace the Byzantine rows of a per-edge gradient stack with an attack.

    ``per_edge`` is ``[n_edges, n_params]``: one flattened gradient per
    contributing neighbour. ``byzantine_rows`` indexes the rows contributed by
    Byzantine neighbours. Returns a new tensor; the input is not mutated.
    """
    if mode == "none" or not byzantine_rows:
        return per_edge
    if mode not in ATTACKS:
        raise ValueError(f"unknown byzantine mode {mode!r}; expected one of {ATTACKS}")

    out = per_edge.clone()
    rows = torch.as_tensor(byzantine_rows, dtype=torch.long, device=per_edge.device)
    honest_mask = torch.ones(per_edge.shape[0], dtype=torch.bool, device=per_edge.device)
    honest_mask[rows] = False
    honest = per_edge[honest_mask]

    if mode == "sign_flip":
        out[rows] = -per_edge[rows] * scale
        return out

    if mode == "gaussian":
        magnitude = per_edge.abs().mean().clamp(min=1e-12)
        noise = torch.randn(
            (len(byzantine_rows), per_edge.shape[1]),
            generator=generator,
            dtype=per_edge.dtype,
            device=per_edge.device,
        )
        out[rows] = noise * magnitude * scale
        return out

    # --- adaptive: sit inside the honest range and bias it ------------------ #
    if honest.shape[0] == 0:
        # No honest reference to hide behind; the attack is undefined rather
        # than zero, and silently degrading it would overstate the defence.
        raise ValueError(
            "the adaptive attack needs at least one honest gradient to fit "
            "itself to; every edge is Byzantine"
        )

    mean = honest.mean(dim=0)
    if honest.shape[0] == 1:
        # A single honest sample has no spread to hide within. Fall back to the
        # gradient's own scale so the attack stays defined, and stay quiet about
        # nothing -- this is recorded by the caller as a degenerate cell.
        std = honest.abs().squeeze(0).clamp(min=1e-12)
    else:
        std = honest.std(dim=0, unbiased=True).clamp(min=1e-12)

    # One consistent direction for every Byzantine row: the whole point is that
    # the contributions accumulate instead of cancelling. Opposing the honest
    # mean drags the aggregate toward zero and past it.
    direction = -torch.sign(mean)
    direction[direction == 0] = 1.0
    out[rows] = (mean + direction * scale * std).unsqueeze(0).expand(len(rows), -1)
    return out
