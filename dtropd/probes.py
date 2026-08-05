"""The Assumption-2 decay probe (SPEC Sec. 6, H5).

H5 asks for two things. The easy one is the per-capita simultaneity gap under a
fixed-degree topology. The hard one -- and the reason this file exists -- is:

    "**plus** directly test the decay assumption -- perturb agent ``j``'s policy,
     record induced change in ``Q^i`` vs. graph distance. *Confirmed if* ...
     sensitivity decays exponentially with hop distance."

That is a test *of the assumption* rather than a result derived *under* it, so
it cannot be computed from the training curves; it needs a deliberate
intervention. The probe perturbs one agent's policy and measures what happens to
every other agent's return as a function of hop distance on the communication
graph.

``Q^i`` is approximated by agent ``i``'s episode return under the perturbed
joint policy, measured against an unperturbed baseline on **identical
environment seeds**, so the difference isolates the perturbation rather than
environment noise. The approximation is stated here rather than hidden: what is
measured is the induced change in agent ``i``'s realised return, which is a
sampled estimate of the change in its value under the joint policy.

The probe never touches the training agents: it deep-copies the actor it
perturbs and restores it afterwards.
"""

from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np
import torch

from .config import EnvConfig
from .env import SyncVectorEnv
from .instrumentation import graph_distances, quantiles
from .utils import derive_seed, get_logger

__all__ = ["decay_probe", "DecayProbeResult"]

logger = get_logger("dtropd.probes")


class DecayProbeResult(dict):
    """A plain dict, so it serialises straight into the metrics file."""


@torch.no_grad()
def _rollout_returns(
    algo: Any,
    env_cfg: EnvConfig,
    episodes: int,
    seed: int,
) -> np.ndarray:
    """Per-agent mean episode return over ``episodes`` fixed-seed episodes."""
    num_envs = max(1, min(episodes, 8))
    vec = SyncVectorEnv(env_cfg, num_envs, seed=seed, autoreset=True)
    obs = vec.reset(seed=seed)
    totals = np.zeros((num_envs, env_cfg.n_agents), dtype=np.float64)
    completed = 0
    running = np.zeros_like(totals)
    for _ in range(env_cfg.max_cycles * (episodes // num_envs + 1)):
        if completed >= episodes:
            break
        actions = algo.act(obs, deterministic=True)
        step = vec.step(actions)
        running += step.rewards
        for i, info in enumerate(step.infos):
            if "episode" in info and completed < episodes:
                totals[i] += running[i]
                running[i] = 0.0
                completed += 1
    vec.close()
    return totals.sum(axis=0) / max(1, completed) * num_envs


def decay_probe(
    algo: Any,
    env_cfg: EnvConfig,
    *,
    n_perturbed: int = 3,
    episodes: int = 4,
    scale: float = 0.1,
    seed: int = 0,
    quantile_list: Sequence[float] = (0.05, 0.5, 0.95),
) -> DecayProbeResult:
    """Perturb agents' policies; record induced |ΔQ^i| against hop distance.

    Returns a dict with per-hop aggregates and the raw pairs, so the analysis
    can fit the exponential decay H5 predicts without re-running anything.
    """
    n_agents = env_cfg.n_agents
    rng = np.random.default_rng(derive_seed(seed, "decay_probe"))
    algo.eval_mode()
    try:
        baseline = _rollout_returns(algo, env_cfg, episodes, seed=seed)

        targets = sorted(
            rng.choice(n_agents, size=min(n_perturbed, n_agents), replace=False).tolist()
        )
        pairs: list[dict[str, Any]] = []
        for j in targets:
            actor = algo.agents[j].actor
            original = copy.deepcopy(actor.state_dict())
            gen = torch.Generator().manual_seed(derive_seed(seed, "perturb", j))
            with torch.no_grad():
                for p in actor.parameters():
                    noise = torch.randn(p.shape, generator=gen, dtype=p.dtype)
                    p.add_(noise * scale * (p.detach().std() + 1e-6))
            try:
                perturbed = _rollout_returns(algo, env_cfg, episodes, seed=seed)
            finally:
                actor.load_state_dict(original)

            hops = graph_distances(algo.neighbors, j)
            for i in range(n_agents):
                if i == j:
                    continue
                pairs.append(
                    {
                        "perturbed": j,
                        "observer": i,
                        "hops": int(hops[i]),
                        "delta_return": float(abs(perturbed[i] - baseline[i])),
                        "signed_delta_return": float(perturbed[i] - baseline[i]),
                    }
                )
    finally:
        algo.train_mode()

    by_hop: dict[str, Any] = {}
    for hop in sorted({p["hops"] for p in pairs}):
        values = [p["delta_return"] for p in pairs if p["hops"] == hop]
        by_hop[str(hop)] = quantiles(values, quantile_list, "")

    fit = fit_decay(by_hop, seed=seed)
    return DecayProbeResult(
        {
            "perturbed_agents": targets,
            "episodes": episodes,
            "scale": scale,
            "q_proxy": "abs change in the observer's mean episode return, paired env seeds",
            "graph_diameter": int(max((p["hops"] for p in pairs), default=0)),
            "by_hop": by_hop,
            "fit": fit,
            "pairs": pairs,
        }
    )


#: SPEC Sec. 12 / D11(a): a decay slope from fewer than this many hop levels is
#: not interpretable and must be reported as such.
MIN_HOP_LEVELS_FOR_A_FIT = 4


def fit_decay(
    by_hop: dict[str, Any], seed: int = 0, n_boot: int = 2000
) -> dict[str, Any]:
    """Fit ``log(mean |dQ^i|)`` against hop distance, with a bootstrap CI.

    H5 predicts sensitivity decays *exponentially* with hop distance, i.e. a
    negative slope in log space. SPEC Sec. 12 / D11(a): a slope from fewer than
    :data:`MIN_HOP_LEVELS_FOR_A_FIT` hop levels is **uninterpretable**, and this
    function says so rather than returning a number that looks like evidence.
    """
    hops = sorted(int(h) for h in by_hop if int(h) > 0)
    means = [by_hop[str(h)].get("mean", float("nan")) for h in hops]
    usable = [(h, m) for h, m in zip(hops, means) if np.isfinite(m) and m > 0.0]
    n_levels = len(usable)

    out: dict[str, Any] = {
        "hop_levels_used": n_levels,
        "hops": [h for h, _ in usable],
        "mean_abs_delta_return": [m for _, m in usable],
        "min_hop_levels_required": MIN_HOP_LEVELS_FOR_A_FIT,
        "interpretable": n_levels >= MIN_HOP_LEVELS_FOR_A_FIT,
        "log_slope": float("nan"),
        "log_slope_ci": {},
    }
    if n_levels < 2:
        out["note"] = (
            f"only {n_levels} usable hop level(s): no slope can be fitted at all. "
            "Use a ring or path topology with a larger diameter (D11(a))."
        )
        return out

    x = np.asarray([h for h, _ in usable], dtype=np.float64)
    y = np.log(np.asarray([m for _, m in usable], dtype=np.float64))
    out["log_slope"] = float(np.polyfit(x, y, 1)[0])

    # Bootstrap over hop levels; with few levels the CI is deliberately wide,
    # which is the honest representation of a low-powered probe.
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_levels, size=n_levels)
        if len(set(x[idx].tolist())) < 2:
            continue
        slopes.append(float(np.polyfit(x[idx], y[idx], 1)[0]))
    if slopes:
        out["log_slope_ci"] = {
            "lo": float(np.quantile(slopes, 0.025)),
            "hi": float(np.quantile(slopes, 0.975)),
            "n_boot": len(slopes),
        }
    out["expected"] = "negative slope = exponential decay with hop distance"
    if not out["interpretable"]:
        out["note"] = (
            f"UNINTERPRETABLE: fitted from {n_levels} hop level(s), fewer than the "
            f"{MIN_HOP_LEVELS_FOR_A_FIT} required by SPEC Sec. 12 / D11(a). Use a "
            "ring or path topology over 12-24 agents to obtain diameter >= 6."
        )
    return out
