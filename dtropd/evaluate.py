"""Evaluation rollouts and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from .config import Config, EnvConfig
from .env import SyncVectorEnv, ZoneKind
from .utils import make_rng

__all__ = ["EvalResult", "evaluate_policy", "evaluate_random_policy"]


@dataclass
class EvalResult:
    """Aggregate statistics over a batch of evaluation episodes."""

    episodes: int
    return_mean: float
    return_std: float
    return_sem: float
    captures_mean: float
    length_mean: float
    zone_returns: dict[str, float] = field(default_factory=dict)
    per_episode_returns: list[float] = field(default_factory=list)
    #: Mean return of EACH agent across the evaluation episodes. Kept so a
    #: SUBSET can be averaged: H6's headline is the honest agents' return, and
    #: folding the saboteurs in measures the attack's self-harm alongside the
    #: defence's success -- a number that moves with `f` even when the defence
    #: is perfect. The environment has always exposed this (env.py, "episode"/
    #: "agent_returns"); it was the evaluation path that dropped it.
    agent_return_means: list[float] = field(default_factory=list)

    def subset_return(self, agents: Sequence[int]) -> float:
        """Mean return over ``agents`` only."""
        if not self.agent_return_means:
            return float("nan")
        if not agents:
            return float("nan")
        return float(np.mean([self.agent_return_means[i] for i in agents]))

    def to_dict(self, prefix: str = "") -> dict[str, Any]:
        out = {
            f"{prefix}episodes": self.episodes,
            f"{prefix}return_mean": self.return_mean,
            f"{prefix}return_std": self.return_std,
            f"{prefix}return_sem": self.return_sem,
            f"{prefix}captures_mean": self.captures_mean,
            f"{prefix}length_mean": self.length_mean,
        }
        for zone, value in self.zone_returns.items():
            out[f"{prefix}zone_return/{zone}"] = value
        return out


def _summarise(
    returns: list[float],
    captures: list[float],
    lengths: list[int],
    zone_returns: list[np.ndarray],
    agent_returns: list[np.ndarray] | None = None,
) -> EvalResult:
    arr = np.asarray(returns, dtype=np.float64)
    n = max(1, len(arr))
    zr = np.stack(zone_returns).mean(axis=0) if zone_returns else np.zeros(len(ZoneKind.NAMES))
    per_agent = (
        np.stack(agent_returns).mean(axis=0).tolist() if agent_returns else []
    )
    return EvalResult(
        episodes=len(returns),
        return_mean=float(arr.mean()) if len(arr) else float("nan"),
        return_std=float(arr.std()) if len(arr) else float("nan"),
        return_sem=float(arr.std() / np.sqrt(n)) if len(arr) else float("nan"),
        captures_mean=float(np.mean(captures)) if captures else float("nan"),
        length_mean=float(np.mean(lengths)) if lengths else float("nan"),
        zone_returns={name: float(zr[i]) for i, name in enumerate(ZoneKind.NAMES)},
        per_episode_returns=[float(x) for x in returns],
        agent_return_means=[float(x) for x in per_agent],
    )


def _run_episodes(
    env_cfg: EnvConfig,
    n_episodes: int,
    seed: int,
    action_fn,
    num_envs: int | None = None,
) -> EvalResult:
    num_envs = num_envs or min(n_episodes, 8)
    vec = SyncVectorEnv(env_cfg, num_envs, seed=seed, autoreset=True)
    obs = vec.reset(seed=seed)
    returns: list[float] = []
    captures: list[float] = []
    lengths: list[int] = []
    zone_returns: list[np.ndarray] = []
    agent_returns: list[np.ndarray] = []

    max_steps = env_cfg.max_cycles * (n_episodes // num_envs + 2)
    for _ in range(max_steps):
        if len(returns) >= n_episodes:
            break
        actions = action_fn(obs)
        step = vec.step(actions)
        obs = step.obs
        for info in step.infos:
            ep = info.get("episode")
            if ep is not None and len(returns) < n_episodes:
                returns.append(ep["return"])
                captures.append(ep["captures"])
                lengths.append(ep["length"])
                zone_returns.append(ep["zone_returns"])
                if "agent_returns" in ep:
                    agent_returns.append(np.asarray(ep["agent_returns"], dtype=np.float64))
    vec.close()
    return _summarise(returns, captures, lengths, zone_returns, agent_returns)


def evaluate_policy(
    algo,
    env_cfg: EnvConfig,
    n_episodes: int = 8,
    seed: int = 0,
    deterministic: bool = True,
    num_envs: int | None = None,
) -> EvalResult:
    """Roll out the current policies and summarise team return."""
    algo.eval_mode()
    try:
        with torch.no_grad():
            result = _run_episodes(
                env_cfg,
                n_episodes,
                seed,
                lambda obs: algo.act(obs, deterministic=deterministic),
                num_envs=num_envs,
            )
    finally:
        algo.train_mode()
    return result


def evaluate_random_policy(
    env_cfg: EnvConfig,
    n_episodes: int = 8,
    seed: int = 0,
    num_envs: int | None = None,
) -> EvalResult:
    """Uniform-random actions in ``[-1, 1]`` -- the reference point."""
    rng = make_rng(seed, "random_policy")

    def action_fn(obs: np.ndarray) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=(obs.shape[0], obs.shape[1], env_cfg.act_dim))

    return _run_episodes(env_cfg, n_episodes, seed, action_fn, num_envs=num_envs)


def evaluate_config(cfg: Config, checkpoint: str, n_episodes: int = 32) -> dict[str, Any]:
    """Load a checkpoint and evaluate it (used by ``run.py eval``)."""
    from .algorithms import build_algorithm

    algo = build_algorithm(cfg, seed=cfg.train.seed, device=cfg.train.device)
    state = torch.load(checkpoint, map_location=cfg.train.device, weights_only=False)
    algo.load_state_dict(state)
    learned = evaluate_policy(
        algo, cfg.env, n_episodes=n_episodes, seed=cfg.train.seed + 99991,
        deterministic=cfg.train.eval_deterministic,
    )
    random = evaluate_random_policy(cfg.env, n_episodes=n_episodes, seed=cfg.train.seed + 99991)
    return {
        "learned": learned.to_dict(),
        "random": random.to_dict(),
        "improvement": learned.return_mean - random.return_mean,
    }
