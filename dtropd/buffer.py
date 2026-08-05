"""Rollout storage and GAE.

Layout is ``(steps, num_envs, n_agents, ...)``.  Each agent's slice is a fully
independent on-policy batch: nothing here ever pools states across agents,
which is what keeps invariant I3 (own-rollout evaluation) structurally
enforceable rather than merely intended.

The rollout also stores each agent's ``(mu, log_std)`` at collection time.
Those *are* ``pi_i^(t)(.|s)`` on the agent's own states, so the KL early stop
of SPEC Sec. 2.4 and the nonstationarity radius of Sec. 2.2 can both be
evaluated in closed form without re-running a frozen snapshot.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch import Tensor

__all__ = ["RolloutBuffer", "compute_gae"]


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation with correct time-limit bootstrapping.

    ``rewards``/``values``/``next_values`` have shape ``(T, ...)``;
    ``terminated``/``truncated`` broadcast against them.

    * ``terminated`` -> the value of the successor state is 0 and the GAE
      recursion is cut.
    * ``truncated``  -> the successor value is bootstrapped from
      ``next_values`` (the true post-step observation) and the recursion is
      still cut, because the next stored transition belongs to a new episode.
    """
    terminated = np.broadcast_to(terminated, rewards.shape).astype(np.float64)
    truncated = np.broadcast_to(truncated, rewards.shape).astype(np.float64)
    done = np.clip(terminated + truncated, 0.0, 1.0)

    advantages = np.zeros_like(rewards, dtype=np.float64)
    last_gae = np.zeros_like(rewards[0], dtype=np.float64)
    for t in reversed(range(rewards.shape[0])):
        delta = (
            rewards[t]
            + gamma * next_values[t] * (1.0 - terminated[t])
            - values[t]
        )
        last_gae = delta + gamma * gae_lambda * (1.0 - done[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


class RolloutBuffer:
    """Fixed-size on-policy storage for ``n_agents`` agents."""

    def __init__(
        self,
        steps: int,
        num_envs: int,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        critic_obs_dim: int | None = None,
    ):
        self.steps = int(steps)
        self.num_envs = int(num_envs)
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        # The critic's input may be wider than the actor's: MAPPO's centralised
        # critic sees the concatenated observations of every agent. It is a
        # separate array rather than a reshape so that the actor's own-rollout
        # slice stays unambiguous (invariant I3 is easier to keep than to prove).
        self.critic_obs_dim = int(critic_obs_dim if critic_obs_dim is not None else obs_dim)

        shape = (self.steps, self.num_envs, self.n_agents)
        self.obs = np.zeros((*shape, obs_dim), dtype=np.float32)
        self.bootstrap_obs = np.zeros((*shape, obs_dim), dtype=np.float32)
        self.critic_obs = np.zeros((*shape, self.critic_obs_dim), dtype=np.float32)
        self.bootstrap_critic_obs = np.zeros((*shape, self.critic_obs_dim), dtype=np.float32)
        self.latent = np.zeros((*shape, act_dim), dtype=np.float32)
        self.mu = np.zeros((*shape, act_dim), dtype=np.float32)
        self.log_std = np.zeros((*shape, act_dim), dtype=np.float32)
        self.log_prob = np.zeros(shape, dtype=np.float32)
        self.value = np.zeros(shape, dtype=np.float32)
        self.next_value = np.zeros(shape, dtype=np.float32)
        self.reward = np.zeros(shape, dtype=np.float32)
        self.terminated = np.zeros((self.steps, self.num_envs), dtype=np.float32)
        self.truncated = np.zeros((self.steps, self.num_envs), dtype=np.float32)
        self.advantage = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self._ptr = 0

    # -- filling ------------------------------------------------------------ #

    def reset(self) -> None:
        self._ptr = 0

    @property
    def full(self) -> bool:
        return self._ptr >= self.steps

    def add(
        self,
        *,
        obs: np.ndarray,
        latent: np.ndarray,
        mu: np.ndarray,
        log_std: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        bootstrap_obs: np.ndarray,
        critic_obs: np.ndarray | None = None,
        bootstrap_critic_obs: np.ndarray | None = None,
    ) -> None:
        if self.full:
            raise RuntimeError("RolloutBuffer is full")
        t = self._ptr
        self.critic_obs[t] = obs if critic_obs is None else critic_obs
        self.bootstrap_critic_obs[t] = (
            bootstrap_obs if bootstrap_critic_obs is None else bootstrap_critic_obs
        )
        self.obs[t] = obs
        self.latent[t] = latent
        self.mu[t] = mu
        self.log_std[t] = log_std
        self.log_prob[t] = log_prob
        self.value[t] = value
        self.reward[t] = reward
        self.terminated[t] = terminated
        self.truncated[t] = truncated
        self.bootstrap_obs[t] = bootstrap_obs
        self._ptr += 1

    # -- GAE ----------------------------------------------------------------- #

    def compute_returns(self, gamma: float, gae_lambda: float) -> None:
        """Fill ``advantage`` and ``returns``; ``next_value`` must be set first."""
        adv, ret = compute_gae(
            self.reward.astype(np.float64),
            self.value.astype(np.float64),
            self.next_value.astype(np.float64),
            self.terminated[..., None],
            self.truncated[..., None],
            gamma,
            gae_lambda,
        )
        self.advantage[:] = adv.astype(np.float32)
        self.returns[:] = ret.astype(np.float32)

    # -- consumption --------------------------------------------------------- #

    @property
    def batch_size(self) -> int:
        return self.steps * self.num_envs

    def agent_tensors(self, agent: int, device: torch.device | str = "cpu") -> dict[str, Tensor]:
        """Flattened ``(steps * num_envs, ...)`` tensors for a single agent."""

        def flat(a: np.ndarray) -> Tensor:
            arr = a[:, :, agent]
            arr = arr.reshape(self.batch_size, *arr.shape[2:])
            return torch.as_tensor(arr, device=device)

        return {
            "obs": flat(self.obs),
            "critic_obs": flat(self.critic_obs),
            "latent": flat(self.latent),
            "mu": flat(self.mu),
            "log_std": flat(self.log_std),
            "log_prob": flat(self.log_prob),
            "value": flat(self.value),
            "advantage": flat(self.advantage),
            "returns": flat(self.returns),
        }

    def minibatch_indices(
        self, num_minibatches: int, generator: torch.Generator | None = None
    ) -> Iterator[Tensor]:
        """Shuffled minibatch index tensors covering the whole batch once."""
        perm = torch.randperm(self.batch_size, generator=generator)
        size = max(1, self.batch_size // num_minibatches)
        for start in range(0, self.batch_size, size):
            yield perm[start : start + size]

    # -- diagnostics ---------------------------------------------------------- #

    def agent_returns_and_values(self, agent: int) -> tuple[np.ndarray, np.ndarray]:
        r = self.returns[:, :, agent].ravel()
        v = self.value[:, :, agent].ravel()
        return r, v
