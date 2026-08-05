"""Tanh-squashed diagonal-Gaussian actor, local critic, and the
``policy_parameters()`` helper demanded by invariant I9.

The actor and the critic are *separate* modules with no shared trunk.  That is
deliberate: invariant I9 requires policy-gradient calls to see a parameter list
that excludes the value head, and a shared trunk would make
``torch.autograd.grad`` over ``policy_parameters()`` either error out or
silently produce partial gradients for the shared layers.
"""

from __future__ import annotations

import math
from typing import Iterator, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .config import NetworkConfig
from .kl import gaussian_entropy

__all__ = [
    "TanhGaussianActor",
    "Critic",
    "ActorCritic",
    "policy_parameters",
    "value_parameters",
]

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
}

# Numerical guard for the tanh log-det-Jacobian term.
_EPS = 1e-6


def _mlp(
    in_dim: int,
    hidden_sizes: Sequence[int],
    out_dim: int,
    activation: str,
    *,
    orthogonal_init: bool,
    output_gain: float,
) -> nn.Sequential:
    act_cls = _ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(act_cls())
        last = h
    layers.append(nn.Linear(last, out_dim))
    module = nn.Sequential(*layers)
    if orthogonal_init:
        linears = [m for m in module if isinstance(m, nn.Linear)]
        for layer in linears[:-1]:
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(linears[-1].weight, gain=output_gain)
        nn.init.zeros_(linears[-1].bias)
    return module


class TanhGaussianActor(nn.Module):
    """``a = tanh(z)``, ``z ~ N(mu(o), sigma(o)^2)`` with diagonal covariance.

    The latent parameters ``(mu, log_std)`` are the only thing the KL machinery
    ever sees (SPEC Sec. 2.5).
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: NetworkConfig):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cfg = cfg
        self.log_std_min = float(cfg.log_std_min)
        self.log_std_max = float(cfg.log_std_max)
        self.state_dependent_std = bool(cfg.state_dependent_std)

        out_dim = act_dim * (2 if self.state_dependent_std else 1)
        self.net = _mlp(
            obs_dim,
            cfg.hidden_sizes,
            out_dim,
            cfg.activation,
            orthogonal_init=cfg.orthogonal_init,
            output_gain=0.01,
        )
        if not self.state_dependent_std:
            self.log_std = nn.Parameter(torch.full((act_dim,), float(cfg.log_std_init)))

    # -- distribution parameters ------------------------------------------- #

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(mu, log_std)``, each of shape ``(..., act_dim)``."""
        out = self.net(obs)
        if self.state_dependent_std:
            mu, log_std = torch.chunk(out, 2, dim=-1)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        else:
            mu = out
            log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
            log_std = log_std.expand_as(mu)
        return mu, log_std

    # -- sampling / evaluation --------------------------------------------- #

    @staticmethod
    def _latent_log_prob(latent: Tensor, mu: Tensor, log_std: Tensor) -> Tensor:
        var = torch.exp(2.0 * log_std)
        lp = -0.5 * ((latent - mu) ** 2 / var) - log_std - 0.5 * math.log(2.0 * math.pi)
        return lp.sum(dim=-1)

    @staticmethod
    def _tanh_log_det(latent: Tensor) -> Tensor:
        """``sum_d log(1 - tanh(z_d)^2)``, computed stably."""
        # log(1 - tanh(z)^2) = 2 * (log(2) - z - softplus(-2z))
        return (
            2.0 * (math.log(2.0) - latent - torch.nn.functional.softplus(-2.0 * latent))
        ).sum(dim=-1)

    def log_prob_from_latent(self, latent: Tensor, mu: Tensor, log_std: Tensor) -> Tensor:
        """Log-density of ``a = tanh(latent)`` under the squashed policy."""
        return self._latent_log_prob(latent, mu, log_std) - self._tanh_log_det(latent)

    def sample(self, obs: Tensor, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        """Sample an action.  Returns latent, action, log-prob and parameters.

        The *latent* is what gets stored in the rollout buffer: re-deriving it
        with ``atanh`` is numerically unstable near the saturation boundary.
        """
        mu, log_std = self(obs)
        noise = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
        latent = mu + noise * torch.exp(log_std)
        action = torch.tanh(latent)
        log_prob = self.log_prob_from_latent(latent, mu, log_std)
        return {
            "latent": latent,
            "action": action,
            "log_prob": log_prob,
            "mu": mu,
            "log_std": log_std,
        }

    def evaluate_latent(self, obs: Tensor, latent: Tensor) -> dict[str, Tensor]:
        """Re-evaluate stored latents under the *current* parameters."""
        mu, log_std = self(obs)
        return {
            "log_prob": self.log_prob_from_latent(latent, mu, log_std),
            "entropy": gaussian_entropy(log_std),
            "mu": mu,
            "log_std": log_std,
        }

    def deterministic_action(self, obs: Tensor) -> Tensor:
        mu, _ = self(obs)
        return torch.tanh(mu)


class Critic(nn.Module):
    """Local state-value function ``V_i(o_i)`` (SPEC Sec. 2: no central critic)."""

    def __init__(self, obs_dim: int, cfg: NetworkConfig):
        super().__init__()
        self.net = _mlp(
            obs_dim,
            cfg.hidden_sizes,
            1,
            cfg.activation,
            orthogonal_init=cfg.orthogonal_init,
            output_gain=1.0,
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """One agent's policy + critic."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: NetworkConfig, critic_obs_dim: int | None = None):
        super().__init__()
        self.actor = TanhGaussianActor(obs_dim, act_dim, cfg)
        self.critic = Critic(critic_obs_dim if critic_obs_dim is not None else obs_dim, cfg)

    # -- invariant I9 ------------------------------------------------------- #

    def policy_parameters(self) -> list[nn.Parameter]:
        """Parameters the policy loss can differentiate w.r.t.

        **Excludes the value head.**  ``torch.autograd.grad`` over the full
        parameter list errors (or, with ``allow_unused=True``, silently yields
        ``None``) for the critic parameters that the policy loss never touches;
        the per-neighbour distillation gradients of SPEC Sec. 2.6 depend on this
        list being exact.
        """
        return list(self.actor.parameters())

    def value_parameters(self) -> list[nn.Parameter]:
        return list(self.critic.parameters())

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, log_std = self.actor(obs)
        value = self.critic(obs)
        return mu, log_std, value


def policy_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Free-function form of :meth:`ActorCritic.policy_parameters`."""
    if isinstance(module, ActorCritic):
        return module.policy_parameters()
    if isinstance(module, TanhGaussianActor):
        return list(module.parameters())
    raise TypeError(f"policy_parameters() does not know how to handle {type(module)}")


def value_parameters(module: nn.Module) -> list[nn.Parameter]:
    if isinstance(module, ActorCritic):
        return module.value_parameters()
    if isinstance(module, Critic):
        return list(module.parameters())
    raise TypeError(f"value_parameters() does not know how to handle {type(module)}")


def parameter_bytes(module: nn.Module) -> int:
    """Total bytes of a parameter snapshot (communication-volume accounting)."""
    return sum(p.numel() * p.element_size() for p in module.parameters())
