"""Algorithms: shared base class + IPPO (Phase 2).

D-TrOPD (Phase 3), MAPPO / HAPPO / NaiveDistill (Phase 4) subclass
:class:`BaseAlgorithm` and are registered in :data:`ALGORITHMS` as they land.
Asking for one before it exists raises a clear, phase-tagged error rather than
silently falling back to something else.

The rollout collection, GAE bootstrapping and logging live in the base class,
so every algorithm consumes byte-identical batches given the same seed.  That
is what makes the Phase 3 acceptance test ("with beta = 0, D-TrOPD reproduces
IPPO bitwise-identically") a meaningful check of the social path rather than a
check of the data pipeline.
"""

from __future__ import annotations

import abc
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .buffer import RolloutBuffer
from .config import Config
from .env import SyncVectorEnv, ZoneKind
from .kl import gaussian_kl, mean_gaussian_kl
from .networks import ActorCritic, parameter_bytes
from .utils import RunningMeanStd, derive_seed, explained_variance, get_logger

__all__ = ["BaseAlgorithm", "IPPO", "ALGORITHMS", "build_algorithm", "build_graph"]

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Communication graph
# --------------------------------------------------------------------------- #


def build_graph(topology: str, n_agents: int, degree: int, rng: np.random.Generator) -> list[list[int]]:
    """Undirected communication graph as adjacency lists (no self-loops)."""
    adj: list[set[int]] = [set() for _ in range(n_agents)]
    if topology == "complete":
        for i in range(n_agents):
            adj[i] = {j for j in range(n_agents) if j != i}
    elif topology == "ring":
        for i in range(n_agents):
            if n_agents > 1:
                adj[i].add((i + 1) % n_agents)
                adj[i].add((i - 1) % n_agents)
    elif topology == "fixed_degree":
        # Circulant graph: deterministic, connected, exactly `degree`-regular
        # for even degree (odd degree additionally uses the antipode).
        half = degree // 2
        for i in range(n_agents):
            for d in range(1, half + 1):
                adj[i].add((i + d) % n_agents)
                adj[i].add((i - d) % n_agents)
            if degree % 2 == 1:
                adj[i].add((i + n_agents // 2) % n_agents)
    elif topology == "erdos_renyi":
        p = degree / max(1, n_agents - 1)
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                if rng.random() < p:
                    adj[i].add(j)
                    adj[j].add(i)
    else:
        raise ValueError(f"unknown topology {topology!r}")
    for i in range(n_agents):
        adj[i].discard(i)
    return [sorted(a) for a in adj]


# --------------------------------------------------------------------------- #
# Base algorithm
# --------------------------------------------------------------------------- #


class BaseAlgorithm(abc.ABC):
    """Per-agent actor-critic bookkeeping, rollout collection and GAE."""

    name: str = "base"

    def __init__(self, cfg: Config, seed: int, device: str | torch.device = "cpu"):
        self.cfg = cfg
        self.seed = int(seed)
        self.device = torch.device(device)
        self.n_agents = cfg.env.n_agents
        self.obs_dim = cfg.env.obs_dim
        self.act_dim = cfg.env.act_dim
        self.epoch = 0

        # One independent actor-critic per agent (no parameter sharing: SPEC
        # Sec. 2 gives every agent its own theta_i and its own critic V_i).
        self.agents: list[ActorCritic] = []
        for i in range(self.n_agents):
            torch.manual_seed(derive_seed(self.seed, "agent_init", i))
            self.agents.append(
                ActorCritic(self.obs_dim, self.act_dim, cfg.network).to(self.device)
            )
        self.optimizers = [
            torch.optim.Adam(ac.parameters(), lr=cfg.ppo.lr, eps=1e-5) for ac in self.agents
        ]
        # Per-agent critic-target normalisers (see utils.RunningMeanStd).
        self.value_normalizers = [RunningMeanStd() for _ in range(self.n_agents)]

        graph_seed = cfg.graph.seed if cfg.graph.seed is not None else derive_seed(seed, "graph")
        self.neighbors = build_graph(
            cfg.graph.topology,
            self.n_agents,
            cfg.graph.degree,
            np.random.default_rng(graph_seed),
        )
        self.snapshot_bytes = parameter_bytes(self.agents[0].actor)

        self.buffer = RolloutBuffer(
            cfg.steps_per_env, cfg.train.num_envs, self.n_agents, self.obs_dim, self.act_dim
        )
        self._rollout_generator = torch.Generator(device="cpu")
        self._minibatch_generator = torch.Generator(device="cpu")

    # -- acting -------------------------------------------------------------- #

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """``obs`` is ``(num_envs, n_agents, obs_dim)``; returns actions in [-1, 1]."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions = np.zeros((obs_t.shape[0], self.n_agents, self.act_dim), dtype=np.float32)
        for i, ac in enumerate(self.agents):
            if deterministic:
                a = ac.actor.deterministic_action(obs_t[:, i])
            else:
                a = ac.actor.sample(obs_t[:, i], generator=self._rollout_generator)["action"]
            actions[:, i] = a.cpu().numpy()
        return actions

    # -- rollout -------------------------------------------------------------- #

    def collect_rollout(self, vec_env: SyncVectorEnv) -> dict[str, Any]:
        """Fill ``self.buffer`` with one epoch of on-policy data."""
        cfg = self.cfg
        buf = self.buffer
        buf.reset()
        self._rollout_generator.manual_seed(derive_seed(self.seed, "rollout", self.epoch))

        obs = vec_env.reset(seed=derive_seed(self.seed, "env_reset", self.epoch))
        episode_returns: list[float] = []
        episode_captures: list[float] = []
        zone_returns: list[np.ndarray] = []
        step_rewards: list[float] = []
        collisions: list[float] = []

        for _ in range(buf.steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            latent = np.zeros((vec_env.num_envs, self.n_agents, self.act_dim), dtype=np.float32)
            mu = np.zeros_like(latent)
            log_std = np.zeros_like(latent)
            log_prob = np.zeros((vec_env.num_envs, self.n_agents), dtype=np.float32)
            actions = np.zeros_like(latent)

            with torch.no_grad():
                for i, ac in enumerate(self.agents):
                    out = ac.actor.sample(obs_t[:, i], generator=self._rollout_generator)
                    latent[:, i] = out["latent"].cpu().numpy()
                    mu[:, i] = out["mu"].cpu().numpy()
                    log_std[:, i] = out["log_std"].cpu().numpy()
                    log_prob[:, i] = out["log_prob"].cpu().numpy()
                    actions[:, i] = out["action"].cpu().numpy()

            step = vec_env.step(actions)
            buf.add(
                obs=obs,
                latent=latent,
                mu=mu,
                log_std=log_std,
                log_prob=log_prob,
                value=np.zeros((vec_env.num_envs, self.n_agents), dtype=np.float32),
                reward=step.rewards,
                terminated=step.terminated,
                truncated=step.truncated,
                bootstrap_obs=step.final_obs,
            )
            obs = step.obs
            step_rewards.append(float(step.rewards.mean()))
            for info in step.infos:
                collisions.append(float(info["collisions"]))
                ep = info.get("episode")
                if ep is not None:
                    episode_returns.append(ep["return"])
                    episode_captures.append(ep["captures"])
                    zone_returns.append(ep["zone_returns"])

        self._fill_values()
        buf.compute_returns(cfg.ppo.gamma, cfg.ppo.gae_lambda)

        stats: dict[str, Any] = {
            "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else float("nan"),
            "episode_return_std": float(np.std(episode_returns)) if episode_returns else float("nan"),
            "episodes": len(episode_returns),
            "captures_mean": float(np.mean(episode_captures)) if episode_captures else float("nan"),
            "step_reward_mean": float(np.mean(step_rewards)),
            "collisions_mean": float(np.mean(collisions)),
        }
        if zone_returns:
            zr = np.stack(zone_returns).mean(axis=0)
            for z, zname in enumerate(ZoneKind.NAMES):
                stats[f"zone_return/{zname}"] = float(zr[z])
        return stats

    @torch.no_grad()
    def _fill_values(self) -> None:
        """Batched critic passes for ``V(s_t)`` and the bootstrap ``V(s_{t+1})``."""
        buf = self.buffer
        t, k = buf.steps, buf.num_envs
        for i, ac in enumerate(self.agents):
            obs = torch.as_tensor(
                buf.obs[:, :, i].reshape(t * k, self.obs_dim), device=self.device
            )
            nxt = torch.as_tensor(
                buf.bootstrap_obs[:, :, i].reshape(t * k, self.obs_dim), device=self.device
            )
            buf.value[:, :, i] = self._denormalize_value(
                i, ac.critic(obs).cpu().numpy()
            ).reshape(t, k)
            buf.next_value[:, :, i] = self._denormalize_value(
                i, ac.critic(nxt).cpu().numpy()
            ).reshape(t, k)

    def _denormalize_value(self, agent: int, value: np.ndarray) -> np.ndarray:
        """Map a critic output back into reward units."""
        if not self.cfg.ppo.normalize_returns:
            return value
        return self.value_normalizers[agent].denormalize(value)

    # -- update ---------------------------------------------------------------- #

    @abc.abstractmethod
    def update(self) -> dict[str, Any]:
        """Consume ``self.buffer`` and return a metrics dict."""

    # -- checkpointing --------------------------------------------------------- #

    def state_dict(self) -> dict[str, Any]:
        return {
            "algo": self.name,
            "epoch": self.epoch,
            "agents": [ac.state_dict() for ac in self.agents],
            "optimizers": [o.state_dict() for o in self.optimizers],
            "value_normalizers": [n.state_dict() for n in self.value_normalizers],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state.get("epoch", 0))
        for ac, s in zip(self.agents, state["agents"]):
            ac.load_state_dict(s)
        for opt, s in zip(self.optimizers, state["optimizers"]):
            opt.load_state_dict(s)
        for norm, s in zip(self.value_normalizers, state.get("value_normalizers", [])):
            norm.load_state_dict(s)

    # -- helpers ---------------------------------------------------------------- #

    def train_mode(self) -> None:
        for ac in self.agents:
            ac.train()

    def eval_mode(self) -> None:
        for ac in self.agents:
            ac.eval()


# --------------------------------------------------------------------------- #
# IPPO
# --------------------------------------------------------------------------- #


class IPPO(BaseAlgorithm):
    """Independent PPO: each agent runs single-agent PPO on its own rollout.

    This is the ``beta = 0`` skeleton of SPEC Sec. 2.4:
    ``L_i = L_CLIP_i - c_V * L_V_i`` (minus an optional entropy bonus), with the
    *improvement* trust region enforced by clipping plus KL early stopping.
    """

    name = "ippo"

    def update(self) -> dict[str, Any]:
        cfg = self.cfg.ppo
        buf = self.buffer
        per_agent: list[dict[str, float]] = []

        for i, (ac, opt) in enumerate(zip(self.agents, self.optimizers)):
            per_agent.append(self._update_agent(i, ac, opt, cfg))

        metrics: dict[str, Any] = {"per_agent": per_agent}
        keys = per_agent[0].keys()
        for key in keys:
            values = [m[key] for m in per_agent]
            metrics[key] = float(np.nanmean(values))
            if key in ("update_kl", "explained_variance"):
                metrics[f"{key}_min"] = float(np.nanmin(values))
                metrics[f"{key}_max"] = float(np.nanmax(values))
        metrics["snapshot_bytes"] = 0  # IPPO exchanges nothing
        return metrics

    # -- one agent's PPO update ------------------------------------------------- #

    def _update_agent(
        self,
        agent_idx: int,
        ac: ActorCritic,
        opt: torch.optim.Optimizer,
        cfg,
    ) -> dict[str, float]:
        data = self.buffer.agent_tensors(agent_idx, device=self.device)
        obs = data["obs"]
        latent = data["latent"]
        old_log_prob = data["log_prob"]
        old_mu = data["mu"]
        old_log_std = data["log_std"]
        returns = data["returns"]
        advantages = data["advantage"]

        # Explained variance is always measured in reward units, on the
        # *pre-update* critic (out-of-sample w.r.t. this batch) -- invariant I4.
        ev = explained_variance(data["value"].cpu().numpy(), returns.cpu().numpy())

        if cfg.normalize_returns:
            normalizer = self.value_normalizers[agent_idx]
            normalizer.update(returns.cpu().numpy())
            value_targets = torch.as_tensor(
                normalizer.normalize(returns.cpu().numpy()), device=returns.device
            )
            old_values_for_clip = torch.as_tensor(
                normalizer.normalize(data["value"].cpu().numpy()), device=returns.device
            )
        else:
            value_targets = returns
            old_values_for_clip = data["value"]

        if cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self._minibatch_generator.manual_seed(
            derive_seed(self.seed, "minibatch", agent_idx, self.epoch)
        )

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        clip_fracs: list[float] = []
        grad_norms: list[float] = []
        inner_epochs_used = 0
        update_kl = 0.0
        early_stopped = False

        for _ in range(cfg.inner_epochs):
            for idx in self.buffer.minibatch_indices(
                cfg.num_minibatches, generator=self._minibatch_generator
            ):
                mb_obs = obs[idx]
                out = ac.actor.evaluate_latent(mb_obs, latent[idx])
                ratio = torch.exp(out["log_prob"] - old_log_prob[idx])
                mb_adv = advantages[idx]

                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                values = ac.critic(mb_obs)
                mb_targets = value_targets[idx]
                if cfg.clip_value_loss:
                    old_values = old_values_for_clip[idx]
                    v_clipped = old_values + torch.clamp(
                        values - old_values, -cfg.value_clip_ratio, cfg.value_clip_ratio
                    )
                    value_loss = 0.5 * torch.max(
                        (values - mb_targets) ** 2, (v_clipped - mb_targets) ** 2
                    ).mean()
                else:
                    value_loss = 0.5 * ((values - mb_targets) ** 2).mean()

                entropy = out["entropy"].mean()

                # SPEC Sec. 2.4: L = L_CLIP - c_V L_V (maximised) -> minimise the
                # negation. Phase 3 subtracts beta(t) * L_soc here.
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy
                    + self._extra_loss(agent_idx, ac, mb_obs, idx)
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)

                opt.step()

                with torch.no_grad():
                    policy_losses.append(float(policy_loss))
                    value_losses.append(float(value_loss))
                    entropies.append(float(entropy))
                    clip_fracs.append(
                        float(((ratio - 1.0).abs() > cfg.clip_ratio).float().mean())
                    )
                    grad_norms.append(float(grad_norm))

            inner_epochs_used += 1

            # --- KL early stopping (SPEC Sec. 2.4, invariant I5) --------------
            # E_s[ KL( pi^(t)(.|s) || pi_theta(.|s) ) ] over the agent's OWN
            # rollout states, in closed form on the pre-squash latents.
            with torch.no_grad():
                mu_new, log_std_new = ac.actor(obs)
                update_kl = float(
                    mean_gaussian_kl(old_mu, old_log_std, mu_new, log_std_new)
                )
            if update_kl >= self.cfg.ppo.kl_early_stop:
                early_stopped = True
                break

        with torch.no_grad():
            _, log_std_final = ac.actor(obs[:1])
            mean_sigma = float(torch.exp(log_std_final).mean())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "clip_fraction": float(np.mean(clip_fracs)),
            "grad_norm": float(np.mean(grad_norms)),
            "explained_variance": float(ev),
            "update_kl": float(update_kl),
            "kl_early_stop_eta": float(self.cfg.ppo.kl_early_stop),
            "inner_epochs_used": float(inner_epochs_used),
            "early_stopped": float(early_stopped),
            "mean_sigma": mean_sigma,
        }

    def _extra_loss(
        self, agent_idx: int, ac: ActorCritic, mb_obs: Tensor, idx: Tensor
    ) -> Tensor:
        """Hook for the Phase 3 social loss.  IPPO adds nothing."""
        return torch.zeros((), device=mb_obs.device, dtype=mb_obs.dtype)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

ALGORITHMS: dict[str, type[BaseAlgorithm]] = {
    "ippo": IPPO,
}

_PLANNED: dict[str, str] = {
    "dtropd": "Phase 3 (D-TrOPD core)",
    "naive_distill": "Phase 4 (baselines)",
    "mappo": "Phase 4 (baselines)",
    "happo": "Phase 4 (baselines)",
}


def build_algorithm(
    cfg: Config, seed: int, device: str | torch.device = "cpu"
) -> BaseAlgorithm:
    name = cfg.train.algo
    if name in ALGORITHMS:
        return ALGORITHMS[name](cfg, seed=seed, device=device)
    if name in _PLANNED:
        raise NotImplementedError(
            f"algorithm {name!r} is scheduled for {_PLANNED[name]} and is not "
            f"implemented yet. Implemented: {sorted(ALGORITHMS)}"
        )
    raise ValueError(f"unknown algorithm {name!r}; known: {sorted(ALGORITHMS) + sorted(_PLANNED)}")
