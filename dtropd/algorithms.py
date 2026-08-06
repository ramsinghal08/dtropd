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
import copy
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .buffer import RolloutBuffer
from .byzantine import apply_attack, byzantine_agents
from .calibration import CalibrationError, CalibrationRecord, env_fingerprint
from .config import Config, ConfigError, DistillConfig
from .env import SyncVectorEnv, ZoneKind
from .kl import gaussian_kl, mean_gaussian_kl
from .networks import ActorCritic, TanhGaussianActor, parameter_bytes
from .instrumentation import (
    SignFlipTracker,
    pairwise_policy_dispersion,
    quantiles,
)
from .robust import aggregate
from .utils import RunningMeanStd, derive_seed, explained_variance, get_logger, git_commit

__all__ = [
    "BaseAlgorithm",
    "IPPO",
    "MAPPO",
    "HAPPO",
    "SnapshotDistillBase",
    "NaiveDistill",
    "DTrOPD",
    "SocialState",
    "TeacherBundle",
    "trust_radius",
    "trust_weight",
    "calibrate_attenuation",
    "ALGORITHMS",
    "build_algorithm",
    "build_graph",
]

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

        # Width of the critic's input. Equal to obs_dim for every decentralised
        # algorithm; MAPPO widens it to the concatenated observation.
        self.critic_obs_dim = self._critic_obs_dim(cfg)

        # One independent actor-critic per agent (no parameter sharing: SPEC
        # Sec. 2 gives every agent its own theta_i and its own critic V_i).
        self.agents: list[ActorCritic] = []
        for i in range(self.n_agents):
            torch.manual_seed(derive_seed(self.seed, "agent_init", i))
            self.agents.append(
                ActorCritic(
                    self.obs_dim,
                    self.act_dim,
                    cfg.network,
                    critic_obs_dim=self.critic_obs_dim,
                ).to(self.device)
            )
        if cfg.ppo.degenerate_critic:
            # H3 ABLATION ONLY (SPEC Sec. 6): a critic carrying no information.
            # Frozen at initialisation rather than merely badly trained, because
            # H3's prediction is about what happens when the value signal is
            # *meaningless*, not when it is noisy. Contradicts I4 by design and
            # says so loudly at config validation.
            for ac in self.agents:
                for p in ac.value_parameters():
                    p.requires_grad_(False)
            logger.warning(
                "ppo.degenerate_critic: critics frozen at initialisation. This "
                "contradicts invariant I4 and is valid ONLY as the H3 ablation; "
                "nothing from this run may be reported as D-TrOPD."
            )
        self.optimizers = [
            torch.optim.Adam(
                [p for p in ac.parameters() if p.requires_grad], lr=cfg.ppo.lr, eps=1e-5
            )
            for ac in self.agents
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

        # SPEC Sec. 11 / D6: every arm in a comparison declares the calibration
        # artifact that comparison used -- including arms that do not consume a
        # delta_base (NaiveDistill has no radius, per D4; IPPO has no social
        # path at all). Declaring it is what makes assert_shared_calibration()
        # able to refuse a mixed sweep.
        self.calibration_record: CalibrationRecord | None = None
        if cfg.distill.calibration_file:
            self.calibration_record = CalibrationRecord.load(cfg.distill.calibration_file)
            expected = env_fingerprint(cfg.env)
            if self.calibration_record.env_fingerprint != expected:
                raise CalibrationError(
                    f"calibration artifact {cfg.distill.calibration_file} was measured "
                    "on a different environment configuration "
                    f"({self.calibration_record.env_fingerprint} != {expected}). "
                    "Calibration is per environment configuration (SPEC Sec. 11 / D6)."
                )
            if self.calibration_record.reward_spec_version != cfg.env.reward_spec_version:
                raise CalibrationError(
                    "calibration artifact was measured under reward_spec_version "
                    f"{self.calibration_record.reward_spec_version!r}, this run uses "
                    f"{cfg.env.reward_spec_version!r}."
                )

        self.buffer = RolloutBuffer(
            cfg.steps_per_env,
            cfg.train.num_envs,
            self.n_agents,
            self.obs_dim,
            self.act_dim,
            critic_obs_dim=self.critic_obs_dim,
        )
        self._rollout_generator = torch.Generator(device="cpu")
        self._minibatch_generator = torch.Generator(device="cpu")

    # -- critic input --------------------------------------------------------- #

    @staticmethod
    def _critic_obs_dim(cfg: Config) -> int:
        """Width of the critic's input. Decentralised critics see own obs only."""
        return cfg.env.obs_dim

    def _critic_obs(self, obs: np.ndarray) -> np.ndarray:
        """Map ``(num_envs, n_agents, obs_dim)`` to the critic's input.

        The identity for every algorithm with a local critic -- which is all of
        them except MAPPO. Kept as a hook rather than a branch so that SPEC
        Sec. 2's "no central critic" holds structurally for D-TrOPD.
        """
        return obs

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
        episode_agent_returns: list[np.ndarray] = []
        episode_captures: list[float] = []
        zone_returns: list[np.ndarray] = []
        zone_collisions: list[np.ndarray] = []
        zone_agent_steps: list[np.ndarray] = []
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
                critic_obs=self._critic_obs(obs),
                bootstrap_critic_obs=self._critic_obs(step.final_obs),
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
                    if "agent_returns" in ep:
                        episode_agent_returns.append(ep["agent_returns"])
                    episode_captures.append(ep["captures"])
                    zone_returns.append(ep["zone_returns"])
                    zone_collisions.append(ep["zone_collisions"])
                    zone_agent_steps.append(ep["zone_agent_steps"])

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
        # H6 reads the honest agents' return. Emitted whenever the per-agent
        # breakdown exists, not only when an attack is configured, so the f=0
        # baseline cell carries the same field as every other cell and the
        # sweep never compares an honest-only number against an all-agent one.
        if episode_agent_returns:
            per_agent = np.stack(episode_agent_returns)
            honest = [i for i in range(self.n_agents) if i not in getattr(self, "_byzantine_set", ())]
            stats["episode_return_mean_honest"] = float(per_agent[:, honest].mean())
            if honest and len(honest) < self.n_agents:
                byz = [i for i in range(self.n_agents) if i not in honest]
                stats["episode_return_mean_byzantine"] = float(per_agent[:, byz].mean())
        if zone_returns:
            zr = np.stack(zone_returns).mean(axis=0)
            for z, zname in enumerate(ZoneKind.NAMES):
                stats[f"zone_return/{zname}"] = float(zr[z])
        if zone_collisions:
            zc = np.stack(zone_collisions).mean(axis=0)
            za = np.stack(zone_agent_steps).mean(axis=0)
            for z, zname in enumerate(ZoneKind.NAMES):
                stats[f"zone_collisions/{zname}"] = float(zc[z])
                stats[f"zone_agent_steps/{zname}"] = float(za[z])
        return stats

    @torch.no_grad()
    def _fill_values(self) -> None:
        """Batched critic passes for ``V(s_t)`` and the bootstrap ``V(s_{t+1})``."""
        buf = self.buffer
        t, k = buf.steps, buf.num_envs
        for i, ac in enumerate(self.agents):
            obs = torch.as_tensor(
                buf.critic_obs[:, :, i].reshape(t * k, self.critic_obs_dim),
                device=self.device,
            )
            nxt = torch.as_tensor(
                buf.bootstrap_critic_obs[:, :, i].reshape(t * k, self.critic_obs_dim),
                device=self.device,
            )
            buf.value[:, :, i] = self._denormalize_value(
                i, ac.critic(obs).cpu().numpy()
            ).reshape(t, k)
            buf.next_value[:, :, i] = self._denormalize_value(
                i, ac.critic(nxt).cpu().numpy()
            ).reshape(t, k)

    # -- Phase 5 instrumentation shared by every algorithm --------------------- #

    def policy_dispersion(self, probe_size: int = 256) -> np.ndarray:
        """Pairwise policy divergence across the swarm on a shared probe batch.

        D9's consensus measure. Computed identically for every algorithm so the
        distillation arms have an uncontaminated IPPO control. This is a
        diagnostic on a *shared* batch and is never used as a trust signal, so
        it does not touch invariant I3.
        """
        obs = self.buffer.obs[:, :, 0].reshape(-1, self.obs_dim)
        take = min(probe_size, obs.shape[0])
        probe = torch.as_tensor(obs[:take], device=self.device)
        return pairwise_policy_dispersion([ac.actor for ac in self.agents], probe)

    def compute_accounting(self) -> dict[str, Any]:
        """Compute cost of one epoch (SPEC Sec. 7: report compute, not just wall-clock)."""
        cfg = self.cfg.ppo
        params = sum(p.numel() for ac in self.agents for p in ac.parameters())
        return {
            "compute/env_steps": self.buffer.batch_size,
            "compute/agent_steps": self.buffer.batch_size * self.n_agents,
            "compute/gradient_steps": self.n_agents * cfg.inner_epochs * cfg.num_minibatches,
            "compute/parameters": params,
            "compute/actor_parameters": sum(
                p.numel() for ac in self.agents for p in ac.policy_parameters()
            ),
        }

    def _denormalize_value(self, agent: int, value: np.ndarray) -> np.ndarray:
        """Map a critic output back into reward units."""
        if not self.cfg.ppo.normalize_returns:
            return value
        return self.value_normalizers[agent].denormalize(value)

    # -- per-epoch hooks --------------------------------------------------------- #

    def on_epoch_start(self) -> None:
        """Called once per epoch *before* the rollout.

        SPEC Sec. 2.1 step 1 ("broadcast a frozen, detached copy of theta_i^(t)")
        happens here, so the snapshot is provably taken before any data is
        collected or any parameter moves.  IPPO has nothing to broadcast.
        """
        return None

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

        results: dict[int, dict[str, float]] = {}
        for i in self._agent_order():
            results[i] = self._update_agent(i, self.agents[i], self.optimizers[i], cfg)
            self._after_agent_update(i, self.agents[i])
        # Reported in agent-index order regardless of the order they were updated.
        per_agent = [results[i] for i in range(self.n_agents)]

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
        critic_obs = data["critic_obs"]  # == obs unless the critic is centralised
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

        # HAPPO's sequential correction factor multiplies the advantage before
        # normalisation; every other algorithm returns None and is unaffected.
        scale = self._advantage_scale(agent_idx)
        if scale is not None:
            advantages = advantages * scale

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

                values = ac.critic(critic_obs[idx])
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
                    + self._extra_loss(agent_idx, ac, mb_obs, idx, out)
                )

                opt.zero_grad(set_to_none=True)
                loss.backward(retain_graph=self._retain_graph)
                self._apply_extra_gradients(agent_idx, ac, out, idx)
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

        # H4/H5: the surrogate's predicted improvement for this agent, on its own
        # rollout, in reward units (raw advantages, not the normalised ones the
        # update used). The analysis pairs it with the realised change in return
        # to get the simultaneity penalty.
        surrogate_gain = float("nan")
        if self.cfg.instrument.surrogate_gain:
            with torch.no_grad():
                raw_adv = data["advantage"]
                out_final = ac.actor.evaluate_latent(obs, latent)
                ratio_final = torch.exp(out_final["log_prob"] - old_log_prob)
                surrogate_gain = float(
                    (ratio_final * raw_adv).mean() - raw_adv.mean()
                )

        return {
            "surrogate_gain": surrogate_gain,
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

    # -- hooks the D-TrOPD social path plugs into ------------------------------- #

    #: whether the inner-loop backward must keep the graph for a second pass
    #: (only the per-neighbour robust-aggregation path needs it)
    _retain_graph: bool = False

    def _extra_loss(
        self,
        agent_idx: int,
        ac: ActorCritic,
        mb_obs: Tensor,
        idx: Tensor,
        out: dict[str, Tensor],
    ) -> Tensor:
        """Additive loss term.  IPPO adds nothing; D-TrOPD adds ``beta * L_soc``."""
        return torch.zeros((), device=mb_obs.device, dtype=mb_obs.dtype)

    def _apply_extra_gradients(
        self, agent_idx: int, ac: ActorCritic, out: dict[str, Tensor], idx: Tensor
    ) -> None:
        """Inject gradients computed outside the main backward.  No-op for IPPO."""
        return None

    def _agent_order(self) -> list[int]:
        """Order in which agents are updated. Simultaneous (index order) here.

        SPEC Sec. 2 requires D-TrOPD to have *no sequential update order*; HAPPO
        overrides this precisely because a random sequential order is what it is.
        """
        return list(range(self.n_agents))

    def _advantage_scale(self, agent_idx: int) -> Tensor | None:
        """Optional per-sample multiplier on the advantage (HAPPO's factor)."""
        return None

    def _after_agent_update(self, agent_idx: int, ac: ActorCritic) -> None:
        """Hook run immediately after one agent's inner loop finishes."""
        return None


# --------------------------------------------------------------------------- #
# Phase 4 baselines: MAPPO and HAPPO
# --------------------------------------------------------------------------- #


class MAPPO(IPPO):
    """Multi-Agent PPO: centralised critic, decentralised actors.

    SPEC Sec. 5 Phase 4: "centralized critic over concatenated observations".
    Every agent keeps its own actor and its own critic *network*, but each
    critic is fed the concatenation of all agents' observations, so it can
    resolve the parts of the return that a local observation cannot.

    Concatenation is in fixed agent-index order, identical for every agent, so
    the critic input at a given transition is the same vector for all of them.
    That is deliberate: it makes MAPPO's advantage over IPPO purely a matter of
    critic information, not of asymmetric inputs.

    This is the baseline that *has* the thing D-TrOPD forgoes -- SPEC Sec. 2
    gives D-TrOPD no central critic -- so it bounds what centralisation buys.
    """

    name = "mappo"

    @staticmethod
    def _critic_obs_dim(cfg: Config) -> int:
        return cfg.env.obs_dim * cfg.env.n_agents

    def _critic_obs(self, obs: np.ndarray) -> np.ndarray:
        # (num_envs, n_agents, obs_dim) -> (num_envs, n_agents, n_agents * obs_dim)
        num_envs, n_agents, obs_dim = obs.shape
        joint = obs.reshape(num_envs, 1, n_agents * obs_dim)
        return np.broadcast_to(joint, (num_envs, n_agents, n_agents * obs_dim)).copy()


class HAPPO(IPPO):
    """Heterogeneous-Agent PPO: sequential updates in a random order.

    SPEC Sec. 5 Phase 4: "sequential updates, random order each epoch".

    Agents are updated one at a time in a permutation re-sampled every epoch.
    What makes the sequence sound rather than merely serial is the HAPPO
    correction factor: when agent ``i`` is updated, its advantage is multiplied
    by

        M_i(s, a) = prod_{j before i in this epoch's order}
                    pi_j^new(a_j | s) / pi_j^old(a_j | s)

    so agent ``i`` optimises against the policies its predecessors have
    *already moved to*, not the ones they started from. Dropping the factor
    would leave a sequential update with no monotonic-improvement argument at
    all, which is the only interesting thing about updating sequentially.

    Because the rollout buffer is indexed ``(step, env, agent)``, transition
    ``(t, k)`` is the same world transition for every agent, so the factor is a
    well-defined per-sample scalar shared across agents.

    Judgment call (see README): this HAPPO keeps IPPO's **local** critics rather
    than adopting a centralised one. SPEC describes HAPPO only as "sequential
    updates, random order", and keeping the critic local isolates the
    sequential-update variable -- MAPPO is the arm that varies critic
    centralisation.
    """

    name = "happo"

    def __init__(self, cfg: Config, seed: int, device: str | torch.device = "cpu"):
        super().__init__(cfg, seed, device)
        self._factor: Tensor | None = None
        self._order: list[int] = list(range(self.n_agents))

    def _agent_order(self) -> list[int]:
        rng = np.random.default_rng(derive_seed(self.seed, "happo_order", self.epoch))
        self._order = [int(i) for i in rng.permutation(self.n_agents)]
        # Reset the sequential correction at the start of every epoch.
        self._factor = torch.ones(self.buffer.batch_size, device=self.device)
        return self._order

    def _advantage_scale(self, agent_idx: int) -> Tensor | None:
        return self._factor

    @torch.no_grad()
    def _after_agent_update(self, agent_idx: int, ac: ActorCritic) -> None:
        """Fold this agent's realised policy change into the factor."""
        if self._factor is None:
            return None
        data = self.buffer.agent_tensors(agent_idx, device=self.device)
        out = ac.actor.evaluate_latent(data["obs"], data["latent"])
        ratio = torch.exp(out["log_prob"] - data["log_prob"])
        # Guard the product against numerical blow-up over many agents.
        self._factor = torch.clamp(self._factor * ratio, min=1e-4, max=1e4)
        return None

    def update(self) -> dict[str, Any]:
        metrics = super().update()
        metrics["happo_order"] = list(self._order)
        if self._factor is not None:
            metrics["happo_factor_mean"] = float(self._factor.mean())
            metrics["happo_factor_max"] = float(self._factor.max())
        return metrics


# --------------------------------------------------------------------------- #
# Trust radius and attenuation (SPEC Sec. 2.2 / 2.3)
# --------------------------------------------------------------------------- #


def trust_radius(delta_base: float, kappa: float, nonstationarity: float) -> float:
    """``delta_i^(t) = delta_base * exp(-kappa * Delta_i^(t))`` (SPEC Sec. 2.2).

    ``nonstationarity`` is ``Delta_i^(t)``, in nats, measured on the agent's own
    rollout states.  At ``t = 0`` it is 0 by definition, so ``delta = delta_base``.
    """
    return float(delta_base * math.exp(-kappa * nonstationarity))


def trust_weight(
    divergence: Tensor | float,
    delta: float,
    gamma: float,
    mode: str = "continuous",
) -> Tensor:
    """Continuous attenuation ``w_ij`` (SPEC Sec. 2.3) and its ablation arms.

    ``continuous`` (the D-TrOPD default)::

        w = exp( -gamma * [ softplus(D - delta) - softplus(-delta) ] )

    The ``- softplus(-delta)`` term is the normaliser that makes ``w(0) = 1``
    exactly (invariant I7).  It is not optional and there is no code path that
    omits it.

    The other two modes exist only as the H1 ablation arms and are named for
    exactly what is wrong with them:

    * ``hard_switch``  -- ``exp(-gamma * max(D - delta, 0))``: continuous, but
      only C^0; the derivative jumps at ``D = delta``.
    * ``hard_cutoff``  -- ``1{D <= delta}``: not even continuous.
    """
    d = divergence if isinstance(divergence, Tensor) else torch.tensor(float(divergence))
    if mode == "continuous":
        normaliser = torch.nn.functional.softplus(torch.tensor(-float(delta), dtype=d.dtype))
        return torch.exp(-gamma * (torch.nn.functional.softplus(d - delta) - normaliser))
    if mode == "hard_switch":
        return torch.exp(-gamma * torch.clamp(d - delta, min=0.0))
    if mode == "hard_cutoff":
        return (d <= delta).to(d.dtype)
    raise ValueError(f"unknown attenuation mode {mode!r}")


def calibrate_attenuation(
    divergences: Sequence[float],
    quantile: float = 0.5,
    tail_quantile: float = 0.95,
    target_w: float = 0.5,
) -> dict[str, float]:
    """The SPEC Sec. 10 / D1 calibration procedure.

    Given observed ``D_ij`` samples, return the ``delta_base`` and ``gamma``
    that put the trust gate inside the data:

    * ``delta_base`` = the ``quantile`` of observed ``D_ij``, so roughly that
      fraction of edges starts inside the radius;
    * ``gamma`` chosen so an edge at ``tail_quantile`` is attenuated to
      ``target_w``.

    One function, used by both the in-run (``distill.calibration: warmup``) and
    the offline (``run.py calibrate``) paths, so the two cannot drift apart.
    """
    d = np.asarray(list(divergences), dtype=np.float64)
    if d.size == 0:
        raise ValueError("cannot calibrate from an empty D_ij sample")
    delta_base = float(np.quantile(d, quantile))
    tail = float(np.quantile(d, tail_quantile))
    # w = exp(-gamma * [softplus(D - delta) - softplus(-delta)]) = target_w
    excess = math.log1p(math.exp(tail - delta_base)) - math.log1p(math.exp(-delta_base))
    if excess <= 0.0 or not math.isfinite(excess):
        raise ValueError(
            "degenerate D_ij distribution: the tail quantile does not exceed the "
            "median, so no gamma can separate them"
        )
    return {
        "delta_base": delta_base,
        "gamma_attenuation": float(-math.log(target_w) / excess),
        "d_ij_median": delta_base,
        "d_ij_tail": tail,
        "n_samples": float(d.size),
    }


# --------------------------------------------------------------------------- #
# Shared distillation machinery (frozen snapshots + social loss plumbing)
# --------------------------------------------------------------------------- #


@dataclass
class TeacherBundle:
    """The frozen neighbour policies agent ``i`` distils from this epoch.

    ``teacher_mu`` / ``teacher_log_std`` are theta_j^(t) evaluated on agent i's
    **own** rollout states (invariant I3), cached once per epoch because both
    the teachers and the states are fixed for its duration (invariant I6).
    """

    neighbors: list[int]
    teacher_mu: Tensor  # (J, B, act_dim)
    teacher_log_std: Tensor  # (J, B, act_dim)


@dataclass
class SocialState(TeacherBundle):
    """A :class:`TeacherBundle` plus D-TrOPD's trust gate.

    Only D-TrOPD has these fields. NaiveDistill deliberately does not (SPEC
    Sec. 10 / D4): if the baseline were reachable from the method by a config
    value, "NaiveDistill" and "miscalibrated D-TrOPD" would be the same run.
    """

    divergence: Tensor  # D_ij, detached, shape (J,)
    weight: Tensor  # w_ij, detached, shape (J,)
    nonstationarity: float  # Delta_i^(t)
    radius: float  # delta_i^(t)


class SnapshotDistillBase(IPPO):
    """Everything distillation needs *except* the weighting rule.

    Subclasses differ in exactly one method, :meth:`_social_terms`:

    * :class:`NaiveDistill` sums the per-neighbour KLs unweighted -- ``w_ij = 1``
      structurally, with no gate to mis-set;
    * :class:`DTrOPD` multiplies each by its trust weight.
    """

    #: subclasses that maintain a trust gate report per-edge diagnostics
    reports_trust_gate: bool = False

    def __init__(self, cfg: Config, seed: int, device: str | torch.device = "cpu"):
        super().__init__(cfg, seed, device)
        d = cfg.distill

        # Invariant I2 is enforced *here*, at the only place a direction could
        # be chosen, and not merely declared in the config dataclass.
        if d.kl_direction != "reverse":
            raise ConfigError(
                f"distill.kl_direction={d.kl_direction!r}: the distillation loss "
                "computes exactly one KL direction (reverse, student first). "
                "Invariant I2 forbids a direction switch inside the update; "
                "forward/switched arms must be separate, explicitly named "
                "ablation algorithms."
            )
        if not d.enabled:
            raise ConfigError(
                f"train.algo={self.name!r} requires distill.enabled=true; running a "
                "distillation update with distillation disabled would silently be IPPO."
            )

        self.snapshots: list[TanhGaussianActor] | None = None
        self._social: list[TeacherBundle | None] = [None] * self.n_agents
        self._beta: float = 0.0
        self.epoch_snapshot_bytes = 0
        self.total_snapshot_bytes = 0

        # Phase 7 / H6. A Byzantine neighbour corrupts the gradient it induces,
        # so the attack has to be injected where the per-edge gradients exist.
        # With plain-sum aggregation those gradients are never materialised --
        # the loss path handles the whole sum at once -- so an active adversary
        # forces the per-edge path on regardless of the aggregation operator.
        # Without that, "plain sum collapses early" would be untestable: the
        # sum arm would silently run unattacked and look robust.
        stress = cfg.stress
        self.byzantine: list[int] = byzantine_agents(
            self.n_agents,
            stress.n_byzantine(self.n_agents),
            stress.byzantine_seed if stress.byzantine_seed is not None else seed,
        )
        self._byzantine_set = set(self.byzantine)
        #: Agents every one of whose neighbours is Byzantine -- an unwinnable
        #: neighbourhood, reported rather than silently counted as a success.
        self._all_byzantine_neighbourhoods: set[int] = set()
        self._byzantine_gen = torch.Generator(device="cpu").manual_seed(
            derive_seed(seed, "byzantine_noise")
        )
        self._uses_loss_path = d.aggregation == "sum" and not self.byzantine
        self._retain_graph = not self._uses_loss_path

        # Communication dropout (Phase 7). Its own stream, so turning dropout on
        # does not shift the Byzantine draw or the environment seeding.
        self._dropout_rng = np.random.default_rng(derive_seed(seed, "comm_dropout"))
        self._epoch_neighbors: list[list[int]] = [list(n) for n in self.neighbors]

        # Per-edge instrumentation state (Phase 5 / H1).
        self._sign_flips = SignFlipTracker(cfg.instrument.sign_flip_buckets)
        self._edge_rows: list[dict[str, Any]] = []
        self._consensus: dict[str, Any] = {}

    # -- Phase 7 / H6: Byzantine neighbours ------------------------------------ #

    def _corrupt_byzantine_edges(self, agent_idx: int, stacked: Tensor) -> Tensor:
        """Replace rows contributed by Byzantine neighbours with their attack.

        Row ``k`` of ``stacked`` is the gradient induced by ``state.neighbors[k]``,
        so the Byzantine rows are found by membership rather than by position --
        neighbour lists differ per agent and are not sorted by agent index.
        """
        if not self.byzantine:
            return stacked
        state = self._social[agent_idx]
        if state is None or not state.neighbors:
            return stacked
        rows = [k for k, j in enumerate(state.neighbors) if j in self._byzantine_set]
        if not rows:
            return stacked
        if len(rows) == len(state.neighbors):
            # Every neighbour of this agent is Byzantine. The adaptive attack has
            # no honest gradient to hide behind, and no aggregation rule can
            # recover a signal that is not present. Leave the edges honest and
            # record it, rather than crashing or pretending a defence held.
            self._all_byzantine_neighbourhoods.add(agent_idx)
            return stacked
        return apply_attack(
            stacked,
            rows,
            self.cfg.stress.byzantine_mode,
            self.cfg.stress.byzantine_scale,
            generator=self._byzantine_gen,
        )

    # -- SPEC Sec. 2.1 step 1: broadcast frozen snapshots ----------------------- #

    @staticmethod
    def _freeze(actor: TanhGaussianActor) -> TanhGaussianActor:
        snapshot = copy.deepcopy(actor)
        snapshot.eval()
        for p in snapshot.parameters():
            p.requires_grad_(False)
            p.detach_()
        return snapshot

    def on_epoch_start(self) -> None:
        self.snapshots = [self._freeze(ac.actor) for ac in self.agents]
        self._epoch_neighbors = self._deliver_snapshots()
        # Communication volume: one actor snapshot per directed edge that
        # actually delivered. Counting the dropped ones would overstate the
        # bandwidth cost that SPEC Sec. 7 asks to be reported.
        self.epoch_snapshot_bytes = sum(
            len(self._epoch_neighbors[i]) * self.snapshot_bytes
            for i in range(self.n_agents)
        )
        self.total_snapshot_bytes += self.epoch_snapshot_bytes

    def _deliver_snapshots(self) -> list[list[int]]:
        """Which neighbours' snapshots arrived this epoch (Phase 7 dropout).

        Dropout is per *ordered* pair, so ``i`` may hear from ``j`` in an epoch
        where ``j`` does not hear from ``i``. A symmetric drop would quietly be a
        different stressor -- a temporarily sparser graph, which both endpoints
        agree on -- and real link failure is not that.
        """
        p = self.cfg.stress.comm_dropout
        if p <= 0.0:
            return [list(n) for n in self.neighbors]
        out = []
        for i in range(self.n_agents):
            keep = self._dropout_rng.random(len(self.neighbors[i])) >= p
            out.append([j for j, k in zip(self.neighbors[i], keep) if k])
        return out

    # -- teacher parameters on the student's own states ------------------------- #

    @torch.no_grad()
    def _teacher_bundle(self, agent_idx: int, data: dict[str, Tensor]) -> TeacherBundle:
        neighbors = list(self._epoch_neighbors[agent_idx])
        if not neighbors:
            empty = torch.zeros(0, 0, self.act_dim, device=self.device)
            return TeacherBundle(neighbors=[], teacher_mu=empty, teacher_log_std=empty)
        assert self.snapshots is not None
        mus, log_stds = [], []
        for j in neighbors:
            mu_j, log_std_j = self.snapshots[j](data["obs"])
            mus.append(mu_j)
            log_stds.append(log_std_j)
        return TeacherBundle(
            neighbors=neighbors,
            teacher_mu=torch.stack(mus).detach(),
            teacher_log_std=torch.stack(log_stds).detach(),
        )

    def _per_neighbor_kl(
        self, state: TeacherBundle, out: dict[str, Tensor], idx: Tensor
    ) -> Tensor:
        """``E_s[ KL( pi_theta_i(.|s) || pi_j^(t)(.|s) ) ]`` per neighbour, shape (J,).

        Reverse KL, student first (invariant I2). Differentiable w.r.t.
        ``theta_i`` only; the teacher tensors are detached constants (I6).
        """
        n = len(state.neighbors)
        mu = out["mu"].unsqueeze(0).expand(n, -1, -1)
        log_std = out["log_std"].unsqueeze(0).expand(n, -1, -1)
        kl = gaussian_kl(
            mu, log_std, state.teacher_mu[:, idx], state.teacher_log_std[:, idx]
        )
        return kl.mean(dim=1)

    def _social_terms(
        self, agent_idx: int, out: dict[str, Tensor], idx: Tensor
    ) -> Tensor | None:
        """Per-neighbour contributions to ``L_soc``. Unweighted in the base."""
        state = self._social[agent_idx]
        if state is None or not state.neighbors:
            return None
        return self._per_neighbor_kl(state, out, idx)

    # -- the inner loop --------------------------------------------------------- #

    def _extra_loss(
        self,
        agent_idx: int,
        ac: ActorCritic,
        mb_obs: Tensor,
        idx: Tensor,
        out: dict[str, Tensor],
    ) -> Tensor:
        zero = torch.zeros((), device=mb_obs.device, dtype=mb_obs.dtype)
        if not self._uses_loss_path:
            return zero  # handled in _apply_extra_gradients
        terms = self._social_terms(agent_idx, out, idx)
        if terms is None:
            return zero
        # L = L_CLIP - beta * L_soc - c_V * L_V is *maximised*; the minimised
        # loss therefore carries +beta * L_soc.
        return self._beta * terms.sum()

    def _apply_extra_gradients(
        self, agent_idx: int, ac: ActorCritic, out: dict[str, Tensor], idx: Tensor
    ) -> None:
        """Byzantine-robust aggregation of per-neighbour distillation gradients.

        SPEC Sec. 2.6: instead of summing ``w_ij * grad KL_ij``, combine the
        per-edge gradients with a robust operator. This is the one place that
        needs ``policy_parameters()`` (invariant I9): ``torch.autograd.grad``
        over the full parameter list would error on the value head, which the
        distillation term never touches.
        """
        if self._uses_loss_path:
            return None
        terms = self._social_terms(agent_idx, out, idx)
        if terms is None or self._beta == 0.0:
            return None

        params = ac.policy_parameters()
        per_edge = []
        for j in range(terms.shape[0]):
            grads = torch.autograd.grad(terms[j], params, retain_graph=True)
            per_edge.append(torch.cat([g.reshape(-1) for g in grads]))
        stacked = torch.stack(per_edge)
        stacked = self._corrupt_byzantine_edges(agent_idx, stacked)
        aggregated = aggregate(
            stacked,
            self.cfg.distill.aggregation,
            trim_fraction=self.cfg.distill.trimmed_mean_fraction,
            n_byzantine=self.cfg.distill.krum_byzantine,
        )
        offset = 0
        for p in params:
            n = p.numel()
            chunk = aggregated[offset : offset + n].view_as(p)
            p.grad = chunk * self._beta if p.grad is None else p.grad + self._beta * chunk
            offset += n
        return None

    # -- epoch update ------------------------------------------------------------ #

    def _prepare_epoch(self, agent_idx: int, data: dict[str, Tensor]) -> TeacherBundle:
        return self._teacher_bundle(agent_idx, data)

    # -- Phase 5: per-edge gradients, sign flips, consensus pull ---------------- #

    def _edge_gap(self, agent_idx: int, edge: int) -> float:
        """``|D_ij - delta_i|`` for H1's bucketing. NaN where there is no gate.

        NaiveDistill has neither ``D_ij`` nor ``delta_i`` (SPEC Sec. 10 / D4),
        so its edges are recorded unbucketed rather than being given a radius
        they must not have.
        """
        return float("nan")

    def _instrument_edges(self, agent_idx: int, ac: ActorCritic) -> None:
        """One extra forward/backward set per edge, outside the training loop.

        Deliberately *not* fused into the inner loop: instrumentation that can
        perturb the update it measures is worse than no instrumentation. This
        runs after the agent's update, on a fixed deterministic subsample.
        """
        inst = self.cfg.instrument
        if not inst.per_edge_gradients:
            return
        if self.epoch % inst.per_edge_gradient_interval != 0:
            return
        state = self._social[agent_idx]
        if state is None or not state.neighbors:
            return

        data = self.buffer.agent_tensors(agent_idx, device=self.device)
        take = min(256, data["obs"].shape[0])
        idx = torch.arange(take, device=self.device)
        out = ac.actor.evaluate_latent(data["obs"][idx], data["latent"][idx])
        terms = self._social_terms(agent_idx, out, idx)
        if terms is None:
            return

        params = ac.policy_parameters()
        edge_grads: list[Tensor] = []
        for j in range(terms.shape[0]):
            grads = torch.autograd.grad(terms[j], params, retain_graph=True)
            edge_grads.append(torch.cat([g.reshape(-1) for g in grads]).detach())

        # Reference magnitude: the policy-improvement gradient on the same
        # subsample, so the consensus share is a like-for-like comparison.
        clip_grad_norm = float("nan")
        if inst.consensus_pull:
            adv = data["advantage"][idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = torch.exp(out["log_prob"] - data["log_prob"][idx])
            clip = self.cfg.ppo.clip_ratio
            pg = -torch.min(
                ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv
            ).mean()
            pg_grads = torch.autograd.grad(pg, params, retain_graph=True, allow_unused=True)
            clip_grad_norm = float(
                torch.cat([g.reshape(-1) for g in pg_grads if g is not None]).norm()
            )

        social_grad = torch.stack(edge_grads).sum(dim=0) * self._beta
        social_norm = float(social_grad.norm())
        denom = social_norm + (clip_grad_norm if np.isfinite(clip_grad_norm) else 0.0)
        share = social_norm / denom if denom > 0 else float("nan")
        self._consensus.setdefault("social_grad_norm", []).append(social_norm)
        self._consensus.setdefault("policy_grad_norm", []).append(clip_grad_norm)
        self._consensus.setdefault("social_grad_fraction", []).append(share)

        sigma_i = float(torch.exp(out["log_std"].detach()).mean())
        for k, j in enumerate(state.neighbors):
            gap = self._edge_gap(agent_idx, k)
            flipped = self._sign_flips.observe(agent_idx, j, edge_grads[k], gap)
            if inst.per_edge_detail:
                teacher_sigma = float(torch.exp(state.teacher_log_std[k]).mean())
                row = {
                    "i": agent_idx,
                    "j": j,
                    "grad_norm": float(edge_grads[k].norm()),
                    "sign_flip": flipped,
                    # H2: sigma trajectories on high-D_ij edges need both ends.
                    "sigma_i": sigma_i,
                    "sigma_j": teacher_sigma,
                }
                row.update(self._edge_detail(agent_idx, k, j))
                self._edge_rows.append(row)

    def _edge_detail(self, agent_idx: int, k: int, j: int) -> dict[str, Any]:
        """Extra per-edge fields. The base has no gate, so nothing to add."""
        return {}

    def _after_agent_update(self, agent_idx: int, ac: ActorCritic) -> None:
        super()._after_agent_update(agent_idx, ac)
        self._instrument_edges(agent_idx, ac)

    def update(self) -> dict[str, Any]:
        d = self.cfg.distill
        if self.snapshots is None:
            raise RuntimeError(
                f"{type(self).__name__}.on_epoch_start() was not called: there are no "
                "frozen teacher snapshots for this epoch (SPEC Sec. 2.1 step 1)."
            )
        self._beta = self._current_beta()
        self._sign_flips.reset_epoch()
        self._edge_rows = []
        self._consensus = {}

        # Prepared for every agent before any parameter moves, so each agent
        # sees the same frozen snapshots (no sequential update order).
        for i in range(self.n_agents):
            data = self.buffer.agent_tensors(i, device=self.device)
            self._social[i] = self._prepare_epoch(i, data)

        metrics = super().update()
        metrics.update(
            {
                "beta": self._beta,
                "snapshot_bytes": self.epoch_snapshot_bytes,
                "snapshot_bytes_total": self.total_snapshot_bytes,
                "calibration_id": (
                    self.calibration_record.calibration_id
                    if self.calibration_record
                    else None
                ),
            }
        )
        metrics.update(self._distill_metrics())
        metrics.update(self._instrumentation_metrics())
        return metrics

    def _instrumentation_metrics(self) -> dict[str, Any]:
        inst = self.cfg.instrument
        qs = inst.quantiles
        out: dict[str, Any] = {}
        if inst.per_edge_gradients:
            out.update(self._sign_flips.summary(qs))
            norms = [r["grad_norm"] for r in self._edge_rows]
            out.update(quantiles(norms, qs, "social_grad_norm_"))
        if inst.consensus_pull:
            for key, values in self._consensus.items():
                out.update(quantiles(values, qs, f"consensus_pull/{key}_"))
        if inst.per_edge_detail and self._edge_rows:
            out["edges"] = self._edge_rows
        return out

    def _current_beta(self) -> float:
        return self.cfg.distill.beta(self.epoch, self.cfg.train.total_epochs)

    def _distill_metrics(self) -> dict[str, Any]:
        return {}

    # -- checkpointing ------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["snapshots"] = (
            [s.state_dict() for s in self.snapshots] if self.snapshots else None
        )
        state["total_snapshot_bytes"] = self.total_snapshot_bytes
        return state

    def _restore_snapshots(self, saved: list[dict] | None) -> list[TanhGaussianActor] | None:
        if not saved:
            return None
        restored = []
        for i, sd in enumerate(saved):
            actor = self._freeze(self.agents[i].actor)
            actor.load_state_dict(sd)
            for p in actor.parameters():
                p.requires_grad_(False)
            restored.append(actor)
        return restored

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.total_snapshot_bytes = int(state.get("total_snapshot_bytes", 0))
        self.snapshots = self._restore_snapshots(state.get("snapshots"))


# --------------------------------------------------------------------------- #
# NaiveDistill (SPEC Sec. 10 / D4)
# --------------------------------------------------------------------------- #


class NaiveDistill(SnapshotDistillBase):
    """Distillation from every neighbour with ``w_ij = 1``, structurally.

    SPEC Sec. 5 Phase 4 calls for "D-TrOPD with ``w_ij = 1``", and SPEC Sec. 10
    / D4 requires that this be its own class rather than a D-TrOPD config.
    Accordingly there is **no trust radius, no attenuation, and no code path
    for either** here: :meth:`SnapshotDistillBase._social_terms` returns the
    unweighted per-neighbour KLs and nothing multiplies them.

    Nothing in this class reads ``delta_base``, ``gamma_attenuation``,
    ``kappa``, ``attenuation``, ``radius_signal`` or ``radius_reference``.
    ``tests/test_baselines.py`` asserts that the class exposes no such
    attribute, so the baseline can never be confused with a mis-calibrated
    D-TrOPD.

    Expected behaviour under heterogeneity: importing at full strength from a
    behaviourally distant neighbour is exactly the failure mode the trust gate
    exists to prevent, so this should visibly collapse where D-TrOPD does not.
    """

    name = "naive_distill"
    reports_trust_gate = False

    def _distill_metrics(self) -> dict[str, Any]:
        # Reported so that a NaiveDistill run is self-identifying in the logs.
        return {"trust_weight_mean": 1.0, "trust_gate": "none"}


# --------------------------------------------------------------------------- #
# D-TrOPD
# --------------------------------------------------------------------------- #


class DTrOPD(SnapshotDistillBase):
    """Decentralized Trust-Region Policy Distillation (SPEC Sec. 2).

    Per epoch, identically and in parallel for every agent:

    1. broadcast a frozen detached copy of ``theta_i^(t)`` (``on_epoch_start``);
    2. roll out locally, GAE (inherited from :class:`BaseAlgorithm`);
    3. ``Delta_i`` from own policy nonstationarity -> ``delta_i`` (Sec. 2.2);
    4. ``D_ij`` (reverse KL, student first) -> ``w_ij`` (Sec. 2.3);
    5. inner loop on ``L_CLIP - beta(t) * L_soc - c_V * L_V`` with KL early
       stopping (Sec. 2.4).

    Both trust regions are live and distinct: ``eta`` bounds how far *this*
    agent moves (clip + early stop), ``delta`` bounds how much it imports from
    each peer (``w_ij``).
    """

    name = "dtropd"
    reports_trust_gate = True

    #: mean w_ij above this means the attenuation is doing essentially nothing
    INERT_GATE_THRESHOLD = 0.95

    def __init__(self, cfg: Config, seed: int, device: str | torch.device = "cpu"):
        super().__init__(cfg, seed, device)
        d = cfg.distill
        if d.attenuation not in DistillConfig._ATTENUATIONS:
            raise ConfigError(f"unknown distill.attenuation {d.attenuation!r}")
        if d.radius_signal not in DistillConfig._RADIUS_SIGNALS:
            raise ConfigError(f"unknown distill.radius_signal {d.radius_signal!r}")
        if d.trust_batch not in DistillConfig._TRUST_BATCHES:
            raise ConfigError(f"unknown distill.trust_batch {d.trust_batch!r}")
        if d.calibration not in DistillConfig._CALIBRATIONS:
            raise ConfigError(f"unknown distill.calibration {d.calibration!r}")
        if d.radius_reference not in DistillConfig._RADIUS_REFERENCES:
            raise ConfigError(f"unknown distill.radius_reference {d.radius_reference!r}")

        # theta^(t-1) snapshots: the previous epoch's broadcast. Storing last
        # epoch's *weights* is what makes Delta measurable -- the buffer's
        # stored (mu, log_std) give pi^(t) on own states, but
        # KL(pi^(t) || pi^(t-1)) needs pi^(t-1) re-evaluated on those states.
        self.prev_snapshots: list[TanhGaussianActor] | None = None

        # Effective radius scale. Starts at the configured constant; D1's
        # calibration may replace it, D5's running_quantile may track it.
        self.delta_base: float = float(d.delta_base)
        self.gamma_attenuation: float = float(d.gamma_attenuation)
        self._calibrated = d.calibration in ("none", "file")
        if d.calibration == "file":
            # SPEC Sec. 11 / D6: pinned from the shared artifact, not re-measured.
            assert self.calibration_record is not None
            self.delta_base = self.calibration_record.delta_base
            self.gamma_attenuation = self.calibration_record.gamma_attenuation
        # beta is held at exactly 0 for the whole calibration window and the
        # warmup ramp starts after it, so the measurement provably cannot be
        # influenced by the gate it is calibrating (SPEC Sec. 10 / D1).
        self._beta_epoch_offset = 0
        self._calibration_samples: list[float] = []
        #: D_ij samples keyed by epoch, so the final N epochs can be pooled (D6).
        self._calibration_by_epoch: dict[int, list[float]] = {}
        self._divergence_history: list[float] = []
        self._first_epoch_median: float | None = None
        self._warned_inert_gate = False

    def on_epoch_start(self) -> None:
        # This epoch's theta^(t) becomes next epoch's theta^(t-1).
        self.prev_snapshots = self.snapshots
        super().on_epoch_start()

    # -- SPEC Sec. 2.2: nonstationarity and radius ------------------------------ #

    @torch.no_grad()
    def _nonstationarity(self, agent_idx: int, data: dict[str, Tensor]) -> float:
        """``Delta_i^(t)``, in nats, on the agent's own rollout states."""
        signal = self.cfg.distill.radius_signal
        if signal == "fixed":
            # kappa is pinned to 0 by config validation; the radius is delta_base.
            return 0.0
        if signal == "nonstationarity":
            if self.prev_snapshots is None:
                return 0.0  # SPEC Sec. 2.2: Delta = 0 at t = 0
            mu_prev, log_std_prev = self.prev_snapshots[agent_idx](data["obs"])
            return float(
                mean_gaussian_kl(data["mu"], data["log_std"], mu_prev, log_std_prev)
            )
        if signal == "value_variance":
            # Superseded design, kept only as the H3 ablation arm. It is the
            # signal invariant I1 forbids: a degenerate critic has zero value
            # variance and therefore earns the *maximal* radius, and the
            # quantity scales with the square of the reward scale.
            return float(data["value"].var(unbiased=False))
        raise ConfigError(f"unknown distill.radius_signal {signal!r}")

    def _effective_delta_base(self) -> float:
        """SPEC Sec. 10 / D5: ``fixed`` (default) or a running quantile of D_ij."""
        if self.cfg.distill.radius_reference == "fixed":
            return self.delta_base
        window = self.cfg.distill.radius_reference_window
        history = self._divergence_history[-window:]
        if not history:
            return self.delta_base
        return float(np.quantile(history, self.cfg.distill.calibration_quantile))

    # -- SPEC Sec. 2.3: social divergence and attenuation ----------------------- #

    @torch.no_grad()
    def _trust_states(
        self, agent_idx: int, data: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """States on which ``D_ij`` is measured, with pi_i evaluated on them.

        Invariant I3: the default is the agent's **own** rollout states, and
        that is what the returned tensors are. ``trust_batch='pooled'`` is the
        Sec. 6 ablation arm and is the only way to get anything else.
        """
        if self.cfg.distill.trust_batch == "own":
            return data["obs"], data["mu"], data["log_std"]

        buf = self.buffer
        pooled = torch.as_tensor(buf.obs.reshape(-1, self.obs_dim), device=self.device)
        assert self.snapshots is not None
        mu_i, log_std_i = self.snapshots[agent_idx](pooled)
        return pooled, mu_i, log_std_i

    @torch.no_grad()
    def _prepare_epoch(self, agent_idx: int, data: dict[str, Tensor]) -> SocialState:
        """SPEC Sec. 2.1 steps 3-4, executed once per epoch per agent."""
        d = self.cfg.distill
        bundle = self._teacher_bundle(agent_idx, data)
        nonstat = self._nonstationarity(agent_idx, data)
        delta = trust_radius(self._effective_delta_base(), d.kappa, nonstat)

        if not bundle.neighbors:
            return SocialState(
                neighbors=[],
                teacher_mu=bundle.teacher_mu,
                teacher_log_std=bundle.teacher_log_std,
                divergence=torch.zeros(0, device=self.device),
                weight=torch.zeros(0, device=self.device),
                nonstationarity=nonstat,
                radius=delta,
            )

        assert self.snapshots is not None
        trust_obs, trust_mu, trust_log_std = self._trust_states(agent_idx, data)
        divergence = torch.stack(
            [
                mean_gaussian_kl(trust_mu, trust_log_std, *self.snapshots[j](trust_obs))
                for j in bundle.neighbors
            ]
        ).detach()

        weight = trust_weight(
            divergence, delta, self.gamma_attenuation, d.attenuation
        ).detach()
        if d.neighbor_selection == "top_k" and d.top_k and d.top_k < len(bundle.neighbors):
            keep = torch.topk(weight, d.top_k).indices
            mask = torch.zeros_like(weight)
            mask[keep] = 1.0
            weight = weight * mask

        return SocialState(
            neighbors=bundle.neighbors,
            teacher_mu=bundle.teacher_mu,
            teacher_log_std=bundle.teacher_log_std,
            divergence=divergence,
            weight=weight,
            nonstationarity=nonstat,
            radius=delta,
        )

    def _social_terms(
        self, agent_idx: int, out: dict[str, Tensor], idx: Tensor
    ) -> Tensor | None:
        state = self._social[agent_idx]
        if state is None or not state.neighbors:
            return None
        assert isinstance(state, SocialState)
        return state.weight * self._per_neighbor_kl(state, out, idx)

    def _edge_gap(self, agent_idx: int, edge: int) -> float:
        """``|D_ij - delta_i|`` -- H1 buckets sign flips by distance to the boundary."""
        state = self._social[agent_idx]
        if not isinstance(state, SocialState) or not state.neighbors:
            return float("nan")
        return abs(float(state.divergence[edge]) - state.radius)

    def _edge_detail(self, agent_idx: int, k: int, j: int) -> dict[str, Any]:
        state = self._social[agent_idx]
        if not isinstance(state, SocialState):
            return {}
        return {
            "D_ij": float(state.divergence[k]),
            "w_ij": float(state.weight[k]),
            "delta_i": state.radius,
            "boundary_gap": abs(float(state.divergence[k]) - state.radius),
        }

    # -- SPEC Sec. 10 / D1: calibration as a documented procedure ---------------- #

    def _collect_calibration_samples(self) -> None:
        epoch_samples: list[float] = []
        for state in self._social:
            if isinstance(state, SocialState) and state.neighbors:
                epoch_samples.extend(state.divergence.cpu().numpy().tolist())
        if epoch_samples:
            self._calibration_by_epoch[self.epoch] = epoch_samples
            self._calibration_samples.extend(epoch_samples)

    def calibrate_from_samples(self, samples: Sequence[float]) -> dict[str, float]:
        """The Sec. 10 / D1 procedure, in one place, used online and offline.

        ``delta_base := quantile_q(D_ij)`` and ``gamma`` chosen so an edge at
        ``quantile_tail(D_ij)`` is attenuated to ``target_w``.
        """
        d = self.cfg.distill
        return calibrate_attenuation(
            samples,
            quantile=d.calibration_quantile,
            tail_quantile=d.calibration_tail_quantile,
            target_w=d.calibration_target_w,
        )

    def _current_beta(self) -> float:
        """beta is exactly 0 until calibration finishes; the ramp starts after.

        SPEC Sec. 10 / D1 requires the calibration measurement to be taken with
        the gate inactive. A linear warmup ramp is *not* zero after its first
        epoch, so simply reusing the warmup window would measure D_ij under a
        partly-active gate. Instead the calibration window holds beta at 0 and
        the (full-length) warmup ramp is shifted to start once calibration is
        done -- invariant I8 still gets its complete ramp.
        """
        if not self._calibrated:
            return 0.0
        return self.cfg.distill.beta(
            self.epoch - self._beta_epoch_offset, self.cfg.train.total_epochs
        )

    def _maybe_calibrate(self) -> dict[str, Any]:
        """The SPEC Sec. 11 / D6 rule, executed in-run.

        A hard ``calibration_epochs`` prefix with beta = 0; ``delta_base`` is the
        median of the D_ij pooled over the FINAL ``calibration_pool_epochs`` of
        that window; gamma pins the p95 of the same pooled sample to
        ``calibration_target_w``.

        The window length is arbitrary and SPEC says so. It is fixed rather than
        searched, and D7 measures how much it matters.
        """
        d = self.cfg.distill
        if self._calibrated or d.calibration != "warmup":
            return {}
        self._collect_calibration_samples()
        if self.epoch + 1 < d.calibration_epochs:
            return {}

        pooled_epochs = sorted(self._calibration_by_epoch)[-d.calibration_pool_epochs :]
        pooled: list[float] = []
        for e in pooled_epochs:
            pooled.extend(self._calibration_by_epoch[e])
        if not pooled:
            return {}

        self.calibration_record = CalibrationRecord.from_samples(
            pooled,
            window_epochs=d.calibration_epochs,
            pooled_epochs=pooled_epochs,
            reward_spec_version=self.cfg.env.reward_spec_version,
            git_commit=git_commit(),
            env_fingerprint=env_fingerprint(self.cfg.env),
            center_quantile=d.calibration_quantile,
            tail_quantile=d.calibration_tail_quantile,
            target_w=d.calibration_target_w,
            seed=self.seed,
            note="measured in-run (distill.calibration=warmup)",
        )
        self.delta_base = self.calibration_record.delta_base
        self.gamma_attenuation = self.calibration_record.gamma_attenuation
        self._calibrated = True
        self._beta_epoch_offset = self.epoch + 1
        logger.info(
            "calibrated at epoch %d from %d edges pooled over epochs %s: "
            "delta_base %.5g -> %.5g, gamma %.5g -> %.5g (id %s)",
            self.epoch,
            len(pooled),
            pooled_epochs,
            d.delta_base,
            self.delta_base,
            d.gamma_attenuation,
            self.gamma_attenuation,
            self.calibration_record.calibration_id,
        )
        return {
            "calibrated_at_epoch": self.epoch,
            "calibrated_delta_base": self.delta_base,
            "calibrated_gamma": self.gamma_attenuation,
            "calibration_pooled_epochs": pooled_epochs,
            "calibration_samples": len(pooled),
            "calibration_id": self.calibration_record.calibration_id,
        }

    # -- metrics ------------------------------------------------------------------ #

    def _distill_metrics(self) -> dict[str, Any]:
        calibration = self._maybe_calibrate()
        deltas, nonstats, divergences, weights = [], [], [], []
        for state in self._social:
            if not isinstance(state, SocialState):
                continue
            deltas.append(state.radius)
            nonstats.append(state.nonstationarity)
            if state.neighbors:
                divergences.append(state.divergence.cpu().numpy())
                weights.append(state.weight.cpu().numpy())

        out: dict[str, Any] = {
            "delta_mean": float(np.mean(deltas)) if deltas else float("nan"),
            "delta_min": float(np.min(deltas)) if deltas else float("nan"),
            "delta_max": float(np.max(deltas)) if deltas else float("nan"),
            "delta_base_effective": self._effective_delta_base(),
            "gamma_attenuation_effective": self.gamma_attenuation,
            "nonstationarity_mean": float(np.mean(nonstats)) if nonstats else float("nan"),
            "radius_reference": self.cfg.distill.radius_reference,
        }
        out.update(calibration)

        if divergences:
            alld = np.concatenate(divergences)
            allw = np.concatenate(weights)
            median = float(np.median(alld))
            self._divergence_history.append(median)
            if self._first_epoch_median is None:
                self._first_epoch_median = median
            # SPEC Sec. 10 / D5: the D_ij drift is logged in BOTH radius_reference
            # arms, so the confound is measured whichever arm is used.
            qs = self.cfg.instrument.quantiles
            # SPEC Sec. 11: distributions, not means. The D1 failure was
            # invisible in the mean and obvious in these quantiles.
            out.update(quantiles(alld, qs, "social_divergence_"))
            out.update(quantiles(allw, qs, "trust_weight_"))
            out.update(
                {
                    "social_divergence_p50": median,
                    # SPEC Sec. 11 / D5: the drift is logged in BOTH radius
                    # reference arms so the confound is always measured.
                    "social_divergence_drift": (
                        median / self._first_epoch_median
                        if self._first_epoch_median and self._first_epoch_median > 0
                        else float("nan")
                    ),
                }
            )
            self._warn_if_gate_is_inert(out)

        for i, state in enumerate(self._social):
            if not isinstance(state, SocialState) or not state.neighbors:
                continue
            out.setdefault("per_edge", {})[str(i)] = {
                "neighbors": state.neighbors,
                "D_ij": state.divergence.cpu().numpy().tolist(),
                "w_ij": state.weight.cpu().numpy().tolist(),
                "delta_i": state.radius,
                "Delta_i": state.nonstationarity,
            }
        return out

    def _warn_if_gate_is_inert(self, metrics: dict[str, Any]) -> None:
        """Loudly flag the one way D-TrOPD degenerates into NaiveDistill.

        If ``delta_base`` sits far above the observed social divergences then
        ``w_ij ~ 1`` on every edge and the trust region is not doing anything --
        with no error and a perfectly plausible learning curve. That is a
        result-invalidating configuration, so it must not pass silently
        (SPEC Sec. 10 / D1).
        """
        mean_w = metrics.get("trust_weight_mean")
        if self._warned_inert_gate or mean_w is None or self._beta == 0.0:
            return
        if mean_w > self.INERT_GATE_THRESHOLD:
            self._warned_inert_gate = True
            logger.warning(
                "trust gate is nearly inert: mean w_ij = %.4f > %.2f with "
                "delta_base=%.4g, gamma=%.4g and median D_ij ~ %.4g. This run is "
                "behaviourally NaiveDistill, not D-TrOPD. Calibrate it "
                "(distill.calibration: warmup, or `run.py calibrate`) before "
                "reporting it -- SPEC Sec. 10 / D1.",
                mean_w,
                self.INERT_GATE_THRESHOLD,
                self._effective_delta_base(),
                self.gamma_attenuation,
                metrics.get("social_divergence_p50", float("nan")),
            )

    # -- checkpointing ------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["prev_snapshots"] = (
            [s.state_dict() for s in self.prev_snapshots] if self.prev_snapshots else None
        )
        state["delta_base"] = self.delta_base
        state["gamma_attenuation"] = self.gamma_attenuation
        state["calibrated"] = self._calibrated
        state["beta_epoch_offset"] = self._beta_epoch_offset
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.prev_snapshots = self._restore_snapshots(state.get("prev_snapshots"))
        self.delta_base = float(state.get("delta_base", self.delta_base))
        self.gamma_attenuation = float(
            state.get("gamma_attenuation", self.gamma_attenuation)
        )
        self._calibrated = bool(state.get("calibrated", self._calibrated))
        self._beta_epoch_offset = int(state.get("beta_epoch_offset", 0))


# --------------------------------------------------------------------------- #
# H2 ablations: the KL direction, as separate named algorithms
# --------------------------------------------------------------------------- #


class ForwardKLDistill(DTrOPD):
    """D-TrOPD with the social KL reversed: ``KL(pi_j || pi_theta_i)``.

    **This is an ablation arm and must never be the default.** Invariant I2
    forbids a direction *switch inside the update*, and it stays forbidden: the
    main path has no branch, `distill.kl_direction` still refuses anything but
    ``reverse``, and the direction is chosen here by which *class* you
    instantiate. SPEC Sec. 2.3's rule is that there is exactly one direction
    everywhere in the D-TrOPD code path -- not that the forward direction may
    never be measured. H2 is precisely the experiment that measures it, and
    SPEC Sec. 5 Phase 4 anticipates it: "forward/switched arms must be separate,
    explicitly named ablation algorithms."

    H2 predicts that forward KL inflates the student's entropy: minimising
    ``KL(teacher || student)`` is mode-covering, so it drives
    ``sigma_i^2 -> sigma_j^2 + (mu_i - mu_j)^2`` -- the student widens to cover
    the teacher's mass *plus* the mean gap. Reverse KL is mode-seeking and holds
    at ``sigma_j^2``. The gap between those two limits is the mismatch term
    ``(mu_i - mu_j)^2``, so the effect is largest exactly on the high-``D_ij``
    edges the trust gate is supposed to be attenuating.
    """

    name = "forward_kl_distill"

    def _per_neighbor_kl(
        self, state: TeacherBundle, out: dict[str, Tensor], idx: Tensor
    ) -> Tensor:
        """``E_s[ KL( pi_j^(t)(.|s) || pi_theta_i(.|s) ) ]`` -- teacher first.

        The teacher tensors remain detached constants, so gradients still flow
        only to ``theta_i`` (I6 is untouched); only the argument order differs.
        """
        n = len(state.neighbors)
        mu = out["mu"].unsqueeze(0).expand(n, -1, -1)
        log_std = out["log_std"].unsqueeze(0).expand(n, -1, -1)
        kl = gaussian_kl(
            state.teacher_mu[:, idx], state.teacher_log_std[:, idx], mu, log_std
        )
        return kl.mean(dim=1)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

ALGORITHMS: dict[str, type[BaseAlgorithm]] = {
    "ippo": IPPO,
    "dtropd": DTrOPD,
    "naive_distill": NaiveDistill,
    "mappo": MAPPO,
    "happo": HAPPO,
    # H2 ablation only -- never a default, never reachable by config switch.
    "forward_kl_distill": ForwardKLDistill,
}

_PLANNED: dict[str, str] = {}


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
