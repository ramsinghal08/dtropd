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
from .config import Config, ConfigError, DistillConfig
from .env import SyncVectorEnv, ZoneKind
from .kl import gaussian_kl, mean_gaussian_kl
from .networks import ActorCritic, TanhGaussianActor, parameter_bytes
from .robust import aggregate
from .utils import RunningMeanStd, derive_seed, explained_variance, get_logger

__all__ = [
    "BaseAlgorithm",
    "IPPO",
    "DTrOPD",
    "SocialState",
    "trust_radius",
    "trust_weight",
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


# --------------------------------------------------------------------------- #
# D-TrOPD
# --------------------------------------------------------------------------- #


@dataclass
class SocialState:
    """Everything agent ``i`` needs from its neighbours for one epoch.

    Built once per epoch at SPEC Sec. 2.1 steps 3-4 and held constant through
    the inner loop: ``w`` is a *gate*, not a differentiable path.
    """

    neighbors: list[int]
    divergence: Tensor  # D_ij, detached, shape (J,)
    weight: Tensor  # w_ij, detached, shape (J,)
    teacher_mu: Tensor  # (J, B, act_dim) -- frozen theta_j^(t) on i's OWN states
    teacher_log_std: Tensor  # (J, B, act_dim)
    nonstationarity: float  # Delta_i^(t)
    radius: float  # delta_i^(t)


class DTrOPD(IPPO):
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

    def __init__(self, cfg: Config, seed: int, device: str | torch.device = "cpu"):
        super().__init__(cfg, seed, device)
        d = cfg.distill

        # Invariant I2 is enforced *here*, at the only place a direction could
        # be chosen, and not merely declared in the config dataclass.
        if d.kl_direction != "reverse":
            raise ConfigError(
                f"distill.kl_direction={d.kl_direction!r}: D-TrOPD computes exactly "
                "one KL direction (reverse, student first). Invariant I2 forbids a "
                "direction switch inside the update; forward/switched arms must be "
                "separate, explicitly named ablation algorithms."
            )
        if not d.enabled:
            raise ConfigError(
                "train.algo='dtropd' requires distill.enabled=true; running the "
                "D-TrOPD update with distillation disabled would silently be IPPO."
            )
        if d.attenuation not in DistillConfig._ATTENUATIONS:
            raise ConfigError(f"unknown distill.attenuation {d.attenuation!r}")
        if d.radius_signal not in DistillConfig._RADIUS_SIGNALS:
            raise ConfigError(f"unknown distill.radius_signal {d.radius_signal!r}")
        if d.trust_batch not in DistillConfig._TRUST_BATCHES:
            raise ConfigError(f"unknown distill.trust_batch {d.trust_batch!r}")

        # theta^(t) snapshots (frozen for the whole epoch, invariant I6) and
        # theta^(t-1) snapshots (the previous epoch's, for the radius signal).
        # Storing last epoch's *weights* is what makes Delta measurable: the
        # buffer's stored (mu, log_std) give pi^(t) on own states, but
        # KL(pi^(t) || pi^(t-1)) needs pi^(t-1) re-evaluated on those states.
        self.snapshots: list[TanhGaussianActor] | None = None
        self.prev_snapshots: list[TanhGaussianActor] | None = None

        self._social: list[SocialState | None] = [None] * self.n_agents
        self._beta: float = 0.0
        self._retain_graph = d.aggregation != "sum"
        self.epoch_snapshot_bytes = 0
        self.total_snapshot_bytes = 0
        self._warned_inert_gate = False

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
        # This epoch's theta^(t) becomes next epoch's theta^(t-1).
        self.prev_snapshots = self.snapshots
        self.snapshots = [self._freeze(ac.actor) for ac in self.agents]
        # Communication volume: one actor snapshot per directed edge.
        self.epoch_snapshot_bytes = sum(
            len(self.neighbors[i]) * self.snapshot_bytes for i in range(self.n_agents)
        )
        self.total_snapshot_bytes += self.epoch_snapshot_bytes

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

    # -- SPEC Sec. 2.3: social divergence and attenuation ----------------------- #

    @torch.no_grad()
    def _trust_states(self, agent_idx: int, data: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """States on which ``D_ij`` is measured, with pi_i evaluated on them.

        Invariant I3: the default is the agent's **own** rollout states, and
        that is what the returned tensors are. ``trust_batch='pooled'`` is the
        Sec. 6 ablation arm and is the only way to get anything else.
        """
        if self.cfg.distill.trust_batch == "own":
            return data["obs"], data["mu"], data["log_std"]

        buf = self.buffer
        pooled = torch.as_tensor(
            buf.obs.reshape(-1, self.obs_dim), device=self.device
        )
        assert self.snapshots is not None
        mu_i, log_std_i = self.snapshots[agent_idx](pooled)
        return pooled, mu_i, log_std_i

    @torch.no_grad()
    def _prepare_social(self, agent_idx: int, data: dict[str, Tensor]) -> SocialState | None:
        """SPEC Sec. 2.1 steps 3-4, executed once per epoch per agent."""
        d = self.cfg.distill
        neighbors = list(self.neighbors[agent_idx])
        nonstat = self._nonstationarity(agent_idx, data)
        delta = trust_radius(d.delta_base, d.kappa, nonstat)
        if not neighbors:
            return SocialState(
                neighbors=[],
                divergence=torch.zeros(0, device=self.device),
                weight=torch.zeros(0, device=self.device),
                teacher_mu=torch.zeros(0, 0, self.act_dim, device=self.device),
                teacher_log_std=torch.zeros(0, 0, self.act_dim, device=self.device),
                nonstationarity=nonstat,
                radius=delta,
            )

        assert self.snapshots is not None
        trust_obs, trust_mu, trust_log_std = self._trust_states(agent_idx, data)

        divergences = []
        teacher_mu = []
        teacher_log_std = []
        for j in neighbors:
            teacher = self.snapshots[j]
            # D_ij = E_s[ KL( pi_i || pi_j^(t) ) ] -- reverse KL, student first.
            mu_j, log_std_j = teacher(trust_obs)
            divergences.append(
                mean_gaussian_kl(trust_mu, trust_log_std, mu_j, log_std_j)
            )
            # Teacher parameters on the agent's OWN states, cached for the whole
            # inner loop: the teacher is frozen and the states are fixed, so
            # this is exact and costs one forward pass per edge per epoch.
            mu_own, log_std_own = teacher(data["obs"])
            teacher_mu.append(mu_own)
            teacher_log_std.append(log_std_own)

        divergence = torch.stack(divergences).detach()
        weight = trust_weight(
            divergence, delta, d.gamma_attenuation, d.attenuation
        ).detach()

        if d.neighbor_selection == "top_k" and d.top_k and d.top_k < len(neighbors):
            keep = torch.topk(weight, d.top_k).indices
            mask = torch.zeros_like(weight)
            mask[keep] = 1.0
            weight = weight * mask

        return SocialState(
            neighbors=neighbors,
            divergence=divergence,
            weight=weight,
            teacher_mu=torch.stack(teacher_mu).detach(),
            teacher_log_std=torch.stack(teacher_log_std).detach(),
            nonstationarity=nonstat,
            radius=delta,
        )

    # -- SPEC Sec. 2.4: the social loss inside the inner loop -------------------- #

    def _social_terms(
        self, agent_idx: int, out: dict[str, Tensor], idx: Tensor
    ) -> Tensor | None:
        """Per-neighbour ``w_ij * E_s[ KL( pi_theta_i || pi_j^(t) ) ]``, shape (J,).

        Differentiable w.r.t. ``theta_i`` through the student parameters only;
        the teacher tensors are detached constants (invariant I6) and ``w`` is a
        detached gate held constant through the inner loop.
        """
        state = self._social[agent_idx]
        if state is None or not state.neighbors:
            return None
        mu = out["mu"].unsqueeze(0).expand(len(state.neighbors), -1, -1)
        log_std = out["log_std"].unsqueeze(0).expand(len(state.neighbors), -1, -1)
        kl = gaussian_kl(
            mu, log_std, state.teacher_mu[:, idx], state.teacher_log_std[:, idx]
        )
        return state.weight * kl.mean(dim=1)

    def _extra_loss(
        self,
        agent_idx: int,
        ac: ActorCritic,
        mb_obs: Tensor,
        idx: Tensor,
        out: dict[str, Tensor],
    ) -> Tensor:
        zero = torch.zeros((), device=mb_obs.device, dtype=mb_obs.dtype)
        if self.cfg.distill.aggregation != "sum":
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
        per-edge gradients with a robust operator.  This is the one place that
        needs ``policy_parameters()`` (invariant I9): ``torch.autograd.grad``
        over the full parameter list would error on the value head, which the
        distillation term never touches.
        """
        if self.cfg.distill.aggregation == "sum":
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

    def update(self) -> dict[str, Any]:
        d = self.cfg.distill
        if self.snapshots is None:
            raise RuntimeError(
                "DTrOPD.on_epoch_start() was not called: there are no frozen "
                "teacher snapshots for this epoch (SPEC Sec. 2.1 step 1)."
            )
        self._beta = d.beta(self.epoch, self.cfg.train.total_epochs)

        # Steps 3-4 for every agent, before any parameter moves, so that every
        # agent sees the same frozen snapshots (no sequential update order).
        for i in range(self.n_agents):
            data = self.buffer.agent_tensors(i, device=self.device)
            self._social[i] = self._prepare_social(i, data)

        metrics = super().update()
        metrics.update(self._social_metrics())
        return metrics

    def _social_metrics(self) -> dict[str, Any]:
        deltas, deltas_raw, divergences, weights = [], [], [], []
        for i, state in enumerate(self._social):
            if state is None:
                continue
            deltas.append(state.radius)
            deltas_raw.append(state.nonstationarity)
            if state.neighbors:
                divergences.append(state.divergence.cpu().numpy())
                weights.append(state.weight.cpu().numpy())
        out: dict[str, Any] = {
            "beta": self._beta,
            "delta_mean": float(np.mean(deltas)) if deltas else float("nan"),
            "delta_min": float(np.min(deltas)) if deltas else float("nan"),
            "delta_max": float(np.max(deltas)) if deltas else float("nan"),
            "nonstationarity_mean": float(np.mean(deltas_raw)) if deltas_raw else float("nan"),
            "snapshot_bytes": self.epoch_snapshot_bytes,
            "snapshot_bytes_total": self.total_snapshot_bytes,
        }
        if divergences:
            alld = np.concatenate(divergences)
            allw = np.concatenate(weights)
            out.update(
                {
                    "social_divergence_mean": float(alld.mean()),
                    "social_divergence_min": float(alld.min()),
                    "social_divergence_max": float(alld.max()),
                    "trust_weight_mean": float(allw.mean()),
                    "trust_weight_min": float(allw.min()),
                    "trust_weight_max": float(allw.max()),
                }
            )
        self._warn_if_gate_is_inert(out)

        # Per-edge detail, so H1/H7 analyses need no re-run.
        for i, state in enumerate(self._social):
            if state is None or not state.neighbors:
                continue
            out.setdefault("per_edge", {})[str(i)] = {
                "neighbors": state.neighbors,
                "D_ij": state.divergence.cpu().numpy().tolist(),
                "w_ij": state.weight.cpu().numpy().tolist(),
                "delta_i": state.radius,
                "Delta_i": state.nonstationarity,
            }
        return out

    #: mean w_ij above this means the attenuation is doing essentially nothing
    INERT_GATE_THRESHOLD = 0.95

    def _warn_if_gate_is_inert(self, metrics: dict[str, Any]) -> None:
        """Loudly flag the one way D-TrOPD degenerates into NaiveDistill.

        If ``delta_base`` sits far above the observed social divergences then
        ``w_ij ~ 1`` on every edge and the trust region is not doing anything --
        with no error and a perfectly plausible learning curve. That is a
        result-invalidating configuration, so it must not pass silently.
        ``experiments/calibrate_delta.py`` measures the D_ij distribution and
        suggests values.
        """
        mean_w = metrics.get("trust_weight_mean")
        if self._warned_inert_gate or mean_w is None or self._beta == 0.0:
            return
        if mean_w > self.INERT_GATE_THRESHOLD:
            self._warned_inert_gate = True
            logger.warning(
                "trust gate is nearly inert: mean w_ij = %.4f > %.2f with "
                "delta_base=%.4g, gamma=%.4g and median D_ij ~ %.4g. This run is "
                "behaviourally NaiveDistill, not D-TrOPD. Calibrate with "
                "experiments/calibrate_delta.py before reporting it.",
                mean_w,
                self.INERT_GATE_THRESHOLD,
                self.cfg.distill.delta_base,
                self.cfg.distill.gamma_attenuation,
                metrics.get("social_divergence_mean", float("nan")),
            )

    # -- checkpointing ------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["snapshots"] = (
            [s.state_dict() for s in self.snapshots] if self.snapshots else None
        )
        state["prev_snapshots"] = (
            [s.state_dict() for s in self.prev_snapshots] if self.prev_snapshots else None
        )
        state["total_snapshot_bytes"] = self.total_snapshot_bytes
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.total_snapshot_bytes = int(state.get("total_snapshot_bytes", 0))
        for key, attr in (("snapshots", "snapshots"), ("prev_snapshots", "prev_snapshots")):
            saved = state.get(key)
            if not saved:
                setattr(self, attr, None)
                continue
            restored = []
            for i, sd in enumerate(saved):
                actor = self._freeze(self.agents[i].actor)
                actor.load_state_dict(sd)
                for p in actor.parameters():
                    p.requires_grad_(False)
                restored.append(actor)
            setattr(self, attr, restored)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

ALGORITHMS: dict[str, type[BaseAlgorithm]] = {
    "ippo": IPPO,
    "dtropd": DTrOPD,
}

_PLANNED: dict[str, str] = {
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
