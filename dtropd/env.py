"""Cooperative-capture MPE environment, heterogeneity zones and stressors.

SPEC Sec. 5, Phase 1: "MPE cooperative navigation wrapper (12 agents, 3
targets, capture requires >= 3 agents co-located), heterogeneity zones
(high-friction: 50% velocity; masked-observation: sensor range 1.5 -> 0.5)".

Physics
-------
The particle dynamics are exactly those of PettingZoo MPE (``mpe2._mpe_utils.
core.World``): semi-implicit integration with damping, soft-margin contact
forces between agents, non-colliding landmarks.  They are re-implemented here
in vectorised numpy for speed -- ``tests/test_env.py`` asserts agreement with a
real ``mpe2`` ``World`` to 1e-9 over random rollouts, so the re-implementation
is checked against the reference rather than trusted.

Why not subclass ``mpe2.simple_spread`` directly:
  * the capture rule (a target counts only when >= k agents are simultaneously
    co-located) is not expressible through ``Scenario.reward`` alone without
    also replacing the observation and the action mapping;
  * stock MPE continuous actions are a 5-dim ``Box(0, 1)`` (no-op / left /
    right / down / up), which does not map cleanly onto the 2-dim tanh-squashed
    Gaussian policy the spec requires;
  * sensor-range masking has to be applied per agent, from the agent's zone.

Action space
------------
``Box(-1, 1, (2,))`` -- an (x, y) force command, the natural codomain of a
tanh-squashed policy.  It is multiplied by ``accel`` (5.0, MPE's ``sensitivity``)
before entering the physics, exactly as MPE does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .config import EnvConfig, Zone

__all__ = [
    "CooperativeCaptureEnv",
    "SyncVectorEnv",
    "ZoneKind",
    "make_env",
    "make_vector_env",
]


try:  # PettingZoo is a hard runtime dependency in practice, but the core
    # package must stay importable without it (CI matrix, minimal Colab cells).
    from pettingzoo.utils.env import ParallelEnv as _ParallelEnvBase
except Exception:  # pragma: no cover - exercised only in minimal installs

    class _ParallelEnvBase:  # type: ignore[no-redef]
        """Fallback stand-in with PettingZoo's ``unwrapped`` contract."""

        @property
        def unwrapped(self):
            return self


class ZoneKind:
    """Integer codes for per-agent zone membership (used for per-zone returns)."""

    NORMAL = 0
    FRICTION = 1
    MASKED = 2
    NAMES = ("normal", "friction", "masked")


# --------------------------------------------------------------------------- #
# Core environment
# --------------------------------------------------------------------------- #


class CooperativeCaptureEnv(_ParallelEnvBase):
    """A PettingZoo ``ParallelEnv`` implementation of a cooperative capture task.

    All agents receive the *same* team reward; the task is fully cooperative.
    Episodes are fixed length (``max_cycles``): they truncate, never terminate,
    so values must be bootstrapped at the horizon.
    """

    metadata = {"name": "cooperative_capture_v0", "is_parallelizable": True}

    #: SPEC Sec. 10 / D3. The reward definition this class implements is frozen
    #: and written out verbatim in SPEC Sec. 10. Changing the reward shape
    #: requires bumping this string, which invalidates every stored curve --
    #: which is the point: a silent reward change would make old and new runs
    #: silently incomparable.
    REWARD_SPEC_VERSION = "v1"

    #: Reward specifications this class implements. SPEC Sec. 12 / D13: versions
    #: coexist; "v1" is H7's primary condition (global team term) and
    #: "v1-local" sums the team term over N_i so Assumption 2 holds, used only
    #: by H5's locality condition.
    IMPLEMENTED_REWARD_SPECS = ("v1", "v1-local")

    def __init__(self, cfg: EnvConfig | None = None, seed: int | None = None):
        self.cfg = cfg or EnvConfig()
        self.cfg.validate()
        if self.cfg.reward_spec_version not in self.IMPLEMENTED_REWARD_SPECS:
            raise ValueError(
                f"env.reward_spec_version={self.cfg.reward_spec_version!r} but this "
                f"environment implements {self.IMPLEMENTED_REWARD_SPECS}. Reward "
                "definitions are frozen (SPEC Sec. 10 / D3); a version mismatch "
                "means the stored curves and this code disagree about what is "
                "being optimised."
            )
        self.reward_spec_version = self.cfg.reward_spec_version
        self._local_team = self.reward_spec_version == "v1-local"
        # Precompute the closed neighbourhood N_i U {i} used by the local team
        # term. Assumption 2(i) requires r_i to depend only on (s_{N_i}, a_{N_i}).
        self._team_members: list[np.ndarray] | None = None
        if self._local_team:
            adj = self.cfg.team_neighborhood or [[] for _ in range(self.cfg.n_agents)]
            self._team_members = [
                np.array(sorted({i, *adj[i]}), dtype=np.int64)
                for i in range(self.cfg.n_agents)
            ]
        self.n_agents = self.cfg.n_agents
        self.n_targets = self.cfg.n_targets
        self.obs_dim = self.cfg.obs_dim
        self.act_dim = self.cfg.act_dim

        self.possible_agents: list[str] = [f"agent_{i}" for i in range(self.n_agents)]
        self.agents: list[str] = list(self.possible_agents)

        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        self.agent_pos = np.zeros((self.n_agents, 2), dtype=np.float64)
        self.agent_vel = np.zeros((self.n_agents, 2), dtype=np.float64)
        self.target_pos = np.zeros((self.n_targets, 2), dtype=np.float64)

        # Static per-pair minimum distance for contact forces.
        self._dist_min = 2.0 * self.cfg.agent_size
        self._eye = np.eye(self.n_agents, dtype=bool)

        # Stressors (Phase 7 hooks; inert by default).
        self._motor_failure: np.ndarray = np.zeros(self.n_agents, dtype=bool)

        self._obs_space: Any = None
        self._act_space: Any = None

    # -- spaces ------------------------------------------------------------- #

    def observation_space(self, agent: str):  # noqa: D102 - PettingZoo API
        # PettingZoo requires the *same* space object on every call.
        if self._obs_space is None:
            from gymnasium import spaces

            self._obs_space = spaces.Box(
                -np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32
            )
        return self._obs_space

    def action_space(self, agent: str):  # noqa: D102 - PettingZoo API
        if self._act_space is None:
            from gymnasium import spaces

            self._act_space = spaces.Box(-1.0, 1.0, shape=(self.act_dim,), dtype=np.float32)
        return self._act_space

    # -- reset / step (array fast path) ------------------------------------- #

    def reset_array(self, seed: int | None = None) -> np.ndarray:
        """Reset and return observations as an ``(n_agents, obs_dim)`` array."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        r = self.cfg.spawn_range
        self.agent_pos = self._rng.uniform(-r, r, size=(self.n_agents, 2))
        self.agent_vel = np.zeros((self.n_agents, 2), dtype=np.float64)
        self.target_pos = self._rng.uniform(-r, r, size=(self.n_targets, 2))
        self._step_count = 0
        self.agents = list(self.possible_agents)
        return self._observations()

    def step_array(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        """Step with an ``(n_agents, act_dim)`` action array.

        Returns ``(obs, rewards, terminated, truncated, info)``.
        """
        actions = np.asarray(actions, dtype=np.float64).reshape(self.n_agents, self.act_dim)
        if not np.all(np.isfinite(actions)):
            raise ValueError("non-finite action passed to CooperativeCaptureEnv.step")
        actions = np.clip(actions, -1.0, 1.0)
        if self._motor_failure.any():
            actions = actions * (~self._motor_failure)[:, None]

        self._integrate(actions)
        self._apply_friction()

        self._step_count += 1
        rewards, info = self._rewards_and_info()
        truncated = self._step_count >= self.cfg.max_cycles
        return self._observations(), rewards, False, truncated, info

    # -- reset / step (PettingZoo dict API) --------------------------------- #

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        obs = self.reset_array(seed)
        return self._to_dict(obs), {a: {} for a in self.agents}

    def step(self, actions: dict[str, np.ndarray]):
        arr = np.stack([np.asarray(actions[a], dtype=np.float64) for a in self.possible_agents])
        obs, rewards, terminated, truncated, info = self.step_array(arr)
        obs_d = self._to_dict(obs)
        rew_d = {a: float(rewards[i]) for i, a in enumerate(self.possible_agents)}
        term_d = {a: bool(terminated) for a in self.possible_agents}
        trunc_d = {a: bool(truncated) for a in self.possible_agents}
        info_d: dict[str, dict[str, Any]] = {}
        for i, a in enumerate(self.possible_agents):
            info_d[a] = dict(info)
            info_d[a]["zone"] = int(info["zones"][i])
        if terminated or truncated:
            self.agents = []
        return obs_d, rew_d, term_d, trunc_d, info_d

    def _to_dict(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        return {a: obs[i] for i, a in enumerate(self.possible_agents)}

    # -- physics ------------------------------------------------------------ #

    def _contact_forces(self) -> np.ndarray:
        """Pairwise soft-contact forces, identical to ``World.get_collision_force``."""
        cfg = self.cfg
        delta = self.agent_pos[:, None, :] - self.agent_pos[None, :, :]  # (N, N, 2)
        dist = np.sqrt(np.sum(np.square(delta), axis=-1))  # (N, N)
        np.fill_diagonal(dist, 1.0)  # avoid 0/0; masked out below
        k = cfg.contact_margin
        penetration = np.logaddexp(0.0, -(dist - self._dist_min) / k) * k
        scale = cfg.contact_force * penetration / dist
        force = delta * scale[..., None]
        force[self._eye] = 0.0
        return force.sum(axis=1)

    def _integrate(self, actions: np.ndarray) -> None:
        cfg = self.cfg
        p_force = actions * cfg.accel
        p_force = p_force + self._contact_forces()

        # MPE order: position first (from the *old* velocity), then damping,
        # then the force impulse, then the optional speed clip.
        self.agent_pos = self.agent_pos + self.agent_vel * cfg.dt
        self.agent_vel = self.agent_vel * (1.0 - cfg.damping)
        self.agent_vel = self.agent_vel + p_force * cfg.dt
        if cfg.max_speed is not None:
            speed = np.sqrt(np.sum(np.square(self.agent_vel), axis=-1))
            over = speed > cfg.max_speed
            if np.any(over):
                self.agent_vel[over] = (
                    self.agent_vel[over] / speed[over, None] * cfg.max_speed
                )

    def _apply_friction(self) -> None:
        """High-friction zones cut velocity to ``friction_velocity_scale``."""
        if not self.cfg.heterogeneous or not self.cfg.friction_zones:
            return
        inside = self._in_zones(self.agent_pos, self.cfg.friction_zones)
        if np.any(inside):
            self.agent_vel[inside] *= self.cfg.friction_velocity_scale

    @staticmethod
    def _in_zones(pos: np.ndarray, zones: Sequence[Zone]) -> np.ndarray:
        inside = np.zeros(pos.shape[0], dtype=bool)
        for z in zones:
            d2 = (pos[:, 0] - z.x) ** 2 + (pos[:, 1] - z.y) ** 2
            inside |= d2 <= z.radius**2
        return inside

    # -- zones -------------------------------------------------------------- #

    def friction_mask(self) -> np.ndarray:
        if not self.cfg.heterogeneous:
            return np.zeros(self.n_agents, dtype=bool)
        return self._in_zones(self.agent_pos, self.cfg.friction_zones)

    def mask_zone_mask(self) -> np.ndarray:
        if not self.cfg.heterogeneous:
            return np.zeros(self.n_agents, dtype=bool)
        return self._in_zones(self.agent_pos, self.cfg.mask_zones)

    def zone_ids(self) -> np.ndarray:
        """Per-agent zone code (friction wins if the zones overlap)."""
        ids = np.full(self.n_agents, ZoneKind.NORMAL, dtype=np.int64)
        masked = self.mask_zone_mask()
        ids[masked] = ZoneKind.MASKED
        friction = self.friction_mask()
        ids[friction] = ZoneKind.FRICTION
        return ids

    def sensor_ranges(self) -> np.ndarray:
        rng = np.full(self.n_agents, self.cfg.sensor_range, dtype=np.float64)
        rng[self.mask_zone_mask()] = self.cfg.masked_sensor_range
        return rng

    # -- observations ------------------------------------------------------- #

    def _observations(self) -> np.ndarray:
        n, t = self.n_agents, self.n_targets
        ranges = self.sensor_ranges()[:, None]

        # targets: (N, T, 2)
        rel_t = self.target_pos[None, :, :] - self.agent_pos[:, None, :]
        dist_t = np.sqrt(np.sum(np.square(rel_t), axis=-1))
        vis_t = (dist_t <= ranges).astype(np.float64)
        rel_t = rel_t * vis_t[..., None]

        # other agents: (N, N, 2) then drop the diagonal -> (N, N-1, 2)
        rel_a = self.agent_pos[None, :, :] - self.agent_pos[:, None, :]
        dist_a = np.sqrt(np.sum(np.square(rel_a), axis=-1))
        vis_a = (dist_a <= ranges).astype(np.float64)
        rel_a = rel_a * vis_a[..., None]
        keep = ~self._eye
        rel_a = rel_a[keep].reshape(n, n - 1, 2)
        vis_a = vis_a[keep].reshape(n, n - 1)

        friction = self.friction_mask().astype(np.float64)[:, None]
        masked = self.mask_zone_mask().astype(np.float64)[:, None]
        remaining = np.full(
            (n, 1), 1.0 - self._step_count / self.cfg.max_cycles, dtype=np.float64
        )

        obs = np.concatenate(
            [
                self.agent_vel,
                self.agent_pos,
                rel_t.reshape(n, 2 * t),
                vis_t,
                rel_a.reshape(n, 2 * (n - 1)),
                vis_a,
                friction,
                masked,
                remaining,
            ],
            axis=1,
        )
        assert obs.shape == (n, self.obs_dim), (obs.shape, self.obs_dim)
        return obs.astype(np.float32)

    def state(self) -> np.ndarray:
        """Global state for centralised critics (MAPPO, Phase 4)."""
        return np.concatenate(
            [
                self.agent_pos.ravel(),
                self.agent_vel.ravel(),
                self.target_pos.ravel(),
            ]
        ).astype(np.float32)

    @property
    def state_dim(self) -> int:
        return 4 * self.n_agents + 2 * self.n_targets

    # -- reward ------------------------------------------------------------- #

    def _local_team_terms(self, d: np.ndarray, colliding: np.ndarray) -> np.ndarray:
        """Per-agent team term restricted to ``N_i U {i}`` (reward spec "v1-local").

        SPEC Sec. 12 / D11(b). Identical in form to the global team term, but
        every sum runs over agent ``i``'s closed neighbourhood only:

        * shaping uses the ``m`` nearest *neighbourhood* members to each target;
        * a target counts as captured for ``i`` only if ``m`` of them are inside
          the radius -- so a neighbourhood smaller than ``m`` can never capture,
          which is the honest consequence of making the signal local;
        * collisions are counted among neighbourhood members only.

        The result depends on no agent outside ``N_i U {i}``, so Assumption 2(i)
        holds -- which is exactly what makes this the environment where H5's
        prediction can be tested rather than assumed.
        """
        cfg = self.cfg
        m = cfg.capture_group_size
        assert self._team_members is not None
        out = np.zeros(self.n_agents, dtype=np.float64)
        for i, members in enumerate(self._team_members):
            sub = d[:, members]  # (T, |N_i U {i}|)
            k = min(m, sub.shape[1])
            nearest_i = np.sort(sub, axis=1)[:, :k]
            shaping_i = -nearest_i.mean(axis=1)
            captured_i = (
                np.all(nearest_i <= cfg.capture_radius, axis=1)
                if sub.shape[1] >= m
                else np.zeros(sub.shape[0], dtype=bool)
            )
            block = colliding[np.ix_(members, members)]
            collisions_i = int(np.sum(np.triu(block, k=1)))
            out[i] = (
                cfg.shaping_weight * float(shaping_i.sum())
                + cfg.capture_bonus * float(captured_i.sum())
                - cfg.collision_penalty * collisions_i
            )
        return out

    def _rewards_and_info(self) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.cfg
        m = cfg.capture_group_size

        # (T, N) distances from every target to every agent.
        d = np.sqrt(
            np.sum(
                np.square(self.target_pos[:, None, :] - self.agent_pos[None, :, :]),
                axis=-1,
            )
        )
        nearest = np.sort(d, axis=1)[:, :m]  # (T, m)
        shaping = -nearest.mean(axis=1)  # (T,)
        captured = np.all(nearest <= cfg.capture_radius, axis=1)  # (T,)

        pair_dist = np.sqrt(
            np.sum(
                np.square(self.agent_pos[:, None, :] - self.agent_pos[None, :, :]), axis=-1
            )
        )
        colliding = pair_dist < self._dist_min
        np.fill_diagonal(colliding, False)
        collisions = int(np.sum(np.triu(colliding, k=1)))
        per_agent_collisions = colliding.sum(axis=1)

        # Shared team term: the coordination signal (>= k agents per target).
        team = (
            cfg.shaping_weight * float(shaping.sum())
            + cfg.capture_bonus * float(captured.sum())
            - cfg.collision_penalty * collisions
        )
        # Per-agent local term, as in PettingZoo MPE's `local_ratio` blend:
        # progress towards this agent's own nearest target plus its own
        # collisions. It carries no capture bonus -- capture stays global.
        local = (
            -cfg.shaping_weight * d.min(axis=0)
            - cfg.collision_penalty * per_agent_collisions
        )
        if self._local_team:
            # SPEC Sec. 12 / D11(b), reward_spec_version "v1-local": the team
            # term is summed over N_i U {i} instead of over all agents, so r_i
            # depends only on the states of agent i's closed neighbourhood --
            # which is what Assumption 2(i) requires and what "v1" violates.
            team_per_agent = self._local_team_terms(d, colliding)
            rewards = (
                1.0 - cfg.local_ratio
            ) * team_per_agent + cfg.local_ratio * local
        else:
            rewards = (1.0 - cfg.local_ratio) * team + cfg.local_ratio * local

        zones = self.zone_ids()
        n_zones = len(ZoneKind.NAMES)
        collisions_by_zone = np.zeros(n_zones, dtype=np.int64)
        agents_by_zone = np.zeros(n_zones, dtype=np.int64)
        for z in range(n_zones):
            sel = zones == z
            agents_by_zone[z] = int(sel.sum())
            if np.any(sel):
                collisions_by_zone[z] = int(per_agent_collisions[sel].sum())
        info = {
            "collisions_by_zone": collisions_by_zone,
            "agents_by_zone": agents_by_zone,
            "team_reward": team,
            "local_rewards": local,
            "team_scope": "neighborhood" if self._local_team else "global",
            "captures": int(captured.sum()),
            "captured": captured.copy(),
            "collisions": collisions,
            "mean_capture_distance": float(nearest.mean()),
            "reward_spec_version": self.reward_spec_version,
            "shaping": float(shaping.sum()),
            "zones": zones,
            "step": self._step_count,
        }
        return rewards, info

    # -- stressors (Phase 7 hooks) ------------------------------------------ #

    def set_motor_failure(self, agent_indices: Iterable[int]) -> None:
        """Zero the actuation of the given agents until cleared."""
        self._motor_failure = np.zeros(self.n_agents, dtype=bool)
        for i in agent_indices:
            self._motor_failure[int(i)] = True

    def clear_motor_failure(self) -> None:
        self._motor_failure = np.zeros(self.n_agents, dtype=bool)

    # -- misc ---------------------------------------------------------------- #

    def render(self) -> None:  # pragma: no cover - no renderer in Phase 1
        raise NotImplementedError("rendering is not implemented")

    def close(self) -> None:
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CooperativeCaptureEnv(n_agents={self.n_agents}, n_targets={self.n_targets}, "
            f"heterogeneous={self.cfg.heterogeneous}, max_cycles={self.cfg.max_cycles})"
        )


# --------------------------------------------------------------------------- #
# Synchronous vector env
# --------------------------------------------------------------------------- #


@dataclass
class VectorStep:
    obs: np.ndarray  # (K, N, obs_dim) -- observation *after* any auto-reset
    rewards: np.ndarray  # (K, N)
    terminated: np.ndarray  # (K,)
    truncated: np.ndarray  # (K,)
    final_obs: np.ndarray  # (K, N, obs_dim) -- pre-reset obs, valid where done
    infos: list[dict[str, Any]]


class SyncVectorEnv:
    """``num_envs`` independent copies of :class:`CooperativeCaptureEnv`.

    Batching over environments (rather than over timesteps) is what keeps the
    CPU path fast: one policy forward pass covers all ``num_envs`` copies.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        num_envs: int,
        seed: int = 0,
        *,
        autoreset: bool = True,
        motor_failures: Sequence[int] | None = None,
    ):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.autoreset = autoreset
        self.envs = [CooperativeCaptureEnv(cfg, seed=seed + 1000 * i) for i in range(self.num_envs)]
        # Phase 7 stressor: dead actuators. Applied to every copy so the failed
        # set is a property of the run, not of one environment instance.
        self.motor_failures = sorted(motor_failures or [])
        if self.motor_failures:
            for env in self.envs:
                env.set_motor_failure(self.motor_failures)
        self.n_agents = cfg.n_agents
        self.obs_dim = cfg.obs_dim
        self.act_dim = cfg.act_dim
        self._seed = seed
        self._episode_returns = np.zeros((self.num_envs, self.n_agents))
        self._episode_zone_returns = np.zeros((self.num_envs, len(ZoneKind.NAMES)))
        self._episode_captures = np.zeros(self.num_envs)
        self._episode_collisions_by_zone = np.zeros((self.num_envs, len(ZoneKind.NAMES)))
        self._episode_steps_by_zone = np.zeros((self.num_envs, len(ZoneKind.NAMES)))
        self._episode_length = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = []
        for i, env in enumerate(self.envs):
            s = None if seed is None else seed + 1000 * i
            obs.append(env.reset_array(s))
        self._episode_returns[:] = 0.0
        self._episode_zone_returns[:] = 0.0
        self._episode_captures[:] = 0.0
        self._episode_collisions_by_zone[:] = 0.0
        self._episode_steps_by_zone[:] = 0.0
        self._episode_length[:] = 0
        return np.stack(obs)

    def step(self, actions: np.ndarray) -> VectorStep:
        actions = np.asarray(actions).reshape(self.num_envs, self.n_agents, self.act_dim)
        obs = np.zeros((self.num_envs, self.n_agents, self.obs_dim), dtype=np.float32)
        final_obs = np.zeros_like(obs)
        rewards = np.zeros((self.num_envs, self.n_agents), dtype=np.float64)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []

        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step_array(actions[i])
            rewards[i] = r
            terminated[i] = term
            truncated[i] = trunc
            self._episode_returns[i] += r
            self._episode_captures[i] += info["captures"]
            self._episode_collisions_by_zone[i] += info["collisions_by_zone"]
            self._episode_steps_by_zone[i] += info["agents_by_zone"]
            self._episode_length[i] += 1
            zones = info["zones"]
            for z in range(len(ZoneKind.NAMES)):
                sel = zones == z
                if np.any(sel):
                    self._episode_zone_returns[i, z] += float(r[sel].mean())
            final_obs[i] = o
            if (term or trunc) and self.autoreset:
                info = dict(info)
                info["episode"] = {
                    "return": float(self._episode_returns[i].mean()),
                    # Per-agent, so a subset can be averaged: H6 reports the
                    # HONEST agents' return, and folding the saboteurs into the
                    # headline number would measure the attack's self-harm
                    # alongside the defence's success.
                    "agent_returns": self._episode_returns[i].copy(),
                    "captures": float(self._episode_captures[i]),
                    "length": int(self._episode_length[i]),
                    "zone_returns": self._episode_zone_returns[i].copy(),
                    "zone_collisions": self._episode_collisions_by_zone[i].copy(),
                    "zone_agent_steps": self._episode_steps_by_zone[i].copy(),
                }
                self._episode_returns[i] = 0.0
                self._episode_zone_returns[i] = 0.0
                self._episode_captures[i] = 0.0
                self._episode_collisions_by_zone[i] = 0.0
                self._episode_steps_by_zone[i] = 0.0
                self._episode_length[i] = 0
                o = env.reset_array()
            obs[i] = o
            infos.append(info)

        return VectorStep(obs, rewards, terminated, truncated, final_obs, infos)

    def state(self) -> np.ndarray:
        return np.stack([env.state() for env in self.envs])

    def close(self) -> None:
        for env in self.envs:
            env.close()


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #


def make_env(cfg: EnvConfig | None = None, seed: int | None = None) -> CooperativeCaptureEnv:
    return CooperativeCaptureEnv(cfg or EnvConfig(), seed=seed)


def make_vector_env(cfg: EnvConfig, num_envs: int, seed: int = 0, **kwargs: Any) -> SyncVectorEnv:
    return SyncVectorEnv(cfg, num_envs, seed=seed, **kwargs)
