"""Phase 1 acceptance (env half): shapes, reward accounting, zones, physics.

"Env passes a random-policy rollout with correct shapes and reward accounting."
"""

from __future__ import annotations

import numpy as np
import pytest

from dtropd.config import EnvConfig, Zone
from dtropd.env import CooperativeCaptureEnv, SyncVectorEnv, ZoneKind


def _cfg(**kw) -> EnvConfig:
    base = dict(
        n_agents=12, n_targets=3, capture_group_size=3, max_cycles=25, local_ratio=0.0
    )
    base.update(kw)
    return EnvConfig(**base)


# --------------------------------------------------------------------------- #
# Shapes and API
# --------------------------------------------------------------------------- #


def test_observation_shapes_match_config():
    cfg = _cfg()
    env = CooperativeCaptureEnv(cfg, seed=0)
    obs = env.reset_array(seed=0)
    assert obs.shape == (12, cfg.obs_dim)
    assert cfg.obs_dim == 4 + 3 * 3 + 3 * 11 + 2 + 1 == 49
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert env.observation_space("agent_0").shape == (cfg.obs_dim,)
    assert env.action_space("agent_0").shape == (2,)


def test_random_policy_rollout_shapes_and_truncation():
    cfg = _cfg(max_cycles=25)
    env = CooperativeCaptureEnv(cfg, seed=1)
    rng = np.random.default_rng(0)
    env.reset_array(seed=1)
    total = 0.0
    for t in range(cfg.max_cycles):
        actions = rng.uniform(-1, 1, size=(cfg.n_agents, cfg.act_dim))
        obs, rewards, term, trunc, info = env.step_array(actions)
        assert obs.shape == (cfg.n_agents, cfg.obs_dim)
        assert rewards.shape == (cfg.n_agents,)
        assert not term
        assert trunc == (t == cfg.max_cycles - 1)
        assert np.all(np.isfinite(obs)) and np.all(np.isfinite(rewards))
        # Fully cooperative: every agent receives the same team reward.
        assert np.allclose(rewards, rewards[0])
        total += rewards[0]
    assert np.isfinite(total)


def test_pettingzoo_dict_api_roundtrip():
    cfg = _cfg(max_cycles=5)
    env = CooperativeCaptureEnv(cfg, seed=2)
    obs, infos = env.reset(seed=2)
    assert set(obs) == set(env.possible_agents)
    assert all(o.shape == (cfg.obs_dim,) for o in obs.values())
    rng = np.random.default_rng(3)
    for _ in range(cfg.max_cycles):
        actions = {a: rng.uniform(-1, 1, size=2) for a in env.possible_agents}
        obs, rew, term, trunc, info = env.step(actions)
        assert set(rew) == set(env.possible_agents)
    assert all(trunc.values())
    assert env.agents == []


def test_actions_are_clipped_and_nonfinite_rejected():
    env = CooperativeCaptureEnv(_cfg(n_agents=2, n_targets=1, capture_group_size=2), seed=0)
    env.reset_array(seed=0)
    huge = np.full((2, 2), 1e6)
    one = np.ones((2, 2))
    env2 = CooperativeCaptureEnv(_cfg(n_agents=2, n_targets=1, capture_group_size=2), seed=0)
    env2.reset_array(seed=0)
    env.step_array(huge)
    env2.step_array(one)
    assert np.allclose(env.agent_vel, env2.agent_vel)

    with pytest.raises(ValueError):
        env.step_array(np.full((2, 2), np.nan))


def test_determinism_same_seed():
    cfg = _cfg(max_cycles=10)
    rng_actions = np.random.default_rng(5).uniform(-1, 1, size=(10, cfg.n_agents, 2))

    def rollout():
        env = CooperativeCaptureEnv(cfg, seed=0)
        env.reset_array(seed=42)
        out = []
        for a in rng_actions:
            obs, r, _, _, _ = env.step_array(a)
            out.append((obs.copy(), r.copy()))
        return out

    a, b = rollout(), rollout()
    for (o1, r1), (o2, r2) in zip(a, b):
        assert np.array_equal(o1, o2)
        assert np.array_equal(r1, r2)


# --------------------------------------------------------------------------- #
# Reward accounting
# --------------------------------------------------------------------------- #


def test_reward_accounting_exact():
    """Hand-computed reward for a hand-placed configuration."""
    cfg = EnvConfig(
        n_agents=4,
        n_targets=2,
        capture_group_size=2,
        capture_radius=0.3,
        shaping_weight=1.0,
        capture_bonus=1.0,
        collision_penalty=0.1,
        agent_size=0.05,
        local_ratio=0.0,
    )
    env = CooperativeCaptureEnv(cfg, seed=0)
    env.reset_array(seed=0)
    # Target 0 at origin with agents 0,1 at distance 0.1 and 0.2 -> captured.
    # Target 1 at (2, 0); its two nearest agents are 2 and 3 at 1.0 and 1.5.
    env.target_pos = np.array([[0.0, 0.0], [2.0, 0.0]])
    env.agent_pos = np.array([[0.1, 0.0], [-0.2, 0.0], [1.0, 0.0], [3.5, 0.0]])

    rewards, info = env._rewards_and_info()

    shaping_0 = -(0.1 + 0.2) / 2
    shaping_1 = -(1.0 + 1.5) / 2
    expected = (shaping_0 + shaping_1) + 1.0 * 1  # one target captured
    assert info["captures"] == 1
    assert info["collisions"] == 0
    assert rewards[0] == pytest.approx(expected)
    assert np.allclose(rewards, rewards[0])
    assert info["shaping"] == pytest.approx(shaping_0 + shaping_1)


def test_capture_requires_the_full_group():
    """Two co-located agents do not capture when the group size is three."""
    cfg = EnvConfig(n_agents=4, n_targets=1, capture_group_size=3, capture_radius=0.3)
    env = CooperativeCaptureEnv(cfg, seed=0)
    env.reset_array(seed=0)
    env.target_pos = np.array([[0.0, 0.0]])
    env.agent_pos = np.array([[0.1, 0.0], [0.0, 0.1], [0.9, 0.0], [-0.9, 0.0]])
    _, info = env._rewards_and_info()
    assert info["captures"] == 0

    env.agent_pos = np.array([[0.1, 0.0], [0.0, 0.1], [0.0, -0.15], [-0.9, 0.0]])
    _, info = env._rewards_and_info()
    assert info["captures"] == 1


def test_collision_penalty_counts_unordered_pairs():
    cfg = EnvConfig(
        n_agents=3, n_targets=1, capture_group_size=1, collision_penalty=1.0,
        shaping_weight=0.0, capture_bonus=0.0, agent_size=0.05, local_ratio=0.0,
    )
    env = CooperativeCaptureEnv(cfg, seed=0)
    env.reset_array(seed=0)
    env.target_pos = np.array([[5.0, 5.0]])
    env.agent_pos = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]])  # all 3 pairs collide
    rewards, info = env._rewards_and_info()
    assert info["collisions"] == 3
    assert rewards[0] == pytest.approx(-3.0)


def test_episode_return_equals_sum_of_step_rewards():
    cfg = _cfg(max_cycles=7)
    vec = SyncVectorEnv(cfg, num_envs=2, seed=0, autoreset=True)
    obs = vec.reset(seed=0)
    rng = np.random.default_rng(0)
    running = np.zeros(2)
    recorded = []
    for _ in range(cfg.max_cycles):
        actions = rng.uniform(-1, 1, size=(2, cfg.n_agents, 2))
        step = vec.step(actions)
        running += step.rewards.mean(axis=1)
        for i, info in enumerate(step.infos):
            if "episode" in info:
                recorded.append((i, info["episode"]["return"]))
    assert len(recorded) == 2
    for i, ret in recorded:
        assert ret == pytest.approx(running[i], rel=1e-6)


# --------------------------------------------------------------------------- #
# Heterogeneity zones
# --------------------------------------------------------------------------- #


def test_friction_zone_halves_velocity():
    cfg = EnvConfig(
        n_agents=2,
        n_targets=1,
        capture_group_size=1,
        heterogeneous=True,
        friction_velocity_scale=0.5,
        friction_zones=[Zone(x=-0.6, y=0.0, radius=0.7)],
        mask_zones=[],
    )
    homo = cfg.homogeneous_copy()

    def run(c):
        env = CooperativeCaptureEnv(c, seed=0)
        env.reset_array(seed=0)
        env.agent_pos = np.array([[-0.6, 0.0], [3.0, 0.0]])  # agent 0 inside, agent 1 outside
        env.agent_vel = np.zeros((2, 2))
        env.step_array(np.array([[1.0, 0.0], [1.0, 0.0]]))
        return env.agent_vel.copy()

    v_het, v_hom = run(cfg), run(homo)
    assert v_het[0, 0] == pytest.approx(0.5 * v_hom[0, 0])
    assert v_het[1, 0] == pytest.approx(v_hom[1, 0])


def test_masked_zone_shrinks_sensor_range():
    cfg = EnvConfig(
        n_agents=2,
        n_targets=1,
        capture_group_size=1,
        heterogeneous=True,
        sensor_range=1.5,
        masked_sensor_range=0.5,
        friction_zones=[],
        mask_zones=[Zone(x=0.6, y=0.0, radius=0.7)],
    )
    env = CooperativeCaptureEnv(cfg, seed=0)
    env.reset_array(seed=0)
    env.agent_pos = np.array([[0.6, 0.0], [3.0, 0.0]])  # agent 0 in the mask zone
    env.target_pos = np.array([[1.6, 0.0]])  # 1.0 away from agent 0

    assert np.allclose(env.sensor_ranges(), [0.5, 1.5])
    obs = env._observations()
    # obs layout: vel(2) pos(2) | target rel(2*T) | target vis(T) | ...
    t_rel = obs[0, 4:6]
    t_vis = obs[0, 4 + 2 : 4 + 3]
    assert t_vis[0] == 0.0
    assert np.allclose(t_rel, 0.0)

    # Outside the mask zone the same 1.0-distance target is visible.
    env.agent_pos = np.array([[-3.0, 0.0], [3.0, 0.0]])
    env.target_pos = np.array([[-2.0, 0.0]])
    obs = env._observations()
    assert obs[0, 6] == 1.0
    assert np.allclose(obs[0, 4:6], [1.0, 0.0])


def test_homogeneous_env_has_no_zone_effects():
    cfg = EnvConfig(n_agents=3, n_targets=1, capture_group_size=1, heterogeneous=False)
    env = CooperativeCaptureEnv(cfg, seed=0)
    env.reset_array(seed=0)
    env.agent_pos = np.array([[-0.6, 0.0], [0.6, 0.0], [0.0, 0.0]])
    assert not env.friction_mask().any()
    assert not env.mask_zone_mask().any()
    assert np.all(env.zone_ids() == ZoneKind.NORMAL)
    assert np.allclose(env.sensor_ranges(), cfg.sensor_range)
    obs = env._observations()
    assert np.allclose(obs[:, -3:-1], 0.0)  # both zone flags


def test_zone_ids_and_flags_in_observation():
    cfg = EnvConfig(
        n_agents=3,
        n_targets=1,
        capture_group_size=1,
        heterogeneous=True,
        friction_zones=[Zone(x=-0.6, y=0.0, radius=0.7)],
        mask_zones=[Zone(x=0.6, y=0.0, radius=0.7)],
    )
    env = CooperativeCaptureEnv(cfg, seed=0)
    env.reset_array(seed=0)
    env.agent_pos = np.array([[-0.6, 0.0], [0.6, 0.0], [5.0, 5.0]])
    assert list(env.zone_ids()) == [ZoneKind.FRICTION, ZoneKind.MASKED, ZoneKind.NORMAL]
    obs = env._observations()
    assert np.allclose(obs[:, -3], [1.0, 0.0, 0.0])  # friction flag
    assert np.allclose(obs[:, -2], [0.0, 1.0, 0.0])  # masked flag


# --------------------------------------------------------------------------- #
# Vector env
# --------------------------------------------------------------------------- #


def test_vector_env_matches_single_env():
    cfg = _cfg(max_cycles=6)
    vec = SyncVectorEnv(cfg, num_envs=1, seed=123, autoreset=False)
    single = CooperativeCaptureEnv(cfg, seed=0)
    obs_v = vec.reset(seed=77)
    obs_s = single.reset_array(seed=77)
    assert np.array_equal(obs_v[0], obs_s)
    rng = np.random.default_rng(9)
    for _ in range(cfg.max_cycles):
        a = rng.uniform(-1, 1, size=(cfg.n_agents, 2))
        step = vec.step(a[None])
        o, r, _, _, _ = single.step_array(a)
        assert np.array_equal(step.obs[0], o)
        assert np.allclose(step.rewards[0], r)


def test_vector_env_autoreset_and_final_obs():
    cfg = _cfg(max_cycles=3)
    vec = SyncVectorEnv(cfg, num_envs=2, seed=0, autoreset=True)
    vec.reset(seed=0)
    rng = np.random.default_rng(0)
    for t in range(3):
        step = vec.step(rng.uniform(-1, 1, size=(2, cfg.n_agents, 2)))
    assert np.all(step.truncated)
    # final_obs is the true post-step observation; obs is the fresh reset.
    assert not np.array_equal(step.final_obs, step.obs)
    assert all("episode" in info for info in step.infos)


# --------------------------------------------------------------------------- #
# Physics fidelity vs the reference MPE implementation
# --------------------------------------------------------------------------- #


def test_physics_matches_mpe2_world():
    """Our vectorised integrator reproduces ``mpe2._mpe_utils.core.World``."""
    core = pytest.importorskip("mpe2._mpe_utils.core")

    cfg = EnvConfig(n_agents=6, n_targets=2, capture_group_size=2, max_cycles=20)
    env = CooperativeCaptureEnv(cfg, seed=0)
    env.reset_array(seed=4)

    world = core.World()
    world.dt = cfg.dt
    world.damping = cfg.damping
    world.contact_force = cfg.contact_force
    world.contact_margin = cfg.contact_margin
    world.dim_c = 0
    for i in range(cfg.n_agents):
        a = core.Agent()
        a.name = f"agent_{i}"
        a.size = cfg.agent_size
        a.movable = True
        a.collide = True
        a.silent = True
        a.u_noise = None
        a.state.p_pos = env.agent_pos[i].copy()
        a.state.p_vel = env.agent_vel[i].copy()
        a.action.u = np.zeros(2)
        world.agents.append(a)
    for k in range(cfg.n_targets):
        lm = core.Landmark()
        lm.name = f"target_{k}"
        lm.size = cfg.target_size
        lm.movable = False
        lm.collide = False
        lm.state.p_pos = env.target_pos[k].copy()
        lm.state.p_vel = np.zeros(2)
        world.landmarks.append(lm)

    rng = np.random.default_rng(11)
    for _ in range(cfg.max_cycles):
        actions = rng.uniform(-1, 1, size=(cfg.n_agents, 2))
        for i, a in enumerate(world.agents):
            a.action.u = actions[i] * cfg.accel
        world.step()
        env.step_array(actions)
        ref_pos = np.stack([a.state.p_pos for a in world.agents])
        ref_vel = np.stack([a.state.p_vel for a in world.agents])
        assert np.allclose(env.agent_pos, ref_pos, atol=1e-9, rtol=0)
        assert np.allclose(env.agent_vel, ref_vel, atol=1e-9, rtol=0)


def test_pettingzoo_parallel_api_compliance():
    """The env satisfies PettingZoo's parallel API contract."""
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test

    cfg = _cfg(n_agents=3, n_targets=1, capture_group_size=2, max_cycles=8)
    env = CooperativeCaptureEnv(cfg, seed=0)
    parallel_api_test(env, num_cycles=20)


def test_local_ratio_blends_local_and_global_terms():
    """local_ratio=0 is a pure team reward; local_ratio>0 differentiates agents."""
    shared = dict(
        n_agents=3, n_targets=1, capture_group_size=2, capture_radius=0.05,
        shaping_weight=1.0, capture_bonus=1.0, collision_penalty=0.0,
    )
    pos = np.array([[0.2, 0.0], [0.0, 0.4], [1.0, 0.0]])
    target = np.array([[0.0, 0.0]])

    def rewards_for(local_ratio):
        env = CooperativeCaptureEnv(EnvConfig(**shared, local_ratio=local_ratio), seed=0)
        env.reset_array(seed=0)
        env.agent_pos = pos.copy()
        env.target_pos = target.copy()
        r, info = env._rewards_and_info()
        return r, info

    r0, info0 = rewards_for(0.0)
    assert np.allclose(r0, r0[0])  # pure team reward
    assert r0[0] == pytest.approx(info0["team_reward"])

    r1, info1 = rewards_for(0.5)
    # local term is -distance to this agent's own nearest target
    local = -np.array([0.2, 0.4, 1.0])
    assert np.allclose(info1["local_rewards"], local)
    assert np.allclose(r1, 0.5 * info1["team_reward"] + 0.5 * local)
    assert not np.allclose(r1, r1[0])
