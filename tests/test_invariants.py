"""SPEC Sec. 3 invariants I1-I9 as executable checks.

All nine bind as of .  The social-path invariants (I1, I3's trust-batch
half, I5's distillation half, I6, I7, I8) are checked against the live
:class:`~dtropd.algorithms.DTrOPD` update, not against config defaults: a
config field nothing reads is not a guarantee.

I7 (normalised attenuation) additionally has a dedicated file,
``tests/test_attenuation.py``, covering w(0) = 1 and C^1 continuity across the
radius boundary for a grid of (delta, gamma).
"""

from __future__ import annotations

import copy
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from dtropd import kl as kl_module
from dtropd.algorithms import (
    IPPO,
    build_algorithm,
    trust_radius,
    trust_weight,
)
from dtropd.config import Config, ConfigError, DistillConfig, NetworkConfig, PPOConfig
from dtropd.env import SyncVectorEnv
from dtropd.networks import ActorCritic, policy_parameters, value_parameters

from test_dtropd import run_epochs, tiny_config

PACKAGE = Path(__file__).resolve().parent.parent / "dtropd"


def _prepared(cfg: Config, seed: int = 0, agent: int = 0):
    """Build a D-TrOPD agent with one epoch of data and its social state ready."""
    algo = build_algorithm(cfg, seed=seed, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=seed)
    algo.on_epoch_start()
    algo.collect_rollout(vec)
    data = algo.buffer.agent_tensors(agent)
    return algo, data


def _tiny_config(n_agents: int = 3) -> Config:
    cfg = Config()
    cfg.env.n_agents = n_agents
    cfg.env.n_targets = 1
    cfg.env.capture_group_size = min(2, n_agents)
    cfg.env.max_cycles = 8
    cfg.train.num_envs = 2
    cfg.train.total_epochs = 1
    cfg.ppo.num_minibatches = 2
    cfg.ppo.inner_epochs = 2
    cfg.train.device = "cpu"
    cfg.validate(emit=False)
    return cfg


def _algo_with_rollout(cfg: Config, seed: int = 0) -> IPPO:
    algo = build_algorithm(cfg, seed=seed, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=seed)
    algo.collect_rollout(vec)
    return algo


# --------------------------------------------------------------------------- #
# I1 -- radius from policy nonstationarity, never value-function variance
# --------------------------------------------------------------------------- #


def test_i1_value_variance_radius_is_not_the_default():
    """The *config-level* half of I1 is checkable today."""
    assert DistillConfig().radius_signal == "nonstationarity"
    # Asking for the pathological signal is possible but must be loud.
    cfg = DistillConfig(radius_signal="value_variance")
    warnings_out = cfg.validate(eta=PPOConfig().kl_early_stop)
    assert any("I1" in w for w in warnings_out)


def test_i1_radius_contracts_under_policy_nonstationarity():
    """delta_i must fall as the agent's own update divergence rises."""
    cfg = tiny_config()
    _, hist = run_epochs(cfg, seed=21, epochs=3)
    assert hist[0]["nonstationarity_mean"] == 0.0  # Delta = 0 at t = 0
    assert hist[0]["delta_mean"] == pytest.approx(cfg.distill.delta_base)
    assert hist[-1]["nonstationarity_mean"] > 0.0
    assert hist[-1]["delta_mean"] < cfg.distill.delta_base

    base, kappa = cfg.distill.delta_base, cfg.distill.kappa
    radii = [trust_radius(base, kappa, d) for d in (0.0, 0.005, 0.01, 0.02)]
    assert radii == sorted(radii, reverse=True)


def test_i1_value_variance_radius_has_the_sign_pathology():
    """A degenerate critic earns the *maximal* radius -- exactly why I1 exists."""
    cfg = tiny_config(radius_signal="value_variance")
    algo, data = _prepared(cfg, seed=22)

    healthy = algo._nonstationarity(0, data)
    assert healthy > 0.0

    degenerate = dict(data)
    degenerate["value"] = torch.full_like(data["value"], 3.14)  # constant critic
    assert algo._nonstationarity(0, degenerate) == pytest.approx(0.0)

    d = cfg.distill
    assert trust_radius(d.delta_base, d.kappa, algo._nonstationarity(0, degenerate)) > (
        trust_radius(d.delta_base, d.kappa, healthy)
    ), "the degenerate critic did not earn more trust -- test setup is wrong"


def test_i1_nonstationarity_is_reward_scale_invariant_value_variance_is_not():
    """H3's prediction, as a unit test on the two radius signals."""
    cfg = tiny_config()
    algo, data = _prepared(cfg, seed=23)
    algo.prev_snapshots = [algo._freeze(ac.actor) for ac in algo.agents]
    with torch.no_grad():
        for p in algo.prev_snapshots[0].parameters():
            p.add_(torch.randn_like(p) * 0.1)

    scaled = dict(data)
    scaled["value"] = data["value"] * 10.0  # a 10x reward rescale

    algo.cfg.distill.radius_signal = "nonstationarity"
    assert algo._nonstationarity(0, scaled) == pytest.approx(
        algo._nonstationarity(0, data), rel=1e-9
    ), "the nonstationarity radius must not depend on the reward scale"

    algo.cfg.distill.radius_signal = "value_variance"
    assert algo._nonstationarity(0, scaled) == pytest.approx(
        100.0 * algo._nonstationarity(0, data), rel=1e-4
    ), "the value-variance radius should scale with the square of the reward scale"


def test_i1_dtropd_default_uses_the_nonstationarity_signal():
    cfg = tiny_config()
    assert cfg.distill.radius_signal == "nonstationarity"
    algo, _ = run_epochs(cfg, seed=24, epochs=2)
    assert algo.cfg.distill.radius_signal == "nonstationarity"


# --------------------------------------------------------------------------- #
# I2 -- exactly one KL direction, no branch switch
# --------------------------------------------------------------------------- #


def test_i2_kl_helper_has_no_direction_switch():
    sig = inspect.signature(kl_module.gaussian_kl)
    assert list(sig.parameters) == ["mu_p", "log_std_p", "mu_q", "log_std_q"]
    for bad in ("direction", "forward", "reverse", "mode"):
        assert bad not in sig.parameters


def test_i2_no_forward_kl_branch_in_the_package():
    """No forward-KL *fallback*: no code path may CHOOSE a direction at runtime.

    This replaced a blanket source scan for the token ``forward_kl``. That scan
    forbade the very thing SPEC prescribes -- Sec. 5 requires forward and
    switched arms to exist as "separate, explicitly named ablation algorithms",
    which H2 needs -- so it contradicted the spec it was protecting.

    What I2 actually forbids is a *branch*: a fallback the D-TrOPD update can
    take, which is what inflates student entropy in proportion to the mismatch
    that triggered it. A separate class you must name explicitly is not a
    fallback; it cannot be reached by misconfiguration, only by asking for it.

    The replacement is stricter where it counts. A token ban is evaded by
    renaming (``fwd_kl`` would have passed it); these checks are structural, and
    the last one below is the one the old scan could not express at all.
    """
    banned = ("kl_forward", "use_forward", "switch_kl", "flip_kl")
    offenders = []
    for path in sorted(PACKAGE.glob("*.py")):
        text = path.read_text()
        for token in banned:
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, f"forward-KL fallback leaked back in: {offenders}"

    # The forward direction may live in exactly one place: the named ablation.
    from dtropd.algorithms import (
        DTrOPD,
        ForwardKLDistill,
        NaiveDistill,
        SnapshotDistillBase,
    )

    for cls in (DTrOPD, NaiveDistill, SnapshotDistillBase):
        src = inspect.getsource(cls)
        # "forward" alone is the neural-net sense (forward pass); the direction
        # concept is the compound token.
        assert "forward_kl" not in src.lower(), (
            f"{cls.__name__} references a forward KL; the method's own path must "
            "carry exactly one direction (I2)"
        )

    # Structural: the shared per-neighbour KL puts the STUDENT first. Reverse KL
    # is KL(student || teacher), and argument order is the whole difference --
    # a swap here would be invisible to any name-based check.
    base_src = inspect.getsource(SnapshotDistillBase._per_neighbor_kl)
    call = base_src[base_src.index("gaussian_kl("):]
    student_at = call.index("mu,")
    teacher_at = call.index("state.teacher_mu")
    assert student_at < teacher_at, (
        "SnapshotDistillBase._per_neighbor_kl passes the teacher first, i.e. it "
        "computes FORWARD KL. Invariant I2 requires student-first (reverse)."
    )

    # The shared per-neighbour KL is student-first, and the ablation overrides it
    # rather than the base branching on a flag.
    assert "_per_neighbor_kl" in ForwardKLDistill.__dict__, (
        "ForwardKLDistill must OVERRIDE the KL, not switch a flag inside the base"
    )

    # And the ablation is unreachable by configuration: no config value produces
    # it. Only naming a different algorithm does. This is the property the token
    # scan could not check, and it is the one that matters.
    cfg = Config()
    cfg.distill.enabled = True
    for direction in ("forward", "switched", "symmetric", "reverse"):
        cfg.distill.kl_direction = direction
        try:
            cfg.distill.validate(eta=cfg.ppo.kl_early_stop)
        except ConfigError:
            continue  # refused outright, which is the point
        assert direction == "reverse", (
            f"kl_direction={direction!r} was accepted; only 'reverse' may be"
        )


def test_i2_config_rejects_any_other_direction():
    for direction in ("forward", "switched", "symmetric"):
        with pytest.raises(ConfigError):
            DistillConfig(kl_direction=direction).validate(eta=0.02)


# --------------------------------------------------------------------------- #
# I3 -- own rollout states, never a pooled global batch
# --------------------------------------------------------------------------- #


def test_i3_agent_update_depends_only_on_its_own_rollout_slice():
    """Corrupting another agent's data must not perturb this agent's update."""
    cfg = _tiny_config(n_agents=3)
    algo = _algo_with_rollout(cfg)
    pristine = copy.deepcopy(algo.state_dict())

    algo.update()
    reference = [p.detach().clone() for p in algo.agents[0].parameters()]

    # Rebuild from the same state and the same buffer, then corrupt agent 2.
    algo.load_state_dict(pristine)
    for arr in (algo.buffer.obs, algo.buffer.latent, algo.buffer.advantage,
                algo.buffer.returns, algo.buffer.mu, algo.buffer.log_std,
                algo.buffer.log_prob, algo.buffer.value):
        arr[:, :, 2] = np.random.default_rng(0).normal(size=arr[:, :, 2].shape).astype(arr.dtype)
    algo.update()

    for a, b in zip(reference, algo.agents[0].parameters()):
        assert torch.equal(a, b.detach()), "agent 0's update saw pooled data"


def test_i3_buffer_slices_are_per_agent():
    cfg = _tiny_config(n_agents=3)
    algo = _algo_with_rollout(cfg)
    t0 = algo.buffer.agent_tensors(0)
    t1 = algo.buffer.agent_tensors(1)
    assert t0["obs"].shape == t1["obs"].shape == (algo.buffer.batch_size, cfg.env.obs_dim)
    assert not torch.equal(t0["obs"], t1["obs"])


def test_i3_pooled_trust_batch_is_not_the_default():
    assert DistillConfig().trust_batch == "own"
    warnings_out = DistillConfig(trust_batch="pooled").validate(eta=0.02)
    assert any("I3" in w for w in warnings_out)


def test_i3_social_divergence_uses_own_rollout_states():
    """D_ij must not move when *another* agent's rollout states change."""
    cfg = tiny_config(n_agents=4)
    algo, data = _prepared(cfg, seed=25)
    baseline = algo._prepare_epoch(0, data).divergence.clone()

    # Corrupt every other agent's slice of the buffer. Under 'own' this is
    # invisible to agent 0; under 'pooled' it would not be.
    rng = np.random.default_rng(0)
    for other in range(1, cfg.env.n_agents):
        algo.buffer.obs[:, :, other] = rng.normal(
            size=algo.buffer.obs[:, :, other].shape
        ).astype(np.float32)
    after = algo._prepare_epoch(0, data).divergence
    assert torch.equal(baseline, after), "D_ij saw another agent's states"

    algo.cfg.distill.trust_batch = "pooled"
    pooled = algo._prepare_epoch(0, data).divergence
    assert not torch.allclose(baseline, pooled), (
        "the pooled arm is indistinguishable from the own-rollout arm"
    )


def test_i3_social_loss_is_evaluated_on_own_states():
    """L_soc's teacher parameters are cached on the agent's own observations."""
    cfg = tiny_config(n_agents=3)
    algo, data = _prepared(cfg, seed=26)
    state = algo._prepare_epoch(0, data)
    j, teacher = 0, algo.snapshots[state.neighbors[0]]
    mu_expected, log_std_expected = teacher(data["obs"])
    assert torch.allclose(state.teacher_mu[j], mu_expected)
    assert torch.allclose(state.teacher_log_std[j], log_std_expected)
    assert state.teacher_mu.shape[1] == algo.buffer.batch_size


# --------------------------------------------------------------------------- #
# I4 -- critic loss present and optimised, explained variance logged
# --------------------------------------------------------------------------- #


def test_i4_critic_loss_is_mandatory_in_config():
    with pytest.raises(ConfigError):
        PPOConfig(value_coef=0.0).validate()


def test_i4_critic_parameters_change_and_ev_is_logged():
    cfg = _tiny_config()
    algo = _algo_with_rollout(cfg)
    before = [p.detach().clone() for p in algo.agents[0].critic.parameters()]
    metrics = algo.update()
    after = list(algo.agents[0].critic.parameters())
    assert any(not torch.equal(a, b.detach()) for a, b in zip(before, after)), (
        "critic parameters did not move: the value loss is not being optimised"
    )
    assert "explained_variance" in metrics
    assert "value_loss" in metrics
    assert np.isfinite(metrics["value_loss"])


def test_i4_explained_variance_definition():
    from dtropd.utils import explained_variance

    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(y, y) == pytest.approx(1.0)
    assert explained_variance(np.full_like(y, y.mean()), y) == pytest.approx(0.0)
    assert explained_variance(-y, y) < 0.0


# --------------------------------------------------------------------------- #
# I5 -- both trust regions
# --------------------------------------------------------------------------- #


def test_i5_kl_early_stop_halts_the_inner_loop():
    cfg = _tiny_config()
    cfg.ppo.inner_epochs = 8
    cfg.ppo.kl_early_stop = 1e-8  # any update immediately exceeds this
    cfg.ppo.lr = 1e-2
    algo = _algo_with_rollout(cfg)
    metrics = algo.update()
    assert metrics["inner_epochs_used"] < cfg.ppo.inner_epochs
    assert metrics["early_stopped"] == 1.0
    assert metrics["update_kl"] >= cfg.ppo.kl_early_stop


def test_i5_kl_early_stop_is_not_optional():
    with pytest.raises(ConfigError):
        PPOConfig(kl_early_stop=0.0).validate()


def test_i5_clipping_is_applied():
    """The PPO ratio clip bounds the surrogate for large ratios."""
    cfg = _tiny_config()
    algo = _algo_with_rollout(cfg)
    metrics = algo.update()
    assert "clip_fraction" in metrics and np.isfinite(metrics["clip_fraction"])
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    assert metrics["update_kl"] <= 10 * cfg.ppo.kl_early_stop, (
        "update moved far past the improvement trust region"
    )


def test_i5_distillation_trust_region_is_enforced():
    """A divergent peer is attenuated; both regions are live simultaneously."""
    cfg = tiny_config(n_agents=3, delta_base=0.05, gamma_attenuation=2.0)
    algo, data = _prepared(cfg, seed=27)
    state = algo._prepare_epoch(0, data)

    # Push one neighbour far away in policy space; its weight must collapse.
    far = state.neighbors[0]
    with torch.no_grad():
        for p in algo.snapshots[far].parameters():
            p.add_(torch.randn_like(p) * 2.0)
    after = algo._prepare_epoch(0, data)
    assert after.divergence[0] > state.divergence[0]
    assert after.weight[0] < state.weight[0]
    assert float(after.weight[0]) < 0.5

    # ...and the improvement region is still enforced in the same update.
    _, hist = run_epochs(
        tiny_config(beta_max=0.5, warmup_epochs=0), seed=27, epochs=2
    )
    assert hist[-1]["update_kl"] <= 10 * cfg.ppo.kl_early_stop
    assert np.isfinite(hist[-1]["clip_fraction"])
    assert hist[-1]["trust_weight_max"] <= 1.0


def test_i5_trust_weight_is_a_detached_gate():
    """w_ij is held constant through the inner loop; it is not a grad path."""
    cfg = tiny_config(n_agents=3)
    algo, data = _prepared(cfg, seed=28)
    state = algo._prepare_epoch(0, data)
    assert not state.weight.requires_grad
    assert not state.divergence.requires_grad
    assert state.weight.grad_fn is None

    before = state.weight.clone()
    algo._social[0] = state
    algo._beta = 0.5
    algo._update_agent(0, algo.agents[0], algo.optimizers[0], algo.cfg.ppo)
    assert torch.equal(before, algo._social[0].weight), "w_ij moved inside the inner loop"


# --------------------------------------------------------------------------- #
# I6/I7/I8 -- frozen snapshots, normalised attenuation, beta warmup
# --------------------------------------------------------------------------- #


def test_i8_beta_warmup_schedule_is_linear_and_starts_at_zero():
    """The schedule itself; test_dtropd.py checks the algorithm honours it."""
    d = DistillConfig(beta_max=0.05, warmup_fraction=0.1)
    total = 100
    assert d.resolved_warmup_epochs(total) == 10
    assert d.beta(0, total) == 0.0
    assert d.beta(5, total) == pytest.approx(0.025)
    assert d.beta(10, total) == pytest.approx(0.05)
    assert d.beta(99, total) == pytest.approx(0.05)
    betas = [d.beta(t, total) for t in range(total)]
    assert betas == sorted(betas), "beta warmup must be monotone non-decreasing"
    assert betas[0] == 0.0 and betas[-1] == pytest.approx(d.beta_max)


def test_i6_teacher_snapshots_are_frozen_for_the_whole_epoch():
    cfg = tiny_config(beta_max=0.5, warmup_epochs=0, n_agents=3)
    algo, _ = _prepared(cfg, seed=29)
    before = [copy.deepcopy(s.state_dict()) for s in algo.snapshots]
    algo.update()
    for saved, snapshot in zip(before, algo.snapshots):
        for key, value in saved.items():
            assert torch.equal(value, snapshot.state_dict()[key]), (
                "a teacher snapshot moved during the epoch"
            )


def test_i6_teacher_parameters_are_detached_and_receive_no_gradient():
    cfg = tiny_config(beta_max=0.5, warmup_epochs=0, n_agents=3)
    algo, data = _prepared(cfg, seed=30)
    assert all(not p.requires_grad for s in algo.snapshots for p in s.parameters())

    algo._beta = 0.5
    algo._social[0] = algo._prepare_epoch(0, data)
    out = algo.agents[0].actor.evaluate_latent(data["obs"], data["latent"])
    idx = torch.arange(data["obs"].shape[0])
    loss = algo._extra_loss(0, algo.agents[0], data["obs"], idx, out)
    assert loss.requires_grad
    loss.backward()
    for snapshot in algo.snapshots:
        for p in snapshot.parameters():
            assert p.grad is None, "gradient flowed into a teacher's parameters"
    # ...while the student did receive one.
    assert any(p.grad is not None for p in algo.agents[0].policy_parameters())


def test_i7_attenuation_is_one_at_zero_divergence():
    """An identical neighbour is fully trusted (details in test_attenuation.py)."""
    assert float(trust_weight(torch.tensor(0.0, dtype=torch.float64), 0.1, 2.0)) == (
        pytest.approx(1.0, abs=1e-12)
    )

    cfg = tiny_config(n_agents=3)
    algo, data = _prepared(cfg, seed=31)
    # Make one neighbour an exact copy of the student: D_ij = 0 -> w_ij = 1.
    twin = algo.neighbors[0][0]
    algo.snapshots[twin].load_state_dict(algo.snapshots[0].state_dict())
    state = algo._prepare_epoch(0, data)
    assert float(state.divergence[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(state.weight[0]) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# I9 -- policy_parameters() excludes the value head
# --------------------------------------------------------------------------- #


def test_i9_policy_parameters_exclude_the_value_head():
    ac = ActorCritic(obs_dim=6, act_dim=2, cfg=NetworkConfig(hidden_sizes=[8]))
    pol = policy_parameters(ac)
    val = value_parameters(ac)
    pol_ids = {id(p) for p in pol}
    val_ids = {id(p) for p in val}
    assert pol_ids and val_ids
    assert pol_ids.isdisjoint(val_ids)
    assert pol_ids | val_ids == {id(p) for p in ac.parameters()}


def test_i9_autograd_grad_over_policy_parameters_works():
    """``torch.autograd.grad`` over the *full* list errors; over the helper it does not."""
    ac = ActorCritic(obs_dim=6, act_dim=2, cfg=NetworkConfig(hidden_sizes=[8]))
    obs = torch.randn(16, 6)
    mu, log_std = ac.actor(obs)
    policy_only_loss = (mu**2).mean() + log_std.mean()

    grads = torch.autograd.grad(policy_only_loss, ac.policy_parameters(), retain_graph=True)
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)

    with pytest.raises(RuntimeError):
        torch.autograd.grad(policy_only_loss, list(ac.parameters()), retain_graph=True)


def test_i9_helper_rejects_unknown_modules():
    with pytest.raises(TypeError):
        policy_parameters(torch.nn.Linear(2, 2))
