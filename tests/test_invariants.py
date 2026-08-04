"""SPEC Sec. 3 invariants I1-I9 as executable checks.

Phases 0-2 implement the environment, the KL machinery and IPPO, so the
invariants that bind *today* are I2 (one KL direction), I3 (own-rollout
batches), I4 (critic present and optimised), the improvement half of I5
(clipping + KL early stop) and I9 (``policy_parameters()`` excludes the value
head).

I1, I6, I7, I8 and the distillation half of I3/I5 constrain the social path,
which lands in Phase 3.  They are present here as explicit skips rather than
absent, so the Phase 3 acceptance run has a checklist to turn green rather
than a blank file.
"""

from __future__ import annotations

import copy
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from dtropd import kl as kl_module
from dtropd.algorithms import IPPO, build_algorithm
from dtropd.config import Config, ConfigError, DistillConfig, NetworkConfig, PPOConfig
from dtropd.env import SyncVectorEnv
from dtropd.networks import ActorCritic, policy_parameters, value_parameters

PACKAGE = Path(__file__).resolve().parent.parent / "dtropd"

PHASE3 = "social path (radius / attenuation / distillation) lands in Phase 3"


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


@pytest.mark.skip(reason=PHASE3)
def test_i1_radius_contracts_under_policy_nonstationarity():
    """delta_i must fall when the agent's own update divergence rises."""


# --------------------------------------------------------------------------- #
# I2 -- exactly one KL direction, no branch switch
# --------------------------------------------------------------------------- #


def test_i2_kl_helper_has_no_direction_switch():
    sig = inspect.signature(kl_module.gaussian_kl)
    assert list(sig.parameters) == ["mu_p", "log_std_p", "mu_q", "log_std_q"]
    for bad in ("direction", "forward", "reverse", "mode"):
        assert bad not in sig.parameters


def test_i2_no_forward_kl_branch_in_the_package():
    """Source scan: no forward-KL fallback may exist anywhere in ``dtropd/``."""
    banned = ("forward_kl", "kl_forward", "use_forward", "switch_kl", "flip_kl")
    offenders = []
    for path in sorted(PACKAGE.glob("*.py")):
        text = path.read_text()
        for token in banned:
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, f"forward-KL path leaked back in: {offenders}"


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


@pytest.mark.skip(reason=PHASE3)
def test_i3_social_divergence_uses_own_rollout_states():
    """D_ij and L_soc must be evaluated on pi_i's own states."""


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


@pytest.mark.skip(reason=PHASE3)
def test_i5_distillation_trust_region_is_enforced():
    """w_ij must bound how much is imported from each peer."""


# --------------------------------------------------------------------------- #
# I6/I7/I8 -- frozen snapshots, normalised attenuation, beta warmup
# --------------------------------------------------------------------------- #


def test_i8_beta_warmup_schedule_is_linear_and_starts_at_zero():
    """The schedule itself is testable today even though nothing consumes it yet."""
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


@pytest.mark.skip(reason=PHASE3)
def test_i6_teacher_snapshots_are_frozen_and_detached():
    """theta_j snapshots must not change within an epoch and must not receive grad."""


@pytest.mark.skip(reason=PHASE3)
def test_i7_attenuation_is_one_at_zero_divergence():
    """Covered in tests/test_attenuation.py once the attenuation lands."""


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
