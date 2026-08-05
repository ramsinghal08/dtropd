"""Phase 3 acceptance: beta = 0 must reproduce IPPO bitwise, and every
``DistillConfig`` switch must be *honoured*, not merely present.

SPEC Sec. 5, Phase 3 acceptance (critical): "with ``beta = 0``, D-TrOPD
reproduces IPPO **bitwise-identically** under the same seed. If curves differ
at all, the social path is leaking."

The second half of this file exists because a config field that nothing reads
is worse than no field at all: it looks like a guarantee. Every switch below is
driven to a value whose effect is observable, and the test asserts the effect.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from dtropd.algorithms import DTrOPD, IPPO, build_algorithm, trust_weight
from dtropd.config import Config, ConfigError
from dtropd.env import SyncVectorEnv
from dtropd.utils import derive_seed


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def tiny_config(algo: str = "dtropd", n_agents: int = 4, **distill: object) -> Config:
    cfg = Config()
    cfg.env.n_agents = n_agents
    cfg.env.n_targets = 1
    cfg.env.capture_group_size = 2
    cfg.env.max_cycles = 8
    cfg.train.algo = algo
    cfg.train.num_envs = 2
    cfg.train.total_epochs = 4
    cfg.train.device = "cpu"
    cfg.ppo.inner_epochs = 2
    cfg.ppo.num_minibatches = 2
    cfg.network.hidden_sizes = [16, 16]
    cfg.distill.enabled = algo == "dtropd"
    cfg.distill.beta_max = 0.05
    for key, value in distill.items():
        setattr(cfg.distill, key, value)
    cfg.validate(emit=False)
    return cfg


def run_epochs(cfg: Config, seed: int = 0, epochs: int | None = None):
    """Run the real per-epoch loop (snapshot -> rollout -> update)."""
    algo = build_algorithm(cfg, seed=seed, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=derive_seed(seed, "env"))
    history = []
    for epoch in range(epochs if epochs is not None else cfg.train.total_epochs):
        algo.epoch = epoch
        algo.on_epoch_start()
        rollout = algo.collect_rollout(vec)
        metrics = algo.update()
        history.append({**rollout, **{k: v for k, v in metrics.items() if k != "per_agent"}})
    return algo, history


def flat_params(algo) -> list[torch.Tensor]:
    return [p.detach().clone() for ac in algo.agents for p in ac.parameters()]


# --------------------------------------------------------------------------- #
# THE Phase 3 acceptance test
# --------------------------------------------------------------------------- #


def test_beta_zero_reproduces_ippo_bitwise():
    """beta = 0 -> every parameter of every agent is bit-identical to IPPO."""
    ippo_cfg = tiny_config(algo="ippo")
    dtropd_cfg = tiny_config(algo="dtropd", beta_max=0.0)

    ippo, ippo_hist = run_epochs(ippo_cfg, seed=7)
    dtropd, dtropd_hist = run_epochs(dtropd_cfg, seed=7)

    assert isinstance(ippo, IPPO) and isinstance(dtropd, DTrOPD)

    for k, (a, b) in enumerate(zip(flat_params(ippo), flat_params(dtropd))):
        assert torch.equal(a, b), f"parameter {k} differs: the social path is leaking"

    # ...and the curves, since that is what SPEC actually says to compare.
    for key in ("episode_return_mean", "explained_variance", "update_kl", "value_loss"):
        a = [h[key] for h in ippo_hist]
        b = [h[key] for h in dtropd_hist]
        assert a == b, f"{key} curves differ: {a} vs {b}"

    # The test is only meaningful if the social path actually ran.
    assert dtropd_hist[-1]["beta"] == 0.0
    assert dtropd_hist[-1]["snapshot_bytes"] > 0
    assert np.isfinite(dtropd_hist[-1]["social_divergence_mean"])
    assert dtropd_hist[-1]["social_divergence_max"] > 0.0
    assert 0.0 < dtropd_hist[-1]["trust_weight_mean"] <= 1.0


def test_nonzero_beta_actually_changes_the_update():
    """The converse guard: with beta > 0 the parameters must diverge from IPPO."""
    ippo, _ = run_epochs(tiny_config(algo="ippo"), seed=7)
    dtropd, hist = run_epochs(
        tiny_config(algo="dtropd", beta_max=0.5, warmup_epochs=0), seed=7
    )
    assert hist[-1]["beta"] == pytest.approx(0.5)
    assert any(
        not torch.equal(a, b) for a, b in zip(flat_params(ippo), flat_params(dtropd))
    ), "beta > 0 left the update unchanged: the social loss is not wired in"


def test_update_without_epoch_start_is_an_error():
    """Missing snapshots must fail loudly, not silently distil from stale weights."""
    cfg = tiny_config()
    algo = build_algorithm(cfg, seed=0, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=0)
    algo.collect_rollout(vec)
    with pytest.raises(RuntimeError, match="on_epoch_start"):
        algo.update()


# --------------------------------------------------------------------------- #
# Previous-policy storage (Delta needs theta^(t-1) weights, not stored outputs)
# --------------------------------------------------------------------------- #


def test_previous_policy_snapshot_is_retained_and_used():
    cfg = tiny_config()
    algo = build_algorithm(cfg, seed=1, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=1)

    algo.epoch = 0
    algo.on_epoch_start()
    assert algo.prev_snapshots is None, "there is no theta^(t-1) at t = 0"
    theta_0 = [copy.deepcopy(s.state_dict()) for s in algo.snapshots]
    algo.collect_rollout(vec)
    metrics = algo.update()
    assert metrics["nonstationarity_mean"] == 0.0, "Delta must be 0 at t = 0"
    assert metrics["delta_mean"] == pytest.approx(cfg.distill.delta_base)

    algo.epoch = 1
    algo.on_epoch_start()
    # theta^(t-1) is exactly the previous epoch's broadcast snapshot.
    assert algo.prev_snapshots is not None
    for saved, prev in zip(theta_0, algo.prev_snapshots):
        for key in saved:
            assert torch.equal(saved[key], prev.state_dict()[key])
    algo.collect_rollout(vec)
    metrics = algo.update()
    assert metrics["nonstationarity_mean"] > 0.0, (
        "Delta stayed 0 after a real update: theta^(t-1) is not being used"
    )
    assert metrics["delta_mean"] < cfg.distill.delta_base


def test_delta_measures_kl_against_previous_weights_not_stored_outputs():
    """Delta must change if theta^(t-1) changes, with the rollout held fixed."""
    cfg = tiny_config()
    algo = build_algorithm(cfg, seed=2, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=2)
    algo.on_epoch_start()
    algo.collect_rollout(vec)

    data = algo.buffer.agent_tensors(0)
    algo.prev_snapshots = [algo._freeze(ac.actor) for ac in algo.agents]
    baseline = algo._nonstationarity(0, data)
    assert baseline == pytest.approx(0.0, abs=1e-6)  # identical weights -> KL 0

    # Perturb theta^(t-1) only; the buffer (pi^(t) on own states) is untouched.
    with torch.no_grad():
        for p in algo.prev_snapshots[0].parameters():
            p.add_(torch.randn_like(p) * 0.1)
    perturbed = algo._nonstationarity(0, data)
    assert perturbed > 1e-4, "Delta ignored theta^(t-1): it is not being evaluated"


# --------------------------------------------------------------------------- #
# Every DistillConfig switch must be honoured
# --------------------------------------------------------------------------- #


def test_kl_direction_is_honoured_at_construction():
    cfg = tiny_config()
    cfg.distill.kl_direction = "forward"  # bypasses Config.validate deliberately
    with pytest.raises(ConfigError, match="I2"):
        build_algorithm(cfg, seed=0, device="cpu")


def test_enabled_flag_is_honoured():
    cfg = tiny_config()
    cfg.distill.enabled = False
    with pytest.raises(ConfigError, match="distill.enabled"):
        build_algorithm(cfg, seed=0, device="cpu")


def test_social_kl_is_reverse_student_first():
    """L_soc uses KL(pi_student || pi_teacher), verified numerically."""
    from dtropd.kl import gaussian_kl

    cfg = tiny_config()
    algo = build_algorithm(cfg, seed=3, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=3)
    algo.on_epoch_start()
    algo.collect_rollout(vec)
    algo._beta = 1.0
    data = algo.buffer.agent_tensors(0)
    algo._social[0] = algo._prepare_epoch(0, data)
    state = algo._social[0]

    # Freshly-initialised agents share log_std, and for equal variances the
    # Gaussian KL is symmetric -- which would make a direction test vacuous.
    # Give the teacher a visibly different scale so the two directions differ.
    state.teacher_log_std = state.teacher_log_std + 1.0
    state.teacher_mu = state.teacher_mu + 0.5

    idx = torch.arange(data["obs"].shape[0])
    out = algo.agents[0].actor.evaluate_latent(data["obs"], data["latent"])
    terms = algo._social_terms(0, out, idx)

    j = 0
    with torch.no_grad():
        student_first = float(
            gaussian_kl(
                out["mu"], out["log_std"], state.teacher_mu[j], state.teacher_log_std[j]
            ).mean()
        )
        teacher_first = float(
            gaussian_kl(
                state.teacher_mu[j], state.teacher_log_std[j], out["mu"], out["log_std"]
            ).mean()
        )

    assert student_first != pytest.approx(teacher_first, rel=1e-3), (
        "test setup failed to make the two directions distinguishable"
    )
    assert float(terms[j].detach()) == pytest.approx(
        float(state.weight[j]) * student_first, rel=1e-5
    ), "L_soc is not KL(student || teacher): the KL direction is wrong"


def test_radius_signal_is_honoured():
    cfg = tiny_config()
    algo = build_algorithm(cfg, seed=4, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=4)
    algo.on_epoch_start()
    algo.collect_rollout(vec)
    data = algo.buffer.agent_tensors(0)
    algo.prev_snapshots = [algo._freeze(ac.actor) for ac in algo.agents]
    with torch.no_grad():
        for p in algo.prev_snapshots[0].parameters():
            p.add_(torch.randn_like(p) * 0.1)

    algo.cfg.distill.radius_signal = "nonstationarity"
    nonstat = algo._nonstationarity(0, data)

    algo.cfg.distill.radius_signal = "value_variance"
    value_var = algo._nonstationarity(0, data)

    algo.cfg.distill.radius_signal = "fixed"
    fixed = algo._nonstationarity(0, data)

    assert nonstat > 0.0
    assert value_var == pytest.approx(float(data["value"].var(unbiased=False)), rel=1e-5)
    assert fixed == 0.0
    assert nonstat != value_var


def test_trust_batch_is_honoured():
    """'own' and 'pooled' must produce different D_ij; 'own' is the default."""
    cfg = tiny_config()
    assert cfg.distill.trust_batch == "own"
    algo = build_algorithm(cfg, seed=5, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=5)
    algo.on_epoch_start()
    algo.collect_rollout(vec)
    data = algo.buffer.agent_tensors(0)

    own = algo._prepare_epoch(0, data).divergence.clone()
    algo.cfg.distill.trust_batch = "pooled"
    pooled = algo._prepare_epoch(0, data).divergence.clone()

    assert own.shape == pooled.shape
    assert not torch.allclose(own, pooled), "the pooled path is dead code"


def test_attenuation_mode_is_honoured():
    cfg = tiny_config(delta_base=1e-3, gamma_attenuation=2.0)
    algo = build_algorithm(cfg, seed=6, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=6)
    algo.on_epoch_start()
    algo.collect_rollout(vec)
    data = algo.buffer.agent_tensors(0)

    weights = {}
    for mode in ("continuous", "hard_switch", "hard_cutoff"):
        algo.cfg.distill.attenuation = mode
        weights[mode] = algo._prepare_epoch(0, data).weight.clone()

    assert torch.all((weights["hard_cutoff"] == 0.0) | (weights["hard_cutoff"] == 1.0))
    assert not torch.allclose(weights["continuous"], weights["hard_cutoff"])
    assert not torch.allclose(weights["continuous"], weights["hard_switch"])


def test_neighbor_selection_is_honoured():
    cfg = tiny_config(n_agents=5)
    algo = build_algorithm(cfg, seed=8, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=8)
    algo.on_epoch_start()
    algo.collect_rollout(vec)
    data = algo.buffer.agent_tensors(0)

    assert algo.cfg.distill.neighbor_selection == "all"
    all_w = algo._prepare_epoch(0, data).weight
    assert torch.all(all_w > 0.0)

    algo.cfg.distill.neighbor_selection = "top_k"
    algo.cfg.distill.top_k = 2
    top_w = algo._prepare_epoch(0, data).weight
    assert int((top_w > 0).sum()) == 2, "top_k did not gate the neighbour set"


def test_aggregation_is_honoured_and_uses_policy_parameters():
    """The robust path changes the update and never touches the value head (I9)."""
    base = tiny_config(beta_max=0.5, warmup_epochs=0)
    plain, _ = run_epochs(base, seed=9, epochs=2)
    robust_cfg = tiny_config(beta_max=0.5, warmup_epochs=0, aggregation="trimmed_mean")
    robust, _ = run_epochs(robust_cfg, seed=9, epochs=2)

    assert any(
        not torch.equal(a, b) for a, b in zip(flat_params(plain), flat_params(robust))
    ), "the robust aggregation path is dead code"

    for name in ("geometric_median", "krum"):
        cfg = tiny_config(beta_max=0.5, warmup_epochs=0, aggregation=name)
        algo, _ = run_epochs(cfg, seed=9, epochs=1)
        assert all(torch.all(torch.isfinite(p)) for p in flat_params(algo))


def test_beta_warmup_is_honoured_by_the_algorithm():
    """Invariant I8: beta really ramps inside the training loop, not just in config."""
    cfg = tiny_config(beta_max=0.4, warmup_epochs=3)
    _, hist = run_epochs(cfg, seed=10, epochs=4)
    betas = [h["beta"] for h in hist]
    assert betas[0] == 0.0
    assert betas == sorted(betas)
    assert betas[3] == pytest.approx(0.4)


def test_communication_volume_is_accounted():
    """SPEC Sec. 7: report bytes of snapshots exchanged."""
    cfg = tiny_config(n_agents=4)
    algo, hist = run_epochs(cfg, seed=11, epochs=2)
    edges = sum(len(n) for n in algo.neighbors)
    assert hist[0]["snapshot_bytes"] == edges * algo.snapshot_bytes
    assert hist[1]["snapshot_bytes_total"] == 2 * hist[0]["snapshot_bytes"]


def test_per_edge_metrics_are_logged():
    cfg = tiny_config(n_agents=4)
    _, hist = run_epochs(cfg, seed=12, epochs=1)
    per_edge = hist[0]["per_edge"]
    assert set(per_edge) == {"0", "1", "2", "3"}
    row = per_edge["0"]
    assert len(row["neighbors"]) == len(row["D_ij"]) == len(row["w_ij"])
    assert all(0.0 < w <= 1.0 for w in row["w_ij"])
    assert row["delta_i"] > 0.0


def test_checkpoint_roundtrip_preserves_snapshots():
    cfg = tiny_config()
    algo, _ = run_epochs(cfg, seed=13, epochs=2)
    state = algo.state_dict()

    restored = build_algorithm(cfg, seed=13, device="cpu")
    restored.load_state_dict(state)
    assert restored.snapshots is not None and restored.prev_snapshots is not None
    for a, b in zip(algo.snapshots, restored.snapshots):
        for key, value in a.state_dict().items():
            assert torch.equal(value, b.state_dict()[key])
    assert all(not p.requires_grad for s in restored.snapshots for p in s.parameters())
