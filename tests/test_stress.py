"""stressors: Byzantine agents, communication dropout, motor failures.

H6 predicts robust aggregation "degrades gracefully to f ~ 0.4 while plain-sum
collapses early". Every test here guards one way that prediction could be
confirmed by the harness rather than by the algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dtropd.algorithms import build_algorithm
from dtropd.byzantine import ATTACKS, apply_attack, byzantine_agents
from dtropd.config import Config, ConfigError
from dtropd.env import SyncVectorEnv
from dtropd.robust import aggregate
from dtropd.utils import derive_seed


def stress_config(**stress) -> Config:
    cfg = Config()
    cfg.env.n_agents = 8
    cfg.env.n_targets = 2
    cfg.env.capture_group_size = 2
    cfg.env.max_cycles = 10
    cfg.train.algo = "dtropd"
    cfg.train.num_envs = 4
    cfg.train.total_epochs = 3
    cfg.ppo.inner_epochs = 2
    cfg.ppo.num_minibatches = 2
    cfg.network.hidden_sizes = [16, 16]
    cfg.distill.enabled = True
    cfg.distill.beta_max = 0.5
    cfg.distill.warmup_fraction = 0.0
    for key, value in stress.items():
        setattr(cfg.stress, key, value)
    cfg.validate(emit=False)
    return cfg


def train_policy(cfg: Config, seed: int = 1, agent: int = 1) -> torch.Tensor:
    """Train briefly and return one agent's flattened policy parameters."""
    algo = build_algorithm(cfg, seed=seed, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=derive_seed(seed, "env"))
    for epoch in range(cfg.train.total_epochs):
        algo.epoch = epoch
        algo.on_epoch_start()
        algo.collect_rollout(vec)
        algo.update()
    return torch.cat([p.detach().reshape(-1) for p in algo.agents[agent].actor.parameters()])


# --------------------------------------------------------------------------- #
# The adaptive adversary is the one worth defending against
# --------------------------------------------------------------------------- #


def test_adaptive_attack_stays_inside_the_honest_range():
    """SPEC: "mimics honest gradients from inside the honest range".

    That phrase is the whole specification of this adversary, and it is testable
    directly: the Byzantine vectors must sit within one honest standard
    deviation, and close to the honest cloud in Euclidean distance, or the
    defences would find them by magnitude alone and H6 would be measuring
    nothing.
    """
    torch.manual_seed(0)
    honest = torch.randn(8, 50) * 0.1 + 1.0
    rows = [0, 1, 2]
    honest_rows = [3, 4, 5, 6, 7]

    attacked = apply_attack(honest, rows, "adaptive", 0.5)
    ref = attacked[honest_rows]
    z = ((attacked[rows] - ref.mean(0)).abs() / ref.std(0)).max().item()
    assert z <= 1.0, (
        f"the adaptive attack sits {z:.2f} honest standard deviations out; past "
        "1.0 it has left the honest range and is just a noisy 'gaussian'"
    )

    loud = apply_attack(honest, rows, "sign_flip", 1.0)
    loud_z = ((loud[rows] - ref.mean(0)).abs() / ref.std(0)).max().item()
    assert loud_z > 5 * z, "sign_flip should be conspicuous; that is its role here"


def test_trimming_defeats_loud_attacks_and_not_the_adaptive_one():
    """The reason H6 needs the adaptive adversary at all.

    Coordinate-wise trimming removes the loud attacks almost entirely. If the
    sweep only ever ran `sign_flip`, "robust aggregation degrades gracefully"
    would be confirmed by an adversary that defeats itself.
    """
    torch.manual_seed(0)
    honest = torch.randn(10, 40) * 0.1 + 1.0
    rows = [0, 1]
    clean_sum = aggregate(honest, "sum")
    clean_trim = aggregate(honest, "trimmed_mean", trim_fraction=0.2)

    def errors(mode, scale):
        att = apply_attack(honest, rows, mode, scale, generator=torch.Generator().manual_seed(0))
        return (
            (aggregate(att, "sum") - clean_sum).norm().item(),
            (aggregate(att, "trimmed_mean", trim_fraction=0.2) - clean_trim).norm().item(),
        )

    loud_sum, loud_trim = errors("sign_flip", 1.0)
    adapt_sum, adapt_trim = errors("adaptive", 0.5)

    assert loud_trim < loud_sum / 3, (
        "trimming should largely neutralise a loud attack; if it does not, the "
        "defence is broken rather than the adversary being clever"
    )
    assert adapt_trim > adapt_sum / 3, (
        "trimming appears to neutralise the adaptive attack, which would mean it "
        "is not staying inside the honest range"
    )


@pytest.mark.parametrize("mode", [m for m in ATTACKS if m != "none"])
def test_every_attack_changes_the_aggregate(mode):
    torch.manual_seed(0)
    honest = torch.randn(6, 20) * 0.1 + 1.0
    att = apply_attack(honest, [0], mode, 0.5, generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(att, honest), f"{mode} left the gradients untouched"
    assert torch.allclose(att[1:], honest[1:]), f"{mode} corrupted an honest row"


def test_adaptive_attack_refuses_when_there_is_nothing_to_hide_behind():
    """With every edge Byzantine there is no honest range to imitate.

    Returning silently (say, zeros) would let a fully-compromised neighbourhood
    be scored as a defended one.
    """
    honest = torch.randn(3, 10)
    with pytest.raises(ValueError, match="every edge is Byzantine"):
        apply_attack(honest, [0, 1, 2], "adaptive", 0.5)


def test_unknown_attack_is_refused():
    with pytest.raises(ValueError, match="unknown byzantine mode"):
        apply_attack(torch.randn(3, 4), [0], "definitely_not_an_attack", 1.0)


# --------------------------------------------------------------------------- #
# Who is Byzantine
# --------------------------------------------------------------------------- #


def test_byzantine_set_is_seed_derived_and_stable():
    a = byzantine_agents(12, 3, seed=7)
    assert a == byzantine_agents(12, 3, seed=7), "the set must not move within a run"
    assert len(a) == 3 and len(set(a)) == 3
    assert a != byzantine_agents(12, 3, seed=8), (
        "a different seed must redraw the set, or a result could be an artifact "
        "of one unlucky assignment repeated ten times"
    )


def test_byzantine_majority_is_refused():
    with pytest.raises(ValueError, match="no honest majority"):
        byzantine_agents(4, 4, seed=0)


@pytest.mark.parametrize(
    "fraction,n_agents,expected", [(0.0, 12, 0), (0.1, 12, 1), (0.25, 8, 2), (0.5, 12, 6)]
)
def test_byzantine_count_rounds_down(fraction, n_agents, expected):
    cfg = Config()
    cfg.stress.byzantine_fraction = fraction
    cfg.stress.byzantine_mode = "adaptive"
    assert cfg.stress.n_byzantine(n_agents) == expected


def test_a_fraction_that_rounds_to_zero_is_flagged():
    """Otherwise an `f=0.1` cell on 8 agents is the baseline wearing a label."""
    cfg = Config()
    cfg.env.n_agents = 8
    cfg.stress.byzantine_fraction = 0.1
    cfg.stress.byzantine_mode = "adaptive"
    assert any("rounds down to 0" in m for m in cfg.validate(emit=False, strict=False))


def test_fraction_without_a_mode_is_flagged():
    cfg = Config()
    cfg.stress.byzantine_fraction = 0.3
    cfg.stress.byzantine_mode = "none"
    assert any("no attack runs" in m for m in cfg.validate(emit=False, strict=False))


def test_adaptive_scale_above_one_std_is_flagged():
    cfg = Config()
    cfg.stress.byzantine_fraction = 0.2
    cfg.stress.byzantine_mode = "adaptive"
    cfg.stress.byzantine_scale = 3.0
    assert any("leaves the honest range" in m for m in cfg.validate(emit=False, strict=False))


@pytest.mark.parametrize("bad", [{"byzantine_fraction": 1.0}, {"comm_dropout": 1.5}, {"byzantine_mode": "nope"}])
def test_invalid_stress_configs_are_refused(bad):
    cfg = Config()
    for k, v in bad.items():
        setattr(cfg.stress, k, v)
    with pytest.raises(ConfigError):
        cfg.validate(emit=False)


# --------------------------------------------------------------------------- #
# The attack has to reach BOTH aggregation paths
# --------------------------------------------------------------------------- #


def test_plain_sum_is_attacked_too():
    """The defect this guards is subtle and would invert H6's headline.

    With ``aggregation: sum`` the per-edge gradients are never materialised --
    the loss path computes the whole sum at once -- so there is nowhere to inject
    an attack. If that path stayed on under an active adversary, the plain-sum
    arm would run *unattacked*, look robust, and refute "plain-sum collapses
    early" for a purely mechanical reason.
    """
    attacked = stress_config(byzantine_fraction=0.25, byzantine_mode="adaptive")
    attacked.distill.aggregation = "sum"
    algo = build_algorithm(attacked, seed=1, device="cpu")
    assert algo.byzantine, "no Byzantine agents were selected"
    assert not algo._uses_loss_path, (
        "plain-sum kept the loss path under an active adversary, so the sum arm "
        "would be silently unattacked"
    )

    clean = stress_config()
    clean.distill.aggregation = "sum"
    assert build_algorithm(clean, seed=1, device="cpu")._uses_loss_path, (
        "with no adversary the cheap loss path should still be used"
    )


@pytest.mark.parametrize("aggregation", ["sum", "trimmed_mean"])
@pytest.mark.parametrize("mode", ["sign_flip", "gaussian", "adaptive"])
def test_attacks_perturb_an_honest_agents_policy(aggregation, mode):
    """End-to-end: the attack must actually reach an honest agent's weights."""
    clean = stress_config()
    clean.distill.aggregation = aggregation
    baseline = train_policy(clean)

    attacked = stress_config(byzantine_fraction=0.25, byzantine_mode=mode)
    attacked.distill.aggregation = aggregation
    drift = (train_policy(attacked) - baseline).norm().item()
    assert drift > 1e-9, (
        f"{mode} under {aggregation} left an honest agent's policy unchanged, so "
        "this H6 cell would report a defence that was never tested"
    )


def test_byzantine_agents_are_excluded_from_the_honest_return():
    """H6's headline is the honest agents' return, not everyone's.

    Byzantine agents are sabotaging; averaging them into the reported number
    measures the attack's self-harm alongside the defence's success, and moves
    with `f` even when the defence is perfect.
    """
    cfg = stress_config(byzantine_fraction=0.25, byzantine_mode="adaptive")
    algo = build_algorithm(cfg, seed=1, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=derive_seed(1, "env"))
    algo.epoch = 0
    algo.on_epoch_start()
    stats = algo.collect_rollout(vec)

    assert "episode_return_mean_honest" in stats
    assert "episode_return_mean_byzantine" in stats
    assert stats["episode_return_mean_honest"] != stats["episode_return_mean_byzantine"]


def test_honest_return_is_present_in_the_baseline_cell_too():
    """The f=0 cell must carry the same field, or the sweep compares an
    honest-only number against an all-agent one along the f axis."""
    cfg = stress_config()
    algo = build_algorithm(cfg, seed=1, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=derive_seed(1, "env"))
    algo.epoch = 0
    algo.on_epoch_start()
    stats = algo.collect_rollout(vec)
    assert "episode_return_mean_honest" in stats
    assert stats["episode_return_mean_honest"] == pytest.approx(
        stats["episode_return_mean"], rel=1e-9
    ), "with no Byzantine agents the honest mean IS the mean"


# --------------------------------------------------------------------------- #
# The POLICY-level adversary (SPEC D19) -- the threat the gate keys on
# --------------------------------------------------------------------------- #


def policy_level_config(**stress) -> Config:
    stress.setdefault("byzantine_level", "policy")
    stress.setdefault("byzantine_mode", "none")
    stress.setdefault("byzantine_fraction", 0.25)
    stress.setdefault("byzantine_scale", 1.0)
    cfg = stress_config(**stress)
    cfg.distill.aggregation = "sum"  # the gate must be the only defence
    cfg.instrument.per_edge_detail = True
    return cfg


def edge_rows(cfg: Config, seed: int = 1) -> tuple[list, set[int]]:
    algo = build_algorithm(cfg, seed=seed, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=derive_seed(seed, "env"))
    algo.epoch = 0
    algo.on_epoch_start()
    algo.collect_rollout(vec)
    algo.update()
    return algo._edge_rows, set(algo.byzantine)


def test_policy_level_adversary_elevates_divergence_and_collapses_the_weight():
    """The pre-registered prediction, as a test.

    A policy-level Byzantine neighbour broadcasts a divergent snapshot, so
    ``D_ij`` on its edges must rise and ``w_ij`` must collapse relative to honest
    edges. If this fails, the gate does not work against the one adversary it was
    designed for, and no amount of the gradient-level H6 evidence would show it.

    The thresholds here are deliberately loose. Given the offset, the *sizes* of
    both effects follow from the attenuation formula and testing them tightly
    would only re-derive arithmetic; what is not guaranteed, and is what this
    guards, is that the corruption reaches ``D_ij`` at all and that the gate is
    wired to respond to it.
    """
    rows, byz = edge_rows(policy_level_config())
    assert byz, "no Byzantine agents were selected"
    assert rows, "per-edge detail was not logged, so the prediction is unmeasurable"

    lied = [r for r in rows if r["j"] in byz]
    honest = [r for r in rows if r["j"] not in byz]
    assert lied and honest

    d_lied = np.mean([r["D_ij"] for r in lied])
    d_honest = np.mean([r["D_ij"] for r in honest])
    w_lied = np.mean([r["w_ij"] for r in lied])
    w_honest = np.mean([r["w_ij"] for r in honest])

    assert d_lied > 5 * d_honest, (
        f"D_ij on Byzantine edges ({d_lied:.4g}) is not markedly above honest "
        f"({d_honest:.4g}); the corruption is not reaching the divergence"
    )
    assert w_lied < 0.5 * w_honest, (
        f"w_ij on Byzantine edges ({w_lied:.4g}) did not collapse relative to "
        f"honest ({w_honest:.4g}); the gate is not attenuating the lie"
    )


def test_gradient_level_adversary_leaves_the_divergence_alone():
    """The other half of the distinction, and the reason H6 came out flat.

    A gradient-level adversary corrupts the aggregation site and broadcasts an
    honest-looking snapshot, so ``D_ij`` is unchanged and the gate has nothing to
    key on. Asserting this keeps the two levels from quietly converging into one
    stressor -- if a future edit made the gradient attack touch the snapshot, the
    published "the gate is blind by construction" finding would be wrong.
    """
    cfg = stress_config(
        byzantine_level="gradient", byzantine_mode="adaptive", byzantine_fraction=0.25
    )
    cfg.instrument.per_edge_detail = True
    rows, byz = edge_rows(cfg)
    lied = np.mean([r["D_ij"] for r in rows if r["j"] in byz])
    honest = np.mean([r["D_ij"] for r in rows if r["j"] not in byz])
    assert lied == pytest.approx(honest, rel=0.5), (
        f"a gradient-level adversary moved D_ij ({lied:.4g} vs {honest:.4g}); it "
        "is supposed to be behaviourally invisible"
    )


def test_the_policy_level_adversary_keeps_its_own_actor_honest():
    """It is an adversary, not a broken agent.

    Only the broadcast snapshot carries the lie. The Byzantine agent's own actor,
    and its own view of itself (the nonstationarity signal, pi_i under a pooled
    trust batch), must be uncorrupted -- otherwise the arm measures an agent
    sabotaging itself, which is a different and much weaker experiment.
    """
    cfg = policy_level_config()
    algo = build_algorithm(cfg, seed=1, device="cpu")
    algo.epoch = 0
    algo.on_epoch_start()
    j = algo.byzantine[0]

    live = torch.cat([p.reshape(-1) for p in algo.agents[j].actor.parameters()])
    own = torch.cat([p.reshape(-1) for p in algo.snapshots[j].parameters()])
    sent = torch.cat([p.reshape(-1) for p in algo._broadcast[j].parameters()])

    assert torch.allclose(live, own), "the agent's own snapshot was corrupted"
    assert not torch.allclose(own, sent), "the broadcast snapshot carries no lie"

    honest_j = next(i for i in range(algo.n_agents) if i not in algo._byzantine_set)
    assert algo._broadcast[honest_j] is algo.snapshots[honest_j], (
        "an honest agent's broadcast was copied rather than shared, which would "
        "change its numerics for no reason"
    )


def test_the_lie_is_the_same_lie_every_epoch():
    """A direction redrawn per epoch is a flapping adversary whose successive
    lies partly cancel -- a weaker attack wearing the same name."""
    cfg = policy_level_config()
    algo = build_algorithm(cfg, seed=1, device="cpu")
    j = algo.byzantine[0]
    offsets = []
    for epoch in range(3):
        algo.epoch = epoch
        algo.on_epoch_start()
        own = torch.cat([p.reshape(-1) for p in algo.snapshots[j].parameters()])
        sent = torch.cat([p.reshape(-1) for p in algo._broadcast[j].parameters()])
        offsets.append(sent - own)
    for later in offsets[1:]:
        assert torch.allclose(offsets[0], later)


def test_policy_level_keeps_the_cheap_loss_path():
    """The per-edge path is forced only for the gradient-level adversary, which
    has nowhere else to be injected. Forcing it here would change the plain-sum
    arm's numerics for an attack that never touches a gradient."""
    assert build_algorithm(policy_level_config(), seed=1, device="cpu")._uses_loss_path


def test_a_policy_level_arm_needs_no_gradient_mode():
    """`byzantine_mode` names a gradient attack. Requiring one at the policy
    level would make every such config declare an attack it does not run."""
    cfg = policy_level_config()
    assert cfg.stress.byzantine_agents_enabled
    assert build_algorithm(cfg, seed=1, device="cpu").byzantine

    noisy = policy_level_config(byzantine_mode="adaptive")
    warnings = noisy.stress.validate(noisy.env.n_agents)
    assert any("NOT applied at the policy level" in w for w in warnings), (
        "a policy-level config that also names a gradient attack must say the "
        "gradient attack does not run"
    )


def test_zero_offset_policy_adversary_is_flagged_as_the_baseline():
    cfg = policy_level_config(byzantine_scale=0.0)
    assert any(
        "honest and this is the f=0 baseline" in w
        for w in cfg.stress.validate(cfg.env.n_agents)
    )


# --------------------------------------------------------------------------- #
# Communication dropout
# --------------------------------------------------------------------------- #


def test_dropout_removes_edges_and_is_asymmetric():
    cfg = stress_config(comm_dropout=0.5)
    algo = build_algorithm(cfg, seed=3, device="cpu")
    algo.epoch = 0
    algo.on_epoch_start()

    full = sum(len(n) for n in algo.neighbors)
    delivered = sum(len(n) for n in algo._epoch_neighbors)
    assert 0 < delivered < full, f"dropout delivered {delivered} of {full} edges"

    asymmetric = any(
        (j in algo._epoch_neighbors[i]) != (i in algo._epoch_neighbors[j])
        for i in range(algo.n_agents)
        for j in algo.neighbors[i]
    )
    assert asymmetric, (
        "every drop was symmetric, which is a sparser graph both endpoints agree "
        "on -- not link failure"
    )


def test_dropout_redraws_each_epoch():
    cfg = stress_config(comm_dropout=0.5)
    algo = build_algorithm(cfg, seed=3, device="cpu")
    seen = []
    for epoch in range(4):
        algo.epoch = epoch
        algo.on_epoch_start()
        seen.append([list(n) for n in algo._epoch_neighbors])
    assert any(seen[0] != s for s in seen[1:]), "dropout froze after the first epoch"


def test_dropout_reduces_reported_communication_volume():
    """SPEC Sec. 7 asks for communication volume; counting undelivered
    snapshots would overstate the bandwidth a dropout run actually used."""
    quiet = stress_config()
    lossy = stress_config(comm_dropout=0.5)
    a, b = (build_algorithm(c, seed=3, device="cpu") for c in (quiet, lossy))
    for algo in (a, b):
        algo.epoch = 0
        algo.on_epoch_start()
    assert b.epoch_snapshot_bytes < a.epoch_snapshot_bytes


def test_zero_dropout_keeps_every_edge():
    cfg = stress_config()
    algo = build_algorithm(cfg, seed=3, device="cpu")
    algo.epoch = 0
    algo.on_epoch_start()
    assert [list(n) for n in algo._epoch_neighbors] == [list(n) for n in algo.neighbors]


# --------------------------------------------------------------------------- #
# Motor failures
# --------------------------------------------------------------------------- #


def test_motor_failure_zeroes_the_agents_actions():
    cfg = Config()
    cfg.env.n_agents = 6
    cfg.env.n_targets = 2
    cfg.env.capture_group_size = 2
    cfg.env.max_cycles = 10
    cfg.validate(emit=False)

    failed = [1, 4]
    healthy = SyncVectorEnv(cfg.env, 2, seed=0)
    broken = SyncVectorEnv(cfg.env, 2, seed=0, motor_failures=failed)
    healthy.reset(seed=0)
    broken.reset(seed=0)

    actions = np.ones((2, cfg.env.n_agents, cfg.env.act_dim), dtype=np.float32)
    for _ in range(5):
        healthy.step(actions)
        broken.step(actions)

    hpos = healthy.envs[0].agent_pos
    bpos = broken.envs[0].agent_pos
    for i in failed:
        assert not np.allclose(hpos[i], bpos[i]), f"agent {i} moved as if healthy"
    assert broken.motor_failures == failed


# --------------------------------------------------------------------------- #
# H6's headline is the HONEST agents' return -- at EVALUATION time too
# --------------------------------------------------------------------------- #


def test_evaluation_records_the_honest_return_not_only_the_team_mean(tmp_path):
    """The plumbing existed at both ends and was never connected in the middle.

    The environment has always exposed per-agent returns for this (env.py, the
    "episode"/"agent_returns" key, with a comment saying so) and the ROLLOUT
    metrics have always split them. The EVALUATION path dropped them -- so
    `final_return`, the number that reaches every manifest, report and curve,
    was the all-agent mean, and `h6_curve.py` labelled that axis "final return
    (honest agents)".
    """
    from dtropd.train import train

    cfg = stress_config(byzantine_fraction=0.25, byzantine_mode="adaptive")
    cfg.train.total_epochs = 1
    cfg.train.eval_episodes = 4
    result = train(cfg, seed=1, out_dir=tmp_path / "run", progress=False)

    summary = result.summary()
    assert "final_return_honest" in summary
    assert summary["final_return_honest"] != summary["final_return"], (
        "the honest mean equals the team mean, so the split did not happen"
    )
    assert result.final_eval["return_mean_byzantine"] != pytest.approx(
        result.final_eval["return_mean_honest"]
    )


def test_the_honest_return_is_present_at_f0_so_the_axis_is_comparable(tmp_path):
    """With no Byzantine agents the honest mean IS the mean. If the key were
    absent at f=0, a curve over f would compare an honest-only number against an
    all-agent one at its own baseline."""
    from dtropd.train import train

    cfg = stress_config()
    cfg.train.total_epochs = 1
    cfg.train.eval_episodes = 4
    summary = train(cfg, seed=1, out_dir=tmp_path / "run", progress=False).summary()
    assert summary["final_return_honest"] == pytest.approx(summary["final_return"], rel=1e-12)
