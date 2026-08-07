"""Phase 5: the instrumentation, the calibration artifact, and the analysis.

The Phase 5 acceptance bar is that ONE run's metrics file supports every H1-H7
measurement. These tests hold each half of that: the measurements are actually
produced, and the analysis can actually consume them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dtropd.analysis import (
    HYPOTHESES,
    analyze_run,
    bootstrap_ci,
    consensus_pull_cost,
    iqm,
)
from dtropd.calibration import (
    CalibrationError,
    CalibrationRecord,
    assert_shared_calibration,
    env_fingerprint,
)
from dtropd.config import Config, ConfigError
from dtropd.env import SyncVectorEnv
from dtropd.instrumentation import (
    SignFlipTracker,
    bucket_label,
    graph_distances,
    quantiles,
)
from dtropd.probes import decay_probe
from dtropd.train import train
from dtropd.utils import read_metrics


def instrumented_config(algo: str = "dtropd", n_agents: int = 5) -> Config:
    cfg = Config()
    cfg.env.n_agents = n_agents
    cfg.env.n_targets = 1
    cfg.env.capture_group_size = 2
    cfg.env.max_cycles = 8
    cfg.env.heterogeneous = True
    cfg.train.algo = algo
    cfg.train.num_envs = 2
    cfg.train.total_epochs = 6
    cfg.train.eval_interval = 0
    cfg.train.eval_episodes = 2
    cfg.train.device = "cpu"
    cfg.ppo.inner_epochs = 2
    cfg.ppo.num_minibatches = 2
    cfg.network.hidden_sizes = [16, 16]
    cfg.distill.enabled = algo in ("dtropd", "naive_distill")
    cfg.distill.beta_max = 0.2
    cfg.distill.warmup_epochs = 1
    cfg.distill.delta_base = 0.01
    cfg.distill.gamma_attenuation = 20.0
    cfg.instrument.decay_probe_interval = 3
    cfg.instrument.decay_probe_agents = 2
    cfg.instrument.decay_probe_episodes = 2
    cfg.validate(emit=False)
    return cfg


@pytest.fixture(scope="module")
def dtropd_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("phase5") / "dtropd"
    cfg = instrumented_config("dtropd")
    result = train(cfg, seed=0, out_dir=out, progress=False)
    return result.run_dir


@pytest.fixture(scope="module")
def ippo_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("phase5c") / "ippo"
    result = train(instrumented_config("ippo"), seed=0, out_dir=out, progress=False)
    return result.run_dir


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def test_quantiles_reports_a_distribution_not_just_a_mean():
    out = quantiles(list(range(101)), [0.05, 0.5, 0.95], "x/")
    assert out["x/p50"] == pytest.approx(50.0)
    assert out["x/p05"] == pytest.approx(5.0)
    assert out["x/p95"] == pytest.approx(95.0)
    assert out["x/count"] == 101
    assert quantiles([], [0.5], "y/") == {"y/count": 0}


def test_bucket_label_covers_the_whole_line():
    edges = [0.0, 0.01, 0.1]
    assert bucket_label(0.005, edges) == "[0,0.01)"
    assert bucket_label(0.05, edges) == "[0.01,0.1)"
    assert bucket_label(5.0, edges) == "[>=0.1)"
    assert bucket_label(-1.0, edges) == "[<0)"


def test_graph_distances_on_a_ring():
    ring = [[4, 1], [0, 2], [1, 3], [2, 4], [3, 0]]
    assert graph_distances(ring, 0) == [0, 1, 2, 2, 1]
    disconnected = [[1], [0], []]
    assert graph_distances(disconnected, 0) == [0, 1, -1]


def test_sign_flip_tracker_detects_reversals():
    tracker = SignFlipTracker([0.0, 0.01, 0.1])
    g = torch.tensor([1.0, 0.0])
    assert tracker.observe(0, 1, g, gap=0.005) is None  # first epoch: no baseline
    assert tracker.observe(0, 1, -g, gap=0.005) is True  # exact reversal
    assert tracker.observe(0, 1, -g, gap=0.005) is False  # unchanged
    summary = tracker.summary([0.5])
    assert summary["h1/sign_flips_total"] == 1
    assert summary["h1/edges_compared"] == 2
    assert summary["h1/sign_flips_by_bucket"]["[0,0.01)"]["flips"] == 1


# --------------------------------------------------------------------------- #
# Calibration artifacts (SPEC Sec. 11 / D6)
# --------------------------------------------------------------------------- #


def _record(**overrides):
    base = dict(
        window_epochs=8,
        pooled_epochs=[6, 7],
        reward_spec_version="v1",
        git_commit="abc123",
        env_fingerprint="fingerprint",
    )
    base.update(overrides)
    return CalibrationRecord.from_samples(np.linspace(0.01, 0.2, 200), **base)


def test_calibration_record_captures_the_rule_and_provenance():
    rec = _record()
    assert rec.window_epochs == 8
    assert rec.pooled_epochs == [6, 7]
    assert rec.pool_size == 2
    assert rec.edge_count == 200
    assert set(rec.quantiles) >= {"p05", "p50", "p95"}
    assert rec.reward_spec_version == "v1"
    assert rec.git_commit == "abc123"
    assert "final 2 epoch(s) of a 8-epoch" in rec.quantile_definition
    assert len(rec.calibration_id) == 16


def test_calibration_roundtrip_and_tamper_detection(tmp_path):
    rec = _record()
    path = rec.save(tmp_path / "calibration.json")
    assert CalibrationRecord.load(path).calibration_id == rec.calibration_id

    tampered = json.loads(path.read_text())
    tampered["delta_base"] = 0.99  # hand-edit the number, keep the stored id
    path.write_text(json.dumps(tampered))
    with pytest.raises(CalibrationError, match="has been edited"):
        CalibrationRecord.load(path)


def test_missing_artifact_says_how_to_make_one(tmp_path):
    with pytest.raises(CalibrationError, match="run.py calibrate"):
        CalibrationRecord.load(tmp_path / "nope.json")


def test_shared_calibration_accepts_one_artifact_and_rejects_two():
    shared = _record()
    assert assert_shared_calibration(
        [("dtropd", shared), ("naive_distill", shared)]
    ) == shared.calibration_id

    separate = CalibrationRecord.from_dict(
        {**shared.to_dict(), "created_at": "1999-01-01T00:00:00+0000"}
    )
    with pytest.raises(CalibrationError, match="calibrated separately"):
        assert_shared_calibration([("dtropd", shared), ("naive_distill", separate)])


def test_shared_calibration_rejects_an_undeclared_arm():
    """D6: arms with no radius still declare which calibration the sweep used."""
    shared = _record()
    with pytest.raises(CalibrationError, match="declare no calibration artifact"):
        assert_shared_calibration([("dtropd", shared), ("naive_distill", None)])


def test_calibration_file_is_pinned_not_remeasured(tmp_path):
    cfg = instrumented_config("dtropd")
    rec = _record(env_fingerprint=env_fingerprint(cfg.env))
    path = rec.save(tmp_path / "cal.json")

    cfg.distill.calibration = "file"
    cfg.distill.calibration_file = str(path)
    cfg.validate(emit=False)
    from dtropd.algorithms import build_algorithm

    algo = build_algorithm(cfg, seed=0, device="cpu")
    assert algo.delta_base == pytest.approx(rec.delta_base)
    assert algo.gamma_attenuation == pytest.approx(rec.gamma_attenuation)
    assert algo._calibrated  # nothing left to measure
    assert algo.calibration_record.calibration_id == rec.calibration_id


def test_calibration_file_rejects_a_different_environment(tmp_path):
    cfg = instrumented_config("dtropd")
    path = _record(env_fingerprint="a-different-env").save(tmp_path / "cal.json")
    cfg.distill.calibration = "file"
    cfg.distill.calibration_file = str(path)
    cfg.validate(emit=False)
    from dtropd.algorithms import build_algorithm

    with pytest.raises(CalibrationError, match="different environment"):
        build_algorithm(cfg, seed=0, device="cpu")


def test_naive_distill_declares_a_calibration_without_owning_a_radius(tmp_path):
    """D6 x D4: declaration for auditability, but still no radius."""
    cfg = instrumented_config("naive_distill")
    rec = _record(env_fingerprint=env_fingerprint(cfg.env))
    cfg.distill.calibration = "file"
    cfg.distill.calibration_file = str(rec.save(tmp_path / "cal.json"))
    cfg.validate(emit=False)
    from dtropd.algorithms import build_algorithm

    algo = build_algorithm(cfg, seed=0, device="cpu")
    assert algo.calibration_record.calibration_id == rec.calibration_id
    assert not hasattr(algo, "delta_base")
    assert not hasattr(algo, "gamma_attenuation")


def test_calibration_config_validation():
    cfg = Config()
    cfg.distill.calibration = "file"
    with pytest.raises(ConfigError, match="calibration_file"):
        cfg.validate(emit=False)
    cfg = Config()
    cfg.distill.calibration_pool_epochs = 99
    with pytest.raises(ConfigError, match="calibration_pool_epochs"):
        cfg.validate(emit=False)


# --------------------------------------------------------------------------- #
# The decay probe (H5's Assumption-2 measurement)
# --------------------------------------------------------------------------- #


def test_decay_probe_reports_by_hop_and_restores_the_policies():
    from dtropd.algorithms import build_algorithm

    cfg = instrumented_config("dtropd", n_agents=5)
    algo = build_algorithm(cfg, seed=0, device="cpu")
    before = [p.detach().clone() for ac in algo.agents for p in ac.actor.parameters()]

    result = decay_probe(algo, cfg.env, n_perturbed=2, episodes=2, scale=0.2, seed=0)

    after = [p.detach() for ac in algo.agents for p in ac.actor.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(before, after)), (
        "the probe left the training policies perturbed"
    )
    assert len(result["perturbed_agents"]) == 2
    assert result["pairs"], "no (observer, perturbed) pairs recorded"
    assert all(p["hops"] >= 1 for p in result["pairs"])
    assert set(result["by_hop"]) == {str(h) for h in {p["hops"] for p in result["pairs"]}}
    for stats in result["by_hop"].values():
        assert stats["count"] > 0
        assert np.isfinite(stats["mean"])


# --------------------------------------------------------------------------- #
# What one run's metrics file contains
# --------------------------------------------------------------------------- #


def test_metrics_file_carries_the_schema_and_reward_versions(dtropd_run):
    meta = json.loads((Path(dtropd_run) / "metadata.json").read_text())
    assert meta["metrics_schema_version"] == "v1"
    assert meta["reward_spec_version"] == "v1"


def test_metrics_file_carries_every_instrumented_quantity(dtropd_run):
    records = read_metrics(Path(dtropd_run) / "metrics.jsonl")
    seen: set[str] = set()
    for r in records:
        seen.update(r)

    required = [
        # improvement trust region
        "update_kl", "kl_early_stop_eta", "early_stopped",
        # radius / gate distributions, not just means
        "delta_mean", "nonstationarity_mean",
        "social_divergence_p05", "social_divergence_p50", "social_divergence_p95",
        "trust_weight_mean", "trust_weight_min",
        # critic / entropy
        "explained_variance", "entropy", "mean_sigma",
        # H1
        "h1/sign_flips_by_bucket", "h1/sign_flip_rate", "social_grad_norm_mean",
        # H2
        "edges",
        # D9
        "consensus_pull/social_grad_fraction_mean",
        "consensus_pull/policy_dispersion_p50",
        # per-zone
        "zone_return/normal", "zone_collisions/normal",
        # communication and compute
        "snapshot_bytes", "snapshot_bytes_total",
        "compute/env_steps", "compute/gradient_steps", "time/elapsed",
    ]
    missing = [k for k in required if k not in seen]
    assert not missing, f"metrics file is missing {missing}"

    # H4/H5 live in the per-agent rows.
    assert any(
        a.get("surrogate_gain") is not None
        for r in records for a in (r.get("per_agent") or [])
    )
    # H5's probe fires at its interval and on the final epoch.
    assert any("decay_probe" in r for r in records)


def test_per_edge_rows_carry_both_sigmas_and_the_boundary_gap(dtropd_run):
    records = read_metrics(Path(dtropd_run) / "metrics.jsonl")
    rows = [e for r in records for e in (r.get("edges") or [])]
    assert rows
    for row in rows[:20]:
        assert {"i", "j", "grad_norm", "sigma_i", "sigma_j", "D_ij", "w_ij", "boundary_gap"} <= set(row)
        assert row["boundary_gap"] >= 0.0


def test_ippo_control_also_reports_dispersion(ippo_run):
    """D9 needs an uncontaminated control, so IPPO logs dispersion too."""
    records = read_metrics(Path(ippo_run) / "metrics.jsonl")
    assert any("consensus_pull/policy_dispersion_p50" in r for r in records)
    # ...but IPPO exchanges nothing and has no edges.
    assert all("edges" not in r for r in records)
    assert all(r.get("snapshot_bytes", 0) == 0 for r in records)


# --------------------------------------------------------------------------- #
# Phase 5 acceptance: every hypothesis computable from one file
# --------------------------------------------------------------------------- #


def test_every_hypothesis_is_computable_from_a_single_run(dtropd_run):
    records = read_metrics(Path(dtropd_run) / "metrics.jsonl")
    metadata = json.loads((Path(dtropd_run) / "metadata.json").read_text())
    results = analyze_run(records, metadata)

    assert set(results) == set(HYPOTHESES)
    uncomputable = {k: v.missing_fields for k, v in results.items() if not v.computable}
    assert not uncomputable, f"not measurable from one run: {uncomputable}"


def test_hypothesis_values_are_populated_not_empty(dtropd_run):
    records = read_metrics(Path(dtropd_run) / "metrics.jsonl")
    metadata = json.loads((Path(dtropd_run) / "metadata.json").read_text())
    r = analyze_run(records, metadata)

    assert r["H1"].value["edges_compared"] > 0
    assert r["H1"].value["flip_rate_by_bucket"]
    assert r["H2"].value["trajectory"]
    assert np.isfinite(r["H3"].value["delta_last"])
    assert r["H4"].value["rounds"] > 0
    assert "decay_probe" in r["H5"].value
    assert r["H5"].value["decay_probe"]["hops"]
    assert r["H7"].value["environment_arm"] == "treatment"
    assert r["H7"].value["per_zone_final"]


def test_analysis_reports_missing_fields_rather_than_crashing():
    """A run with instruments off must produce a precise gap list, not an error."""
    bare = [{"epoch": 0, "episode_return_mean": -1.0}, {"epoch": 1, "episode_return_mean": -0.5}]
    results = analyze_run(bare, {})
    assert not results["H1"].computable
    assert "h1/sign_flips_by_bucket" in results["H1"].missing_fields
    assert not results["H2"].computable
    assert not results["H4"].computable


def test_h7_marks_the_homogeneous_arm_as_the_control(tmp_path):
    """D8: the homogeneous environment is a control condition, and is labelled."""
    cfg = instrumented_config("dtropd")
    cfg.env.heterogeneous = False
    cfg.train.total_epochs = 3
    cfg.validate(emit=False)
    result = train(cfg, seed=0, out_dir=tmp_path / "homo", progress=False)
    records = read_metrics(result.run_dir / "metrics.jsonl")
    metadata = json.loads((result.run_dir / "metadata.json").read_text())
    assert analyze_run(records, metadata)["H7"].value["environment_arm"] == "control"


# --------------------------------------------------------------------------- #
# D9 consensus-pull cost
# --------------------------------------------------------------------------- #


def test_consensus_pull_cost_pairs_seeds_when_it_can():
    control = [10.0, 12.0, 11.0]
    distiller = [8.0, 9.0, 9.5]
    paired = consensus_pull_cost(control, distiller)
    assert paired["paired"] is True
    assert paired["cost"] == pytest.approx(np.mean(np.array(control) - np.array(distiller)))
    assert paired["ci"]["lo"] < paired["cost"] < paired["ci"]["hi"]

    unpaired = consensus_pull_cost(control, distiller[:2])
    assert unpaired["paired"] is False


def test_bootstrap_ci_and_iqm():
    values = list(range(100))
    ci = bootstrap_ci(values, seed=0)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert ci["n"] == 100
    assert iqm(values) == pytest.approx(np.mean(values[25:75]), rel=0.05)
    assert bootstrap_ci([])["n"] == 0


# --------------------------------------------------------------------------- #
# D7 / D8: the pre-registered decision rules, as executable code
# --------------------------------------------------------------------------- #


def test_d7_rule_reports_stability_when_the_verdict_holds():
    from dtropd.analysis import D7_GRID_POINTS, d7_sensitivity_verdict

    out = d7_sensitivity_verdict({d: True for d in D7_GRID_POINTS})
    assert out["stable"] is True
    assert out["complete"] is True
    assert out["report_as_limitation"] is False
    assert "stable across the pre-registered" in out["conclusion"]


def test_d7_rule_calls_a_flip_a_finding_not_a_reason_to_reselect():
    from dtropd.analysis import D7_GRID_POINTS, d7_sensitivity_verdict

    flipped = dict.fromkeys(D7_GRID_POINTS, True)
    flipped[D7_GRID_POINTS[-1]] = False
    out = d7_sensitivity_verdict(flipped)
    assert out["stable"] is False
    assert out["report_as_limitation"] is True
    assert "CALIBRATION-SENSITIVE" in out["conclusion"]
    assert "do not select the window" in out["conclusion"]


def test_d7_rule_flags_an_incomplete_grid():
    from dtropd.analysis import d7_sensitivity_verdict

    out = d7_sensitivity_verdict({"w8": True})
    assert out["complete"] is False
    assert out["missing_arms"] == ["w6", "w10", "x2"]


def test_d7_grid_constant_matches_the_shipped_configs():
    """The constant went stale once and nothing noticed.

    `D7_DELTA_BASE_GRID` read `(0.022, 0.026, 0.049)` -- a placeholder from
    before the grid was measured on the real H7 environment, never updated when
    D17 signed off the four measured points. Nothing consumed it on a real
    sweep, so the staleness was invisible. This is the guard: the numbers in the
    analysis module and the numbers the sweeps will actually run must agree, or
    the D7 verdict is computed against a grid nobody ran.
    """
    from pathlib import Path as _Path

    from dtropd.analysis import D7_DELTA_BASE, D7_GRID_POINTS
    from dtropd.sweep import load_sweep_config

    configs = _Path(__file__).resolve().parent.parent / "experiments" / "configs"
    for point in D7_GRID_POINTS:
        plan = load_sweep_config(configs / f"h7_{point}.yaml")
        for arm in plan.arms:
            if arm.algo != "dtropd":
                continue
            expected = D7_DELTA_BASE[arm.condition][point]
            actual = arm.config(plan.base).distill.delta_base
            assert actual == pytest.approx(expected, rel=1e-12), (
                f"h7_{point}.yaml arm {arm.name!r} runs delta_base={actual!r} but "
                f"D7_DELTA_BASE says {expected!r}"
            )


def test_the_interaction_carries_its_own_confidence_interval():
    """D8's claim is the interaction, so the interaction needs a CI.

    Without one the test reduces to eyeballing two separate CIs, which is not a
    test of their difference. The pairing must also be by SEED, not by position:
    a resumed or reordered sweep would otherwise pair one seed's treatment
    against another's control and report it with a confident interval.
    """
    from dtropd.analysis import h7_interaction

    het = {"dtropd": {0: 10.0, 1: 12.0, 2: 11.0}, "ippo": {0: 8.0, 1: 9.0, 2: 9.0}}
    hom = {"dtropd": {0: 5.0, 1: 6.0, 2: 5.5}, "ippo": {0: 6.0, 1: 7.0, 2: 6.5}}
    out = h7_interaction(het, hom)
    assert out["interaction_n_pairs"] == 3
    assert out["interaction_ci"]["lo"] > 0.0
    assert out["predicted_sign_flip_observed"] is True

    # Shuffling the seed order must not move anything: the pairing is by key.
    shuffled = {a: dict(reversed(list(v.items()))) for a, v in het.items()}
    assert h7_interaction(shuffled, hom)["interaction_ci"] == out["interaction_ci"]

    # A list has no seed labels, so no paired interaction CI is claimed.
    listed = {a: [v[k] for k in sorted(v)] for a, v in het.items()}
    assert h7_interaction(listed, hom)["interaction_ci"] == {}


def test_h7_interaction_detects_the_predicted_sign_flip():
    from dtropd.analysis import h7_interaction

    het = {"dtropd": [10.0, 11.0, 12.0], "ippo": [8.0, 9.0, 9.0]}
    hom = {"dtropd": [5.0, 5.5, 6.0], "ippo": [7.0, 7.5, 8.0]}
    out = h7_interaction(het, hom)
    assert out["heterogeneous_difference"]["mean"] > 0
    assert out["homogeneous_difference"]["mean"] < 0
    assert out["predicted_sign_flip_observed"] is True
    assert out["interaction"] > 0

    # Same direction in both environments is NOT the predicted interaction.
    no_flip = h7_interaction(het, {"dtropd": [9.0, 10.0, 11.0], "ippo": [8.0, 9.0, 9.0]})
    assert no_flip["predicted_sign_flip_observed"] is False
