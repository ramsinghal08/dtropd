"""Phase 6: the experiment harness, and the D11/D13 guards it enforces.

The harness's job is to make a comparison hard to invalidate by accident. Each
test here corresponds to one way a comparison could quietly become meaningless.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dtropd.algorithms import build_graph
from dtropd.calibration import (
    CalibrationRecord,
    d7_grid_from_records,
    env_fingerprint,
)
from dtropd.config import Config, ConfigError, EnvConfig
from dtropd.env import CooperativeCaptureEnv
from dtropd.probes import MIN_HOP_LEVELS_FOR_A_FIT, fit_decay
from dtropd.sweep import (
    SweepArm,
    SweepError,
    SweepPlan,
    build_plan,
    check_calibration,
    check_paired_seeds,
    check_reward_spec_versions,
    collect_final_returns,
    load_sweep_config,
    run_sweep,
    sweep_report,
)

CONFIGS = Path(__file__).resolve().parent.parent / "experiments" / "configs"


def smoke_plan(tmp_path: Path, seeds=(0, 1)) -> SweepPlan:
    plan = load_sweep_config(CONFIGS / "sweep_smoke.yaml")
    plan.seeds = list(seeds)
    plan.out_dir = tmp_path / "sweep"
    plan.auto_calibrate = True
    for arm in plan.arms:
        arm.overrides.pop("distill.calibration", None)
        arm.overrides.pop("distill.calibration_file", None)
    return plan


# --------------------------------------------------------------------------- #
# Shipped sweep configs
# --------------------------------------------------------------------------- #


def _sweep_config_paths() -> list[Path]:
    """Every shipped sweep config, discovered rather than listed.

    A hardcoded list means a new sweep config is unvalidated until someone
    remembers to add it -- and the configs are exactly the artifacts where a
    silent structural error costs a multi-hour run.
    """
    out = []
    for path in sorted(CONFIGS.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        if isinstance(raw, dict) and "arms" in raw:
            out.append(path)
    return out


def test_sweep_config_discovery_finds_the_known_ones():
    """Guards the guard: if discovery silently matched nothing, every config
    below would 'pass' by not being collected."""
    found = {p.stem for p in _sweep_config_paths()}
    assert {"sweep_smoke", "h7", "h5_locality", "h6_byzantine"} <= found, found


@pytest.mark.parametrize("path", _sweep_config_paths(), ids=lambda p: p.stem)
def test_shipped_sweep_configs_load_and_pass_structural_checks(path):
    plan = load_sweep_config(path)
    assert plan.arms and plan.seeds
    check_reward_spec_versions(plan)
    check_paired_seeds(plan)


def test_h7_spans_both_environments_with_a_control():
    """SPEC Sec. 10 / D8: the homogeneous arm is a control, not a throwaway."""
    plan = load_sweep_config(CONFIGS / "h7.yaml")
    conditions = {arm.condition for arm in plan.arms}
    assert conditions == {"heterogeneous", "homogeneous"}
    for condition in conditions:
        algos = {a.algo for a in plan.arms if a.condition == condition}
        assert {"ippo", "dtropd"} <= algos, condition
    assert len(plan.seeds) == 10  # SPEC Sec. 7


def test_h7_is_single_reward_version_and_h5_is_cross_version():
    h7 = check_reward_spec_versions(load_sweep_config(CONFIGS / "h7.yaml"))
    assert h7["algorithm_arms_version"] == ["v1"]
    assert h7["cross_version"] is False

    h5 = check_reward_spec_versions(load_sweep_config(CONFIGS / "h5_locality.yaml"))
    assert h5["algorithm_arms_version"] == []  # no algorithm arms at all
    assert h5["environment_property_versions"] == ["v1", "v1-local"]
    assert h5["cross_version"] is True


# --------------------------------------------------------------------------- #
# D13: algorithm comparisons may not span reward spec versions
# --------------------------------------------------------------------------- #


def test_algorithm_comparison_spanning_reward_versions_is_refused():
    plan = load_sweep_config(CONFIGS / "sweep_smoke.yaml")
    plan.arms.append(
        SweepArm(
            name="dtropd_local",
            algo="dtropd",
            condition="heterogeneous",
            overrides={"env.reward_spec_version": "v1-local"},
            purpose="algorithm",  # the mistake D13 exists to catch
        )
    )
    with pytest.raises(SweepError, match="across reward spec versions"):
        check_reward_spec_versions(plan)


def test_environment_property_arms_may_span_versions():
    """The D13 exception: the environments are the subject, and it is labelled."""
    plan = load_sweep_config(CONFIGS / "h5_locality.yaml")
    result = check_reward_spec_versions(plan)
    assert result["passed"] and result["cross_version"]


@pytest.mark.slow
def test_parallel_workers_do_not_change_the_result(tmp_path):
    """Parallelism must be a scheduling change and nothing more.

    Every cell is seeded from its own (arm, seed) and writes its own directory,
    so running them concurrently cannot alter a curve. This is worth pinning
    because the alternative -- a sweep whose numbers depend on the worker count
    -- would be undetectable by eye and would invalidate any comparison run at a
    different parallelism.
    """
    sequential = run_sweep(smoke_plan(tmp_path / "seq", seeds=(0, 1)), progress=False)
    parallel = run_sweep(
        smoke_plan(tmp_path / "par", seeds=(0, 1)), progress=False, workers=2
    )

    def cells(manifest):
        return {
            (r["arm"], r["seed"]): r.get("final_return")
            for r in manifest["runs"]
        }

    seq, par = cells(sequential), cells(parallel)
    assert seq.keys() == par.keys()
    for key, value in seq.items():
        assert par[key] == pytest.approx(value, rel=1e-9), (
            f"{key} differs between 1 and 2 workers: {value} vs {par[key]}"
        )
    assert [r["arm"] for r in sequential["runs"]] == [r["arm"] for r in parallel["runs"]], (
        "the manifest must stay in plan order regardless of completion order"
    )


def _scale_points(name: str) -> list[int]:
    plan = load_sweep_config(CONFIGS / f"{name}.yaml")
    return sorted(a.config(plan.base).env.n_agents for a in plan.arms)


def test_scale_sweep_caps_are_asymmetric_as_d15_requires():
    """SPEC Sec. 13 / D15: H4 stops at 24, H5 keeps 48, and that is the point.

    A single cap in either direction is the failure mode: capping both at 24
    throws away the point that tests H5's flatness prediction hardest, and
    running both to 48 spends roughly four times N=24 on a complete graph to add
    one more point to a line H4 can already fit.
    """
    h4, h5 = _scale_points("h4_scale"), _scale_points("h5_scale")
    assert h4 == [4, 8, 12, 24], h4
    assert h5 == [4, 8, 12, 24, 48], h5
    assert max(h5) > max(h4), (
        "the caps have become symmetric; D15's asymmetry is deliberate and its "
        "reason is the topology's edge count, not convenience"
    )


def test_scale_sweep_topologies_match_their_hypotheses():
    """H4 needs dense coupling and H5 needs fixed degree; swapping them would
    invert what each sweep measures while still running to completion."""
    h4 = load_sweep_config(CONFIGS / "h4_scale.yaml")
    h5 = load_sweep_config(CONFIGS / "h5_scale.yaml")
    assert h4.base.graph.topology == "complete", "H4 specifies densely coupled"
    assert h5.base.graph.topology == "fixed_degree", "H5 specifies fixed-degree"

    # Fixed degree means the neighbourhood does not grow with N, or "per capita"
    # would be measuring coupling rather than size.
    degrees = {a.config(h5.base).graph.degree for a in h5.arms if a.config(h5.base).env.n_agents >= 8}
    assert degrees == {4}, f"H5's degree varies with N: {degrees}"


def test_h6_sweep_covers_the_full_fraction_axis_for_both_aggregators():
    """H6's claim is a comparison of two curves over f, so both need the axis."""
    plan = load_sweep_config(CONFIGS / "h6_byzantine.yaml")
    by_condition: dict[str, set[float]] = {}
    for arm in plan.arms:
        cfg = arm.config(plan.base)
        by_condition.setdefault(arm.condition, set()).add(cfg.stress.byzantine_fraction)

    for condition in ("adaptive_trimmed_mean", "adaptive_sum"):
        assert by_condition[condition] == {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}, (
            f"{condition} does not span the f axis SPEC H6 names"
        )


def test_h6_includes_the_adaptive_adversary_not_only_loud_ones():
    """Without `adaptive`, H6 would be confirmed by an adversary that defeats
    itself: trimming removes a sign-flip on magnitude alone."""
    plan = load_sweep_config(CONFIGS / "h6_byzantine.yaml")
    modes = {arm.config(plan.base).stress.byzantine_mode for arm in plan.arms}
    assert "adaptive" in modes, modes
    assert "sign_flip" in modes, "the loud positive control is missing"

    scales = {
        arm.config(plan.base).stress.byzantine_scale
        for arm in plan.arms
        if arm.config(plan.base).stress.byzantine_mode == "adaptive"
    }
    assert all(s <= 1.0 for s in scales), (
        f"adaptive scale {scales} exceeds one honest sd, which leaves the honest "
        "range and stops being an adaptive attack"
    )


def test_h6_does_not_tune_the_trim_fraction_to_the_byzantine_fraction():
    """A trim fitted per-f assumes the defender knows f, which is exactly the
    knowledge the defence is supposed not to need. It would also be a tuning
    asymmetry: the robust arm tuned, the plain-sum arm not (SPEC Sec. 9 rule 5).
    """
    plan = load_sweep_config(CONFIGS / "h6_byzantine.yaml")
    trims = {arm.config(plan.base).distill.trimmed_mean_fraction for arm in plan.arms}
    assert len(trims) == 1, f"trimmed_mean_fraction varies across arms: {trims}"


def _manifest(*runs: dict) -> dict:
    """A manifest built by hand, as ``--report-only`` would find one on disk."""
    return {"runs": list(runs), "completed": len(runs), "skipped": 0, "in_progress": False}


def _run_entry(arm, algo, version, seed=0, purpose="algorithm", condition="heterogeneous"):
    return {
        "arm": arm,
        "algo": algo,
        "condition": condition,
        "purpose": purpose,
        "seed": seed,
        "run_dir": f"/nonexistent/{arm}_{seed}",
        "config_fingerprint": "x",
        "reward_spec_version": version,
        "status": "completed",
    }


def test_report_refuses_an_algorithm_comparison_spanning_versions():
    """D13 binds the *figure and table*, not only the launcher.

    ``run.py sweep --report-only`` reads a manifest off disk and rebuilds the
    comparison figure without going through ``run_sweep``, so ``preflight`` --
    and with it ``check_reward_spec_versions`` -- never runs. A manifest that
    did not come from a checked plan (merged from two sweep directories, or
    produced before the config was edited) would render a cross-version
    comparison silently. The artifact-producing call has to refuse on its own.
    """
    manifest = _manifest(
        _run_entry("dtropd", "dtropd", "v1"),
        _run_entry("mappo", "mappo", "v1-local"),
    )
    with pytest.raises(SweepError, match="across reward spec versions"):
        sweep_report(manifest)


def test_report_allows_environment_property_arms_to_span_versions(tmp_path):
    """The D13 exception survives the report-layer guard, and stays labelled."""
    runs = []
    for arm, version, purpose in (
        ("ippo_v1", "v1", "algorithm"),
        ("probe_v1", "v1", "environment_property"),
        ("probe_local", "v1-local", "environment_property"),
    ):
        entry = _run_entry(arm, "ippo", version, purpose=purpose)
        run_dir = tmp_path / arm
        run_dir.mkdir()
        (run_dir / "metrics.jsonl").write_text("")
        entry["run_dir"] = str(run_dir)
        runs.append(entry)

    report = sweep_report(_manifest(*runs))
    block = report["environment_property"]
    assert block["cross_version"] is True
    assert "CROSS-VERSION" in block["label"]


def test_cross_version_figures_carry_the_label(tmp_path):
    """D13: labelled cross-version *wherever it appears*, figures included.

    A figure is where someone is most likely to read the measurement without
    the report open beside them, so the label has to survive onto the canvas
    rather than living only in ``report.json``.
    """
    pytest.importorskip("matplotlib")
    from dtropd.plot import plot_comparison

    report = {
        "per_arm": {"heterogeneous": {"ippo": {"iqm": 1.0, "ci": {"lo": 0.5, "hi": 1.5}, "n_seeds": 2}}},
        "conditions": ["heterogeneous"],
        "environment_property": {
            "cross_version": True,
            "reward_spec_versions": ["v1", "v1-local"],
            "label": "CROSS-VERSION environment-property measurement",
        },
    }
    out = plot_comparison(report, tmp_path / "cross.png")
    assert out.exists() and out.stat().st_size > 0

    texts = _figure_texts(report, tmp_path / "probe.png")
    assert any("CROSS-VERSION" in t for t in texts), (
        "a cross-version figure rendered without the D13 label"
    )
    assert any("v1-local" in t for t in texts), "the label must name the versions"

    # A single-version report must NOT be stamped -- the label has to mean
    # something when it appears.
    single = {k: v for k, v in report.items() if k != "environment_property"}
    assert not any("CROSS-VERSION" in t for t in _figure_texts(single, tmp_path / "s.png"))


def _figure_texts(report: dict, path: Path) -> list[str]:
    """Render a comparison figure and return every string drawn on it.

    ``plot_comparison`` closes the figure before returning, so the close is
    suppressed for the duration of the call to inspect what was actually drawn.
    """
    from unittest.mock import patch

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dtropd.plot import plot_comparison

    plt.close("all")
    with patch.object(plt, "close"):
        plot_comparison(report, path)
    texts = [
        t.get_text() for num in plt.get_fignums() for t in plt.figure(num).texts
    ]
    plt.close("all")
    return texts


# --------------------------------------------------------------------------- #
# D6: one calibration per environment, shared by every arm in it
# --------------------------------------------------------------------------- #


def test_calibration_is_grouped_by_environment_not_by_sweep(tmp_path):
    """D6 x D8: a two-environment sweep legitimately carries two artifacts."""
    plan = load_sweep_config(CONFIGS / "sweep_smoke.yaml")
    plan.out_dir = tmp_path / "s"
    result = check_calibration(plan)
    assert result["n_environments"] == 2
    ids = {g["calibration_id"] for g in result["per_environment"]}
    assert len(ids) == 2, "the two environments must have distinct calibrations"
    for group in result["per_environment"]:
        assert len(group["arms"]) == 2  # ippo + dtropd share one artifact


def test_arms_in_one_environment_calibrated_separately_are_refused(tmp_path):
    plan = load_sweep_config(CONFIGS / "sweep_smoke.yaml")
    shared = CalibrationRecord.load(plan.arms[0].overrides["distill.calibration_file"])
    forked = CalibrationRecord.from_dict(
        {**shared.to_dict(), "created_at": "1999-01-01T00:00:00+0000"}
    )
    other = forked.save(tmp_path / "forked.json")
    plan.arms[1].overrides["distill.calibration_file"] = str(other)
    with pytest.raises(SweepError, match="calibrated separately"):
        check_calibration(plan)


def test_calibration_from_a_different_environment_is_refused(tmp_path):
    plan = load_sweep_config(CONFIGS / "sweep_smoke.yaml")
    shared = CalibrationRecord.load(plan.arms[0].overrides["distill.calibration_file"])
    wrong = CalibrationRecord.from_dict(
        {**shared.to_dict(), "env_fingerprint": "some-other-env"}
    ).save(tmp_path / "wrong.json")
    plan.arms[0].overrides["distill.calibration_file"] = str(wrong)
    with pytest.raises(SweepError, match="different environment"):
        check_calibration(plan)


# --------------------------------------------------------------------------- #
# Paired seeds
# --------------------------------------------------------------------------- #


def test_paired_seeds_are_enforced():
    plan = load_sweep_config(CONFIGS / "sweep_smoke.yaml")
    assert check_paired_seeds(plan)["passed"]

    plan.arms[0].overrides["train.seed"] = 99
    with pytest.raises(SweepError, match="breaks seed pairing"):
        check_paired_seeds(plan)


def test_arms_in_one_condition_must_share_an_environment():
    plan = load_sweep_config(CONFIGS / "sweep_smoke.yaml")
    plan.arms[1].overrides["env.max_cycles"] = 12  # same condition, different env
    with pytest.raises(SweepError, match="not paired"):
        check_paired_seeds(plan)


def test_paired_arms_see_identical_environment_streams():
    """SPEC Sec. 7: identical env instances for D-TrOPD vs IPPO at one seed."""
    from dtropd.env import SyncVectorEnv
    from dtropd.utils import derive_seed

    plan = load_sweep_config(CONFIGS / "sweep_smoke.yaml")
    ippo = next(a for a in plan.arms if a.name == "ippo_het").config(plan.base)
    dtropd = next(a for a in plan.arms if a.name == "dtropd_het").config(plan.base)

    # One fixed action sequence, replayed against both arms' environments: the
    # thing under test is the env stream, so the actions must not vary.
    actions = np.random.default_rng(0).uniform(
        -1, 1, size=(3, ippo.train.num_envs, ippo.env.n_agents, 2)
    )
    streams = []
    for cfg in (ippo, dtropd):
        vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=derive_seed(0, "env"))
        obs = [vec.reset(seed=derive_seed(0, "env_reset", 0))]
        for a in actions:
            obs.append(vec.step(a).obs)
        streams.append(obs)
    for a, b in zip(*streams):
        assert np.array_equal(a, b), "paired arms saw different environment instances"


# --------------------------------------------------------------------------- #
# End-to-end: run, resume, report, figure
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_sweep_runs_resumes_and_reports(tmp_path):
    plan = smoke_plan(tmp_path)
    manifest = run_sweep(plan, progress=False)
    assert manifest["in_progress"] is False
    assert manifest["completed"] == len(plan.arms) * len(plan.seeds)
    assert manifest["skipped"] == 0
    # D6: the sweep derived one artifact per environment and pinned it.
    derived = manifest["checks"]["auto_calibration"]["derived"]
    assert derived, "auto-calibration derived nothing"
    for block in derived.values():
        assert len(block["arms_pinned"]) == 2

    # Resume: nothing left to do, everything skipped, no re-training.
    again = run_sweep(smoke_plan(tmp_path), progress=False)
    assert again["completed"] == 0
    assert again["skipped"] == len(plan.arms) * len(plan.seeds)

    report = sweep_report(manifest)
    assert set(report["conditions"]) == {"heterogeneous", "homogeneous"}
    for condition in report["per_arm"].values():
        for block in condition.values():
            assert block["n_seeds"] == len(plan.seeds)
            assert np.isfinite(block["iqm"])
            assert block["ci"]["lo"] <= block["iqm"] <= block["ci"]["hi"]

    # D8: the interaction is reported, with the sign flip called explicitly.
    interaction = report["h7_interaction"]
    assert "predicted_sign_flip_observed" in interaction
    assert interaction["heterogeneous_difference"]["paired"] is True
    # D9: the consensus-pull cost against the homogeneous IPPO control.
    assert "dtropd" in report["consensus_pull_cost"]


@pytest.mark.slow
def test_resume_skips_only_matching_configs(tmp_path):
    """A changed config must re-run, not be silently skipped as 'complete'."""
    plan = smoke_plan(tmp_path, seeds=(0,))
    run_sweep(plan, progress=False)

    changed = smoke_plan(tmp_path, seeds=(0,))
    changed.base.ppo.lr = 5e-4  # different config, same run directories
    manifest = run_sweep(changed, progress=False)
    assert manifest["skipped"] == 0, "a changed config was skipped as complete"


@pytest.mark.slow
def test_figure_is_regenerable_from_stored_metrics_alone(tmp_path):
    """SPEC Sec. 7: one command, no manual steps, nothing but the metrics."""
    pytest.importorskip("matplotlib")
    from dtropd.plot import plot_comparison

    plan = smoke_plan(tmp_path, seeds=(0, 1))
    manifest = run_sweep(plan, progress=False)
    report = sweep_report(manifest)
    figure = plot_comparison(report, tmp_path / "figures" / "comparison.png")
    assert figure.exists() and figure.stat().st_size > 0

    # Rebuild from the manifest on disk, with no in-memory state at all.
    reloaded = json.loads((plan.out_dir / "manifest.json").read_text())
    again = sweep_report(reloaded)
    assert again["per_arm"].keys() == report["per_arm"].keys()
    assert plot_comparison(again, tmp_path / "figures" / "again.png").exists()


# --------------------------------------------------------------------------- #
# D7-AMENDED: the grid is derived, and the spread is checked
# --------------------------------------------------------------------------- #


def _rec(delta: float, window: int) -> CalibrationRecord:
    scale = delta / 0.1
    return CalibrationRecord.from_samples(
        np.linspace(0.01, 0.2, 200) * scale,
        window_epochs=window,
        pooled_epochs=[window - 2, window - 1],
        reward_spec_version="v1",
        git_commit="abc",
        env_fingerprint="env",
    )


def test_d7_grid_is_derived_from_real_calibrations():
    records = {6: _rec(0.05, 6), 8: _rec(0.10, 8), 10: _rec(0.20, 10)}
    grid = d7_grid_from_records(records)
    assert grid["windows"] == [6, 8, 10]
    assert grid["spread_max_over_min"] == pytest.approx(4.0, rel=1e-6)
    assert grid["widened"] is False
    assert len(grid["grid"]) == 3


def test_d7_grid_widens_deliberately_when_the_spread_is_too_narrow():
    """A grid that tests nothing must be widened, and say that it was."""
    records = {6: _rec(0.100, 6), 8: _rec(0.105, 8), 10: _rec(0.110, 10)}
    grid = d7_grid_from_records(records)
    assert grid["spread_max_over_min"] < 1.5
    assert grid["widened"] is True
    assert grid["widening_factor"] == 2.0
    assert max(grid["grid"]) / min(grid["grid"]) == pytest.approx(4.0)
    assert "disclosed deviation" in grid["widening_reason"]


def test_d7_grid_requires_calibrations():
    from dtropd.calibration import CalibrationError

    with pytest.raises(CalibrationError):
        d7_grid_from_records({})


# --------------------------------------------------------------------------- #
# D11: Assumption 2 -- violated by v1, satisfied by v1-local
# --------------------------------------------------------------------------- #


def _env(version: str, n: int = 6, closest: int = 3) -> CooperativeCaptureEnv:
    """A 6-ring with hand-placed agents, so the test controls what is nearest.

    Whether moving an agent changes a team term depends on whether it was among
    the ``m`` nearest to some target. Leaving that to a random reset makes the
    test pass or fail by luck, so ``closest`` is placed right on target 0 and is
    therefore guaranteed to be in the nearest set.
    """
    adj = build_graph("ring", n, 2, np.random.default_rng(0))
    cfg = EnvConfig(
        n_agents=n,
        n_targets=2,
        capture_group_size=2,
        reward_spec_version=version,
        team_neighborhood=adj if version == "v1-local" else None,
        local_ratio=0.5,
    )
    cfg.validate()
    env = CooperativeCaptureEnv(cfg, seed=0)
    env.reset_array(seed=0)
    env.target_pos = np.array([[0.0, 0.0], [5.0, 5.0]])
    # Everyone on a wide arc, except `closest`, which sits on target 0.
    env.agent_pos = np.array([[2.0 + 0.3 * i, 0.0] for i in range(n)])
    env.agent_pos[closest] = np.array([0.01, 0.0])
    return env


#: v1's rewards for the hand-placed configuration in ``_env``. D13's first
#: sentence -- "v1 is frozen and unedited" -- only means something if editing it
#: fails loudly, so the values are pinned here. Every stored curve, every
#: calibration artifact and every H7 arm is denominated in this reward; a change
#: to it silently invalidates all of them rather than announcing itself.
V1_FROZEN_REWARDS = [
    -4.136071339172461,
    -4.2860713391724605,
    -4.436071339172461,
    -3.141071339172461,
    -4.736071339172462,
    -4.886071339172461,
]


def test_v1_reward_spec_is_frozen():
    """SPEC Sec. 12 / D13: `v1` is frozen and unedited.

    If this fails, the question is not "update the constants" -- it is whether
    the edit was intended, because every result already denominated in `v1`
    becomes incomparable to everything produced afterwards. A deliberate change
    is a *new* reward spec version, which is the mechanism D13 exists to provide.
    """
    env = _env("v1", closest=3)
    rewards, _ = env._rewards_and_info()
    np.testing.assert_allclose(
        rewards,
        V1_FROZEN_REWARDS,
        rtol=1e-12,
        err_msg=(
            "the v1 reward changed. D13 freezes v1: if this edit is intended, it "
            "must be a new reward spec version, not a redefinition of v1, or "
            "every stored v1 curve silently stops meaning what it meant."
        ),
    )


def test_v1_local_is_used_only_by_the_h5_locality_condition():
    """SPEC Sec. 12 / D13: `v1-local` is used ONLY by H5's locality condition.

    Checked against the shipped configs rather than trusted, because the way
    this constraint breaks is someone reaching for the localised reward in an
    unrelated sweep and quietly making its arms incomparable to everything else.
    """
    def settings(node, trail=()):
        """Every ``*reward_spec_version: <value>`` setting, comments excluded."""
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).endswith("reward_spec_version") and isinstance(value, str):
                    yield ".".join(trail + (str(key),)), value
                else:
                    yield from settings(value, trail + (str(key),))
        elif isinstance(node, list):
            for i, item in enumerate(node):
                yield from settings(item, trail + (str(i),))

    offenders = {}
    for path in sorted(CONFIGS.glob("*.yaml")):
        if path.name == "h5_locality.yaml":
            continue
        found = [where for where, v in settings(yaml.safe_load(path.read_text()))
                 if v == "v1-local"]
        if found:
            offenders[path.name] = found
    assert not offenders, (
        f"v1-local appears outside H5's locality condition: {offenders}. D13 "
        "scopes it to that condition; anywhere else it makes a comparison span "
        "reward spec versions."
    )


def test_v1_violates_assumption_2_and_v1_local_satisfies_it():
    """D11(c): the violation in the primary env is a result in its own right."""
    results = {}
    for version in ("v1", "v1-local"):
        # Agent 3 is 3 hops from agent 0 on a 6-ring -- outside N_0 -- and is
        # placed on target 0, so it is definitely in the global nearest set.
        env = _env(version, closest=3)
        baseline, _ = env._rewards_and_info()
        moved = _env(version, closest=3)
        moved.agent_pos[3] += 10.0
        after, _ = moved._rewards_and_info()
        results[version] = abs(float(after[0]) - float(baseline[0]))

    assert results["v1"] > 1e-6, (
        "v1's team term is global, so a distant agent MUST move agent 0's "
        "reward -- that is the Assumption 2(i) violation D11 documents"
    )
    assert results["v1-local"] == pytest.approx(0.0, abs=1e-12), (
        "v1-local's team term is summed over N_i, so a non-neighbour must not "
        "affect agent 0's reward at all"
    )


def test_v1_local_still_rewards_its_own_neighbourhood():
    """Locality must not mean deafness: N_i still drives agent i's team term."""
    # Agent 1 IS a neighbour of agent 0 on the ring, and is placed on target 0.
    env = _env("v1-local", closest=1)
    baseline, info = env._rewards_and_info()
    assert info["team_scope"] == "neighborhood"
    moved = _env("v1-local", closest=1)
    moved.agent_pos[1] += 10.0
    after, _ = moved._rewards_and_info()
    assert abs(float(after[0]) - float(baseline[0])) > 1e-6


def test_v1_local_requires_the_adjacency():
    cfg = EnvConfig(n_agents=4, n_targets=1, capture_group_size=2, reward_spec_version="v1-local")
    warnings_out = cfg.validate()
    assert any("team_neighborhood" in w for w in warnings_out)

    bad = EnvConfig(
        n_agents=4,
        n_targets=1,
        capture_group_size=2,
        reward_spec_version="v1-local",
        team_neighborhood=[[1], [0]],
    )
    with pytest.raises(ConfigError, match="one adjacency list per agent"):
        bad.validate()


def test_unknown_reward_spec_version_is_refused():
    with pytest.raises(ConfigError, match="not a known"):
        EnvConfig(reward_spec_version="v99").validate()


def test_reward_spec_version_is_reported_per_step():
    for version in ("v1", "v1-local"):
        env = _env(version)
        _, info = env._rewards_and_info()
        assert info["reward_spec_version"] == version


# --------------------------------------------------------------------------- #
# D11(a): the decay fit refuses to be interpreted from too few hop levels
# --------------------------------------------------------------------------- #


def test_decay_fit_is_uninterpretable_below_four_hop_levels():
    by_hop = {"1": {"mean": 1.0}, "2": {"mean": 0.5}, "3": {"mean": 0.25}}
    fit = fit_decay(by_hop)
    assert fit["hop_levels_used"] == 3 < MIN_HOP_LEVELS_FOR_A_FIT
    assert fit["interpretable"] is False
    assert "UNINTERPRETABLE" in fit["note"]
    assert np.isfinite(fit["log_slope"])  # the number exists, but is labelled


def test_decay_fit_is_interpretable_with_enough_hop_levels():
    by_hop = {str(h): {"mean": float(np.exp(-0.7 * h))} for h in range(1, 7)}
    fit = fit_decay(by_hop)
    assert fit["hop_levels_used"] == 6
    assert fit["interpretable"] is True
    assert fit["log_slope"] == pytest.approx(-0.7, rel=1e-6)
    assert fit["log_slope_ci"]["lo"] <= fit["log_slope"] <= fit["log_slope_ci"]["hi"]


def test_decay_fit_reports_when_no_slope_is_possible():
    assert fit_decay({"1": {"mean": 1.0}})["hop_levels_used"] == 1
    assert "no slope can be fitted" in fit_decay({"1": {"mean": 1.0}})["note"]


def test_h5_locality_config_has_enough_graph_diameter():
    """D11(a): a ring over 16 agents gives diameter 8, so >= 4 hop levels."""
    from dtropd.instrumentation import graph_distances

    plan = load_sweep_config(CONFIGS / "h5_locality.yaml")
    cfg = plan.arms[0].config(plan.base)
    adj = build_graph(cfg.graph.topology, cfg.env.n_agents, cfg.graph.degree, np.random.default_rng(0))
    diameter = max(max(graph_distances(adj, i)) for i in range(cfg.env.n_agents))
    assert diameter >= MIN_HOP_LEVELS_FOR_A_FIT + 2, diameter
