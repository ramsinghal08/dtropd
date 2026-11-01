"""The config-driven multi-seed experiment harness.

SPEC Sec. 5 asks for a launcher, resumable checkpoints, rliable-style
plotting, and a paired-seed comparison mode, with every figure regenerable from
stored metrics by one command.

Three enforcement rules are wired in here rather than left to discipline,
because each of them is a way a comparison can be quietly invalid:

* **D6** -- every arm must load the *same* calibration artifact. A sweep whose
  arms were calibrated separately is refused before a single epoch runs.
* **D13** -- no algorithm-performance comparison may span reward spec versions.
  Environment-property measurements may, and are labelled ``cross_version``.
* **paired seeds** -- arms sharing a seed get identical environment instances,
  so a D-TrOPD/IPPO difference is a difference in the algorithm and not in the
  episodes it happened to see.

Resumability is deliberately dumb: a run is complete iff its ``summary.json``
exists, its recorded config fingerprint matches what the sweep asked for, and
its metrics file parses without duplicate epochs. Killing a sweep and restarting
it re-plans everything and skips what is already done. There is no partial-epoch
resume, and that is a design choice, not an omission -- a half-written metrics
file is worse than a re-run.

That last clause was aspirational until the writer stopped opening in append
mode: a killed cell's partial file used to be concatenated with the re-run's,
producing exactly the half-written file this claims to avoid. The metrics check
in ``_is_complete`` is what makes an already-corrupted sweep self-healing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from .analysis import (
    analyze_run,
    bootstrap_ci,
    consensus_pull_cost,
    d7_sensitivity_verdict,
    h7_interaction,
    iqm,
)
from .calibration import (
    CalibrationError,
    CalibrationRecord,
    assert_shared_calibration,
    env_fingerprint,
)
from .config import Config, ConfigError
from .utils import CorruptMetricsError, get_logger, read_metrics

__all__ = [
    "SweepArm",
    "SweepPlan",
    "SweepError",
    "build_plan",
    "run_sweep",
    "load_sweep_config",
    "check_reward_spec_versions",
    "check_manifest_reward_spec_versions",
    "sweep_report",
]

logger = get_logger("dtropd.sweep")


class SweepError(RuntimeError):
    """Raised when a sweep is structurally invalid and must not be run."""


#: Algorithms whose update includes the distillation term, so `distill.enabled`
#: must be on for them. This drives CONFIGURATION -- forgetting an entry here
#: silently turns that algorithm into IPPO, which the SnapshotDistillBase
#: constructor refuses loudly rather than allowing.
DISTILLERS = ("dtropd", "naive_distill", "forward_kl_distill")

#: Algorithms the D9 consensus-pull cost is REPORTED for. Deliberately narrower
#: than DISTILLERS: `forward_kl_distill` is an H2 ablation that violates the
#: single-direction rule by construction, so it belongs in no headline table.
REPORTED_DISTILLERS = ("dtropd", "naive_distill")


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #


@dataclass
class SweepArm:
    """One cell of the sweep: an algorithm, an environment, a calibration."""

    name: str
    algo: str
    overrides: dict[str, Any] = field(default_factory=dict)
    #: free-form label used to group arms in the comparison figure
    condition: str = "default"
    #: purpose of the arm: "algorithm" arms are compared with each other and are
    #: subject to D13; "environment_property" arms measure the environment and
    #: may span reward spec versions.
    purpose: str = "algorithm"

    def config(self, base: Config) -> Config:
        cfg = Config.from_dict(dataclasses.asdict(base))
        cfg.train.algo = self.algo
        cfg.distill.enabled = self.algo in DISTILLERS
        for dotted, value in self.overrides.items():
            _set_dotted(cfg, dotted, value)
        cfg.validate(emit=False)
        return cfg


def _set_dotted(cfg: Config, dotted: str, value: Any) -> None:
    target: Any = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ConfigError(f"unknown config section {part!r} in override {dotted!r}")
        target = getattr(target, part)
    if not hasattr(target, parts[-1]):
        raise ConfigError(f"unknown config key {dotted!r}")
    setattr(target, parts[-1], value)


@dataclass
class SweepPlan:
    name: str
    base: Config
    arms: list[SweepArm]
    seeds: list[int]
    out_dir: Path
    notes: str = ""
    #: Derive any missing calibration artifact ONCE per environment before the
    #: sweep starts, then pin it for every arm in that environment. This is the
    #: D6 workflow, not an exception to it: what D6 forbids is arms calibrating
    #: *separately*, and a sweep-level derive-once-then-pin is exactly the
    #: guarantee it asks for -- while keeping the acceptance a single
    #: command. Set false to require the artifacts to pre-exist.
    auto_calibrate: bool = True

    @property
    def runs(self) -> list[tuple[SweepArm, int]]:
        return [(arm, seed) for arm in self.arms for seed in self.seeds]

    def run_dir(self, arm: SweepArm, seed: int) -> Path:
        return self.out_dir / "runs" / f"{arm.name}_seed{seed}"


def config_fingerprint(cfg: Config) -> str:
    payload = json.dumps(dataclasses.asdict(cfg), sort_keys=True, default=str).encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_sweep_config(path: str | Path) -> SweepPlan:
    """Load a sweep YAML: a base config plus the arms and seeds to run."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if "base" not in raw or "arms" not in raw:
        raise SweepError(
            f"{path}: a sweep config needs 'base' (a full run config) and 'arms'"
        )
    base = Config.from_dict(raw["base"])
    arms = [
        SweepArm(
            name=a["name"],
            algo=a["algo"],
            overrides=a.get("overrides", {}) or {},
            condition=a.get("condition", "default"),
            purpose=a.get("purpose", "algorithm"),
        )
        for a in raw["arms"]
    ]
    return SweepPlan(
        name=raw.get("name", Path(path).stem),
        base=base,
        arms=arms,
        seeds=list(raw.get("seeds", [0])),
        out_dir=Path(raw.get("out_dir", f"runs/{Path(path).stem}")),
        notes=raw.get("notes", ""),
        auto_calibrate=bool(raw.get("auto_calibrate", True)),
    )


def build_plan(
    path: str | Path, seeds: int | Sequence[int] | None = None, out_dir: str | Path | None = None
) -> SweepPlan:
    plan = load_sweep_config(path)
    if seeds is not None:
        plan.seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    if out_dir is not None:
        plan.out_dir = Path(out_dir)
    return plan


# --------------------------------------------------------------------------- #
# Pre-flight guards -- run before any epoch
# --------------------------------------------------------------------------- #


def check_reward_spec_versions(plan: SweepPlan) -> dict[str, Any]:
    """SPEC Sec. 12 / D13: algorithm comparisons may not span reward versions.

    Environment-property arms are exempt *and labelled*, because for them the
    environments are the subject rather than a confound.
    """
    rows = []
    for arm in plan.arms:
        cfg = arm.config(plan.base)
        rows.append((arm, cfg.env.reward_spec_version))

    algorithm_versions = {
        v for arm, v in rows if arm.purpose == "algorithm"
    }
    property_versions = {v for arm, v in rows if arm.purpose != "algorithm"}

    if len(algorithm_versions) > 1:
        detail = ", ".join(
            f"{arm.name}={v}" for arm, v in rows if arm.purpose == "algorithm"
        )
        raise SweepError(
            "this sweep compares algorithms across reward spec versions, which "
            f"makes the comparison meaningless: {detail}. SPEC Sec. 12 / D13 "
            "allows versions to coexist but forbids an algorithm-performance "
            "comparison from spanning them. Split it into one sweep per version, "
            "or mark the cross-version arms purpose: environment_property."
        )
    return {
        "algorithm_arms_version": sorted(algorithm_versions),
        "environment_property_versions": sorted(property_versions),
        "cross_version": len(algorithm_versions | property_versions) > 1,
        "passed": True,
    }


def check_calibration(plan: SweepPlan, require: bool | None = None) -> dict[str, Any]:
    """SPEC Sec. 11 / D6: arms sharing an environment share one calibration.

    D6 says the calibration is measured "once per environment configuration,
    then pinned for every arm in that comparison". A D8 sweep spans *two*
    environments (heterogeneous treatment + homogeneous control), so it
    legitimately carries two artifacts -- one per environment -- and the
    sharing requirement applies **within** each environment, not across the
    whole sweep. Requiring one artifact for the entire sweep would make D6 and
    D8 mutually unsatisfiable.

    Arms are therefore grouped by environment fingerprint, and every arm in a
    group must pin the same artifact -- including arms with no radius of their
    own, which declare it for auditability (D4 x D6).
    """
    groups: dict[str, list[tuple[str, CalibrationRecord | None]]] = {}
    needs: dict[str, bool] = {}
    for arm in plan.arms:
        cfg = arm.config(plan.base)
        fingerprint = env_fingerprint(cfg.env)
        record = None
        if cfg.distill.calibration_file:
            record = CalibrationRecord.load(cfg.distill.calibration_file)
            if record.env_fingerprint != fingerprint:
                raise SweepError(
                    f"arm {arm.name!r} pins a calibration measured on a different "
                    f"environment configuration ({record.env_fingerprint} != "
                    f"{fingerprint}). Calibration is per environment configuration "
                    "(SPEC Sec. 11 / D6)."
                )
        groups.setdefault(fingerprint, []).append((arm.name, record))
        needs[fingerprint] = needs.get(fingerprint, False) or arm.algo == "dtropd"

    shared: dict[str, str | None] = {}
    for fingerprint, rows in groups.items():
        required = needs[fingerprint] if require is None else require
        try:
            shared[fingerprint] = assert_shared_calibration(
                rows, require_calibration=required
            )
        except CalibrationError as exc:
            names = ", ".join(name for name, _ in rows)
            raise SweepError(
                f"environment group [{names}]: {exc}"
            ) from exc

    return {
        "per_environment": [
            {
                "env_fingerprint": fingerprint,
                "arms": [name for name, _ in rows],
                "calibration_id": shared[fingerprint],
                "calibration_required": needs[fingerprint] if require is None else require,
            }
            for fingerprint, rows in groups.items()
        ],
        "n_environments": len(groups),
        "passed": True,
    }


def check_paired_seeds(plan: SweepPlan) -> dict[str, Any]:
    """Arms sharing a seed must see identical environment instances.

    SPEC Sec. 7: "Paired seeds and identical env instances for D-TrOPD vs IPPO."
    Env streams are derived from the run seed alone, so this holds by
    construction; the check makes that explicit rather than assumed, and fails
    loudly if an arm overrides something that would break it.
    """
    offenders = []
    for arm in plan.arms:
        if any(k.startswith("train.seed") for k in arm.overrides):
            offenders.append(arm.name)
    if offenders:
        raise SweepError(
            f"arms {offenders} override train.seed, which breaks seed pairing: "
            "arms in one sweep must share the seed list so their environment "
            "instances are identical (SPEC Sec. 7)."
        )
    env_cfgs = {
        arm.name: env_fingerprint(arm.config(plan.base).env)
        for arm in plan.arms
        if arm.purpose == "algorithm"
    }
    by_condition: dict[str, set[str]] = {}
    for arm in plan.arms:
        if arm.purpose == "algorithm":
            by_condition.setdefault(arm.condition, set()).add(env_cfgs[arm.name])
    mismatched = {c: sorted(v) for c, v in by_condition.items() if len(v) > 1}
    if mismatched:
        raise SweepError(
            "arms within one condition have different environment configurations, "
            f"so their seeds are not paired: {mismatched}"
        )
    return {
        "seeds": plan.seeds,
        "conditions": {c: sorted(v)[0] for c, v in by_condition.items()},
        "passed": True,
    }


def ensure_calibrations(plan: SweepPlan, seed: int | None = None) -> dict[str, Any]:
    """Derive any missing calibration artifact once per environment, then pin it.

    SPEC Sec. 11 / D6 requires the attenuation to be "measured ONCE per
    environment configuration, then PINNED for every arm in that comparison".
    That is what this does: one measurement per distinct environment, written to
    the sweep directory, and every arm in that environment rewritten to point at
    it. No arm ever measures its own.

    Runs before any training epoch, so the artifacts are fixed for the sweep.
    """
    from .run import _measure_calibration  # local import: CLI helper, one copy

    if not plan.auto_calibrate:
        return {"auto_calibrate": False, "derived": {}}

    groups: dict[str, list[SweepArm]] = {}
    for arm in plan.arms:
        groups.setdefault(env_fingerprint(arm.config(plan.base).env), []).append(arm)

    derived: dict[str, Any] = {}
    cal_dir = plan.out_dir / "calibrations"
    for fingerprint, arms in groups.items():
        if not any(a.algo == "dtropd" for a in arms):
            continue
        existing = {
            a.config(plan.base).distill.calibration_file
            for a in arms
            if a.config(plan.base).distill.calibration_file
        }
        if len(existing) == 1 and Path(next(iter(existing))).exists():
            continue  # already pinned by the config; leave it alone
        cal_dir.mkdir(parents=True, exist_ok=True)
        path = cal_dir / f"{fingerprint}.json"
        if not path.exists():
            # Prefer an UNSTRESSED arm to measure on. delta_base is a quantile of
            # the observed D_ij distribution, and a policy-level adversary's
            # edges sit far out in that distribution -- calibrating with one
            # active lets the adversary widen the very radius that is supposed to
            # exclude it. Measured on a smoke config at f=0.5: delta_base 0.1 ->
            # 1.2545, i.e. the gate largely calibrated itself away.
            candidates = [a for a in arms if a.algo == "dtropd"]
            clean = [a for a in candidates if not a.config(plan.base).stress.active]
            template_arm = clean[0] if clean else candidates[0]
            if not clean:
                logger.warning(
                    "environment %s has no unstressed D-TrOPD arm, so the shared "
                    "calibration is measured on %r WITH its stressor active. "
                    "delta_base is a quantile of the observed D_ij distribution, "
                    "so an adversary present during calibration moves the trust "
                    "radius itself. Add an f=0 arm if the radius is meant to be "
                    "independent of the attack.",
                    fingerprint,
                    template_arm.name,
                )
            template = template_arm.config(plan.base)
            logger.info(
                "deriving the shared calibration for environment %s from arm %r "
                "(%d arms)",
                fingerprint,
                template_arm.name,
                len(arms),
            )
            record = _measure_calibration(
                template,
                window=template.distill.calibration_epochs,
                seed=plan.seeds[0] if seed is None else seed,
                source=plan.name,
            )
            record.save(path)
            derived[fingerprint] = {
                "artifact": str(path),
                "calibration_id": record.calibration_id,
                "delta_base": record.delta_base,
                "gamma_attenuation": record.gamma_attenuation,
                "window_epochs": record.window_epochs,
                "arms_pinned": [a.name for a in arms],
            }
        for arm in arms:
            arm.overrides["distill.calibration"] = "file"
            arm.overrides["distill.calibration_file"] = str(path)
    return {"auto_calibrate": True, "derived": derived}


def preflight(plan: SweepPlan, require_calibration: bool | None = None) -> dict[str, Any]:
    """Every structural check, run before a single epoch."""
    return {
        "reward_spec": check_reward_spec_versions(plan),
        "calibration": check_calibration(plan, require=require_calibration),
        "paired_seeds": check_paired_seeds(plan),
    }


# --------------------------------------------------------------------------- #
# Execution (resumable)
# --------------------------------------------------------------------------- #


def _is_complete(run_dir: Path, fingerprint: str) -> bool:
    summary = run_dir / "summary.json"
    if not summary.exists():
        return False
    meta = run_dir / "metadata.json"
    if not meta.exists():
        return False
    try:
        recorded = json.loads(meta.read_text()).get("config_fingerprint")
    except json.JSONDecodeError:  # pragma: no cover - a truncated write
        return False
    if recorded != fingerprint:
        return False
    # A cell can be "finished" and still be unusable: before MetricsWriter
    # truncated, a killed-then-resumed cell wrote summary.json at the end of the
    # SECOND attempt while metrics.jsonl held both. Checking only the sentinel
    # files marks that complete forever. Reading the metrics is what makes the
    # resume path self-healing.
    metrics = run_dir / "metrics.jsonl"
    if metrics.exists():
        try:
            read_metrics(metrics)
        except CorruptMetricsError as exc:
            logger.warning("re-running %s: %s", run_dir.name, exc)
            return False
    return True


def _run_cell(job: tuple[Any, int, Any, str, Path, str]) -> dict[str, Any]:
    """Train one (arm, seed) cell. Top-level so it is picklable by workers."""
    from .train import train

    arm_name, seed, cfg, fingerprint, run_dir, sweep_name = job
    result = train(cfg, seed=seed, out_dir=run_dir, progress=False)
    # SPEC Sec. 12 / D12: the fingerprint is an audit artifact, so it is
    # written after the run it audits, not before.
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["config_fingerprint"] = fingerprint
    meta["sweep"] = {"name": sweep_name, "arm": arm_name}
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return _cell_returns(result.summary())


#: What a manifest entry records about a cell's outcome. `final_return` is the
#: ALL-AGENT mean and keeps that meaning; `final_return_honest` excludes the
#: Byzantine agents and equals it when there are none, so a consumer can read one
#: key along the whole `f` axis without comparing an honest-only number against
#: an all-agent one at `f = 0`.
_RETURN_KEYS = ("final_return", "final_return_honest")


def _cell_returns(summary: dict[str, Any]) -> dict[str, Any]:
    out = {"final_return": summary.get("final_return")}
    out["final_return_honest"] = summary.get("final_return_honest", out["final_return"])
    return out


def _returns_from_disk(run_dir: Path) -> dict[str, Any]:
    """A SKIPPED cell's outcome, read back from the run it is skipping.

    Without this a resumed sweep's manifest carries no return at all for the
    cells it skipped, so any consumer reading the manifest silently analyses
    only the cells that happened to run this time -- and a resumed sweep is the
    normal case, not the exception.
    """
    summary = run_dir / "summary.json"
    if not summary.exists():
        return {}
    try:
        return _cell_returns(json.loads(summary.read_text()))
    except json.JSONDecodeError:  # pragma: no cover - a truncated write
        return {}


def run_sweep(
    plan: SweepPlan,
    *,
    resume: bool = True,
    progress: bool = False,
    require_calibration: bool | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Run (or resume) every cell of the sweep and write a manifest.

    ``workers`` runs cells in parallel processes. This does **not** change any
    result: every cell is seeded from its own ``(arm, seed)`` and writes to its
    own directory, so cells never share state. What it changes is only which
    cell is running when -- and the manifest is still assembled in plan order,
    so the artifact is identical regardless of ``workers``.

    Each worker sets its own torch thread count to 1 (via ``TrainConfig``), so
    ``workers`` should stay at or below the core count; oversubscribing makes
    every cell slower without finishing any sooner.
    """
    plan.out_dir.mkdir(parents=True, exist_ok=True)
    calibration = ensure_calibrations(plan)
    checks = preflight(plan, require_calibration=require_calibration)
    checks["auto_calibration"] = calibration
    checks["workers"] = workers

    manifest_path = plan.out_dir / "manifest.json"
    started = time.time()
    entries: list[dict[str, Any]] = []
    skipped = 0

    # Decide skip-vs-run for every cell up front, so the parallel path and the
    # sequential path make the same resume decisions from the same state.
    planned: list[dict[str, Any]] = []
    for arm, seed in plan.runs:
        cfg = arm.config(plan.base)
        fingerprint = config_fingerprint(cfg)
        run_dir = plan.run_dir(arm, seed)
        entry = {
            "arm": arm.name,
            "algo": arm.algo,
            "condition": arm.condition,
            "purpose": arm.purpose,
            "seed": seed,
            "run_dir": str(run_dir),
            "config_fingerprint": fingerprint,
            "reward_spec_version": cfg.env.reward_spec_version,
        }
        planned.append(
            {"entry": entry, "cfg": cfg, "arm": arm, "seed": seed,
             "run_dir": run_dir, "fingerprint": fingerprint,
             "skip": bool(resume and _is_complete(run_dir, fingerprint))}
        )

    pending = [p for p in planned if not p["skip"]]
    results: dict[int, dict[str, Any]] = {}
    if workers > 1 and pending:
        import multiprocessing as mp

        jobs = [
            (p["arm"].name, p["seed"], p["cfg"], p["fingerprint"], p["run_dir"], plan.name)
            for p in pending
        ]
        logger.info("running %d cells on %d workers", len(jobs), workers)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for i, out in enumerate(pool.imap(_run_cell, jobs)):
                results[id(pending[i])] = out
                logger.info("done %s seed %d (%d/%d)",
                            pending[i]["arm"].name, pending[i]["seed"], i + 1, len(jobs))
                # Progress marker written as results arrive. Without it a
                # parallel sweep produces no manifest at all until the very last
                # cell finishes, so a five-hour run killed at hour four looks
                # like a run that never started. Resume reads run directories
                # rather than this file, so it stays correct either way -- what
                # this restores is visibility, not correctness.
                manifest_path.write_text(
                    json.dumps(
                        {
                            "sweep": plan.name,
                            "notes": plan.notes,
                            "seeds": plan.seeds,
                            "checks": checks,
                            "runs": [],
                            "in_progress": True,
                            "cells_finished": i + 1,
                            "cells_total": len(jobs),
                        },
                        indent=2,
                        default=str,
                    )
                )

    for p in planned:
        entry = p["entry"]
        if p["skip"]:
            entry["status"] = "skipped (already complete)"
            skipped += 1
            entry.update(_returns_from_disk(p["run_dir"]))
            logger.info("skip %s seed %d (already complete)", p["arm"].name, p["seed"])
        elif id(p) in results:
            entry["status"] = "completed"
            entry.update({k: results[id(p)].get(k) for k in _RETURN_KEYS})
        else:
            logger.info("run  %s seed %d", p["arm"].name, p["seed"])
            out = _run_cell(
                (p["arm"].name, p["seed"], p["cfg"], p["fingerprint"], p["run_dir"], plan.name)
            )
            entry["status"] = "completed"
            entry.update({k: out.get(k) for k in _RETURN_KEYS})
        entries.append(entry)
        # Written incrementally so a killed sweep still has an accurate manifest.
        manifest_path.write_text(
            json.dumps(
                {
                    "sweep": plan.name,
                    "notes": plan.notes,
                    "seeds": plan.seeds,
                    "checks": checks,
                    "runs": entries,
                    "in_progress": True,
                },
                indent=2,
                default=str,
            )
        )

    manifest = {
        "sweep": plan.name,
        "notes": plan.notes,
        "seeds": plan.seeds,
        "checks": checks,
        "runs": entries,
        "in_progress": False,
        "completed": sum(1 for e in entries if e["status"] == "completed"),
        "skipped": skipped,
        "wall_clock_seconds": time.time() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def collect_final_returns(
    manifest: dict[str, Any], *, by_seed: bool = False
) -> dict[str, dict[str, Any]]:
    """``{condition: {algo: [final return per seed]}}`` from stored metrics only.

    With ``by_seed=True`` the innermost value is ``{seed: return}`` instead of a
    list. Any PAIRED statistic must use that form: a bare list is aligned only by
    the manifest happening to be in plan order, which is true today and is not a
    guarantee anything should rest on -- a resumed or reordered sweep would pair
    seed 3's treatment against seed 7's control and report it with a confident CI.
    """
    out: dict[str, dict[str, Any]] = {}
    for entry in manifest["runs"]:
        if entry["purpose"] != "algorithm":
            continue
        records = read_metrics(Path(entry["run_dir"]) / "metrics.jsonl")
        if not records:
            continue
        evals = [r["eval/return_mean"] for r in records if "eval/return_mean" in r]
        final = evals[-1] if evals else records[-1].get("episode_return_mean", float("nan"))
        block = out.setdefault(entry["condition"], {})
        if by_seed:
            block.setdefault(entry["algo"], {})[int(entry["seed"])] = float(final)
        else:
            block.setdefault(entry["algo"], []).append(float(final))
    return out


def check_manifest_reward_spec_versions(manifest: dict[str, Any]) -> dict[str, Any]:
    """SPEC Sec. 12 / D13, enforced again at the artifact layer.

    :func:`check_reward_spec_versions` guards the *plan*, but D13 binds "any
    figure or table comparing algorithms", and not every figure comes from a
    checked plan. ``run.py sweep --report-only`` reads a manifest off disk and
    rebuilds the figure without calling :func:`run_sweep`, so ``preflight``
    never runs; a manifest merged from two sweep directories, or written before
    the config was edited, would render a cross-version comparison silently.

    Environment-property runs are exempt, per the D13 exception -- for them the
    environments are the subject rather than a confound.
    """
    versions = {
        entry["reward_spec_version"]
        for entry in manifest["runs"]
        if entry.get("purpose", "algorithm") == "algorithm"
        and "reward_spec_version" in entry
    }
    if len(versions) > 1:
        detail = ", ".join(
            f"{entry['arm']}={entry['reward_spec_version']}"
            for entry in manifest["runs"]
            if entry.get("purpose", "algorithm") == "algorithm"
        )
        raise SweepError(
            "this manifest compares algorithms across reward spec versions, so "
            f"no figure or table may be drawn from it: {detail}. SPEC Sec. 12 / "
            "D13 allows versions to coexist but forbids an algorithm-performance "
            "comparison from spanning them. Report each version separately, or "
            "mark the cross-version runs purpose: environment_property."
        )
    return {"algorithm_arms_version": sorted(versions), "passed": True}


def sweep_report(manifest: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """The D8 interaction, the D9 cost and per-arm IQM/CIs -- all from disk."""
    # Refuse before doing any work, so an invalid comparison cannot half-render.
    check_manifest_reward_spec_versions(manifest)
    blocks = collect_final_returns(manifest)
    per_arm = {
        condition: {
            algo: {
                "iqm": iqm(values),
                "ci": bootstrap_ci(values, seed=seed),
                "n_seeds": len(values),
                "values": values,
            }
            for algo, values in algos.items()
        }
        for condition, algos in blocks.items()
    }

    report: dict[str, Any] = {"per_arm": per_arm, "conditions": sorted(blocks)}

    het = blocks.get("heterogeneous", {})
    hom = blocks.get("homogeneous", {})
    if het and hom:
        # SPEC Sec. 10 / D8: the claim is the sign flip across environments.
        # Passed SEED-KEYED so the interaction CI is genuinely paired.
        by_seed = collect_final_returns(manifest, by_seed=True)
        report["h7_interaction"] = h7_interaction(
            by_seed.get("heterogeneous", het), by_seed.get("homogeneous", hom), seed=seed
        )
    if hom.get("ippo") and any(a in hom for a in REPORTED_DISTILLERS):
        # SPEC Sec. 11 / D9: the floor the heterogeneous benefit must clear.
        report["consensus_pull_cost"] = {
            algo: consensus_pull_cost(hom["ippo"], hom[algo], seed=seed)
            for algo in REPORTED_DISTILLERS
            if algo in hom
        }

    # Environment-property arms are reported separately and labelled, per D13.
    property_runs = [e for e in manifest["runs"] if e["purpose"] != "algorithm"]
    if property_runs:
        versions = sorted({e["reward_spec_version"] for e in property_runs})
        report["environment_property"] = {
            "cross_version": len(versions) > 1,
            "reward_spec_versions": versions,
            "label": (
                "CROSS-VERSION environment-property measurement (SPEC Sec. 12 / "
                "D13 exception): compares whether Assumption 2 holds, NOT how "
                "algorithms perform."
            ),
            "decay_probe_by_version": _decay_by_version(property_runs),
        }
    return report


def _decay_by_version(runs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for entry in runs:
        records = read_metrics(Path(entry["run_dir"]) / "metrics.jsonl")
        metadata_path = Path(entry["run_dir"]) / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        result = analyze_run(records, metadata)["H5"]
        probe = result.value.get("decay_probe")
        if probe:
            out.setdefault(entry["reward_spec_version"], []).append(
                {
                    "arm": entry["arm"],
                    "seed": entry["seed"],
                    "log_slope": probe.get("log_slope"),
                    "log_slope_ci": probe.get("log_slope_ci"),
                    "hop_levels_used": probe.get("hop_levels_used"),
                    "interpretable": probe.get("interpretable"),
                    "verdict": probe.get("verdict"),
                }
            )
    return out
