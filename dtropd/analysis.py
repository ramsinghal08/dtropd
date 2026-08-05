"""H1-H7 measurements, computed from a stored metrics file.

SPEC Sec. 5, Phase 5 acceptance:

    "a single training run produces a metrics file from which every H1-H7
     measurement in Sec. 6 can be computed **without re-running**."

That is the contract this module implements and
``experiments/analyze_hypotheses.py`` demonstrates. Every function here takes
parsed metrics records (plus, where a hypothesis is inherently a
*between-run* comparison, several runs) and returns either a value or an
explicit statement of what is missing. A hypothesis whose measurement cannot be
computed from the file is a Phase 5 failure, so ``missing`` is a first-class
outcome rather than an exception.

Single-run vs multi-run: H1-H5 are measurable within one run. H6 and H7 are
comparisons *across* runs by construction (Byzantine fraction, algorithm), so
their single-run functions report the per-run quantity the comparison consumes
and say so. That distinction is recorded rather than papered over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "HypothesisResult",
    "analyze_run",
    "HYPOTHESES",
    "bootstrap_ci",
    "consensus_pull_cost",
    "d7_sensitivity_verdict",
    "D7_DELTA_BASE_GRID",
    "h7_interaction",
]


@dataclass
class HypothesisResult:
    """One hypothesis' measurement, or a precise account of why it is missing."""

    key: str
    title: str
    computable: bool
    scope: str  # "single-run" | "multi-run"
    value: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.key,
            "title": self.title,
            "computable_from_this_file": self.computable,
            "scope": self.scope,
            "value": self.value,
            "missing_fields": self.missing_fields,
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _series(records: Sequence[dict], key: str) -> list[float]:
    return [r[key] for r in records if key in r and r[key] is not None]


def _present(records: Sequence[dict], keys: Sequence[str]) -> list[str]:
    """Return the subset of ``keys`` that never appears in any record."""
    seen = set()
    for r in records:
        seen.update(r.keys())
    return [k for k in keys if k not in seen]


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI of the mean (SPEC Sec. 7)."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(draws, alpha)),
        "hi": float(np.quantile(draws, 1.0 - alpha)),
        "n": int(arr.size),
    }


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean (SPEC Sec. 7)."""
    arr = np.sort(np.asarray(list(values), dtype=np.float64))
    if arr.size == 0:
        return float("nan")
    lo, hi = int(0.25 * arr.size), int(math.ceil(0.75 * arr.size))
    core = arr[lo:hi] if hi > lo else arr
    return float(core.mean())


# --------------------------------------------------------------------------- #
# H1 -- no chattering
# --------------------------------------------------------------------------- #


def h1_chattering(records: Sequence[dict]) -> HypothesisResult:
    required = ["h1/sign_flips_by_bucket", "h1/sign_flip_rate"]
    missing = _present(records, required)
    if missing:
        return HypothesisResult(
            "H1", "no chattering", False, "single-run", missing_fields=missing,
            note="requires instrument.per_edge_gradients",
        )

    per_bucket: dict[str, dict[str, float]] = {}
    for r in records:
        for label, row in (r.get("h1/sign_flips_by_bucket") or {}).items():
            acc = per_bucket.setdefault(label, {"flips": 0.0, "edges": 0.0})
            acc["flips"] += row["flips"]
            acc["edges"] += row["edges"]
    rates = {
        label: (acc["flips"] / acc["edges"] if acc["edges"] else float("nan"))
        for label, acc in per_bucket.items()
    }
    # H1's discriminator: is the flip rate elevated in the bucket nearest the
    # radius boundary relative to the rest?
    ordered = sorted(rates)
    near = rates.get(ordered[0], float("nan")) if ordered else float("nan")
    far = (
        float(np.nanmean([rates[k] for k in ordered[1:]]))
        if len(ordered) > 1
        else float("nan")
    )
    return HypothesisResult(
        "H1",
        "no chattering: per-edge sign flips bucketed by |D_ij - delta_i|",
        True,
        "single-run",
        value={
            "flip_rate_by_bucket": rates,
            "flip_rate_overall": float(np.nanmean(_series(records, "h1/sign_flip_rate"))),
            "nearest_bucket": ordered[0] if ordered else None,
            "flip_rate_nearest_boundary": near,
            "flip_rate_elsewhere": far,
            "boundary_excess": (near - far) if np.isfinite(near) and np.isfinite(far) else float("nan"),
            "edges_compared": int(sum(a["edges"] for a in per_bucket.values())),
        },
        note="compare boundary_excess between the continuous and switched arms",
    )


# --------------------------------------------------------------------------- #
# H2 -- forward-KL inflates entropy
# --------------------------------------------------------------------------- #


def h2_entropy_inflation(records: Sequence[dict], high_d_quantile: float = 0.75) -> HypothesisResult:
    missing = _present(records, ["edges"])
    if missing:
        return HypothesisResult(
            "H2", "forward-KL inflates entropy", False, "single-run",
            missing_fields=missing, note="requires instrument.per_edge_detail",
        )

    trajectory = []
    for r in records:
        rows = [e for e in (r.get("edges") or []) if "D_ij" in e]
        if not rows:
            continue
        d = np.asarray([e["D_ij"] for e in rows])
        cut = float(np.quantile(d, high_d_quantile))
        high = [e for e in rows if e["D_ij"] >= cut]
        low = [e for e in rows if e["D_ij"] < cut]
        if not high:
            continue
        trajectory.append(
            {
                "epoch": r.get("epoch"),
                "d_cut": cut,
                "sigma_student_high_d": float(np.mean([e["sigma_i"] for e in high])),
                "sigma_teacher_high_d": float(np.mean([e["sigma_j"] for e in high])),
                "sigma_student_low_d": (
                    float(np.mean([e["sigma_i"] for e in low])) if low else float("nan")
                ),
                "mu_gap_proxy": float(np.mean([e["D_ij"] for e in high])),
            }
        )
    if not trajectory:
        return HypothesisResult(
            "H2", "forward-KL inflates entropy", False, "single-run",
            missing_fields=["edges[].D_ij"],
            note="no per-edge rows carried a D_ij (arm has no divergence measure)",
        )
    return HypothesisResult(
        "H2",
        "sigma trajectories on high-D_ij edges",
        True,
        "single-run",
        value={
            "high_d_quantile": high_d_quantile,
            "trajectory": trajectory,
            "final_sigma_student_high_d": trajectory[-1]["sigma_student_high_d"],
            "final_sigma_teacher_high_d": trajectory[-1]["sigma_teacher_high_d"],
            "sigma_ratio_final": (
                trajectory[-1]["sigma_student_high_d"]
                / max(1e-12, trajectory[-1]["sigma_teacher_high_d"])
            ),
        },
        note=(
            "reverse-KL predicts the student's sigma tracks the teacher's; "
            "forward-KL predicts sigma_i^2 -> sigma_j^2 + (mu_i - mu_j)^2"
        ),
    )


# --------------------------------------------------------------------------- #
# H3 -- radius pathologies
# --------------------------------------------------------------------------- #


def h3_radius_pathology(records: Sequence[dict]) -> HypothesisResult:
    required = ["delta_mean", "nonstationarity_mean", "radius_reference"]
    missing = _present(records, required)
    if missing:
        return HypothesisResult(
            "H3", "radius pathologies", False, "single-run", missing_fields=missing,
            note="only arms with a trust radius log these",
        )
    delta = _series(records, "delta_mean")
    nonstat = _series(records, "nonstationarity_mean")
    ev = _series(records, "explained_variance")
    corr = float("nan")
    if len(delta) > 2 and np.std(delta) > 0 and np.std(nonstat) > 0:
        corr = float(np.corrcoef(delta, nonstat)[0, 1])
    return HypothesisResult(
        "H3",
        "radius responds to nonstationarity, not to critic quality",
        True,
        "single-run",
        value={
            "radius_signal": records[-1].get("radius_signal"),
            "radius_reference": records[-1].get("radius_reference"),
            "delta_first": delta[0] if delta else float("nan"),
            "delta_last": delta[-1] if delta else float("nan"),
            "delta_vs_nonstationarity_corr": corr,
            "expected_sign": "negative (delta contracts as Delta rises)",
            "explained_variance_last": ev[-1] if ev else float("nan"),
        },
        note="the full H3 verdict needs the degenerate-critic and 10x-rescale arms",
    )


# --------------------------------------------------------------------------- #
# H4 / H5 -- simultaneity penalty
# --------------------------------------------------------------------------- #


def _simultaneity_gap(records: Sequence[dict]) -> list[dict[str, Any]]:
    rows = []
    for prev, cur in zip(records, records[1:]):
        agents = prev.get("per_agent") or []
        gains = [a["surrogate_gain"] for a in agents if a.get("surrogate_gain") is not None]
        if not gains or "episode_return_mean" not in prev or "episode_return_mean" not in cur:
            continue
        surrogate_total = float(np.nansum(gains))
        realized = float(cur["episode_return_mean"] - prev["episode_return_mean"])
        rows.append(
            {
                "epoch": prev.get("epoch"),
                "surrogate_total": surrogate_total,
                "realized_improvement": realized,
                "gap": surrogate_total - realized,
                "n_agents": len(gains),
                "gap_per_agent": (surrogate_total - realized) / max(1, len(gains)),
            }
        )
    return rows


def h4_simultaneity(records: Sequence[dict]) -> HypothesisResult:
    rows = _simultaneity_gap(records)
    if not rows:
        return HypothesisResult(
            "H4", "simultaneity penalty grows with N", False, "single-run",
            missing_fields=["per_agent[].surrogate_gain"],
            note="requires instrument.surrogate_gain",
        )
    gaps = [r["gap"] for r in rows]
    return HypothesisResult(
        "H4",
        "surrogate gain minus realised improvement, per round",
        True,
        "single-run",
        value={
            "n_agents": rows[-1]["n_agents"],
            "gap_mean": float(np.mean(gaps)),
            "gap_ci": bootstrap_ci(gaps),
            "gap_per_agent_mean": float(np.mean([r["gap_per_agent"] for r in rows])),
            "rounds": len(rows),
        },
        note="H4's verdict is the slope of gap_mean vs N across the scale sweep",
    )


def h5_locality(records: Sequence[dict]) -> HypothesisResult:
    rows = _simultaneity_gap(records)
    probes = [r["decay_probe"] for r in records if r.get("decay_probe")]
    if not rows and not probes:
        return HypothesisResult(
            "H5", "size-independence under locality", False, "single-run",
            missing_fields=["per_agent[].surrogate_gain", "decay_probe"],
        )
    value: dict[str, Any] = {}
    if rows:
        value["gap_per_capita_mean"] = float(np.mean([r["gap_per_agent"] for r in rows]))
        value["gap_per_capita_ci"] = bootstrap_ci([r["gap_per_agent"] for r in rows])
    if probes:
        latest = probes[-1]
        fit = latest.get("fit") or {}
        value["decay_probe"] = {
            **fit,
            "graph_diameter": latest.get("graph_diameter"),
            "perturbed_agents": latest.get("perturbed_agents"),
            "pairs": len(latest.get("pairs", [])),
            # SPEC Sec. 12 / D11(a): a slope from < 4 hop levels is reported as
            # uninterpretable rather than as a number that looks like evidence.
            "verdict": (
                "uninterpretable (too few hop levels)"
                if not fit.get("interpretable", False)
                else (
                    "consistent with exponential decay"
                    if fit.get("log_slope", 0.0) < 0.0
                    else "NOT consistent with exponential decay"
                )
            ),
        }
    return HypothesisResult(
        "H5",
        "per-capita gap, plus the Assumption-2 decay probe",
        bool(rows and probes),
        "single-run",
        value=value,
        missing_fields=[] if (rows and probes) else (["decay_probe"] if rows else ["per_agent[].surrogate_gain"]),
        note="the flat-in-N part of H5 needs the scale sweep; the decay probe is per-run",
    )


# --------------------------------------------------------------------------- #
# H6 -- Byzantine tolerance
# --------------------------------------------------------------------------- #


def h6_byzantine(records: Sequence[dict], metadata: dict[str, Any] | None = None) -> HypothesisResult:
    """One (aggregation, f, attack) cell of the H6 curve.

    The reported return is the **honest agents'** return. Byzantine agents are
    sabotaging, so averaging their returns into the headline number measures the
    attack's self-harm alongside the defence's success and moves with `f` even
    when the defence is perfect.
    """
    metadata = metadata or {}
    config = metadata.get("config") or {}
    distill = config.get("distill", {})
    stress = config.get("stress", {})
    honest = _series(records, "episode_return_mean_honest")
    ret = honest or _series(records, "episode_return_mean")
    return HypothesisResult(
        "H6",
        "Byzantine tolerance: honest-agent final return vs Byzantine fraction",
        True,
        "multi-run",
        value={
            "aggregation": distill.get("aggregation"),
            "byzantine_fraction": stress.get("byzantine_fraction", 0.0),
            "byzantine_mode": stress.get("byzantine_mode", "none"),
            "byzantine_scale": stress.get("byzantine_scale"),
            "return_is_honest_only": bool(honest),
            "final_return": ret[-1] if ret else float("nan"),
            "final_return_ci_over_epochs": bootstrap_ci(ret[-5:]) if ret else {},
        },
        note=(
            "this run contributes one (aggregation, f, attack) cell; the H6 curve "
            "is the sweep over f in {0,.1,.2,.3,.4,.5} x {trimmed_mean, sum} x "
            "{sign_flip, adaptive}. The adaptive adversary is the one the "
            "prediction is worth testing against -- the loud attacks are caught "
            "by trimming on magnitude alone."
        ),
    )


# --------------------------------------------------------------------------- #
# H7 -- headline interaction (SPEC Sec. 10 / D8)
# --------------------------------------------------------------------------- #


def h7_headline(records: Sequence[dict], metadata: dict[str, Any] | None = None) -> HypothesisResult:
    metadata = metadata or {}
    env_cfg = (metadata.get("config") or {}).get("env", {})
    ret = _series(records, "episode_return_mean")
    eval_ret = _series(records, "eval/return_mean")
    zones = {
        k.split("/")[-1]: _series(records, k)
        for k in ("zone_return/normal", "zone_return/friction", "zone_return/masked")
    }
    dispersion = _series(records, "consensus_pull/policy_dispersion_p50")
    share = _series(records, "consensus_pull/social_grad_fraction_mean")
    return HypothesisResult(
        "H7",
        "headline interaction: benefit under heterogeneity, cost under homogeneity",
        True,
        "multi-run",
        value={
            "algo": metadata.get("algo"),
            "heterogeneous": env_cfg.get("heterogeneous"),
            "environment_arm": "treatment" if env_cfg.get("heterogeneous") else "control",
            "calibration_id": next(
                (r.get("calibration_id") for r in reversed(records) if r.get("calibration_id")),
                None,
            ),
            "delta_base_effective": next(
                (r.get("delta_base_effective") for r in reversed(records)
                 if r.get("delta_base_effective") is not None), None
            ),
            "final_return": ret[-1] if ret else float("nan"),
            "final_return_iqm_last5": iqm(ret[-5:]) if ret else float("nan"),
            "final_eval_return": eval_ret[-1] if eval_ret else float("nan"),
            "per_zone_final": {k: (v[-1] if v else float("nan")) for k, v in zones.items()},
            "consensus_pull_dispersion_final": dispersion[-1] if dispersion else float("nan"),
            "consensus_pull_share_mean": float(np.mean(share)) if share else float("nan"),
            "env_steps": records[-1].get("env_steps") if records else None,
            "snapshot_bytes_total": records[-1].get("snapshot_bytes_total") if records else None,
            "wall_clock_seconds": records[-1].get("time/elapsed") if records else None,
        },
        note=(
            "one arm of the H7 interaction. The verdict is the SIGN FLIP of "
            "R(D-TrOPD) - R(IPPO) between the heterogeneous and homogeneous "
            "environments (SPEC Sec. 10 / D8), across the delta_base sensitivity "
            "sweep of D7, with the consensus-pull cost of D9 reported alongside."
        ),
    )


def h7_interaction(
    heterogeneous: dict[str, Sequence[float]],
    homogeneous: dict[str, Sequence[float]],
    treatment: str = "dtropd",
    control: str = "ippo",
    seed: int = 0,
) -> dict[str, Any]:
    """H7's interaction test (SPEC Sec. 10 / D8).

    Takes ``{algo: [per-seed final returns]}`` for each environment and returns
    the per-environment difference, the interaction (difference of differences),
    and whether the predicted **sign flip** occurred. The claim is the flip, not
    either difference alone.
    """
    def paired_diff(block: dict[str, Sequence[float]]) -> dict[str, Any]:
        t = np.asarray(list(block.get(treatment, [])), dtype=np.float64)
        c = np.asarray(list(block.get(control, [])), dtype=np.float64)
        if t.size == 0 or c.size == 0:
            return {"mean": float("nan"), "ci": {}, "n": 0}
        if t.size == c.size:
            ci = bootstrap_ci(t - c, seed=seed)
            return {"mean": ci["mean"], "ci": ci, "n": int(t.size), "paired": True}
        return {
            "mean": float(t.mean() - c.mean()),
            "ci": {},
            "n": int(min(t.size, c.size)),
            "paired": False,
        }

    het, hom = paired_diff(heterogeneous), paired_diff(homogeneous)
    interaction = het["mean"] - hom["mean"]
    sign_flip = (
        np.isfinite(het["mean"])
        and np.isfinite(hom["mean"])
        and het["mean"] > 0.0
        and hom["mean"] < 0.0
    )
    return {
        "treatment": treatment,
        "control": control,
        "heterogeneous_difference": het,
        "homogeneous_difference": hom,
        "interaction": interaction,
        "predicted_sign_flip_observed": bool(sign_flip),
        "claim": (
            "D-TrOPD beats IPPO under heterogeneity while costing something "
            "under homogeneity (SPEC Sec. 10 / D8); the sign flip IS the claim"
        ),
    }


# --------------------------------------------------------------------------- #
# D9 -- consensus-pull cost (a between-run quantity)
# --------------------------------------------------------------------------- #


def consensus_pull_cost(
    control_returns: Sequence[float],
    distiller_returns: Sequence[float],
    seed: int = 0,
) -> dict[str, Any]:
    """``R(IPPO) - R(distiller)`` on the homogeneous control, with a bootstrap CI.

    SPEC Sec. 11 / D9: this is the floor the heterogeneous benefit must clear
    for the method to net out positive, so it is reported with CIs in the H7
    table rather than mentioned in prose.
    """
    control = np.asarray(list(control_returns), dtype=np.float64)
    distiller = np.asarray(list(distiller_returns), dtype=np.float64)
    if control.size == 0 or distiller.size == 0:
        return {"cost": float("nan"), "n_control": int(control.size), "n_distiller": int(distiller.size)}
    if control.size == distiller.size:
        # Paired seeds (SPEC Sec. 7) -- pair the differences rather than the means.
        diffs = control - distiller
        ci = bootstrap_ci(diffs, seed=seed)
        return {"cost": ci["mean"], "ci": ci, "paired": True, "n": int(control.size)}
    ci_c, ci_d = bootstrap_ci(control, seed=seed), bootstrap_ci(distiller, seed=seed + 1)
    return {
        "cost": ci_c["mean"] - ci_d["mean"],
        "control_ci": ci_c,
        "distiller_ci": ci_d,
        "paired": False,
    }


# --------------------------------------------------------------------------- #
# D7 -- the calibration-sensitivity decision rule, fixed before any result
# --------------------------------------------------------------------------- #

#: The three delta_base values H7 runs at: what the 6-, 8- and 10-epoch
#: calibration windows actually produced on this environment (SPEC Sec. 11 / D7).
D7_DELTA_BASE_GRID: tuple[float, ...] = (0.022, 0.026, 0.049)


def d7_sensitivity_verdict(verdicts: dict[float, bool]) -> dict[str, Any]:
    """Apply D7's pre-registered rule to per-delta_base verdicts.

    ``verdicts`` maps ``delta_base`` to that arm's H7 verdict (True = confirmed).

    The rule, fixed in SPEC Sec. 11 / D7 before any result was seen:

    * identical at all three -> "conclusion stable across a 2x range of
      delta_base"; the window choice is immaterial;
    * flips -> **that is the finding**: the benefit is calibration-sensitive,
      reported as a limitation.

    It is a function rather than a paragraph so that the rule cannot be quietly
    reinterpreted once the numbers are in, and so "do not pick the window that
    gives the nicer answer" is enforced rather than promised.
    """
    if not verdicts:
        raise ValueError("no verdicts supplied")
    missing = [d for d in D7_DELTA_BASE_GRID if d not in verdicts]
    values = set(verdicts.values())
    stable = len(values) == 1
    return {
        "grid": list(D7_DELTA_BASE_GRID),
        "verdicts": {str(k): v for k, v in sorted(verdicts.items())},
        "missing_arms": missing,
        "complete": not missing,
        "stable": stable,
        "conclusion": (
            "conclusion stable across a 2x range of delta_base; the window "
            "choice is immaterial"
            if stable
            else "the benefit is CALIBRATION-SENSITIVE: the verdict flips across "
            "the pre-registered delta_base grid. Report as a limitation; do not "
            "select the window that gives the nicer answer (SPEC Sec. 11 / D7)."
        ),
        "report_as_limitation": not stable,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

HYPOTHESES = ("H1", "H2", "H3", "H4", "H5", "H6", "H7")


def analyze_run(
    records: Sequence[dict], metadata: dict[str, Any] | None = None
) -> dict[str, HypothesisResult]:
    """Every H1-H7 measurement obtainable from one run's metrics file."""
    return {
        "H1": h1_chattering(records),
        "H2": h2_entropy_inflation(records),
        "H3": h3_radius_pathology(records),
        "H4": h4_simultaneity(records),
        "H5": h5_locality(records),
        "H6": h6_byzantine(records, metadata),
        "H7": h7_headline(records, metadata),
    }
