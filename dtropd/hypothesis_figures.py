"""Per-hypothesis figures: each hypothesis plots ITS OWN decision quantity.

The defect this exists to remove
--------------------------------

``plot_comparison`` draws final return. That is the right outcome variable for
**H7 only**. Drawing it for H1, H2, H3 or H6 produces a figure that is not wrong
so much as irrelevant -- it answers a question nobody asked, with confidence
intervals that look authoritative. This happened three times before it was
caught, which is a property of the *default*, not of the individual mistakes:
the plotter had one outcome variable and every sweep silently inherited it.

So the outcome variable is now selected by hypothesis, and a sweep with no
registered hypothesis figure gets no hypothesis figure -- rather than getting
the return plot by default and looking like it was measured.

What each hypothesis actually decides on (SPEC Sec. 6):

===  ==========================================================================
H1   per-edge social-gradient SIGN FLIPS bucketed by ``|D_ij - delta_i|``, as a
     separate series per attenuation mode. The CONTRAST between modes is the
     hypothesis, so the modes are never pooled.
H2   student ``sigma_i`` on high-``D_ij`` edges, reverse vs forward KL, against
     the predicted limits ``sigma_j`` and ``sqrt(sigma_j^2 + (mu_i-mu_j)^2)``.
H3   ``delta_i`` under a degenerate critic and under a 10x reward rescale, for
     both radius signals. The claim is INVARIANCE for nonstationarity and
     non-invariance for value-variance.
H6   final return vs Byzantine fraction ``f`` (see ``experiments/h6_curve.py``).
H7   final return -- the one hypothesis whose outcome variable it is.
===  ==========================================================================
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import bootstrap_ci
from .utils import get_logger, read_metrics

__all__ = [
    "HYPOTHESIS_FIGURES",
    "figure_for_sweep",
    "boundary_crossing_rate",
    "h1_figure",
    "h2_figure",
    "h3_figure",
]

logger = get_logger("dtropd.hypothesis_figures")


def _agg(manifest: dict[str, Any]) -> dict[str, list[Path]]:
    """arm name -> run directories, from a manifest."""
    out: dict[str, list[Path]] = defaultdict(list)
    for row in manifest["runs"]:
        out[row["arm"]].append(Path(row["run_dir"]))
    return out


def _records(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    return read_metrics(path) if path.exists() else []


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# --------------------------------------------------------------------------- #
# H1 -- sign flips vs distance to the trust boundary, per attenuation mode
# --------------------------------------------------------------------------- #


def _bucket_lower(label: str) -> float:
    """Numeric lower edge of a bucket label like ``[0.005,0.01)``.

    The previous key was ``float(s.split("_")[0]) if s[0].isdigit() else 1e9``.
    Every label starts with ``[``, so ``s[0].isdigit()`` was always False and
    EVERY bucket sorted to the same key -- leaving the x-axis in arbitrary order,
    with ``[0.05,0.1)`` drawn leftmost. H1's criterion is "hard_* spike AT THE
    LEFT", i.e. nearest the boundary, so the axis order is not cosmetic: the
    stated test could not be evaluated against the plot at all.
    """
    match = re.match(r"^[\[(]\s*([-\d.eE+]+)", label.strip())
    return float(match.group(1)) if match else float("inf")


def boundary_crossing_rate(manifest: dict[str, Any]) -> dict[str, Any]:
    """How often an edge CROSSES ``delta_i`` between consecutive epochs.

    Chattering requires crossings. If edges sit firmly on one side of the trust
    boundary for the whole run, a switched gate never switches, and H1 is
    **untestable at this horizon** rather than refuted -- a distinction the
    sign-flip plot alone cannot make, because "no flips" looks identical in both
    cases.

    Computed from stored per-edge rows (``D_ij`` and ``delta_i`` per epoch), so
    this is re-analysis of existing runs.
    """
    out: dict[str, Any] = {}
    for arm, dirs in _agg(manifest).items():
        crossings = total = 0
        for run in dirs:
            side: dict[tuple[int, int], int] = {}
            for rec in _records(run):
                for e in rec.get("edges") or []:
                    key = (int(e["i"]), int(e["j"]))
                    now = 1 if e["D_ij"] > e["delta_i"] else -1
                    if key in side:
                        total += 1
                        if now != side[key]:
                            crossings += 1
                    side[key] = now
        out[arm] = {
            "crossings": crossings,
            "edge_epoch_transitions": total,
            "rate": (crossings / total) if total else float("nan"),
        }
    rates = [v["rate"] for v in out.values() if v["edge_epoch_transitions"]]
    out["_verdict"] = (
        "NO DATA"
        if not rates
        else (
            "H1 UNTESTABLE at this horizon: edges essentially never cross the "
            "trust boundary, so a switched gate never switches. A null here is "
            "not evidence against chattering."
            if max(rates) < 0.01
            else "crossings occur; the sign-flip contrast is interpretable"
        )
    )
    return out


def h1_figure(manifest: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """Sign-flip rate by ``|D_ij - delta_i|`` bucket, one series per mode.

    H1 is confirmed if the switched variants spike NEAR THE BOUNDARY relative to
    their own far-from-boundary buckets, while the continuous one stays flat.
    That is a within-arm comparison across buckets: an arm having a higher
    overall rate is a different gate, not chattering.
    """
    # Each bucket logs {flips, edges, rate}. `rate` is the quantity to plot:
    # raw flip counts scale with how many edges land in a bucket, so a bucket
    # holding most of the edges would dominate for a reason unrelated to
    # chattering.
    per_arm: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    support: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for arm, dirs in _agg(manifest).items():
        for run in dirs:
            for rec in _records(run):
                buckets = rec.get("h1/sign_flips_by_bucket")
                if not buckets:
                    continue
                for label, cell in buckets.items():
                    if isinstance(cell, dict):
                        per_arm[arm][label].append(float(cell.get("rate", float("nan"))))
                        support[arm][label] += int(cell.get("edges", 0))
                    else:  # older runs logged a bare count
                        per_arm[arm][label].append(float(cell))
    if not per_arm:
        return {"available": False, "reason": "no h1/sign_flips_by_bucket in metrics"}

    labels = sorted({b for arm in per_arm.values() for b in arm}, key=_bucket_lower)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    summary: dict[str, Any] = {"available": True, "buckets": labels, "arms": {}}
    for arm in sorted(per_arm):
        means, los, his = [], [], []
        for b in labels:
            vals = per_arm[arm].get(b, [])
            m = float(np.mean(vals)) if vals else float("nan")
            ci = bootstrap_ci(vals) if len(vals) > 1 else {"lo": m, "hi": m}
            means.append(m)
            los.append(max(0.0, m - ci["lo"]))
            his.append(max(0.0, ci["hi"] - m))
        ax.errorbar(range(len(labels)), means, yerr=[los, his], fmt="-o", capsize=3, label=arm)
        summary["arms"][arm] = {
            "rate_by_bucket": means,
            "edges_by_bucket": [support[arm].get(b, 0) for b in labels],
        }

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_xlabel(r"$|D_{ij} - \delta_i|$  (distance to the trust boundary)")
    ax.set_ylabel("social-gradient sign-flip RATE (flips / edges in bucket)")
    ax.set_title("H1: chattering — flips near the boundary, per attenuation mode\n"
                 "confirmed if hard_* spike at the left and continuous stays flat")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    crossings = boundary_crossing_rate(manifest)
    summary["boundary_crossings"] = crossings
    verdict = crossings.get("_verdict", "")
    fig.text(0.5, 0.005, f"boundary-crossing check: {verdict}", ha="center",
             fontsize=8, color="tab:red" if "UNTESTABLE" in verdict else "black")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return summary


# --------------------------------------------------------------------------- #
# H2 -- student sigma on high-D_ij edges, against the predicted limits
# --------------------------------------------------------------------------- #


def _mean_gap_sq(d_ij: float, sigma_i: float, sigma_j: float, act_dim: int) -> float:
    """Recover ``(mu_i - mu_j)^2`` per dimension from the logged reverse KL.

    For a diagonal Gaussian, per dimension,
    ``KL = log(s_j/s_i) + (s_i^2 + gap^2) / (2 s_j^2) - 0.5``. The logged
    ``sigma_i``/``sigma_j`` are means over dimensions, so inverting gives the
    mean-squared gap under the approximation that the per-dimension scales are
    similar. Reconstructed, not measured -- labelled as such wherever plotted,
    and it exists so H2's predicted limit can be drawn WITHOUT re-running to log
    one more field.
    """
    if sigma_j <= 0 or sigma_i <= 0 or act_dim <= 0:
        return float("nan")
    per_dim = d_ij / act_dim
    gap_sq = 2.0 * sigma_j**2 * (per_dim + 0.5 - math.log(sigma_j / sigma_i)) - sigma_i**2
    return max(0.0, gap_sq)


def h2_figure(manifest: dict[str, Any], out_path: Path, top_quantile: float = 0.75) -> dict[str, Any]:
    """``sigma_i`` on the high-``D_ij`` edges, reverse vs forward KL.

    H2 predicts forward KL is mode-covering and drives
    ``sigma_i -> sqrt(sigma_j^2 + (mu_i-mu_j)^2)``, while reverse KL is
    mode-seeking and holds at ``sigma_j``. Both limits are drawn.
    """
    series: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    targets: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for arm, dirs in _agg(manifest).items():
        for run in dirs:
            meta_path = run / "metadata.json"
            act_dim = 2
            if meta_path.exists():
                try:
                    act_dim = int(json.loads(meta_path.read_text())["config"]["env"].get("act_dim", 2))
                except Exception:
                    act_dim = 2
            for rec in _records(run):
                edges = rec.get("edges")
                if not edges:
                    continue
                ds = np.array([e["D_ij"] for e in edges], dtype=float)
                if ds.size == 0:
                    continue
                cut = float(np.quantile(ds, top_quantile))
                hi = [e for e in edges if e["D_ij"] >= cut]
                if not hi:
                    continue
                epoch = int(rec["epoch"])
                series[arm][epoch].extend(float(e["sigma_i"]) for e in hi)
                for e in hi:
                    gap = _mean_gap_sq(e["D_ij"], e["sigma_i"], e["sigma_j"], act_dim)
                    targets[arm][epoch].append(
                        (float(e["sigma_j"]), math.sqrt(e["sigma_j"] ** 2 + gap))
                    )
    if not series:
        return {"available": False, "reason": "no per-edge rows (instrument.per_edge_detail)"}

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = {}
    summary: dict[str, Any] = {"available": True, "top_quantile": top_quantile, "arms": {}}
    for arm in sorted(series):
        epochs = sorted(series[arm])
        vals = [float(np.mean(series[arm][e])) for e in epochs]
        line, = ax.plot(epochs, vals, "-o", ms=3, label=f"{arm}: student $\\sigma_i$")
        colours[arm] = line.get_color()
        summary["arms"][arm] = {"epochs": epochs, "sigma_i": vals, "final_sigma_i": vals[-1]}
    # Predicted limits, drawn once from the reverse arm's edges.
    ref = next((a for a in sorted(targets) if "forward" not in a), sorted(targets)[0])
    epochs = sorted(targets[ref])
    lo = [float(np.mean([t[0] for t in targets[ref][e]])) for e in epochs]
    hi = [float(np.mean([t[1] for t in targets[ref][e]])) for e in epochs]
    ax.plot(epochs, lo, "k--", lw=1.2, label=r"reverse-KL limit  $\sigma_j$")
    ax.plot(epochs, hi, "k:", lw=1.5,
            label=r"forward-KL limit  $\sqrt{\sigma_j^2+(\mu_i-\mu_j)^2}$ (reconstructed)")
    summary["predicted_reverse_limit"] = lo
    summary["predicted_forward_limit"] = hi

    ax.set_xlabel("epoch")
    ax.set_ylabel(r"student $\sigma_i$ on high-$D_{ij}$ edges")
    ax.set_title("H2: KL direction and entropy inflation\n"
                 f"edges above the {int(top_quantile*100)}th percentile of $D_{{ij}}$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return summary


# --------------------------------------------------------------------------- #
# H3 -- delta_i invariance, per radius signal
# --------------------------------------------------------------------------- #


def h3_figure(manifest: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """``delta_i`` per condition, both radius signals.

    The two claims are (1) scale INVARIANCE under a 10x reward rescale for the
    nonstationarity radius, and (2) CONTRACTION under a degenerate critic. The
    value-variance radius is expected to fail both -- it is the I1 counterexample.

    Final return is deliberately NOT plotted here. Under a 10x reward rescale the
    return shifts by ~10x as a UNIT CHANGE, which looks like a catastrophic
    result and is nothing of the kind.
    """
    per_arm: dict[str, list[float]] = defaultdict(list)
    traj: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for arm, dirs in _agg(manifest).items():
        for run in dirs:
            for rec in _records(run):
                if "delta_mean" not in rec:
                    continue
                traj[arm][int(rec["epoch"])].append(float(rec["delta_mean"]))
            recs = [r for r in _records(run) if "delta_mean" in r]
            if recs:
                per_arm[arm].append(float(np.mean([r["delta_mean"] for r in recs[-10:]])))
    if not per_arm:
        return {"available": False, "reason": "no delta_mean in metrics"}

    plt = _mpl()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for arm in sorted(traj):
        epochs = sorted(traj[arm])
        ax.plot(epochs, [float(np.mean(traj[arm][e])) for e in epochs], "-", label=arm)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\delta_i$  (trust radius)")
    ax.set_title(r"H3: $\delta_i$ trajectory per condition")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    arms = sorted(per_arm)
    means = [float(np.mean(per_arm[a])) for a in arms]
    cis = [bootstrap_ci(per_arm[a]) if len(per_arm[a]) > 1 else {"lo": m, "hi": m}
           for a, m in zip(arms, means)]
    ax2.errorbar(range(len(arms)), means,
                 yerr=[[max(0.0, m - c["lo"]) for m, c in zip(means, cis)],
                       [max(0.0, c["hi"] - m) for m, c in zip(means, cis)]],
                 fmt="o", capsize=4)
    ax2.set_xticks(range(len(arms)))
    ax2.set_xticklabels(arms, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel(r"final $\delta_i$")
    ax2.set_title("H3: invariance check\n"
                  "nonstationarity should be FLAT across 1x/10x; value_variance should not")
    ax2.grid(alpha=0.3, axis="y")

    ratios: dict[str, float] = {}
    for signal in ("nonstat", "valvar"):
        base = next((a for a in arms if a.startswith(signal) and "healthy" in a), None)
        for other, tag in ((f"{signal}_reward10x", "scale_invariance"),
                           (f"{signal}_degenerate", "contraction")):
            if base and other in per_arm:
                ratios[f"{signal}:{tag}"] = float(np.mean(per_arm[other]) / np.mean(per_arm[base]))

    fig.suptitle("H3 decides on $\\delta_i$, NOT on return "
                 "(a 10x reward rescale shifts return by ~10x as a unit change)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"available": True, "final_delta_by_arm": {a: m for a, m in zip(arms, means)},
            "ratios_vs_healthy": ratios,
            "expected": {"nonstat:scale_invariance": "~1.0", "nonstat:contraction": "<1.0",
                         "valvar:scale_invariance": "far from 1.0"}}


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

#: sweep name prefix -> (figure function, filename). A sweep with no entry gets
#: NO hypothesis figure, rather than silently inheriting the return plot.
HYPOTHESIS_FIGURES = {
    "h1": (h1_figure, "h1_sign_flips.png"),
    "h2": (h2_figure, "h2_sigma.png"),
    "h3": (h3_figure, "h3_delta.png"),
}

#: The only hypothesis whose decision quantity IS final return.
RETURN_IS_THE_OUTCOME = ("h7",)


def figure_for_sweep(sweep_name: str, manifest: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Draw the hypothesis figure for this sweep, if one is registered."""
    for prefix, (fn, filename) in HYPOTHESIS_FIGURES.items():
        if sweep_name.startswith(prefix):
            path = out_dir / filename
            try:
                summary = fn(manifest, path)
            except Exception as exc:  # pragma: no cover - plotting is best-effort
                logger.error("could not draw the %s figure: %s", prefix, exc)
                return {"available": False, "reason": str(exc)}
            if summary.get("available"):
                logger.info("wrote %s", path)
                summary["figure"] = str(path)
            else:
                logger.warning(
                    "%s figure not drawn: %s", prefix, summary.get("reason")
                )
            return summary
    return {}
