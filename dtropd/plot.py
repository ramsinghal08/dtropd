"""Figures from stored metrics.

SPEC Sec. 7: "Every figure regenerable from stored metrics by one command, no
manual steps."  The sweep harness adds the ``rliable`` IQM / bootstrap-CI comparison
figures; this module also provides the learning-curve plot for individual
runs, driven entirely by ``metrics.jsonl`` files on disk.

``matplotlib`` is imported lazily so that the core package stays importable in
a minimal environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .utils import read_metrics

__all__ = [
    "load_runs",
    "learning_curves",
    "plot_learning_curves",
    "plot_comparison",
]


def load_runs(run_dirs: Iterable[str | Path]) -> dict[str, list[dict[str, Any]]]:
    """Map ``run_dir -> metrics records``."""
    out: dict[str, list[dict[str, Any]]] = {}
    for d in run_dirs:
        path = Path(d)
        metrics = path / "metrics.jsonl" if path.is_dir() else path
        if metrics.exists():
            out[str(path)] = read_metrics(metrics)
    return out


def learning_curves(
    runs: dict[str, list[dict[str, Any]]],
    key: str = "episode_return_mean",
    x_key: str = "env_steps",
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-seed curves onto a common x grid (truncated to the shortest)."""
    xs, ys = [], []
    for records in runs.values():
        xs.append([r[x_key] for r in records if key in r])
        ys.append([r[key] for r in records if key in r])
    if not ys:
        return np.zeros((0, 0)), np.zeros((0, 0))
    n = min(len(y) for y in ys)
    return (
        np.asarray([x[:n] for x in xs], dtype=np.float64),
        np.asarray([y[:n] for y in ys], dtype=np.float64),
    )


def plot_learning_curves(
    run_dirs: Sequence[str | Path],
    out_path: str | Path,
    key: str = "episode_return_mean",
    x_key: str = "env_steps",
    label: str | None = None,
    title: str | None = None,
) -> Path:
    """Mean +/- s.e.m. learning curve over seeds; returns the written path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_runs(run_dirs)
    x, y = learning_curves(runs, key=key, x_key=x_key)
    if y.size == 0:
        raise ValueError(f"no records with key {key!r} found in {list(run_dirs)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    mean = y.mean(axis=0)
    sem = y.std(axis=0) / max(1.0, np.sqrt(y.shape[0]))
    ax.plot(x[0], mean, label=label or key)
    ax.fill_between(x[0], mean - sem, mean + sem, alpha=0.25)
    ax.set_xlabel(x_key)
    ax.set_ylabel(key)
    ax.set_title(title or f"{key} ({y.shape[0]} seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# The main comparison figure (SPEC Sec. 5, Sec. 7)
# --------------------------------------------------------------------------- #


def plot_comparison(
    report: dict[str, Any],
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """IQM with 95% bootstrap CIs per arm, plus the D8 interaction panel.

    ``report`` is the output of :func:`dtropd.sweep.sweep_report`, which is
    computed entirely from stored metrics -- so this figure is regenerable by
    one command with no manual steps (SPEC Sec. 7).

    rliable's headline estimator is the interquartile mean with stratified
    bootstrap CIs; those are computed in ``dtropd.analysis`` (``iqm``,
    ``bootstrap_ci``) so the figure has no dependency the core package lacks.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_arm = report.get("per_arm", {})
    if not per_arm:
        raise ValueError("report has no per_arm block: nothing to plot")

    conditions = sorted(per_arm)
    interaction = report.get("h7_interaction")
    ncols = 2 if interaction else 1
    fig, axes = plt.subplots(
        1, ncols, figsize=(6 * ncols, 4.2), squeeze=False
    )

    ax = axes[0][0]
    algos = sorted({a for c in conditions for a in per_arm[c]})
    width = 0.8 / max(1, len(conditions))
    for ci, condition in enumerate(conditions):
        xs, ys, lo, hi = [], [], [], []
        for ai, algo in enumerate(algos):
            block = per_arm[condition].get(algo)
            if not block:
                continue
            xs.append(ai + ci * width - 0.4 + width / 2)
            ys.append(block["iqm"])
            lo.append(block["iqm"] - block["ci"]["lo"])
            hi.append(block["ci"]["hi"] - block["iqm"])
        if not xs:
            continue
        n = per_arm[condition][algos[0]]["n_seeds"] if algos[0] in per_arm[condition] else 0
        ax.errorbar(
            xs, ys, yerr=[lo, hi], fmt="o", capsize=4, label=f"{condition} (n={n})"
        )
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos, rotation=15)
    ax.set_ylabel("final return -- IQM, 95% bootstrap CI")
    ax.set_title(title or "Algorithm comparison")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    if interaction:
        ax2 = axes[0][1]
        labels, values, errs = [], [], []
        for key, label in (
            ("heterogeneous_difference", "heterogeneous"),
            ("homogeneous_difference", "homogeneous (control)"),
        ):
            block = interaction.get(key, {})
            if not np.isfinite(block.get("mean", float("nan"))):
                continue
            labels.append(label)
            values.append(block["mean"])
            ci = block.get("ci") or {}
            errs.append(
                [block["mean"] - ci.get("lo", block["mean"]),
                 ci.get("hi", block["mean"]) - block["mean"]]
            )
        if labels:
            err = np.asarray(errs).T if errs else None
            ax2.bar(labels, values, color=["tab:blue", "tab:orange"][: len(labels)])
            if err is not None:
                ax2.errorbar(labels, values, yerr=err, fmt="none", ecolor="k", capsize=5)
        ax2.axhline(0.0, color="k", lw=1)
        flip = interaction.get("predicted_sign_flip_observed")
        ax2.set_ylabel(
            f"{interaction['treatment']} - {interaction['control']} (final return)"
        )
        ax2.set_title(
            "H7 interaction: predicted sign flip "
            + ("OBSERVED" if flip else "not observed")
        )
        ax2.grid(alpha=0.3, axis="y")

    # SPEC Sec. 12 / D13: an environment-property measurement that spans reward
    # spec versions is allowed, but must be "labelled as cross-version wherever
    # it appears" -- and a figure is where someone is most likely to read it
    # without the report open beside them.
    prop = report.get("environment_property") or {}
    if prop.get("cross_version"):
        versions = "/".join(prop.get("reward_spec_versions", []))
        fig.text(
            0.5,
            0.005,
            f"CROSS-VERSION ({versions}): environment-property measurement, "
            "NOT an algorithm comparison (SPEC Sec. 12 / D13)",
            ha="center",
            va="bottom",
            fontsize=8,
            color="tab:red",
            weight="bold",
        )

    fig.tight_layout(rect=(0, 0.035, 1, 1) if prop.get("cross_version") else None)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
