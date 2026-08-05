"""Figures from stored metrics.

SPEC Sec. 7: "Every figure regenerable from stored metrics by one command, no
manual steps."  Phase 6 adds the ``rliable`` IQM / bootstrap-CI comparison
figures; what exists today is the learning-curve plot used to eyeball Phase 2
runs, driven entirely by ``metrics.jsonl`` files on disk.

``matplotlib`` is imported lazily so that the core package stays importable in
a minimal environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .utils import read_metrics

__all__ = ["load_runs", "learning_curves", "plot_learning_curves"]


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
