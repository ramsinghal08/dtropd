"""Seeding, logging and run-metadata helpers.

Every random source in the project is derived from a single master seed via
:func:`derive_seed`, so a run is reproducible from ``(config, seed)`` alone.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

__all__ = [
    "derive_seed",
    "seed_everything",
    "make_rng",
    "git_commit",
    "run_metadata",
    "MetricsWriter",
    "RunningMeanStd",
    "configure_logging",
    "get_logger",
    "explained_variance",
]


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #


def derive_seed(master_seed: int, *tags: Any) -> int:
    """Deterministically derive a sub-seed from a master seed and string tags.

    Uses BLAKE2b rather than :func:`hash` so the mapping is stable across
    processes and Python versions.
    """
    payload = "|".join([str(master_seed), *(str(t) for t in tags)]).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def seed_everything(master_seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed python, numpy and torch from one master seed."""
    random.seed(derive_seed(master_seed, "python"))
    np.random.seed(derive_seed(master_seed, "numpy"))
    torch.manual_seed(derive_seed(master_seed, "torch"))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(derive_seed(master_seed, "torch", "cuda"))
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_rng(master_seed: int, *tags: Any) -> np.random.Generator:
    """A dedicated numpy Generator for one component."""
    return np.random.default_rng(derive_seed(master_seed, *tags))


# --------------------------------------------------------------------------- #
# Run metadata
# --------------------------------------------------------------------------- #


def git_commit(repo_root: str | Path | None = None) -> str:
    """Current git commit (with a ``-dirty`` suffix), or ``"unknown"``."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.call(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"{commit}-dirty" if dirty else commit
    except Exception:  # pragma: no cover - git is not guaranteed to exist
        return "unknown"


def run_metadata(config: Any, seed: int, **extra: Any) -> dict[str, Any]:
    """The JSON metadata header written next to every metrics file."""
    cfg_dict = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else config
    meta: dict[str, Any] = {
        "config": cfg_dict,
        "seed": seed,
        "git_commit": git_commit(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "argv": list(sys.argv),
    }
    meta.update(extra)
    return meta


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


class MetricsWriter:
    """Append-only JSONL metrics writer with a JSON metadata sidecar.

    ``<out_dir>/metadata.json``  -- config + git commit + seed (SPEC Sec. 4)
    ``<out_dir>/metrics.jsonl``  -- one JSON object per logged epoch
    """

    def __init__(self, out_dir: str | Path, metadata: dict[str, Any] | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.jsonl"
        self.metadata_path = self.out_dir / "metadata.json"
        if metadata is not None:
            self.metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_default))
        self._fh = self.metrics_path.open("a", buffering=1)

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=_json_default) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return str(obj)


def read_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL metrics file back into a list of records."""
    records = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level if isinstance(level, int) else level.upper(),
        format=_LOG_FORMAT,
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str = "dtropd") -> logging.Logger:
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #


class RunningMeanStd:
    """Welford running mean/variance, used to normalise critic targets.

    Value-target normalisation is an optimisation detail, not an algorithmic
    one: the critic learns ``(G - mu) / sigma`` and every value it reports is
    de-normalised back into reward units, so advantages, GAE, explained
    variance and the trust-radius signal are all unchanged.  It matters because
    returns here live around -30 with an untrained critic initialised near 0.
    """

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        batch_mean, batch_var, batch_count = float(x.mean()), float(x.var()), float(x.size)
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta**2 * self.count * batch_count / tot) / tot
        self.mean, self.var, self.count = new_mean, max(new_var, 1e-8), tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.mean, self.var, self.count = state["mean"], state["var"], state["count"]


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """1 - Var(y_true - y_pred) / Var(y_true); invariant I4's headline number."""
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    var_true = y_true.var()
    if var_true < 1e-12:
        return float("nan")
    return float(1.0 - (y_true - y_pred).var() / var_true)


def set_torch_threads(n: int) -> None:
    try:
        torch.set_num_threads(max(1, int(n)))
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, int(n))))
    except Exception:  # pragma: no cover
        pass


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else float("nan")
