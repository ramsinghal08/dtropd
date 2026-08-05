"""Phase 0/2 smoke tests: the CPU debug path always works (SPEC Sec. 9, rule 6)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dtropd.config import Config
from dtropd.train import train
from dtropd.utils import read_metrics

ROOT = Path(__file__).resolve().parent.parent


def _smoke_config(tmp_path: Path, steps: int = 200) -> Config:
    cfg = Config()
    cfg.env.n_agents = 2
    cfg.env.n_targets = 1
    cfg.env.capture_group_size = 2
    cfg.env.max_cycles = 25
    cfg.train.algo = "ippo"
    cfg.train.num_envs = 2
    cfg.train.total_epochs = max(1, steps // (25 * 2))
    cfg.train.eval_interval = 0
    cfg.train.eval_episodes = 2
    cfg.train.log_dir = str(tmp_path)
    cfg.ppo.num_minibatches = 2
    cfg.ppo.inner_epochs = 2
    cfg.validate(emit=False)
    return cfg


def test_two_agent_200_step_cpu_run_completes(tmp_path):
    """A 2-agent, >=200-step CPU run completes well inside two minutes."""
    cfg = _smoke_config(tmp_path, steps=200)
    start = time.time()
    result = train(cfg, seed=0, out_dir=tmp_path / "smoke", progress=False)
    elapsed = time.time() - start

    assert elapsed < 120, f"smoke run took {elapsed:.1f}s (budget: 120s)"
    assert result.epochs * cfg.batch_size >= 200

    records = read_metrics(result.run_dir / "metrics.jsonl")
    assert len(records) == cfg.train.total_epochs
    assert (result.run_dir / "metadata.json").exists()
    assert (result.run_dir / "checkpoint_final.pt").exists()

    meta = json.loads((result.run_dir / "metadata.json").read_text())
    assert meta["seed"] == 0
    assert "git_commit" in meta and "config" in meta

    last = records[-1]
    for key in ("episode_return_mean", "explained_variance", "update_kl", "value_loss"):
        assert key in last, f"missing metric {key}"


def test_run_py_help_exits_zero():
    """Phase 0 acceptance: ``python run.py --help`` works."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "D-TrOPD" in proc.stdout
    for command in ("train", "eval", "plot", "validate", "smoke"):
        assert command in proc.stdout


def test_run_py_validate_default_config():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "validate"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "validation: OK" in proc.stdout


def test_unimplemented_algorithms_fail_loudly():
    from dtropd.algorithms import build_algorithm

    cfg = Config()
    cfg.train.algo = "mappo"
    with pytest.raises(NotImplementedError, match="Phase 4"):
        build_algorithm(cfg, seed=0)
    cfg.train.algo = "nope"
    with pytest.raises(ValueError):
        build_algorithm(cfg, seed=0)
