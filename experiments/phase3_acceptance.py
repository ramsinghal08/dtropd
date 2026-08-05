#!/usr/bin/env python3
"""Phase 3 acceptance test (SPEC Sec. 5).

    "Accept (critical): with ``beta = 0``, D-TrOPD reproduces IPPO
     **bitwise-identically** under the same seed. If curves differ at all, the
     social path is leaking. Also: ``test_invariants.py`` passes I1-I9;
     ``test_attenuation.py`` confirms ``w(0)=1`` and derivative continuity at
     ``D = delta``."

Three criteria, checked in that order:

  A1  for every seed, every parameter of every agent after N epochs of
      D-TrOPD with ``beta_max = 0`` is bit-identical (``torch.equal``) to IPPO's,
      and every logged scalar curve matches exactly -- while the social path
      demonstrably ran (non-trivial D_ij and w_ij were computed and logged);
  A2  ``pytest tests/test_invariants.py`` is green (I1-I9);
  A3  ``pytest tests/test_attenuation.py`` is green (w(0)=1, monotone, C^1).

Usage::

    python experiments/phase3_acceptance.py --seeds 0 1 2 --epochs 15
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtropd.algorithms import build_algorithm  # noqa: E402
from dtropd.config import Config  # noqa: E402
from dtropd.env import SyncVectorEnv  # noqa: E402
from dtropd.utils import configure_logging, derive_seed, get_logger  # noqa: E402

logger = get_logger("phase3")

CURVE_KEYS = (
    "episode_return_mean",
    "explained_variance",
    "update_kl",
    "value_loss",
    "policy_loss",
    "entropy",
    "clip_fraction",
    "inner_epochs_used",
)


def _run(cfg: Config, seed: int, epochs: int) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    algo = build_algorithm(cfg, seed=seed, device=cfg.train.device)
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=derive_seed(seed, "env"))
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        algo.epoch = epoch
        algo.on_epoch_start()
        rollout = algo.collect_rollout(vec)
        metrics = algo.update()
        history.append(
            {**rollout, **{k: v for k, v in metrics.items() if k not in ("per_agent", "per_edge")}}
        )
    params = [p.detach().clone() for ac in algo.agents for p in ac.parameters()]
    return params, history


def _base_config(path: str | None, epochs: int) -> Config:
    cfg = Config.from_yaml(path) if path else Config()
    if path is None:
        cfg.env.n_agents = 6
        cfg.env.n_targets = 2
        cfg.env.capture_group_size = 3
        cfg.env.max_cycles = 25
        cfg.train.num_envs = 8
        cfg.network.hidden_sizes = [32, 32]
    cfg.train.total_epochs = epochs
    cfg.train.device = "cpu"
    return cfg


def check_bitwise(cfg_path: str | None, seeds: list[int], epochs: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        ippo_cfg = _base_config(cfg_path, epochs)
        ippo_cfg.train.algo = "ippo"
        ippo_cfg.distill.enabled = False
        ippo_cfg.validate(emit=False)

        dtropd_cfg = _base_config(cfg_path, epochs)
        dtropd_cfg.train.algo = "dtropd"
        dtropd_cfg.distill.enabled = True
        dtropd_cfg.distill.beta_max = 0.0
        dtropd_cfg.validate(emit=False)

        logger.info("[seed %d] running IPPO for %d epochs", seed, epochs)
        ippo_params, ippo_hist = _run(ippo_cfg, seed, epochs)
        logger.info("[seed %d] running D-TrOPD (beta_max=0) for %d epochs", seed, epochs)
        dt_params, dt_hist = _run(dtropd_cfg, seed, epochs)

        mismatched = [
            k for k, (a, b) in enumerate(zip(ippo_params, dt_params)) if not torch.equal(a, b)
        ]
        max_abs_diff = max(
            (float((a - b).abs().max()) for a, b in zip(ippo_params, dt_params)), default=0.0
        )
        curve_mismatch = {
            key: (
                [h.get(key) for h in ippo_hist],
                [h.get(key) for h in dt_hist],
            )
            for key in CURVE_KEYS
            if [h.get(key) for h in ippo_hist] != [h.get(key) for h in dt_hist]
        }

        last = dt_hist[-1]
        social_ran = bool(
            last.get("snapshot_bytes", 0) > 0
            and last.get("social_divergence_max", 0.0) > 0.0
            and 0.0 < last.get("trust_weight_mean", 0.0) <= 1.0
        )
        rows.append(
            {
                "seed": seed,
                "epochs": epochs,
                "n_tensors": len(ippo_params),
                "mismatched_tensors": mismatched,
                "max_abs_param_diff": max_abs_diff,
                "curve_mismatch_keys": sorted(curve_mismatch),
                "social_path_ran": social_ran,
                "final_beta": last.get("beta"),
                "final_delta_mean": last.get("delta_mean"),
                "final_social_divergence_mean": last.get("social_divergence_mean"),
                "final_trust_weight_mean": last.get("trust_weight_mean"),
                "snapshot_bytes_total": last.get("snapshot_bytes_total"),
                "passed": not mismatched and not curve_mismatch and social_ran,
            }
        )
        logger.info(
            "[seed %d] bitwise: %s (max |diff| = %.3g, social path ran: %s)",
            seed,
            "PASS" if rows[-1]["passed"] else "FAIL",
            max_abs_diff,
            social_ran,
        )
    return {"per_seed": rows, "passed": all(r["passed"] for r in rows)}


def check_pytest(target: str) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:]
    return {"target": target, "returncode": proc.returncode, "summary": tail, "passed": proc.returncode == 0}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML config (default: a small CPU config)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--out", default="runs/phase3")
    args = p.parse_args(argv)

    configure_logging("INFO")
    a1 = check_bitwise(args.config, args.seeds, args.epochs)
    a2 = check_pytest("tests/test_invariants.py")
    a3 = check_pytest("tests/test_attenuation.py")

    report = {
        "acceptance": "Phase 3 -- D-TrOPD core",
        "criteria": {
            "A1_beta_zero_reproduces_ippo_bitwise": a1,
            "A2_invariants_I1_to_I9": a2,
            "A3_attenuation_w0_and_c1": a3,
        },
        "passed": bool(a1["passed"] and a2["passed"] and a3["passed"]),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase3_acceptance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nPHASE 3 ACCEPTANCE:", "PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
