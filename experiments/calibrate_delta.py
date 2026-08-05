#!/usr/bin/env python3
"""Measure the empirical ``D_ij`` distribution and suggest ``delta_base``/``gamma``.

Why this exists: ``delta_base`` and ``gamma_attenuation`` are the only D-TrOPD
constants SPEC.md does not pin down, and a mis-set ``delta_base`` silently
turns the whole mechanism off.  If ``delta_base`` sits far above the observed
social divergences, then ``w_ij ~ 1`` for every edge and D-TrOPD *is*
NaiveDistill -- with no error, no warning, and a plausible-looking learning
curve.  This script measures the distribution instead of guessing at it.

It reports percentiles of ``D_ij`` over the first ``--epochs`` epochs of a real
run and suggests:

* ``delta_base`` = the ``--quantile`` (default 0.5) of observed ``D_ij``, so
  roughly half the edges start inside the radius and half outside -- the regime
  where the attenuation has something to do;
* ``gamma`` such that an edge at the 95th percentile of ``D_ij`` is attenuated
  to ``--target-w`` (default 0.5).

Usage::

    python experiments/calibrate_delta.py \
        --config experiments/configs/dtropd_heterogeneous.yaml --epochs 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtropd.algorithms import build_algorithm, trust_weight  # noqa: E402
from dtropd.config import Config  # noqa: E402
from dtropd.env import SyncVectorEnv  # noqa: E402
from dtropd.utils import configure_logging, derive_seed, get_logger  # noqa: E402

logger = get_logger("calibrate")


def collect_divergences(cfg: Config, seed: int, epochs: int) -> tuple[np.ndarray, np.ndarray]:
    algo = build_algorithm(cfg, seed=seed, device=cfg.train.device)
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=derive_seed(seed, "env"))
    divergences, radii = [], []
    for epoch in range(epochs):
        algo.epoch = epoch
        algo.on_epoch_start()
        algo.collect_rollout(vec)
        metrics = algo.update()
        for row in metrics.get("per_edge", {}).values():
            divergences.extend(row["D_ij"])
            radii.append(row["delta_i"])
        logger.info(
            "epoch %d: D_ij median %.4f, delta %.4f, w mean %.4f",
            epoch,
            float(np.median(divergences[-64:])) if divergences else float("nan"),
            metrics.get("delta_mean", float("nan")),
            metrics.get("trust_weight_mean", float("nan")),
        )
    return np.asarray(divergences), np.asarray(radii)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "experiments/configs/dtropd_heterogeneous.yaml"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--quantile", type=float, default=0.5)
    p.add_argument("--target-w", type=float, default=0.5)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    configure_logging("INFO")
    cfg = Config.from_yaml(args.config)
    cfg.validate(emit=False)

    d, radii = collect_divergences(cfg, args.seed, args.epochs)
    if d.size == 0:
        raise SystemExit("no edges: the communication graph is empty")

    qs = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    percentiles = {f"p{int(q * 100)}": float(np.quantile(d, q)) for q in qs}

    suggested_delta = float(np.quantile(d, args.quantile))
    d95 = percentiles["p95"]
    # w = exp(-gamma * [softplus(D - delta) - softplus(-delta)]) = target_w
    excess = math.log1p(math.exp(d95 - suggested_delta)) - math.log1p(
        math.exp(-suggested_delta)
    )
    suggested_gamma = float(-math.log(args.target_w) / excess) if excess > 0 else float("nan")

    current_w = float(
        trust_weight(
            __import__("torch").tensor(percentiles["p50"]),
            float(np.mean(radii)),
            cfg.distill.gamma_attenuation,
            cfg.distill.attenuation,
        )
    )

    report = {
        "config": args.config,
        "epochs": args.epochs,
        "edges_sampled": int(d.size),
        "D_ij_percentiles": percentiles,
        "D_ij_mean": float(d.mean()),
        "delta_i_mean_observed": float(radii.mean()),
        "current": {
            "delta_base": cfg.distill.delta_base,
            "gamma_attenuation": cfg.distill.gamma_attenuation,
            "w_at_median_D": current_w,
            "attenuation_is_effectively_off": bool(current_w > 0.95),
        },
        "suggested": {
            "delta_base": suggested_delta,
            "gamma_attenuation": suggested_gamma,
            "rationale": (
                f"delta_base = q{args.quantile:g} of observed D_ij so the radius sits "
                f"inside the data; gamma chosen so an edge at the 95th percentile "
                f"(D = {d95:.4f}) is attenuated to w = {args.target_w:g}."
            ),
        },
    }
    if report["current"]["attenuation_is_effectively_off"]:
        logger.warning(
            "w = %.4f at the median D_ij: the trust gate is nearly inert at these "
            "settings, so this configuration is behaviourally NaiveDistill.",
            current_w,
        )
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
