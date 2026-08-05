#!/usr/bin/env python3
"""Can a shortened budget express H6's effect at all? (SPEC Sec. 15 / D16)

H6's result is the **shape of a curve over the Byzantine fraction `f`**. That
makes a short budget dangerous in a specific way: if training stops before the
attack has time to bite, every cell returns roughly the same number, the curve
comes out flat, and a flat curve is **indistinguishable** between

    "robust aggregation held"        (the hypothesis confirmed)
    "nothing had time to happen"     (the budget was too short)

Those are opposite conclusions with identical plots. So D16 forbids trimming the
budget until the trimmed budget is shown to be able to express the effect --
and the cheapest sufficient test is the two endpoints of the `f` axis.

    Run `f = 0` and `f = 0.5` at the proposed budget, plain-sum aggregation
    (the arm H6 predicts collapses), under the adaptive adversary. Their final
    returns must separate with **non-overlapping bootstrap CIs**.

`f = 0.5` is the right upper endpoint because it is the strongest cell in the
sweep: half the neighbourhood Byzantine, no honest majority. If the effect is
not visible *there*, it will not be visible at `f = 0.2`, and the interior of
the curve carries no information the endpoints did not already deny.

This is a **precondition, not a result.** Passing it does not support H6; it
only establishes that the budget is capable of showing something. Failing it
means the budget must grow, not that the defence works.

Usage::

    python experiments/h6_budget_check.py --epochs 60 --num-envs 20 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtropd.algorithms import build_algorithm  # noqa: E402
from dtropd.analysis import bootstrap_ci  # noqa: E402
from dtropd.config import load_config  # noqa: E402
from dtropd.env import SyncVectorEnv  # noqa: E402
from dtropd.utils import configure_logging, derive_seed, get_logger  # noqa: E402

logger = get_logger("h6_budget")


def run_cell(fraction: float, seed: int, epochs: int, num_envs: int) -> float:
    """One cell; returns the honest agents' final mean episode return."""
    cfg = load_config(
        str(ROOT / "experiments/configs/h6_byzantine.yaml".replace("h6_byzantine", "h7_env_het")),
        validate=False,
    )
    cfg.train.total_epochs = epochs
    cfg.train.num_envs = num_envs
    cfg.train.algo = "dtropd"
    cfg.distill.enabled = True
    cfg.distill.aggregation = "sum"      # the arm H6 predicts collapses
    cfg.distill.calibration = "warmup"
    cfg.stress.byzantine_mode = "adaptive"
    cfg.stress.byzantine_fraction = fraction
    cfg.validate(emit=False)

    algo = build_algorithm(cfg, seed=seed, device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=derive_seed(seed, "env"))
    last = float("nan")
    for epoch in range(epochs):
        algo.epoch = epoch
        algo.on_epoch_start()
        stats = algo.collect_rollout(vec)
        algo.update()
        last = stats.get("episode_return_mean_honest", stats["episode_return_mean"])
    return float(last)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, required=True, help="proposed trimmed epoch budget")
    p.add_argument("--num-envs", type=int, default=20, help="proposed trimmed env count")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="runs/h6_budget_check.json")
    args = p.parse_args(argv)

    configure_logging("INFO")
    started = time.time()
    results: dict[str, list[float]] = {}
    for fraction in (0.0, 0.5):
        key = f"f={fraction}"
        results[key] = []
        for seed in range(args.seeds):
            logger.info("%s seed %d", key, seed)
            results[key].append(run_cell(fraction, seed, args.epochs, args.num_envs))

    lo, hi = results["f=0.0"], results["f=0.5"]
    ci_lo, ci_hi = bootstrap_ci(lo), bootstrap_ci(hi)
    separated = ci_lo["lo"] > ci_hi["hi"] or ci_hi["lo"] > ci_lo["hi"]

    report = {
        "budget": {"epochs": args.epochs, "num_envs": args.num_envs, "seeds": args.seeds},
        "aggregation": "sum",
        "adversary": "adaptive",
        "f=0.0": {"returns": lo, "mean": float(np.mean(lo)), "ci": ci_lo},
        "f=0.5": {"returns": hi, "mean": float(np.mean(hi)), "ci": ci_hi},
        "separated": bool(separated),
        "verdict": (
            "USABLE: f=0 and f=0.5 separate with non-overlapping CIs, so this "
            "budget can express H6's effect."
            if separated else
            "NOT USABLE: f=0 and f=0.5 do not separate. A curve run at this "
            "budget would be flat regardless of whether the defence works, so "
            "it could not distinguish 'robust aggregation held' from 'nothing "
            "had time to happen'. Increase the budget rather than reading the "
            "flat curve."
        ),
        "precondition_only": (
            "Passing establishes the budget is capable of showing an effect. It "
            "is NOT evidence for or against H6."
        ),
        "elapsed_seconds": time.time() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0 if separated else 1


if __name__ == "__main__":
    raise SystemExit(main())
