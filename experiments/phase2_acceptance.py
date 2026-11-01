#!/usr/bin/env python3
"""acceptance test (SPEC Sec. 5).

    "IPPO learns on the *homogeneous* env -- mean return over 3 seeds beats a
     random policy by a clear margin, and critic explained variance rises
     above 0.3."

"Clear margin" is operationalised here, once, before looking at any results:

  C1  every seed's final evaluation return exceeds the random-policy return on
      the same environment by at least ``--margin-sems`` (default 10) standard
      errors of the random-policy baseline;
  C2  the improvement is positive for *every* seed (not just on average);
  C3  every seed's critic explained variance rises above ``--ev-threshold``
      (default 0.3) at some point during training, and the final-epoch value is
      also reported.

Usage::

    python experiments/phase2_acceptance.py \
        --config experiments/configs/ippo_homogeneous.yaml \
        --seeds 0 1 2 --workers 3 --out runs/phase2
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtropd.config import Config  # noqa: E402
from dtropd.train import train  # noqa: E402
from dtropd.utils import configure_logging  # noqa: E402


def _run_one(payload: tuple[dict[str, Any], int, str]) -> dict[str, Any]:
    cfg_dict, seed, out_dir = payload
    configure_logging("INFO")
    cfg = Config.from_dict(cfg_dict)
    result = train(cfg, seed=seed, out_dir=Path(out_dir) / f"seed{seed}", progress=True)
    evs = [
        r["explained_variance"]
        for r in result.history
        if r.get("explained_variance") == r.get("explained_variance")  # not NaN
    ]
    return {
        "seed": seed,
        "run_dir": str(result.run_dir),
        "final_return": result.final_eval["return_mean"],
        "final_return_sem": result.final_eval["return_sem"],
        "random_return": result.random_eval["return_mean"],
        "random_return_sem": result.random_eval["return_sem"],
        "improvement": result.final_eval["return_mean"] - result.random_eval["return_mean"],
        "max_explained_variance": max(evs) if evs else float("nan"),
        "final_explained_variance": evs[-1] if evs else float("nan"),
        "final_captures": result.final_eval["captures_mean"],
        "random_captures": result.random_eval["captures_mean"],
        "wall_clock_seconds": result.wall_clock_seconds,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "experiments/configs/ippo_homogeneous.yaml"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", default="runs/phase2")
    p.add_argument("--epochs", type=int, default=None, help="override train.total_epochs")
    p.add_argument("--margin-sems", type=float, default=10.0)
    p.add_argument("--ev-threshold", type=float, default=0.3)
    args = p.parse_args(argv)

    configure_logging("INFO")
    cfg = Config.from_yaml(args.config)
    if args.epochs is not None:
        cfg.train.total_epochs = args.epochs
    cfg.validate(emit=False)
    assert not cfg.env.heterogeneous, "acceptance is defined on the homogeneous env"

    payloads = [(cfg.to_dict(), seed, args.out) for seed in args.seeds]
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(args.workers, len(payloads))) as pool:
            rows = pool.map(_run_one, payloads)
    else:
        rows = [_run_one(x) for x in payloads]

    improvements = [r["improvement"] for r in rows]
    margins = [
        r["improvement"] / r["random_return_sem"] if r["random_return_sem"] > 0 else float("inf")
        for r in rows
    ]
    c1 = all(m >= args.margin_sems for m in margins)
    c2 = all(i > 0 for i in improvements)
    c3 = all(r["max_explained_variance"] > args.ev_threshold for r in rows)

    report = {
        "acceptance": "-- IPPO learns on the homogeneous env",
        "config": args.config,
        "seeds": args.seeds,
        "criteria": {
            "C1_margin_in_random_sems": {
                "threshold": args.margin_sems,
                "values": margins,
                "passed": bool(c1),
            },
            "C2_all_seeds_improve": {"values": improvements, "passed": bool(c2)},
            "C3_explained_variance": {
                "threshold": args.ev_threshold,
                "values": [r["max_explained_variance"] for r in rows],
                "passed": bool(c3),
            },
        },
        "passed": bool(c1 and c2 and c3),
        "mean_final_return": sum(r["final_return"] for r in rows) / len(rows),
        "mean_random_return": sum(r["random_return"] for r in rows) / len(rows),
        "mean_improvement": sum(improvements) / len(improvements),
        "runs": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase2_acceptance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nPHASE 2 ACCEPTANCE:", "PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
