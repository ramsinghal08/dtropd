#!/usr/bin/env python3
"""How much of the update is the social term? (the number behind H6's flatness)

Reports ``||beta * grad L_soc|| / (||beta * grad L_soc|| + ||grad L_CLIP||)``
over training, from metrics already on disk. No re-running.

Why the GRADIENT ratio and not the loss ratio
---------------------------------------------

Comparing the *values* ``beta*L_soc`` and ``L_CLIP`` is misleading and would
give a badly wrong answer here. ``L_CLIP`` is a policy-gradient surrogate whose
value sits near zero by construction (it is a clipped ratio times a normalised
advantage, and normalised advantages average to zero), while ``L_soc`` is a sum
of KL divergences and is strictly positive. Their ratio can be order 1 while the
social term contributes almost nothing to the actual parameter update.

What determines the update is the **gradient**, so that is what is instrumented:
``consensus_pull/social_grad_fraction`` in ``metrics.jsonl``, computed on the
same minibatch as the policy gradient so the comparison is like-for-like.

Requires ``instrument.per_edge_gradients: true`` in the run being read -- the
fraction is computed inside the per-edge gradient block. Runs with it off (the
H4/H5 scale sweeps) carry no such field, and this script says so rather than
silently reporting nothing.

Usage::

    python experiments/social_share.py runs/h6_byzantine/runs/*/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KEY = "consensus_pull/social_grad_fraction_p50"
ALT = ("consensus_pull/social_grad_fraction_mean", "consensus_pull/social_grad_fraction_p50")
NORMS = ("consensus_pull/social_grad_norm_p50", "consensus_pull/policy_grad_norm_p50")


def series(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl" if run_dir.is_dir() else run_dir
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories (or metrics.jsonl paths)")
    p.add_argument("--out", default=None, help="write JSON summary here")
    args = p.parse_args(argv)

    fractions, rows, missing = [], [], []
    for raw in args.runs:
        run = Path(raw)
        recs = series(run)
        if not recs:
            continue
        key = next((k for k in ALT if k in recs[-1]), None)
        if key is None:
            missing.append(str(run))
            continue
        vals = [(r["epoch"], r[key]) for r in recs if key in r]
        fractions.extend(v for _, v in vals)
        rows.append({"run": str(run), "n_epochs": len(vals),
                     "first": vals[0][1], "last": vals[-1][1],
                     "median": float(np.median([v for _, v in vals]))})

    if missing and not rows:
        print("NOT LOGGED in any run given.", file=sys.stderr)
        print(f"  {len(missing)} run(s) lack '{KEY}'.", file=sys.stderr)
        print("  That field is only produced when instrument.per_edge_gradients", file=sys.stderr)
        print("  is true. The H4/H5 scale sweeps set it false; H6 sets it true.", file=sys.stderr)
        return 2

    arr = np.asarray(fractions, dtype=float)
    arr = arr[np.isfinite(arr)]
    summary = {
        "metric": "||beta * grad L_soc|| / (||beta * grad L_soc|| + ||grad L_CLIP||)",
        "runs_with_field": len(rows),
        "runs_missing_field": len(missing),
        "n_observations": int(arr.size),
        "quantiles": {
            q: float(np.quantile(arr, float(q))) for q in ("0.05", "0.25", "0.5", "0.75", "0.95")
        } if arr.size else {},
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "per_run": rows,
        "interpretation": (
            "This is the share of the parameter update attributable to the "
            "social term. If it is a few percent, distillation is a small "
            "perturbation on PPO -- which would explain simultaneously: H6's "
            "flat curve (corrupting a small term changes little), the small "
            "consensus-pull cost, and a likely small D-TrOPD-minus-IPPO gap in "
            "H7, since that gap is generated ENTIRELY by this term."
        ),
    }
    text = json.dumps(summary, indent=2, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)
    if arr.size:
        print(f"\nSOCIAL SHARE OF THE UPDATE: median {np.median(arr):.4f} "
              f"({100*np.median(arr):.2f}%), mean {arr.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
