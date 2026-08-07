#!/usr/bin/env python3
"""H7's verdict across the whole D7 grid. REPORTING ONLY -- runs nothing.

What H7 claims, and what this reports
-------------------------------------

**The interaction is the claim (SPEC Sec. 10 / D8), not either environment
alone.** D-TrOPD is predicted to help under heterogeneity and cost something
under homogeneity; the result is the SIGN FLIP between the two. A positive
heterogeneous difference on its own is not H7 -- an algorithm that is simply
better everywhere would produce it, and would not support the paper's claim.

So each grid point reports three numbers with CIs:

  het   = return(D-TrOPD) - return(IPPO)  on the heterogeneous environment
  hom   = the same difference on the homogeneous control
  inter = het - hom, the interaction

all paired by seed -- arms share the seed list and therefore the environment
instances, so the seed-to-seed spread cancels within a difference.

**The D7 sensitivity rule is applied, not chosen (SPEC Sec. 11 / D7).** The
verdict is computed at all four grid points and `d7_sensitivity_verdict` decides:
identical at every point means the conclusion is stable and the calibration
window is immaterial; a flip means the benefit is calibration-sensitive and THAT
IS THE FINDING, reported as a limitation. Selecting the point with the nicer
answer is the failure mode the rule exists to make impossible, so this script
prints every point whatever they say.

**The D9 consensus-pull cost is reported alongside (SPEC Sec. 11 / D9)** -- the
homogeneous-environment cost of distillation, which is the floor the
heterogeneous benefit has to clear. It is reported SEPARATELY from return and
never as a proxy for it: THEORY_CHANGES §12 measured a 26% swing in honest
consensus weight with zero detectable effect on return.

Usage
-----
    python experiments/h7_report.py                       # runs/h7_<point>
    python experiments/h7_report.py --root /data --out h7.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtropd.analysis import (  # noqa: E402
    D7_DELTA_BASE,
    D7_GRID_POINTS,
    d7_sensitivity_verdict,
)


def _fmt(d: dict) -> str:
    ci = d.get("ci") or {}
    lo, hi = ci.get("lo"), ci.get("hi")
    if lo is None:
        return f"{d.get('mean', float('nan')):+8.3f}  [   n/a          ]"
    return f"{d.get('mean', float('nan')):+8.3f}  [{lo:+8.3f}, {hi:+8.3f}]"


def _confirmed(inter: dict, sign_flip: bool | None) -> bool | None:
    """H7 at one grid point, per D8: **the SIGN FLIP is the claim**.

    Both conditions are required, and both are reported separately above so a
    partial result stays visible instead of collapsing into one bit:

    * the interaction's CI excludes zero on the positive side -- the statistical
      form of "the heterogeneous benefit exceeds the homogeneous cost";
    * the predicted sign flip is observed -- `het > 0` AND `hom < 0`.

    An algorithm that is simply better everywhere satisfies the first and not the
    second, and would not support the paper's claim. `None` when the CI is
    missing: an unknown verdict must not be silently counted as either.
    """
    ci = (inter or {}).get("ci") or {}
    lo, hi = ci.get("lo"), ci.get("hi")
    if lo is None or hi is None:
        return None
    return bool(lo > 0.0 and sign_flip)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs", help="directory holding runs/h7_<point>")
    p.add_argument("--out", default="runs/h7_grid_report.json")
    p.add_argument("--points", nargs="*", default=list(D7_GRID_POINTS))
    args = p.parse_args(argv)

    root = Path(args.root)
    points: dict[str, dict] = {}
    for point in args.points:
        report_path = root / f"h7_{point}" / "report.json"
        if not report_path.exists():
            print(f"  MISSING  {report_path}", file=sys.stderr)
            continue
        report = json.loads(report_path.read_text())
        inter = report.get("h7_interaction") or {}
        points[point] = {
            "delta_base": D7_DELTA_BASE["heterogeneous"].get(point),
            "delta_base_homogeneous": D7_DELTA_BASE["homogeneous"].get(point),
            "heterogeneous": inter.get("heterogeneous_difference", {}),
            "homogeneous": inter.get("homogeneous_difference", {}),
            "interaction": {
                "mean": inter.get("interaction"),
                "ci": inter.get("interaction_ci", {}),
                "n_pairs": inter.get("interaction_n_pairs"),
            },
            "sign_flip": inter.get("predicted_sign_flip_observed"),
            "confirmed": None,
            "consensus_pull_cost": report.get("consensus_pull_cost", {}),
            "n_seeds": (inter.get("heterogeneous_difference") or {}).get("n"),
            "paired": (inter.get("heterogeneous_difference") or {}).get("paired"),
        }
        points[point]["confirmed"] = _confirmed(
            points[point]["interaction"], points[point]["sign_flip"]
        )

    if not points:
        print("no H7 grid points found; run experiments/run_h7_grid.sh", file=sys.stderr)
        return 2

    print("=" * 78)
    print("H7 -- the INTERACTION is the claim (D8). Paired by seed within each")
    print("environment; the predicted result is the SIGN FLIP between them.")
    print("=" * 78)
    for point, d in points.items():
        seeds = d["n_seeds"]
        flag = "" if (seeds or 0) >= 10 else f"   << {seeds} seeds, SPEC Sec. 7 requires >=10"
        print(f"\n-- {point}   delta_base het {d['delta_base']:.6g} / hom "
              f"{d['delta_base_homogeneous']:.6g}   n={seeds} paired={d['paired']}{flag}")
        print(f"   heterogeneous  D-TrOPD - IPPO   {_fmt(d['heterogeneous'])}")
        print(f"   homogeneous    D-TrOPD - IPPO   {_fmt(d['homogeneous'])}")
        print(f"   INTERACTION    het - hom        {_fmt(d['interaction'])}"
              f"  (n_pairs={d['interaction'].get('n_pairs')})")
        verdict = ("CONFIRMED" if d["confirmed"]
                   else "NO CI" if d["confirmed"] is None else "not confirmed")
        print(f"   sign flip (het>0 and hom<0): {d['sign_flip']}"
              f"   |   interaction CI excludes 0: "
              f"{(d['interaction'].get('ci') or {}).get('lo', 0) > 0}"
              f"   ->  H7 {verdict}")
        for algo, cost in (d["consensus_pull_cost"] or {}).items():
            print(f"   D9 consensus-pull cost, homogeneous, {algo}: "
                  f"{cost.get('cost', float('nan')):+8.3f}")

    known = {k: v["confirmed"] for k, v in points.items() if v["confirmed"] is not None}
    unknown = [k for k, v in points.items() if v["confirmed"] is None]
    out: dict = {"points": points, "unresolved_points": unknown}

    print("\n" + "=" * 78)
    if unknown:
        print(f"D7 VERDICT WITHHELD: no CI at {unknown}. A point without a verdict is")
        print("not a confirming point; complete the grid before concluding.")
        out["d7"] = {"complete": False, "reason": f"no CI at {unknown}"}
    else:
        verdict = d7_sensitivity_verdict(known, grid=args.points)
        out["d7"] = verdict
        print("D7 SENSITIVITY (rule fixed before any result, SPEC Sec. 11 / D7)")
        print(f"   verdicts: {verdict['verdicts']}")
        print(f"   complete: {verdict['complete']}   stable: {verdict['stable']}")
        print(f"   -> {verdict['conclusion']}")
        if verdict["report_as_limitation"]:
            print("\n   The flip IS the finding. Report it; do not pick a point.")
    print("=" * 78)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
