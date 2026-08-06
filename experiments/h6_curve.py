#!/usr/bin/env python3
"""Regenerate H6's figure as return vs Byzantine fraction. REPORTING ONLY.

Does not re-run anything: reads the existing manifest and plots what is already
on disk.

Why this exists
---------------

`sweep_report` groups by ``(condition, algo)``, and `h6_byzantine.yaml` gives all
six `f` arms of an aggregator the *same* condition. So the report pooled the
`f` axis into a single number per condition -- marginalising away the very axis
H6's hypothesis is about. The resulting figure showed four points with
overlapping CIs and looked like "no effect", when it had not yet asked the
question.

H6's claim is the *shape of a curve over `f`*, and the comparison that carries
it is **paired**: every arm runs the same ten seeds on the same environment
streams, so the seed-to-seed spread (sd ~7) cancels in a within-seed difference
while the between-condition effect (sd ~0.6) does not. Plotting unpaired means
throws that away and buries any effect under seed noise.

Usage::

    python experiments/h6_curve.py --manifest runs/h6_byzantine/manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtropd.analysis import bootstrap_ci  # noqa: E402

#: arm names look like `<adversary>_<aggregator>_f<NN>`
ARM = re.compile(r"^(?P<adv>adaptive|signflip)_(?P<agg>sum|trim)_f(?P<f>\d+)$")


def load(manifest_path: Path) -> dict[tuple[str, str], dict[float, dict[int, float]]]:
    manifest = json.loads(manifest_path.read_text())
    out: dict[tuple[str, str], dict[float, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in manifest["runs"]:
        m = ARM.match(row["arm"])
        if not m or "final_return" not in row:
            continue
        f = int(m["f"]) / 100.0
        out[(m["adv"], m["agg"])][f][row["seed"]] = row["final_return"]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="runs/h6_byzantine/manifest.json")
    p.add_argument("--out", default="runs/h6_byzantine/figures/h6_curve.png")
    p.add_argument("--json-out", default="runs/h6_byzantine/h6_curve.json")
    args = p.parse_args(argv)

    series = load(Path(args.manifest))
    if not series:
        print("no H6 arms found in that manifest", file=sys.stderr)
        return 2

    # Baseline for the paired difference: f=0 of the SAME (adversary, aggregator).
    report: dict[str, dict] = {}
    for (adv, agg), by_f in sorted(series.items()):
        key = f"{adv}_{agg}"
        fs = sorted(by_f)
        base = by_f.get(0.0)
        rows = []
        for f in fs:
            seeds = sorted(by_f[f])
            vals = [by_f[f][s] for s in seeds]
            row = {
                "f": f,
                "n_seeds": len(vals),
                "mean": float(np.mean(vals)),
                "ci": bootstrap_ci(vals),
            }
            if base:
                shared = sorted(set(seeds) & set(base))
                diff = [by_f[f][s] - base[s] for s in shared]
                row["paired_vs_f0"] = {
                    "mean": float(np.mean(diff)),
                    "ci": bootstrap_ci(diff),
                    "n_pairs": len(diff),
                }
            rows.append(row)
        report[key] = {"points": rows}

    # Variance decomposition -- the number that explains the flat curve.
    all_vals = [v for by_f in series.values() for d in by_f.values() for v in d.values()]
    cond_means = [
        np.mean([v for v in d.values()]) for by_f in series.values() for d in by_f.values()
    ]
    report["_variance"] = {
        "seed_sd_within_cell": float(
            np.mean([np.std(list(d.values()), ddof=1) for by_f in series.values() for d in by_f.values()])
        ),
        "condition_sd_between_cells": float(np.std(cond_means, ddof=1)),
        "note": (
            "If the seed sd dwarfs the between-condition sd, a flat curve is "
            "consistent with BOTH 'the defence held' and 'the attack never bit'. "
            "That is a manipulation-check failure, not a result."
        ),
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(json.dumps(report, indent=2, default=str))
        print("\nmatplotlib not available; wrote JSON only", file=sys.stderr)
        return 0

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for (adv, agg), by_f in sorted(series.items()):
        fs = sorted(by_f)
        mean = [np.mean(list(by_f[f].values())) for f in fs]
        ci = [bootstrap_ci(list(by_f[f].values())) for f in fs]
        lo = [m - c["lo"] for m, c in zip(mean, ci)]
        hi = [c["hi"] - m for m, c in zip(mean, ci)]
        style = "-o" if agg == "trim" else "--s"
        ax.errorbar(fs, mean, yerr=[lo, hi], fmt=style, capsize=3,
                    label=f"{adv} / {'trimmed mean' if agg=='trim' else 'plain sum'}")
    ax.set_xlabel("Byzantine fraction $f$")
    ax.set_ylabel("final return (honest agents)")
    ax.set_title("H6: return vs $f$ — unpaired, 95% bootstrap CI")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    for (adv, agg), by_f in sorted(series.items()):
        base = by_f.get(0.0)
        if not base:
            continue
        fs = sorted(by_f)
        d, dlo, dhi = [], [], []
        for f in fs:
            shared = sorted(set(by_f[f]) & set(base))
            diff = [by_f[f][s] - base[s] for s in shared]
            c = bootstrap_ci(diff)
            d.append(float(np.mean(diff)))
            dlo.append(d[-1] - c["lo"])
            dhi.append(c["hi"] - d[-1])
        style = "-o" if agg == "trim" else "--s"
        ax2.errorbar(fs, d, yerr=[dlo, dhi], fmt=style, capsize=3,
                     label=f"{adv} / {'trimmed mean' if agg=='trim' else 'plain sum'}")
    ax2.axhline(0.0, color="k", lw=1)
    ax2.set_xlabel("Byzantine fraction $f$")
    ax2.set_ylabel("return at $f$  −  return at $f=0$  (same seeds)")
    ax2.set_title("H6 paired vs $f=0$ — the test with seed noise removed")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    v = report["_variance"]
    fig.suptitle(
        f"H6 — seed sd {v['seed_sd_within_cell']:.2f} vs between-condition sd "
        f"{v['condition_sd_between_cells']:.2f}: INCONCLUSIVE pending manipulation check",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"wrote {args.json_out}")
    print(f"\nseed sd within a cell:      {v['seed_sd_within_cell']:.2f}")
    print(f"condition sd between cells: {v['condition_sd_between_cells']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
