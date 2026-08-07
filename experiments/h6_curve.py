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

def _series_key(cfg: dict, varying: set[str]) -> str:
    """Label this cell's series by everything that varies EXCEPT `f`.

    Read from the run's recorded config, not parsed out of the arm name. The
    name regex this replaced (`<adversary>_<aggregator>_f<NN>`) matched only
    `h6_byzantine.yaml`; every other H6 sweep -- the NaiveDistill control, the
    policy-level arm -- fell through it and the script reported "no H6 arms
    found in that manifest". A naming convention is not a data model.
    """
    parts = []
    if "algo" in varying:
        parts.append(str(cfg["algo"]))
    if "level" in varying:
        parts.append(f"{cfg['level']}-level")
    if "mode" in varying:
        parts.append(str(cfg["mode"]))
    if "aggregation" in varying:
        parts.append("trimmed mean" if cfg["aggregation"] == "trimmed_mean" else "plain sum")
    return " / ".join(parts) or "all arms"


def _cell(manifest_path: Path, row: dict) -> dict | None:
    """(f, series descriptor, honest return) for one manifest row."""
    run_dir = Path(row.get("run_dir", ""))
    if not run_dir.is_absolute():
        run_dir = (manifest_path.parent.parent.parent / run_dir) if not run_dir.exists() else run_dir
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    cfg = meta.get("config") or {}
    stress = cfg.get("stress") or {}

    # The HONEST agents' return. Byzantine agents are sabotaging, so folding
    # them into the headline measures the attack's self-harm alongside the
    # defence's success -- a number that moves with `f` even when the defence is
    # perfect. Falls back to final_return only when the honest key is absent,
    # and says so, rather than quietly plotting an all-agent mean under an
    # honest-agent axis label (which is what this figure used to do).
    ret = row.get("final_return_honest")
    honest = ret is not None
    if not honest:
        ret = row.get("final_return")
    if ret is None:
        return None

    return {
        "f": float(stress.get("byzantine_fraction", 0.0)),
        "seed": row["seed"],
        "return": float(ret),
        "honest_only": honest,
        "descriptor": {
            "algo": meta.get("algo"),
            "level": stress.get("byzantine_level", "gradient"),
            "mode": stress.get("byzantine_mode", "none"),
            "aggregation": (cfg.get("distill") or {}).get("aggregation"),
        },
    }


def load(manifest_path: Path) -> tuple[dict[str, dict[float, dict[int, float]]], bool]:
    manifest = json.loads(manifest_path.read_text())
    cells = [c for row in manifest["runs"] for c in [_cell(manifest_path, row)] if c]
    if not cells:
        return {}, False

    # Only label a series by what actually distinguishes it. Naming every arm by
    # all four fields makes a single-algorithm sweep unreadable.
    varying = {
        k for k in ("algo", "level", "mode", "aggregation")
        if len({str(c["descriptor"][k]) for c in cells}) > 1
    }
    if not varying:
        varying = {"algo"}

    out: dict[str, dict[float, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for c in cells:
        out[_series_key(c["descriptor"], varying)][c["f"]][c["seed"]] = c["return"]
    return out, all(c["honest_only"] for c in cells)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="runs/h6_byzantine/manifest.json")
    p.add_argument("--out", default="runs/h6_byzantine/figures/h6_curve.png")
    p.add_argument("--json-out", default="runs/h6_byzantine/h6_curve.json")
    args = p.parse_args(argv)

    series, honest_only = load(Path(args.manifest))
    if not series:
        print("no runs with a recorded config in that manifest", file=sys.stderr)
        return 2
    ylabel = (
        "final return (honest agents)" if honest_only
        else "final return (ALL agents -- honest-only not recorded)"
    )
    if not honest_only:
        print(
            "WARNING: these runs predate the honest-only evaluation metric, so "
            "the curve is the ALL-AGENT mean. At high f that includes the "
            "saboteurs' own returns and moves with f even under a perfect "
            "defence. Re-run to get the honest split.",
            file=sys.stderr,
        )

    # Baseline for the paired difference: f=0 of the SAME (adversary, aggregator).
    report: dict[str, dict] = {}
    for key, by_f in sorted(series.items()):
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
    report["_honest_only_return"] = honest_only
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
    for idx, (key, by_f) in enumerate(sorted(series.items())):
        fs = sorted(by_f)
        mean = [np.mean(list(by_f[f].values())) for f in fs]
        ci = [bootstrap_ci(list(by_f[f].values())) for f in fs]
        lo = [m - c["lo"] for m, c in zip(mean, ci)]
        hi = [c["hi"] - m for m, c in zip(mean, ci)]
        style = "-o" if idx % 2 == 0 else "--s"
        ax.errorbar(fs, mean, yerr=[lo, hi], fmt=style, capsize=3, label=key)
    ax.set_xlabel("Byzantine fraction $f$")
    ax.set_ylabel(ylabel)
    ax.set_title("H6: return vs $f$ — unpaired, 95% bootstrap CI")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    for idx, (key, by_f) in enumerate(sorted(series.items())):
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
        style = "-o" if idx % 2 == 0 else "--s"
        ax2.errorbar(fs, d, yerr=[dlo, dhi], fmt=style, capsize=3, label=key)
    ax2.axhline(0.0, color="k", lw=1)
    ax2.set_xlabel("Byzantine fraction $f$")
    ax2.set_ylabel("return at $f$  −  return at $f=0$  (same seeds)")
    ax2.set_title("H6 paired vs $f=0$ — the test with seed noise removed")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    v = report["_variance"]
    fig.suptitle(
        f"H6 — seed sd {v['seed_sd_within_cell']:.2f} vs between-condition sd "
        f"{v['condition_sd_between_cells']:.2f}. A flat curve is only "
        "interpretable if the second number is not swamped by the first.",
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
