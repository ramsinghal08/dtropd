#!/usr/bin/env python3
"""Does the trust gate attenuate Byzantine neighbours? (SPEC D19)

**This is the readout for the policy-level arm, and final return is not it.**
The gate's claim is about `w_ij`, so the measurement is `w_ij` split by whether
the neighbour on that edge is Byzantine -- reporting final return here would
answer a question nobody asked, which has already happened three times in this
project.

What it computes, per run and pooled across seeds:

  * mean `D_ij` and `w_ij` on **Byzantine edges** vs **honest edges**;
  * the **paired per-seed difference**, which is the analysis that matters --
    seeds differ far more from each other than the conditions differ, so an
    unpaired comparison drowns the effect in seed variance;
  * a bootstrap CI on that paired difference.

The Byzantine set is recomputed from the run's own seed and fraction through
`byzantine_agents`, i.e. the same function the run used, so the split cannot
drift from what was actually attacked.

Usage
-----
    python experiments/gate_split.py runs/h6_policy_level/runs/*/
    python experiments/gate_split.py --manifest runs/h6_policy_level/manifest.json
    python experiments/gate_split.py runs/h6_byzantine/runs/*/ --out split.json

Cells with no Byzantine agents (`f = 0`) are skipped with a note rather than
silently: they have no split to make, and counting them as "no difference found"
would dilute the result with cells that could not have shown one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtropd.analysis import bootstrap_ci  # noqa: E402
from dtropd.byzantine import byzantine_agents  # noqa: E402


def run_split(run_dir: Path) -> dict | None:
    """Byzantine-vs-honest edge statistics for one run, or None if inapplicable."""
    meta_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not meta_path.exists() or not metrics_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    cfg = meta.get("config") or {}
    stress = cfg.get("stress") or {}
    n_agents = int((cfg.get("env") or {}).get("n_agents", 0))
    fraction = float(stress.get("byzantine_fraction", 0.0))
    mode = stress.get("byzantine_mode", "none")
    level = stress.get("byzantine_level", "gradient")
    seed = int(meta.get("seed", 0))

    enabled = fraction > 0.0 and (level == "policy" or mode != "none")
    n_byz = int(fraction * n_agents) if enabled else 0
    if n_byz == 0:
        return {"run": run_dir.name, "seed": seed, "skipped": "no Byzantine agents"}

    # Prefer the set the run itself recorded. Re-deriving is a fallback for runs
    # written before the field existed, and it is only correct for as long as
    # nobody changes how the set is drawn -- a mislabelled split would fail
    # silently, in the direction of "no effect found".
    recorded = meta.get("byzantine")
    derived = byzantine_agents(
        n_agents,
        n_byz,
        stress.get("byzantine_seed") if stress.get("byzantine_seed") is not None else seed,
    )
    if recorded is not None and sorted(recorded) != sorted(derived):
        raise SystemExit(
            f"{run_dir}: the recorded Byzantine set {sorted(recorded)} does not "
            f"match the one re-derived from (seed, fraction) {sorted(derived)}. "
            "One of them is wrong and every split computed here would be too."
        )
    byz = set(recorded if recorded is not None else derived)

    fields = ("D_ij", "w_ij")
    acc: dict[tuple[str, bool], list[float]] = defaultdict(list)
    epochs = 0
    with metrics_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            edges = rec.get("edges")
            if not edges:
                continue
            epochs += 1
            for row in edges:
                if "w_ij" not in row:
                    # NaiveDistill has no gate, so it logs no w_ij. Its edges are
                    # ungated by construction and there is nothing to split.
                    continue
                is_byz = row["j"] in byz
                for f in fields:
                    acc[(f, is_byz)].append(float(row[f]))

    if not acc:
        return {
            "run": run_dir.name,
            "seed": seed,
            "skipped": "no per-edge gate detail logged (instrument.per_edge_detail)",
        }

    out = {
        "run": run_dir.name,
        "seed": seed,
        "algo": meta.get("algo"),
        "byzantine_level": level,
        "byzantine_fraction": fraction,
        "byzantine_scale": stress.get("byzantine_scale"),
        "n_byzantine": n_byz,
        "byzantine_set_source": "recorded" if recorded is not None else "re-derived",
        "instrumented_epochs": epochs,
        "n_byzantine_edges": len(acc[("w_ij", True)]),
        "n_honest_edges": len(acc[("w_ij", False)]),
    }
    for f in fields:
        b, h = acc[(f, True)], acc[(f, False)]
        out[f"{f}_byzantine"] = float(np.mean(b)) if b else float("nan")
        out[f"{f}_honest"] = float(np.mean(h)) if h else float("nan")
        out[f"{f}_diff"] = out[f"{f}_byzantine"] - out[f"{f}_honest"]
    return out


def pooled(rows: list[dict]) -> dict:
    """Paired-by-seed summary. One difference per seed, then a CI over seeds."""
    usable = [r for r in rows if "w_ij_diff" in r]
    if not usable:
        return {"cells": len(rows), "usable": 0}
    out: dict = {
        "cells": len(rows),
        "usable": len(usable),
        "byzantine_set_source": sorted({r["byzantine_set_source"] for r in usable}),
    }
    for f in ("D_ij", "w_ij"):
        diffs = [r[f"{f}_diff"] for r in usable]
        out[f] = {
            "byzantine_mean": float(np.mean([r[f"{f}_byzantine"] for r in usable])),
            "honest_mean": float(np.mean([r[f"{f}_honest"] for r in usable])),
            "paired_diff_mean": float(np.mean(diffs)),
            "paired_diff_ci": bootstrap_ci(diffs),
            "paired_diff_per_seed": diffs,
        }
    lo = out["w_ij"]["paired_diff_ci"].get("lo")
    hi = out["w_ij"]["paired_diff_ci"].get("hi")
    if lo is not None and hi is not None:
        out["gate_responds"] = bool(hi < 0.0)
        out["verdict"] = (
            "the gate ATTENUATES Byzantine edges"
            if hi < 0.0
            else "no attenuation: the paired CI on w_ij spans zero or is positive"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="run directories")
    p.add_argument("--manifest", default=None, help="take the run list from a sweep manifest")
    p.add_argument("--out", default=None, help="write the full report here as JSON")
    args = p.parse_args(argv)

    dirs = [Path(r) for r in args.runs]
    if args.manifest:
        man = json.loads(Path(args.manifest).read_text())
        root = Path(args.manifest).parent
        for r in man.get("runs", []):
            d = r.get("run_dir") or r.get("dir")
            if d:
                path = Path(d)
                dirs.append(path if path.is_absolute() else root.parent.parent / path)
    if not dirs:
        p.error("no run directories given")

    rows = [r for r in (run_split(d) for d in dirs if d.is_dir()) if r]
    by_group: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if "skipped" in r:
            continue
        by_group[(r["algo"], r["byzantine_level"], r["byzantine_fraction"])].append(r)

    report = {"groups": {}, "skipped": [r for r in rows if "skipped" in r]}
    for key, group in sorted(by_group.items(), key=lambda kv: str(kv[0])):
        label = f"{key[0]} / {key[1]} / f={key[2]}"
        summary = pooled(group)
        report["groups"][label] = summary
        src = ", ".join(summary.get("byzantine_set_source", []))
        print(f"== {label}   ({summary['usable']} seeds; Byzantine set {src})")
        for f in ("D_ij", "w_ij"):
            s = summary.get(f)
            if not s:
                continue
            ci = s["paired_diff_ci"]
            print(
                f"   {f:5s}  byzantine {s['byzantine_mean']:.5f}   honest {s['honest_mean']:.5f}"
                f"   paired diff {s['paired_diff_mean']:+.5f} "
                f"[{ci.get('lo', float('nan')):+.5f}, {ci.get('hi', float('nan')):+.5f}]"
            )
        if "verdict" in summary:
            print(f"   -> {summary['verdict']}")
        print()

    for r in report["skipped"]:
        print(f"skipped {r['run']}: {r['skipped']}", file=sys.stderr)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0 if report["groups"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
