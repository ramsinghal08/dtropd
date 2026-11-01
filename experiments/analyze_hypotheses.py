#!/usr/bin/env python3
"""Compute every H1-H7 measurement from ONE run's metrics file.

This script *is* the acceptance demonstration (SPEC Sec. 5):

    "a single training run produces a metrics file from which every H1-H7
     measurement in Sec. 6 can be computed without re-running."

It reads `metrics.jsonl` + `metadata.json` from a single run directory and emits
a value for each of H1-H7, or an explicit list of the fields that were missing.
It exits non-zero if any hypothesis is not computable, because "we'll add that
logging later" is exactly the failure mode exists to prevent.

H6 and H7 are comparisons across runs by construction (Byzantine fraction,
algorithm x environment). For those, "computable" means the *per-run* quantity
the comparison consumes is present and complete -- which is the only thing a
single run can be asked for. The script says so rather than silently counting
them as satisfied.

Usage::

    python experiments/analyze_hypotheses.py --run runs/phase5/dtropd_seed0
    python experiments/analyze_hypotheses.py --run <dir> --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtropd.analysis import HYPOTHESES, analyze_run  # noqa: E402
from dtropd.utils import read_metrics  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="a single run directory")
    p.add_argument("--out", default=None, help="write the report here as JSON")
    p.add_argument(
        "--require",
        nargs="*",
        default=list(HYPOTHESES),
        help="hypotheses that must be computable for a zero exit code",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    run_dir = Path(args.run)
    metrics_path = run_dir / "metrics.jsonl" if run_dir.is_dir() else run_dir
    meta_path = (run_dir if run_dir.is_dir() else run_dir.parent) / "metadata.json"

    records = read_metrics(metrics_path)
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if not records:
        print(f"no metrics records in {metrics_path}", file=sys.stderr)
        return 2

    # `--require h3` used to die with `KeyError: 'h3'` three frames down. The
    # keys are "H1".."H7"; asking for a hypothesis by the name it is called
    # everywhere else in the project should not be a stack trace.
    required = [k.upper() for k in args.require]
    unknown = [k for k in required if k not in HYPOTHESES]
    if unknown:
        print(
            f"unknown hypothes{'is' if len(unknown) == 1 else 'es'} {unknown}; "
            f"expected any of {list(HYPOTHESES)} (case-insensitive)",
            file=sys.stderr,
        )
        return 2

    results = analyze_run(records, metadata)
    report = {
        "run": str(run_dir),
        "epochs": len(records),
        "metrics_schema_version": metadata.get("metrics_schema_version"),
        "reward_spec_version": metadata.get("reward_spec_version"),
        "calibration_id": metadata.get("calibration_id"),
        "algo": metadata.get("algo"),
        "hypotheses": {k: v.to_dict() for k, v in results.items()},
    }
    uncomputable = [k for k in required if not results[k].computable]
    report["all_required_computable"] = not uncomputable
    report["uncomputable"] = uncomputable

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))

    if not args.quiet:
        print(json.dumps(report, indent=2, default=str))

    width = max(len(results[k].title) for k in HYPOTHESES)
    print("\n--- : can every hypothesis be measured from this one file? ---")
    for key in HYPOTHESES:
        r = results[key]
        mark = "OK  " if r.computable else "MISS"
        detail = "" if r.computable else f"  missing: {r.missing_fields}"
        print(f"  {mark} {key}  {r.title:<{width}}  [{r.scope}]{detail}")
    print("\nRESULT:", "ALL MEASURABLE" if not uncomputable else f"MISSING {uncomputable}")
    return 0 if not uncomputable else 1


if __name__ == "__main__":
    raise SystemExit(main())
