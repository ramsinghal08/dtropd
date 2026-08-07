#!/usr/bin/env python3
"""Find run directories whose metrics file cannot be trusted.

Two corruptions, one cause. Before ``MetricsWriter`` truncated, a cell killed
mid-write left a partial line; ``_is_complete`` correctly re-ran the cell, and
the re-run then APPENDED to the dead attempt:

  * **malformed line** -- where the two runs meet. Self-announcing: anything
    reading the file raises.
  * **duplicate epochs** -- on either side of it. Silent, and the dangerous one:
    the file parses, and every summary over it averages across two runs.

Neither is repairable. Dropping the bad line still leaves two runs' records in
one file, so the only correct fix is to delete the run directory and let the
sweep re-run the cell -- which is cheap, and which the resume path now does on
its own once the directory is gone.

Usage
-----
    python experiments/check_runs.py runs/                # report
    python experiments/check_runs.py runs/ --delete       # report and remove

``--delete`` removes only directories this reports as corrupt, and prints each
one. Clean cells are never touched, so a sweep re-run after it costs only the
affected cells.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtropd.utils import CorruptMetricsError, read_metrics  # noqa: E402


def check(run_dir: Path) -> str | None:
    """The reason this run is untrustworthy, or None if it is fine."""
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return None  # not a finished run; the sweep will handle it
    try:
        records = read_metrics(metrics)
    except CorruptMetricsError as exc:
        return str(exc).split(". ")[0]
    if not records:
        return f"{metrics} has no records"
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="directories to scan recursively")
    p.add_argument(
        "--delete",
        action="store_true",
        help="remove the corrupt run directories so the sweep re-runs those cells",
    )
    args = p.parse_args(argv)

    dirs = sorted({m.parent for root in args.roots for m in Path(root).rglob("metrics.jsonl")})
    corrupt = [(d, r) for d in dirs for r in [check(d)] if r]

    print(f"scanned {len(dirs)} run directories")
    for d, reason in corrupt:
        print(f"  CORRUPT  {d}\n           {reason}")
    if not corrupt:
        print("  all clean")
        return 0

    print(f"\n{len(corrupt)} of {len(dirs)} are corrupt.")
    if not args.delete:
        print("Re-run with --delete to remove them, then re-run the sweep; only")
        print("these cells will be recomputed.")
        return 1

    for d, _ in corrupt:
        shutil.rmtree(d)
        print(f"  removed {d}")
    print("\nNow re-run the sweep. Completed cells are skipped by fingerprint;")
    print("only the removed ones run again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
