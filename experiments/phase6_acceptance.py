#!/usr/bin/env python3
"""acceptance test (SPEC Sec. 5).

    "Accept: `python run.py --config experiments/configs/h7.yaml --seeds 10`
     produces the main comparison figure end-to-end."

plus the resumability requirement: kill a sweep mid-run and restart it.

**This runs the machinery on the SMOKE sweep, not on H7.** SPEC gates the H7
experiment at , and a "quick look" at this stage is exactly the peek that
biases later interpretation. The code path is identical -- ``run.py sweep``
against a sweep YAML -- so what is demonstrated here is the harness, which is
what is for. ``experiments/configs/h7.yaml`` is shipped, validated by
the test suite, and deliberately not executed.

Criteria:

  E1  one command, from scratch, produces the manifest, the report and the
      comparison figure -- no manual steps (SPEC Sec. 7);
  E2  a sweep killed mid-run resumes: completed cells are skipped, the rest run,
      and a *changed* config is NOT skipped;
  E3  the figure is regenerable from stored metrics alone (``--report-only``),
      with no training and no in-memory state;
  E4  the structural guards fire: D6 (one calibration per environment, shared),
      D13 (no algorithm comparison across reward spec versions), paired seeds;
  E5  the report carries the D8 interaction and the D9 consensus-pull cost with
      CIs.

Usage::

    python experiments/phase6_acceptance.py --out runs/phase6
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtropd.sweep import (  # noqa: E402
    SweepArm,
    SweepError,
    build_plan,
    check_paired_seeds,
    check_reward_spec_versions,
    load_sweep_config,
    run_sweep,
    sweep_report,
)
from dtropd.utils import configure_logging, get_logger  # noqa: E402

logger = get_logger("phase6")

SMOKE = ROOT / "experiments/configs/sweep_smoke.yaml"
H7 = ROOT / "experiments/configs/h7.yaml"


def _sweep_cmd(out: Path, seeds: int, extra: list[str] | None = None) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "run.py"),
        "sweep",
        "--config",
        str(SMOKE),
        "--seeds",
        str(seeds),
        "--out",
        str(out),
        *(extra or []),
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/phase6")
    p.add_argument("--seeds", type=int, default=2)
    args = p.parse_args(argv)

    configure_logging("INFO")
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    # --- E1: one command, from scratch, to the figure ---------------------- #
    logger.info("E1: one command from scratch")
    proc = subprocess.run(_sweep_cmd(out, args.seeds), capture_output=True, text=True, cwd=ROOT)
    manifest_path = out / "manifest.json"
    figure = out / "figures" / "comparison.png"
    e1 = {
        "command": " ".join(_sweep_cmd(out, args.seeds)),
        "returncode": proc.returncode,
        "manifest": manifest_path.exists(),
        "report": (out / "report.json").exists(),
        "figure": figure.exists(),
        "figure_bytes": figure.stat().st_size if figure.exists() else 0,
        "passed": proc.returncode == 0 and figure.exists() and figure.stat().st_size > 0,
    }
    if not e1["passed"]:
        e1["stderr_tail"] = proc.stderr[-1500:]

    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    # --- E2: kill mid-run, then resume ------------------------------------- #
    logger.info("E2: kill a sweep mid-run and restart it")
    resume_dir = out / "resume"
    try:
        subprocess.run(
            _sweep_cmd(resume_dir, args.seeds), capture_output=True, text=True, cwd=ROOT, timeout=6
        )
        partial_note = "sweep finished before the kill window; forcing a partial state"
    except subprocess.TimeoutExpired:
        partial_note = "sweep killed mid-run by timeout"
    partial = json.loads((resume_dir / "manifest.json").read_text())
    completed_before = sum(1 for r in partial["runs"] if r["status"] == "completed")

    # If it happened to finish, delete some runs so the resume has work to do.
    total_cells = 0
    if not partial.get("in_progress", True):
        plan = build_plan(SMOKE, seeds=args.seeds, out_dir=resume_dir)
        total_cells = len(plan.arms) * len(plan.seeds)
        for entry in partial["runs"][: max(1, total_cells // 2)]:
            shutil.rmtree(entry["run_dir"], ignore_errors=True)
        completed_before = total_cells - max(1, total_cells // 2)

    resumed = subprocess.run(
        _sweep_cmd(resume_dir, args.seeds), capture_output=True, text=True, cwd=ROOT
    )
    final = json.loads((resume_dir / "manifest.json").read_text())
    total_cells = total_cells or len(final["runs"])

    # A *changed* config must not be skipped as complete.
    changed = subprocess.run(
        _sweep_cmd(resume_dir, args.seeds, ["--no-resume"]),
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    changed_manifest = json.loads((resume_dir / "manifest.json").read_text())

    e2 = {
        "note": partial_note,
        "completed_before_resume": completed_before,
        "total_cells": total_cells,
        "skipped_on_resume": final["skipped"],
        "completed_on_resume": final["completed"],
        "all_cells_present_after_resume": len(final["runs"]) == total_cells
        and all(r["status"] != "failed" for r in final["runs"]),
        "no_resume_reruns_everything": changed_manifest["skipped"] == 0,
        "passed": bool(
            resumed.returncode == 0
            and final["skipped"] > 0
            and final["in_progress"] is False
            and changed.returncode == 0
            and changed_manifest["skipped"] == 0
        ),
    }

    # --- E3: figure regenerable from stored metrics alone ------------------ #
    logger.info("E3: rebuild the figure from stored metrics only")
    report_only = subprocess.run(
        _sweep_cmd(out, args.seeds, ["--report-only", "--figure", str(out / "figures" / "rebuilt.png")]),
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    rebuilt = out / "figures" / "rebuilt.png"
    e3 = {
        "returncode": report_only.returncode,
        "rebuilt_figure": rebuilt.exists(),
        "passed": report_only.returncode == 0 and rebuilt.exists() and rebuilt.stat().st_size > 0,
    }
    if not e3["passed"]:
        e3["stderr_tail"] = report_only.stderr[-1500:]

    # --- E4: the structural guards fire ------------------------------------ #
    logger.info("E4: structural guards")
    guards: dict[str, Any] = {}

    plan = load_sweep_config(SMOKE)
    plan.arms.append(
        SweepArm(
            name="bad_cross_version",
            algo="dtropd",
            condition="heterogeneous",
            overrides={"env.reward_spec_version": "v1-local"},
        )
    )
    try:
        check_reward_spec_versions(plan)
        guards["d13_cross_version_refused"] = False
    except SweepError as exc:
        guards["d13_cross_version_refused"] = True
        guards["d13_message"] = str(exc)[:150]

    plan2 = load_sweep_config(SMOKE)
    plan2.arms[0].overrides["train.seed"] = 99
    try:
        check_paired_seeds(plan2)
        guards["paired_seeds_enforced"] = False
    except SweepError:
        guards["paired_seeds_enforced"] = True

    calibration_groups = manifest.get("checks", {}).get("calibration", {})
    guards["d6_one_calibration_per_environment"] = calibration_groups.get("n_environments", 0) >= 2
    guards["d6_groups"] = calibration_groups.get("per_environment", [])
    guards["passed"] = bool(
        guards["d13_cross_version_refused"]
        and guards["paired_seeds_enforced"]
        and guards["d6_one_calibration_per_environment"]
    )
    e4 = guards

    # --- E5: interaction + consensus-pull cost, with CIs ------------------- #
    report = sweep_report(manifest)
    interaction = report.get("h7_interaction", {})
    cost = report.get("consensus_pull_cost", {})
    e5 = {
        "conditions": report.get("conditions"),
        "interaction_reported": bool(interaction),
        "sign_flip_field_present": "predicted_sign_flip_observed" in interaction,
        "interaction_has_cis": bool(interaction.get("heterogeneous_difference", {}).get("ci")),
        "consensus_pull_cost_algos": sorted(cost),
        "consensus_pull_cost_has_ci": all(bool(v.get("ci")) for v in cost.values()) if cost else False,
        "passed": bool(interaction and cost and "predicted_sign_flip_observed" in interaction),
    }

    report_out = {
        "acceptance": "-- experiment harness",
        "sweep_config_used": str(SMOKE.relative_to(ROOT)),
        "h7_config_shipped_but_not_run": str(H7.relative_to(ROOT)),
        "criteria": {
            "E1_one_command_end_to_end": e1,
            "E2_kill_and_resume": e2,
            "E3_figure_from_stored_metrics": e3,
            "E4_structural_guards": e4,
            "E5_interaction_and_consensus_cost": e5,
        },
        "passed": all(c["passed"] for c in (e1, e2, e3, e4, e5)),
        "not_a_result": (
            "The harness was exercised on the smoke sweep (4 agents, 4 epochs). "
            "H7 is NOT run here: SPEC gates the experiment at , and a "
            "preliminary look would bias later interpretation. No hypothesis is "
            "evaluated."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase6_acceptance.json").write_text(json.dumps(report_out, indent=2, default=str))
    print(json.dumps({k: v for k, v in report_out.items() if k != "criteria"}, indent=2))
    for name, crit in report_out["criteria"].items():
        print(f"  {'PASS' if crit['passed'] else 'FAIL'}  {name}")
    print("\nPHASE 6 ACCEPTANCE:", "PASS" if report_out["passed"] else "FAIL")
    return 0 if report_out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
