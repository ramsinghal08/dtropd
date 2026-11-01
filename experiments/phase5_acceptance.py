#!/usr/bin/env python3
"""acceptance test (SPEC Sec. 5).

    "Accept: a single training run produces a metrics file from which every
     H1-H7 measurement in Sec. 6 can be computed without re-running."

Demonstrated literally: run ONE fully-instrumented training job, then run the
analysis over its metrics file alone and emit a value for each of H1-H7.

Criteria:

  C1  the run completes and writes metrics.jsonl + metadata.json, with the
      metrics schema version and reward spec version stamped;
  C2  every H1-H7 measurement is computable from that one file -- a hypothesis
      that is not is a failure, not a problem;
  C3  the D6 calibration artifact is emitted, self-consistent, and records the
      window, the pooled epochs, the edge count and the quantile table;
  C4  the shared-calibration check refuses a sweep whose arms were calibrated
      separately (SPEC Sec. 11 / D6);
  C5  the D9 consensus-pull instrumentation is present for a distillation arm
      *and* for the IPPO control, so the cost has something to be measured
      against.

Usage::

    python experiments/phase5_acceptance.py --out runs/phase5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtropd.analysis import HYPOTHESES, analyze_run, consensus_pull_cost  # noqa: E402
from dtropd.calibration import (  # noqa: E402
    CalibrationError,
    CalibrationRecord,
    assert_shared_calibration,
)
from dtropd.config import Config  # noqa: E402
from dtropd.train import train  # noqa: E402
from dtropd.utils import configure_logging, get_logger, read_metrics  # noqa: E402

logger = get_logger("phase5")


def _run(cfg: Config, algo: str, seed: int, out_dir: Path) -> dict[str, Any]:
    run_cfg = Config.from_dict(dataclasses.asdict(cfg))
    run_cfg.train.algo = algo
    run_cfg.distill.enabled = algo in ("dtropd", "naive_distill")
    if algo not in ("dtropd",):
        # only D-TrOPD measures a radius; the others declare, not calibrate
        run_cfg.distill.calibration = "none"
    run_cfg.validate(emit=False)
    result = train(run_cfg, seed=seed, out_dir=out_dir / f"{algo}_seed{seed}", progress=False)
    return {
        "algo": algo,
        "seed": seed,
        "run_dir": str(result.run_dir),
        "final_return": result.final_eval["return_mean"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "experiments/configs/phase5_instrumented.yaml"))
    p.add_argument("--out", default="runs/phase5")
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args(argv)

    configure_logging("INFO")
    cfg = Config.from_yaml(args.config)
    if args.epochs is not None:
        cfg.train.total_epochs = args.epochs
    cfg.validate(emit=False)
    out = Path(args.out)

    # --- C1: one fully-instrumented run ----------------------------------- #
    logger.info("running the instrumented D-TrOPD job")
    main_run = _run(cfg, "dtropd", 0, out)
    run_dir = Path(main_run["run_dir"])
    records = read_metrics(run_dir / "metrics.jsonl")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    c1 = {
        "epochs": len(records),
        "metrics_schema_version": metadata.get("metrics_schema_version"),
        "reward_spec_version": metadata.get("reward_spec_version"),
        "passed": bool(
            records
            and metadata.get("metrics_schema_version")
            and metadata.get("reward_spec_version")
        ),
    }

    # --- C2: every hypothesis measurable from that ONE file ---------------- #
    results = analyze_run(records, metadata)
    c2 = {
        "hypotheses": {k: results[k].to_dict() for k in HYPOTHESES},
        "uncomputable": [k for k in HYPOTHESES if not results[k].computable],
    }
    c2["passed"] = not c2["uncomputable"]
    logger.info("H1-H7 from a single file: %s", "PASS" if c2["passed"] else "FAIL")

    # --- C3: the calibration artifact -------------------------------------- #
    artifact_path = run_dir / "calibration.json"
    c3: dict[str, Any] = {"artifact": str(artifact_path), "passed": False}
    if artifact_path.exists():
        record = CalibrationRecord.load(artifact_path)
        c3.update(
            {
                "calibration_id": record.calibration_id,
                "delta_base": record.delta_base,
                "gamma_attenuation": record.gamma_attenuation,
                "window_epochs": record.window_epochs,
                "pooled_epochs": record.pooled_epochs,
                "edge_count": record.edge_count,
                "quantiles": record.quantiles,
                "reward_spec_version": record.reward_spec_version,
                "git_commit": record.git_commit,
                "rule": record.quantile_definition,
                "passed": bool(
                    record.window_epochs == cfg.distill.calibration_epochs
                    and len(record.pooled_epochs) == cfg.distill.calibration_pool_epochs
                    and record.edge_count > 0
                    and record.quantiles
                ),
            }
        )

    # --- C4: the shared-calibration check refuses a mixed sweep ------------ #
    c4: dict[str, Any] = {}
    if artifact_path.exists():
        shared = CalibrationRecord.load(artifact_path)
        other = CalibrationRecord.from_dict(
            {**shared.to_dict(), "created_at": "1999-01-01T00:00:00+0000"}
        )
        try:
            assert_shared_calibration([("dtropd", shared), ("naive_distill", shared)])
            same_ok = True
        except CalibrationError:
            same_ok = False
        try:
            assert_shared_calibration([("dtropd", shared), ("naive_distill", other)])
            mixed_rejected = False
        except CalibrationError as exc:
            mixed_rejected = True
            c4["rejection_message"] = str(exc)[:160]
        try:
            assert_shared_calibration([("dtropd", shared), ("ippo", None)])
            missing_rejected = False
        except CalibrationError:
            missing_rejected = True
        c4.update(
            {
                "shared_accepted": same_ok,
                "separately_calibrated_rejected": mixed_rejected,
                "undeclared_arm_rejected": missing_rejected,
                "passed": bool(same_ok and mixed_rejected and missing_rejected),
            }
        )

    # --- C5: D9 has a control to be measured against ----------------------- #
    logger.info("running the IPPO control for the consensus-pull comparison")
    control = _run(cfg, "ippo", 0, out)
    control_records = read_metrics(Path(control["run_dir"]) / "metrics.jsonl")
    key = "consensus_pull/policy_dispersion_p50"
    c5 = {
        "distiller_has_dispersion": any(key in r for r in records),
        "control_has_dispersion": any(key in r for r in control_records),
        "distiller_has_grad_share": any(
            "consensus_pull/social_grad_fraction_mean" in r for r in records
        ),
        "example_cost": consensus_pull_cost(
            [control["final_return"]], [main_run["final_return"]]
        ),
    }
    c5["passed"] = bool(
        c5["distiller_has_dispersion"]
        and c5["control_has_dispersion"]
        and c5["distiller_has_grad_share"]
    )

    report = {
        "acceptance": "-- instrumentation",
        "config": args.config,
        "criteria": {
            "C1_run_and_schema": c1,
            "C2_all_hypotheses_from_one_file": c2,
            "C3_calibration_artifact": c3,
            "C4_shared_calibration_enforced": c4,
            "C5_consensus_pull_has_a_control": c5,
        },
        "passed": bool(
            c1["passed"] and c2["passed"] and c3["passed"] and c4.get("passed") and c5["passed"]
        ),
        "not_a_result": (
            "This runs one seed to demonstrate measurability. No hypothesis is "
            "evaluated here; H7 and the multi-seed sweeps are gated at ."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase5_acceptance.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "criteria"}, indent=2))
    for name, crit in report["criteria"].items():
        print(f"  {'PASS' if crit.get('passed') else 'FAIL'}  {name}")
    print("\nPHASE 5 ACCEPTANCE:", "PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
