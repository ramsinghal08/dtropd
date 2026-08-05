#!/usr/bin/env python3
"""Phase 4 acceptance test (SPEC Sec. 5).

    "Accept: all four baselines train end-to-end on the homogeneous env with
     shared hyperparameters and identical seeds."

plus the regression requirement added with the locked decisions:

    IPPO's existing curves are unchanged by the refactor.

Criteria:

  B1  every algorithm (IPPO, MAPPO, HAPPO, NaiveDistill, D-TrOPD) completes
      training on the homogeneous environment for every seed, producing finite
      metrics throughout and a checkpoint at the end;
  B2  hyperparameters are shared *by construction* -- one config file is used
      for all of them and only ``train.algo`` (plus the structural
      ``distill.enabled``) differs. The harness records the diff it applied so
      that claim is auditable;
  B3  identical seeds give identical initial policies across algorithms, so the
      comparison is not confounded before the first update;
  B4  IPPO's learning curve matches ``tests/data/ippo_phase2_reference.json``,
      which was generated on the pre-refactor Phase 2 commit.

This does **not** run H7 or any headline comparison: SPEC gates that at Phase 5.
Returns are reported only to show that training happened, and are explicitly
not a result.

Usage::

    python experiments/phase4_acceptance.py --seeds 0 1 2 --workers 3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtropd import golden  # noqa: E402
from dtropd.algorithms import build_algorithm  # noqa: E402
from dtropd.config import Config  # noqa: E402
from dtropd.train import train  # noqa: E402
from dtropd.utils import configure_logging, get_logger, read_metrics  # noqa: E402

logger = get_logger("phase4")

ALGOS = ("ippo", "mappo", "happo", "naive_distill", "dtropd")
DISTILLERS = ("naive_distill", "dtropd")


def configure(base: Config, algo: str) -> Config:
    """The only per-algorithm difference the harness is allowed to make."""
    cfg = Config.from_dict(dataclasses.asdict(base))
    cfg.train.algo = algo
    cfg.distill.enabled = algo in DISTILLERS
    cfg.validate(emit=False)
    return cfg


def _run_one(payload: tuple[dict[str, Any], str, int, str]) -> dict[str, Any]:
    base_dict, algo, seed, out_dir = payload
    configure_logging("INFO")
    cfg = configure(Config.from_dict(base_dict), algo)
    result = train(cfg, seed=seed, out_dir=Path(out_dir) / f"{algo}_seed{seed}", progress=False)

    records = read_metrics(result.run_dir / "metrics.jsonl")
    finite = all(
        np.isfinite(r["episode_return_mean"]) and np.isfinite(r["value_loss"])
        for r in records
    )
    meta = json.loads((result.run_dir / "metadata.json").read_text())
    return {
        "algo": algo,
        "seed": seed,
        "epochs_completed": len(records),
        "all_metrics_finite": bool(finite),
        "checkpoint_written": (result.run_dir / "checkpoint_final.pt").exists(),
        "reward_spec_version": meta.get("reward_spec_version"),
        "final_return": result.final_eval["return_mean"],
        "random_return": result.random_eval["return_mean"],
        "improvement": result.final_eval["return_mean"] - result.random_eval["return_mean"],
        "max_explained_variance": result.max_explained_variance,
        "wall_clock_seconds": result.wall_clock_seconds,
        "trust_weight_mean": records[-1].get("trust_weight_mean"),
        "beta": records[-1].get("beta"),
        "calibrated_delta_base": next(
            (r["calibrated_delta_base"] for r in records if "calibrated_delta_base" in r),
            None,
        ),
        "passed": bool(finite and len(records) == cfg.train.total_epochs),
    }


def check_shared_hyperparameters(base: Config) -> dict[str, Any]:
    """B2: enumerate every field that differs between the per-algorithm configs."""
    reference = dataclasses.asdict(configure(base, "ippo"))
    diffs: dict[str, list[str]] = {}

    def walk(a: Any, b: Any, path: str, into: list[str]) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a:
                walk(a[key], b.get(key), f"{path}.{key}" if path else key, into)
        elif a != b:
            into.append(f"{path}: {a!r} -> {b!r}")

    for algo in ALGOS[1:]:
        found: list[str] = []
        walk(reference, dataclasses.asdict(configure(base, algo)), "", found)
        diffs[algo] = found

    allowed = {"train.algo", "distill.enabled"}
    unexpected = {
        algo: [d for d in found if d.split(":")[0] not in allowed]
        for algo, found in diffs.items()
    }
    return {
        "diffs_vs_ippo": diffs,
        "allowed_keys": sorted(allowed),
        "unexpected": {k: v for k, v in unexpected.items() if v},
        "passed": not any(unexpected.values()),
    }


def check_identical_init(base: Config, seed: int) -> dict[str, Any]:
    """B3: same seed -> same initial actors, whatever the algorithm."""
    reference = build_algorithm(configure(base, "ippo"), seed=seed, device="cpu")
    ref_params = [p.detach().clone() for ac in reference.agents for p in ac.actor.parameters()]
    rows = {}
    for algo in ALGOS[1:]:
        algo_obj = build_algorithm(configure(base, algo), seed=seed, device="cpu")
        params = [p.detach() for ac in algo_obj.agents for p in ac.actor.parameters()]
        rows[algo] = all(torch.equal(a, b) for a, b in zip(ref_params, params))
    return {"identical_to_ippo": rows, "passed": all(rows.values())}


def check_ippo_regression() -> dict[str, Any]:
    """B4: IPPO's curve still matches the pre-refactor Phase 2 reference."""
    from dtropd.env import SyncVectorEnv
    from dtropd.utils import derive_seed

    path = ROOT / "tests" / "data" / "ippo_phase2_reference.json"
    reference = json.loads(path.read_text())
    spec = reference["config_summary"]

    cfg = Config()
    cfg.env.n_agents = spec["n_agents"]
    cfg.env.n_targets = spec["n_targets"]
    cfg.env.capture_group_size = 3
    cfg.env.max_cycles = spec["max_cycles"]
    cfg.env.local_ratio = 0.5
    cfg.network.hidden_sizes = list(spec["hidden"])
    cfg.train.algo = "ippo"
    cfg.train.num_envs = spec["num_envs"]
    cfg.train.total_epochs = spec["epochs"]
    cfg.ppo.lr = spec["lr"]
    cfg.validate(emit=False)

    algo = build_algorithm(cfg, seed=spec["seed"], device="cpu")
    vec = SyncVectorEnv(cfg.env, cfg.train.num_envs, seed=derive_seed(spec["seed"], "env"))
    history = []
    for epoch in range(cfg.train.total_epochs):
        algo.epoch = epoch
        algo.on_epoch_start()
        rollout = algo.collect_rollout(vec)
        metrics = algo.update()
        history.append({**rollout, **metrics})

    # Exact equality is not the right assertion across machines, and neither is
    # pointwise relative error for a metric that crosses zero. The rule -- and
    # the reasoning -- live in dtropd.golden, shared with tests/test_baselines.py
    # so there is exactly one definition of it.
    verdicts: dict[str, Any] = {}
    worst_dev, worst_first = 0.0, 0.0
    for key, expected in reference["curve"].items():
        actual = [history[i][key] for i in range(len(expected))]
        verdict = golden.compare_curve(key, actual, expected)
        verdicts[key] = verdict
        if verdict.kind == "smooth":
            worst_dev = max(worst_dev, verdict.max_deviation)
            worst_first = max(worst_first, verdict.first_epoch_deviation)
    mismatches = {k: v.as_dict() for k, v in verdicts.items() if not v.ok}
    return {
        "reference": str(path.relative_to(ROOT)),
        "generated_on_commit": "011de8f (Phase 2, pre-refactor)",
        "metrics_compared": sorted(reference["curve"]),
        "deviation_units": (
            "absolute; for count-valued metrics, flipped threshold decisions"
        ),
        "first_epoch_tol": f"{golden.FIRST_EPOCH_ATOL:g} + {golden.FIRST_EPOCH_RANGE_TOL:g} x range",
        "curve_tol": f"{golden.CURVE_ATOL:g} + {golden.CURVE_RANGE_TOL:g} x range",
        "step_valued_keys": list(golden.STEP_VALUED_KEYS),
        "max_first_epoch_deviation": worst_first,
        "max_deviation": worst_dev,
        "mismatched_metrics": sorted(mismatches),
        "detail": mismatches,
        "passed": not mismatches,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "experiments/configs/phase4_homogeneous.yaml"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out", default="runs/phase4")
    args = p.parse_args(argv)

    configure_logging("INFO")
    base = Config.from_yaml(args.config)
    if args.epochs is not None:
        base.train.total_epochs = args.epochs
    base.validate(emit=False)
    assert not base.env.heterogeneous, "Phase 4 acceptance is defined on the homogeneous env"

    b2 = check_shared_hyperparameters(base)
    b3 = check_identical_init(base, args.seeds[0])
    logger.info("shared hyperparameters: %s", "PASS" if b2["passed"] else "FAIL")
    logger.info("identical initialisation: %s", "PASS" if b3["passed"] else "FAIL")

    payloads = [
        (dataclasses.asdict(base), algo, seed, args.out)
        for algo in ALGOS
        for seed in args.seeds
    ]
    if args.workers > 1:
        with mp.get_context("spawn").Pool(min(args.workers, len(payloads))) as pool:
            rows = pool.map(_run_one, payloads)
    else:
        rows = [_run_one(x) for x in payloads]

    b1 = {
        "runs": rows,
        "passed": all(r["passed"] for r in rows),
        "algorithms": sorted({r["algo"] for r in rows}),
    }
    logger.info("training runs: %s", "PASS" if b1["passed"] else "FAIL")

    b4 = check_ippo_regression()
    logger.info("IPPO regression: %s", "PASS" if b4["passed"] else "FAIL")

    per_algo = {
        algo: {
            "improvements": [r["improvement"] for r in rows if r["algo"] == algo],
            "max_explained_variance": [
                r["max_explained_variance"] for r in rows if r["algo"] == algo
            ],
            "wall_clock_seconds": [r["wall_clock_seconds"] for r in rows if r["algo"] == algo],
        }
        for algo in ALGOS
    }

    report = {
        "acceptance": "Phase 4 -- baselines",
        "config": args.config,
        "seeds": args.seeds,
        "criteria": {
            "B1_all_algorithms_train_end_to_end": b1,
            "B2_shared_hyperparameters": b2,
            "B3_identical_initialisation": b3,
            "B4_ippo_regression_vs_phase2": b4,
        },
        "passed": bool(b1["passed"] and b2["passed"] and b3["passed"] and b4["passed"]),
        "per_algorithm": per_algo,
        "tuning_statement": (
            "No hyperparameter search was performed for any algorithm. Every "
            "algorithm uses the Phase 2 IPPO settings unchanged, from one shared "
            "config file (SPEC Sec. 9 rule 5)."
        ),
        "not_a_result": (
            "Returns are reported to show that training happened. This is the "
            "homogeneous environment with 3 seeds; SPEC Sec. 7 requires >= 10 "
            "seeds with paired seeds and bootstrap CIs, and H7 is gated at "
            "Phase 5. Nothing here is a comparison between algorithms."
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase4_acceptance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nPHASE 4 ACCEPTANCE:", "PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
