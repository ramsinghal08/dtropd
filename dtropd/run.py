"""CLI entrypoint.

Examples
--------
    python run.py --help
    python run.py train --config experiments/configs/smoke.yaml
    python run.py train --config experiments/configs/h7_ippo_homogeneous.yaml --seeds 3
    python run.py eval  --config experiments/configs/smoke.yaml --checkpoint runs/.../checkpoint_final.pt
    python run.py plot  --runs runs/ippo_seed0 runs/ippo_seed1 --out figures/curve.png
    python run.py validate  --config experiments/configs/smoke.yaml
    python run.py calibrate --config experiments/configs/dtropd_heterogeneous.yaml \
        --out experiments/configs/dtropd_heterogeneous_static.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .config import Config, ConfigError, load_config
from .utils import configure_logging, get_logger

logger = get_logger("dtropd.run")


def _parse_overrides(pairs: Sequence[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "D-TrOPD: Decentralized Trust-Region Policy Distillation. "
            "Training, evaluation and figure generation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO", help="logging verbosity")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=str, default=None, help="path to a YAML config")
        p.add_argument(
            "--set",
            dest="overrides",
            action="append",
            metavar="KEY=VALUE",
            help="dotted-key config override, e.g. --set train.total_epochs=5",
        )
        p.add_argument(
            "--no-strict",
            action="store_true",
            help="downgrade the kappa*eta range check from an error to a warning",
        )

    p_train = sub.add_parser("train", help="train an algorithm")
    add_common(p_train)
    p_train.add_argument("--algo", type=str, default=None, help="override train.algo")
    p_train.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="number of seeds (e.g. 10) or an explicit list (e.g. 0,1,2)",
    )
    p_train.add_argument("--device", type=str, default=None, help="cpu | cuda")
    p_train.add_argument("--out", type=str, default=None, help="output directory")
    p_train.add_argument("--epochs", type=int, default=None, help="override train.total_epochs")
    p_train.add_argument("--quiet", action="store_true", help="suppress per-epoch logging")

    p_eval = sub.add_parser("eval", help="evaluate a checkpoint")
    add_common(p_eval)
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--episodes", type=int, default=32)

    p_plot = sub.add_parser("plot", help="regenerate a figure from stored metrics")
    p_plot.add_argument("--runs", nargs="+", required=True, help="run directories")
    p_plot.add_argument("--out", type=str, required=True, help="output image path")
    p_plot.add_argument("--key", type=str, default="episode_return_mean")
    p_plot.add_argument("--x-key", type=str, default="env_steps")

    p_val = sub.add_parser("validate", help="load and validate a config, then print it")
    add_common(p_val)

    p_cal = sub.add_parser(
        "calibrate",
        help="measure the D_ij distribution and emit a calibrated config (SPEC D1)",
    )
    add_common(p_cal)
    p_cal.add_argument(
        "--out", type=str, default=None, help="write the calibration ARTIFACT (json) here"
    )
    p_cal.add_argument("--epochs", type=int, default=10, help="calibration window length")
    p_cal.add_argument("--seed", type=int, default=None)

    p_sweep = sub.add_parser(
        "sweep-check",
        help="refuse a sweep whose arms were calibrated separately (SPEC D6)",
    )
    p_sweep.add_argument(
        "--configs", nargs="+", required=True, help="one config per arm of the sweep"
    )
    p_sweep.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="permit a sweep in which no arm declares a calibration artifact",
    )

    p_analyze = sub.add_parser(
        "analyze", help="compute every H1-H7 measurement from one run's metrics file"
    )
    p_analyze.add_argument("--run", required=True)
    p_analyze.add_argument("--out", default=None)

    p_sweepr = sub.add_parser(
        "sweep", help="run a multi-seed sweep and emit the comparison figure (Phase 6)"
    )
    p_sweepr.add_argument("--config", required=True, help="a sweep YAML")
    p_sweepr.add_argument(
        "--seeds", type=str, default=None, help="count (e.g. 10) or list (0,1,2)"
    )
    p_sweepr.add_argument("--out", type=str, default=None, help="sweep output directory")
    p_sweepr.add_argument(
        "--no-resume", action="store_true", help="re-run cells that already completed"
    )
    p_sweepr.add_argument(
        "--report-only",
        action="store_true",
        help="skip training; rebuild the report and figure from stored metrics",
    )
    p_sweepr.add_argument("--figure", type=str, default=None, help="figure output path")
    p_sweepr.add_argument(
        "--workers",
        type=int,
        default=1,
        help="run cells in parallel processes; does not change any result, "
        "since each cell is independently seeded and writes its own directory",
    )

    p_grid = sub.add_parser(
        "calibration-grid",
        help="derive H7's delta_base sensitivity grid (SPEC D7-AMENDED)",
    )
    add_common(p_grid)
    p_grid.add_argument("--windows", type=int, nargs="+", default=None)
    p_grid.add_argument("--out", type=str, default="experiments/calibrations")
    p_grid.add_argument("--seed", type=int, default=None)

    p_smoke = sub.add_parser(
        "smoke", help="2-agent CPU smoke run (SPEC Sec. 9 rule 6: always works)"
    )
    p_smoke.add_argument("--steps", type=int, default=200, help="minimum env steps to run")
    p_smoke.add_argument(
        "--algo", type=str, default="ippo", help="algorithm to smoke-test"
    )
    return parser


def _resolve_seeds(spec: str | None, default: int) -> list[int]:
    if spec is None:
        return [default]
    if "," in spec:
        return [int(s) for s in spec.split(",") if s.strip()]
    value = int(spec)
    return list(range(value))


def _load(args: argparse.Namespace) -> Config:
    overrides = _parse_overrides(getattr(args, "overrides", None))
    if getattr(args, "algo", None):
        overrides["train.algo"] = args.algo
    if getattr(args, "device", None):
        overrides["train.device"] = args.device
    if getattr(args, "epochs", None):
        overrides["train.total_epochs"] = args.epochs
    return load_config(
        args.config, overrides, strict=not getattr(args, "no_strict", False)
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level if hasattr(args, "log_level") else "INFO")

    try:
        if args.command == "train":
            return _cmd_train(args)
        if args.command == "eval":
            return _cmd_eval(args)
        if args.command == "plot":
            return _cmd_plot(args)
        if args.command == "validate":
            return _cmd_validate(args)
        if args.command == "calibrate":
            return _cmd_calibrate(args)
        if args.command == "sweep-check":
            return _cmd_sweep_check(args)
        if args.command == "sweep":
            return _cmd_sweep(args)
        if args.command == "calibration-grid":
            return _cmd_calibration_grid(args)
        if args.command == "analyze":
            return _cmd_analyze(args)
        if args.command == "smoke":
            return _cmd_smoke(args)
    except ConfigError as exc:
        logger.error("config error: %s", exc)
        return 2
    except NotImplementedError as exc:
        logger.error("%s", exc)
        return 3
    return 1


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import aggregate_results, train_many_seeds

    cfg = _load(args)
    seeds = _resolve_seeds(args.seeds, cfg.train.seed)
    results = train_many_seeds(cfg, seeds, out_dir=args.out, progress=not args.quiet)
    summary = aggregate_results(results)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_config

    cfg = _load(args)
    print(json.dumps(evaluate_config(cfg, args.checkpoint, n_episodes=args.episodes), indent=2))
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from .plot import plot_learning_curves

    out = plot_learning_curves(args.runs, args.out, key=args.key, x_key=args.x_key)
    print(str(out))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    import yaml

    cfg = _load(args)
    warnings_out = cfg.validate(strict=not args.no_strict, emit=False)
    print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    for w in warnings_out:
        logger.warning("%s", w)
    print(f"# validation: OK ({len(warnings_out)} warning(s))")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    """SPEC Sec. 5 Phase 6: one command, from scratch, to the comparison figure."""
    from .plot import plot_comparison
    from .sweep import SweepError, build_plan, run_sweep, sweep_report

    seeds: Any = args.seeds
    if seeds is not None:
        seeds = [int(s) for s in seeds.split(",")] if "," in seeds else int(seeds)

    try:
        plan = build_plan(args.config, seeds=seeds, out_dir=args.out)
    except SweepError as exc:
        logger.error("sweep config rejected: %s", exc)
        return 6

    manifest_path = plan.out_dir / "manifest.json"
    try:
        if args.report_only:
            if not manifest_path.exists():
                logger.error("no manifest at %s; run the sweep first", manifest_path)
                return 7
            manifest = json.loads(manifest_path.read_text())
        else:
            manifest = run_sweep(
                plan,
                resume=not args.no_resume,
                progress=False,
                workers=max(1, int(getattr(args, "workers", 1))),
            )
    except SweepError as exc:
        logger.error("sweep rejected: %s", exc)
        return 6

    report = sweep_report(manifest)
    (plan.out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

    # Each hypothesis plots ITS OWN decision quantity. Final return is the
    # outcome variable for H7 and nothing else; drawing it for H1/H2/H3/H6
    # produces an authoritative-looking figure of the wrong thing, which
    # happened three times before the default was changed.
    from .hypothesis_figures import RETURN_IS_THE_OUTCOME, figure_for_sweep

    hyp = figure_for_sweep(plan.name, manifest, plan.out_dir / "figures")
    if hyp:
        report["hypothesis_figure"] = hyp
        (plan.out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
        if not any(plan.name.startswith(p) for p in RETURN_IS_THE_OUTCOME):
            logger.info(
                "%s decides on its own quantity, not final return; the return "
                "comparison figure is skipped for it.", plan.name,
            )
            print(json.dumps({
                "manifest": str(manifest_path),
                "report": str(plan.out_dir / "report.json"),
                "figure": hyp.get("figure"),
            }, indent=2))
            return 0

    figure = Path(args.figure) if args.figure else plan.out_dir / "figures" / "comparison.png"
    if not report.get("per_arm"):
        # An environment-property sweep (H4/H5 scale, the D11(c) probe) has no
        # algorithm arms, so the comparison figure has nothing to draw and its
        # absence is not a failure. Erroring here made a legitimate sweep type
        # unable to exit cleanly -- and, chained with &&, silently prevented the
        # next sweep from starting at all.
        logger.info(
            "no algorithm arms in this sweep, so no comparison figure: %s",
            "every arm is purpose: environment_property",
        )
        figure = None
    else:
        try:
            plot_comparison(report, figure, title=f"{plan.name}: final return")
            logger.info("wrote %s", figure)
        except Exception as exc:  # pragma: no cover - matplotlib is optional
            logger.error("could not render the figure: %s", exc)
            return 8

    print(json.dumps({
        "manifest": str(manifest_path),
        "report": str(plan.out_dir / "report.json"),
        "figure": str(figure) if figure else None,
    }, indent=2))
    return 0


def _cmd_calibration_grid(args: argparse.Namespace) -> int:
    """SPEC Sec. 12 / D7-AMENDED: derive the grid, check the spread, pin artifacts."""
    from .calibration import D7_WINDOW_LENGTHS, d7_grid_from_records

    cfg = _load(args)
    windows = args.windows or list(D7_WINDOW_LENGTHS)
    seed = args.seed if args.seed is not None else cfg.train.seed
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = {}
    for window in windows:
        logger.info("calibrating with a %d-epoch window", window)
        record = _measure_calibration(cfg, window=window, seed=seed, source=args.config)
        path = out_dir / f"{Path(args.config).stem}_w{window}.json"
        record.save(path)
        records[window] = record
        logger.info("  delta_base %.6g  gamma %.6g  -> %s", record.delta_base, record.gamma_attenuation, path)

    grid = d7_grid_from_records(records)
    grid["artifacts"] = {
        str(w): str(out_dir / f"{Path(args.config).stem}_w{w}.json") for w in windows
    }
    (out_dir / f"{Path(args.config).stem}_d7_grid.json").write_text(
        json.dumps(grid, indent=2, default=str)
    )
    print(json.dumps(grid, indent=2, default=str))
    if grid["widened"]:
        logger.warning("%s", grid["widening_reason"])
    return 0


def _measure_calibration(cfg: Config, window: int, seed: int, source: str | None):
    """Run the D6 procedure once at a given window length."""
    import copy as _copy

    from .algorithms import build_algorithm
    from .env import SyncVectorEnv
    from .utils import derive_seed

    run_cfg = _copy.deepcopy(cfg)
    run_cfg.train.algo = "dtropd"
    run_cfg.distill.enabled = True
    run_cfg.distill.calibration = "warmup"
    run_cfg.distill.calibration_file = None
    run_cfg.distill.calibration_epochs = window
    run_cfg.train.total_epochs = max(run_cfg.train.total_epochs, window)
    run_cfg.validate(emit=False)

    algo = build_algorithm(run_cfg, seed=seed, device=run_cfg.train.device)
    vec = SyncVectorEnv(run_cfg.env, run_cfg.train.num_envs, seed=derive_seed(seed, "env"))
    for epoch in range(window):
        algo.epoch = epoch
        algo.on_epoch_start()
        algo.collect_rollout(vec)
        algo.update()
    if algo.calibration_record is None:
        raise RuntimeError(f"calibration produced no samples at window {window}")
    algo.calibration_record.source_config = source
    return algo.calibration_record


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """SPEC Sec. 11 / D6: the calibration procedure as a single command.

    Emits the **artifact** (a CalibrationRecord JSON), which is what every arm
    of a comparison pins via ``distill.calibration_file``. Runs the real
    training loop with ``distill.calibration: warmup``, so the offline command
    and the in-run path execute the same code and cannot drift apart; beta is
    held at exactly 0 for the whole window, so the measurement is taken with the
    gate inactive.
    """
    from .algorithms import trust_weight

    cfg = _load(args)
    seed = args.seed if args.seed is not None else cfg.train.seed
    window = args.epochs
    record = _measure_calibration(cfg, window=window, seed=seed, source=args.config)

    before = float(
        trust_weight(
            torch.tensor(record.quantiles["p50"]),
            cfg.distill.delta_base,
            cfg.distill.gamma_attenuation,
            cfg.distill.attenuation,
        )
    )
    after = float(
        trust_weight(
            torch.tensor(record.quantiles["p50"]),
            record.delta_base,
            record.gamma_attenuation,
            cfg.distill.attenuation,
        )
    )
    report = {
        "procedure": "SPEC Sec. 11 / D6",
        "source_config": args.config,
        "seed": seed,
        "window_epochs": record.window_epochs,
        "pooled_epochs": record.pooled_epochs,
        "edges_sampled": record.edge_count,
        "D_ij_quantiles": record.quantiles,
        "calibration_id": record.calibration_id,
        "before": {
            "delta_base": cfg.distill.delta_base,
            "gamma_attenuation": cfg.distill.gamma_attenuation,
            "w_at_median_D": before,
            "gate_inert": bool(before > 0.95),
        },
        "after": {
            "delta_base": record.delta_base,
            "gamma_attenuation": record.gamma_attenuation,
            "w_at_median_D": after,
            "gate_inert": bool(after > 0.95),
        },
    }
    if args.out:
        record.save(args.out)
        report["artifact"] = str(args.out)
        logger.info("wrote calibration artifact to %s", args.out)
    print(json.dumps(report, indent=2, default=str))
    if report["before"]["gate_inert"]:
        logger.warning(
            "the source config's gate was inert (w=%.4f at the median D_ij): "
            "results from it would have been NaiveDistill, not D-TrOPD.",
            before,
        )
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    """A <2-minute, 2-agent CPU run (SPEC Sec. 9, rule 6)."""
    from .train import train

    cfg = Config()
    cfg.env.n_agents = 2
    cfg.env.n_targets = 1
    cfg.env.capture_group_size = 2
    cfg.env.max_cycles = 25
    cfg.train.algo = args.algo
    cfg.distill.enabled = args.algo == "dtropd"
    cfg.train.num_envs = 4
    cfg.train.total_epochs = max(1, args.steps // (25 * 4))
    cfg.train.eval_interval = 0
    cfg.train.eval_episodes = 4
    cfg.train.log_dir = "runs/smoke"
    cfg.validate(emit=False)
    result = train(cfg, seed=0, progress=True)
    print(json.dumps(result.summary(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
