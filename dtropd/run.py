"""CLI entrypoint.

Examples
--------
    python run.py --help
    python run.py train --config experiments/configs/smoke.yaml
    python run.py train --config experiments/configs/h7_ippo_homogeneous.yaml --seeds 3
    python run.py eval  --config experiments/configs/smoke.yaml --checkpoint runs/.../checkpoint_final.pt
    python run.py plot  --runs runs/ippo_seed0 runs/ippo_seed1 --out figures/curve.png
    python run.py validate --config experiments/configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

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
