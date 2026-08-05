"""Training loop, logging, checkpointing and seeding."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .algorithms import BaseAlgorithm, build_algorithm
from .config import Config
from .env import SyncVectorEnv
from .evaluate import EvalResult, evaluate_policy, evaluate_random_policy
from .utils import (
    MetricsWriter,
    configure_logging,
    derive_seed,
    get_logger,
    run_metadata,
    seed_everything,
    set_torch_threads,
)

__all__ = ["TrainResult", "train", "train_many_seeds"]

logger = get_logger("dtropd.train")


@dataclass
class TrainResult:
    run_dir: Path
    seed: int
    algo: str
    epochs: int
    final_eval: dict[str, Any]
    random_eval: dict[str, Any]
    best_return: float
    final_explained_variance: float
    max_explained_variance: float
    wall_clock_seconds: float
    history: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "seed": self.seed,
            "algo": self.algo,
            "epochs": self.epochs,
            "final_return": self.final_eval.get("return_mean"),
            "random_return": self.random_eval.get("return_mean"),
            "improvement": (
                self.final_eval.get("return_mean", float("nan"))
                - self.random_eval.get("return_mean", float("nan"))
            ),
            "best_return": self.best_return,
            "final_explained_variance": self.final_explained_variance,
            "max_explained_variance": self.max_explained_variance,
            "wall_clock_seconds": self.wall_clock_seconds,
        }


def _run_dir(cfg: Config, seed: int) -> Path:
    name = cfg.train.run_name or f"{cfg.train.algo}"
    return Path(cfg.train.log_dir) / f"{name}_seed{seed}"


def train(
    cfg: Config,
    seed: int | None = None,
    *,
    out_dir: str | Path | None = None,
    progress: bool = True,
) -> TrainResult:
    """Run one training job and return its summary.

    Every random source is derived from ``seed``; the run directory receives a
    ``metadata.json`` header (config + git commit + seed) and a ``metrics.jsonl``
    stream, per SPEC Sec. 4.
    """
    seed = cfg.train.seed if seed is None else int(seed)
    cfg.validate(emit=False)

    set_torch_threads(cfg.train.torch_threads)
    seed_everything(seed)

    run_dir = Path(out_dir) if out_dir is not None else _run_dir(cfg, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    algo: BaseAlgorithm = build_algorithm(cfg, seed=seed, device=cfg.train.device)
    vec_env = SyncVectorEnv(
        cfg.env, cfg.train.num_envs, seed=derive_seed(seed, "env"), autoreset=True
    )

    metadata = run_metadata(
        cfg,
        seed,
        algo=cfg.train.algo,
        # SPEC Sec. 10 / D3: every run records which frozen reward it optimised.
        reward_spec_version=cfg.env.reward_spec_version,
        obs_dim=cfg.env.obs_dim,
        act_dim=cfg.env.act_dim,
        batch_size_per_agent=cfg.batch_size,
        neighbors=algo.neighbors,
        actor_snapshot_bytes=algo.snapshot_bytes,
    )
    writer = MetricsWriter(run_dir, metadata)

    random_eval = evaluate_random_policy(
        cfg.env, n_episodes=max(8, cfg.train.eval_episodes), seed=derive_seed(seed, "random_eval")
    )
    logger.info(
        "[seed %d] random-policy reference return: %.3f +/- %.3f",
        seed,
        random_eval.return_mean,
        random_eval.return_sem,
    )

    history: list[dict[str, Any]] = []
    best_return = -float("inf")
    max_ev = -float("inf")
    last_ev = float("nan")
    final_eval: EvalResult | None = None
    start = time.time()

    for epoch in range(cfg.train.total_epochs):
        algo.epoch = epoch
        t0 = time.time()
        # SPEC Sec. 2.1 step 1: snapshots are broadcast and frozen before any
        # data is collected. No-op for algorithms that exchange nothing.
        algo.on_epoch_start()
        rollout_stats = algo.collect_rollout(vec_env)
        t1 = time.time()
        update_stats = algo.update()
        t2 = time.time()

        record: dict[str, Any] = {
            "epoch": epoch,
            "seed": seed,
            "algo": cfg.train.algo,
            "env_steps": (epoch + 1) * cfg.batch_size,
            "time/rollout": t1 - t0,
            "time/update": t2 - t1,
            "time/elapsed": t2 - start,
        }
        record.update(rollout_stats)
        record.update({k: v for k, v in update_stats.items() if k != "per_agent"})
        record["per_agent"] = update_stats.get("per_agent", [])

        ev = record.get("explained_variance", float("nan"))
        if np.isfinite(ev):
            last_ev = ev
            max_ev = max(max_ev, ev)
        best_return = max(best_return, record.get("episode_return_mean", -float("inf")))

        do_eval = (
            cfg.train.eval_interval > 0
            and (epoch + 1) % cfg.train.eval_interval == 0
        ) or epoch == cfg.train.total_epochs - 1
        if do_eval:
            eval_res = evaluate_policy(
                algo,
                cfg.env,
                n_episodes=cfg.train.eval_episodes,
                seed=derive_seed(seed, "eval", epoch),
                deterministic=cfg.train.eval_deterministic,
            )
            record.update(eval_res.to_dict(prefix="eval/"))
            final_eval = eval_res

        writer.write(record)
        history.append(record)

        if progress and (epoch % max(1, cfg.train.log_interval) == 0 or do_eval):
            logger.info(
                "[seed %d] epoch %3d | train R %8.2f | EV %6.3f | KL %.4f/%.3f | "
                "inner %.1f | %s",
                seed,
                epoch,
                record.get("episode_return_mean", float("nan")),
                record.get("explained_variance", float("nan")),
                record.get("update_kl", float("nan")),
                cfg.ppo.kl_early_stop,
                record.get("inner_epochs_used", float("nan")),
                (
                    f"eval R {record['eval/return_mean']:8.2f}"
                    if "eval/return_mean" in record
                    else f"{t2 - t0:.2f}s/epoch"
                ),
            )

        if cfg.train.checkpoint_interval > 0 and (epoch + 1) % cfg.train.checkpoint_interval == 0:
            torch.save(algo.state_dict(), run_dir / f"checkpoint_{epoch + 1}.pt")

    torch.save(algo.state_dict(), run_dir / "checkpoint_final.pt")
    wall = time.time() - start

    if final_eval is None:  # pragma: no cover - only if total_epochs == 0
        final_eval = evaluate_policy(algo, cfg.env, n_episodes=cfg.train.eval_episodes, seed=seed)

    result = TrainResult(
        run_dir=run_dir,
        seed=seed,
        algo=cfg.train.algo,
        epochs=cfg.train.total_epochs,
        final_eval=final_eval.to_dict(),
        random_eval=random_eval.to_dict(),
        best_return=best_return,
        final_explained_variance=last_ev,
        max_explained_variance=max_ev,
        wall_clock_seconds=wall,
        history=history,
    )
    (run_dir / "summary.json").write_text(json.dumps(result.summary(), indent=2))
    writer.close()
    vec_env.close()
    logger.info("[seed %d] done in %.1fs -> %s", seed, wall, run_dir)
    return result


def train_many_seeds(
    cfg: Config,
    seeds: list[int],
    *,
    out_dir: str | Path | None = None,
    progress: bool = True,
) -> list[TrainResult]:
    """Sequentially train one job per seed (SPEC Sec. 7: never a single seed)."""
    results = []
    for seed in seeds:
        run_dir = None
        if out_dir is not None:
            run_dir = Path(out_dir) / f"{cfg.train.algo}_seed{seed}"
        results.append(train(cfg, seed=seed, out_dir=run_dir, progress=progress))
    return results


def aggregate_results(results: list[TrainResult]) -> dict[str, Any]:
    """Mean / std / min over seeds for the numbers acceptance tests care about."""
    finals = np.array([r.final_eval["return_mean"] for r in results], dtype=np.float64)
    randoms = np.array([r.random_eval["return_mean"] for r in results], dtype=np.float64)
    evs = np.array([r.max_explained_variance for r in results], dtype=np.float64)
    return {
        "seeds": [r.seed for r in results],
        "final_return_mean": float(finals.mean()),
        "final_return_std": float(finals.std()),
        "final_returns": finals.tolist(),
        "random_return_mean": float(randoms.mean()),
        "random_returns": randoms.tolist(),
        "improvement_mean": float((finals - randoms).mean()),
        "improvement_min": float((finals - randoms).min()),
        "max_explained_variance_mean": float(evs.mean()),
        "max_explained_variance_min": float(evs.min()),
        "wall_clock_seconds": [r.wall_clock_seconds for r in results],
    }
