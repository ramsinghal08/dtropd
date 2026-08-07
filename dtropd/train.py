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
from .instrumentation import quantiles as _quantiles
from .probes import decay_probe
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
            # Equal to final_return when nothing is Byzantine, so a sweep can
            # read one key along the whole f axis without comparing an
            # honest-only number against an all-agent one.
            "final_return_honest": self.final_eval.get(
                "return_mean_honest", self.final_eval.get("return_mean")
            ),
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
        # SPEC Sec. 11: the metrics schema is versioned too. A schema change
        # after H7 starts invalidates stored curves, same as a reward bump.
        metrics_schema_version=cfg.instrument.metrics_schema_version,
        calibration_id=(
            algo.calibration_record.calibration_id if algo.calibration_record else None
        ),
        calibration=(
            algo.calibration_record.to_dict() if algo.calibration_record else None
        ),
        obs_dim=cfg.env.obs_dim,
        act_dim=cfg.env.act_dim,
        batch_size_per_agent=cfg.batch_size,
        neighbors=algo.neighbors,
        actor_snapshot_bytes=algo.snapshot_bytes,
        # Who was actually Byzantine, recorded rather than left to be recomputed.
        # Any analysis that splits edges by attacker has to know the set exactly;
        # re-deriving it downstream from (seed, fraction) works only for as long
        # as nobody changes how it is drawn, and a mislabelled split fails
        # silently in the direction of "no effect found".
        byzantine=list(getattr(algo, "byzantine", [])),
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
        record.update(algo.compute_accounting())

        # D9: the consensus measure, computed identically for every algorithm so
        # the distillation arms have an uncontaminated IPPO control.
        if cfg.instrument.consensus_pull:
            record.update(
                _quantiles(
                    algo.policy_dispersion(),
                    cfg.instrument.quantiles,
                    "consensus_pull/policy_dispersion_",
                )
            )

        # H5's second measurement: the Assumption-2 decay probe. Interval-gated
        # because it needs its own paired rollouts, but always run on the final
        # epoch so every run's metrics file carries at least one.
        inst = cfg.instrument
        if inst.decay_probe and (
            (epoch + 1) % inst.decay_probe_interval == 0
            or epoch == cfg.train.total_epochs - 1
        ):
            record["decay_probe"] = decay_probe(
                algo,
                cfg.env,
                n_perturbed=inst.decay_probe_agents,
                episodes=inst.decay_probe_episodes,
                scale=inst.decay_probe_scale,
                seed=derive_seed(seed, "probe", epoch),
                quantile_list=inst.quantiles,
            )

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
    # SPEC Sec. 11 / D6: an in-run calibration is measured *during* training, so
    # it cannot be in the metadata written at start-up. Persist the artifact so
    # every other arm of this comparison can pin itself to it.
    if getattr(algo, "calibration_record", None) is not None:
        algo.calibration_record.save(run_dir / "calibration.json")
        metadata["calibration_id"] = algo.calibration_record.calibration_id
        metadata["calibration"] = algo.calibration_record.to_dict()
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    wall = time.time() - start

    if final_eval is None:  # pragma: no cover - only if total_epochs == 0
        final_eval = evaluate_policy(algo, cfg.env, n_episodes=cfg.train.eval_episodes, seed=seed)

    # H6's headline is the HONEST agents' return. The environment has always
    # exposed per-agent returns for this, and the rollout metrics have always
    # split them -- but the EVALUATION path dropped them, so `final_return`, the
    # number that reaches every manifest, report and curve, was the all-agent
    # mean. Recorded alongside rather than replacing it: `final_return` keeps its
    # meaning, and anything H6 must not fold saboteurs into reads the honest key.
    final_eval_dict = final_eval.to_dict()
    byzantine = sorted(getattr(algo, "byzantine", []) or [])
    honest = [i for i in range(cfg.env.n_agents) if i not in set(byzantine)]
    final_eval_dict["return_mean_honest"] = (
        final_eval.subset_return(honest) if byzantine else final_eval.return_mean
    )
    if byzantine:
        final_eval_dict["return_mean_byzantine"] = final_eval.subset_return(byzantine)

    result = TrainResult(
        run_dir=run_dir,
        seed=seed,
        algo=cfg.train.algo,
        epochs=cfg.train.total_epochs,
        final_eval=final_eval_dict,
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
