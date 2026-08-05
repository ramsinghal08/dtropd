"""Calibration artifacts (SPEC Sec. 11 / D6).

The attenuation constants are *measured*, and the measurement is a first-class
artifact rather than two numbers pasted into a YAML file. An artifact records
everything needed to audit it -- the window, the epochs pooled, the edge count,
the full quantile table, the reward spec it was measured under, and the commit
-- and every arm in a comparison must load the same one.

Why this exists: ``delta_base`` measured separately per arm is a researcher
degree of freedom that no amount of care in the analysis can undo. Pinning it to
a shared file makes "all arms used the same calibration" checkable rather than
asserted, and :func:`assert_shared_calibration` refuses to run a sweep that
violates it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "CalibrationRecord",
    "CalibrationError",
    "QUANTILES",
    "assert_shared_calibration",
]

#: Quantiles recorded for every calibration. SPEC Sec. 11: log distributions,
#: not means -- the D1 failure was invisible in the mean.
QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


class CalibrationError(RuntimeError):
    """Raised when a comparison's arms disagree about their calibration."""


@dataclass
class CalibrationRecord:
    """One measurement of the attenuation constants for one environment config."""

    delta_base: float
    gamma_attenuation: float

    # --- the rule that produced it (SPEC Sec. 11 / D6) --------------------- #
    window_epochs: int
    pooled_epochs: list[int]
    pool_size: int
    edge_count: int
    quantiles: dict[str, float]
    quantile_definition: str

    # --- provenance -------------------------------------------------------- #
    reward_spec_version: str
    git_commit: str
    env_fingerprint: str
    created_at: str = ""
    seed: int | None = None
    source_config: str | None = None
    note: str = ""
    calibration_target_w: float = 0.5
    tail_quantile: float = 0.95
    center_quantile: float = 0.5

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    # -- identity ----------------------------------------------------------- #

    @property
    def calibration_id(self) -> str:
        """Stable hash over the measurement, not over the file it lives in.

        Two arms that loaded the same artifact share this id; two arms that
        calibrated separately do not, even if their numbers happen to be close.
        """
        payload = json.dumps(
            {
                "delta_base": self.delta_base,
                "gamma_attenuation": self.gamma_attenuation,
                "window_epochs": self.window_epochs,
                "pooled_epochs": list(self.pooled_epochs),
                "edge_count": self.edge_count,
                "reward_spec_version": self.reward_spec_version,
                "env_fingerprint": self.env_fingerprint,
                "created_at": self.created_at,
            },
            sort_keys=True,
        ).encode()
        return hashlib.blake2b(payload, digest_size=8).hexdigest()

    # -- IO ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["calibration_id"] = self.calibration_id
        return out

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationRecord":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationRecord":
        p = Path(path)
        if not p.exists():
            raise CalibrationError(
                f"calibration artifact {p} not found. Produce one with "
                f"`python run.py calibrate --config <cfg> --out {p}` "
                "(SPEC Sec. 11 / D6)."
            )
        record = cls.from_dict(json.loads(p.read_text()))
        stored = json.loads(p.read_text()).get("calibration_id")
        if stored and stored != record.calibration_id:
            raise CalibrationError(
                f"calibration artifact {p} has been edited: its recorded id "
                f"{stored} does not match the id of its contents "
                f"{record.calibration_id}. Re-measure rather than hand-edit."
            )
        return record

    # -- construction from a sample ------------------------------------------ #

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[float],
        *,
        window_epochs: int,
        pooled_epochs: Sequence[int],
        reward_spec_version: str,
        git_commit: str,
        env_fingerprint: str,
        center_quantile: float = 0.5,
        tail_quantile: float = 0.95,
        target_w: float = 0.5,
        seed: int | None = None,
        source_config: str | None = None,
        note: str = "",
    ) -> "CalibrationRecord":
        from .algorithms import calibrate_attenuation

        arr = np.asarray(list(samples), dtype=np.float64)
        if arr.size == 0:
            raise CalibrationError("no D_ij samples: cannot calibrate")
        result = calibrate_attenuation(
            arr, quantile=center_quantile, tail_quantile=tail_quantile, target_w=target_w
        )
        return cls(
            delta_base=result["delta_base"],
            gamma_attenuation=result["gamma_attenuation"],
            window_epochs=int(window_epochs),
            pooled_epochs=[int(e) for e in pooled_epochs],
            pool_size=len(list(pooled_epochs)),
            edge_count=int(arr.size),
            quantiles={
                f"p{int(round(q * 100)):02d}": float(np.quantile(arr, q))
                for q in QUANTILES
            },
            quantile_definition=(
                f"delta_base = q{center_quantile:g} of D_ij pooled over the final "
                f"{len(list(pooled_epochs))} epoch(s) of a {window_epochs}-epoch "
                f"beta=0 window; gamma set so a q{tail_quantile:g} edge attenuates "
                f"to w = {target_w:g} (SPEC Sec. 11 / D6)"
            ),
            reward_spec_version=reward_spec_version,
            git_commit=git_commit,
            env_fingerprint=env_fingerprint,
            seed=seed,
            source_config=source_config,
            note=note,
            calibration_target_w=target_w,
            tail_quantile=tail_quantile,
            center_quantile=center_quantile,
        )


def env_fingerprint(env_cfg: Any) -> str:
    """Stable hash of the environment configuration a calibration applies to.

    Calibration is per *environment configuration* (D6). Reusing an artifact
    across a different environment is a silent error, so the fingerprint is
    recorded and checked.
    """
    payload = json.dumps(dataclasses.asdict(env_cfg), sort_keys=True, default=str).encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def assert_shared_calibration(
    records: Iterable[tuple[str, "CalibrationRecord | None"]],
    *,
    require_calibration: bool = True,
) -> str | None:
    """Refuse a sweep whose arms were calibrated separately (SPEC Sec. 11 / D6).

    ``records`` is an iterable of ``(arm_name, record_or_None)``. Every arm must
    reference the *same* artifact -- including arms that do not consume
    ``delta_base`` (NaiveDistill has no radius per D4, but still declares which
    calibration its comparison used, so the sweep is auditable).

    Returns the shared ``calibration_id``, or ``None`` if the sweep legitimately
    uses no calibration at all.
    """
    rows = list(records)
    if not rows:
        raise CalibrationError("no arms supplied to assert_shared_calibration")

    missing = [name for name, rec in rows if rec is None]
    present = [(name, rec) for name, rec in rows if rec is not None]

    if not present:
        if require_calibration:
            raise CalibrationError(
                "no arm in this sweep declares a calibration artifact. SPEC "
                "Sec. 11 / D6 requires the attenuation to be measured once per "
                "environment configuration and pinned for every arm."
            )
        return None

    if missing:
        raise CalibrationError(
            f"arms {sorted(missing)} declare no calibration artifact while "
            f"{sorted(n for n, _ in present)} do. Every arm in a comparison must "
            "load the same artifact -- including arms with no radius of their own, "
            "which declare it for auditability (SPEC Sec. 11 / D6)."
        )

    ids = {rec.calibration_id for _, rec in present}
    if len(ids) > 1:
        detail = ", ".join(
            f"{name}={rec.calibration_id} (delta_base={rec.delta_base:.5g})"
            for name, rec in present
        )
        raise CalibrationError(
            "arms in this sweep were calibrated separately, which makes the "
            f"comparison uninterpretable: {detail}. SPEC Sec. 11 / D6 requires a "
            "single artifact shared by every arm."
        )

    fingerprints = {rec.env_fingerprint for _, rec in present}
    if len(fingerprints) > 1:  # pragma: no cover - implied by the id check
        raise CalibrationError("arms disagree about the environment fingerprint")

    return present[0][1].calibration_id
