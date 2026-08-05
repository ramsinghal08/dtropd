#!/usr/bin/env python3
"""Generate the four H7 sensitivity-sweep configs (SPEC Sec. 16 / D17).

D7-AMENDED says the grid is a procedure and the sweep is run once per resulting
`delta_base`. This script materialises that: one sweep config per grid point,
each pinning the calibration artifact for its point so D6's "one artifact per
environment configuration, shared by every arm" still holds.

The four points
---------------

Three are **measured**, from `run.py calibration-grid --windows 6 8 10` on the
actual H7 treatment environment, and are signed off as pre-registered:

    0.051703   (window 6)
    0.079151   (window 8 -- the calibrated value)
    0.092786   (window 10)

The fourth is **derived by scaling**, at `2 x` the calibrated value:

    0.158302   (2 x window 8)

It is derived rather than measured because no calibration window produces it --
that is the point. The measured grid spans 0.65x-1.17x of calibrated, i.e.
almost entirely *tighter* gates, so the loose-gate direction was untested. Its
artifact is written with `gamma` carried over from window 8 unchanged and a note
recording exactly how it was produced, so it stays auditable and keeps its
content hash; it is never presented as a measurement.

Usage::

    python experiments/make_h7_grid_configs.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtropd.calibration import CalibrationRecord  # noqa: E402

CAL = ROOT / "experiments/calibrations"
CONFIGS = ROOT / "experiments/configs"

#: H7 spans TWO environments -- the heterogeneous treatment and the homogeneous
#: control -- and D6 gives each its own artifact. Pinning one environment's
#: artifact to both arms is refused by the D6 guard, correctly, so the grid is
#: built per environment and the arms are pinned by which environment they run.
ENVIRONMENTS = {"het": "h7_env_het", "hom": "h7_env_hom"}
WINDOWS = (6, 8, 10)
#: SPEC Sec. 16 / D17: the added loose-gate point, as a multiple of calibrated.
ADDED_MULTIPLE = 2.0
CALIBRATED_WINDOW = 8
#: Arms whose environment is the homogeneous control (they override
#: env.heterogeneous to false in h7.yaml).
HOM_SUFFIX = "_hom"


def make_scaled_artifact(env_stem: str) -> tuple[Path, float]:
    """Write the 2x artifact for one environment, labelled as derived."""
    base = CalibrationRecord.load(CAL / f"{env_stem}_w{CALIBRATED_WINDOW}.json")
    scaled = dataclasses.replace(
        base,
        delta_base=base.delta_base * ADDED_MULTIPLE,
        created_at="",  # re-stamped by __post_init__
        note=(
            f"DERIVED, NOT MEASURED. delta_base = {ADDED_MULTIPLE}x the calibrated "
            f"value from window {CALIBRATED_WINDOW} "
            f"({base.delta_base:.6f} -> {base.delta_base * ADDED_MULTIPLE:.6f}); "
            "gamma carried over unchanged. Added per SPEC Sec. 16 / D17 to test "
            "the LOOSE-gate direction, which the measured grid does not reach. "
            "Pre-outcome and strictly harder for the hypothesis: a looser gate "
            "is closer to NaiveDistill, so a benefit that survives here is not "
            "attributable to gating."
        ),
    )
    path = CAL / f"{env_stem}_x2.json"
    scaled.save(path)
    return path, scaled.delta_base


def grid_points(env_stem: str) -> dict[str, tuple[Path, float]]:
    out: dict[str, tuple[Path, float]] = {}
    for window in WINDOWS:
        path = CAL / f"{env_stem}_w{window}.json"
        out[f"w{window}"] = (path, CalibrationRecord.load(path).delta_base)
    out["x2"] = make_scaled_artifact(env_stem)
    return out


def main() -> int:
    CAL.mkdir(parents=True, exist_ok=True)
    per_env = {tag: grid_points(stem) for tag, stem in ENVIRONMENTS.items()}

    base_cfg = yaml.safe_load((CONFIGS / "h7.yaml").read_text())
    written = []
    for point in ("w6", "w8", "w10", "x2"):
        het_path, het_delta = per_env["het"][point]
        hom_path, hom_delta = per_env["hom"][point]

        cfg = json.loads(json.dumps(base_cfg))  # deep copy through plain types
        cfg["name"] = f"h7_{point}"
        cfg["out_dir"] = f"runs/h7_{point}"
        cfg["auto_calibrate"] = False  # artifacts are pinned, not derived
        cfg["notes"] = (
            f"H7 headline sweep at D7 grid point {point} "
            f"(delta_base: het={het_delta:.6f}, hom={hom_delta:.6f}). One of four; "
            "see SPEC Sec. 16 / D17. reward_spec_version v1 throughout "
            "(D2/D3/D13)."
        )
        # Treatment environment on the base; the control arms override it, the
        # same way they already override env.heterogeneous.
        d = cfg["base"].setdefault("distill", {})
        d["calibration"] = "file"
        d["calibration_file"] = str(het_path.relative_to(ROOT))
        d["delta_base"] = float(het_delta)
        for arm in cfg["arms"]:
            if arm["name"].endswith(HOM_SUFFIX):
                ov = arm.setdefault("overrides", {})
                ov["distill.calibration_file"] = str(hom_path.relative_to(ROOT))
                ov["distill.delta_base"] = float(hom_delta)

        path = CONFIGS / f"h7_{point}.yaml"
        path.write_text(
            "# GENERATED by experiments/make_h7_grid_configs.py -- do not hand-edit.\n"
            f"# D7 grid point {point}: delta_base het={het_delta:.6f} "
            f"hom={hom_delta:.6f}\n"
            + yaml.safe_dump(cfg, sort_keys=False, width=100)
        )
        written.append((point, het_delta, hom_delta, path))

    print("D7 sensitivity grid -- four sweep configs written:")
    print(f"{'point':6s} {'delta_base(het)':>16s} {'delta_base(hom)':>16s}  kind")
    for point, het, hom, path in written:
        kind = "measured" if point != "x2" else "DERIVED (2x calibrated)"
        print(f"{point:6s} {het:16.6f} {hom:16.6f}  {kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
