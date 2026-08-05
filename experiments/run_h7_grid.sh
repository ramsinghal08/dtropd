#!/usr/bin/env bash
# H7 -- the headline interaction sweep, at all FOUR D7 grid points.
# SPEC Sec. 12 / D7-AMENDED and Sec. 16 / D17.
#
# Four sweeps of 80 cells (8 arms x 10 seeds) = 320 cells total.
#
#   point   delta_base (het)   delta_base (hom)   provenance
#   w6      0.051703           0.047977           measured, window 6
#   w8      0.079151           0.076916           measured, window 8 = CALIBRATED
#   w10     0.092786           0.096086           measured, window 10
#   x2      0.158302           0.153832           DERIVED, 2x calibrated (D17)
#
# H7 spans two environments -- heterogeneous treatment and homogeneous control
# -- so D6 gives each its own artifact and each grid point pins both. The
# control arms carry their own delta_base via per-arm overrides.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   ./experiments/run_h7_grid.sh              # 8 workers (default)
#   WORKERS=16 ./experiments/run_h7_grid.sh
#   OUT_ROOT=/data/h7 ./experiments/run_h7_grid.sh
#   POINTS="w8" ./experiments/run_h7_grid.sh  # a single grid point
#
# RESUMABLE, per point and across points. Re-run the same command: finished
# cells are skipped by config fingerprint and finished points are skipped whole.
#
# Cost: ~5 min/cell on one core => ~27 CPU-hours for all four points.
#   8 workers ~= 3.5 h   16 workers ~= 1.7 h
#
# The four configs are GENERATED. Regenerate after any calibration change:
#   python experiments/make_h7_grid_configs.py

set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS="${WORKERS:-8}"
OUT_ROOT="${OUT_ROOT:-runs}"
POINTS="${POINTS:-w6 w8 w10 x2}"

if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python
fi

echo "H7 headline sweep -- D7 sensitivity grid"
echo "  points : $POINTS"
echo "  workers: $WORKERS"
echo "  out    : $OUT_ROOT/h7_<point>"
echo "  cells  : 80 per point; resumable, safe to re-run"
echo

for point in $POINTS; do
  cfg="experiments/configs/h7_${point}.yaml"
  out="${OUT_ROOT}/h7_${point}"
  [ -f "$cfg" ] || { echo "missing $cfg -- run make_h7_grid_configs.py first"; exit 2; }

  # Skip a point that already finished, so re-running the script is cheap.
  if [ -f "$out/manifest.json" ] && $PY -c "
import json,sys
m=json.load(open('$out/manifest.json'))
sys.exit(0 if m.get('in_progress') is False else 1)" 2>/dev/null; then
    echo "== $point already complete, skipping =="
    continue
  fi

  echo "== grid point $point =="
  $PY run.py sweep --config "$cfg" --out "$out" --workers "$WORKERS"
  echo
done

echo "All requested H7 grid points complete."
echo "Per-point reports: $OUT_ROOT/h7_<point>/report.json"
echo "Figures:           $OUT_ROOT/h7_<point>/figures/comparison.png"
echo
echo "The D7 verdict is whether H7's INTERACTION (the predicted sign flip)"
echo "holds across all four points -- not whether any single point is positive."
