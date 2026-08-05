#!/usr/bin/env bash
# H6 -- Byzantine tolerance, at FULL budget (SPEC Sec. 15 / D16).
#
# 16 arms x 10 seeds = 160 cells, 150 epochs, 40 envs, 12 agents.
# No epoch trim, no env trim, no seed cut. If a shorter budget ever becomes
# necessary, D16 requires experiments/h6_budget_check.py to pass FIRST.
#
# Why this runs outside the agent container: that environment executes only
# while a session is actively issuing calls, so a long sweep makes no progress
# between turns. This script has no such constraint -- run it anywhere with
# CPU and time.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   ./experiments/run_h6.sh                # auto-detect workers
#   WORKERS=8 ./experiments/run_h6.sh      # pin worker count
#   OUT=/data/h6 ./experiments/run_h6.sh   # pin output directory
#
# RESUMABLE. Kill it and re-run the same command: completed cells are skipped
# by config fingerprint, and only the remainder runs. This has been tested --
# it is how H6 recovered from two container reaps during development.
#
# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
# Measured at 12 agents / 40 envs / complete graph on a 4-core CPU box:
#   ~2.2 s/epoch with an active adversary  ->  ~5.6 min per 150-epoch cell
#   160 cells ~= 15 CPU-hours
# Divide by worker count for wall-clock: 8 workers ~= 2 h, 4 workers ~= 4 h.
# A GPU helps less than you would expect here; the networks are 64x64 and the
# bottleneck is per-edge autograd, so cores matter more than device.
#
# ---------------------------------------------------------------------------
# Colab
# ---------------------------------------------------------------------------
#   !git clone https://github.com/ramsinghal08/dtropd.git && cd dtropd
#   !pip install -q torch --index-url https://download.pytorch.org/whl/cpu
#   !pip install -q -r requirements.txt
#   !WORKERS=2 OUT=/content/drive/MyDrive/h6 bash experiments/run_h6.sh
# Point OUT at Drive so a disconnect does not lose finished cells -- Colab
# recycles VMs, and the resume path is only useful if the runs survive.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=experiments/configs/h6_byzantine.yaml
OUT="${OUT:-runs/h6_byzantine}"
if [ -n "${WORKERS:-}" ]; then
  W="$WORKERS"
else
  CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
  W=$(( CORES > 2 ? CORES - 1 : 1 ))   # leave one core for the OS
fi
# Prefer an in-repo venv when one exists, else whatever `python` resolves to.
# Override with PYTHON=... for conda, pyenv, or a Colab interpreter.
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python
fi

echo "H6 Byzantine tolerance -- full budget"
echo "  config : $CONFIG"
echo "  out    : $OUT"
echo "  workers: $W"
echo "  cells  : 160 (16 arms x 10 seeds); resumable, safe to re-run"
echo

# Refuse to start on a config that has been trimmed without the D16 check.
$PY - "$CONFIG" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
train = cfg["base"]["train"]
seeds = cfg["seeds"]
problems = []
if train["total_epochs"] < 150: problems.append(f"total_epochs={train['total_epochs']} < 150")
if train["num_envs"] < 40:      problems.append(f"num_envs={train['num_envs']} < 40")
if len(seeds) < 10:             problems.append(f"{len(seeds)} seeds < 10 (SPEC Sec. 7)")
if problems:
    print("REFUSING: this config has been trimmed below the D16 full budget:")
    for p in problems: print("  -", p)
    print()
    print("D16 requires experiments/h6_budget_check.py to pass on the shorter")
    print("budget FIRST -- f=0 and f=0.5 must separate with non-overlapping CIs.")
    print("Otherwise a flat H6 curve cannot be told apart from a working defence.")
    sys.exit(2)
print("budget check: full budget confirmed (150 epochs, 40 envs, 10 seeds)")
PYEOF

echo
exec $PY run.py sweep --config "$CONFIG" --out "$OUT" --workers "$W"
