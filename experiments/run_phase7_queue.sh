#!/usr/bin/env bash
# Queue: H6 at full budget, then H7 at all four D7 grid points.
#
# One command. H7 starts automatically when H6 finishes, and ONLY if H6
# actually finished -- a failed or interrupted H6 stops the queue rather than
# letting H7 start on a half-tested harness.
#
#   ./experiments/run_phase7_queue.sh              # 8 workers (default)
#   WORKERS=16 ./experiments/run_phase7_queue.sh
#   nohup ./experiments/run_phase7_queue.sh > phase7.log 2>&1 &   # detached
#
# BOTH STAGES ARE RESUMABLE. If this dies at any point -- machine sleeps, Colab
# recycles, you Ctrl-C it -- re-run the identical command. Completed cells are
# skipped by config fingerprint and completed stages are skipped whole, so the
# queue picks up where it stopped rather than starting over.
#
# Total cost: H6 ~15 CPU-hours + H7 ~27 CPU-hours = ~42 CPU-hours.
#   8 workers  ~= 5.5 h
#   16 workers ~= 2.7 h
#
# Progress is visible without stopping anything:
#   watch -n60 'ls runs/h6_byzantine/runs | wc -l; ls -d runs/h7_* 2>/dev/null'

set -euo pipefail
cd "$(dirname "$0")/.."

export WORKERS="${WORKERS:-8}"

echo "=============================================================="
echo " queue -- H6 (full budget), then H7 (four grid points)"
echo " workers: $WORKERS      started: $(date)"
echo "=============================================================="
echo

echo "### STAGE 1/2 -- H6 Byzantine tolerance, full budget ###"
echo
if ! ./experiments/run_h6.sh; then
  echo
  echo "H6 did not complete. STOPPING -- H7 is not started."
  echo "Re-run this same command to resume H6 from where it stopped."
  exit 1
fi

echo
echo "### STAGE 2/2 -- H7 headline, D7 sensitivity grid (4 points) ###"
echo
./experiments/run_h7_grid.sh

echo
echo "=============================================================="
echo " queue complete: $(date)"
echo "=============================================================="
echo
echo "Next:"
echo "  python experiments/phase7_acceptance.py      # should now pass F4"
echo "  python experiments/analyze_hypotheses.py     # H1-H7 verdicts"
echo
echo "H7's verdict is the INTERACTION -- the predicted sign flip between the"
echo "heterogeneous and homogeneous environments -- and whether it survives all"
echo "four delta_base grid points. A single positive point is not the result."
