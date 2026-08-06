#!/usr/bin/env bash
# The three queued follow-ups, in the order they were prioritised.
#
#   ./experiments/run_queue.sh manipulation   # 1. H6(b): is the adversary landing?
#   ./experiments/run_queue.sh valvar         # 2. H3's value-variance arms, kappa fixed
#   ./experiments/run_queue.sh policy         # 3. H6(c): the policy-level adversary
#   ./experiments/run_queue.sh all            # all three in order, ~10 h
#
#   WORKERS=8 ./experiments/run_queue.sh manipulation
#   OUT_ROOT=/data/dtropd ./experiments/run_queue.sh policy
#
# ONE STAGE PER INVOCATION IS THE DEFAULT, so results can be read before the
# next stage starts -- but that is a REVIEW cadence, not a compute dependency.
# No stage's outcome changes whether another is worth running: stage 1 tests the
# gradient-level adversary, stage 3 tests a different adversary against a
# different defence (SPEC D19), and stage 2 is unrelated to both. `all` is
# therefore safe, and is the right choice for an unattended overnight run.
#
# `all` DOES NOT STOP ON A FAILED STAGE. A crash three hours in would otherwise
# waste the rest of the night; each stage is independent, so the remaining ones
# still run and a summary at the end names anything that failed. Every stage is
# resumable, so a failed one picks up where it stopped on the next invocation.
#
# Run these OUTSIDE the agent container (SPEC D16): that environment only
# executes while a session is issuing calls, so a multi-hour sweep makes no
# progress between turns.
#
# RESUMABLE. Re-running the same command skips completed cells by config
# fingerprint. This is load-bearing for stage 2 -- see below.
#
# ---------------------------------------------------------------------------
# 1. manipulation -- h6_manipulation.yaml, 40 cells, ~3.7 h at 4 workers
# ---------------------------------------------------------------------------
# NaiveDistill (w = 1, ungated) vs D-TrOPD at f = 0.5 under the SAME adaptive
# adversary, unchanged. The discriminator for H6's flat curve:
#   * NaiveDistill collapses, D-TrOPD does not -> something in D-TrOPD confers
#     the robustness. Not the gate (it is blind to this threat model, D19), so
#     the aggregator or the KL early-stop absorbing the corrupted step.
#   * Neither moves -> the adversary is not landing and H6 is uninterpretable
#     as run.
# Highest-value item in the queue: it is the only one that can retire H6's open
# question.
#
# ---------------------------------------------------------------------------
# 2. valvar -- h3_radius.yaml, 3 arms x 10 seeds re-run, ~0.6 h at 4 workers
# ---------------------------------------------------------------------------
# The kappa fix (SPEC D18). kappa = 100 was derived for a KL-valued signal and
# pinned every value-variance arm at delta ~ 1e-141, so those arms could not
# express their prediction at all. kappa_valvar = 0.636, derived ONCE on the 1x
# healthy arm and pinned across all three.
#
# THE WHOLE CONFIG IS RE-RUN, and only the three valvar arms actually execute:
# their fingerprint changed with kappa, the three nonstationarity arms' did not.
# That is the resume path doing the selection, which is safer than hand-picking
# arms -- if the nonstat arms HAD changed, silently skipping them would be wrong.
# Requires the existing runs/h3_radius directory; without it, all 60 cells run.
#
# ---------------------------------------------------------------------------
# 3. policy -- h6_policy_level.yaml, 60 cells, ~5.6 h at 4 workers
# ---------------------------------------------------------------------------
# The policy-level adversary (SPEC D19): broadcasts a divergent snapshot while
# acting normally. The threat the trust gate was designed for, and it has never
# been run. Plain-sum aggregation so the gate is the only defence in play.
# Pre-registered prediction is in the config header, dated, written before the
# first run.
#
# Outranks stage 2 if compute is tight.
# ---------------------------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-}"
OUT_ROOT="${OUT_ROOT:-runs}"
# Default to cores-1 rather than a fixed 4: an overnight run should use the
# machine. One core is left for the OS so the box stays responsive.
if [ -z "${WORKERS:-}" ]; then
  CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)"
  WORKERS=$(( CORES > 2 ? CORES - 1 : 1 ))
fi

if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python
fi

# --- `all`: every stage in order, in ONE process ------------------------------
# Sequentially, never concurrently. Three sweeps running at once each want
# $WORKERS processes, so on an 8-core laptop that is ~24 workers thrashing 8
# cores -- same total work, worse wall-clock, and three times the peak memory.
# One at a time with every core is strictly faster.
if [ "$STAGE" = "all" ]; then
  failed=()
  for s in manipulation valvar policy; do
    echo
    echo "############################################################"
    echo "# queue: $s   ($(date))"
    echo "############################################################"
    if "$0" "$s"; then :; else
      echo "!! stage $s exited $? -- continuing; it is resumable" >&2
      failed+=("$s")
    fi
  done
  echo
  echo "############################################################"
  echo "# queue finished ($(date))"
  if [ ${#failed[@]} -eq 0 ]; then
    echo "# all three stages completed"
  else
    echo "# FAILED: ${failed[*]}   -- re-run those stages; completed cells skip"
  fi
  echo "############################################################"
  [ ${#failed[@]} -eq 0 ]
  exit $?
fi

case "$STAGE" in
  manipulation) NAME=h6_manipulation ; CELLS="40 cells, ~3.7 h at 4 workers"   ;;
  valvar)       NAME=h3_radius       ; CELLS="30 cells of 60; the rest resume-skip" ;;
  policy)       NAME=h6_policy_level ; CELLS="60 cells, ~5.6 h at 4 workers" ;;
  *)
    echo "usage: $0 {manipulation|valvar|policy|all}" >&2
    echo >&2
    echo "One stage at a time lets each be read before the next starts;" >&2
    echo "'all' runs the three in order, ~10 h, for an unattended run." >&2
    exit 2
    ;;
esac

CONFIG="experiments/configs/${NAME}.yaml"
OUT="${OUT_ROOT}/${NAME}"

echo "queue stage: $STAGE"
echo "  config : $CONFIG"
echo "  out    : $OUT"
echo "  workers: $WORKERS"
echo "  size   : $CELLS   (resumable, safe to re-run)"
echo

if [ "$STAGE" = "valvar" ] && [ ! -f "$OUT/manifest.json" ]; then
  echo "NOTE: no existing $OUT -- the three nonstationarity arms have nothing to"
  echo "      resume from and will run too (60 cells rather than 30)."
  echo
fi

$PY run.py sweep --config "$CONFIG" --out "$OUT" --workers "$WORKERS"

echo
echo "Done: $STAGE. Analysis commands for this stage:"
case "$STAGE" in
  manipulation)
    echo "  $PY experiments/h6_curve.py --manifest $OUT/manifest.json \\"
    echo "      --out $OUT/figures/h6b_curve.png --json-out $OUT/h6b_curve.json"
    echo "  -> paired return vs f, per algorithm. The comparison that matters is"
    echo "     each algorithm against ITS OWN f=0, not against the other."
    echo "  $PY experiments/gate_split.py $OUT/runs/*/"
    echo "  -> confirms on THESE runs that the gate is still blind at the"
    echo "     gradient level; NaiveDistill logs no w_ij and is skipped, which is"
    echo "     itself the control's sanity check."
    ;;
  valvar)
    echo "  $PY experiments/analyze_hypotheses.py --run $OUT/runs/<cell> --require h3"
    echo "  -> delta must now be nonzero in the valvar arms. If it is still ~0 the"
    echo "     kappa fix did not take and nothing downstream is interpretable."
    ;;
  policy)
    echo "  $PY experiments/gate_split.py $OUT/runs/*/ --out $OUT/gate_split.json"
    echo "  -> D_ij and w_ij split by Byzantine vs honest edge, paired per seed."
    echo "     Predictions (1) and (2) live THERE, not in final return."
    echo "  $PY experiments/h6_curve.py --manifest $OUT/manifest.json \\"
    echo "      --out $OUT/figures/h6c_curve.png --json-out $OUT/h6c_curve.json"
    echo "  -> prediction (3), the return curve, and the weakest of the three."
    ;;
esac
