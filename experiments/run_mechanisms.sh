#!/usr/bin/env bash
# H1, H2, H3 -- the cheap, decisive mechanism experiments, queued in sequence.
#
# They also DIAGNOSE H6. H6 came out flat, which is consistent with two opposite
# readings; H2 discriminates them:
#
#   * If H2 shows a large, clear effect, the social term IS material, and H6's
#     flatness means the ADVERSARY is inert.
#   * If H2 is weak, the social term is immaterial overall -- which explains H6
#     and predicts a null H7, since D-TrOPD differs from IPPO only through it.
#
#   ./experiments/run_mechanisms.sh              # 8 workers (default)
#   WORKERS=16 ./experiments/run_mechanisms.sh
#   ONLY="h2" ./experiments/run_mechanisms.sh    # a single experiment
#
# RESUMABLE per experiment and across them: completed cells are skipped by
# config fingerprint, completed experiments are skipped whole.
#
# Cost: 55 cells total (H1 15, H2 10, H3 30) at ~60 epochs / 20 envs.
# Roughly 1/12th of H6 -- these are the cheap ones.
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED DECISION RULES  (SPEC Sec. 6; full text in each config)
# ---------------------------------------------------------------------------
# H1  no chattering under continuous attenuation
#     DECIDES ON: sign-flip rate in the nearest-boundary bucket
#                 (|D_ij - delta_i| <= 0.005), WITHIN each arm, versus that
#                 arm's own far-from-boundary buckets.
#     CONFIRMED IF: hard_switch / hard_cutoff spike near the boundary while
#                 continuous stays flat across buckets.
#
# H2  forward KL inflates student entropy
#     DECIDES ON: mean_sigma (and per-edge sigma_i on top-quartile-D_ij edges),
#                 forward_kl_distill vs dtropd, paired by seed.
#     CONFIRMED IF: forward drives sigma_i^2 -> sigma_j^2 + (mu_i - mu_j)^2
#                 while reverse holds near sigma_j^2.
#
# H3  trust-radius pathologies
#     DECIDES ON: (1) delta_mean ratio under 10x reward vs 1x -- nonstationarity
#                 should be ~1.0 (scale-invariant), value_variance should not be;
#                 (2) delta_mean with a degenerate critic vs healthy --
#                 nonstationarity should CONTRACT, value_variance should not.
#     CONFIRMED IF: both hold. Failing (2) is the I1 sign pathology.
# ---------------------------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS="${WORKERS:-8}"
ONLY="${ONLY:-h1 h2 h3}"
OUT_ROOT="${OUT_ROOT:-runs}"

if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python
fi

declare -a NAMES=(h1:h1_chattering h2:h2_kl_direction h3:h3_radius)

echo "Mechanism experiments -- H1, H2, H3"
echo "  running: $ONLY     workers: $WORKERS"
echo

for entry in "${NAMES[@]}"; do
  tag="${entry%%:*}"; cfgname="${entry##*:}"
  case " $ONLY " in *" $tag "*) ;; *) continue ;; esac

  cfg="experiments/configs/${cfgname}.yaml"
  out="${OUT_ROOT}/${cfgname}"
  if [ -f "$out/manifest.json" ] && $PY -c "
import json,sys
sys.exit(0 if json.load(open('$out/manifest.json')).get('in_progress') is False else 1)" 2>/dev/null; then
    echo "== $tag ($cfgname) already complete, skipping =="
    continue
  fi

  echo "== $tag -- $cfgname =="
  $PY run.py sweep --config "$cfg" --out "$out" --workers "$WORKERS"
  echo
done

echo "Done. Next:"
echo "  python experiments/social_share.py ${OUT_ROOT}/h1_chattering/runs/*/"
echo "     -> the social share of the update; the common explanation for H6's"
echo "        flatness, the small consensus-pull cost, and any null in H7."
echo "  python experiments/analyze_hypotheses.py --run ${OUT_ROOT}/h2_kl_direction/runs/<cell>"
