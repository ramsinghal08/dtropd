"""Cross-machine comparison of a training curve against a stored golden reference.

This is the single definition of "does IPPO still behave as it did at Phase 2".
It lives in the package rather than in ``tests/`` because two callers need it --
``tests/test_baselines.py`` and ``experiments/phase4_acceptance.py`` -- and when
the rule was written out twice, one copy was fixed and the other was not.

Why a tolerance at all
----------------------

The reference was generated on a different machine. Floating-point addition is
not associative, so a different BLAS build reorders reductions and changes the
last few bits of every dot product. Training is chaotic, so those last bits
amplify: on CI the observed disagreement was ~1e-8 of a curve's dynamic range at
epoch 0, growing by roughly two orders of magnitude over twelve epochs. Nothing
about the *behaviour* differs. Asserting exact equality was a bug -- it asserted
a property achievable only on the machine that produced the reference.

Why the noise floor is *absolute*
---------------------------------

Two repairs failed before this one, each because it assumed the round-off scales
with something it does not.

*Relative error*, ``|a-e|/|e|``, reddened CI on ``explained_variance``. An
untrained critic explains nothing, so that curve *starts at* -0.0099 and rises to
0.58: it passes through zero by construction. Dividing a 6.2e-9 round-off by a
value that is 1.7% of the curve's own range manufactured a 6.3e-7 "relative
error" out of the same round-off that read as 1.7e-9 on the return curve.

*Fraction of dynamic range* then reddened CI on ``update_kl``, whose absolute
deviation was 8.9e-9 -- the same round-off again -- but whose range is only
0.0062, so the normalisation inflated it to 1.4e-6 of range.

The three deviations CI actually produced say what the right scale is::

    episode_return_mean   4.9e-8 absolute   (magnitude 28)
    explained_variance    6.2e-9 absolute   (magnitude 0.0099 at epoch 0)
    update_kl             8.9e-9 absolute   (magnitude 0.0091)

Three metrics whose magnitudes span three orders of magnitude, and whose
round-off is the same ~1e-8 in absolute terms. That is what should be expected:
all of them are built from O(1) intermediates -- log-probs, values, normalised
advantages -- so BLAS reduction order perturbs them at a fixed absolute scale
regardless of how large the reported statistic happens to be. ``update_kl``
looks noisy only because it is an O(0.01) *difference* of O(1) quantities, and
the cancellation amplifies its relative error by ~100x without changing its
absolute error at all.

So the tolerance is ``atol + rtol * range``: an absolute floor that covers the
common noise scale, plus a proportional term giving large-magnitude curves
headroom in case their round-off does scale with magnitude on some other build.

What each tier detects
----------------------

* **Epoch 0**, to ``FIRST_EPOCH_ATOL + FIRST_EPOCH_RANGE_TOL * range``
  (1e-6 + 1e-5 x range). This is the sharp assertion and the one doing the real
  work: a single update from bit-identical initial weights, before chaos has
  anything to amplify. It clears the worst observed round-off by ~20x and sits
  100x or more below a 1%-of-magnitude change on *every* metric in the
  reference -- which ``test_first_epoch_tolerance_still_detects_a_real_change``
  checks mechanically for each key rather than asserting in prose here.
* **The whole curve**, to ``CURVE_ATOL + CURVE_RANGE_TOL * range``
  (1e-4 + 1e-2 x range). Deliberately loose: chaotic amplification is
  exponential, so no tight bound on later epochs is defensible, and pretending
  otherwise just buys another red build. What this tier catches is gross
  divergence -- IPPO plateauing somewhere else, or a bug in a code path that only
  runs after the first epoch -- which is O(1) of range, a hundredfold over the
  threshold.

Step-valued metrics
-------------------

Some of these metrics are not continuous at all. ``inner_epochs_used`` is a
*count* -- how many inner epochs ran before the KL early-stop tripped, averaged
over agents -- and ``clip_fraction`` is the fraction of samples the PPO ratio
clipped. Their response to round-off is a step function, not a smooth drift: a
borderline comparison decides the other way and the value jumps by one whole
step. For ``inner_epochs_used`` that step is ``1/6`` on the reference config,
which is 100% of that curve's range; for ``clip_fraction`` it is ``1/4800``,
which is 0.6% of its range and still thousands of times the epoch-0 tolerance.
No tolerance on a continuous scale can both accept a single flipped decision and
mean anything.

Such curves are therefore compared in their own units: at most
:data:`MAX_FLIPPED_DECISIONS` flipped decision per epoch. That is a weaker
assertion than for the smooth curves and it is the strongest true one available
-- a real change to the update math flips these decisions in bulk, not one at a
time.

Which curves are step-valued is **inferred**, by :func:`infer_step`, rather than
listed. Twice now this comparison has reddened CI because a metric was assumed
smooth and was not, so the classification should not depend on remembering to
add a name to a list. A count-fraction takes values ``k/N`` for integer ``k``, so
the smallest ``N`` that renders every reference value integral recovers the
granularity directly from the stored curve, with no config plumbing and nothing
to keep in sync. :data:`STEP_VALUED_KEYS` records the keys already known to be
discrete, and a test asserts the inference still finds them.

The genuinely bitwise guarantees live elsewhere and are untouched by any of this:
``test_beta_zero_reproduces_ippo_bitwise`` and
``test_algorithms_are_deterministic_under_a_seed`` compare runs in the *same*
process, where exactness is both achievable and meaningful. This module is for
the cross-machine, cross-time comparison, which is the only place a tolerance
belongs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "FIRST_EPOCH_ATOL",
    "FIRST_EPOCH_RANGE_TOL",
    "CURVE_ATOL",
    "CURVE_RANGE_TOL",
    "STEP_VALUED_KEYS",
    "MAX_FLIPPED_DECISIONS",
    "MAX_DENOMINATOR",
    "CurveVerdict",
    "infer_step",
    "compare_curve",
]

#: Epoch-0 tolerance: absolute floor, ~20x the worst round-off CI has produced.
FIRST_EPOCH_ATOL = 1e-6
#: Epoch-0 tolerance: proportional term, in case round-off scales with magnitude.
FIRST_EPOCH_RANGE_TOL = 1e-5
#: Whole-curve tolerance: absolute floor, for curves with a very small range.
CURVE_ATOL = 1e-4
#: Whole-curve tolerance: proportional term bounding chaotic amplification.
CURVE_RANGE_TOL = 1e-2
#: Keys known to be counts rather than continuous quantities. Not the mechanism
#: -- :func:`infer_step` is -- but a test asserts the inference still finds these.
STEP_VALUED_KEYS = ("inner_epochs_used", "clip_fraction")
#: How many flipped threshold decisions per epoch are attributable to round-off.
MAX_FLIPPED_DECISIONS = 1
#: Largest denominator :func:`infer_step` will consider. Above this, the sample
#: count is fine enough that a single flip is lost in the smooth tolerance anyway.
MAX_DENOMINATOR = 100_000


@dataclass
class CurveVerdict:
    """The outcome of comparing one metric's curve against the reference."""

    key: str
    kind: str  # "smooth" | "step"
    ok: bool
    reason: str = ""
    scale: float = float("nan")
    first_epoch_deviation: float = float("nan")
    max_deviation: float = float("nan")
    max_deviation_epoch: int = -1
    tolerances: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "ok": self.ok,
            "reason": self.reason,
            "scale": self.scale,
            "first_epoch_deviation": self.first_epoch_deviation,
            "max_deviation": self.max_deviation,
            "max_deviation_epoch": self.max_deviation_epoch,
            "tolerances": self.tolerances,
        }


#: Relative error the stored metrics themselves carry, from float32 accumulation
#: inside the training loop. ``clip_fraction`` is exactly ``223/4800`` but is
#: stored as 0.046458332954595484, which is ``222.999998/4800``.
ACCUMULATION_EPS = 1e-7
#: An inferred granularity is only believed while the tolerance it needs stays
#: this tight; past that, a count is indistinguishable from a continuous value
#: and claiming otherwise would match noise.
MAX_DISCRIMINATING_TOL = 1e-2


def infer_step(
    expected: Sequence[float],
    max_denominator: int = MAX_DENOMINATOR,
) -> float | None:
    """Recover a count-fraction's granularity from the reference curve itself.

    A metric averaged over a fixed number ``N`` of discrete decisions takes only
    the values ``k/N``. Returns ``1/N`` for the smallest such ``N``, or ``None``
    if the curve is continuous.

    The comparison cannot use a fixed absolute tolerance on ``v*N``: the stored
    values carry :data:`ACCUMULATION_EPS` of relative error from float32
    accumulation, which ``v*N`` multiplies up. The tolerance therefore grows with
    ``N`` -- and the scan stops once it exceeds
    :data:`MAX_DISCRIMINATING_TOL`, because a tolerance that loose accepts
    anything and would start reporting continuous curves as counts.

    False positives are negligible while the scan runs: a continuous value lands
    within ``tol`` of a multiple of ``1/N`` with probability ``~2*tol <= 0.02``,
    and all of the curve's points must do so at once.
    """
    e = np.asarray(list(expected), dtype=np.float64)
    if e.size == 0 or not np.all(np.isfinite(e)):
        return None
    magnitude = float(np.max(np.abs(e)))
    for n in range(1, max_denominator + 1):
        tol = max(1e-9, n * magnitude * ACCUMULATION_EPS)
        if tol > MAX_DISCRIMINATING_TOL:
            return None
        scaled = e * n
        if np.all(np.abs(scaled - np.round(scaled)) <= tol):
            return 1.0 / n
    return None


def _dynamic_range(e: np.ndarray) -> float:
    """The curve's own scale. Falls back sensibly for a constant curve."""
    span = float(np.max(e) - np.min(e))
    if span > 0.0:
        return span
    magnitude = float(np.max(np.abs(e)))
    return magnitude if magnitude > 0.0 else 1.0


def compare_curve(
    key: str,
    actual: Sequence[float],
    expected: Sequence[float],
    *,
    granularity: float | None = None,
    first_epoch_atol: float = FIRST_EPOCH_ATOL,
    first_epoch_range_tol: float = FIRST_EPOCH_RANGE_TOL,
    curve_atol: float = CURVE_ATOL,
    curve_range_tol: float = CURVE_RANGE_TOL,
    max_flipped: int = MAX_FLIPPED_DECISIONS,
) -> CurveVerdict:
    """Compare one metric's curve against the stored reference.

    Whether the curve is step-valued is inferred from the reference (see
    :func:`infer_step`). Pass ``granularity`` to override that inference; passing
    it for a curve that is continuous has no effect.
    """
    a = np.asarray(list(actual), dtype=np.float64)
    e = np.asarray(list(expected), dtype=np.float64)
    if a.shape != e.shape:
        return CurveVerdict(
            key=key,
            kind="smooth",
            ok=False,
            reason=f"{key}: got {a.size} epochs, reference has {e.size} epochs",
        )

    step = granularity if granularity and granularity > 0.0 else infer_step(e)
    if step is not None:
        # Compared in units of one flipped threshold decision, not on a
        # continuous scale -- see the module docstring.
        steps = np.abs(a - e) / step
        worst = float(steps.max())
        verdict = CurveVerdict(
            key=key,
            kind="step",
            ok=worst <= max_flipped + 1e-6,
            scale=float(step),
            first_epoch_deviation=float(steps[0]),
            max_deviation=worst,
            max_deviation_epoch=int(steps.argmax()),
            tolerances={"max_flipped_decisions": float(max_flipped)},
        )
        if not verdict.ok:
            verdict.reason = (
                f"IPPO's {key} diverged from the Phase 2 reference: {worst:.3g} "
                f"flipped decisions (steps of {step:.6g}) at epoch "
                f"{verdict.max_deviation_epoch}, more than the {max_flipped} "
                f"attributable to round-off tipping a single borderline "
                f"comparison.\n  reference: {list(e)}\n  current:   {list(a)}"
            )
        return verdict

    # Absolute deviations, against an absolute floor plus a proportional term.
    # The floor is what matters: the round-off is ~1e-8 whatever the metric's
    # magnitude, because all of these are built from O(1) intermediates.
    scale = _dynamic_range(e)
    dev = np.abs(a - e)
    first_tol = first_epoch_atol + first_epoch_range_tol * scale
    curve_tol = curve_atol + curve_range_tol * scale
    verdict = CurveVerdict(
        key=key,
        kind="smooth",
        ok=True,
        scale=scale,
        first_epoch_deviation=float(dev[0]),
        max_deviation=float(dev.max()),
        max_deviation_epoch=int(dev.argmax()),
        tolerances={"first_epoch": first_tol, "curve": curve_tol},
    )
    if verdict.first_epoch_deviation > first_tol:
        verdict.ok = False
        verdict.reason = (
            f"IPPO's {key} changed since Phase 2 at the FIRST epoch: absolute "
            f"deviation {verdict.first_epoch_deviation:.3g} exceeds {first_tol:.3g} "
            f"({first_epoch_atol:g} + {first_epoch_range_tol:g} x range {scale:.4g}). "
            f"One update from identical initial weights should not move at this "
            f"magnitude unless the update math changed.\n"
            f"  reference[0]: {e[0]!r}\n  current[0]:   {a[0]!r}"
        )
    elif verdict.max_deviation > curve_tol:
        verdict.ok = False
        verdict.reason = (
            f"IPPO's {key} curve diverged from the Phase 2 reference: max absolute "
            f"deviation {verdict.max_deviation:.3g} at epoch "
            f"{verdict.max_deviation_epoch} exceeds {curve_tol:.3g} "
            f"({curve_atol:g} + {curve_range_tol:g} x range {scale:.4g}).\n"
            f"  reference: {list(e)}\n  current:   {list(a)}"
        )
    return verdict
