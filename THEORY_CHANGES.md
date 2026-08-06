# Theory changes and paper-relevant empirical findings

Findings from the implementation and the Phase 7 experiments that bear on the
paper's claims. Each entry states what was measured, on what data, and what
follows for the text. Dated 2026-08-06 unless noted.

Nothing here is a result of a hypothesis test unless it says so. Several entries
record hypotheses that were **refuted**, including three of mine.

---

## 1. The "social term is immaterial" hypothesis — TESTED AND REFUTED

I proposed, three times and with increasing confidence, that H6's flat curve,
H1's null and a likely H7 null had one common cause: that `beta_max = 0.05`
makes the distillation term a rounding error next to the policy gradient. It
does not. Measured on the calibrated D-TrOPD heterogeneous H6 cell
(`adaptive_trim_f00_seed0`, 12 agents, complete graph, 150 epochs, 29
instrumented epochs):

| quantity | median while β > 0 | range |
|---|---|---|
| loss ratio `\|β·L_soc\| / \|L_CLIP\|` | **2.36** | 1.07 – 2.96 |
| gradient ratio `‖β·∇L_soc‖ / ‖∇L_CLIP‖` | **0.307** | 0.167 – 0.428 |
| gradient share `soc / (soc + clip)` | **23.5%** | — |

**The social term contributes ~24% of the parameter update, rising to ~34% late
in training.** The common-cause hypothesis is dead, and the three quiet results
need three separate explanations. This is worth stating in the paper because the
obvious reading of a small `β` is exactly the wrong one.

*Method note:* the gradient norms are logged directly and measured on the same
minibatch as the policy gradient. The loss values are **not** logged; `|β·L_soc|`
is reconstructed as `n_edges_per_agent × mean(w_ij) × mean(D_ij) × β`, i.e.
`L_soc` at inner-loop start where `KL_ij = D_ij`. That is an upper bound, since
distillation reduces `KL_ij` within the inner loop.

---

## 2. Implementation notes: the obvious diagnostic lies

Two independent cases where a standard, reasonable-looking instrument gives a
confidently wrong answer. Both cost real time here, and neither is obvious.

### 2a. The loss ratio misleads by ~8x

`|β·L_soc| / |L_CLIP| = 2.36` says distillation **dominates** training. The
gradient ratio says it contributes a **minority** 23.5%. Both are computed from
the same run at the same epochs.

The cause: `L_CLIP` is a clipped-ratio surrogate against **normalised**
advantages, so its *value* sits near zero by construction — normalised
advantages average to zero, and the clip is near 1 — while its *gradient* is
large. `L_soc` is a sum of positive KL divergences, so its value is
unambiguously positive. Comparing the values compares two quantities that are
not on the same footing.

**Anyone instrumenting a composite RL objective is exposed to this.** The rule:
to ask what a loss term contributes to learning, compare **gradient norms on the
same minibatch**, never the loss values.

### 2b. Explained variance does not detect a frozen critic

Invariant I4 asks for "critic loss present and optimized; explained variance
logged and rising". EV is the natural check. It does not work.

With the critic's parameters **frozen at initialisation** — verified directly,
parameter drift exactly `0.000000e+00`, zero value parameters requiring grad —
EV was essentially unchanged against a healthy critic (0.1176 vs 0.1379, and
tracking it epoch by epoch: 0.0913/0.184/0.4277 vs 0.0913/0.1859/0.4279).

The cause: the **value normaliser keeps adapting** even when the critic cannot
learn. `RunningMeanStd` tracks the return distribution, so EV partly measures
the normaliser rather than the critic, and a critic that has learned nothing can
post a healthy-looking EV.

**Consequence for I4:** parameter drift is the authoritative degeneracy check;
EV is secondary and can be actively misleading. Recorded as an amendment in
SPEC. EV is a common critic-health metric, so this failure mode is worth a line
in the paper's implementation notes.

---

## 3. The social term's influence GROWS over training

The gradient ratio rises from **0.25 to 0.34** over 150 epochs (loss ratio 1.95
→ 2.96). Distillation becomes *more* influential as training proceeds, not less.

This is not obvious a priori — one might expect neighbours to converge and the
social term to fade. It interacts with the fixed-`delta_base` question in D5: a
radius calibrated early is applied to a regime where the term it gates is
strengthening. Worth a sentence, and it is an argument for revisiting whether
`delta_base` should track the `D_ij` distribution rather than being pinned once.

---

## 4. Remark 2 is stronger than stated: the value-variance radius is unusable
   across scales without re-tuning

The paper says the value-variance radius has the wrong sign and is
reward-scale dependent. The empirically demonstrated claim is stronger.

`delta = delta_base · exp(-kappa · signal)`. With `kappa = 100` — chosen for
**nonstationarity**, a KL in nats of order `1e-5` — applied to a
**value-variance** signal of order `3.25`, the radius is
`exp(-325) ≈ 1e-141`: numerically zero. Under a 10x reward rescale the signal
becomes `~324`, giving `exp(-32440)`, a hard zero.

So the radius does not merely have the wrong sign; **it saturates to zero and
stops being a radius at all**, and the `kappa` that works at 1x is off by ~100x
at 10x because `Var(V)` scales with reward². Every value-variance arm was pinned
at `delta = 0` regardless of its critic, which is why the degenerate-critic
pathology could not appear even though the critic was verifiably frozen.

**The nonstationarity radius needs no re-tuning under reward rescaling, because
KL is dimensionless.** That is the sharp form of the claim, and it is an
argument from units rather than from a benchmark — which makes it stronger.

---

## 5. The trust gate is blind to a gradient-level adversary, by construction

H6's manipulation check, on existing logs (adaptive attack, trimmed mean,
f = 0.1, 10 seeds x 30 instrumented epochs, 1936 Byzantine edges vs 21296
honest):

| | Byzantine edges | honest edges |
|---|---|---|
| mean `w_ij` | 0.87074 | 0.87193 |
| mean `D_ij` | 0.02803 | 0.02775 |

Paired per-seed difference **-0.0029, 95% CI [-0.0071, +0.0008]** — spans zero.
The gate does not attenuate Byzantine neighbours at all.

**This is not a failure of the gate; it is a category distinction the paper
should make explicitly.** `w_ij` keys on `D_ij`, the *behavioural* divergence of
the neighbour's broadcast policy. A gradient-level adversary corrupts the
per-edge gradient at the aggregation site and never touches its policy snapshot,
so its behaviour is indistinguishable and the gate has nothing to key on.

The two defences protect against different things:

* the **trust gate** against neighbours whose *policies* have diverged;
* **robust aggregation** against neighbours whose *gradients* are corrupted.

H6 as constructed therefore tests the aggregators, and can neither support nor
undermine any gate-based robustness claim. A gate-based claim needs a
**policy-level** adversary — one that broadcasts a corrupted snapshot — which is
a different experiment.
