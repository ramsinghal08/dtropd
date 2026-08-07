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

## 5. Two threat models, two defences: the trust gate and robust aggregation
   protect against different adversaries

**This is the most transferable finding here, and it is a taxonomy rather than a
result.** It began as an apparent null in H6 and turned out to be a category
error in how the experiment was scoped.

### 5a. The distinction

| | **gradient-level** adversary | **policy-level** adversary |
|---|---|---|
| what it corrupts | the per-edge gradient, at the aggregation site | the broadcast policy snapshot |
| what it broadcasts | an honest-looking snapshot | a divergent snapshot |
| effect on `D_ij` | none — behaviour is unchanged | elevated |
| effect on `w_ij` | none — **the gate is blind by construction** | should collapse |
| what defends | robust aggregation (Prop. 5) | the trust gate |

`w_ij` keys on `D_ij`, the *behavioural* divergence of the neighbour's broadcast
policy. An adversary that never changes its behaviour is invisible to it. This
is not a weakness in the gate; it is the gate's domain of applicability, and the
same reasoning says a robust aggregator cannot help against a peer whose
*policy* has genuinely diverged — trimming operates on gradients that are, in
that case, faithful reports of a bad teacher.

### 5b. Consequences for the paper

* **Prop. 5 must be scoped explicitly to the gradient-level model.** It is a
  statement about robust aggregation, and that is the model robust aggregation
  addresses. Stating it without the scope invites the reading that D-TrOPD's
  gate contributes to Byzantine tolerance, which the measurement below rules
  out.
* **H6 as built is correctly scoped, not inconclusive.** It uses a
  gradient-level adversary, so it tests the *aggregators* — exactly Prop. 5's
  subject. Its earlier "inconclusive" label was wrong about what it measured.
* **A gate-based robustness claim needs a policy-level adversary**, which is a
  different experiment. It had never been run; see 5d.

### 5c. The measurement

H6's manipulation check, on existing logs (adaptive attack, trimmed mean,
f = 0.1, 10 seeds x 30 instrumented epochs, 1936 Byzantine edges vs 21296
honest):

| | Byzantine edges | honest edges |
|---|---|---|
| mean `w_ij` | 0.87074 | 0.87193 |
| mean `D_ij` | 0.02803 | 0.02775 |

Paired per-seed difference **-0.0029, 95% CI [-0.0071, +0.0008]** — spans zero.
The gate does not attenuate Byzantine neighbours at all.

The gate does not attenuate Byzantine neighbours at all — as 5a says it cannot.

### 5d. The policy-level arm, added afterwards

A policy-level adversary is implemented in `dtropd/byzantine.py`
(`corrupt_snapshot`): it offsets the *broadcast* snapshot's pre-squash mean while
leaving the agent's own actor untouched, so the adversary keeps acting and
learning normally and only its report is a lie. The arm
(`experiments/configs/h6_policy_level.yaml`) runs at `f ∈ {0, 0.2, 0.5}`, 10
paired seeds, D-TrOPD vs NaiveDistill, with **plain-sum aggregation** so that the
gate is the only defence in play.

It was **added 2026-08-06, after the distinction above was found and before any
policy-level run existed**; the prediction is written into the config header,
pre-outcome. One caveat is recorded there and is worth repeating: with a large
enough offset, "`D_ij` rises and `w_ij` collapses" follows from the attenuation
formula and is close to arithmetic. What the arm establishes is that the wiring
is real and that nothing short-circuits it — and, on the open question, whether
the attenuation translates into protected *return*. The genuinely informative
follow-up is the **stealth threshold**: how close to `delta_i` an adversary can
sit and still be admitted, the policy-level analogue of what `adaptive` does to
trimming.

### 5e. An adaptive radius can be widened by the thing it polices

Found while building 5d, and it generalises beyond this algorithm.

`delta_base` is calibrated as a **quantile of the observed `D_ij`
distribution**. A policy-level adversary's edges sit far out in that
distribution, so an adversary present during calibration **widens the very
radius meant to exclude it**. Measured on a smoke config containing only
`f = 0.5` arms: `delta_base` `0.1 → 1.2545`, `gamma` `1 → 0.472` — the gate
largely calibrated itself away before training began.

Any scheme that fits a threshold to the distribution it is meant to police has
this exposure. The mitigation used here is to measure the calibration on an
unattacked arm, which is only available because the sweep contains one.

---

## 6. Proposition 4's bound holds with about two orders of magnitude to spare

`beta / beta_max` measured across training on the calibrated heterogeneous cell:
**median 0.0063, worst case 0.0106**. The operating point sits roughly 100×
inside the bound.

The margin matters more than the compliance. `beta_max` depends on `F_max`,
which is estimated rather than known, and an estimate of a maximum over an
unbounded state space is the kind of quantity that can be wrong by a large
factor. A **100× error in `F_max` would still leave the run inside the bound**,
so the guarantee does not rest on the estimate being tight.

This is the third confirmed mechanism result, alongside Prop. 3 and the
invariance half of Remark 2.

---

## 7. The trust gate works against the adversary it was designed for — and the
   protection has a compounding cost

Two results from the policy-level arm (60 cells, 10 paired seeds, plain-sum
aggregation so the gate is the only defence in play).

### 7a. Complete exclusion, and a 300× contrast between threat models

| threat model | `w_ij` byz | `w_ij` honest | honest weight removed |
|---|---|---|---|
| **policy**-level, `f = 0.5` | **0.00000** | 0.7077 | **100%** |
| **policy**-level, `f = 0.2` | **0.00000** | 0.8337 | **100%** |
| **gradient**-level, `f = 0.1` | 0.8628 | 0.8652 | 0.3% |

The same gate, the same measurement, two adversaries: total exclusion against
one, nothing against the other. This is the sharpest form of the §5 taxonomy —
the two threat models are not points on a severity axis, and a single "Byzantine
tolerance" number that pools them would be meaningless.

Both halves are near-arithmetic given the divergences involved (`D_ij` 3.83 vs
0.035 on lying edges, `w = exp(-γ[softplus(D-δ) - softplus(-δ)])` with
`γ = 10.67`). Their value is not that the numbers are surprising; it is that the
mechanism is demonstrated end-to-end and the scoping claim is now measured
rather than argued.

### 7b. Excluding neighbours costs consensus among the ones that remain

Not predicted. Honest-edge statistics, same runs:

| | `f = 0` | `f = 0.2` | `f = 0.5` |
|---|---|---|---|
| honest `D_ij` | ~0.0074 | 0.0349 | 0.0665 |
| honest `w_ij` | ~0.957 | 0.834 | 0.708 |

**Honest agents diverge ~9× more at `f = 0.5` than unattacked**, and the gate
attenuates them too. The loop is direct: excluding a fraction `f` of every
neighbourhood removes that much consensus pull → the surviving honest agents
drift apart → their `D_ij` rises → their `w_ij` falls → less pull still.

**The cost of the defence is not the excluded edges; it is the compounding loss
of consensus among the agents that remain.** Any attenuation-based robustness
scheme on a communication graph has this exposure, and it is invisible if only
the excluded edges are reported. Worth a paragraph.

---

## 8. Implementation note: append-mode metrics silently merged two runs

A third entry for §2's collection, and the costliest, because it corrupted
already-committed results.

The metrics writer opened its JSONL in **append** mode. A cell killed mid-write
left a partial line; the resume path correctly re-ran the cell, and the re-run
appended to the dead attempt. The result was one file holding two runs.

The malformed line where they meet is self-announcing. **The duplicate epochs on
either side are not** — the file parses, and every summary over it averages
across two runs. Three cells of the completed H6 sweep were affected (59, 56 and
7 duplicated epochs) and had been analysed without anyone noticing.

**The rule:** a writer for a re-runnable unit of work must truncate, and a reader
must treat duplicate keys as corruption rather than data. "Append-only" is the
right shape for a log and the wrong shape for a result. The self-announcing
failure is a courtesy; design for the silent one.
