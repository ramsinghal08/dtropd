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

## 4. Remark 2, RESTATED: one premise of its argument is refuted by our own
   experiment; the conclusion stands on stronger ground

**The argument as written does not survive; the claim does.** Recorded this way
deliberately — a premise refuted by the project's own measurement is worth more
than a conclusion nobody stress-tested.

### The refuted premise

Remark 2 currently reasons:

> degenerate critic → the agent is unstable → it *should* earn a SMALL radius →
> but value-variance grants a LARGE one → sign pathology.

**The middle step is false.** Measured (§11): a critic frozen at initialisation
made the policy **12% MORE stable** (`Delta_healthy = 4.52e-3` nats vs
`Delta_degenerate = 3.97e-3`). Uninformative advantages give a weaker policy
gradient and therefore *less* movement between epochs. A broken critic is a
source of policy **inertia**, not instability — so "the agent is unstable" cannot
be assumed, and the sign pathology cannot be argued from it.

Same category as H1's "untestable at this horizon": the mechanism did exactly
what its formula specifies, and it was the hypothesis's assumed causal chain that
failed.

### The three grounds that survive

**(b) NON-STATIONARITY WITHIN A RUN — the lead argument.** With
`kappa_valvar = 0.636` derived on an observed `Var(V)` range of `[0.124, 3.270]`,
the late-training radius is `1.53e-7`. Inverting
`delta = delta_base·exp(-kappa·Var(V))` gives late-training **`Var(V) = 20.71`,
6.3× beyond the window the constant was derived on, at UNCHANGED reward scale.**

So a value-variance radius must be re-calibrated **within a single run**. This
leads because it argues from what the quantity *is* — `Var(V)` is a
dimensionful, drifting statistic of a function that is still being learned —
rather than from any benchmark, and it is the version that survives review. A KL
signal has neither problem: dimensionless, and its scale does not drift with the
spread of the value function.

**(a) SCALE NON-INVARIANCE.** `Var(V)` scales with reward², so a `kappa`
calibrated at 1× underflows the radius at 10×: measured `delta = 0.0` exactly,
a hard zero, against `1.53e-7` at 1×. The nonstationarity radius over the same
rescale gives ratio **1.099** — invariant, as an argument from units predicts.

**(c) MIS-RANKING, measured.** The degenerate critic's `Var(V)` is 8.11 against
the healthy critic's 20.71, so **the critic that explains nothing earns a radius
3037× larger**. This stands as direct evidence of the sign pathology and needs no
claim about stability to make its point: whatever a degenerate critic does to the
policy, ranking it as *more* trustworthy than a working one is the defect I1
exists to prevent.

Ground (c) was unmeasurable before D18 — both arms were pinned at exactly zero by
the `kappa` units error — so the pathology could only be asserted, never shown.

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

---

## 9. The trust gate is necessary and sufficient against a policy-level
   adversary — the ungated learner collapses by 7×

Honest-agent final return, 10 paired seeds, plain-sum aggregation so the gate is
the only defence in play, paired against **each algorithm's own `f = 0`**.

| `f` | D-TrOPD (gated) | paired Δ | NaiveDistill (`w=1`) | paired Δ |
|---|---|---|---|---|
| 0.0 | −36.51 | — | −35.80 | — |
| 0.2 | −35.97 | +0.54 [−1.07, +2.08] | **−196.96** | **−161.16 [−261.91, −61.58]** |
| 0.5 | −35.35 | +1.16 [−0.48, +3.11] | **−252.70** | **−216.91 [−301.59, −122.85]** |

The two algorithms are indistinguishable at `f = 0` and separated by an order of
magnitude at `f > 0`. Every gated CI spans zero; no ungated CI comes near it.

Read with §7: the gate removes **100%** of a lying neighbour's weight, and that
translates into **complete** protection of return. It also answers §7b's open
question — the compounding consensus loss among honest agents (their `w_ij`
falls to 0.708 at `f = 0.5`) **does not cost return**. The defence's second-order
cost is real in the consensus metric and invisible in the outcome.

*Scope, stated as it was before the run:* NaiveDistill is the no-defence
ablation, not a Byzantine-robust baseline. This establishes the gate as necessary
and sufficient against a policy-level adversary here — not that D-TrOPD beats a
robust aggregator, which is a different comparison against a different threat
model (§5).

---

## 10. (folded into §4)

The within-run `Var(V)` drift and the 3037× mis-ranking are grounds (b) and (c)
of the restated Remark 2. Kept as a numbered stub rather than renumbering, so
references to §4 and §11 elsewhere stay valid.

---

## 11. A refuted premise: a degenerate critic does not destabilise the policy

H3 predicted the nonstationarity radius CONTRACTS under an intentionally
degenerate critic, reasoning that a destabilised policy has larger `Delta_i`.
Measured ratio **1.057** — the radius slightly *grew*.

Inverting the radii gives the reason: `Delta_healthy = 4.52e-3` nats against
`Delta_degenerate = 3.97e-3`. **The frozen critic made the policy 12% MORE
stable.** A critic frozen at initialisation produces near-uninformative
advantages, so the policy gradient is weaker and the policy moves less between
epochs.

The radius did exactly what `delta = delta_base·exp(-kappa·Delta)` specifies.
What failed is the hypothesis's assumed causal chain from critic degeneracy to
policy instability. Worth a line in the paper because the intuition is natural
and wrong: **a broken critic is not a source of policy instability; it is a
source of policy inertia.**

---

## 12. The consensus cost of the trust gate is REAL in the mechanism and ZERO in
    the outcome

The standing objection to any attenuation gate is that it discards useful peers.
Measured, on the policy-level arm — where the gate removes 100% of every lying
neighbour's weight and `f` of each neighbourhood is excluded:

| | `f = 0` | `f = 0.2` | `f = 0.5` |
|---|---|---|---|
| honest-edge `D_ij` | ~0.0074 | 0.0349 | 0.0665 |
| honest-edge `w_ij` | ~0.957 | 0.834 | **0.708** |
| honest final return | −36.51 | −35.97 | −35.35 |
| paired Δ return vs `f=0` | — | +0.54 [−1.07, +2.08] | +1.16 [−0.48, +3.11] |

**The mechanism pays and the outcome does not.** Excluding `f` of every
neighbourhood removes that much consensus pull; the surviving honest agents drift
~9× further apart, so the gate attenuates *them* too, `w_ij` falling from 0.957 to
0.708 — a **26% loss of honest consensus weight**, compounding, and entirely
invisible in return. Both paired CIs span zero, and the point estimates are
mildly positive.

This is the direct answer to "the gate discards useful peers": at this `beta`,
in this environment, **it does, and it costs nothing.** Stated with its scope —
one environment, one `beta`, one adversary — because the honest version of this
result is not "the cost is zero" but "the cost is invisible in the outcome under
conditions we can name."

**Consequence for D9.** The consensus-pull cost is a first-class measured
quantity, and this is the first case where the measurement and the outcome
disagree. A consensus metric moving by 26% with no detectable effect on return
means **the consensus metric is not a proxy for performance** and must not be
reported as though it were. Both belong in the H7 table, separately.

---

## 13. H7 REFUTED: the mechanisms work and do not pay for themselves

The headline hypothesis. 320 cells, four pre-registered `delta_base` points,
8 arms, 10 paired seeds, all runs verified uncorrupted.

| point | het (D-TrOPD − IPPO) | hom (D-TrOPD − IPPO) | interaction |
|---|---|---|---|
| w6  | −7.780 [−10.264, −5.365] | −6.607 [−9.554, −3.386] | −1.173 [−5.292, +2.860] |
| w8  | −8.708 [−11.429, −6.153] | −8.403 [−11.238, −5.492] | −0.305 [−4.425, +3.872] |
| w10 | −8.497 [−10.920, −6.233] | −7.414 [−10.272, −4.454] | −1.083 [−5.147, +3.000] |
| x2  | −8.541 [−11.089, −5.975] | −9.631 [−12.623, −6.459] | +1.090 [−2.890, +4.706] |

**The predicted sign flip requires `het > 0`. It is ≈ −8, with a CI excluding
zero at every point.** D-TrOPD is not neutral under heterogeneity; it is
significantly worse than IPPO. The interaction is null (|mean| ≤ 1.2 against a
~4 half-width), so distillation's cost is *uniform across environments* and
heterogeneity confers no differential benefit. Stable across the whole grid, so
this is not a calibration artifact.

### The result that matters more than the verdict

**Every mechanism this paper claims was confirmed, and the composite still loses
to IPPO.** The reverse-KL direction separates as predicted (H2); the radius is
scale-invariant where the theory says and pathological where the theory says
(H3, 3 of 4); `beta` sits two orders inside Prop. 4's bound; the trust gate
removes 100% of a policy-level adversary's weight and fully protects return,
where the ungated learner collapses 7×.

The distillation term is not a rounding error either — it is ~24% of the
parameter update (§1), rising to 34%. It is doing a lot of work, correctly, in
the wrong direction for return.

**A method can be right in every component and wrong in aggregate.** That is the
finding, and it is only sayable because the components were measured separately
rather than inferred from the headline. A paper that reported only H7 would have
a null; a paper that reports the mechanisms and then H7 has an explanation.

### The cost is distillation's, and the gate is not part of it

The sweep ran MAPPO and HAPPO too, none of the five arms tuned beyond the shared
base config. Heterogeneous, paired against IPPO:

| arm | vs IPPO |
|---|---|
| **HAPPO** | **+0.148 [−1.847, +2.002]** |
| MAPPO | −3.500 [−5.184, −1.713] |
| NaiveDistill | −8.154 [−10.529, −6.099] |
| D-TrOPD | −8.708 [−11.429, −6.153] |

**HAPPO ties IPPO.** The deficit is therefore not the environment, the horizon
or the untuned shared config — a trust-region baseline reaches IPPO's level on
the same budget. The environment discriminates rather than flattening: MAPPO's
centralised critic is significantly *worse* than independent learning under
heterogeneity.

And the gate contributes nothing. D-TrOPD − NaiveDistill, paired by seed, at all
four grid points in both environments: **all eight CIs span zero, every |mean| ≤
1.6** against deficits of ~8. The entire difference between the two algorithms is
the trust gate, and it is worth nothing in return here. F-1 predicted this from
`w_ij ≈ 0.96` at degree 11; this is the same conclusion read off *return*, with
paired seeds, in H7's own environment.

The refutation is also unanimous rather than marginal: D-TrOPD loses to IPPO on
**79 of 80** paired seed comparisons (10/10 in seven of the eight
environment × grid-point cells).

So the claim is not "D-TrOPD did not beat IPPO". It is: **every mechanism is
confirmed, the trust gate contributes nothing to return, and distillation itself
costs ~8 uniformly across heterogeneity — while a trust-region baseline on the
same budget ties IPPO.**

### Where the cost comes from, and what is not yet ruled out

D9's consensus-pull cost — `R(IPPO) − R(D-TrOPD)` on the homogeneous control —
is +6.6 to +9.6, essentially the whole gap. D9 framed it as the floor a
heterogeneous benefit must clear. **There is no heterogeneous benefit; the floor
is the result.**

One scope statement, no longer a hypothesis: the H7 environment is a complete
graph on 12 agents, F-1 established that **the trust gate is inert at high
degree** (`w ≈ 0.96`), and the D-TrOPD − NaiveDistill table above confirms it in
return terms. So what H7 refutes is **distillation**, with the gate shown to be
inactive rather than assumed to be.

A lower-degree test would be a **new experiment** needing its own pre-registered
rule, and this refutation stands regardless of its outcome. Running environment
variants until one confirms is the same practice D7's calibration rule forbids,
and the rule does not stop at calibration windows.

---

## 14. THE LOCKED RESULT STATEMENT

The paper's lead claim, recorded verbatim and not to be reworded by a later run
(SPEC D21):

> **Every mechanism is confirmed; the trust gate contributes nothing to return;
> and distillation itself costs ~8 uniformly across heterogeneity — while a
> trust-region baseline (HAPPO) on the same budget ties IPPO.**

**(a) HAPPO +0.148 [−1.847, +2.002] vs IPPO.** The environment supports
competitive performance, so the deficit belongs to distillation and not to the
task, the horizon, or the shared untuned hyperparameters. **This control is what
makes the result interpretable at all** — without it, `−8` cannot be told apart
from "the environment is hard for everything", and the paper would have a null
instead of a finding.

**(b) D-TrOPD − NaiveDistill: all 8 contrasts span zero, |mean| ≤ 1.6** against
deficits of ~8. The gate is the entire difference between the two algorithms and
is worth nothing in return, in either environment, at every grid point.

**(c) D-TrOPD loses to IPPO on 79 of 80 paired seed comparisons.** Unanimous,
not marginal.

*Secondary, worth a line:* **MAPPO −3.500 [−5.184, −1.713] vs IPPO** — a
centralised critic is significantly **worse** than independent learning under
heterogeneity.

### The D7 sensitivity verdict

Identical at all four grid points: **conclusion stable across the calibration
grid; the window choice is immaterial.** The widening clause did not fire, and
no grid point was selected for its answer — all four were run, all four are
reported, and the rule was executed in code rather than applied by hand.

Stability here costs the result nothing and buys something: the refutation
cannot be attributed to a calibration choice.

### What the whole programme demonstrates

Nine of the ten recorded results are confirmations, refined claims, or refuted
premises found by our own measurements. The tenth — the headline — is a
refutation. **The mechanisms are right and the composite loses**, and that
sentence is only available because every mechanism was measured separately
instead of being inferred from the headline.

Three of the refuted things were mine, one was a premise of the paper's own
Remark 2, and one was an experiment scoped to the wrong threat model. Each is
recorded with the measurement that killed it. A programme that reported only
what worked would have had less to say and more to defend.
