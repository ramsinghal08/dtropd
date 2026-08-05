

## 1. Mission

Implement **D-TrOPD** (Decentralized Trust-Region Policy Distillation) for networked multi-agent RL, plus baselines and a pre-registered experiment suite.

Each agent runs PPO locally and additionally distills from neighbors, but *gates* how much it imports from each neighbor by a trust weight based on behavioral divergence. There is no central critic and no sequential update order.

**A previous prototype of this project failed** because it implemented a superseded design. Section 3 lists the exact corrections. Treat those as hard invariants: if any code path violates one, that is a bug regardless of whether results look good.

**Target hardware:** development on a laptop GPU (GTX 1650, 4 GB VRAM); full runs on Google Colab T4 (15 GB). Everything must run CPU-only for smoke tests and scale to a single GPU for real runs. No multi-GPU, no cluster assumptions.

---

## 2. Algorithm specification (authoritative — implement exactly this)

Agents `i ∈ V = {1..N}` on an undirected graph `G`, neighborhood `N_i`. Each agent has a tanh-squashed diagonal-Gaussian policy `π_i(a|o; θ_i)` and its own local critic `V_i`.

### 2.1 Per-epoch loop (all agents in parallel, identically)

1. Broadcast a **frozen, detached** copy of `θ_i^(t)` to neighbors; freeze all received snapshots for the whole epoch.
2. Roll out `π_i^(t)` locally. Collect `(o, a, r)`, compute GAE advantages `Â_i` and returns `Ĝ_i`.
3. Compute own nonstationarity and set the trust radius (Sec. 2.2).
4. For each neighbor `j ∈ N_i`: compute social divergence `D_ij` and trust weight `w_ij` (Sec. 2.3).
5. Inner optimization loop with KL early stopping (Sec. 2.4).

### 2.2 Trust radius — from **own policy nonstationarity**

```
Δ_i^(t) = E_{s ~ own rollout states} [ KL( π_i^(t)(·|s) || π_i^(t-1)(·|s) ) ]
δ_i^(t) = δ_base * exp( -κ * Δ_i^(t) )
```

- `Δ` is the *previous epoch's* update divergence, measured in nats on the agent's **own** rollout states.
- Requires keeping the previous epoch's policy snapshot per agent.
- At `t = 0`, set `Δ = 0` so `δ = δ_base`.
- Choose `κ` such that `κ * η ∈ [1, 3]` (η is the KL early-stop cap) so the radius has real dynamic range. Assert this at config validation time and warn otherwise.

### 2.3 Social divergence and continuous attenuation

```
D_ij  = E_{s ~ π_i's own rollout states} [ KL( π_i(·|s) || π_j^(t)(·|s) ) ]     # reverse KL, student first
w_ij  = exp( -γ * [ softplus(D_ij - δ_i) - softplus(-δ_i) ] )
L_soc = Σ_{j ∈ N_i} w_ij * E_{s ~ own states} [ KL( π_θi(·|s) || π_j^(t)(·|s) ) ]
```

- **Always reverse KL.** There is no forward-KL branch and no direction switch anywhere in the codebase.
- The `- softplus(-δ_i)` term normalizes so that `w_ij = 1` exactly at `D_ij = 0`. Do not omit it.
- `w_ij` is computed from the detached `D_ij` at step 4 and held **constant** through the inner loop (it is a gate, not a differentiable path). The KL inside `L_soc` **is** differentiable w.r.t. `θ_i`.
- Teacher parameters are always detached; gradients must never flow to `θ_j`.
- **Neighbor selection:** default is all neighbors (`N_k(i) = N_i`) — the weights do the filtering. A hard top-k is available behind a config flag but is off by default because it reintroduces a discontinuity at the selection boundary.

### 2.4 Objective and both trust regions

```
L_i(θ_i) = L_CLIP_i(θ_i) - β(t) * L_soc_i(θ_i) - c_V * L_V_i(θ_i)
```

- `L_CLIP` is standard PPO clipping on the agent's **own** rollout, with GAE.
- `L_V` is the critic MSE against returns. **It must be present and optimized** — both the advantages and the radius signal depend on a trained critic. Log critic explained variance to prove it is learning.
- **β warmup:** `β(t)` ramps linearly from 0 to `β_max` over `warmup_epochs` (default 10% of training). Without this, agents anchor to a random consensus early in training and never recover.
- **KL early stopping:** halt the inner loop when `E_s[ KL( π_i^(t)(·|s) || π_θi(·|s) ) ] ≥ η_i`. This is what enforces the *policy-improvement* trust region.
- Two distinct trust regions, never conflated:
  - *improvement* region (radius `η`) = how far **I** move → enforced by clip + KL early stop.
  - *distillation* region (radius `δ`) = how much I import from **peer j** → enforced by `w_ij`.

### 2.5 Gaussian KL in closed form

Policies are tanh-squashed diagonal Gaussians. KL is invariant under the shared tanh diffeomorphism, so compute KL analytically on the **pre-squash latent** `(μ, log σ)`. Never estimate it by Monte Carlo sampling.

Per dimension: `KL(N(μ₁,σ₁) || N(μ₂,σ₂)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²) / (2σ₂²) - 0.5`, summed over action dims.

### 2.6 Byzantine-robust aggregation (optional path, default off)

When enabled, aggregate per-neighbor distillation gradients `g_ij = ∇_θi KL(π_θi || π_j)` with a robust operator instead of a weighted sum. Implement: coordinate-wise trimmed mean, geometric median, and Krum. Selectable by config.

---

## 3. Non-negotiable invariants (the prototype violated all of these)

Encode each as a runtime assertion or an automated test. Failing any one invalidates the run.

| # | Invariant | Why |
|---|-----------|-----|
| I1 | Trust radius comes from **policy nonstationarity**, never from value-function variance | Value-variance has a sign pathology: a degenerate critic would earn maximal trust |
| I2 | Exactly one KL direction (reverse) everywhere; no branch switch | Forward-KL fallback inflates student entropy in proportion to the mismatch that triggered it |
| I3 | `D_ij` and `L_soc` evaluated on the agent's **own** rollout states, never a pooled global batch | Pooled batches break the improvement bound and open a Byzantine spoofing surface |
| I4 | Critic loss present and optimized; explained variance logged and rising | Advantages *and* the radius signal both depend on it |
| I5 | Both trust regions enforced (clip + KL early stop **and** `w_ij`) | Bounding social divergence alone guarantees nothing about improvement |
| I6 | Teacher snapshots frozen and detached for the whole epoch | Otherwise every agent chases a moving target |
| I7 | `w_ij = 1` at `D_ij = 0` (normalized attenuation) | An identical neighbor must be fully trusted |
| I8 | β warmup schedule active | Prevents anchoring to random consensus early in training |
| I9 | Policy-gradient calls use a `policy_parameters()` helper that **excludes the value head** | `torch.autograd.grad` over all params errors or silently corrupts gradients when the value head is untouched by the policy loss |

---

## 4. Repository structure

```
dtropd/
  config.py        # dataclass configs, validation (incl. κ·η ∈ [1,3] check), YAML load
  env.py           # PettingZoo MPE wrapper + heterogeneity zones + stressors
  networks.py      # tanh-Gaussian actor, critic, policy_parameters() helper
  buffer.py        # rollout storage, GAE
  kl.py            # closed-form Gaussian KL utilities (single source of truth)
  algorithms.py    # DTrOPD, IPPO, MAPPO, HAPPO, NaiveDistill — shared base class
  robust.py        # trimmed mean, geometric median, Krum
  train.py         # training loop, logging, checkpointing, seeding
  evaluate.py      # eval rollouts, metrics
  plot.py          # rliable-based figures (IQM, bootstrap CIs)
  run.py           # CLI entrypoint
tests/
  test_kl.py           # analytic KL vs high-sample MC agreement
  test_invariants.py   # I1–I9 as executable checks
  test_attenuation.py  # w(0)=1, monotone decreasing, C¹ (finite-difference derivative continuity)
  test_smoke.py        # 2-agent 200-step CPU run completes
experiments/
  configs/*.yaml   # one per hypothesis H1–H7 + ablations
notebooks/
  D_TrOPD.ipynb    # self-contained Colab notebook (pip installs, clones, runs, plots)
README.md
```

**Conventions:** Python 3.10+, PyTorch, type hints throughout, dataclass configs (no dicts of magic strings), every random source seeded from one master seed, all runs write a JSON metadata header (config + git commit + seed) next to their metrics.

---

## 5. Build order — phased, each with an acceptance test

**Do not proceed past a failing acceptance test.** Report results and stop.

### Phase 0 — Skeleton
Repo structure, configs, seeding, logging, CI-runnable pytest.
**Accept:** `pytest` runs green on an empty-but-valid test suite; `python run.py --help` works.

### Phase 1 — Env + networks + KL
MPE cooperative navigation wrapper (12 agents, 3 targets, capture requires ≥3 agents co-located), heterogeneity zones (high-friction: 50% velocity; masked-observation: sensor range 1.5→0.5), tanh-Gaussian actor, critic, `kl.py`.
**Accept:** `test_kl.py` — analytic Gaussian KL matches a 1e6-sample Monte Carlo estimate within 1%. Env passes a random-policy rollout with correct shapes and reward accounting.

### Phase 2 — IPPO
Single-agent PPO logic applied independently per agent. GAE, clipping, KL early stopping, critic.
**Accept:** IPPO learns on the *homogeneous* env — mean return over 3 seeds beats a random policy by a clear margin, and critic explained variance rises above 0.3.

### Phase 3 — D-TrOPD core
Radius, attenuation, social loss, frozen snapshots, β warmup, both trust regions.
**Accept (critical):** with `β = 0`, D-TrOPD reproduces IPPO **bitwise-identically** under the same seed. If curves differ at all, the social path is leaking. Also: `test_invariants.py` passes I1–I9; `test_attenuation.py` confirms `w(0)=1` and derivative continuity at `D = δ`.

### Phase 4 — Baselines
MAPPO (centralized critic over concatenated observations), HAPPO (sequential updates, random order each epoch), NaiveDistill (D-TrOPD with `w_ij ≡ 1` — must visibly collapse under heterogeneity).
**Accept:** all four baselines train end-to-end on the homogeneous env with shared hyperparameters and identical seeds.

### Phase 5 — Instrumentation
Log per epoch, per agent: update KL vs `η`; `δ_i`; every `D_ij` and `w_ij`; critic EV; entropy; per-edge social gradient norms; per-zone returns; snapshot bytes exchanged.
**Accept:** a single training run produces a metrics file from which every H1–H7 measurement in Sec. 6 can be computed without re-running.

### Phase 6 — Experiment harness
Config-driven multi-seed launcher, resumable checkpoints, `rliable` plotting (IQM + 95% bootstrap CIs), paired-seed comparison mode.
**Accept:** `python run.py --config experiments/configs/h7.yaml --seeds 10` produces the main comparison figure end-to-end.

### Phase 7 — Robustness + scale
Byzantine agents (including an **adaptive** adversary that mimics honest gradients from inside the honest range), robust aggregators, communication dropout, motor failures, scale sweep `N ∈ {4,8,12,24,48}`.
**Accept:** H6 and H4/H5 configs run to completion on Colab T4 within a documented wall-clock budget.

---

## 6. Pre-registered hypotheses (build the measurement, not the conclusion)

Each needs a config and an analysis function. **Report verdicts honestly — "refuted" is a valid, publishable outcome; "undocumented" is not.**

- **H1 (no chattering).** Compare continuous attenuation vs. a switched objective. *Measure:* per-edge social-gradient sign flips per epoch, bucketed by `|D_ij − δ_i|`. *Confirmed if* the switched variant spikes near the boundary and the attenuated one doesn't.
- **H2 (forward-KL inflates entropy).** *Measure:* student `σ_i` trajectories on high-`D_ij` edges under each KL direction. *Confirmed if* forward-KL drives `σ_i² → σ_j² + (μ_i−μ_j)²` while reverse-KL holds at `σ_j²`.
- **H3 (radius pathologies).** *Measure:* `δ_i` under (a) an intentionally degenerate critic and (b) a 10× reward rescale, for both radius definitions. *Confirmed if* the nonstationarity radius is scale-invariant and contracts under instability, while value-variance shows the predicted pathologies.
- **H4 (simultaneity penalty grows with N).** *Measure:* per-round `surrogate − realized improvement` across the scale sweep, densely coupled topology. *Confirmed if* roughly linear in `N`.
- **H5 (size-independence under locality).** *Measure:* same gap per capita with fixed-degree topology; **plus** directly test the decay assumption — perturb agent `j`'s policy, record induced change in `Q^i` vs. graph distance. *Confirmed if* the per-capita gap is flat in `N` **and** sensitivity decays exponentially with hop distance.
- **H6 (Byzantine tolerance).** *Measure:* final return vs. Byzantine fraction `f ∈ {0, .1, .2, .3, .4, .5}`, trimmed-mean vs. plain-sum, including the adaptive adversary. *Confirmed if* robust aggregation degrades gracefully to `f ≈ 0.4` while plain-sum collapses early.
- **H7 (headline) — restated as an INTERACTION hypothesis (SPEC Sec. 10 / D8).**
  *"D-TrOPD beats IPPO **under heterogeneity**, while **costing** something under
  homogeneity."* The claim is the **predicted sign flip across environments**, not
  a one-sided improvement. Both environments are in the sweep and the homogeneous
  arm is a **control condition**, never a throwaway.
  *Measure:* learning curves and final returns on **both** environments, **10
  seeds**, paired seeds, 95% bootstrap CIs, IQM; the per-environment difference
  `R(D-TrOPD) - R(IPPO)` **and the interaction** (that difference's sign flip
  between heterogeneous and homogeneous); per-zone returns so gains are not an
  artifact of one sector; the consensus-pull cost of D9; and the calibration
  sensitivity sweep of D7.
  *Confirmed if* the difference is positive under heterogeneity **and** the
  interaction is significant — reporting the homogeneous cost explicitly — and
  D-TrOPD ≥ NaiveDistill everywhere.
  *Rationale for the restatement:* a predicted sign flip is far harder to obtain
  by accident than a single-direction improvement, so it is the stronger claim.
  A homogeneous-env cost is therefore not a negative result to be buried; it is
  half of the prediction.

### Ablation grid

| Ablation | Variants |
|---|---|
| Attenuation | continuous / hard switch / hard cutoff |
| KL direction | reverse / forward / switched |
| Radius signal | nonstationarity / value-variance / fixed (κ=0) |
| Trust-test batch | own rollout / pooled |
| Aggregation | trimmed mean / geometric median / plain sum |
| Neighbor selection | all / hard top-k |
| β | {0, 0.01, 0.05, 0.1, 0.5} vs. the theoretical β_max |

---

## 7. Reporting standards

- ≥10 seeds per configuration; report mean, 95% bootstrap CIs, and IQM (`rliable`). Never a single seed.
- Paired seeds and identical env instances for D-TrOPD vs. IPPO.
- Report compute, wall-clock, **and communication volume** (bytes of snapshots exchanged) — decentralization's cost/benefit claim depends on it.
- Every figure regenerable from stored metrics by one command, no manual steps.

---

## 8. Watchlist — regressions from the failed prototype

If any of these appear, an old bug has returned:

- Sudden per-agent entropy blow-ups → a forward-KL path leaked back in (violates I2).
- Trust radius *growing* as training destabilizes → value-variance radius leaked back in (violates I1).
- Same neighbor oscillating in/out across consecutive epochs → selection churn; confirm all-neighbor default is active.
- Trust scores disagreeing between pooled-batch and own-rollout evaluation → pooled path is live (violates I3). Own-rollout is authoritative.
- D-TrOPD tracking IPPO during warmup then degrading once β reaches full strength → **do not dismiss this.** It was observed in preliminary prototype runs. Check I1–I9 first, then treat it as a real empirical finding requiring multi-seed GPU confirmation.

---

## 9. Rules of engagement for Claude Code

1. **One phase at a time.** Stop at each acceptance test, report the result, wait for approval.
2. **Tests before features** for anything touching Sec. 2 math.
3. **Ask, don't assume.** If this spec is ambiguous, ask rather than picking a plausible default — the prototype failed on exactly this kind of silent choice.
4. **Never soften an invariant to make a test pass.** If an invariant seems to make something impossible, say so and stop.
5. **No hidden hyperparameter tuning** of D-TrOPD without equal tuning of baselines. Unfair comparisons are worse than no comparisons.
6. **CPU smoke path always works:** every feature must be runnable in a <2-minute 2-agent CPU config for debugging.
7. Commit per phase with a message naming the phase and its acceptance result.

---

## 10. Locked decisions — 2026-08-05

These resolve ambiguities surfaced during Phases 0–3. They are **final**: they
have the same standing as Sec. 2 and Sec. 3, and changing one requires the same
deliberation as changing an invariant. Each names the code that enforces it.

### D1 — Headline arm is the *calibrated* configuration; calibration is part of the method

Measurement from Phase 3: with the shipped `delta_base = 0.1`, `gamma = 1.0`,
the observed social divergences on the cooperative-capture environment are
`median D_ij = 0.026` over 1056 edges, giving `mean w_ij = 0.987`. The trust
gate is inert, and D-TrOPD at those settings **is** NaiveDistill.

H7 and every headline comparison must therefore run on a calibrated
configuration. Calibration is a **documented procedure**, not a tuned constant:

```
1. Run a calibration window with beta held at exactly 0, so the gate is
   inactive and calibration cannot influence the policies it measures. The
   beta warmup ramp starts after this window, at its full configured length,
   so invariant I8 is unaffected. (A linear ramp is not zero after its first
   epoch, which is why the calibration window is not merely "the warmup".)
2. Collect every observed D_ij over that window.
3. delta_base := median(D_ij)
4. gamma      := -ln(0.5) / [ softplus(p95(D_ij) - delta_base)
                              - softplus(-delta_base) ]
   i.e. an edge at the 95th percentile of observed divergence is attenuated
   to w = 0.5.
```

A per-task magic constant is a reviewer objection; a measurement procedure with
a stated rule is not. Reproducible two ways, both required to exist:

* in-run, `distill.calibration: warmup` — the procedure above executes inside
  training during the beta-warmup window;
* offline, one command — `python run.py calibrate --config <in> --out <out>`
  writes a calibrated config with its provenance recorded in `notes`.

Defaults stay uncalibrated so no existing config silently changes; the shipped
headline configs set `calibration: warmup`. `DTrOPD` emits a loud warning
whenever mean `w_ij > 0.95` with `beta > 0`, so an inert gate can never pass
unnoticed again.

### D2 — `env.local_ratio` stays at 0.5 for the primary condition

The pure-global-reward variant (`local_ratio: 0.0`) is a **secondary, clearly
labelled condition** and must never be headlined.

Rationale, recorded so it is not relitigated: under a pure global reward one
agent in twelve moves the shared return by roughly 1/9 and IPPO does not learn
at all. That produces a floored baseline, which is a weak comparison. More
importantly, the pure-global failure is a **credit-assignment** failure, and
this work is about **distribution-mismatch** failure. Beating a baseline that
failed for the wrong reason proves nothing about the mechanism under test.

### D3 — Reward shape is frozen at `reward_spec_version = "v1"`

The environment asserts the version it implements
(`CooperativeCaptureEnv.REWARD_SPEC_VERSION`); a config asking for any other
version fails at construction. Every run stamps `reward_spec_version` into its
`metadata.json`. Any change to the definition below requires a version bump and
**invalidates every stored curve**.

**Reward specification v1** (verbatim; `dtropd/env.py::_rewards_and_info`).
Symbols: `N` agents at positions `x_i`, `T` targets at positions `y_k`,
`m = capture_group_size`, `d(i,k) = ||x_i - y_k||`.

```
For each target k, let n_k(1) <= n_k(2) <= ... <= n_k(m) be the m smallest
values of d(i,k) over all agents i:

  shaping_k = - (1/m) * sum_{r=1..m} n_k(r)
  captured_k = 1 if n_k(m) <= capture_radius else 0

Collisions are unordered agent pairs closer than 2 * agent_size:

  C     = #{ (i,j) : i < j, ||x_i - x_j|| < 2 * agent_size }
  c_i   = #{ j != i : ||x_i - x_j|| < 2 * agent_size }

Global (shared) term:

  G = shaping_weight * sum_k shaping_k
    + capture_bonus  * sum_k captured_k
    - collision_penalty * C

Per-agent local term (no capture bonus: capture is always a shared signal):

  L_i = - shaping_weight * min_k d(i,k)
        - collision_penalty * c_i

Reward delivered to agent i on this timestep:

  r_i = (1 - local_ratio) * G + local_ratio * L_i
```

Defaults: `shaping_weight = 1.0`, `capture_bonus = 1.0`,
`collision_penalty = 0.1`, `local_ratio = 0.5`, `capture_radius = 0.3`,
`capture_group_size = 3`, `agent_size = 0.05`. Episodes are fixed length
(`max_cycles`): they truncate, never terminate, so values are bootstrapped at
the horizon.

### D4 — NaiveDistill is its own algorithm class

NaiveDistill is **not** expressible as D-TrOPD with `gamma_attenuation = 0` or
with an inflated `delta_base`. It is an explicit class in which `w_ij = 1`
structurally: no trust radius, no attenuation, no code path for either.

Reason: if the baseline is reachable from the method by a config value, then
"NaiveDistill" and "miscalibrated D-TrOPD" are the same run, and the headline
comparison is ambiguous by construction — which is exactly the failure D1
documents. `tests/test_baselines.py` asserts that `NaiveDistill` exposes no
radius or attenuation attribute.

### D5 — Adaptive `delta_base` is an ablation arm, never the default

`D_ij` drifts upward during training (0.0001 -> 0.038 over 40 epochs on the
heterogeneous environment), so a *fixed* `delta_base` means the gate tightens
over training for reasons unrelated to the mechanism the theory describes.

`distill.radius_reference` selects between:

* `fixed` (**default**) — `delta_base` is a constant, as Sec. 2.2 states;
* `running_quantile` — `delta_base` tracks a running quantile of observed
  `D_ij`, making trust *relative*: "unusually divergent for this swarm right
  now" rather than absolute.

Both arms log the `D_ij` distribution per epoch (p5/p50/p95 and drift relative
to the first epoch) so the confound is measured either way.

**`running_quantile` is not what the theory currently states.** Adopting it as
the default would require re-deriving the scale-invariance proposition of
Sec. 2.2, because a radius defined from a running quantile of `D_ij` is no
longer a fixed function of the agent's own nonstationarity.

---

## 11. Locked decisions — 2026-08-05 (second set)

Continues Sec. 10. Same standing: these are final, and changing one requires the
same deliberation as changing an invariant.

### D6 — The calibration rule is fixed and pre-registered

There is **no principled window length, and no principled value exists**. The
arbitrariness is made harmless by fixing the rule and testing sensitivity to it
(D7), not by searching for a "true" value.

The rule, exactly:

```
1. A hard beta = 0 prefix of 8 EPOCHS. The beta warmup ramp starts after it.
2. delta_base := median D_ij, pooled over ALL edges across the FINAL 2 EPOCHS
   of that window. (Pooling two epochs damps the drift relative to a
   single-epoch reading; D_ij rises steeply early in training.)
3. gamma set so an edge at the p95 of the same pooled sample attenuates to
   w = 0.5.
4. Measured ONCE per environment configuration, then PINNED for every arm in
   that comparison -- D-TrOPD, NaiveDistill, and any ablation sharing the env.
```

**The number 8 is arbitrary.** It has no theoretical justification and none is
offered. It is fixed so that it is not a researcher degree of freedom, and its
influence is measured by D7.

The measurement is emitted as a **calibration artifact** (`calibration.json`)
recording: window length, which epochs were pooled, edge count, the full `D_ij`
quantile table, `reward_spec_version`, and the git commit. Every arm in a
comparison must load **the same artifact file**;
`dtropd.calibration.assert_shared_calibration()` refuses to run a sweep whose
arms were calibrated separately, and `run.py sweep-check` exposes it.

Arms that have no radius (NaiveDistill, per D4) still **declare** the artifact
their comparison used, so the sweep is auditable, without acquiring a
`delta_base` they must not have.

### D7 — Calibration sensitivity is pre-registered as part of H7

H7 runs at `delta_base` in **{0.022, 0.026, 0.049}** -- the three values the
6-, 8- and 10-epoch windows actually produced on this environment.

**Decision rule, fixed now, before any result is seen:**

* verdict identical at all three -> report *"conclusion stable across a 2x range
  of delta_base"*; the window choice is then immaterial;
* verdict flips between them -> **that is the finding**: the benefit is
  calibration-sensitive, and it is reported as a limitation.

**Do not pick the window that gives the nicer answer.** The sweep is run in
full and all three are reported regardless of which way they come out.

### D8 — H7 is an interaction hypothesis

Restated in Sec. 6. Both environments are in the sweep; the homogeneous arm is a
control condition; the predicted **sign flip** is the claim; the interaction is
reported explicitly.

### D9 — The consensus-pull cost is a first-class measured quantity

Observed in Phase 4: both distillation arms sit below IPPO on the homogeneous
environment, together, with the gap opening early and staying flat. Reading:
distillation pulls agents toward consensus, which costs where agents are already
similar and there is nothing to gain.

This is **not an anecdote**. It is the floor the heterogeneous benefit must
clear for the method to net out positive, so it is measured directly and
reported with CIs in the H7 table. Three quantities, all in the metrics file:

* `consensus_pull/social_grad_fraction` -- the share of the update attributable
  to `beta * grad L_soc`, per agent per epoch;
* `consensus_pull/policy_dispersion_*` -- quantiles of pairwise `D_ij` across
  the graph; consensus shows up as dispersion collapsing;
* `consensus_pull_cost` -- `R(IPPO) - R(distiller)` on the homogeneous control,
  with a bootstrap CI, computed by `dtropd.analysis`.

### D10 — Housekeeping

Merged PR #1's description was edited after it merged and describes Phase 3 work
its diff does not contain. A correction comment is posted on it. Git history is
evidence in a research project and must not be quietly misleading.

### Metrics schema versioning

The metrics schema is versioned (`metrics_schema_version`) and stamped into
every `metadata.json`. A schema change after H7 starts invalidates stored curves
exactly as a `reward_spec_version` bump does.

Logging preference, now a rule: **log distributions (quantiles), not just
means.** The `delta_base` failure of Sec. 10 / D1 was invisible in the mean and
obvious in the `D_ij` distribution.

---

## 12. Amendments and locked decisions — 2026-08-05 (third set)

Continues Sec. 10-11. Same standing.

### D7-AMENDED — the sensitivity grid is a procedure, not three numbers

**Original wording (Sec. 11 / D7):** "H7 runs at `delta_base` in
**{0.022, 0.026, 0.049}** -- the three values the 6-, 8- and 10-epoch windows
actually produced on this environment."

**Amendment.** Those literal values were derived from a 6/8/10-epoch window
sweep on an environment configuration that has since changed; the Phase 5
environment calibrates to ~0.12, outside the grid entirely. Holding the literal
numbers would not test window sensitivity -- it would test three arbitrary
*severe mis-calibrations*, which is a different and far less interesting
experiment.

**Amended rule:**

```
Run the D6 calibration at window lengths 6, 8 and 10 on the ACTUAL H7
environment. The three resulting delta_base values ARE the sensitivity grid.
```

**Spread check.** After deriving the grid, report `max/min`. If the spread is
under **1.5x** the check tests nothing; widen deliberately to **+/-2x** around
the calibrated value and record that this was done and why.

**Decision rule: UNCHANGED.** Identical verdict at all three -> report
"conclusion stable across the calibration-window range". Verdict flips ->
report calibration-sensitivity as a limitation. **Never pick the window with
the nicer answer.**

**Provenance.** This is a **disclosed pre-outcome amendment, not a post-hoc
revision.** No H7 run has been performed, so no result influenced it. The
reason is that the literal values derived from a superseded environment config;
D7's intent was the *spread*, not the numbers. Both the original wording and
the amendment are recorded above so the change is auditable.

### D11 — Assumption 2 is violated by the primary environment; H5 gets its own condition

The reward (Sec. 10 / D3) is
`r_i = (1 - local_ratio) * team + local_ratio * local_i`, where `team` sums
shaping, capture bonus and collisions over **all** agents. At
`local_ratio = 0.5`, half of every agent's reward is a global quantity
depending on every agent at every hop distance, **with weight independent of
distance**.

The theory paper's Assumption 2(i) requires `r_i` to depend only on
`(s_{N_i}, a_{N_i})`. It does not. **Assumption 2 is violated by construction
in the primary condition**, and the decay probe's positive log-slope (+0.78
observed in Phase 5) may be reporting that correctly rather than being an
artifact of low power.

Three required actions:

**(a) Fix the probe's power, regardless of the above.** Two hop levels cannot
support a decay fit. Use a ring or path topology over 12-24 agents to obtain
**diameter >= 6**; raise `decay_probe_episodes` substantially; run multiple
seeds; report the fit **with a CI and the number of hop levels it used**. A
slope derived from **fewer than 4 hop levels must be reported as
uninterpretable**, and the code labels it so.

**(b) Add a local-reward condition, used by H5 only.** The team term is summed
over `N_i` rather than over all agents -- preferred over `local_ratio = 1.0`
because it keeps the coordination signal while making it local. This is an
environment in which Assumption 2 **actually holds**, and is the honest place
to test the theorem's prediction. H5 becomes *"the size-independent bound holds
where its assumption holds."*

**(c) Measure and report the violation in the primary environment** as a result
in its own right: *"Assumption 2 fails in standard MPE-style cooperative
navigation because the team reward is global; Theorem 2 therefore does not
cover the most common MARL benchmark class."* The decay probe is reported in
**both** conditions, side by side.

**Scope note (do not conflate with D2).** This does **not** reopen D2.
`local_ratio = 0.5` remains H7's primary arm; the argument that a pure-global
reward tests credit assignment rather than distribution mismatch still holds.
**D2 governs H7.** H5's locality condition is the *subject* of H5's test and
therefore needs its own environment. The two scopes are separate.

### D12 — Audit artifacts are written after the thing they audit

Standing rule, generalised from the Phase 5 metadata fix: **any artifact whose
purpose is auditability is written at the END of the run**, with only a
crash-safety stub written early.

A calibration measured at epoch 7 cannot appear in a `metadata.json` written at
epoch 0; writing it early and never revisiting it silently defeats the
auditability the artifact exists for. Applies to every future artifact of this
kind, not just calibration.

### D13 — Reward spec versions may coexist; comparisons may not span them

`v1` is **frozen and unedited**. `v1-local` is a **new** version used **only**
by H5's locality condition (team term summed over `N_i` rather than all
agents). H7 remains entirely on `v1` per D2.

**Rule:** no **algorithm-performance** comparison may span reward spec
versions. Any figure or table comparing algorithms must draw all of its arms
from a single version, and the launcher **refuses otherwise** -- the same
enforcement pattern as the D6 calibration sweep-check.

**Exception, explicitly allowed and explicitly labelled:**
**environment-property** measurements *may* span versions, because the
environments themselves are the subject. D11(c)'s side-by-side decay probe on
`v1` vs `v1-local` is such a measurement: it compares whether Assumption 2
holds, not how algorithms perform. It is labelled **cross-version** wherever it
appears.

**Enforcement.** Four guards, because each clause of the rule can be broken
independently:

1. `check_reward_spec_versions` refuses a *plan* whose algorithm arms span
   versions (the launcher guard the rule names).
2. `check_manifest_reward_spec_versions` refuses to build a *report or figure*
   from a manifest whose algorithm runs span versions, and runs before the
   report touches disk. The launcher guard alone is not sufficient: D13 binds
   "any figure or table", and `run.py sweep --report-only` rebuilds the figure
   from a manifest on disk without going through `run_sweep`, so `preflight`
   never executes. A manifest merged from two sweep directories, or written
   before the config was edited, would otherwise render silently.
3. `test_v1_reward_spec_is_frozen` pins `v1`'s rewards for a hand-placed
   configuration. "Frozen" only means something if editing it fails loudly: a
   change to `v1` silently invalidates every stored curve, calibration artifact
   and H7 arm already denominated in it. A deliberate change is a *new* version,
   which is the mechanism this decision exists to provide.
4. `test_v1_local_is_used_only_by_the_h5_locality_condition` parses every
   shipped config and refuses `v1-local` outside `h5_locality.yaml` -- the way
   the scope restriction breaks in practice is someone reaching for the
   localised reward in an unrelated sweep.

The cross-version label is rendered onto the figure itself, not only into
`report.json`, since a figure is where it is most likely to be read without the
report open beside it.

---

## 13. Amendments and locked decisions — 2026-08-05 (fourth set)

### D14 — H7's gate is stated here and nowhere else

Two statements of when H7 may run had drifted apart:

* an **instruction** given while Phase 6 was being built -- *"do not run H7;
  Phase 6 builds the harness, Phase 7 runs the experiment"*;
* **SPEC Sec. 5**, which places only **robustness + scale** in Phase 7 and
  attaches H7 to Phase 6's acceptance criterion.

**For the record: the Phase 7 gate was a sequencing choice, not a SPEC
requirement.** It was made to stop a preliminary multi-seed look from biasing
later interpretation, which is a good reason, but it was never written into the
build order, so the build order and the instruction disagreed. Neither was
wrong; they were two different documents.

**The gate, stated once, superseding both.** H7 runs when **both** hold:

1. **The D7 sensitivity grid has been derived on the actual H7 environment and
   confirmed.** Not the literal values from any earlier environment
   configuration -- see D7-AMENDED. The grid is an input to H7, so it cannot be
   chosen after seeing H7's outcome.
2. **The harness has been load-tested by H6 and H4/H5.** Those sweeps exercise
   resume, multi-seed pairing and the report path at real scale. Discovering a
   harness defect *during* the headline run is how a headline run gets repeated,
   and a repeated run is a run whose first outcome is already known.
   *(Amended below — see D14-AMENDED. Condition 2 is satisfied; H6 is not
   required for it.)*

Phase membership is *not* a gate. H7 does not require Phase 7 to be complete,
only these two conditions. Anything else that reads like a gate on H7 elsewhere
in this document is superseded by this decision.

### D15 — The scale sweep's cap is asymmetric, and the asymmetry is the point

SPEC Sec. 5 asks for `N ∈ {4,8,12,24,48}`. Cost does not grow the same way in
the two hypotheses that use it, so a single cap would either waste compute or
throw away the measurement that matters most:

| hypothesis | topology | edges | cap |
|---|---|---|---|
| **H4** — simultaneity penalty grows with `N` | densely coupled (complete) | `O(N²)` | **N ≤ 24** |
| **H5** — size-independence under locality | fixed-degree | `O(N·d)` | **N ≤ 48** |

H4's prediction is that the gap is *roughly linear in N*; a slope is
identifiable from `{4, 8, 12, 24}`, and `N=48` on a complete graph costs
roughly four times `N=24` for one more point on a line. H5's prediction is that
the per-capita gap is *flat in N*, and flatness is exactly the claim that a
larger `N` tests hardest -- so H5 keeps `N=48`, which its fixed-degree topology
makes affordable.

**If `N=48` fixed-degree does not fit the execution environment, the measured
limit is reported** -- the largest `N` that completed, with its wall-clock --
rather than the point being silently dropped. A scale sweep that quietly stops
early reads as a completed sweep.

---

## 14. Phase 7 findings — 2026-08-05

Measurements, not decisions. Recorded here because they bear on how H7 should
be read, and because the first of them was found by running the scale sweep
rather than by reasoning about the design.

### F-1 — The calibrated trust gate goes inert as **degree** grows, not as `N` grows

**This supersedes the first version of F-1, which attributed the effect to `N`.
The H5 fixed-degree sweep refuted that reading, and the correction matters: it
changes what would fix it.**

Measured on the H4 scale sweep (complete graph, so degree = `N-1`), 5 seeds per
point, `w` and `D_ij` at end of run; `delta_base` from the D6 artifact:

| `N` | degree | `delta_base` (measured at β=0) | operating `D_ij` p50 (β>0) | ratio | mean `w_ij` |
|---|---|---|---|---|---|
| 4 | 3 | 0.103 | 0.094 | 1.1x | 0.68 — active |
| 8 | 7 | 0.090 | 0.036 | 2.5x | 0.89 |
| 12 | 11 | 0.114 | 0.022 | 5.1x | 0.93 — **H7's scale** |
| 24 | 23 | 0.082 | 0.011 | 7.5x | 0.95 — inert by the D1 threshold |

Read alone, that table says the gate dies as the system grows. The H5 sweep --
identical in every respect except a **fixed** degree of 4 -- says otherwise:

| `N` | degree | mean `w_ij` |
|---|---|---|
| 4 | 2 | 0.60 |
| 8 | 4 | 0.78 |
| 12 | 4 | 0.78 |
| 24 | 4 | 0.67 |
| 48 | 4 | 0.68 |

At fixed degree the gate stays active out to `N = 48` and is **flat in `N`**.
Sorting both sweeps by degree instead makes the relationship monotone and the
confound obvious:

| degree | 2 | 3 | 4 | 7 | 11 | 23 |
|---|---|---|---|---|---|---|
| mean `w_ij` | 0.60 | 0.68 | 0.67–0.78 | 0.89 | 0.93 | 0.95 |

`N = 12` appears in both sweeps: `w = 0.78` at degree 4, `w = 0.93` at degree 11.
Same size, same everything else, different gate.

**Mechanism, and it is not a bug.** D1 requires the calibration to be taken
with the gate *inactive*, so the window holds `β = 0` exactly. But the gate then
operates under `β > 0`, and distillation is itself a force pulling neighbours
together — so the `D_ij` distribution the gate meets is compressed relative to
the one it was fitted to. **Each neighbour is one more source of pull**, so the
compression scales with the number of neighbours an agent averages over, which
is degree. `N` only appeared to drive it because the H4 sweep uses a complete
graph, where degree and `N` move together. Both D1 and D6 are being followed
exactly; the finding is that following them yields a weak gate in
densely-connected topologies.

**Why it matters for H7.** H7 specifies `topology: complete` at `N = 12`, i.e.
degree 11, where the gate attenuates about 7%. D-TrOPD is then behaviourally
close to NaiveDistill — the comparison H7's headline rests on. A null result
would be ambiguous between "the trust gate does not help" and "the trust gate
was barely switched on", and those are very different conclusions. Note the
implication of the correction: this is a property of H7's **topology choice**,
not of its size, and the same `N = 12` at degree 4 gives an active gate
(`w = 0.78`).

**What is deliberately NOT changed.** D5 fixes adaptive `delta_base` as an
ablation arm and never the default; D6 fixes the calibration rule; D2 and the
H7 config fix the topology. None is touched here. Re-fitting the calibration —
or quietly switching H7's topology — after seeing that the gate comes out weak
would be selecting a condition against an outcome, which is the thing the
pre-registration exists to prevent. Recorded as a measurement, left for an
explicit decision.

**How this was gotten wrong first.** The initial version of F-1 read the H4
sweep alone and attributed the effect to `N`. That reading survived because H4's
complete graph confounds degree with `N` perfectly, and it was only refuted when
the H5 fixed-degree sweep — which was run for a different hypothesis entirely —
produced a flat `w` out to `N = 48`. The correction is recorded rather than
silently rewritten because the two versions imply different remedies: the first
suggests the method does not scale, the second suggests it is sensitive to how
densely the graph is wired.

### F-2 — `N = 48` fixed-degree fits; the D15 asymmetry is vindicated

| sweep | `N` | topology | edges | s/epoch | peak RSS |
|---|---|---|---|---|---|
| H4 | 24 | complete | 552 | 2.20 | 0.74 GB |
| H5 | 24 | fixed-degree | 96 | 2.15 | 0.74 GB |
| H5 | 48 | fixed-degree | 192 | 4.10 | 0.87 GB |

D15's fallback — report the measured limit if `N = 48` does not fit — is not
needed. The edge counts show why the cap is asymmetric rather than arbitrary:
`N = 24` on a complete graph carries nearly three times the edges of `N = 48`
fixed-degree.

---

## 15. Amendments — 2026-08-05 (fifth set)

### D14-AMENDED — the load-test gate names a property, not a particular sweep

D14 condition 2 required the harness to be load-tested "by H6 and H4/H5". That
named three sweeps when what it meant was a **property**: that the headline run
should not be the first real exercise of the machinery.

**The requirement, restated.** Condition 2 is satisfied by **any completed
multi-seed sweep driven through the same launcher path** — `run.py sweep` →
`build_plan` → `preflight` → `run_sweep` → manifest → `sweep_report`. That path
is what H7 will use, and a sweep that has run it end to end has tested it.

**Condition 2 is now satisfied.** `h4_scale` (20 cells, 4 arms x 5 seeds,
9.0 min) and `h5_scale` (25 cells, 5 arms x 5 seeds, 16.1 min, out to `N = 48`)
both completed through exactly that path. Between them they exercised
auto-calibration grouped by environment fingerprint, the D6 refusal when no arm
declares an artifact, multi-seed pairing, parallel workers, resume, manifest
assembly and the report path — and they found four real defects in it, which is
the strongest available evidence that the load-test did its job.

**H6 is therefore NOT a gate on H7.** It remains a Phase 7 deliverable and a
hypothesis in its own right; it is simply not load-bearing for the headline.
Nothing about H6's *content* was ever the reason for condition 2 — Byzantine
tolerance tells you nothing about whether the launcher works — so requiring it
was over-specification, not caution.

**What is unchanged.** Condition 1 stands untouched: H7 still requires the D7
grid derived on the actual H7 environment **and confirmed**. That is a separate
gate with a separate rationale, and it is not affected by this amendment.

### D16 — H6 runs at full budget, elsewhere; a trimmed budget must earn its way in

**H6 runs at the budget `h6_byzantine.yaml` already specifies**: 150 epochs,
40 envs, 10 seeds, 16 arms, 160 cells. No epoch trim, no env trim, no seed cut.
SPEC Sec. 7's ">= 10 seeds per configuration" is a reporting standard, and a
Byzantine-tolerance curve read off 5 seeds is not the curve.

It runs **outside this execution environment** — see the runner in
`experiments/run_h6.sh`. The reason is environmental rather than scientific:
this container executes only while a session is actively issuing calls, so
background progress stops between turns. Measured directly — H6 completed 13
cells during dense activity, then produced zero cells across two 20-minute idle
windows, while a foreground calibration finished in 50 seconds.

**If a trimmed budget ever becomes necessary**, it must first be shown to be
interpretable, *before* the full sweep is run on it:

> Run `f = 0` and `f = 0.5` at the proposed shorter budget. Their final returns
> must separate visibly, with non-overlapping bootstrap CIs. Only then may the
> trimmed budget be used.

The reason is that H6's result is the *shape of a curve over `f`*. At a budget
too short for the attack to bite, every cell returns roughly the same number and
the curve comes out flat — and a flat curve is indistinguishable between "robust
aggregation held" and "nothing had time to happen". The endpoints are the
cheapest possible test of whether the budget can express the effect at all, and
`experiments/h6_budget_check.py` runs exactly that check.
