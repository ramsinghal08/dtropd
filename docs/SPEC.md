

## 1. Mission

Implement **D-TrOPD** (Decentralized Trust-Region Policy Distillation) for networked multi-agent RL, plus baselines and a pre-registered experiment suite.

Each agent runs PPO locally and additionally distills from neighbors, but *gates* how much it imports from each neighbor by a trust weight based on behavioral divergence. There is no central critic and no sequential update order.

Section 3 lists invariants that every code path must satisfy. A violation is a bug regardless of whether results look good.

**Target hardware:** development on a laptop GPU (4 GB VRAM); full runs on a single 15 GB GPU. Everything runs CPU-only for smoke tests and scales to a single GPU for real runs. There are no multi-GPU or cluster assumptions.

---

## 2. Algorithm specification

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

## 3. Invariants

Each is enforced as a runtime assertion or an automated test. Failing any one invalidates the run.

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

## 5. Components and acceptance criteria

The implementation is organised into stages, each with an acceptance test.

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

## 8. Diagnostics

Symptoms that indicate a regression:

- Sudden per-agent entropy blow-ups: a forward-KL path has leaked back in (violates I2).
- Trust radius growing as training destabilises: a value-variance radius has leaked back in (violates I1).
- The same neighbour oscillating in and out across consecutive epochs: selection churn; confirm the all-neighbour default is active.
- Trust scores disagreeing between pooled-batch and own-rollout evaluation: a pooled path is live (violates I3). Own-rollout is authoritative.
- D-TrOPD tracking IPPO during warmup and degrading once beta reaches full strength: check I1 to I9 first, then treat it as an empirical finding requiring multi-seed confirmation.

---

## 9. Methodology

1. Tests precede features for anything touching the math in Sec. 2.
2. No invariant is relaxed to make a test pass.
3. No hyperparameter tuning of D-TrOPD without equal tuning of the baselines.
4. Every feature is runnable in a two-minute, 2-agent CPU configuration.
5. Each stage is committed with its acceptance result.

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

---

## 16. D7 sign-off and grid extension — 2026-08-05

### D17 — The measured grid is signed off; one point is ADDED, and it is not a widening

**Sign-off, recorded explicitly.** The D7 sensitivity grid derived on the actual
H7 treatment environment (`h7_env_het.yaml`, 264 edges per window) stands as
**pre-registered**:

| window | `delta_base` | `gamma` |
|---|---|---|
| 6 | **0.051703** | 14.8832 |
| 8 | **0.079151** | 10.1647 |
| 10 | **0.092786** | 9.4748 |

**Spread max/min = 1.795, which clears the 1.5x threshold, so the D7-AMENDED
widening clause does NOT fire.** `widened: false` in the artifact. This is
recorded rather than left implicit because "the grid was widened" and "the grid
was extended" would otherwise be indistinguishable in the record, and they have
opposite meanings: the first is a rule firing automatically on a degenerate
spread, the second is a deliberate addition.

**The addition.** A fourth point is added at **2x the calibrated value**:

    delta_base = 2 x 0.079151 = 0.158302   (treatment)
                 2 x 0.076916 = 0.153832   (control)

**This is NOT a D7 widening.** The spread check passed; nothing triggered. It is
a separate, deliberate extension, and the reason is the F-1 inertness finding.

**Why.** The pre-registered grid spans **0.65x to 1.17x** of the calibrated
value — almost entirely *tighter* gates. The loose-gate direction is untested,
and that is precisely where the question *"is this benefit gating, or is it just
distillation?"* lives. A looser `delta_base` admits more neighbours inside the
trust radius, pushing D-TrOPD toward NaiveDistill. Without a point in that
direction, a positive H7 result could not distinguish a benefit that comes from
the trust gate from one that comes from distillation per se, which is the
central claim of the method.

**Why it is legitimate to add it now.** Two properties, both checkable:

1. **Pre-outcome.** No H7 arm has run. The addition cannot have been chosen to
   produce a preferred result because no result exists to prefer.
2. **Strictly harder.** A looser gate is *closer to the baseline it must beat*.
   Adding this point can only make H7 harder to confirm, never easier. An
   extension that could only hurt the hypothesis is not one that a preferred
   outcome would motivate.

**Provenance is kept honest.** The three measured points come from calibration
windows. The fourth does not — no window produces it — so its artifact is
written by scaling the window-8 record, carries `gamma` over unchanged, and
records in its own `note` field that it is **DERIVED, NOT MEASURED**. It keeps
its content hash, so it is as tamper-evident as the measured ones, and it is
never presented as a measurement.

**Both environments.** H7 spans the heterogeneous treatment and the homogeneous
control, and D6 gives each its own artifact — the D6 guard refused the first
attempt to pin one environment's artifact to both, correctly. Each grid point
therefore pins two artifacts, and the control arms carry their own `delta_base`
by per-arm override. The control's own measured grid has spread 2.003x, which
also does not trigger widening.

**H7 is APPROVED to run at these four points.** Condition 1 of D14 is now
satisfied (grid derived *and* confirmed); condition 2 was satisfied by
`h4_scale` and `h5_scale` under D14-AMENDED. The verdict is the **interaction**
— the predicted sign flip between environments — and whether it survives all
four grid points. A single positive point is not the result.

---

## 17. H6 verdict and the mechanism experiments — 2026-08-05

### H6 — INCONCLUSIVE PENDING MANIPULATION CHECK

> **Scope corrected by D19 (Sec. 19), 2026-08-06.** What follows stands as
> written, but "inconclusive" was the wrong label for *scope*: H6 runs a
> gradient-level adversary and therefore tests the **aggregators**, which is
> precisely Prop. 5's subject. It is correctly scoped, and it can neither
> support nor undermine a claim about the trust gate. The open question below —
> whether the adversary bit at all — is unaffected.

**Explicitly NOT "robust".** 160/160 cells, 10 seeds, full budget, 803.7 min.

| `f` | adaptive / plain sum | adaptive / trimmed mean |
|---|---|---|
| 0.0 | -36.21 | -36.21 |
| 0.5 | -36.08 | -35.92 |

Paired against each aggregator's own `f = 0`, **not one of the 15 attacked arms
differs from baseline** -- every bootstrap CI spans zero. Plain sum was predicted
to collapse early and did not move at all.

**Why this is not a confirmation.** Seed sd within a cell is **6.81**; sd between
conditions is **0.58**. An effect an order of magnitude below the noise it must
be read against is not evidence of robustness -- a flat curve is equally
consistent with *"robust aggregation held"* and *"the attack never bit"*, and
those are opposite conclusions with identical plots.

The D16 endpoint test -- `f = 0` versus `f = 0.5` must separate with
non-overlapping CIs -- **fails at the full budget**. D16 introduced that test as
a precondition for a *shortened* budget, on the assumption that the full budget
was safe. It was not.

**The control arm is what makes this diagnosable.** `sign_flip` at `f = 0.4`
with plain sum is the most damaged cell the design contains: loudest attack, no
defence, 40% of neighbours negating their gradients. Against baseline it gives
**+0.47 [-1.37, +2.40]** -- nothing. If negating 40% of the distillation
gradients is undetectable in final return, the Byzantine channel is not
load-bearing, and no aggregation rule can be credited for surviving it. Including
a positive control that was expected to *fail* is the only reason this is
distinguishable from success.

**Required before H6 can be given any verdict:** a manipulation check showing the
social term materially affects the update at the budget used. `beta * L_soc`'s
gradient share is the quantity (`experiments/social_share.py`), and H2 is the
experiment.

**The adversary is NOT modified.** It behaves as specified and was verified to
sit inside the honest range by construction. Changing it now, after seeing a
flat curve, would be tuning an experimental condition against its own outcome.

### H6 reporting defect (fixed, no re-run)

`sweep_report` groups by `(condition, algo)`, and `h6_byzantine.yaml` gave all
six `f` arms of an aggregator the same condition -- so the report **pooled the
`f` axis away**, marginalising out the hypothesis. The published figure showed
four points with overlapping CIs and looked like a clean null, having never
asked the question. `experiments/h6_curve.py` regenerates return-vs-`f` with
paired CIs from the existing manifest. Reporting fix only; no cell was re-run.

### H1 / H2 / H3 — the mechanism experiments, and what they decide about H6

Run by `experiments/run_mechanisms.sh` (55 cells, ~1/12th of H6). Each config
carries its pre-registered rule and the deciding quantity in its header.

**H2 is the discriminator for H6.** If the social term produces a large clear
effect there, H6's flatness means the *adversary* is inert. If H2 is weak, the
social term is immaterial overall -- which explains H6 and predicts a null H7,
since D-TrOPD differs from IPPO through that term and nothing else.

Two ablations were built for these, both **loud but legal**, on the existing
precedent that `radius_signal='value_variance'` contradicts I1 and exists solely
to measure the pathology I1 forbids:

* `forward_kl_distill` -- a **separate named algorithm**, which is what SPEC
  Sec. 5 Phase 4 requires of forward/switched arms. I2 is untouched: the main
  path still has no direction branch, and `distill.kl_direction` still refuses
  anything but `reverse`. The direction is chosen by which class you instantiate.
* `ppo.degenerate_critic` -- freezes the critic at initialisation, giving H3 the
  "intentionally degenerate critic" it requires. This contradicts I4 and warns
  so. `value_coef = 0` was rejected as the mechanism: I4's hard check exists for
  a reason and softening it to make an experiment runnable is the wrong trade.

---

## 18. Amendment — 2026-08-06

### D18 — `kappa` is per radius SIGNAL, derived once and pinned across arms

**The original value and why it was wrong.** `kappa = 100` was chosen to satisfy
SPEC Sec. 2.2's `kappa * eta in [1, 3]` rule, where the radius signal is
**nonstationarity** -- a KL in nats, of order `1e-5` in these runs. The H3
ablation applied the same `kappa` to a **value-variance** signal of order `3.25`,
giving `delta = delta_base * exp(-325) ~ 1e-141`, i.e. numerically zero, and
`exp(-32440)` under the 10x reward rescale.

This is a **units mismatch, not a performance problem.** `Var(V)` is not a KL,
is not dimensionless, and scales with reward^2. One `kappa` cannot serve two
signals whose magnitudes differ by five orders.

**No outcome influenced this change.** Every value-variance arm was pinned at
`delta = 0`, which is uninterpretable, so there was no result to prefer. The
correction makes the arm able to express its prediction at all -- the same
category as D16's manipulation check.

**Derived value.** `kappa_valvar = 0.636`, from the **1x healthy arm alone**:
observed `Var(V)` range `3.1463` (min 0.1240, max 3.2703), giving
`kappa * range = 2.00` -- the midpoint of the `[1, 3]` band, mirroring Sec. 2.2's
rule for the KL-valued signal.

**The pinning constraint, which matters more than the value.**

> `kappa_valvar` is derived ONCE, on the 1x healthy arm, and **pinned across all
> three value-variance arms** (healthy, degenerate, reward10x).

Deriving it per-arm from each arm's own `Var(V)` would hand the 10x arm a
`kappa` ~100x smaller, exactly cancelling the reward scaling and
**manufacturing the invariance H3 exists to detect**. The arm would then
"demonstrate" scale-invariance by construction. `test_valvar_arms_share_one_pinned_kappa`
asserts the three arms share one value, and that it differs from the
nonstationarity arms'.

**The validation rule is now signal-aware.** `kappa * eta in [1, 3]` is enforced
for `nonstationarity` and **not** applied to `value_variance`, because `eta` is
the KL early-stop cap and the product is meaningless for a non-KL signal --
enforcing it is what demands `kappa ~ 100`. The value-variance arm instead
carries a loud warning that its `kappa` must be derived against the observed
`Var(V)` range and pinned.

### I4 — amended: parameter drift is authoritative, explained variance is secondary

I4 reads "critic loss present and optimized; explained variance logged and
rising". EV is the natural check and **it does not detect a frozen critic**.

Measured: with the critic frozen at initialisation -- parameter drift exactly
`0.000000e+00`, zero value parameters requiring grad -- EV tracked a healthy
critic almost exactly (0.0913 / 0.184 / 0.4277 vs 0.0913 / 0.1859 / 0.4279).
The **value normaliser keeps adapting** when the critic cannot learn, so EV
partly measures the normaliser.

**Amendment:** for any degeneracy check, **critic parameter drift is
authoritative**; EV is a secondary signal and can be actively misleading. EV
remains logged and I4's requirement that the critic be optimized is unchanged --
what changes is which instrument settles the question.

---

## 19. Threat models — 2026-08-06

### D19 — A Byzantine agent has a LEVEL, and the two levels are different threats

`stress.byzantine_level` selects **where** the corruption is injected. This is a
threat model, not an intensity, and the two are not comparable points on one
axis.

| | `gradient` (the original) | `policy` (added here) |
|---|---|---|
| corrupts | the per-edge gradient, at the aggregation site | the broadcast policy snapshot |
| broadcasts | an honest-looking snapshot | a divergent snapshot |
| `D_ij` | unchanged | elevated |
| `w_ij` | unchanged — **the gate is blind by construction** | should collapse |
| defended by | robust aggregation (Prop. 5) | the trust gate |
| `byzantine_scale` units | honest gradient standard deviations | pre-squash mean offset |

`byzantine_scale` means different things at the two levels, so **a scale must
never be carried across a level change without being re-derived**. Config
validation warns when a `policy` arm also names a gradient `byzantine_mode`,
because the gradient attack is not applied there and the config would otherwise
declare an attack it does not run.

**H6's scope is corrected, not its result.** H6 was run at the `gradient` level
and was earlier labelled *inconclusive*. That label was wrong about what the
experiment measured. A gradient-level adversary is behaviourally invisible, so
the gate cannot respond to it — measured, on H6's own logs: Byzantine vs honest
`w_ij` 0.87074 vs 0.87193, paired difference **-0.0029, 95% CI [-0.0071,
+0.0008]**. H6 therefore tests the **aggregators**, which is exactly Prop. 5's
subject, and is **correctly scoped to Prop. 5**. The open question about the
adversary biting at all (Sec. 17) is unchanged and is still what the
manipulation check answers.

**Consequence for the paper: Prop. 5 must be scoped explicitly to the
gradient-level model.** Left unscoped it invites the reading that D-TrOPD's gate
contributes to Byzantine tolerance, which the measurement above rules out.

### The policy-level arm — added after the distinction was found, pre-outcome

`experiments/configs/h6_policy_level.yaml`, **added 2026-08-06**, after the
threat-model distinction above was discovered and **before any policy-level run
existed**. This is recorded plainly because the arm is favourable to D-TrOPD by
design — it is the one adversary the gate is built for — and an arm added late
in a project's life needs its provenance visible.

* `f ∈ {0, 0.2, 0.5}`, 10 paired seeds, D-TrOPD vs NaiveDistill (`w = 1`), 60
  cells.
* **Plain-sum aggregation, deliberately.** With a robust aggregator left on, a
  null would be attributable to either defence and the arm would answer nothing.
* The prediction is written in the config header, dated, pre-outcome.
* The adversary corrupts the **snapshot only**; `agents[j].actor` is untouched,
  so it acts and learns normally. `_rebuild_broadcast` keeps what an agent *is*
  separate from what it *sends*, so the adversary's own radius and
  nonstationarity signal stay honest. Corrupting `snapshots[j]` directly would
  model a broken agent, not an adversary.

### A calibration attack surface, found while building the arm

`delta_base` is calibrated as a **quantile of the observed `D_ij` distribution**.
A policy-level adversary's edges sit far out in that distribution, so an
adversary present during calibration **widens the very radius meant to exclude
it**. Measured on a smoke config with only `f = 0.5` arms: `delta_base` went
`0.1 -> 1.2545` and `gamma` `1 -> 0.472` — the gate largely calibrated itself
away before training began.

This is a property of the calibration rule, not of this arm, and it applies to
any adaptive-radius scheme fitted to the divergences it is meant to police. It
is worth a line in the paper.

`ensure_calibrations` now **prefers an unstressed D-TrOPD arm** as the
measurement template, falling back with a loud warning if a sweep has none.
Previously the template was just the first D-TrOPD arm, so the property held in
every existing config **by arm order alone** — a dependency that survives a
rename and breaks silently. `test_the_shared_calibration_is_measured_on_an_unattacked_arm`
pins it for all three H6 configs.

**What this arm can and cannot establish, stated before it runs.** Given the
offset, "`D_ij` rises and `w_ij` collapses" follows from the attenuation formula
and is close to arithmetic. What it establishes is that the wiring is real —
the corruption reaches `D_ij`, the gate responds, nothing short-circuits it —
which had never been shown. The open question is whether that attenuation
protects **return**. The informative follow-up, not run here, is the **stealth
threshold**: how close to `delta_i` an adversary can sit and still be admitted,
the policy-level analogue of what `adaptive` does to trimming. `D_ij` is logged
per edge, so it is a scope change rather than a code change.

---

## 20. H6(c) results, and a data-integrity defect — 2026-08-07

### H6(c) — POLICY-LEVEL ADVERSARY: predictions (1) and (2) CONFIRMED

60/60 cells, 10 paired seeds, plain-sum aggregation, `δ_base = 0.074296`,
`γ = 10.674` (calibrated on the unstressed `dtropd_f00` arm).

| `f` | `D_ij` byz | `D_ij` honest | `w_ij` byz | `w_ij` honest | paired Δ`w` |
|---|---|---|---|---|---|
| 0.2 | 3.825 | 0.0349 | **0.00000** | 0.8337 | −0.8337 [−0.8352, −0.8323] |
| 0.5 | 3.875 | 0.0665 | **0.00000** | 0.7077 | −0.7077 [−0.7137, −0.7011] |

The gate excludes a lying neighbour **completely** — `w_ij = 0` to five decimals,
100% of the weight removed, on every seed. Against the gradient-level adversary
the same measurement gives 0.3%. The contrast between the two threat models is
**a factor of ~300**, and it is the point of D19.

Predictions (1) and (2) are confirmed. As the config header said before the run,
both largely follow from the attenuation formula once the offset is fixed; what
they establish is that the wiring is real. **Prediction (3), the return curve,
is still outstanding.**

### An unpredicted second-order cost: protection thins the neighbourhood

Not predicted, and it is the more interesting number.

| | `f = 0` | `f = 0.2` | `f = 0.5` |
|---|---|---|---|
| honest-edge `D_ij` | ~0.0074 | 0.0349 | 0.0665 |
| honest-edge `w_ij` | ~0.957 | 0.834 | 0.708 |

**Honest agents diverge from each other ~9× more at `f = 0.5` than unattacked**,
and the gate then attenuates *them* too. The mechanism is direct: excluding a
fraction `f` of every neighbourhood removes that much consensus pull, the
surviving honest agents drift further apart, their own `D_ij` rises, and their
`w_ij` falls — which removes still more pull.

So the gate's defence is not free, and its cost is **not** the excluded edges. It
is the compounding loss of consensus among the agents that remain. This is a
paper-relevant result and it is measured, not argued.

### A data-integrity defect that had already corrupted committed results

`MetricsWriter` opened `metrics.jsonl` in **append** mode. A cell killed
mid-write leaves a partial line; `_is_complete` correctly re-ran the cell, and
the re-run then **appended to the dead attempt**. One file, two runs:

* a **malformed line** where they meet — self-announcing, and how this was found
  (it crashed the `h6_manipulation` report and `gate_split.py`);
* **duplicate epochs** on either side — silent. The file parses, and every
  summary over it (`records[-10:]`, any per-epoch series) averages two runs.

**Three cells of the completed H6 sweep were already corrupted this way**:
`adaptive_trim_f10_seed{3,4,5}`, with 59/56/7 duplicated epochs. Those are among
the cells the gradient-level gate-blindness number was computed on.

**Effect on that finding: none material.** The Byzantine-vs-honest split is made
*within* each record, so duplicated epochs reweight the epoch axis rather than
mixing the compared groups. On the 3 clean `f = 0.1` cells the difference is
**−0.00246 [−0.00563, −0.00035]** against **−0.00291 [−0.00711, +0.00081]** on
all 10 — the same ~0.3% of honest weight either way. The full-10-seed number will
be requoted once the cells are re-run.

**Fixes.** `MetricsWriter` truncates by default (`append=True` exists for a
caller that genuinely continues a stream; nothing does). `read_metrics` raises
`CorruptMetricsError` naming file and line, and **refuses duplicate epochs** —
the silent half now fails as loudly as the noisy one. `_is_complete` reads the
metrics, so an already-corrupted sweep re-runs the affected cells by itself.
`experiments/check_runs.py` reports and optionally deletes corrupt cells.

Nothing is repaired in place. Dropping the bad line still leaves two runs' data
in one file; the only correct fix is to re-run the cell.

### `gate_split.py`'s verdict was sign-based, and that was wrong

It reported "the gate ATTENUATES" for any paired CI excluding zero. On the clean
`f = 0.1` subset that fires for a **0.3%** attenuation — true of the arithmetic,
false of the claim. The verdict now keys on the **fraction of honest weight
removed** and reports anything under 5% as `NEGLIGIBLE`, separated from zero or
not.

### The honest-only return was never wired into the EVALUATION path

Found when `h6_curve.py` refused the policy-level manifest. Three defects, one
of them consequential.

**1. `final_return` is the ALL-AGENT mean, and the H6 curve labelled it
"final return (honest agents)".** `env.py` has always exposed per-agent returns
in the episode info — with a comment saying it exists so H6 can report the
honest subset — and the *rollout* metrics have always split them
(`episode_return_mean_honest`). The **evaluation** path dropped them, so the
number reaching every manifest, report and curve folded the saboteurs in. That
number moves with `f` even under a perfect defence, which is precisely what SPEC
Sec. 14 says the H6 headline must not do.

Fixed: `EvalResult.agent_return_means` + `subset_return()`; `train` records
`return_mean_honest` (and `return_mean_byzantine`), and `summary()` exposes
`final_return_honest`, **equal to `final_return` when nothing is Byzantine** so
one key can be read along the whole `f` axis. `final_return` keeps its old
meaning rather than being redefined under existing consumers.

**2. A resumed sweep's manifest carried no return for skipped cells.** Only
freshly-run cells got `final_return`, so any consumer reading the manifest
analysed *the cells that happened to run this time*. A resumed sweep is the
normal case. `_returns_from_disk` reads them back from `summary.json`.

**3. `h6_curve.py` parsed `f` out of arm names** with a regex matching only
`h6_byzantine.yaml`'s `<adversary>_<aggregator>_f<NN>` convention. Every other
H6 sweep fell through it and the script reported "no H6 arms found in that
manifest" — a naming convention used as a data model. It now reads `f`, level,
mode and aggregation from each run's recorded config, labels series by whatever
actually varies, and **warns loudly** when only the all-agent return is
available instead of silently mislabelling the axis.

Also: `analyze_hypotheses.py --require h3` died with `KeyError: 'h3'` because the
keys are uppercase. Case-insensitive, with a named error for unknown keys.

**Prediction (3) for the policy-level arm cannot be read off the existing runs**
— they predate the honest split. It needs `runs/h6_policy_level` re-run, which
also gets it the corrected metric everywhere else.

---

## 21. H6(c) prediction (3), and H3's verdict — 2026-08-07

### H6(c) prediction (3) — CONFIRMED, and it is the load-bearing one

Honest-agent final return, 10 paired seeds, paired against **each algorithm's
own `f = 0`**.

| `f` | D-TrOPD (gated) | paired Δ vs own `f=0` | NaiveDistill (`w=1`) | paired Δ vs own `f=0` |
|---|---|---|---|---|
| 0.0 | −36.51 | — | −35.80 | — |
| 0.2 | −35.97 | **+0.54 [−1.07, +2.08]** | **−196.96** | **−161.16 [−261.91, −61.58]** |
| 0.5 | −35.35 | **+1.16 [−0.48, +3.11]** | **−252.70** | **−216.91 [−301.59, −122.85]** |

**The ungated learner collapses by 5.5× and 7.1×. The gated one does not move.**
Both D-TrOPD CIs span zero; both NaiveDistill CIs are nowhere near it. At `f = 0`
the two algorithms are equal (−36.51 vs −35.80), so this is not a baseline
difference wearing an `f` axis.

**This also retires H6's open question, for this threat model.** The ungated
control collapsing proves the adversary lands, and the gated arm's flatness is
therefore a defence rather than an inert channel — exactly the discriminator
H6(b) was built to provide, here supplied by H6(c). The variance decomposition
points the right way for the first time: between-condition sd **99.14** against
within-cell seed sd **59.08**, where H6 had 0.58 against 6.81.

**What it does and does not establish, stated as plainly as before the run.**
NaiveDistill is the *no-defence ablation*, not a Byzantine-robust baseline: it
distils toward the lie with `w = 1`, so its collapse is expected. What the arm
shows is that the trust gate is **necessary and sufficient** against a
policy-level adversary in this setting — not that D-TrOPD beats a robust
aggregator. Since predictions (1) and (2) are near-arithmetic given the offset,
**(3) is the result that carries the claim**.

### H3 — 3 of 4 criteria confirmed; the miss is a false PREMISE, not a bad radius

`delta` is recovered per arm by inverting `delta = delta_base·exp(-kappa·signal)`.

| criterion | signal | ratio | expected | verdict |
|---|---|---|---|---|
| (1) scale invariance | nonstationarity | **1.099** | ~1.0 | **CONFIRMED** |
| (1) scale invariance | value variance | **0.0** | far from 1 | **CONFIRMED** (hard zero) |
| (2) contraction under instability | nonstationarity | **1.057** | < 1.0 | **not confirmed** — see below |
| (2) sign pathology | value variance | **3037×** | grows | **CONFIRMED** |

**(2) nonstationarity: the premise fails, not the mechanism.** H3 assumed a
degenerate critic *destabilises* the policy, giving larger `Delta_i` and hence a
smaller radius. Inverting the measured radii gives
`Delta_healthy = 4.52e-3` nats and `Delta_degenerate = 3.97e-3` — the frozen
critic made the policy **12% MORE stable**. A critic frozen at initialisation
produces near-uninformative advantages, so the policy gradient is weaker and the
policy moves *less* between epochs. The radius did exactly what its formula says;
the hypothesis's assumed causal chain from critic degeneracy to policy
instability does not hold in this environment. Recorded as a refuted premise, in
the same category as H1's "untestable at this horizon" — not as a failed radius.

**(2) value variance: the I1 pathology, at full strength.** The degenerate arm's
`Var(V)` is 8.11 against the healthy arm's 20.71, so the critic that explains
nothing earns a radius **3037× larger**. This is precisely what I1 forbids, and
it was unmeasurable before D18 because both arms were pinned at exactly zero.

### D18's `kappa` fix worked as specified, and exposed a further problem

The fix did what D18 claimed: the arm can now express its prediction (the 3037×
ratio above). But the healthy `valvar` radius is still numerically negligible at
`1.53e-7`, and inverting it says why — late-training `Var(V) = 20.71`, against a
calibration window whose maximum was 3.27. **`Var(V)` grew 6.3× beyond the window
`kappa` was derived on.**

This sharpens Remark 2 again (THEORY_NOTES §4). The value-variance radius does
not merely need re-tuning across *reward scales*; it needs re-tuning across
**training time**, because the magnitude of `Var(V)` is itself non-stationary. A
KL-valued signal has no such problem — it is dimensionless *and* its scale does
not drift with the value function's spread. One calibration cannot serve a
value-variance radius over a whole run, which is a stronger objection to it than
the reward-scaling one.

---

## 22. H7 approved and launched — 2026-08-07

### D20 — H7's reporting contract, fixed BEFORE any result

Recorded here rather than decided when the numbers land.

1. **The INTERACTION is the claim (D8).** Three quantities per grid point, each
   with a CI: the heterogeneous difference, the homogeneous difference, and
   `interaction = het − hom`. The predicted **sign flip** (`het > 0` and
   `hom < 0`) is the result. A point counts as confirming only if the
   interaction's CI excludes zero on the positive side **and** the sign flip is
   observed. An algorithm merely better everywhere satisfies the first and not
   the second, and does not support the paper's claim.
2. **The D7 sensitivity verdict is computed at all four grid points.** Identical
   at every point → the conclusion is stable and the window choice is
   immaterial. A flip → the benefit is calibration-sensitive and **that is the
   finding**, reported as a limitation. Never select the point with the nicer
   answer.
3. **≥10 seeds, paired, identical environment instances** for D-TrOPD vs IPPO.
4. **The D9 consensus-pull cost appears in the H7 table**, separately from
   return and never as a proxy for it — THEORY_NOTES §12 measured a 26% swing
   in honest consensus weight with zero detectable effect on return.
5. **H7 uses final return as its outcome variable**, and is the one hypothesis
   for which that is correct (`RETURN_IS_THE_OUTCOME = ("h7",)`).

`experiments/h7_report.py` implements all five and is the artifact the verdict
is read from. It was exercised on synthetic fixtures including a
verdict-flipping grid **before** any real point existed, so its behaviour on an
unfavourable result is not first observed on an unfavourable result.

### Three defects fixed before launch, all of which would have corrupted H7

**`D7_DELTA_BASE_GRID` was stale.** It read `(0.022, 0.026, 0.049)` — a
placeholder from before the grid was measured on the real H7 environment, never
updated when D17 signed off the four measured points. Nothing consumed it on a
real sweep, so it went stale invisibly, and `d7_sensitivity_verdict` would have
reported `complete: False` with every real point listed as missing. Replaced by
`D7_GRID_POINTS` (labels, since `delta_base` differs per environment and a bare
float can never name a point) plus `D7_DELTA_BASE`, with
`test_d7_grid_constant_matches_the_shipped_configs` asserting they match the
shipped YAML.

**The interaction had no confidence interval.** `h7_interaction` returned it as
a bare float, so D8's claim reduced to eyeballing two separate CIs — which is
not a test of their difference. It now returns `interaction_ci`, bootstrapped
over per-seed interactions. The environments differ, so the episodes are not
shared; what the pairing removes is the seed's shared effect on initialisation,
which both environments inherit.

**The pairing was positional.** Blocks were `{algo: [value per seed]}`, aligned
only by the manifest happening to be in plan order. `collect_final_returns(...,
by_seed=True)` returns `{algo: {seed: value}}` and the interaction pairs on the
key. A resumed or reordered sweep would otherwise have paired one seed's
treatment against another's control and reported it with a confident CI.

### The run

`./experiments/run_h7_grid.sh` — 4 points × 80 cells (8 arms × 10 seeds) = **320
cells**, ~27 CPU-hours, resumable per point and per cell. Arms: IPPO, D-TrOPD,
NaiveDistill, MAPPO, HAPPO on the heterogeneous environment; IPPO, D-TrOPD,
NaiveDistill on the homogeneous control.

**The verdict is reported whatever it is.** A refuted H7 at this level of rigour
is a publishable result, and the paper carries seven confirmed mechanism
findings that do not depend on it.

---

## 23. H7 — REFUTED — 2026-08-07

**320/320 cells, 4 grid points × 8 arms × 10 paired seeds. All clean
(`check_runs.py`: 480 run directories, no corruption).**

| point | het (D-TrOPD − IPPO) | hom (D-TrOPD − IPPO) | INTERACTION | verdict |
|---|---|---|---|---|
| w6  | −7.780 [−10.264, −5.365] | −6.607 [−9.554, −3.386] | −1.173 [−5.292, +2.860] | not confirmed |
| w8  | −8.708 [−11.429, −6.153] | −8.403 [−11.238, −5.492] | −0.305 [−4.425, +3.872] | not confirmed |
| w10 | −8.497 [−10.920, −6.233] | −7.414 [−10.272, −4.454] | −1.083 [−5.147, +3.000] | not confirmed |
| x2  | −8.541 [−11.089, −5.975] | −9.631 [−12.623, −6.459] | +1.090 [−2.890, +4.706] | not confirmed |

**D7 verdict: STABLE.** Identical at all four points, so the conclusion does not
depend on the calibration window and the refutation is not a calibration
artifact. Per D7 that is reported as stability, not as a limitation.

### What was found, stated as the numbers say it

1. **The predicted sign flip did not occur at any grid point.** It requires
   `het > 0`. Measured `het ≈ −8`, with a CI excluding zero at every point.
   D-TrOPD is not neutral under heterogeneity — it is **significantly worse than
   IPPO**, by about 8 return.
2. **The interaction is null.** |mean| ≤ 1.2 against a CI half-width of ~4, at
   every point. Distillation's cost is **uniform across the two environments**;
   heterogeneity confers no differential benefit.
3. **The D9 consensus-pull cost is large and it is not recovered.** `R(IPPO) −
   R(D-TrOPD)` on the homogeneous control runs +6.6 to +9.6 depending on the
   grid point, against NaiveDistill's +8.19. D9 framed this cost as the floor a
   heterogeneous benefit must clear. There is no heterogeneous benefit to clear
   it.

**H7 is REFUTED.** Not weakly, not pending more seeds: the effect has the wrong
sign, at every point of a pre-registered grid, with 10 paired seeds and CIs that
exclude zero.

### A scope statement, offered as a hypothesis and not as a defence

The H7 environment is a **complete graph on 12 agents**, i.e. degree 11. Finding
F-1, measured *before* H7 ran, is that the trust gate is **inert at high degree**
— `w_ij ≈ 0.96` against a calibrated radius, because neighbours on a complete
graph stay close and there is nothing for the gate to attenuate. H7's own logs
show it again throughout.

So H7 tested D-TrOPD in a regime where its distinguishing mechanism is
**inactive**, and the arm is behaviourally close to NaiveDistill. The measured
costs agree: D-TrOPD +8.40 vs NaiveDistill +8.19 at the calibrated point.

**This does not soften the verdict, and it must not be used to.** Three things
follow, in this order:

* **H7 as pre-registered is refuted, and that is what is reported.** The
  environment was fixed in D2/D8 long before the result.
* A lower-degree H7 would be a **new experiment**, not a re-analysis. If it is
  run, it needs its own pre-registered decision rule, and **this refutation is
  reported regardless of what it finds**. Running variants until one confirms,
  and publishing that one, is precisely the practice D7's rule exists to
  prevent — the rule applies to environments exactly as it applies to
  calibration windows.
* The scope statement rests on a measurement taken before the result, which is
  the only reason it is admissible at all.

### Fair comparison

MAPPO and HAPPO ran in every heterogeneous cell. **None of the five arms has
been tuned beyond the shared base config**, so the comparison is fair by
symmetry rather than by matched search — stated explicitly because an unfair
comparison is worse than no comparison. `h7_report.py` now prints every arm's
paired difference against IPPO in both environments, and per-arm IQM, so the
arm the hypothesis is about is not the only one reported.

### What survives

Seven confirmed mechanism findings do not depend on H7: Prop. 3, Prop. 4,
Remark 2's invariance half, the threat-model taxonomy, H2's KL-direction
separation, H3 (3 of 4 criteria), and the policy-level gate result — where the
gate removes 100% of a lying neighbour's weight and the ungated learner
collapses by 7× while the gated one does not move.

The honest summary of this project is that **the mechanisms work as specified
and do not, in this environment, produce a net return benefit.**

### The full arm ladder — the cost is DISTILLATION's, not the environment's

Heterogeneous environment, paired against IPPO, 10 seeds. IPPO, MAPPO and HAPPO
carry no `delta_base`, so their rows are identical at every grid point.

| arm | vs IPPO | IQM |
|---|---|---|
| IPPO | — | −26.21 |
| **HAPPO** | **+0.148 [−1.847, +2.002]** | −26.99 |
| MAPPO | −3.500 [−5.184, −1.713] | −31.11 |
| NaiveDistill | −8.154 [−10.529, −6.099] | −35.48 |
| D-TrOPD | −8.708 [−11.429, −6.153] | −36.53 |

**HAPPO ties IPPO.** That settles the question the −8 could not answer on its
own: the environment is not simply hard for everything, and the deficit is not
an artifact of the task, the horizon or the shared hyperparameters. A trust-region
baseline reaches IPPO's level on the same budget with the same untuned config.
The environment also discriminates — MAPPO's centralised critic is significantly
*worse* than independent learning under heterogeneity, which is its own small
observation and worth a line.

**The ~8-return deficit belongs to distillation.**

### The gate buys nothing over ungated distillation — measured, not inferred

D-TrOPD − NaiveDistill, paired by seed, at every grid point in both environments:

| point | heterogeneous | homogeneous |
|---|---|---|
| w6  | +0.374 [−1.371, +1.755] | +1.579 [−0.444, +3.635] |
| w8  | −0.553 [−1.852, +0.363] | −0.217 [−2.840, +2.322] |
| w10 | −0.343 [−2.037, +1.090] | +0.772 [−1.053, +2.560] |
| x2  | −0.387 [−2.118, +1.102] | −1.445 [−3.412, +0.365] |

**All eight span zero; every |mean| ≤ 1.6 against deficits of ~8.** The trust
gate — the entire difference between D-TrOPD and NaiveDistill — contributes
nothing to return here, in either environment, at any point of the grid.

This converts the earlier scope statement from a hypothesis into a measurement.
F-1 predicted gate inertness at degree 11 from `w_ij ≈ 0.96`; this is the same
conclusion read off *return*, in H7's own environment, with paired seeds. What
H7 refutes is **distillation**, and the gate is not what is costing the 8.

### The refutation is unanimous, not marginal

D-TrOPD is worse than IPPO on **10/10 seeds** in 7 of the 8 environment × grid
point cells, and 9/10 in the eighth. 79 of 80 paired seed comparisons. This is
not a CI that happens to exclude zero.

### The sentence the paper leads with

Not "D-TrOPD did not beat IPPO." The measured claim is:

> **Every mechanism is confirmed; the trust gate contributes nothing to return;
> and distillation itself costs ~8 return in this environment, uniformly across
> heterogeneity, while a trust-region baseline on the same budget ties IPPO.**

---

## 24. Result set FROZEN — 2026-08-07

### D21 — The locked H7 result statement

Recorded verbatim as the paper's lead claim. Not to be reworded by a later run.

> **Every mechanism is confirmed; the trust gate contributes nothing to return;
> and distillation itself costs ~8 uniformly across heterogeneity — while a
> trust-region baseline (HAPPO) on the same budget ties IPPO.**

Supported by three facts, each paired, 10 seeds, four pre-registered grid points:

**(a) HAPPO +0.148 [−1.847, +2.002] vs IPPO.** The environment supports
competitive performance, so the deficit is attributable to distillation and not
to the task, the horizon, or the shared untuned hyperparameters. **This control
is what makes the whole result interpretable** — without it, `−8` is
indistinguishable from "the environment is hard for everything".

**(b) D-TrOPD − NaiveDistill: all 8 contrasts span zero, |mean| ≤ 1.6** against
deficits of ~8. The trust gate is the entire difference between the two
algorithms and is worth nothing in return, in either environment, at every grid
point.

**(c) D-TrOPD loses to IPPO on 79/80 paired seed comparisons** — 10/10 seeds in
seven of the eight environment × grid-point cells, 9/10 in the eighth. The
refutation is unanimous, not marginal.

*Secondary observation, worth a line in the paper:* **MAPPO −3.500 [−5.184,
−1.713] vs IPPO.** A centralised critic is significantly **worse** than
independent learning under heterogeneity.

### D22 — The D7 sensitivity verdict, stated

Applying D7's rule, fixed before any result was seen:

> **The verdict is identical at all four grid points. Conclusion stable across
> the calibration grid; the window choice is immaterial.**

* `verdicts: {w6: false, w8: false, w10: false, x2: false}`, `complete: true`,
  `stable: true`, `report_as_limitation: false`.
* **The widening clause did not fire.** D7's alternative branch — "the verdict
  flips, therefore the benefit is calibration-sensitive, report as a limitation"
  — is not invoked, because nothing flipped.
* **No grid point was selected for its answer.** All four were run, all four are
  reported, and the rule was executed by `d7_sensitivity_verdict` rather than
  applied by hand.

This is the favourable direction of D7's rule and it costs the result nothing:
stability here means the refutation cannot be attributed to a calibration
choice.

### D23 — The ACTIVE-GATE experiment: PRE-REGISTERED, NOT RUN

Dated 2026-08-07, written **before** any degree sweep exists.

**Why it is well-posed now.** At degree 11 the gate is inactive by construction
(`w_ij ≈ 0.96`), so D-TrOPD ≈ NaiveDistill and the gate had no opportunity to
act — measured two independent ways (F-1's `w_ij`, and D21(b)'s return
contrast). Lower degree makes `w_ij` vary.

**Design.** Fixed-degree topology at fixed `N = 12`, degrees `{11, 8, 6, 4, 2}`.
Arms: D-TrOPD, NaiveDistill, IPPO. Paired seeds, ≥10. Calibration per D6, one
artifact per environment configuration.

**The degrees are not guessed.** They are chosen from the policy-level arm's
honest-edge data, where the gate excluding a fraction `f` of each neighbourhood
gives an effective degree — measured `(11, 0.957)`, `(8.8, 0.834)`,
`(5.5, 0.708)`, a near-linear `w ≈ 0.0447·deg + 0.456` (R² = 0.988). Extrapolated:

| degree | predicted mean `w_ij` |
|---|---|
| 11 | 0.95 |
| 8 | 0.81 |
| 6 | 0.72 |
| 4 | 0.63 |
| 2 | 0.55 |

Gating a neighbour to `w = 0` is not identical to its absence from the graph, so
these are a calibrated expectation and not a prediction of the sweep's own
`w_ij`. They are recorded so the grid cannot later be described as chosen for
its answer.

**Decision rule, pre-registered:**

* The **gate is vindicated** if `D-TrOPD − NaiveDistill` becomes reliably
  positive as degree falls (CI excluding zero at the low-degree points).
* **Distillation's cost is unaffected by gating** if the `D-TrOPD − IPPO` deficit
  persists at every degree.

**The ceiling, stated in advance.** Gating can only **reduce** how much
distillation occurs. The most optimistic realistic outcome is therefore
D-TrOPD converging toward IPPO **from below**, not exceeding it. **This
experiment can rescue the GATE's value; it cannot rescue distillation's ~8 cost,
which NaiveDistill pays too.** Writing that down now removes the temptation to
present a smaller deficit as a positive result.

**The H7 refutation is reported UNCHANGED regardless of this experiment's
outcome.**

*No config is shipped for it.* The parameters above are complete enough to
implement, and leaving a runnable file in the tree would make it one command to
extend a frozen result set. Writing the config is the first step of running it,
and that step needs a decision, not a `git pull`.

### D24 — The result set is FROZEN

The experimental programme is closed as of tag `results-frozen-2026-08-07`.

* **No arm is added to the reported result set without a dated
  pre-registration**, written before the run, in the manner of D17, D19 and D23.
* **Nothing already reported is revised by a later run.** A later experiment may
  add scope; it may not retroactively change a recorded verdict.
* **Next work is writing, not running.**

**The reported set, complete:**

| result | verdict |
|---|---|
| H1 chattering | untestable at this horizon (edges never cross the boundary) |
| H2 KL direction | **CONFIRMED** — clean separation |
| H3 radius | **3 of 4 criteria confirmed**; the miss is a refuted premise (§11) |
| H4 / H5 scale | measured; D15's asymmetric cap documented |
| H6 gradient-level | correctly scoped to the aggregator (D19); gate blind by construction, 0.3% |
| H6(c) policy-level | **CONFIRMED on all three predictions** — `w_ij = 0`, 100% weight removed, ungated learner collapses 7× |
| H7 headline | **REFUTED**, stable across the D7 grid, unanimous on 79/80 seeds |
| Prop. 3 | confirmed |
| Prop. 4 | **empirically validated**, ~100× margin |
| Remark 2 | restated; one premise refuted, conclusion on stronger ground (§4) |
