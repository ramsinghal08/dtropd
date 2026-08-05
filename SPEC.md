

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
- **H7 (headline).** D-TrOPD > IPPO in sample efficiency and final return under heterogeneity, and ≥ NaiveDistill everywhere. *Measure:* learning curves, final returns, **10 seeds**, 95% bootstrap CIs, IQM, plus per-zone returns so gains aren't an artifact of one sector.

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
