# D-TrOPD — Decentralized Trust-Region Policy Distillation

Implementation of D-TrOPD for networked multi-agent RL, plus baselines and a
pre-registered experiment suite. `SPEC.md` is the authoritative specification;
this README describes only what is built and how to run it.

**Build status: Phases 0–6 complete. Phase 7 (robustness + scale) not started.**

**No hypothesis has been evaluated.** H7 and every multi-seed headline
experiment are gated at Phase 7; the harness is built and exercised on a smoke
sweep only.

SPEC.md §10–§12 record the **locked decisions** (D1–D13) that govern how these
phases are configured and reported. They have the same standing as the
invariants in §3.

| Phase | Scope | State |
|---|---|---|
| 0 | Skeleton: repo structure, configs, seeding, logging, CI-runnable pytest | ✅ |
| 1 | Env + networks + closed-form KL | ✅ |
| 2 | IPPO | ✅ |
| 3 | D-TrOPD core (radius, attenuation, social loss, β warmup) | ✅ |
| 4 | Baselines (MAPPO, HAPPO, NaiveDistill) | ✅ |
| 5 | Instrumentation | ✅ |
| 6 | Experiment harness (rliable figures, paired seeds) | ✅ |
| 7 | Robustness + scale | ⬜ |

Asking for an unbuilt algorithm fails loudly and names the phase it belongs to,
rather than silently substituting something else.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. Everything runs CPU-only; `train.device=cuda` moves the networks
to a single GPU. No multi-GPU or cluster assumptions anywhere.

Note on PettingZoo: the MPE scenarios were split out of `pettingzoo` at 1.25 and
now live in the `mpe2` package. `mpe2` is used as the *reference implementation*
that our vectorised physics is tested against (`tests/test_env.py::
test_physics_matches_mpe2_world`), not as a runtime dependency of training.

## Quick start

`run.py` is subcommand-driven — every invocation takes one of
`train`, `eval`, `plot`, `validate`, `calibrate`, `sweep-check`, `analyze`,
`smoke`:

```bash
python run.py --help
python run.py validate --config experiments/configs/ippo_homogeneous.yaml
python run.py smoke                       # 2-agent IPPO CPU debug path, < 2 minutes
python run.py smoke --algo dtropd         # ...and the D-TrOPD debug path
python run.py train --config experiments/configs/ippo_homogeneous.yaml --seeds 3
python run.py train --config experiments/configs/dtropd_heterogeneous.yaml --seeds 3
python run.py eval  --config experiments/configs/dtropd_heterogeneous.yaml \
    --checkpoint runs/dtropd_heterogeneous/dtropd_seed0/checkpoint_final.pt
python run.py plot --runs runs/phase2/seed0 runs/phase2/seed1 --out figures/curve.png
python run.py analyze --run runs/phase5/dtropd_seed0     # every H1–H7 measurement
python run.py sweep-check --configs experiments/configs/phase5_arm_*.yaml
python run.py sweep --config experiments/configs/sweep_smoke.yaml --seeds 2
python run.py calibration-grid --config <env>.yaml --windows 6 8 10
pytest                                    # full suite, ~40 s on CPU
```

`train` and `eval` also accept `--set KEY=VALUE` overrides and `--no-strict`
(which downgrades the `κ·η ∈ [1,3]` check from an error to a warning).

Any config value can be overridden from the CLI:

```bash
python run.py train --config experiments/configs/smoke.yaml \
    --set train.total_epochs=20 --set ppo.lr=1e-3
```

## Repository layout

```
dtropd/
  config.py      dataclass configs, validation (incl. κ·η ∈ [1,3]), YAML load
  env.py         cooperative-capture MPE env + heterogeneity zones + stressors
  networks.py    tanh-Gaussian actor, critic, policy_parameters() helper
  buffer.py      rollout storage, GAE (with time-limit bootstrapping)
  kl.py          closed-form Gaussian KL utilities (single source of truth)
  algorithms.py  base class, IPPO, MAPPO, HAPPO, NaiveDistill, D-TrOPD
  robust.py      trimmed mean, geometric median, Krum
  train.py       training loop, logging, checkpointing, seeding
  evaluate.py    eval rollouts, metrics
  plot.py        figures regenerated from stored metrics
  run.py         CLI entrypoint
  utils.py       seeding, JSONL metrics writer, run metadata, explained variance
tests/           test_kl, test_env, test_invariants, test_attenuation, test_smoke, …
experiments/
  configs/*.yaml
  phase2_acceptance.py  phase3_acceptance.py  phase4_acceptance.py
  _ippo_reference_curve.py   # regenerates the Phase 2 golden curve
tests/data/ippo_phase2_reference.json   # golden curve, made on commit 011de8f
notebooks/D_TrOPD.ipynb
```

## Algorithms

| name | critic | update order | distillation |
|---|---|---|---|
| `ippo` | local `V_i(o_i)` | simultaneous | none |
| `mappo` | centralised, over concatenated observations | simultaneous | none |
| `happo` | local | **sequential, random order each epoch**, with the HAPPO importance factor | none |
| `naive_distill` | local | simultaneous | every neighbour, `w_ij = 1` **structurally** |
| `dtropd` | local (SPEC §2: no central critic) | simultaneous | trust-gated, `w_ij` from `D_ij` vs `δ_i` |

`naive_distill` is its own class, not a D-TrOPD config (SPEC §10 / D4): it has
no radius and no attenuation code path at all, so "NaiveDistill" and
"miscalibrated D-TrOPD" can never be the same run.

## Calibrating the attenuation (SPEC §10 / D1)

`δ_base` and `γ` are the only D-TrOPD constants SPEC does not pin down, and a
mis-set `δ_base` silently disables the whole mechanism. They are therefore
**measured, not chosen**:

```bash
# offline: emit a calibrated config, with its provenance recorded in `notes`
python run.py calibrate --config experiments/configs/dtropd_heterogeneous.yaml \
    --epochs 10 --out experiments/configs/dtropd_heterogeneous_static.yaml
```

```yaml
# in-run: the same procedure, executed during training
distill:
  calibration: warmup      # beta is held at exactly 0 for the window
```

Both paths call the same `calibrate_attenuation()`, so they cannot drift apart.
`δ_base` is set to the median observed `D_ij` and `γ` so that a 95th-percentile
edge is attenuated to `w = 0.5`. Headline configs set `calibration: warmup`;
defaults stay uncalibrated so no existing config silently changes. D-TrOPD warns
loudly whenever mean `w_ij > 0.95` with β > 0.

## Instrumentation (Phase 5)

The acceptance bar is that **one** run's metrics file supports **every** H1–H7
measurement without re-running. That is demonstrated literally:

```bash
python experiments/phase5_acceptance.py          # runs one job, then measures H1–H7 from it
python run.py analyze --run runs/phase5/dtropd_seed0
```

| hypothesis | what the file carries |
|---|---|
| H1 | per-edge social-gradient **sign-flip counts bucketed by `\|D_ij − δ_i\|`**, plus gradient cosines |
| H2 | per-edge `σ_i` (student) and `σ_j` (teacher) alongside `D_ij`, so σ trajectories on high-`D_ij` edges are recoverable |
| H3 | `δ_i`, `Δ_i`, the radius signal and reference in use, critic EV |
| H4 | per-agent **surrogate gain** vs the realised change in return, per round |
| H5 | the same per capita, **plus the Assumption-2 decay probe** — perturb agent *j*, record induced \|ΔQ^i\| against hop distance |
| H6 | per-run aggregation, Byzantine fraction and final return (the sweep cell) |
| H7 | per-environment returns, per-zone returns, calibration id, consensus-pull share and dispersion |

Everything distributional is logged as **quantiles, not means** (SPEC §11): the
`δ_base` failure was invisible in the mean and obvious in the `D_ij` quantiles.
The metrics schema is versioned (`metrics_schema_version`) and stamped into
`metadata.json`; a schema change after H7 starts invalidates stored curves
exactly as a `reward_spec_version` bump does.

`experiments/analyze_hypotheses.py` exits non-zero if any hypothesis is not
computable — "we'll add that logging later" is the failure Phase 5 exists to
prevent.

## Calibration is pinned across a comparison (SPEC §11 / D6)

The rule is fixed and pre-registered: a hard **8-epoch β = 0 window**, `δ_base`
= the median `D_ij` **pooled over that window's final 2 epochs**, `γ` set so a
p95 edge attenuates to `w = 0.5`. **The number 8 is arbitrary** — SPEC says so
in those words. It is fixed so it is not a researcher degree of freedom, and D7
measures how much it matters.

The measurement is an **artifact**, not two numbers in a YAML:

```bash
python run.py calibrate --config <cfg> --epochs 8 --out experiments/calibrations/<env>.json
python run.py sweep-check --configs experiments/configs/phase5_arm_*.yaml
```

The artifact records the window, the epochs pooled, the edge count, the full
`D_ij` quantile table, `reward_spec_version` and the git commit, and is
tamper-evident (its id is a hash of its contents). Every arm of a comparison
pins the same file — **including arms with no radius of their own**: NaiveDistill
declares which calibration its comparison used without acquiring a `δ_base`
(D4 × D6). `sweep-check` refuses a sweep whose arms were calibrated separately.

## The experiment harness (Phase 6)

One command, from scratch, to the main comparison figure:

```bash
python run.py sweep --config experiments/configs/h7.yaml --seeds 10
```

It derives the calibration, runs every arm × seed, writes the manifest and the
report, and renders the figure — no manual steps (SPEC §7). `--report-only`
rebuilds the report and figure **from stored metrics alone**, and `--no-resume`
forces a re-run.

**Resumable.** A cell is complete iff its `summary.json` exists *and* the
config fingerprint recorded in its `metadata.json` matches what the sweep asked
for. Kill a sweep and restart it: finished cells are skipped, the rest run, and
a *changed* config is never silently skipped. There is no partial-epoch resume —
a half-written metrics file is worse than a re-run.

**Three guards run before a single epoch**, because each is a way a comparison
becomes quietly meaningless:

| guard | refuses |
|---|---|
| D6 | arms in one environment that were calibrated **separately**. One artifact per *environment configuration*, pinned by every arm in it — a two-environment H7 sweep legitimately carries two |
| D13 | an **algorithm-performance** comparison spanning reward spec versions. Environment-property arms may span, and are labelled `cross_version` |
| paired seeds | arms overriding `train.seed`, or arms in one condition with different environment configs — either breaks the identical-env-instance guarantee |

The report carries the **D8 interaction** (per-environment difference plus the
predicted sign flip) and the **D9 consensus-pull cost**, both with bootstrap
CIs, and per-arm IQM — rliable's headline estimators, computed in
`dtropd.analysis` so the core package needs no extra dependency.

![Phase 6 harness demonstration](figures/phase6_harness_demo.png)

*Smoke sweep only (4 agents, 4 epochs, 2 seeds). This is a picture of the
machinery working, **not a result**.*

## The environment

`dtropd.env.CooperativeCaptureEnv` is a PettingZoo `ParallelEnv`.

* **12 agents, 3 targets.** A target counts as captured on a timestep only when
  **≥ 3 agents are simultaneously within `capture_radius`** of it.
* **Reward**, per timestep, blended by `local_ratio` exactly as PettingZoo
  MPE blends its local and global terms:

  ```
  global   = shaping_weight · Σ_k ( −mean distance of the 3 nearest agents to target k )
           + capture_bonus  · #{captured targets}
           − collision_penalty · #{colliding agent pairs}
  local_i  = − shaping_weight · (distance from agent i to its nearest target)
           − collision_penalty · #{collisions involving agent i}

  r_i = (1 − local_ratio) · global + local_ratio · local_i
  ```

  `local_ratio = 0` gives a pure shared team reward (identical for every
  agent); the default `0.5` matches MPE's default. The capture bonus is always
  global — that is the coordination signal. See "Judgment calls" below.

* **Physics** are exactly PettingZoo MPE's `World.step` (semi-implicit
  integration, damping 0.25, soft-margin contact forces, `dt = 0.1`,
  `sensitivity = 5`), re-implemented in vectorised numpy and checked against a
  live `mpe2` `World` to 1e-9 in the test suite.
* **Actions** are `Box(-1, 1, (2,))` — an (x, y) force command, the natural
  codomain of a tanh-squashed policy.
* **Observations** (49-dim for 12 agents / 3 targets): own velocity and
  position, per-target relative position + visibility flag, per-other-agent
  relative position + visibility flag, two zone flags, and the remaining
  fraction of the episode. Entities beyond the observer's sensor range are
  zeroed and flagged invisible.
* **Heterogeneity zones** (`heterogeneous: true`): a high-friction zone scales
  an agent's velocity to 50% while it is inside, and a masked-observation zone
  shrinks its sensor range from 1.5 to 0.5.
* **Episodes are fixed length** (`max_cycles`): they truncate, never terminate,
  so values are bootstrapped at the horizon.

## Judgment calls

SPEC.md is silent on three points that had to be decided to make Phase 2's
acceptance test meaningful. Each is a config knob, each defaults to the
learnable setting, and each can be reverted in one line.

1. **`env.local_ratio` (default 0.5).** With a purely global team reward, one
   agent out of twelve moves the shared return by ~1/9, and IPPO does not learn
   within any budget we could run: over 60 epochs it stayed at the
   random-policy level. MPE's own `simple_spread` blends a local and a global
   term for exactly this reason, so we blend the same way. `local_ratio: 0.0`
   restores the pure team reward.
2. **Remaining-horizon feature in the observation.** Episodes are fixed length,
   so the return from step *t* depends strongly on how many steps remain.
   Without that feature in the observation the critic cannot predict its own
   target and explained variance is capped near zero no matter how well it is
   trained — which would make invariant I4's "explained variance rising" test
   unpassable for reasons that have nothing to do with the critic.
3. **`ppo.normalize_returns` (default true).** The critic learns
   `(G − μ)/σ` with a running per-agent normaliser and every value it reports
   is de-normalised back into reward units, so advantages, GAE, explained
   variance and (later) the radius signal all see unmodified values. This one
   change moved explained variance from ~0.14 to ~0.87 on the same run.

Points 2 and 3 are optimisation fixes with no algorithmic content. Point 1
changes the task, so it is called out here rather than buried in a default.
It is now settled by SPEC §10 / D2: `local_ratio = 0.5` is the primary
condition, and `experiments/configs/secondary_pure_global.yaml` carries the
pure-global variant as an explicitly-labelled secondary condition.

### 4. δ_base and γ are measured, not chosen — and the window length matters

*(Resolved by SPEC §10 / D1; the finding that prompted it is kept below because
it is the reason the decision exists.)*

New caveat surfaced while implementing the procedure: **the calibrated value
depends on how long the calibration window is.** `D_ij` drifts upward during
training, so measuring over 8 epochs gave `δ_base = 0.026` and over 10 epochs
`δ_base = 0.049` on the same config and seed. That is D5's confound showing up
inside D1's procedure. The window length is therefore a reported parameter of
the method, recorded in the emitted config's `notes`, not an implementation
detail. Both the fixed and `running_quantile` arms log the per-epoch `D_ij`
percentiles and the drift ratio so this is always visible.

### 4a. The original finding

SPEC.md fixes the form of the attenuation but not its two constants. The
shipped defaults (`delta_base: 0.1`, `gamma_attenuation: 1.0`) turn out to sit
far above the social divergences this environment actually produces:

```
median D_ij over 1056 edges = 0.026     mean w_ij = 0.987
```

At `w ≈ 0.99` on every edge the trust gate is doing nothing, so a run at these
defaults **is NaiveDistill wearing D-TrOPD's name** — with no error and a
perfectly plausible learning curve. Two things now prevent that from passing
silently:

* `experiments/calibrate_delta.py` measures the `D_ij` distribution and
  suggests values (`δ_base` = median observed `D_ij`, `γ` such that a
  95th-percentile edge is attenuated to `w = 0.5`);
* `DTrOPD` logs a loud warning whenever mean `w_ij > 0.95` with β > 0.

`experiments/configs/dtropd_heterogeneous_calibrated.yaml` carries the measured
values (`δ_base = 0.026`, `γ = 26`), under which the gate is genuinely active
(mean `w` falls 0.998 → 0.61, min 0.46 over 40 epochs). The defaults are left
**unchanged** so nothing silently shifts under an existing config — which of the
two is used for the H7 headline is a science decision, not a build decision.

## Reproducibility

Every run writes, next to its metrics:

* `metadata.json` — full config, git commit, seed, platform, package versions;
* `metrics.jsonl` — one JSON record per epoch, including per-agent detail;
* `summary.json`, `checkpoint_final.pt`.

All randomness is derived from a single master seed through
`utils.derive_seed(master, *tags)` (BLAKE2b, stable across processes), so a run
is reproducible from `(config, seed)` alone. Minibatch shuffling is driven by a
per-agent, per-epoch generator, so it does not depend on how much randomness
the rest of the update consumed — which is what makes the Phase 3 acceptance
test (β = 0 must reproduce IPPO bitwise) a test of the social path rather than
of RNG bookkeeping.

## Invariants

SPEC Sec. 3 lists nine non-negotiable invariants. All nine bind as of Phase 3
and `tests/test_invariants.py` encodes each as an executable check against the
live update — not against config defaults. A config field that nothing reads is
not a guarantee, so every `DistillConfig` switch is driven to a value whose
effect is observable and the test asserts the effect
(`tests/test_dtropd.py::test_*_is_honoured`).

* **I1** — radius from policy nonstationarity: δ = δ_base at t = 0 and
  contracts as Δ rises; Δ is shown invariant to a 10× reward rescale while the
  `value_variance` arm scales by 100× and hands a *degenerate* critic the
  maximal radius — the sign pathology I1 exists to forbid.
* **I2** — one KL direction, no branch: `kl.gaussian_kl` has no `direction`
  argument, a source scan rejects any forward-KL token in the package, the
  config refuses `kl_direction != "reverse"`, `DTrOPD.__init__` raises on it,
  and `L_soc` is checked numerically to be KL(student ‖ teacher).
* **I3** — own-rollout batches: corrupting another agent's buffer slice
  provably leaves this agent's update *and* its `D_ij` bit-identical, while the
  `pooled` arm demonstrably differs.
* **I4** — critic present and optimised: `value_coef = 0` is a config error,
  critic parameters are asserted to move, explained variance is logged.
* **I5** — both regions: clipping plus KL early stopping (a tiny η halts the
  inner loop), *and* a divergent peer's `w_ij` is shown to collapse while `w`
  stays bit-constant through the inner loop.
* **I6** — teacher snapshots are byte-identical before and after an epoch's
  update, and a backward through `L_soc` leaves every teacher's `.grad` `None`.
* **I7** — `w(0) = 1` to 1e-12 for every (δ, γ) tested, including end-to-end:
  an exact policy twin gets `w = 1`.
* **I8** — β warmup is checked *inside* the training loop, not just in config.
* **I9** — `policy_parameters()` excludes the value head; `torch.autograd.grad`
  over the full parameter list is shown to raise, over the helper to succeed.
  The robust-aggregation path depends on this and is exercised.

## Acceptance results

| Phase | Test | Result |
|---|---|---|
| 0 | `pytest` green; `python run.py --help` works | **pass** — 168 passed, 0 skipped |
| 1 | analytic Gaussian KL vs 1e6-sample MC within 1%; env random-policy rollout shapes + reward accounting | **pass** — max relative error 0.10% over three parameter pairs |
| 2 | IPPO on the homogeneous env, 3 seeds, beats random by a clear margin, critic EV > 0.3 | **pass** — see below |

Phase 2 numbers (150 epochs, 12 agents, 3 seeds, ~5 min/seed on one CPU core):

| seed | final return | random return | improvement | margin (random SEMs) | max critic EV | captures/ep (vs random) |
|---|---|---|---|---|---|---|
| 0 | −23.06 | −60.58 | **+37.52** | 17.5 | **0.943** | 35.7 (2.9) |
| 1 | −7.59 | −63.24 | **+55.65** | 21.8 | **0.941** | 59.9 (3.3) |
| 2 | −24.01 | −62.23 | **+38.22** | 10.9 | **0.947** | 34.6 (2.4) |

Mean final return −18.22 vs −62.02 random; mean improvement +43.80.

![Phase 2 return](figures/phase2_return.png)
![Phase 2 explained variance](figures/phase2_ev.png)

### Phase 3 — β = 0 reproduces IPPO bitwise

```bash
python experiments/phase3_acceptance.py --seeds 0 1 2 --epochs 15
```

| seed | tensors compared | mismatched | max abs param diff | curves differing | social path ran |
|---|---|---|---|---|---|
| 0 | 78 | 0 | **0.0** | none | yes (mean D_ij 0.320, mean w 0.848) |
| 1 | 78 | 0 | **0.0** | none | yes (mean D_ij 0.190, mean w 0.909) |
| 2 | 78 | 0 | **0.0** | none | yes (mean D_ij 0.247, mean w 0.882) |

"Social path ran" matters: β = 0 would also be trivially satisfied by a
D-TrOPD that never computed anything. The acceptance script requires that
snapshots were exchanged and that non-degenerate `D_ij` / `w_ij` were produced
in the same run whose parameters it then compares.

`tests/test_invariants.py` (I1–I9) and `tests/test_attenuation.py` (w(0)=1,
monotone, C¹ at D = δ over a 5×4 grid of (δ, γ)) are both green.

Regenerate Phase 2:

```bash
python experiments/phase2_acceptance.py --seeds 0 1 2 --workers 3 --out runs/phase2
```

The "clear margin" criterion is operationalised in that script (every seed
must beat the random policy by ≥ 10 standard errors of the random baseline,
every seed must improve, and every seed's critic explained variance must exceed
0.3), and it prints the numbers it judged.

### Judgment calls made while implementing D1–D5 and Phase 4

5. **The calibration window holds β at exactly 0, and shifts the warmup ramp.**
   D1's rationale requires the `D_ij` measurement to be taken with the gate
   inactive. A linear β warmup is *not* zero after its first epoch, so reusing
   the warmup window as the calibration window would have measured `D_ij` under
   a partly-active gate — quietly violating the stated rationale. Instead the
   calibration window pins β = 0 and the full-length warmup ramp starts after
   it, so invariant I8 still gets a complete ramp. The alternatives were to
   calibrate on epoch 0 alone (a much smaller sample) or to accept the
   contamination; both seemed worse than shifting the ramp.

6. **HAPPO keeps local critics, and includes the importance factor.** SPEC
   describes HAPPO only as "sequential updates, random agent order re-sampled
   each epoch". Two gaps had to be filled. First, the critic: HAPPO is
   implemented with IPPO's *local* critics rather than a centralised one, so
   that the sequential-update variable is isolated and MAPPO remains the arm
   that varies critic centralisation. Second, the correction factor: a
   sequential update without HAPPO's `M_i = Π_{j before i} π_j^new/π_j^old` has
   no monotonic-improvement argument at all, which is the only interesting
   property of updating sequentially — so it is implemented. The factor is
   clamped to `[1e-4, 1e4]`, since a product over 12 agents can otherwise
   overflow; the clamp is logged.

7. **MAPPO concatenates in fixed agent-index order**, giving every agent the
   *same* joint critic input at a transition. This keeps MAPPO's advantage over
   IPPO purely a matter of critic information rather than asymmetric inputs.

8. **Phase 4 uses one shared config file, not five.** `train.algo` (and the
   structural `distill.enabled`) are the only per-algorithm overrides, and the
   acceptance harness enumerates every field that differs and fails if anything
   else does. "Shared hyperparameters" is then a property of the code rather
   than of five YAML files staying in sync by inspection.

9. **`run.py calibrate` runs the real training loop** rather than a separate
   measurement path, so the offline command and `calibration: warmup` execute
   the same code and cannot disagree about what the procedure is.

**No hyperparameter search was performed for any algorithm.** Every algorithm in
Phase 4 uses the Phase 2 IPPO settings unchanged (SPEC §9 rule 5). If tuning is
done later for D-TrOPD, the same search must be run for MAPPO and HAPPO and
documented.

### Judgment calls made while implementing D6–D10 and Phase 5

10. **D6 × D4: NaiveDistill declares a calibration it does not consume.** D6
    requires the calibration to be "pinned for every arm in that comparison
    (D-TrOPD, NaiveDistill, and any ablation sharing the env)", while D4
    requires NaiveDistill to have no `δ_base` at all. Resolved by separating
    *declaration* from *consumption*: every algorithm — including IPPO — loads
    and stamps the artifact for provenance and for `sweep-check`, but only
    D-TrOPD reads `delta_base` out of it. `test_naive_distill_declares_a_
    calibration_without_owning_a_radius` holds both halves.

11. **The calibration window is its own knob, not the β warmup.** D6 fixes an
    8-epoch β = 0 prefix; the previous implementation derived the window from
    `warmup_fraction`. These are now separate (`distill.calibration_epochs`), so
    changing the training length no longer silently changes the calibration
    rule.

12. **Sign flips are measured by gradient cosine, not by a scalar sign.** A
    per-edge social gradient is a vector, so "sign flip" needs a definition. A
    flip is recorded when the cosine between an edge's gradient this epoch and
    last epoch is negative. The alternative — tracking the sign of some scalar
    projection — would have depended on an arbitrary choice of projection.

13. **The per-edge gradient pass runs *outside* the inner loop**, on a fixed
    256-sample subsample, after the agent's update. Instrumentation that can
    perturb the update it measures is worse than no instrumentation, and this
    also keeps the β = 0 bitwise-IPPO property intact.

14. **`Q^i` in the decay probe is the observer's realised episode return** under
    the perturbed joint policy, measured against an unperturbed baseline on
    identical environment seeds. A true `Q^i` is not available; the proxy and
    its pairing are stated in `dtropd/probes.py` rather than left implicit.

15. **D7 and D8's decision rules are code, not prose.**
    `analysis.d7_sensitivity_verdict()` applies the pre-registered rule to the
    three `δ_base` arms, and `analysis.h7_interaction()` computes the predicted
    sign flip. Written before any H7 run, so neither can be quietly
    reinterpreted once numbers exist.

16. **The decay probe always fires on the final epoch**, in addition to its
    interval, so every run's metrics file carries at least one — otherwise H5's
    measurability would depend on whether `total_epochs` happened to be a
    multiple of the interval.

### Judgment calls made while implementing D7-amended, D11–D13 and Phase 6

17. **D6 × D8 collided, and D6's own wording resolved it.** The calibration
    check originally demanded one artifact for the whole sweep, which made D6
    and D8 mutually unsatisfiable: H7 spans two environments, and D6 says the
    calibration is measured "once per **environment configuration**". Arms are
    now grouped by environment fingerprint and must share an artifact *within*
    each group. A two-environment sweep carries two artifacts, which is correct
    rather than a loophole.

19. **`run.py calibrate --out` now emits the artifact, not a config YAML.** In
    Phase 5 it wrote a calibrated config; but `distill.calibration_file` points
    at a `CalibrationRecord`, so the command that produces the thing arms pin
    must produce *that*. Caught by the D6 guard refusing a sweep whose pinned
    artifacts did not exist.

20. **The sweep derives missing calibrations once, then pins them
    (`auto_calibrate`).** Without this the Phase 6 acceptance could not be one
    command, since D6 requires artifacts to pre-exist. Deriving once per
    environment at sweep start and pinning for every arm *is* the D6 workflow —
    what D6 forbids is arms calibrating separately. Set `auto_calibrate: false`
    to require pre-existing artifacts.

21. **`v1-local`'s neighbourhood cannot capture if it is smaller than
    `capture_group_size`.** With the team term summed over `N_i ∪ {i}`, a
    neighbourhood of 2 can never satisfy a 3-agent capture. That is the honest
    consequence of making the signal local, not a bug — but it means H5's
    locality condition needs a topology with degree ≥ `capture_group_size`, and
    `h5_locality.yaml` uses a ring over 16 agents accordingly.

22. **The env now receives the communication graph** (`env.team_neighborhood`)
    for `v1-local`. It is passed in rather than rediscovered so the environment
    and the algorithm cannot disagree about the topology — a disagreement would
    silently break exactly the locality property the condition exists to
    provide.

23. **Phase 6's acceptance was demonstrated on the smoke sweep, not on H7.**
    SPEC's literal acceptance command is `run.py sweep --config h7.yaml --seeds
    10`; the code path is identical and `h7.yaml` is shipped and validated by
    the test suite, but running it would be the peek Phase 7 gates. This is a
    deliberate deviation from the letter of the acceptance in favour of its
    intent.

24. **Two YAML schemas now live in `experiments/configs/`** — single-run
    configs and sweep configs. `test_shipped_experiment_configs_validate` routes
    by the presence of an `arms` key rather than by filename, so a new file
    cannot dodge validation by being named unexpectedly.
