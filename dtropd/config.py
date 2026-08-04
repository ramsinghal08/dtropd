"""Dataclass configuration objects, validation and YAML loading.

Every knob in the project lives here.  There are no dictionaries of magic
strings anywhere else in the codebase; algorithms receive typed config objects.

Validation encodes the parts of SPEC.md Sec. 2 that are checkable statically:

* ``kappa * eta in [1, 3]``               (SPEC Sec. 2.2)
* KL direction is always ``"reverse"``    (SPEC Sec. 2.3 / invariant I2)
* trust-test batch defaults to own rollout (invariant I3)
* radius signal defaults to nonstationarity (invariant I1)
* beta warmup is active                   (invariant I8)
"""

from __future__ import annotations

import dataclasses
import typing
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Zone",
    "EnvConfig",
    "NetworkConfig",
    "PPOConfig",
    "DistillConfig",
    "GraphConfig",
    "TrainConfig",
    "Config",
    "ConfigError",
    "load_config",
]


class ConfigError(ValueError):
    """Raised when a configuration violates a hard specification constraint."""


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


@dataclass
class Zone:
    """A circular heterogeneity zone in world coordinates."""

    x: float
    y: float
    radius: float

    def contains(self, pos_x: float, pos_y: float) -> bool:
        return (pos_x - self.x) ** 2 + (pos_y - self.y) ** 2 <= self.radius**2


@dataclass
class EnvConfig:
    """Cooperative-capture MPE environment (SPEC Sec. 5, Phase 1).

    12 agents, 3 targets, a target counts as captured only when at least
    ``capture_group_size`` agents are simultaneously inside ``capture_radius``.
    """

    n_agents: int = 12
    n_targets: int = 3
    capture_group_size: int = 3
    capture_radius: float = 0.3
    max_cycles: int = 50

    # --- reward accounting -------------------------------------------------
    shaping_weight: float = 1.0
    capture_bonus: float = 1.0
    collision_penalty: float = 0.1
    # Blend between the per-agent local term and the shared global term, as in
    # PettingZoo MPE's `local_ratio`. 0.0 = purely global team reward.
    local_ratio: float = 0.5

    # --- observation -------------------------------------------------------
    sensor_range: float = 1.5
    masked_sensor_range: float = 0.5

    # --- heterogeneity zones (SPEC Sec. 5, Phase 1) ------------------------
    heterogeneous: bool = False
    friction_velocity_scale: float = 0.5
    friction_zones: list[Zone] = field(
        default_factory=lambda: [Zone(x=-0.6, y=0.0, radius=0.7)]
    )
    mask_zones: list[Zone] = field(
        default_factory=lambda: [Zone(x=0.6, y=0.0, radius=0.7)]
    )

    # --- physics (mirrors mpe2._mpe_utils.core.World defaults) -------------
    dt: float = 0.1
    damping: float = 0.25
    contact_force: float = 1e2
    contact_margin: float = 1e-3
    agent_size: float = 0.05
    target_size: float = 0.05
    accel: float = 5.0
    max_speed: float | None = None
    spawn_range: float = 1.0

    def validate(self) -> list[str]:
        warnings_out: list[str] = []
        if self.n_agents < 1:
            raise ConfigError("env.n_agents must be >= 1")
        if self.n_targets < 1:
            raise ConfigError("env.n_targets must be >= 1")
        if self.capture_group_size < 1:
            raise ConfigError("env.capture_group_size must be >= 1")
        if self.capture_group_size > self.n_agents:
            raise ConfigError(
                "env.capture_group_size cannot exceed env.n_agents "
                f"({self.capture_group_size} > {self.n_agents})"
            )
        if self.n_agents < self.capture_group_size * self.n_targets:
            warnings_out.append(
                "fewer agents than capture_group_size * n_targets -- not all "
                "targets can be captured simultaneously."
            )
        if self.masked_sensor_range > self.sensor_range:
            raise ConfigError(
                "env.masked_sensor_range must be <= env.sensor_range "
                "(masked zones shrink the sensor range, they do not grow it)"
            )
        if not 0.0 <= self.friction_velocity_scale <= 1.0:
            raise ConfigError("env.friction_velocity_scale must lie in [0, 1]")
        if not 0.0 <= self.local_ratio <= 1.0:
            raise ConfigError("env.local_ratio must lie in [0, 1]")
        if self.local_ratio == 1.0:
            warnings_out.append(
                "local_ratio=1.0 removes the shared team term entirely; the "
                "capture bonus then has no effect on any agent's reward."
            )
        if self.max_cycles < 1:
            raise ConfigError("env.max_cycles must be >= 1")
        return warnings_out

    @property
    def obs_dim(self) -> int:
        """Observation width.

        self vel (2) + self pos (2) + targets (2 rel + 1 visible each)
        + other agents (2 rel + 1 visible each) + zone flags (2)
        + remaining-horizon fraction (1).

        The horizon feature is not decoration: episodes are fixed length, so
        the return from step ``t`` depends strongly on how many steps remain.
        Without it the critic cannot predict its own target and explained
        variance is capped near zero regardless of how well it is trained
        (invariant I4 is about the critic *learning*, which needs the value
        function to be learnable in the first place).
        """
        return 4 + 3 * self.n_targets + 3 * (self.n_agents - 1) + 2 + 1

    @property
    def act_dim(self) -> int:
        return 2

    def homogeneous_copy(self) -> "EnvConfig":
        """Return the same environment with all heterogeneity zones disabled."""
        return dataclasses.replace(self, heterogeneous=False)


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #


@dataclass
class NetworkConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"
    log_std_init: float = -0.5
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    state_dependent_std: bool = False
    orthogonal_init: bool = True

    def validate(self) -> list[str]:
        if self.activation not in ("tanh", "relu", "elu"):
            raise ConfigError(f"network.activation {self.activation!r} not supported")
        if not self.hidden_sizes:
            raise ConfigError("network.hidden_sizes must be non-empty")
        if self.log_std_min >= self.log_std_max:
            raise ConfigError("network.log_std_min must be < network.log_std_max")
        return []


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #


@dataclass
class PPOConfig:
    """Local PPO (SPEC Sec. 2.4 ``L_CLIP`` + ``L_V`` + KL early stop)."""

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5  # c_V in SPEC Sec. 2.4
    max_grad_norm: float = 0.5
    inner_epochs: int = 4
    num_minibatches: int = 4
    kl_early_stop: float = 0.02  # eta -- the *improvement* trust radius
    normalize_advantages: bool = True
    normalize_returns: bool = True  # critic targets only; values are de-normalised
    clip_value_loss: bool = False
    value_clip_ratio: float = 0.2

    def validate(self) -> list[str]:
        warnings_out: list[str] = []
        if self.kl_early_stop <= 0.0:
            raise ConfigError(
                "ppo.kl_early_stop (eta) must be > 0: the improvement trust region "
                "is mandatory (invariant I5)"
            )
        if self.value_coef <= 0.0:
            raise ConfigError(
                "ppo.value_coef (c_V) must be > 0: the critic loss must be present "
                "and optimized (invariant I4)"
            )
        if not 0.0 < self.clip_ratio < 1.0:
            raise ConfigError("ppo.clip_ratio must lie in (0, 1)")
        if not 0.0 <= self.gamma <= 1.0:
            raise ConfigError("ppo.gamma must lie in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ConfigError("ppo.gae_lambda must lie in [0, 1]")
        if self.inner_epochs < 1:
            raise ConfigError("ppo.inner_epochs must be >= 1")
        if self.num_minibatches < 1:
            raise ConfigError("ppo.num_minibatches must be >= 1")
        return warnings_out


# --------------------------------------------------------------------------- #
# Distillation (D-TrOPD -- Phase 3; declared here so configs validate today)
# --------------------------------------------------------------------------- #


@dataclass
class DistillConfig:
    """Social distillation knobs (SPEC Sec. 2.2, 2.3, 2.6).

    Consumed by the D-TrOPD algorithm in Phase 3.  The fields exist now so that
    the kappa*eta validation demanded by Phase 0 has something to validate and
    so that experiment configs are stable across phases.
    """

    enabled: bool = False
    beta_max: float = 0.05
    warmup_fraction: float = 0.1  # SPEC Sec. 2.4: default 10% of training
    warmup_epochs: int | None = None  # explicit override
    kappa: float = 100.0  # radius contraction rate; kappa * eta in [1, 3]
    delta_base: float = 0.1
    gamma_attenuation: float = 1.0  # gamma in w_ij = exp(-gamma * [...])
    attenuation: str = "continuous"  # continuous | hard_switch | hard_cutoff
    kl_direction: str = "reverse"  # invariant I2: reverse everywhere, always
    radius_signal: str = "nonstationarity"  # nonstationarity | value_variance | fixed
    trust_batch: str = "own"  # invariant I3: own | pooled
    aggregation: str = "sum"  # sum | trimmed_mean | geometric_median | krum
    neighbor_selection: str = "all"  # SPEC Sec. 2.3: all neighbours by default
    top_k: int | None = None
    trimmed_mean_fraction: float = 0.2
    krum_byzantine: int = 1

    _ATTENUATIONS = ("continuous", "hard_switch", "hard_cutoff")
    _RADIUS_SIGNALS = ("nonstationarity", "value_variance", "fixed")
    _TRUST_BATCHES = ("own", "pooled")
    _AGGREGATIONS = ("sum", "trimmed_mean", "geometric_median", "krum")
    _NEIGHBOR_SELECTIONS = ("all", "top_k")

    def resolved_warmup_epochs(self, total_epochs: int) -> int:
        if self.warmup_epochs is not None:
            return int(self.warmup_epochs)
        return max(1, int(round(self.warmup_fraction * total_epochs)))

    def beta(self, epoch: int, total_epochs: int) -> float:
        """Linear beta warmup from 0 to ``beta_max`` (SPEC Sec. 2.4, invariant I8)."""
        warmup = self.resolved_warmup_epochs(total_epochs)
        if warmup <= 0:
            return self.beta_max
        return self.beta_max * min(1.0, epoch / warmup)

    def validate(self, eta: float, *, strict: bool = True) -> list[str]:
        warnings_out: list[str] = []

        if self.kl_direction != "reverse":
            raise ConfigError(
                "distill.kl_direction must be 'reverse'. SPEC Sec. 2.3 / invariant "
                "I2: there is exactly one KL direction in this codebase and no "
                "direction switch. The forward/switched ablations of Sec. 6 are "
                "implemented as separate, explicitly-named ablation algorithms, "
                "never as a branch inside the D-TrOPD update."
            )
        if self.attenuation not in self._ATTENUATIONS:
            raise ConfigError(
                f"distill.attenuation must be one of {self._ATTENUATIONS}"
            )
        if self.radius_signal not in self._RADIUS_SIGNALS:
            raise ConfigError(
                f"distill.radius_signal must be one of {self._RADIUS_SIGNALS}"
            )
        if self.trust_batch not in self._TRUST_BATCHES:
            raise ConfigError(f"distill.trust_batch must be one of {self._TRUST_BATCHES}")
        if self.aggregation not in self._AGGREGATIONS:
            raise ConfigError(f"distill.aggregation must be one of {self._AGGREGATIONS}")
        if self.neighbor_selection not in self._NEIGHBOR_SELECTIONS:
            raise ConfigError(
                f"distill.neighbor_selection must be one of {self._NEIGHBOR_SELECTIONS}"
            )
        if self.neighbor_selection == "top_k" and not self.top_k:
            raise ConfigError("distill.top_k must be set when neighbor_selection='top_k'")
        if self.beta_max < 0.0:
            raise ConfigError("distill.beta_max must be >= 0")
        if self.delta_base <= 0.0:
            raise ConfigError("distill.delta_base must be > 0")
        if self.gamma_attenuation < 0.0:
            raise ConfigError("distill.gamma_attenuation must be >= 0")
        if not 0.0 <= self.warmup_fraction <= 1.0:
            raise ConfigError("distill.warmup_fraction must lie in [0, 1]")
        if self.warmup_epochs is not None and self.warmup_epochs < 0:
            raise ConfigError("distill.warmup_epochs must be >= 0")

        # Invariant-flavoured warnings: these knobs exist only to *measure* the
        # pathologies of Sec. 6, so using them is loud but legal.
        if self.radius_signal == "value_variance":
            warnings_out.append(
                "distill.radius_signal='value_variance' contradicts invariant I1. "
                "This is only valid as the H3 ablation arm; results from it must "
                "never be reported as D-TrOPD."
            )
        if self.trust_batch == "pooled":
            warnings_out.append(
                "distill.trust_batch='pooled' contradicts invariant I3. Own-rollout "
                "is authoritative; this arm exists only for the Sec. 6 ablation."
            )
        if self.attenuation != "continuous":
            warnings_out.append(
                f"distill.attenuation={self.attenuation!r} is an H1 ablation arm, "
                "not the D-TrOPD default."
            )
        if self.neighbor_selection == "top_k":
            warnings_out.append(
                "distill.neighbor_selection='top_k' reintroduces a discontinuity at "
                "the selection boundary (SPEC Sec. 2.3); the default is 'all'."
            )
        if self.enabled and self.beta_max > 0.0 and self.warmup_fraction == 0.0 and (
            self.warmup_epochs in (None, 0)
        ):
            warnings_out.append(
                "beta warmup disabled — invariant I8 requires an active warmup "
                "schedule outside of deliberate ablations."
            )

        warnings_out.extend(self._validate_kappa_eta(eta, strict=strict))
        return warnings_out

    def _validate_kappa_eta(self, eta: float, *, strict: bool) -> list[str]:
        """SPEC Sec. 2.2: choose kappa such that ``kappa * eta in [1, 3]``.

        The ``fixed`` radius ablation pins ``kappa = 0`` on purpose, so it is
        exempt; every other configuration must satisfy the range.
        """
        if self.radius_signal == "fixed":
            if self.kappa != 0.0:
                raise ConfigError(
                    "distill.radius_signal='fixed' requires distill.kappa == 0"
                )
            return []
        if self.kappa == 0.0:
            raise ConfigError(
                "distill.kappa == 0 collapses the trust radius to a constant. "
                "Use radius_signal='fixed' to request that explicitly."
            )
        product = self.kappa * eta
        if 1.0 <= product <= 3.0:
            return []
        message = (
            f"kappa * eta = {product:.4g} is outside [1, 3] "
            f"(kappa={self.kappa:.4g}, eta={eta:.4g}). SPEC Sec. 2.2 requires this "
            "range so the trust radius has real dynamic range."
        )
        if strict:
            raise ConfigError(message)
        return [message]


# --------------------------------------------------------------------------- #
# Communication graph
# --------------------------------------------------------------------------- #


@dataclass
class GraphConfig:
    """Undirected communication graph G over the agents (SPEC Sec. 2)."""

    topology: str = "complete"  # complete | ring | fixed_degree | erdos_renyi
    degree: int = 4  # fixed_degree / erdos_renyi expected degree
    seed: int | None = None  # None -> derived from the master seed

    _TOPOLOGIES = ("complete", "ring", "fixed_degree", "erdos_renyi")

    def validate(self, n_agents: int) -> list[str]:
        if self.topology not in self._TOPOLOGIES:
            raise ConfigError(f"graph.topology must be one of {self._TOPOLOGIES}")
        if self.topology in ("fixed_degree", "erdos_renyi"):
            if self.degree < 1:
                raise ConfigError("graph.degree must be >= 1")
            if self.degree >= n_agents:
                raise ConfigError(
                    f"graph.degree ({self.degree}) must be < n_agents ({n_agents})"
                )
            if self.topology == "fixed_degree" and (self.degree * n_agents) % 2 != 0:
                raise ConfigError(
                    "fixed_degree topology needs degree * n_agents to be even"
                )
        return []


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    algo: str = "ippo"
    total_epochs: int = 200
    num_envs: int = 20
    steps_per_env: int | None = None  # None -> one full episode per env per epoch
    seed: int = 0
    device: str = "cpu"
    torch_threads: int = 1
    log_dir: str = "runs"
    run_name: str | None = None
    eval_interval: int = 10
    eval_episodes: int = 8
    eval_deterministic: bool = True
    checkpoint_interval: int = 0  # 0 -> only at the end
    log_interval: int = 1

    def validate(self) -> list[str]:
        if self.total_epochs < 1:
            raise ConfigError("train.total_epochs must be >= 1")
        if self.num_envs < 1:
            raise ConfigError("train.num_envs must be >= 1")
        if self.steps_per_env is not None and self.steps_per_env < 1:
            raise ConfigError("train.steps_per_env must be >= 1 when set")
        if self.device not in ("cpu", "cuda") and not self.device.startswith("cuda:"):
            raise ConfigError("train.device must be 'cpu' or 'cuda[:i]'")
        return []


# --------------------------------------------------------------------------- #
# Root config
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    notes: str = ""

    # ---- validation ------------------------------------------------------ #

    def validate(self, *, strict: bool = True, emit: bool = True) -> list[str]:
        """Validate the whole config.

        Hard violations raise :class:`ConfigError`.  Soft ones are returned as
        warning strings and (by default) also emitted through :mod:`warnings`.

        ``strict`` controls one specific case that SPEC Sec. 2.2 describes as
        "assert ... and warn otherwise": ``kappa * eta`` outside ``[1, 3]``
        raises when strict and warns when not.
        """
        msgs: list[str] = []
        msgs += [f"env: {m}" for m in self.env.validate()]
        msgs += [f"network: {m}" for m in self.network.validate()]
        msgs += [f"ppo: {m}" for m in self.ppo.validate()]
        msgs += [
            f"distill: {m}"
            for m in self.distill.validate(self.ppo.kl_early_stop, strict=strict)
        ]
        msgs += [f"graph: {m}" for m in self.graph.validate(self.env.n_agents)]
        msgs += [f"train: {m}" for m in self.train.validate()]

        steps = self.train.steps_per_env or self.env.max_cycles
        batch = steps * self.train.num_envs
        if batch % self.ppo.num_minibatches != 0:
            msgs.append(
                f"ppo: batch size {batch} is not divisible by num_minibatches "
                f"{self.ppo.num_minibatches}; the last minibatch will be smaller."
            )
        if emit:
            for m in msgs:
                warnings.warn(m, stacklevel=2)
        return msgs

    # ---- (de)serialisation ------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return _build(cls, data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: top-level YAML document must be a mapping")
        return cls.from_dict(raw)

    # ---- derived ---------------------------------------------------------- #

    @property
    def steps_per_env(self) -> int:
        return self.train.steps_per_env or self.env.max_cycles

    @property
    def batch_size(self) -> int:
        """Transitions collected per agent per epoch."""
        return self.steps_per_env * self.train.num_envs


# --------------------------------------------------------------------------- #
# Recursive dataclass builder (typed, no magic-string dicts leak out)
# --------------------------------------------------------------------------- #


def _build(cls: type, data: Any) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping to build {cls.__name__}, got {type(data)}")
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(
            f"unknown key(s) for {cls.__name__}: {sorted(unknown)}; "
            f"valid keys are {sorted(known)}"
        )
    hints = typing.get_type_hints(cls)
    for name, value in data.items():
        kwargs[name] = _coerce(hints.get(name, Any), value, f"{cls.__name__}.{name}")
    return cls(**kwargs)


def _coerce(hint: Any, value: Any, path: str) -> Any:
    if value is None:
        return None
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    if origin is typing.Union or str(origin) == "types.UnionType":
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return _coerce(non_none[0], value, path)
        return value
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value)}")
        inner = args[0] if args else Any
        return [_coerce(inner, v, f"{path}[]") for v in value]
    if is_dataclass(hint) and isinstance(hint, type):
        return _build(hint, value)
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    strict: bool = True,
    validate: bool = True,
) -> Config:
    """Load a config from YAML (or defaults) and apply dotted-key overrides."""
    cfg = Config.from_yaml(path) if path is not None else Config()
    for key, value in (overrides or {}).items():
        _apply_override(cfg, key, value)
    if validate:
        cfg.validate(strict=strict)
    return cfg


def _apply_override(cfg: Config, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    target: Any = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ConfigError(f"unknown config section {part!r} in override {dotted!r}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise ConfigError(f"unknown config key {dotted!r}")
    current = getattr(target, leaf)
    if isinstance(current, bool) and isinstance(value, str):
        value = value.lower() in ("1", "true", "yes", "on")
    elif isinstance(current, int) and not isinstance(current, bool) and isinstance(value, str):
        value = int(value)
    elif isinstance(current, float) and isinstance(value, (str, int)):
        value = float(value)
    setattr(target, leaf, value)
