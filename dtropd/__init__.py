"""D-TrOPD: Decentralized Trust-Region Policy Distillation.

See ``docs/SPEC.md`` for the authoritative algorithm specification.  Module map:

``config``      typed dataclass configs + validation (incl. kappa*eta in [1, 3])
``env``         cooperative-capture MPE env, heterogeneity zones, stressors
``networks``    tanh-Gaussian actor, local critic, ``policy_parameters()``
``buffer``      rollout storage and GAE
``kl``          closed-form Gaussian KL -- the single source of truth
``algorithms``  shared base class + IPPO (D-TrOPD and baselines land later)
``robust``      trimmed mean, geometric median, Krum
``train``       training loop, logging, checkpointing, seeding
``evaluate``    evaluation rollouts and metrics
``plot``        figures regenerated from stored metrics
``run``         CLI entrypoint
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, ConfigError, EnvConfig, load_config  # noqa: F401

__all__ = ["Config", "ConfigError", "EnvConfig", "load_config", "__version__"]
