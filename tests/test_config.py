"""Config dataclasses, validation and YAML round-tripping."""

from __future__ import annotations

import pytest
import yaml

from dtropd.config import (
    Config,
    ConfigError,
    DistillConfig,
    EnvConfig,
    GraphConfig,
    load_config,
)


def test_default_config_validates():
    cfg = Config()
    warnings_out = cfg.validate(emit=False)
    assert warnings_out == [], warnings_out


def test_kappa_eta_product_must_lie_in_range():
    """SPEC Sec. 2.2: choose kappa such that kappa * eta lies in [1, 3]."""
    cfg = Config()
    cfg.ppo.kl_early_stop = 0.02
    cfg.distill.kappa = 100.0  # product = 2.0
    assert cfg.validate(emit=False) == []

    cfg.distill.kappa = 10.0  # product = 0.2 -> too small
    with pytest.raises(ConfigError, match=r"kappa \* eta"):
        cfg.validate(emit=False)

    cfg.distill.kappa = 1000.0  # product = 20 -> too large
    with pytest.raises(ConfigError, match=r"kappa \* eta"):
        cfg.validate(emit=False)


def test_kappa_eta_can_be_downgraded_to_a_warning():
    cfg = Config()
    cfg.distill.kappa = 10.0
    warnings_out = cfg.validate(strict=False, emit=False)
    assert any("kappa * eta" in w for w in warnings_out)


def test_fixed_radius_ablation_is_exempt_and_requires_kappa_zero():
    d = DistillConfig(radius_signal="fixed", kappa=0.0)
    assert d.validate(eta=0.02) == []
    with pytest.raises(ConfigError):
        DistillConfig(radius_signal="fixed", kappa=5.0).validate(eta=0.02)
    with pytest.raises(ConfigError, match="radius_signal='fixed'"):
        DistillConfig(kappa=0.0).validate(eta=0.02)


def test_kl_direction_is_locked_to_reverse():
    assert DistillConfig().kl_direction == "reverse"
    with pytest.raises(ConfigError, match="I2"):
        DistillConfig(kl_direction="forward").validate(eta=0.02)


def test_env_dimensions():
    cfg = EnvConfig(n_agents=12, n_targets=3)
    assert cfg.obs_dim == 49
    assert cfg.act_dim == 2
    assert EnvConfig(n_agents=4, n_targets=2).obs_dim == 4 + 6 + 9 + 2 + 1


def test_env_validation_errors():
    with pytest.raises(ConfigError):
        EnvConfig(n_agents=2, capture_group_size=5).validate()
    with pytest.raises(ConfigError):
        EnvConfig(sensor_range=0.5, masked_sensor_range=1.5).validate()
    with pytest.raises(ConfigError):
        EnvConfig(friction_velocity_scale=2.0).validate()


def test_graph_validation():
    GraphConfig(topology="complete").validate(12)
    GraphConfig(topology="fixed_degree", degree=4).validate(12)
    with pytest.raises(ConfigError):
        GraphConfig(topology="nope").validate(12)
    with pytest.raises(ConfigError):
        GraphConfig(topology="fixed_degree", degree=12).validate(12)


def test_yaml_roundtrip(tmp_path):
    cfg = Config()
    cfg.env.n_agents = 6
    cfg.env.capture_group_size = 3
    cfg.train.total_epochs = 7
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    loaded = Config.from_yaml(path)
    assert loaded.env.n_agents == 6
    assert loaded.train.total_epochs == 7
    assert loaded.to_dict() == cfg.to_dict()
    assert isinstance(loaded.env.friction_zones[0].radius, float)


def test_unknown_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"env": {"n_agents": 3, "typo_key": 1}}))
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_yaml(path)


def test_dotted_overrides():
    cfg = load_config(None, {"train.total_epochs": "5", "env.n_agents": "4",
                             "env.capture_group_size": "2", "ppo.lr": "1e-3"})
    assert cfg.train.total_epochs == 5
    assert cfg.env.n_agents == 4
    assert cfg.ppo.lr == pytest.approx(1e-3)
    with pytest.raises(ConfigError):
        load_config(None, {"train.nonexistent": "1"})


def test_beta_warmup_defaults_to_ten_percent():
    d = DistillConfig()
    assert d.warmup_fraction == 0.1
    assert d.resolved_warmup_epochs(200) == 20


def test_homogeneous_copy_disables_zones():
    cfg = EnvConfig(heterogeneous=True)
    homo = cfg.homogeneous_copy()
    assert cfg.heterogeneous and not homo.heterogeneous
    assert homo.friction_zones == cfg.friction_zones  # kept, simply inactive


def test_shipped_experiment_configs_validate():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "experiments" / "configs"
    files = sorted(root.glob("*.yaml"))
    assert files, "no experiment configs shipped"
    for path in files:
        cfg = Config.from_yaml(path)
        cfg.validate(emit=False)
