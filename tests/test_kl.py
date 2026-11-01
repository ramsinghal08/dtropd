"""Acceptance: analytic Gaussian KL vs a 1e6-sample Monte Carlo estimate.

SPEC Sec. 2.5 / Sec. 5 : "analytic Gaussian KL matches a 1e6-sample
Monte Carlo estimate within 1%".

The tanh-invariance test is the part that licenses computing the KL on the
pre-squash latents at all: the Jacobian of the shared squashing map cancels,
so the MC estimate taken with *squashed* log-densities must equal the analytic
latent KL to the same tolerance.
"""

from __future__ import annotations

import math

import pytest
import torch

from dtropd.config import NetworkConfig
from dtropd.kl import gaussian_entropy, gaussian_kl, mean_gaussian_kl
from dtropd.networks import TanhGaussianActor

N_SAMPLES = 1_000_000
TOL = 0.01  # 1%


def _mc_kl(mu_p, log_std_p, mu_q, log_std_q, n=N_SAMPLES, seed=0):
    """E_{x~p}[log p(x) - log q(x)] on the *latent* (unsquashed) variable."""
    g = torch.Generator().manual_seed(seed)
    std_p, std_q = log_std_p.exp(), log_std_q.exp()
    x = mu_p + std_p * torch.randn((n, mu_p.shape[-1]), generator=g, dtype=torch.float64)
    log_p = (-0.5 * ((x - mu_p) / std_p) ** 2 - log_std_p - 0.5 * math.log(2 * math.pi)).sum(-1)
    log_q = (-0.5 * ((x - mu_q) / std_q) ** 2 - log_std_q - 0.5 * math.log(2 * math.pi)).sum(-1)
    return (log_p - log_q).mean(), x


@pytest.mark.parametrize(
    "mu_p, ls_p, mu_q, ls_q",
    [
        ([0.0, 0.0], [0.0, 0.0], [1.0, -0.5], [0.4054651, 0.2]),
        ([0.3, -0.7], [-0.3, 0.1], [-0.2, 0.4], [0.1, -0.2]),
        ([1.5, 1.5], [-0.5, -0.5], [0.0, 0.0], [0.0, 0.0]),
    ],
)
def test_analytic_kl_matches_monte_carlo(mu_p, ls_p, mu_q, ls_q):
    """1e6-sample MC estimate agrees with the closed form to within 1%."""
    t = lambda v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
    mu_p, ls_p, mu_q, ls_q = t(mu_p), t(ls_p), t(mu_q), t(ls_q)

    analytic = gaussian_kl(mu_p, ls_p, mu_q, ls_q)
    mc, _ = _mc_kl(mu_p, ls_p, mu_q, ls_q)

    assert analytic > 0.05, "test case too close to zero KL to be informative"
    rel = abs(float(analytic) - float(mc)) / float(analytic)
    assert rel < TOL, f"analytic={float(analytic):.6f} mc={float(mc):.6f} rel={rel:.4%}"


def test_kl_invariant_under_tanh_squash():
    """KL of the tanh-squashed policies equals the latent KL (SPEC Sec. 2.5)."""
    t = lambda v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
    mu_p, ls_p = t([0.4, -0.2]), t([-0.1, 0.2])
    mu_q, ls_q = t([-0.3, 0.5]), t([0.2, -0.1])

    analytic = float(gaussian_kl(mu_p, ls_p, mu_q, ls_q))

    # MC over the squashed density: log p(a) = log p(z) - log|det J(z)|, and the
    # Jacobian term is shared, so it must cancel exactly.
    g = torch.Generator().manual_seed(7)
    z = mu_p + ls_p.exp() * torch.randn((N_SAMPLES, 2), generator=g, dtype=torch.float64)
    log_det = (2.0 * (math.log(2.0) - z - torch.nn.functional.softplus(-2.0 * z))).sum(-1)

    def squashed_logp(mu, ls):
        lp = (-0.5 * ((z - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
        return lp - log_det

    mc = float((squashed_logp(mu_p, ls_p) - squashed_logp(mu_q, ls_q)).mean())
    rel = abs(analytic - mc) / analytic
    assert rel < TOL, f"analytic={analytic:.6f} squashed-mc={mc:.6f} rel={rel:.4%}"


def test_kl_matches_torch_distributions():
    torch.manual_seed(0)
    mu_p, ls_p = torch.randn(16, 3), torch.randn(16, 3) * 0.3
    mu_q, ls_q = torch.randn(16, 3), torch.randn(16, 3) * 0.3
    p = torch.distributions.Normal(mu_p, ls_p.exp())
    q = torch.distributions.Normal(mu_q, ls_q.exp())
    reference = torch.distributions.kl_divergence(p, q).sum(-1)
    assert torch.allclose(gaussian_kl(mu_p, ls_p, mu_q, ls_q), reference, atol=1e-6)


def test_kl_zero_for_identical_distributions():
    torch.manual_seed(1)
    mu, ls = torch.randn(32, 2), torch.randn(32, 2) * 0.2
    kl = gaussian_kl(mu, ls, mu.clone(), ls.clone())
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)  # float32 epsilon
    assert float(mean_gaussian_kl(mu, ls, mu, ls)) == pytest.approx(0.0, abs=1e-6)


def test_kl_is_non_negative_and_asymmetric():
    torch.manual_seed(2)
    mu_p, ls_p = torch.randn(256, 2), torch.randn(256, 2) * 0.4
    mu_q, ls_q = torch.randn(256, 2), torch.randn(256, 2) * 0.4
    forward = gaussian_kl(mu_p, ls_p, mu_q, ls_q)
    reverse = gaussian_kl(mu_q, ls_q, mu_p, ls_p)
    assert torch.all(forward >= -1e-9)
    assert torch.all(reverse >= -1e-9)
    assert not torch.allclose(forward, reverse)


def test_kl_shape_mismatch_raises():
    with pytest.raises(ValueError):
        gaussian_kl(
            torch.zeros(4, 2), torch.zeros(4, 2), torch.zeros(4, 3), torch.zeros(4, 3)
        )


def test_gaussian_entropy_matches_torch():
    torch.manual_seed(3)
    ls = torch.randn(8, 3) * 0.5
    reference = torch.distributions.Normal(torch.zeros_like(ls), ls.exp()).entropy().sum(-1)
    assert torch.allclose(gaussian_entropy(ls), reference, atol=1e-6)


def test_actor_log_prob_matches_torch_transformed_distribution():
    """The tanh log-density matches torch's ``TransformedDistribution`` exactly."""
    cfg = NetworkConfig(hidden_sizes=[8], log_std_init=-0.3)
    actor = TanhGaussianActor(obs_dim=4, act_dim=2, cfg=cfg).double()
    obs = torch.randn(64, 4, dtype=torch.float64)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        out = actor.sample(obs, generator=g)
        base = torch.distributions.Normal(out["mu"], out["log_std"].exp())
        squashed = torch.distributions.TransformedDistribution(
            base, [torch.distributions.transforms.TanhTransform(cache_size=1)]
        )
        reference = squashed.log_prob(out["action"]).sum(-1)
    assert torch.allclose(out["log_prob"], reference, atol=1e-6)


def test_tanh_log_det_is_numerically_stable():
    """The stable rewrite agrees with the naive form and survives saturation."""
    z = torch.tensor([-30.0, -3.0, 0.0, 2.0, 25.0], dtype=torch.float64).reshape(-1, 1)
    stable = TanhGaussianActor._tanh_log_det(z)
    naive = torch.log(1.0 - torch.tanh(z) ** 2 + 1e-300).sum(-1)
    assert torch.all(torch.isfinite(stable))
    assert torch.allclose(stable[1:4], naive[1:4], atol=1e-9)
    # The naive form underflows to -inf at |z| = 30; the stable one does not.
    assert float(stable[0]) == pytest.approx(-2 * 30.0 + 2 * math.log(2.0), abs=1e-6)
