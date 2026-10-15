"""Tests for the bridge-matching regression target (``xi_form`` = c3 / eq24)."""

import pytest
import torch

from bms.experiment.train import bridge_matching_target
from bms.process.sde import VESDE
from bms.process.source import GaussianSource


@pytest.mark.parametrize("scale_1", [1.0, 2.0])
def test_xi_forms_conditional_mean(scale_1):
    """Prop. C.3 and eq. (24) (both with c = gamma) have the same conditional mean
    given X_t when X_1 follows the target (fixed point). Otherwise they differ by
    (1 - gamma) E[grad log(p_target / p_1)(X_1) | X_t], computed here in closed form
    for Gaussians on the 1D mean-free subspace of 2 particles in 1D."""
    torch.manual_seed(0)
    n, t = 200_000, 0.05
    scale_0, scale_target = 2.0, 1.0
    sde = VESDE(sigma_min=0.01, sigma_max=1.5)
    data_0 = GaussianSource(num_atoms=2, spatial_dim=1, scale=scale_0).sample(n)
    data_1 = GaussianSource(num_atoms=2, spatial_dim=1, scale=scale_1).sample(n)
    time = torch.full((n,), t)
    data_t = sde.sample_posterior(time, data_0, data_1)

    # E[y | X_t] is linear with zero intercept (all variables are centered Gaussians).
    x = data_t[:, 0, 0]
    slopes = {}
    for form in ("c3", "eq24"):
        y = bridge_matching_target(
            form,
            sde,
            time=time,
            data_0=data_0,
            data_1=data_1,
            data_t=data_t,
            prior_score=-data_0 / scale_0**2,
            target_score=-data_1 / scale_target**2,
        )[:, 0, 0]
        assert torch.isfinite(y).all()
        slopes[form] = ((x * y).mean() / (x * x).mean()).item()

    kappa_1 = sde.total_variance.item()
    gamma = (sde.cond_var_t0(time[:1], data_0[:1]).item()) / kappa_1
    var_t = (
        (1 - gamma) ** 2 * scale_0**2
        + gamma**2 * scale_1**2
        + kappa_1 * gamma * (1 - gamma)
    )
    regression_1_on_t = gamma * scale_1**2 / var_t  # E[X_1 | X_t] = this * X_t
    expected = (1 - gamma) * (1 / scale_1**2 - 1 / scale_target**2) * regression_1_on_t
    assert slopes["eq24"] - slopes["c3"] == pytest.approx(expected, abs=0.02)
    if scale_1 != scale_target:
        assert abs(expected) > 0.2  # the check above is discriminative


def test_xi_form_unknown():
    x = torch.zeros(1, 2, 1)
    with pytest.raises(ValueError):
        bridge_matching_target("eq25", VESDE(), torch.ones(1), x, x, x, x, x)
