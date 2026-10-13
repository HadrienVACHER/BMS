"""Tests for the n-body systems (DW-4, LJ-13, LJ-55): potentials, EGNN, prior,
terminal cost, and evaluator. Run with ``pytest tests/``."""

from pathlib import Path

import numpy as np
import pytest
import torch

from bms.evaluation.nbody import NBodyEvaluator
from bms.model.egnn import EGNNDynamics
from bms.potential.nbody import DoubleWellPotential, LennardJonesPotential
from bms.process.sde import VESDE
from bms.process.source import GaussianSource
from bms.process.terminal_cost import BMSTerminalCost
from bms.utils.geometry import is_mean_free

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "nbody"

SYSTEMS = [
    (DoubleWellPotential, 4, 2),
    (LennardJonesPotential, 13, 3),
    (LennardJonesPotential, 55, 3),
]


def random_rotation(dim: int) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, dtype=torch.double))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.float()


def lattice(n: int, dim: int, batch: int = 8) -> torch.Tensor:
    """Well-separated configurations (avoids the LJ singularity at d = 0)."""
    side = int(np.ceil(n ** (1 / dim)))
    grid = torch.stack(
        torch.meshgrid(*[torch.arange(side)] * dim, indexing="ij"), -1
    ).reshape(-1, dim)[:n]
    pos = 1.1 * grid.float() + 0.05 * torch.randn(batch, n, dim)
    return pos - pos.mean(dim=1, keepdim=True)


@pytest.mark.parametrize("cls,n,dim", SYSTEMS)
def test_potential_invariances_and_forces(cls, n, dim):
    torch.manual_seed(0)
    potential = cls(n_particles=n, spatial_dim=dim)
    pos = lattice(n, dim)
    energy = potential.energy(pos)

    rot = random_rotation(dim)
    perm = torch.randperm(n)
    shift = torch.randn(1, 1, dim)
    torch.testing.assert_close(potential.energy(pos @ rot.T), energy, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(potential.energy(pos[:, perm]), energy, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(potential.energy(pos + shift), energy, rtol=1e-4, atol=1e-3)

    out = potential(pos)
    torch.testing.assert_close(out["energy"], energy)
    pos_req = pos.clone().requires_grad_(True)
    grad = torch.autograd.grad(potential.energy(pos_req).sum(), pos_req)[0]
    torch.testing.assert_close(out["forces"], -grad)
    # Translation invariance => forces sum to zero.
    net = out["forces"].sum(dim=1).abs().max() / out["forces"].abs().max()
    assert net < 1e-4


@pytest.mark.parametrize("n,dim", [(4, 2), (13, 3)])
def test_egnn_equivariance(n, dim):
    torch.manual_seed(0)
    model = EGNNDynamics(n_particles=n, spatial_dim=dim, hidden_nf=32, n_layers=3)
    with torch.no_grad():  # make the (near-zero initialized) output non-trivial
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    pos = torch.randn(5, n, dim)
    pos = pos - pos.mean(dim=1, keepdim=True)
    time = torch.rand(5)
    out = model(time, pos)

    assert out.shape == pos.shape
    assert is_mean_free(out)
    assert out.abs().max() > 1e-3

    rot = random_rotation(dim)
    torch.testing.assert_close(model(time, pos @ rot.T), out @ rot.T, rtol=1e-4, atol=1e-4)
    perm = torch.randperm(n)
    torch.testing.assert_close(model(time, pos[:, perm]), out[:, perm], rtol=1e-4, atol=1e-4)
    # Scalar time (as used by the integrator) and [B, 1, 1] time are accepted.
    torch.testing.assert_close(
        model(time.view(-1, 1, 1), pos), out, rtol=1e-5, atol=1e-6
    )
    assert model(torch.tensor(0.3), pos).shape == pos.shape


def test_gaussian_source_and_bridge_2d():
    source = GaussianSource(num_atoms=4, spatial_dim=2, scale=2.0)
    data_0 = source.sample(16)
    data_1 = source.sample(16)
    assert data_0.shape == (16, 4, 2)
    assert is_mean_free(data_0)

    sde = VESDE(sigma_min=0.001, sigma_max=1.5)
    data_t = sde.sample_posterior(torch.rand(16), data_0, data_1)
    assert data_t.shape == (16, 4, 2)


def test_terminal_cost_clipped_and_centered():
    potential = LennardJonesPotential(n_particles=13, spatial_dim=3)
    cost = BMSTerminalCost(
        potential, temperature=1.0, thermal_energy=1.0, grad_clip_val=10.0, center_grad=True
    )
    pos = torch.randn(8, 13, 3) * 0.6  # close particles -> huge, clipped gradients
    pos = pos - pos.mean(dim=1, keepdim=True)
    grad = cost(pos)["terminal_cost"]
    assert torch.isfinite(grad).all()
    assert is_mean_free(grad, atol=1e-4)
    # Before centering every particle gradient is clipped to norm <= 10.
    assert grad.norm(dim=-1).max() <= 2 * 10.0 + 1e-3


@pytest.mark.parametrize(
    "cls,n,dim,filename",
    [
        (DoubleWellPotential, 4, 2, "test_split_DW4.npy"),
        (LennardJonesPotential, 13, 3, "test_split_LJ13-1000.npy"),
    ],
)
def test_evaluator_reference_vs_reference(cls, n, dim, filename):
    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip("Reference data missing; run scripts/download_nbody_data.sh")
    torch.manual_seed(0)
    potential = cls(n_particles=n, spatial_dim=dim)
    evaluator = NBodyEvaluator(
        potential, str(path), n, dim, max_samples=100, make_plots=False
    )
    ref = evaluator.ref_samples[torch.randperm(len(evaluator.ref_samples))[:100]]
    metrics = evaluator(ref)
    assert metrics["eval/num_samples"] == 100
    # Two independent draws from the reference distribution are close...
    assert metrics["eval/dist_w2"] < 0.05
    # ...and much closer than samples from the prior.
    prior = GaussianSource(num_atoms=n, spatial_dim=dim).sample(100)
    prior_metrics = evaluator(prior)
    assert prior_metrics["eval/eq_w2"] > metrics["eval/eq_w2"]
    assert prior_metrics["eval/energy_w2"] > metrics["eval/energy_w2"]
