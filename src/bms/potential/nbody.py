# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Synthetic n-body particle potentials (DW-4, LJ-13, LJ-55).

The energies follow the conventions of DEM / adjoint_samplers (which the paper's
n-body results were computed with), so that the energies -- and therefore the
reference data and the ``energy_w2`` metric -- are directly comparable:

* Double well: ``sum_{i<j} a (d_ij - d0)^4 + b (d_ij - d0)^2`` with
  ``a = 0.9``, ``b = -4``, ``d0 = 4``.
* Lennard-Jones: the pair term is summed over *ordered* pairs ``i != j`` (i.e. every
  pair is counted twice), plus a harmonic oscillator ``0.5 * c * ||x - x_COM||^2``.

Energies are dimensionless (``tau = 1``) and positions have shape ``(B, N, D)``.
"""

from abc import abstractmethod

import torch

from bms.potential.base import BasePotential


class ParticlePotential(BasePotential):
    """Base class for potentials defined via an analytic energy ``E(x)``."""

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def energy(self, pos: torch.Tensor) -> torch.Tensor:
        """Energy of a batch of configurations ``(B, N, D) -> (B,)``."""

    def forward(self, pos: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        with torch.enable_grad():
            pos = pos.detach().requires_grad_(True)
            energy = self.energy(pos)
            grad = torch.autograd.grad(energy.sum(), pos)[0]
        return {"energy": energy.detach(), "forces": -grad.detach()}


class DoubleWellPotential(ParticlePotential):
    """Pairwise double-well potential (DW-4: 4 particles in 2D)."""

    def __init__(
        self,
        n_particles: int = 4,
        spatial_dim: int = 2,
        a: float = 0.9,
        b: float = -4.0,
        c: float = 0.0,
        offset: float = 4.0,
    ):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.a, self.b, self.c, self.offset = a, b, c, offset
        rows, cols = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)

    def energy(self, pos: torch.Tensor) -> torch.Tensor:
        diff = pos[:, self.rows] - pos[:, self.cols]  # (B, P, D)
        dists = torch.sqrt(diff.pow(2).sum(-1) + 1e-8) - self.offset
        energies = self.a * dists**4 + self.b * dists**2 + self.c
        return energies.sum(-1)


class LennardJonesPotential(ParticlePotential):
    """Lennard-Jones cluster with a harmonic oscillator on the center of mass."""

    def __init__(
        self,
        n_particles: int = 13,
        spatial_dim: int = 3,
        eps: float = 1.0,
        rm: float = 1.0,
        oscillator_scale: float = 1.0,
    ):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.eps = eps
        self.rm = rm
        self.oscillator_scale = oscillator_scale
        off_diag = ~torch.eye(n_particles, dtype=torch.bool)
        self.register_buffer("off_diag", off_diag, persistent=False)

    def energy(self, pos: torch.Tensor) -> torch.Tensor:
        # All ordered pairs i != j, as in bgflow's ``distance_vectors``.
        diff = pos[:, :, None, :] - pos[:, None, :, :]  # (B, N, N, D)
        # Same epsilon as bgflow's ``distances_from_vectors``.
        dists = torch.sqrt(diff[:, self.off_diag].pow(2).sum(-1) + 1e-6)
        ratio = self.rm / dists
        lj = self.eps * (ratio**12 - 2 * ratio**6)
        energy = lj.sum(-1)

        centered = pos - pos.mean(dim=1, keepdim=True)
        osc = 0.5 * centered.pow(2).sum(dim=(-2, -1))
        return energy + self.oscillator_scale * osc
