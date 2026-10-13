# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluation for the synthetic n-body systems (DW-4, LJ-13, LJ-55).

Ported from ``adjoint_samplers`` (``SyntheticEenergyEvaluator``), whose metrics are
the ones reported in the paper (Table 2):

* ``eq_w2``     -- Wasserstein-2 between samples and reference samples, where the
  pairwise cost is the RMSD after optimal particle permutation (Hungarian) and
  rigid alignment (Kabsch). Reported as "W2" in Table 2.
* ``energy_w2`` -- 1D Wasserstein-2 between the energy distributions. Reported as
  "E(.) W2" in Table 2.
* ``dist_w2``   -- 1D *squared* Wasserstein-2 between interatomic distance
  distributions (kept as in the reference implementation).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from scipy.optimize import linear_sum_assignment

from bms.potential.nbody import ParticlePotential
from bms.utils.geometry import subtract_mean


class NBodyEvaluator:
    """Compares generated samples against reference (long-run MCMC) samples."""

    def __init__(
        self,
        potential: ParticlePotential,
        ref_samples_path: str,
        n_particles: int,
        spatial_dim: int,
        max_samples: int = 2000,
        compute_eq_w2: bool = True,
        make_plots: bool = True,
    ):
        self.potential = potential
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.max_samples = max_samples
        self.compute_eq_w2 = compute_eq_w2
        self.make_plots = make_plots

        path = Path(ref_samples_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Reference samples not found at {path}. "
                "Run scripts/download_nbody_data.sh first."
            )
        ref = torch.as_tensor(np.load(path, allow_pickle=True), dtype=torch.float)
        self.ref_samples = subtract_mean(ref.reshape(-1, n_particles, spatial_dim))

    @torch.no_grad()
    def __call__(self, samples: torch.Tensor) -> dict:
        samples = samples.reshape(-1, self.n_particles, self.spatial_dim)
        samples = samples.detach().float().cpu()
        # eq_w2 is quadratic in the number of samples, so cap it (2000 as in the
        # reference implementation).
        num = min(samples.size(0), self.ref_samples.size(0), self.max_samples)
        samples = samples[:num]
        idxs = torch.randperm(self.ref_samples.size(0))[:num]
        ref_samples = self.ref_samples[idxs]

        gen_energy = self._energy(samples)
        ref_energy = self._energy(ref_samples)
        energy_w2 = pot.emd2_1d(ref_energy, gen_energy) ** 0.5

        gen_dist = interatomic_dist(samples).numpy()
        ref_dist = interatomic_dist(ref_samples).numpy()
        dist_w2 = pot.emd2_1d(gen_dist.reshape(-1), ref_dist.reshape(-1))

        metrics = {
            "eval/energy_w2": float(energy_w2),
            "eval/dist_w2": float(dist_w2),
            "eval/num_samples": num,
        }

        if self.compute_eq_w2:
            M = dist_point_clouds(samples, ref_samples)
            weights = torch.ones(M.shape[0], dtype=torch.double) / M.shape[0]
            eq_w2 = pot.emd2(M=M.double() ** 2, a=weights, b=weights) ** 0.5
            metrics["eval/eq_w2"] = float(eq_w2)

        if self.make_plots:
            metrics["eval/histograms"] = _histogram_figure(
                gen_dist, ref_dist, gen_energy, ref_energy
            )
        return metrics

    def _energy(self, pos: torch.Tensor) -> np.ndarray:
        # Evaluate on the potential's device without moving the (shared) potential.
        device = next(self.potential.buffers()).device
        return self.potential.energy(pos.to(device)).cpu().numpy()


def _histogram_figure(gen_dist, ref_dist, gen_energy, ref_energy):
    """Interatomic-distance and energy histograms (cf. Figure 7 of the paper)."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    bins = np.linspace(
        min(gen_dist.min(), ref_dist.min()), max(gen_dist.max(), ref_dist.max()), 100
    )
    axes[0].hist(ref_dist.reshape(-1), bins=bins, density=True, alpha=0.5, label="reference")
    axes[0].hist(gen_dist.reshape(-1), bins=bins, density=True, alpha=0.5, label="BMS")
    axes[0].set_xlabel("interatomic distance")
    axes[0].legend()

    lo, hi = np.percentile(ref_energy, [0.5, 99.5])
    span = hi - lo
    bins = np.linspace(lo - 0.25 * span, hi + 0.25 * span, 100)
    # Clip so that out-of-range (e.g. early-training) samples pile up at the edges.
    axes[1].hist(ref_energy, bins=bins, density=True, alpha=0.5, label="reference")
    axes[1].hist(
        np.clip(gen_energy, bins[0], bins[-1]),
        bins=bins,
        density=True,
        alpha=0.5,
        label="BMS",
    )
    axes[1].set_xlabel("energy")
    axes[1].legend()
    fig.tight_layout()
    return fig


def interatomic_dist(x: torch.Tensor) -> torch.Tensor:
    """Pairwise distances ``(B, N, D) -> (B, N (N - 1) / 2)``, without duplicates."""
    n_particles = x.size(1)
    rows, cols = torch.triu_indices(n_particles, n_particles, offset=1)
    return torch.linalg.norm(x[:, rows] - x[:, cols], dim=-1)


# From https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
def find_rigid_alignment(A: torch.Tensor, B: torch.Tensor):
    """Kabsch algorithm: optimal rotation/translation mapping point cloud A onto B."""
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    R = V.mm(U.T)
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    return R, t.T.squeeze()


# From https://github.com/jarridrb/DEM/blob/main/dem/models/components/distribution_distances.py
def _ot_reorder(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    dists = torch.cdist(x0, x1)
    _, col_ind = linear_sum_assignment(dists)
    return x1[col_ind]


def dist_point_clouds(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Cost matrix between two sets of point clouds, invariant to permutations,
    rotations, and translations. Shapes ``(B0, N, D)``, ``(B1, N, D)`` -> ``(B0, B1)``."""
    M = []
    for i in range(len(x0)):
        reordered = torch.stack([_ot_reorder(x0[i], x1[j]) for j in range(len(x1))])
        R, _ = torch.vmap(find_rigid_alignment)(
            x0[i][None].repeat(len(x1), 1, 1), reordered
        )
        superimposed = torch.matmul(reordered, R)
        M.append(torch.cdist(x0[i].reshape(1, -1), superimposed.reshape(len(x1), -1)))
    return torch.stack(M).squeeze()
