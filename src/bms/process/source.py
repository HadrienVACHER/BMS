# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Prior (source) distribution from which BMS samples the initial states ``X_0``."""

import torch
import torch.nn as nn

from bms.utils.composition import composition_to_atomic_numbers
from bms.utils.geometry import is_mean_free, subtract_mean


class GaussianSource(nn.Module):
    """Generates a batch of mean-free Gaussian positions of shape ``(B, N, D)``."""

    def __init__(
        self,
        num_atoms: int | None = None,
        composition: str | None = None,
        scale: float = 1.0,
        center: bool = True,
        spatial_dim: int = 3,
    ):
        super().__init__()
        if not ((num_atoms is None) ^ (composition is None)):
            raise ValueError("Exactly one of num_atoms or composition must be provided.")

        if composition is not None:
            atomic_numbers = composition_to_atomic_numbers(composition)
            num_atoms = len(atomic_numbers)

        self.num_atoms = num_atoms
        self.spatial_dim = spatial_dim
        self.center = center
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float))

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample a batch of positions of shape ``(B, N, D)``."""
        pos = torch.randn(
            batch_size,
            self.num_atoms,
            self.spatial_dim,
            dtype=torch.float,
            device=self.scale.device,
        )
        pos = pos * self.scale

        if self.center:
            pos = subtract_mean(pos)
            if not is_mean_free(pos):
                raise ValueError("Sampled data is not perfectly mean-free.")

        return pos
