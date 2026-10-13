# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Terminal cost (target score) for the bridge-matching objective.

For BMS the path-dependent target score at the terminal time is the gradient of
the (thermally scaled) potential energy, i.e. ``grad U / (k_B T)``. For
dimensionless energies (e.g. the n-body systems) the thermal scale can be set
directly via ``thermal_energy``.
"""

from abc import ABC, abstractmethod

import ase.units as units
import torch
import torch.nn as nn

from bms.potential.base import BasePotential
from bms.utils.geometry import subtract_mean


class BaseTerminalCost(nn.Module, ABC):
    """Base class for terminal cost functions."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    @torch.no_grad()
    def forward(self, pos_1: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Compute the terminal cost for a geometry at time ``t = 1``."""


class PotentialGradientMixin:
    """Mixin providing thermally scaled potential gradients."""

    def __init__(
        self,
        potential: BasePotential,
        temperature: float,
        grad_clip_val: float | None = None,
        thermal_energy: float | None = None,
        center_grad: bool = False,
    ):
        self.potential = potential
        self.potential.eval()
        self.grad_clip_val = grad_clip_val
        self.temperature = temperature
        self.center_grad = center_grad
        # Thermal scale (k_B T); defaults to eV for molecular force fields.
        if thermal_energy is None:
            thermal_energy = units.kB * temperature
        self.thermal_energy = thermal_energy

    @torch.no_grad()
    def potential_grad(
        self,
        pos: torch.Tensor,  # [B, N, D]
        eps: float = 1e-10,
        return_grad_norm: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute ``grad U / (k_B T)`` (negative thermal forces) and the energy.

        The gradient is clipped per particle to norm ``grad_clip_val`` and, if
        ``center_grad``, projected back onto the mean-free subspace.
        """
        pot_result = self.potential(pos, **kwargs)
        grad = -pot_result["forces"] / self.thermal_energy

        grad_norm = torch.sqrt(grad.pow(2).sum(dim=-1, keepdim=True) + eps)
        if self.grad_clip_val is not None:
            clip_coefficient = torch.clamp(self.grad_clip_val / grad_norm, max=1.0)
            grad = grad * clip_coefficient
        if self.center_grad:
            grad = subtract_mean(grad)

        result = {"energy": pot_result["energy"], "grad": grad}
        if return_grad_norm:
            result["grad_norm"] = grad_norm.squeeze(-1)  # [B, N]
        return result


class BMSTerminalCost(BaseTerminalCost, PotentialGradientMixin):
    """Terminal cost (target score) for the Bridge Matching Sampler."""

    def __init__(
        self,
        potential: BasePotential,
        temperature: float,
        grad_clip_val: float | None = None,
        thermal_energy: float | None = None,
        center_grad: bool = False,
    ):
        BaseTerminalCost.__init__(self)
        PotentialGradientMixin.__init__(
            self, potential, temperature, grad_clip_val, thermal_energy, center_grad
        )

    @torch.no_grad()
    def forward(
        self,
        pos_1: torch.Tensor,  # [B, N, D]
        is_init_stage: bool = False,
        return_energy: bool = False,
        return_grad_norm: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        grad_result = self.potential_grad(
            pos_1, return_grad_norm=return_grad_norm, **kwargs
        )
        result = {"terminal_cost": grad_result["grad"]}

        if return_energy:
            result["energy"] = grad_result["energy"]
        if return_grad_norm:
            result["grad_norm"] = grad_result["grad_norm"]
        return result
