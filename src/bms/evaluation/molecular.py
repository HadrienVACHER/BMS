# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluation of molecular samples (e.g. ALA2) against boltzkit reference data.

boltzkit (and therefore OpenMM) is only imported here, so that targets which do
not need it (e.g. the n-body systems) can be trained without it.
"""

import torch

from bms.potential.base import BasePotential


class BoltzkitEvaluator:
    """Energy histogram, torsion marginals, TICA, and dihedral-angle metrics."""

    def __init__(self, potential: BasePotential, temperature: float):
        from boltzkit.evaluation.eval import EnergyHistEval
        from boltzkit.evaluation.molecular_eval import (
            DihedralAngleEval,
            TicaEval,
            TorsionMarginalEval,
        )

        # The boltzkit system is the first term of the (summed) potential.
        self.system = potential.potentials[0].system
        self.val_data = self.system.load_dataset(T=temperature, type="val").get_samples()

        topology = self.system.get_mdtraj_topology()
        self.eval_pipeline = [
            EnergyHistEval(),
            TorsionMarginalEval(topology),
            TicaEval(topology, self.system.get_tica_model()),
            DihedralAngleEval(topology, self.system.get_z_matrix()),
        ]

    @torch.no_grad()
    def __call__(self, samples: torch.Tensor) -> dict:
        from boltzkit.evaluation import EvalData, run_eval
        from boltzkit.evaluation.eval import make_wandb_compatible

        xyz = samples.detach().cpu().numpy()
        pred_flat = xyz.reshape(xyz.shape[0], -1)
        min_size = min(pred_flat.shape[0], self.val_data.shape[0])
        eval_data = EvalData(
            true_samples_target_log_prob=self.system.get_log_prob(
                self.val_data[:min_size]
            ),
            pred_samples_target_log_prob=self.system.get_log_prob(pred_flat[:min_size]),
            samples_true=self.val_data[:min_size],
            samples_pred=pred_flat[:min_size],
        )
        metrics = run_eval(data=eval_data, evals=self.eval_pipeline)
        return make_wandb_compatible(metrics, dpi=100)
