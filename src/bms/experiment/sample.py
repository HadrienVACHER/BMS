# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Generate samples from a trained BMS checkpoint and evaluate them against
reference data with the configured ``evaluator``.

Example
-------
    python -u -m bms.experiment.sample \
        experiment=ala2 \
        checkpoint_directory=ckpts/ala2/checkpoints \
        num_samples=100000

    python -u -m bms.experiment.sample \
        experiment=lj13 \
        checkpoint_directory=outputs/lj13/checkpoints \
        num_samples=2000
"""

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from bms.process.sde import ControlledSDE
from bms.utils.topology import save_data_to_pdb


def _latest_checkpoint(directory: Path) -> Path:
    checkpoints = list(directory.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints (*.pt) found in {directory}")
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def _normalize_state_dict(state_dict: dict) -> dict:
    """Map a saved controller ``state_dict`` onto a plain ``ClippedModel``.

    Training may wrap the controller in ``torch.compile`` (``_orig_mod.`` prefix)
    and/or an ``EMA`` module. For EMA we keep the averaged ("shadow") weights.
    """
    # Strip the torch.compile wrapper prefix first, if present.
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    # Then prefer the EMA-averaged ("shadow") weights when present.
    if any(k.startswith("shadow.") for k in state_dict):
        state_dict = {
            k[len("shadow."):]: v
            for k, v in state_dict.items()
            if k.startswith("shadow.")
        }
    return state_dict


@torch.no_grad()
def generate(cfg: DictConfig, device: torch.device):
    controller = hydra.utils.instantiate(cfg.controller).to(device)

    checkpoint_path = _latest_checkpoint(Path(cfg.checkpoint_directory))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["controller"]
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    controller.load_state_dict(_normalize_state_dict(state_dict))
    controller.eval()

    source = hydra.utils.instantiate(cfg.source).to(device)
    base_sde = hydra.utils.instantiate(cfg.base_sde).to(device)
    sde = ControlledSDE(base_sde=base_sde, controller=controller)
    integrator = hydra.utils.instantiate(cfg.integrator, sde=sde).to(device)

    num_samples = int(cfg.num_samples)
    batch_size = int(cfg.inference_batch_size)
    num_batches = (num_samples + batch_size - 1) // batch_size
    samples = []
    for _ in tqdm(range(num_batches), desc="sampling"):
        data_0 = source.sample(batch_size).to(device)
        data_1 = integrator.run(
            initial_data=data_0,
            center_every_step=cfg.mean_free,
            zero_last_step_noise=False,
            return_trajectory=False,
            progress_bar=False,
        )
        samples.append(data_1.cpu())
    return torch.cat(samples, dim=0)[:num_samples]


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = generate(cfg, device)  # (num_samples, n_atoms, spatial_dim)

    out_dir = Path(HydraConfig.get().run.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"pos": samples}, out_dir / "samples.pt")
    if cfg.get("topology_pdb_file"):
        save_data_to_pdb(
            data=samples.numpy(),
            topology_file=cfg.topology_pdb_file,
            output_file=str(out_dir / "samples.pdb"),
        )

    # Evaluate against the reference data.
    potential = hydra.utils.instantiate(cfg.potential)
    evaluator = hydra.utils.instantiate(cfg.evaluator, potential=potential)
    metrics = evaluator(samples)

    print("Evaluation metrics:")
    for key, value in metrics.items():
        if isinstance(value, plt.Figure):
            filename = out_dir / f"{key.replace('/', '_')}.png"
            value.savefig(filename, dpi=150)
            plt.close(value)
        elif isinstance(value, (int, float)):
            print(f"  {key}: {value:.5f}")


if __name__ == "__main__":
    main()
