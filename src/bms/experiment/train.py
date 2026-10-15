# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Training entry point for the Bridge Matching Sampler (BMS).

Each outer epoch performs one step of the generalized fixed-point iteration:

1.  *Simulate & couple* -- sample ``X_0`` from the prior, integrate the current
    controlled SDE to obtain ``X_1``, and pair it with an independent fresh
    prior sample (the BMS independent coupling).
2.  *Target score* -- evaluate the terminal cost ``grad U / (k_B T)`` at ``X_1``
    and cache ``(X_0, X_1, target_score)`` in a replay buffer.
3.  *Markovianize* -- regress the Markovian control against the bridge-matching
    target (Nelson's identity). When ``damping > 0`` an extra L2 term towards the
    previous iterate ``u_i`` implements the damped fixed-point update.

Evaluation against reference data is delegated to the configured ``evaluator``
(boltzkit for molecules, see ``bms.evaluation``).
"""

import copy
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from lightning.fabric import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_warn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from bms.process.sde import ControlledSDE
from bms.utils.topology import save_data_to_pdb
from bms.utils.training import EMA, TrainingCurriculum

XI_FORMS = ("c3", "eq24")


def bridge_matching_target(
    xi_form: str,
    base_sde,
    time: torch.Tensor,  # [B]
    data_0: torch.Tensor,  # [B, N, D]
    data_1: torch.Tensor,  # [B, N, D]
    data_t: torch.Tensor,  # [B, N, D]
    prior_score: torch.Tensor,  # grad log p_prior(X_0), [B, N, D]
    target_score: torch.Tensor,  # grad log p_target(X_1), [B, N, D]
) -> torch.Tensor:
    """Regression target ``sigma(t)^-1 xi(X, t)`` with control variate ``c = gamma``.

    Both forms have the same fixed point but different variance, and they give
    different updates away from it (``c3`` matches ``eq24`` with ``c = gamma^2``):

    * ``c3``  (Prop. C.3, as in the released code):
      ``gamma_t * (prior_score + target_score) - grad_{X_1} log p_{1|0}(X_1 | X_0)``
    * ``eq24`` (Prop. 2.10, eq. (24)):
      ``prior_score + target_score - grad_{X_t} log p_{t|0}(X_t | X_0)``. Its last
      term scales as ``1 / kappa(t)``, so the training time needs a cutoff ``t > 0``.
    """
    if xi_form == "c3":
        time_1 = torch.ones_like(time)
        kappa_t = base_sde.cond_var_t0(time=time, data=data_0)
        kappa_1 = base_sde.cond_var_t0(time=time_1, data=data_0)
        ref_score_1_0 = base_sde.cond_score_t0(
            time=time_1, data_0=data_0, data_t=data_1
        )
        return kappa_t / kappa_1 * (prior_score + target_score) - ref_score_1_0
    if xi_form == "eq24":
        ref_score_t_0 = base_sde.cond_score_t0(time=time, data_0=data_0, data_t=data_t)
        return prior_score + target_score - ref_score_t_0
    raise ValueError(f"Unknown xi_form {xi_form!r}; expected one of {XI_FORMS}.")


class BMSTrainer:
    """Trains the Bridge Matching Sampler via the damped fixed-point iteration."""

    TASKS: list[str] = ["bridge_matching"]

    ############################################################################
    # Initialization
    ############################################################################

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        if cfg.xi_form not in XI_FORMS:
            raise ValueError(f"Unknown xi_form {cfg.xi_form!r}; expected one of {XI_FORMS}.")
        if cfg.xi_form == "eq24" and cfg.train_time_min <= 0:
            raise ValueError(
                "xi_form=eq24 divides by kappa(t), which vanishes at t = 0; "
                "set train_time_min > 0 (e.g. 1e-3)."
            )

        wandb_logger = WandbLogger(
            project=cfg.get("project_name", "BridgeMatchingSampler"),
            name=cfg.get("name", "my-run"),
            tags=[cfg.get("tag", "my-run")],
            config=OmegaConf.to_container(cfg, resolve=True),
            save_dir=HydraConfig.get().run.dir,
        )

        self.fabric = hydra.utils.instantiate(cfg.fabric, loggers=wandb_logger)
        self.fabric.launch()
        self.rank_seed = cfg.seed ^ self.fabric.global_rank
        seed_everything(self.rank_seed, workers=False)

        with self.fabric.init_module():
            # Controller (the Markovian control u).
            controller = hydra.utils.instantiate(cfg.controller)
            if cfg.ema_decay is not None:
                self.controller = EMA(controller, cfg.ema_decay, cfg.ema_start_step)
            else:
                self.controller = controller
            if cfg.compile:
                self.controller = torch.compile(self.controller)

            # Replay buffer.
            self.buffer = hydra.utils.instantiate(cfg.buffer)

            # Prior, reference SDE, controlled SDE, and integrator.
            self.source = hydra.utils.instantiate(cfg.source)
            self.base_sde = hydra.utils.instantiate(cfg.base_sde)
            self.sde = ControlledSDE(base_sde=self.base_sde, controller=self.controller)
            self.integrator = hydra.utils.instantiate(cfg.integrator, sde=self.sde)

            # Target potential and terminal cost (target score).
            self.potential = hydra.utils.instantiate(cfg.potential)
            self.terminal_cost = hydra.utils.instantiate(
                cfg.terminal_cost, potential=self.potential
            )
            self.annealer = hydra.utils.instantiate(cfg.annealer)

        self.curriculum = TrainingCurriculum(
            curriculum=cfg.curriculum, valid_tasks=self.TASKS
        )

        self.controller_optimizer = hydra.utils.instantiate(
            cfg.controller_optimizer,
            params=self.controller.parameters(),
        )
        self.controller, self.controller_optimizer = self.fabric.setup(
            self.controller, self.controller_optimizer
        )
        if hasattr(cfg, "controller_scheduler"):
            self.controller_scheduler = hydra.utils.instantiate(
                cfg.controller_scheduler,
                optimizer=self.controller_optimizer,
            )

        # Reference data and evaluation metrics.
        self.evaluator = hydra.utils.instantiate(cfg.evaluator, potential=self.potential)

    ############################################################################
    # Training loop
    ############################################################################

    def fit(self):
        self.global_step = 0
        self.current_epoch = 0
        if getattr(self.cfg, "pretrained_controller_checkpoint", None):
            self.load_pretrained_controller_checkpoint()
        self.load_checkpoint()

        while self.current_epoch < len(self.curriculum):
            # Snapshot the previous iterate u_i for the damped update.
            if self.current_epoch % self.cfg.prev_model_interval_epoch == 0:
                self.snapshot_prev_controller()

            current_task = self.curriculum[self.current_epoch]
            self.populate_buffer(current_task)
            dataloader = self.create_dataloader(current_task)
            for batch in tqdm(
                dataloader,
                desc=f"Epoch {self.current_epoch + 1}/{len(self.curriculum)}",
                leave=False,
                disable=self.fabric.global_rank != 0 or not self.cfg.progress_bar,
            ):
                self.training_step(batch, current_task)
                self.global_step += 1
            self.current_epoch += 1
            if self.current_epoch % self.cfg.save_checkpoint_interval_epoch == 0:
                self.save_checkpoint()
            if self.current_epoch % self.cfg.eval_interval_epoch == 0:
                self.eval()

    def snapshot_prev_controller(self):
        self.prev_controller = copy.deepcopy(self.controller)
        self.prev_controller.eval()
        for param in self.prev_controller.parameters():
            param.requires_grad = False

    ############################################################################
    # Simulation, coupling, and buffering
    ############################################################################

    @torch.no_grad()
    def populate_buffer(self, current_task):
        buffer = self.buffer
        if len(buffer) == 0:
            num_samples = self.cfg.initial_buffer_samples
        else:
            num_samples = self.cfg.buffer_samples_per_epoch
        if num_samples % self.cfg.inference_batch_size != 0:
            rank_zero_warn(
                f"Number of samples ({num_samples}) is not divisible by inference "
                f"batch size ({self.cfg.inference_batch_size}). Truncating to the "
                "nearest lower multiple."
            )
        num_extend_cycles = num_samples // self.cfg.inference_batch_size

        controller_training_state = self.controller.training
        self.controller.eval()

        for _ in range(num_extend_cycles):
            # Simulate X_1 under the current control, starting from a prior sample.
            data_0 = self.annealer(
                self.source.sample(self.cfg.inference_batch_size), self.current_epoch
            )
            data_1 = self.integrator.run(
                initial_data=data_0,
                center_every_step=self.cfg.mean_free,
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=self.fabric.global_rank == 0,
            )

            # Independent coupling: pair X_1 with a fresh, independent prior sample.
            data_0 = self.annealer(
                self.source.sample(self.cfg.inference_batch_size), self.current_epoch
            )

            populate_data = {"data_0": data_0, "data_1": data_1}
            results = self.terminal_cost(
                pos_1=data_1,
                is_init_stage=current_task.is_init_stage,
                return_energy=self.cfg.log_buffer_energy,
                return_grad_norm=self.cfg.log_buffer_grad_norm,
            )
            populate_data["terminal_cost"] = results["terminal_cost"]
            if self.cfg.log_buffer_energy:
                populate_data["energy"] = results["energy"]
            if self.cfg.log_buffer_grad_norm:
                populate_data["grad_norm"] = results["grad_norm"]
            buffer.extend(**populate_data)

        if self.cfg.log_buffer_energy:
            median_energy = torch.as_tensor(buffer.storage["energy"]).median()
            median_energy = self.fabric.all_reduce(
                self.fabric.to_device(median_energy), reduce_op="mean"
            )
            self.fabric.log(
                "train/buffer_median_energy", median_energy, step=self.global_step
            )
            self.fabric.log(
                "train/temperature", self.cfg.temperature, step=self.global_step
            )
            if self.fabric.global_rank == 0:
                if self.current_epoch % self.cfg.save_sample_interval_epoch == 0:
                    self.save_samples(data_1)

        if self.cfg.log_buffer_grad_norm:
            median_grad_norm = torch.cat(buffer.storage["grad_norm"], dim=0).median()
            median_grad_norm = self.fabric.all_reduce(
                self.fabric.to_device(median_grad_norm), reduce_op="max"
            )
            self.fabric.log(
                "train/buffer_median_grad_norm",
                median_grad_norm,
                step=self.global_step,
            )

        self.fabric.log("train/buffer_size", len(buffer), step=self.global_step)
        self.fabric.barrier()
        self.controller.train(controller_training_state)

    def save_samples(self, data: torch.Tensor):
        if self.fabric.global_rank != 0:
            return
        sample_directory = Path(self.cfg.sample_directory)
        sample_directory.mkdir(parents=True, exist_ok=True)
        if self.cfg.get("topology_pdb_file"):
            filename = sample_directory / f"epoch_{self.current_epoch:04d}.pdb"
            save_data_to_pdb(
                data=data.detach().cpu().numpy(),
                topology_file=self.cfg.topology_pdb_file,
                output_file=str(filename),
            )
        else:
            filename = sample_directory / f"epoch_{self.current_epoch:04d}.pt"
            torch.save({"pos": data.detach().cpu()}, filename)

    def create_dataloader(self, current_task) -> DataLoader:
        dataset = self.buffer
        num_samples = self.cfg.train_batch_size * self.cfg.steps_per_epoch
        epoch_seed = self.rank_seed ^ (self.current_epoch << 16)
        generator = torch.Generator().manual_seed(epoch_seed)
        sampler = RandomSampler(
            dataset, replacement=True, num_samples=num_samples, generator=generator
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train_batch_size,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )
        return self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=False)

    ############################################################################
    # Markovianization step (bridge matching with optional damping)
    ############################################################################

    def training_step(self, batch, current_task) -> None:
        loss, matching_loss, damping_loss = self.bridge_matching_step(batch)
        model = self.controller
        optimizer = self.controller_optimizer

        optimizer.zero_grad()
        self.fabric.backward(loss)

        if self.global_step % self.cfg.log_loss_interval_step == 0:
            self.log_norms()

        if self.cfg.gradient_clip_val is not None:
            self.fabric.clip_gradients(
                model, optimizer, max_norm=self.cfg.gradient_clip_val
            )
        optimizer.step()

        if self.cfg.ema_decay is not None:
            model.update_ema(self.global_step)

        if hasattr(self, "controller_scheduler"):
            if (
                not hasattr(self.cfg, "skip_scheduler_steps")
                or self.global_step >= self.cfg.skip_scheduler_steps
            ):
                self.controller_scheduler.step()

        if self.global_step % self.cfg.log_loss_interval_step == 0:
            self.fabric.log(
                "train/lr",
                optimizer.param_groups[0]["lr"],
                step=self.global_step,
            )
            loss = self.fabric.all_reduce(loss.detach(), reduce_op="mean")
            self.fabric.log("train/loss", loss, step=self.global_step)
            matching_loss = self.fabric.all_reduce(
                matching_loss.detach(), reduce_op="mean"
            )
            self.fabric.log("train/matching_loss", matching_loss, step=self.global_step)
            if damping_loss is not None:
                damping_loss = self.fabric.all_reduce(
                    damping_loss.detach(), reduce_op="mean"
                )
                self.fabric.log(
                    "train/damping_loss", damping_loss, step=self.global_step
                )

    def bridge_matching_step(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        data_0 = batch["data_0"]
        data_1 = batch["data_1"]
        terminal_cost = batch["terminal_cost"]
        batch_size = data_0.size(0)

        # Sample a bridge state X_t given the coupling (X_0, X_1).
        time_min = self.cfg.train_time_min
        time = time_min + (1 - time_min) * torch.rand(batch_size, dtype=torch.float)
        time = self.fabric.to_device(time)
        data_t = self.base_sde.sample_posterior(time=time, data_0=data_0, data_1=data_1)

        control = self.controller(time, data_t)

        # Target drift via Nelson's identity and the generalized target score identity.
        target_score = -terminal_cost
        prior_score = -data_0 / self.source.scale ** 2
        nelson = bridge_matching_target(
            self.cfg.xi_form,
            self.base_sde,
            time=time,
            data_0=data_0,
            data_1=data_1,
            data_t=data_t,
            prior_score=prior_score,
            target_score=target_score,
        )

        matching_loss = (control - nelson).pow(2).mean()

        # Damped fixed-point update: regularize towards the previous iterate u_i.
        if self.cfg.damping > 0:
            prev_control = self.prev_controller(time, data_t)
            damping_loss = (control - prev_control).pow(2).mean()
            loss = matching_loss + self.cfg.damping * damping_loss
            return loss, matching_loss, damping_loss

        return matching_loss, matching_loss, None

    ############################################################################
    # Checkpointing
    ############################################################################

    def load_pretrained_controller_checkpoint(self):
        checkpoint_path = Path(self.cfg.pretrained_controller_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Pretrained controller checkpoint not found: {checkpoint_path}"
            )
        self.fabric.load(checkpoint_path, {"controller": self.controller})

    def load_checkpoint(self):
        state = {
            "controller": self.controller,
            "controller_optimizer": self.controller_optimizer,
        }
        self.snapshot_prev_controller()
        if hasattr(self, "controller_scheduler"):
            state["controller_scheduler"] = self.controller_scheduler

        if not self.cfg.get("resume", True):
            return
        checkpoint_directory = Path(self.cfg.checkpoint_directory)
        if not checkpoint_directory.exists():
            return
        checkpoint_files = list(checkpoint_directory.glob("*.pt"))
        if checkpoint_files:
            checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            remainder = self.fabric.load(checkpoint_path, state)
            self.global_step = remainder.pop("global_step")
            self.current_epoch = remainder.pop("current_epoch")
            self.load_buffer()

    def save_checkpoint(self):
        checkpoint_directory = Path(self.cfg.checkpoint_directory)
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_directory / f"epoch_{self.current_epoch:04d}.pt"
        state = {
            "controller": self.controller,
            "controller_optimizer": self.controller_optimizer,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
        }
        if hasattr(self, "controller_scheduler"):
            state["controller_scheduler"] = self.controller_scheduler
        self.fabric.save(checkpoint_path, state)
        self.save_buffer()

    def _buffer_path(self, epoch: int) -> Path:
        # Buffers are per rank and kept in a subfolder, so that the ``*.pt`` glob
        # used to find the latest checkpoint does not pick them up.
        rank = self.fabric.global_rank
        return (
            Path(self.cfg.checkpoint_directory)
            / "buffer"
            / f"epoch_{epoch:04d}_rank{rank}.pt"
        )

    def save_buffer(self):
        """Save this rank's replay buffer, keeping only the latest copy."""
        path = self._buffer_path(self.current_epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.buffer.state_dict(), path)
        rank = self.fabric.global_rank
        for old in path.parent.glob(f"epoch_*_rank{rank}.pt"):
            if old != path:
                old.unlink()

    def load_buffer(self):
        """Restore this rank's replay buffer for the resumed epoch, if saved."""
        path = self._buffer_path(self.current_epoch)
        if path.exists():
            self.buffer.load_state_dict(torch.load(path, weights_only=True))
        else:
            rank_zero_warn(
                f"No replay buffer saved at {path}; resuming with an empty buffer."
            )

    ############################################################################
    # Logging and evaluation
    ############################################################################

    @torch.no_grad()
    def log_norms(self):
        total_grad_norm = 0.0
        total_param_norm = 0.0
        for p in self.controller.parameters():
            if p.requires_grad:
                total_param_norm += p.detach().norm(2) ** 2
                if p.grad is not None:
                    total_grad_norm += p.grad.detach().norm(2) ** 2
        self.fabric.log(
            "train/model_grad_norm", total_grad_norm ** 0.5, step=self.global_step
        )
        self.fabric.log(
            "train/model_param_norm", total_param_norm ** 0.5, step=self.global_step
        )

    @torch.no_grad()
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Simulate ``X_1`` under the current control for ``num_samples`` prior samples."""
        controller_training_state = self.controller.training
        self.controller.eval()
        batch_size = self.cfg.inference_batch_size
        samples = []
        for start in range(0, num_samples, batch_size):
            data_0 = self.source.sample(min(batch_size, num_samples - start))
            data_1 = self.integrator.run(
                initial_data=data_0,
                center_every_step=self.cfg.mean_free,
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=False,
            )
            samples.append(data_1)
        self.controller.train(controller_training_state)
        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def eval(self):
        shape = (self.cfg.num_atoms, self.cfg.spatial_dim)

        if self.cfg.eval_source == "buffer":
            # Evaluate on the most recent samples stored in the replay buffer.
            local_positions = [
                batch.to(self.fabric.device).reshape(-1, *shape)
                for batch in self.buffer.storage.get("data_1", [])
            ]
            if local_positions:
                local_tensor = torch.cat(local_positions, dim=0)
            else:
                local_tensor = torch.zeros((0, *shape), device=self.fabric.device)
        elif self.cfg.eval_source == "fresh":
            # Simulate fresh samples, split evenly across ranks.
            world_size = self.fabric.world_size
            num_local = -(-self.cfg.num_eval_samples // world_size)
            local_tensor = self.generate_samples(num_local)
        else:
            raise ValueError(f"Unknown eval_source: {self.cfg.eval_source}")

        gathered_tensor = self.fabric.all_gather(local_tensor)
        if self.fabric.global_rank != 0:
            return

        samples = gathered_tensor.reshape(-1, *shape).detach().cpu()
        if self.cfg.eval_source == "fresh":
            samples = samples[: self.cfg.num_eval_samples]
        if samples.shape[0] == 0:
            return

        metrics = self.evaluator(samples)
        self.fabric.log_dict(to_loggable(metrics), step=self.global_step)


def to_loggable(metrics: dict) -> dict:
    """Convert matplotlib figures to W&B images (and close them)."""
    loggable = {}
    for key, value in metrics.items():
        if isinstance(value, plt.Figure):
            loggable[key] = wandb.Image(value)
            plt.close(value)
        else:
            loggable[key] = value
    return loggable


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("highest")
    trainer = BMSTrainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
