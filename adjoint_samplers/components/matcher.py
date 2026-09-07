# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch.func import grad
from torch.utils.data import DataLoader

from adjoint_samplers.components.buffer import BatchBuffer
from adjoint_samplers.components.sde import BaseSDE, sdeint
from adjoint_samplers.components.state_cost import GradStateCost, ZeroGradStateCost
from adjoint_samplers.components.term_cost import GradEnergy


class Matcher:
    def __init__(
        self,
        sde: BaseSDE | None = None,
        buffer: BatchBuffer | None = None,
        resample_size: int | None = None,
        duplicates: int | None = None,
        loss_scale: float = 1,
        **kwargs,
    ):
        self.sde = sde
        self.buffer = buffer
        self.resample_size = resample_size
        self.duplicates = duplicates
        self.loss_scale = loss_scale

    def build_dataloader(self, batch_size, collate_fn=None) -> DataLoader:
        dataset = self.buffer.build_dataset(self.duplicates)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def populate_buffer(self):
        raise NotImplementedError()

    def prepare_target(self):
        raise NotImplementedError()


class AdjointMatcher(Matcher):
    def __init__(
        self,
        grad_term_cost: GradEnergy | None = None,
        grad_state_cost: GradStateCost | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.grad_term_cost = grad_term_cost
        self.grad_state_cost = grad_state_cost

    @torch.no_grad()
    def _backward_simulate(self, adjoint1, timesteps, xs):
        (T, B, D) = xs.shape
        assert len(timesteps) == T and T > 1
        assert adjoint1.shape == (B, D)

        adjoint = adjoint1.clone()
        adjoints = [adjoint]
        for j in range(T - 1, 0, -1):
            dt = timesteps[j] - timesteps[j - 1]
            assert dt > 0

            t = timesteps[j].repeat((B, 1))
            x = xs[j]
            assert t.shape == (B, 1) and x.shape == (B, D)

            # Compute a^T ∇ b(t, x)
            ref_sde = self.sde.ref_sde
            if not ref_sde.has_drift:
                f = torch.zeros_like(x)
            else:
                with torch.enable_grad():
                    f = grad(
                        lambda x: torch.sum(adjoint * ref_sde.drift(t, x))
                    )(x)

            # Compute gradient of state cost
            f = f + self.grad_state_cost(t, x)

            adjoint = adjoint + f * dt
            adjoints.append(adjoint)

        adjoints = torch.stack(adjoints[::-1])
        assert adjoints.shape == (T, B, D)
        return adjoints

    def _compute_adjoint1(self, x1, is_asbs_init_stage):
        if is_asbs_init_stage:
            # IPF init: First Adjoint Matching stage
            # of ASBS uses zero corrector
            adjoint1 = self.grad_term_cost.grad_E(x1)
        else:
            adjoint1 = self.grad_term_cost(x1)
        return adjoint1

    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
    ):
        (B, D), T = x0.shape, len(timesteps)
        assert x0.device == timesteps.device

        ts = timesteps.unsqueeze(1).repeat((1, B))[..., None]
        assert ts.shape == (T, B, 1)

        xs = sdeint(
            self.sde,
            x0,
            timesteps,
            only_boundary=False,
        )
        xs = torch.stack(xs)
        assert xs.shape == (T, B, D)

        adjoint1 = self._compute_adjoint1(xs[-1], is_asbs_init_stage).clone()
        adjoints = self._backward_simulate(adjoint1, timesteps, xs)
        assert adjoints.shape == (T, B, D)

        # note: use entire traj as one smaple. this improves training.
        ts = ts.transpose(0, 1)
        xs = xs.transpose(0, 1)
        adjoints = adjoints.transpose(0, 1)
        assert ts.shape == (B, T, 1)
        assert adjoints.shape == xs.shape == (B, T, D)

        self.buffer.add({
            "t": ts.reshape(B, T).detach().cpu(),
            "xt": xs.reshape(B, T * D).detach().cpu(),
            "adjointt": adjoints.reshape(B, T * D).detach().cpu(),
        })

    def prepare_target(self, data, device):
        t = data["t"].to(device)
        xt = data["xt"].to(device)
        adjointt = data["adjointt"].to(device)

        (B, T), D = t.shape, xt.shape[1] // t.shape[1]
        assert xt.shape == adjointt.shape == (B, T * D)

        # randomly select B index
        # TODO(ghliu) not used for AS / ASBS. Refac this logic.
        idx = torch.randint(high=T, size=(B,))
        idx = [i*T+id for i, id in enumerate(idx)]

        t = t.reshape(B * T, 1)[idx]
        xt = xt.reshape(B * T, D)[idx]
        adjointt = adjointt.reshape(B * T, D)[idx]
        return (t, xt), - adjointt


class AdjointVEMatcher(AdjointMatcher):
    """ Efficient computation of AM when the base SDE has no drift (e.g., VE)
        and the SOC problem has no state cost.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._check_soc_problem()

    def _check_soc_problem(self):
        assert not self.sde.ref_sde.has_drift
        assert isinstance(self.grad_state_cost, ZeroGradStateCost)

    def _check_buffer_sample_shape(self, x0, x1, adjoint1):
        (B, D) = x0.shape
        assert x1.shape == adjoint1.shape == (B, D)

    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
    ):
        (x0, x1) = sdeint(
            self.sde,
            x0,
            timesteps,
            only_boundary=True,
        )
        adjoint1 = self._compute_adjoint1(x1, is_asbs_init_stage).clone()

        self._check_buffer_sample_shape(x0, x1, adjoint1)
        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
            "adjoint1": adjoint1.to("cpu"),
        })

    def sample_t(self, x):
        (B, D) = x.shape
        return torch.rand(B, 1)

    def _check_target_shape(self, t, xt, adjoint):
        (B, D) = xt.shape
        assert t.shape == (B, 1) and adjoint.shape == (B, D)

    def prepare_target(self, data, device):
        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)
        adjoint1 = data["adjoint1"].to(device)

        t = self.sample_t(x0).to(device)
        xt = self.sde.sample_base_posterior(t, x0, x1)
        adjoint = adjoint1 # const w.r.t. time in this case

        self._check_target_shape(t, xt, adjoint)
        return (t, xt), - adjoint


class AdjointVPMatcher(AdjointVEMatcher):
    """ Efficient computation of AM when the base SDE has linear drift (e.g., VP)
        and the SOC problem has no state cost.
    """
    def _check_soc_problem(self):
        assert self.sde.ref_sde.has_drift
        assert isinstance(self.grad_state_cost, ZeroGradStateCost)

    def prepare_target(self, data, device):
        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)
        adjoint1 = data["adjoint1"].to(device)

        t = self.sample_t(x0).to(device)
        xt = self.sde.sample_base_posterior(t, x0, x1)
        adjoint = adjoint1 # const w.r.t. time in this case
        adjoint = adjoint * torch.exp(self.sde.ref_sde.coeff2(t))

        self._check_target_shape(t, xt, adjoint)
        return (t, xt), - adjoint


class CorrectorMatcher(Matcher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _check_buffer_sample_shape(self, x0, x1):
        (B, D) = x0.shape
        assert x1.shape == (B, D)

    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
    ):
        # IPF init: First Corrector Matching stage
        # of ASBS uses zero controller (i.e., ref_sde)
        sde = self.sde.ref_sde if is_asbs_init_stage else self.sde

        (x0, x1) = sdeint(
            sde,
            x0,
            timesteps,
            only_boundary=True,
        )

        self._check_buffer_sample_shape(x0, x1)
        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
        })

    def _check_target_shape(self, t1, x1, score):
        (B, D) = x1.shape
        assert t1.shape == (B, 1) and score.shape == (B, D)

    def prepare_target(self, data, device):
        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)

        t1 = torch.ones(x0.shape[0], 1).to(device)
        score = self.sde.ref_sde.cond_score(x0, t1, x1)

        self._check_target_shape(t1, x1, score)
        return (t1, x1,), score

class BMSMatcher(Matcher):
    """ Bridge Matching Sampler (Blessing et al.), couplage indépendant P0 ⊗ P^u_T,
        c(t) = γ(t) = κ(t)/κ(1).
        Cible (Prop. C.3, forme numériquement stable utilisée par le code officiel) :
            σ^{-1} ξ = γ(t) [∇log p_prior(X0) − ∇E(X1)] − ∇_{X1} log P_{1|0}(X1 | X0)
        Aucune division par κ(t) => pas de singularité en t=0, pas besoin de t_min.
    """
    def __init__(self, source=None, grad_term_cost=None, t_min: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        assert not self.sde.ref_sde.has_drift
        self.source = source
        self.grad_term_cost = grad_term_cost
        self.t_min = t_min

    def populate_buffer(self, x0, timesteps, is_asbs_init_stage):
        # Simulation avec u_i (sdeint est @no_grad => "detached")
        (_, x1) = sdeint(self.sde, x0, timesteps, only_boundary=True)
        # ∇E évalué une seule fois par échantillon
        grad_E1 = self.grad_term_cost.grad_E(x1)
        mask = torch.isfinite(grad_E1).all(dim=-1) & torch.isfinite(x1).all(dim=-1)
        x1, grad_E1 = x1[mask], grad_E1[mask]
        self.buffer.add({
            "x1": x1.to("cpu"),
            "grad_E1": grad_E1.to("cpu"),
        })

    def prepare_target(self, data, device):
        x1 = data["x1"].to(device)
        grad_E1 = data["grad_E1"].to(device)
        B = x1.shape[0]
        x0 = self.source.sample([B,]).to(device)
        t = self.t_min + (1 - self.t_min) * torch.rand(B, 1, device=device)   # cutoff 1e-3 (Annexe C.3)
        xt = self.sde.ref_sde.sample_posterior(t, x0, x1)
        score_prior = self.source.score(x0)                        # ∇log p_prior(X0)
        score_t0 = self.sde.ref_sde.cond_score(x0, t, xt)          # ∇_{Xt} log P_{t|0} = (X0−Xt)/κ(t)
        target = score_prior - grad_E1 - score_t0                  # Prop. 2.10, c = γ
        return (t, xt), target

    def loss_weight(self, t):
        ref = self.sde.ref_sde
        return ref._diffsquare_integral(t) / ref._diffsquare_integral(torch.ones_like(t))   # γ(t) ∈ [0,1]