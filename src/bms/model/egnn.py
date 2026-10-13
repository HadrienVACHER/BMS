# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""E(n)-equivariant graph neural network for identical-particle systems.

Ported from ``adjoint_samplers`` (which in turn takes it from DEM,
https://github.com/jarridrb/DEM/blob/main/dem/models/components/egnn.py). This is
the architecture used for the n-body experiments (DW-4, LJ-13, LJ-55). All
particles are identical, so the only node feature is the diffusion time.

Only the coordinate output is used, so the parameters that cannot influence it (the
output node embedding and the node update of the last layer) are omitted. This
keeps the output identical to the reference implementation while making the model
compatible with DDP (no unused parameters).
"""

import torch
import torch.nn as nn

from bms.utils.geometry import subtract_mean


class EGNNDynamics(nn.Module):
    """Fully connected EGNN mapping ``(time, pos)`` to a mean-free vector field."""

    def __init__(
        self,
        n_particles: int,
        spatial_dim: int,
        hidden_nf: int = 128,
        n_layers: int = 5,
        recurrent: bool = True,
        attention: bool = True,
        condition_time: bool = True,
        tanh: bool = True,
        agg: str = "sum",
    ):
        super().__init__()
        self.egnn = EGNN(
            in_node_nf=1,
            in_edge_nf=1,
            hidden_nf=hidden_nf,
            act_fn=nn.SiLU(),
            n_layers=n_layers,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            agg=agg,
        )
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.condition_time = condition_time

        rows, cols = torch.where(~torch.eye(n_particles, dtype=torch.bool))
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        self._edges_cache: dict[tuple[int, torch.device], list[torch.Tensor]] = {}

    def _batched_edges(self, n_batch: int, device: torch.device) -> list[torch.Tensor]:
        key = (n_batch, device)
        if key not in self._edges_cache:
            offsets = torch.arange(n_batch, device=device)[:, None] * self.n_particles
            rows = (self.rows.to(device)[None] + offsets).reshape(-1)
            cols = (self.cols.to(device)[None] + offsets).reshape(-1)
            self._edges_cache = {key: [rows, cols]}
        return self._edges_cache[key]

    def forward(
        self,
        time: torch.Tensor,  # scalar, [B], or [B, 1, 1]
        pos: torch.Tensor,  # [B, N, D]
        return_potential: bool = False,
    ) -> torch.Tensor:
        if return_potential:
            raise NotImplementedError("EGNNDynamics does not output a potential.")

        n_batch = pos.size(0)
        edges = self._batched_edges(n_batch, pos.device)
        x = pos.reshape(n_batch * self.n_particles, self.spatial_dim)

        h = torch.ones(n_batch, self.n_particles, device=pos.device, dtype=pos.dtype)
        if self.condition_time:
            h = h * time.reshape(-1, 1).to(pos).expand(n_batch, 1)
        h = h.reshape(n_batch * self.n_particles, 1)

        edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
        vel = (x_final - x).view(n_batch, self.n_particles, self.spatial_dim)
        return subtract_mean(vel)


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf: int,
        in_edge_nf: int,
        hidden_nf: int,
        act_fn: nn.Module = nn.SiLU(),
        n_layers: int = 4,
        recurrent: bool = True,
        attention: bool = False,
        norm_diff: bool = True,
        tanh: bool = False,
        coords_range: float = 15,
        agg: str = "sum",
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range) / self.n_layers
        if agg == "mean":
            self.coords_range_layer = self.coords_range_layer * 19
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.layers = nn.ModuleList(
            [
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=self.coords_range_layer,
                    agg=agg,
                    update_nodes=i < n_layers - 1,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, h, x, edges, edge_attr=None):
        """Return the updated coordinates."""
        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, edges, x, edge_attr=edge_attr)
        return x


class E_GCL(nn.Module):
    """E(n)-equivariant graph convolutional layer."""

    def __init__(
        self,
        input_nf: int,
        output_nf: int,
        hidden_nf: int,
        edges_in_d: int = 0,
        act_fn: nn.Module = nn.SiLU(),
        recurrent: bool = True,
        attention: bool = False,
        norm_diff: bool = True,
        tanh: bool = False,
        coords_range: float = 1,
        agg: str = "sum",
        update_nodes: bool = True,
    ):
        super().__init__()
        self.update_nodes = update_nodes
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.agg_type = agg
        self.tanh = tanh
        self.coords_range = coords_range
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        if update_nodes:
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, output_nf),
            )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            out = out * self.att_mlp(out)
        return out

    def node_model(self, x, edge_index, edge_attr):
        row, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        out = self.node_mlp(torch.cat([x, agg], dim=1))
        if self.recurrent:
            out = x + out
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, _ = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.tanh:
            trans = trans * self.coords_range
        if self.agg_type == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.agg_type == "mean":
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise ValueError(f"Unknown coordinate aggregation type: {self.agg_type}")
        return coord + agg

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        if self.norm_diff:
            norm = torch.sqrt(radial + 1e-8)
            coord_diff = coord_diff / (norm + 1)
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        if self.update_nodes:
            h = self.node_model(h, edge_index, edge_feat)
        return h, coord


def unsorted_segment_sum(data, segment_ids, num_segments):
    """PyTorch version of TensorFlow's ``unsorted_segment_sum``."""
    result = data.new_zeros((num_segments, data.size(1)))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
