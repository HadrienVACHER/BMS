# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Replay buffer storing simulated endpoints and their target scores.

The buffer enables data reuse across fixed-point iterations: each outer iteration
appends freshly simulated ``(X_0, X_1, target_score)`` tuples and the matching
objective is regressed against samples drawn (with replacement) from the buffer.
"""

from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset


class ReplayBuffer(Dataset):
    """A buffer to store tensors for training. Intended to be instantiated per rank
    in a distributed setting."""

    def __init__(self, max_size: int, filter_nan: bool = True):
        self.max_size = max_size
        self.storage = defaultdict(list)
        self.filter_nan = filter_nan

    def extend(self, **kwargs: torch.Tensor) -> None:
        """Extend the buffer with a dictionary of tensors keyed by name."""
        if not kwargs:
            return

        first_tensor = next(iter(kwargs.values()))
        if not isinstance(first_tensor, torch.Tensor):
            raise TypeError("Buffer only accepts torch.Tensor objects.")

        batch_size = first_tensor.size(0)

        for key, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Expected torch.Tensor, got {type(value)} for key '{key}'"
                )
            if value.size(0) != batch_size:
                raise ValueError(
                    f"Batch dimension mismatch for '{key}': expected {batch_size}, "
                    f"got {value.size(0)}"
                )

            value = value.to("cpu")  # always store on cpu
            # Slice along the batch dimension, e.g. (B, N, 3) -> B tensors of (N, 3).
            value_list = [value[i] for i in range(batch_size)]
            self.storage[key].extend(value_list)

        # Drop whole samples that contain NaNs in any field.
        if self.filter_nan:
            nan_indices = set()
            for key, value_list in self.storage.items():
                for i, tensor in enumerate(value_list):
                    if torch.isnan(tensor).any():
                        nan_indices.add(i)
            if nan_indices:
                for key in self.storage:
                    self.storage[key] = [
                        v
                        for i, v in enumerate(self.storage[key])
                        if i not in nan_indices
                    ]

        # Enforce the maximum buffer size (FIFO).
        if len(self) > self.max_size:
            for key in self.storage:
                self.storage[key] = self.storage[key][-self.max_size:]

    def __len__(self) -> int:
        try:
            key = next(iter(self.storage.keys()))
            return len(self.storage[key])
        except StopIteration:
            return 0

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Stack the stored samples per key (for checkpointing)."""
        return {key: torch.stack(values) for key, values in self.storage.items() if values}

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Restore the samples saved by ``state_dict``."""
        self.storage = defaultdict(list)
        for key, value in state_dict.items():
            self.storage[key] = list(value.unbind(0))

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.storage.items()}

    @staticmethod
    def collate_fn(batch: Sequence[Any]) -> Any:
        """Collate a batch back into a single dictionary, stacking the batch dim."""
        element = batch[0]
        if isinstance(element, torch.Tensor):
            return torch.stack(batch, dim=0)
        elif isinstance(element, float):
            return torch.as_tensor(batch, dtype=torch.float)
        elif isinstance(element, int):
            return torch.as_tensor(batch, dtype=torch.long)
        elif isinstance(element, str):
            return batch
        elif isinstance(element, Mapping):
            return {
                key: ReplayBuffer.collate_fn([item[key] for item in batch])
                for key in element.keys()
            }
        elif isinstance(element, Sequence):  # already not str
            return [ReplayBuffer.collate_fn(item) for item in zip(*batch)]
        else:
            raise TypeError(f"Unsupported type {type(element)} for collate_fn.")
