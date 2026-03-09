"""
lambast/mitigation_methods/density_ratio/datasets.py

Creates datasets for source and target domains, including
with weights from density ratio estimation.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .util import _to_float_tensor

TensorLike = Union[np.ndarray, torch.Tensor]


class DomainDataset(Dataset):
    """
    Dataset for training a domain classifier: source vs target.

    Exposes two indexing modes:
      - standard indexing: returns (x, domain_y) with domain_y in {0,1}
      - domain-specific indexing via get_source(i) / get_target(i)
        to support balanced batching logic.

    Expected shapes:
      X_source: (N_s, C, T)
      X_target: (N_t, C, T)
    """

    def __init__(
        self,
        X_source: TensorLike,
        X_target: TensorLike,
        return_domain_label: bool = True,
    ):
        self.Xs = _to_float_tensor(X_source)
        self.Xt = _to_float_tensor(X_target)

        if self.Xs.ndim != 3 or self.Xt.ndim != 3:
            raise ValueError(
                "X_source and X_target must be 3D tensors of shape (N, C, T). "
                f"Got X_source.ndim={self.Xs.ndim}"
                f", X_target.ndim={self.Xt.ndim}."
            )
        if self.Xs.shape[1] != self.Xt.shape[1]:
            raise ValueError(
                "Source/target must have same number of channels C. "
                f"Got {self.Xs.shape[1]} vs {self.Xt.shape[1]}."
            )
        if self.Xs.shape[2] != self.Xt.shape[2]:
            raise ValueError(
                "Source/target must have same time length T by default. "
                f"Got {self.Xs.shape[2]} vs {self.Xt.shape[2]}."
            )

        self.return_domain_label = return_domain_label

        self.n_source = int(self.Xs.shape[0])
        self.n_target = int(self.Xt.shape[0])
        self._len = self.n_source + self.n_target

    def __len__(self) -> int:
        return self._len

    def get_source(self, i: int) -> torch.Tensor:
        """Return a single source example x."""
        x = self.Xs[i]
        return x

    def get_target(self, i: int) -> torch.Tensor:
        """Return a single target example x."""
        x = self.Xt[i]
        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (x, domain_y) where domain_y is float tensor scalar:
          0.0 for source, 1.0 for target
        """
        if idx < 0 or idx >= self._len:
            raise IndexError(idx)

        if idx < self.n_source:
            x = self.get_source(idx)
            y = torch.tensor(0.0)
        else:
            x = self.get_target(idx - self.n_source)
            y = torch.tensor(1.0)

        if self.return_domain_label:
            return x, y
        # If caller wants only x, still return a consistent tuple
        return x, y


class WeightedTaskDataset(Dataset):
    """
    Dataset for downstream supervised training on SOURCE with sample weights.

    Expected shapes:
      X: (N, C, T)
      y: (N,) or (N,1)  (binary labels typically)
      w: (N,)           (positive weights)
    """

    def __init__(
        self,
        X: TensorLike,
        y: TensorLike,
        w: TensorLike,
    ):
        self.X = _to_float_tensor(X)
        if self.X.ndim != 3:
            raise ValueError(
                "X must be 3D tensor of shape (N, C, T). "
                f"Got X.ndim={self.X.ndim}."
            )

        self.y = _to_float_tensor(y).view(-1)
        self.w = _to_float_tensor(w).view(-1)

        n = int(self.X.shape[0])
        if self.y.shape[0] != n:
            raise ValueError(
                f"y must have length N={n}; got {self.y.shape[0]}.")
        if self.w.shape[0] != n:
            raise ValueError(
                f"w must have length N={n}; got {self.w.shape[0]}.")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        x = self.X[idx]
        y = self.y[idx]
        w = self.w[idx]
        return x, y, w


@dataclass(frozen=True)
class BalancedDomainBatch:
    """
    Convenience container for a balanced domain batch.
    x: (B, C, T)
    y: (B,) float labels 0/1
    """
    x: torch.Tensor
    y: torch.Tensor


def make_balanced_domain_batch(
    domain_ds: DomainDataset,
    batch_size: int,
    *,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    device: Optional[torch.device] = None,
) -> BalancedDomainBatch:
    """
    Build a balanced domain batch explicitly.
    Caller supplies indices for source and target.

    batch_size must be even; half from source, half from target.
    """
    if batch_size % 2 != 0:
        raise ValueError(
            "batch_size must be even for balanced domain batching.")

    half = batch_size // 2
    if source_indices.numel() != half or target_indices.numel() != half:
        raise ValueError(
            f"Expected {half} source indices and {half} target indices, "
            f"got {source_indices.numel()} and {target_indices.numel()}."
        )

    xs = torch.stack([domain_ds.get_source(int(i))
                     for i in source_indices], dim=0)
    xt = torch.stack([domain_ds.get_target(int(i))
                     for i in target_indices], dim=0)

    x = torch.cat([xs, xt], dim=0)
    y = torch.cat([torch.zeros(half, dtype=torch.float32),
                   torch.ones(half, dtype=torch.float32)], dim=0, )

    # Shuffle within the batch so the model doesn't see ordered domains.
    perm = torch.randperm(batch_size)
    x = x[perm]
    y = y[perm]

    if device is not None:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

    return BalancedDomainBatch(x=x, y=y)
