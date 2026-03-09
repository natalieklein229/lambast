"""
lambast/mitigation_methods/density_ratio/train_task.py

Training utilities for binary classification task.

"""


from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .util import _to_float_tensor

TensorLike = Union[np.ndarray, torch.Tensor]


@torch.no_grad()
def eval_binary_accuracy(
    model: nn.Module,
    X: TensorLike,
    y: TensorLike,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, float]:
    model.eval()
    X_t = _to_float_tensor(X)
    y_t = _to_float_tensor(y).view(-1)

    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    correct = 0.0
    total = 0.0
    loss_sum = 0.0
    bce = nn.BCEWithLogitsLoss(reduction="sum")

    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss_sum += float(bce(logits, yb).item())
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += float((preds == yb).float().sum().item())
        total += float(yb.numel())

    return {
        "acc": correct / max(total, 1.0),
        "bce": loss_sum / max(total, 1.0),
    }


def train_binary_classifier(
    model: nn.Module,
    X: TensorLike,
    y: TensorLike,
    *,
    X_val=None,
    y_val=None,
    sample_weight: Optional[TensorLike] = None,
    device: Optional[torch.device] = None,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[nn.Module, dict[str, list[float]]]:
    """
    Train a binary classifier on (X, y), optionally with per-example weights.

    X: (N, C, T)
    y: (N,) in {0,1}
    sample_weight: (N,) positive weights (optional)
    """
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    X_t = _to_float_tensor(X)
    y_t = _to_float_tensor(y).view(-1)

    if sample_weight is None:
        ds = TensorDataset(X_t, y_t)
    else:
        w_t = _to_float_tensor(sample_weight).view(-1)
        if w_t.shape[0] != X_t.shape[0]:
            raise ValueError("sample_weight must have same length as X.")
        ds = TensorDataset(X_t, y_t, w_t)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    has_val = (X_val is not None) and (y_val is not None)
    if has_val:
        Xv = _to_float_tensor(X_val)
        yv = _to_float_tensor(y_val).view(-1)
        val_ds = TensorDataset(Xv, yv)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        val_dl = None

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce_none = nn.BCEWithLogitsLoss(reduction="none")

    history: Dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for _ in range(epochs):
        loss_sum = 0.0
        correct = 0.0
        total = 0.0
        for batch in dl:
            if sample_weight is None:
                xb, yb = batch
                wb = None
            else:
                xb, yb, wb = batch

            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            per_ex_loss = bce_none(logits, yb)

            if wb is not None:
                wb = wb.to(device)
                loss = (per_ex_loss * wb).mean()
            else:
                loss = per_ex_loss.mean()

            loss.backward()
            opt.step()

            loss_sum += float(per_ex_loss.sum().item())
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += float((preds == yb).float().sum().item())
            total += float(yb.numel())
        train_loss = loss_sum / max(total, 1.0)
        train_acc = correct / max(total, 1.0)
        if val_dl is not None:
            val_loss, val_acc = _eval_binary_epoch(model, val_dl, device)
        else:
            val_loss, val_acc = float("nan"), float("nan")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return model, history


@torch.no_grad()
def _eval_binary_epoch(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Returns (mean_loss, accuracy) over the dataloader.
    Loss is unweighted BCE (for evaluation).
    """
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")

    loss_sum = 0.0
    correct = 0.0
    total = 0.0

    for batch in dl:
        if len(batch) == 2:
            xb, yb = batch
        else:
            xb, yb, _wb = batch  # ignore weights for eval

        xb = xb.to(device)
        yb = yb.to(device).view(-1)

        logits = model(xb)
        loss_sum += float(bce(logits, yb).item())

        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += float((preds == yb).float().sum().item())
        total += float(yb.numel())

    mean_loss = loss_sum / max(total, 1.0)
    acc = correct / max(total, 1.0)

    model.train()
    return mean_loss, acc
