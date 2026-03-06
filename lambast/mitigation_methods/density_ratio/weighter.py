
"""
lambast/mitigation_methods/density_ratio/weighter.py

Density ratio weighting.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .datasets import DomainDataset, make_balanced_domain_batch
from .models import DomainClassifier
from .util import _to_float_tensor

TensorLike = Union[np.ndarray, torch.Tensor]

class DensityRatioWeighter:
    """
    Train a domain classifier to distinguish source vs target,
    then compute density-ratio weights for source samples:

        w(x) = d(x) / (1 - d(x))

    where d(x) = P(target | x) from a balanced domain classifier.
    Stores:
      - diagnostics_: summary metrics after fit
      - history_: per-epoch train/val loss & accuracy for the domain classifier
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        *,
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        eps: float = 1e-4,
        w_clip: Tuple[float, float] = (0.1, 10.0),
        normalize: bool = True,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.w_clip = w_clip
        self.normalize = normalize

        self.device = (
            device if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model
        self._fitted = False
        self.diagnostics_ = {}
        self.history_ = {}

    # ----------------------------
    # Public API
    # ----------------------------

    def fit(self, X_source: TensorLike, X_target: TensorLike, X_source_val=None, X_target_val=None) -> "DensityRatioWeighter":
        """
        Train domain classifier on source vs target.
        Parameters
        ----------
        X_source_train: (Ns_train, C, T)
        X_target_train: (Nt_train, C, T)
        X_source_val:   (Ns_val, C, T) optional
        X_target_val:   (Nt_val, C, T) optional
        """

        domain_ds = DomainDataset(X_source, X_target)
        n_source = domain_ds.n_source
        n_target = domain_ds.n_target

        has_val = (X_source_val is not None) and (X_target_val is not None)
        if has_val:
            val_ds = DomainDataset(X_source_val, X_target_val)
        else:
            val_ds = None

        if self.model is None:
            self.model = DomainClassifier(in_channels=domain_ds.Xs.shape[1])

        self.model.to(self.device)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        criterion = nn.BCEWithLogitsLoss()

        half_batch = self.batch_size // 2
        if self.batch_size % 2 != 0:
            raise ValueError("batch_size must be even for balanced domain training.")

        self.model.train()
        self.history_ = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        for epoch in range(self.epochs):
            # number of balanced steps per epoch
            steps = max(n_source, n_target) // half_batch
            running_loss = 0.0
            running_acc = 0.0
            for _ in range(steps):
                src_idx = torch.randint(0, n_source, (half_batch,))
                tgt_idx = torch.randint(0, n_target, (half_batch,))

                batch = make_balanced_domain_batch(
                    domain_ds,
                    self.batch_size,
                    source_indices=src_idx,
                    target_indices=tgt_idx,
                    device=self.device,
                )

                optimizer.zero_grad()
                logits = self.model(batch.x)
                eps_ls = 0.05  # try 0.05, 0.1
                y = batch.y * (1.0 - 2 * eps_ls) + eps_ls
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    acc = (preds == batch.y).float().mean()

                running_loss += float(loss.item())
                running_acc += float(acc.item())
            train_loss = running_loss / steps
            train_acc = running_acc / steps
            # optional val evaluation (full-pass)
            if val_ds is not None:
                val_loss, val_acc = self._eval_domain_full(val_ds)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            self.history_["train_loss"].append(train_loss)
            self.history_["train_acc"].append(train_acc)
            self.history_["val_loss"].append(val_loss)
            self.history_["val_acc"].append(val_acc)

        self._fitted = True

        # Basic diagnostics
        ref_ds = val_ds if val_ds is not None else domain_ds
        final_loss, final_acc = self._eval_domain_full(ref_ds)
        self.diagnostics_["domain_loss"] = float(final_loss)
        self.diagnostics_["domain_accuracy"] = float(final_acc)
        self.diagnostics_["used_validation_for_diagnostics"] = 1.0 if val_ds is not None else 0.0

        return self

    def compute_weights(self, X_source: TensorLike, alpha=1) -> torch.Tensor:
        """
        Compute density-ratio weights for source samples.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before compute_weights().")

        Xs = _to_float_tensor(X_source).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(Xs)
            d = torch.sigmoid(logits)

        d = torch.clamp(d, self.eps, 1.0 - self.eps)

        # odds = p(target|x) / p(source|x)
        w = torch.pow(d / (1.0 - d),alpha)

        # clip
        w = torch.clamp(w, self.w_clip[0], self.w_clip[1])

        # normalize to mean 1
        if self.normalize:
            w = w / w.mean()

        return w.detach().cpu()

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _compute_domain_diagnostics(self, domain_ds: DomainDataset) -> None:
        """
        Compute simple domain accuracy as a sanity check.
        """

        self.model.eval()
        with torch.no_grad():
            Xs = domain_ds.Xs.to(self.device)
            Xt = domain_ds.Xt.to(self.device)

            logits_s = self.model(Xs)
            logits_t = self.model(Xt)

            ds = torch.sigmoid(logits_s)
            dt = torch.sigmoid(logits_t)

            # predicted target if prob > 0.5
            acc_s = (ds < 0.5).float().mean()
            acc_t = (dt > 0.5).float().mean()
            acc = 0.5 * (acc_s + acc_t)

        self.diagnostics_["domain_accuracy"] = float(acc.cpu())

    def _domain_epoch_eval(self, model, Xs, Xt, device):
        model.eval()
        bce = torch.nn.BCEWithLogitsLoss(reduction="mean")
        with torch.no_grad():
            # Source label 0
            ys = torch.zeros(Xs.shape[0], dtype=torch.float32, device=device)
            # Target label 1
            yt = torch.ones(Xt.shape[0], dtype=torch.float32, device=device)

            ls = bce(model(Xs), ys).item()
            lt = bce(model(Xt), yt).item()
            loss = 0.5 * (ls + lt)

            ps = (torch.sigmoid(model(Xs)) < 0.5).float().mean().item()
            pt = (torch.sigmoid(model(Xt)) > 0.5).float().mean().item()
            acc = 0.5 * (ps + pt)

        model.train()
        return loss, acc
    
    @torch.no_grad()
    def _eval_domain_full(self, domain_ds: DomainDataset) -> Tuple[float, float]:
        """
        Full evaluation on all source and all target examples.

        Returns
        -------
        loss: float (average of source-loss and target-loss)
        acc:  float (average of source-acc and target-acc)
        """
        self.model.eval()
        bce = nn.BCEWithLogitsLoss(reduction="mean")

        Xs = domain_ds.Xs.to(self.device)
        Xt = domain_ds.Xt.to(self.device)

        ys = torch.zeros(Xs.shape[0], dtype=torch.float32, device=self.device)
        yt = torch.ones(Xt.shape[0], dtype=torch.float32, device=self.device)

        logits_s = self.model(Xs)
        logits_t = self.model(Xt)

        loss_s = bce(logits_s, ys).item()
        loss_t = bce(logits_t, yt).item()
        loss = 0.5 * (loss_s + loss_t)

        ds = torch.sigmoid(logits_s)
        dt = torch.sigmoid(logits_t)

        acc_s = (ds < 0.5).float().mean().item()
        acc_t = (dt > 0.5).float().mean().item()
        acc = 0.5 * (acc_s + acc_t)

        self.model.train()
        return float(loss), float(acc)