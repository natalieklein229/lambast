"""
lambast/mitigation_methods/density_ratio/models.py

PyTorch models and modules for binary classification and domain classification.

"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureNet1D(nn.Module):
    """
    Feature extractor for 1D time series.

    Input:  (B, C, T)
    Output: (B, D)
    """

    def __init__(self, in_channels: int = 2, hidden: int = 32,
                 out_dim: int = 64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, hidden, 1)
        self.proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)              # (B, hidden, T)
        z = self.pool(z).squeeze(-1)  # (B, hidden)
        z = self.proj(z)              # (B, out_dim)
        return z


class DomainHead(nn.Module):
    """
    Domain classifier head.

    Input:  (B, D)
    Output: (B,) logits
    """

    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(-1)


class DomainClassifier(nn.Module):
    """
    Full domain classifier: FeatureNet + DomainHead

    Input:  (B, C, T)
    Output: (B,) logits
    """

    def __init__(self, in_channels: int = 2, hidden: int = 32,
                 feat_dim: int = 64):
        super().__init__()
        self.feature = FeatureNet1D(
            in_channels=in_channels,
            hidden=hidden,
            out_dim=feat_dim)
        self.head = DomainHead(in_dim=feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feature(x)
        return self.head(z)


class BinaryCNN(nn.Module):
    """
    Simple 1D CNN for binary classification.
    Input:  (B, C, T)
    Output: (B,) logits
    """

    def __init__(self, in_channels: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).squeeze(-1)      # (B, hidden)
        return self.head(z).squeeze(-1)  # (B,)
