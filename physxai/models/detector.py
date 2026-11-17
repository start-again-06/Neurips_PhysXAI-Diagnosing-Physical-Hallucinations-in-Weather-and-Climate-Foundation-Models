"""Output heads: detection (h), localisation (m), violation type (d) and Beta uncertainty (alpha, beta)."""
import torch.nn.functional as F
from torch import nn


class DetectionHead(nn.Module):
    """Logit of the hallucination score h = sigmoid(w^T z) (paper Theorem 2 form)."""

    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, z):
        return self.fc(z)


class LocalizationHead(nn.Module):
    """Per-pixel logits for the localisation heatmap m."""

    def __init__(self, d):
        super().__init__()
        self.conv = nn.Conv2d(d, 1, 1)

    def forward(self, Z):
        return self.conv(Z)


class DiagnosisHead(nn.Module):
    """Logits over K violation types."""

    def __init__(self, d, n_classes):
        super().__init__()
        self.fc = nn.Linear(d, n_classes)

    def forward(self, pooled):
        return self.fc(pooled)


class UncertaintyHead(nn.Module):
    """Small MLP giving positive Beta parameters (alpha, beta)."""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, pooled):
        out = F.softplus(self.net(pooled)) + 1e-3
        return out[:, 0], out[:, 1]
