"""Projection head and InfoNCE objective (paper Sec. 3.3).

Embeddings are L2-normalised before the dot product (SimCLR-style cosine similarity). The paper
writes raw dot products, but normalisation keeps the temperature tau meaningful.
"""
import torch
import torch.nn.functional as F
from torch import nn


class ProjectionHead(nn.Module):
    """Pools the fused map Z (B, D, H, W) and projects it to the latent vector z (B, L)."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_in), nn.GELU(), nn.Linear(d_in, d_out))

    def forward(self, Z):
        return self.net(Z.mean(dim=(2, 3)))


def info_nce(anchor, positive, negatives, tau):
    """InfoNCE: each anchor i scores its own positive against the shared negative set."""
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    n = F.normalize(negatives, dim=-1)
    pos = (a * p).sum(dim=-1, keepdim=True) / tau
    neg = a @ n.t() / tau
    logits = torch.cat([pos, neg], dim=1)
    target = torch.zeros(a.shape[0], dtype=torch.long, device=a.device)
    return F.cross_entropy(logits, target)
