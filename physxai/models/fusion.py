"""Cross-physics attention fusion (paper Sec. 3.2).

Per pixel, the encoder features form the query and each physics branch forms one key/value.
Attention is a softmax over the branch axis, so each pixel weights the physics branches it needs.
The paper's written form, sum_k softmax(Q K_k^T) V_k, is degenerate when applied per branch
(a softmax over one key is 1), so the softmax is taken across branches instead.
"""
import math

import torch
from torch import nn


class CrossPhysicsAttention(nn.Module):
    def __init__(self, d_enc, d_branch, d_model, n_branches, n_heads):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_branches = n_branches
        self.q = nn.Conv2d(d_enc, d_model, 1)
        self.k = nn.ModuleList([nn.Conv2d(d_branch, d_model, 1) for _ in range(n_branches)])
        self.v = nn.ModuleList([nn.Conv2d(d_branch, d_model, 1) for _ in range(n_branches)])
        self.e_proj = nn.Conv2d(d_enc, d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, E, feats):
        B, _, H, W = E.shape
        N = H * W
        q = self.q(E).flatten(2).transpose(1, 2)  # (B, N, d_model)
        k = torch.stack([c(f).flatten(2).transpose(1, 2) for c, f in zip(self.k, feats)], dim=2)
        v = torch.stack([c(f).flatten(2).transpose(1, 2) for c, f in zip(self.v, feats)], dim=2)
        q = q.reshape(B, N, self.n_heads, self.d_head)
        k = k.reshape(B, N, self.n_branches, self.n_heads, self.d_head)
        v = v.reshape(B, N, self.n_branches, self.n_heads, self.d_head)
        scores = torch.einsum("bnhd,bnkhd->bnhk", q, k) / math.sqrt(self.d_head)
        weights = scores.softmax(dim=-1)
        z = torch.einsum("bnhk,bnkhd->bnhd", weights, v).reshape(B, N, -1)
        e_p = self.e_proj(E).flatten(2).transpose(1, 2)
        out = self.norm(z + e_p)  # Z_final = LayerNorm(Z + E_proj)
        return out.transpose(1, 2).reshape(B, -1, H, W)
