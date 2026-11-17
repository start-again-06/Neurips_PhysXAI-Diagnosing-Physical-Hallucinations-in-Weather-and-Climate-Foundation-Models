"""Thresholds and labels for the paper's physical-hallucination definition (Definition 1)."""
import torch


def branch_scores(branch_maps):
    """Per-sample magnitude of each residual branch: mean |r_k| over channels and grid. Returns (B, K)."""
    return torch.stack([m.abs().mean(dim=(1, 2, 3)) for m in branch_maps.values()], dim=1)


def calibrate_thresholds(scores, p_normal=0.95, p_severe=0.999):
    """theta_k: 95th percentile of each residual on physically consistent data. Theta_k: 99.9th."""
    return torch.quantile(scores, p_normal, dim=0), torch.quantile(scores, p_severe, dim=0)


def hallucination_label(scores, rmse, theta, Theta, delta):
    """Definition 1. y = 1 if (|r_k| > theta_k and rmse < delta) for some k, or |r_k| > Theta_k for some k.

    scores: (B, K); rmse: (B,); theta, Theta: (K,); delta: scalar median forecast RMSE.
    """
    accurate_but_inconsistent = (scores > theta) & (rmse.unsqueeze(1) < delta)
    severe = scores > Theta
    return (accurate_but_inconsistent.any(dim=1) | severe.any(dim=1)).float()
