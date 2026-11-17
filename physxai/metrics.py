"""Evaluation metrics."""
import torch


def auroc(scores, labels):
    """Area under the ROC curve via the Mann-Whitney rank statistic. Returns NaN if one class is absent."""
    scores = scores.detach().cpu().double().flatten()
    labels = labels.detach().cpu().flatten() > 0.5
    n = scores.numel()
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.float64)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
