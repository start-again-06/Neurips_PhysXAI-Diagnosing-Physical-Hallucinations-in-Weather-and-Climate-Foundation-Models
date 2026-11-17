"""Multi-task objective L = lam_det BCE + lam_loc Dice + lam_diag CE + lam_unc NLL + lam_cont InfoNCE.

Weights are learned homoscedastic uncertainties (Kendall et al., 2018), which is how Table 3's
"learnable uncertainty parameter sigma_i" is read here. The paper's adaptive weighting in
Sec. 4.6 is not present in the manuscript, so this is my choice.
"""
import torch
import torch.nn.functional as F
from torch import nn

from ..models.contrastive import info_nce

TASKS = ("det", "loc", "diag", "unc", "cont")


def dice_loss(pred, target, smooth=1.0):
    inter = (pred * target).sum(dim=(1, 2))
    denom = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau
        self.log_vars = nn.Parameter(torch.zeros(len(TASKS)))

    def forward(self, out, batch):
        y = batch["y"]
        pos = y > 0.5
        parts = {}
        parts["det"] = F.binary_cross_entropy_with_logits(out["h_logit"], y)
        parts["loc"] = dice_loss(out["m"], batch["mask"])
        if bool(pos.any()):
            parts["diag"] = F.cross_entropy(out["diag_logits"][pos], batch["family"][pos].long())
        else:
            parts["diag"] = out["diag_logits"].sum() * 0.0
        # The Beta density is evaluated at a smoothed label (0.02 clean, 0.98 perturbed).
        target = 0.02 + 0.96 * y
        dist = torch.distributions.Beta(out["alpha"], out["beta"])
        parts["unc"] = -dist.log_prob(target).mean()
        parts["cont"] = self._contrastive(out["z"], y)

        total = 0.0
        for i, name in enumerate(TASKS):
            s = self.log_vars[i]
            total = total + 0.5 * torch.exp(-s) * parts[name] + 0.5 * s
        return total, {k: float(v.detach()) for k, v in parts.items()}

    def _contrastive(self, z, y):
        clean = z[y < 0.5]
        perturbed = z[y > 0.5]
        if clean.shape[0] < 2 or perturbed.shape[0] < 1:
            return z.sum() * 0.0
        positives = clean.roll(1, dims=0)  # another consistent state as the positive
        return info_nce(clean, positives, perturbed, self.tau)
