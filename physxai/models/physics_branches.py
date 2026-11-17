"""Per-branch CNN that turns a residual map into a feature map F_k (paper Sec. 3.1)."""
from torch import nn


class BranchEncoder(nn.Module):
    def __init__(self, in_ch, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, residual_map):
        return self.net(residual_map)
