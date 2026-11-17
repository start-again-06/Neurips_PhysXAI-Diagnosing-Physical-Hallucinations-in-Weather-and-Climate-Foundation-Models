"""Two-level U-Net encoder over the stacked forecast state (paper Sec. 3.2)."""
import torch
from torch import nn


def _conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.GELU(),
    )


class UNetEncoder(nn.Module):
    """(B, in_ch, H, W) -> (B, out_ch, H, W). H and W must be even."""

    def __init__(self, in_ch, base, out_ch):
        super().__init__()
        self.enc1 = _conv_block(in_ch, base)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = _conv_block(base, 2 * base)
        self.up = nn.ConvTranspose2d(2 * base, base, kernel_size=2, stride=2)
        self.dec = _conv_block(2 * base, base)
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        d = self.dec(torch.cat([self.up(s2), s1], dim=1))
        return self.head(d)
