"""Spatiotemporal branch: finite-difference advective tendency residuals for T and q.

Residual = dX/dt + u dX/dx + v dX/dy, which is zero for pure advection. Diabatic terms are ignored.
Without a previous state the branch returns zeros.
"""
import torch


def spatiotemporal_residuals(f, grid, prev=None, dt=None):
    """Returns (B, 2P, H, W): advective residual for T, then for q."""
    T = f["T"]
    B, P, H, W = T.shape
    if prev is None or dt is None:
        return T.new_zeros(B, 2 * P, H, W)
    u, v = f["u"], f["v"]
    out = []
    for name in ("T", "q"):
        cur = f[name]
        dfdt = (cur - prev[name]) / dt
        out.append(dfdt + u * grid.ddx(cur) + v * grid.ddy(cur))
    return torch.cat(out, dim=1)
