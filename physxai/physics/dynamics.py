"""Dynamics branch: hydrostatic balance and geostrophic balance."""
import torch

from ..grid import R_DRY


def hydrostatic_residual(T, Phi, grid):
    """r_H = dPhi/dp + R T / p, shape (B, P, H, W)."""
    p = grid.p_pa.view(-1, 1, 1)
    return grid.ddp(Phi) + R_DRY * T / p


def geostrophic_residual(u, v, Phi, grid, min_abs_lat=5.0):
    """r_G = || (u, v) - (u_g, v_g) ||, with u_g = -(1/f) dPhi/dy and v_g = (1/f) dPhi/dx.

    Geostrophic balance is undefined near the equator, so |lat| < min_abs_lat is zeroed.
    """
    f = grid.f.view(-1, 1)
    f_safe = torch.where(f.abs() < 1e-6, torch.full_like(f, 1e-6), f)
    u_g = -grid.ddy(Phi) / f_safe
    v_g = grid.ddx(Phi) / f_safe
    mask = grid.latitude_mask(min_abs_lat)
    return torch.sqrt((u - u_g) ** 2 + (v - v_g) ** 2 + 1e-12) * mask


def dynamics_residuals(f, grid):
    """Returns (B, 2P, H, W): hydrostatic residual then geostrophic residual per level."""
    r_h = hydrostatic_residual(f["T"], f["Phi"], grid)
    r_g = geostrophic_residual(f["u"], f["v"], f["Phi"], grid)
    return torch.cat([r_h, r_g], dim=1)
