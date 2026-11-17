"""Conservation branch: column dry-air mass, column total energy and column divergence."""
import torch

from ..grid import CP, LV


def column_mass(q, grid):
    """Dry-air column mass proxy: integral of (1 - q) dp / g. (B, P, H, W) -> (B, H, W)."""
    return grid.column_integral(1.0 - q)


def column_energy(T, Phi, q, grid):
    """Column total energy: integral of (cp T + Phi + Lv q) dp / g. (B, P, H, W) -> (B, H, W)."""
    return grid.column_integral(CP * T + Phi + LV * q)


def conservation_residuals(f, grid, ref_stats=None):
    """Returns (B, 3, H, W): local column divergence, global mass drift, global energy drift.

    ref_stats: optional dict with scalar tensors 'mass' and 'energy' (analysis or climatological
    reference). Without it the two global channels are zero.
    """
    div_col = grid.column_integral(grid.ddx(f["u"]) + grid.ddy(f["v"]))
    B, H, W = div_col.shape
    mass = grid.area_mean(column_mass(f["q"], grid))
    energy = grid.area_mean(column_energy(f["T"], f["Phi"], f["q"], grid))
    if ref_stats is None:
        d_mass = torch.zeros_like(mass)
        d_energy = torch.zeros_like(energy)
    else:
        d_mass = mass - ref_stats["mass"]
        d_energy = energy - ref_stats["energy"]
    global_maps = [
        d_mass.view(B, 1, 1, 1).expand(B, 1, H, W),
        d_energy.view(B, 1, 1, 1).expand(B, 1, H, W),
    ]
    return torch.cat([div_col.unsqueeze(1)] + global_maps, dim=1)
