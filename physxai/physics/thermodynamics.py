"""Cross-variable thermodynamic branch: Clausius-Clapeyron, static stability and hypsometric balance."""
import torch

from ..grid import CP, R_DRY


def saturation_humidity(T, p_hpa):
    """Saturation specific humidity (kg/kg) from temperature (K) and pressure (hPa), Bolton (1980)."""
    es = 6.112 * torch.exp(17.67 * (T - 273.15) / (T - 29.65))  # hPa
    return 0.622 * es / (p_hpa - 0.378 * es)


def potential_temperature(T, p_hpa):
    return T * (1000.0 / p_hpa) ** (R_DRY / CP)


def hypsometric_residual(T, Phi, grid):
    """Layer thickness check between adjacent levels: dPhi - R * Tmean * ln(p_bottom / p_top).

    Returns (B, P, H, W), zero-padded at the top level so the shape matches the other branches.
    """
    dphi = Phi[..., 1:, :, :] - Phi[..., :-1, :, :]
    t_mean = 0.5 * (T[..., 1:, :, :] + T[..., :-1, :, :])
    log_ratio = torch.log(grid.p_pa[:-1] / grid.p_pa[1:]).view(-1, 1, 1)
    r = dphi - R_DRY * t_mean * log_ratio
    return torch.cat([r, torch.zeros_like(r[..., :1, :, :])], dim=-3)


def thermodynamic_residuals(f, grid):
    """Returns (B, 3P, H, W): Clausius-Clapeyron supersaturation, static instability, hypsometric residual."""
    T, q, Phi = f["T"], f["q"], f["Phi"]
    p_hpa = grid.pressures_hpa.view(-1, 1, 1)
    supersat = torch.relu(q - saturation_humidity(T, p_hpa)) * 1000.0  # g/kg
    instability = torch.relu(grid.ddp(potential_temperature(T, p_hpa)))  # stable air has d(theta)/dp <= 0
    hypso = hypsometric_residual(T, Phi, grid)
    return torch.cat([supersat, instability, hypso], dim=-3)
