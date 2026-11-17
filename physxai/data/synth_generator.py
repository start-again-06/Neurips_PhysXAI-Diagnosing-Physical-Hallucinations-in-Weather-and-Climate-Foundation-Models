"""Synthetic clean atmospheric states and physically motivated perturbations (negatives).

These stand in for reanalysis and forecast data so the pipeline runs end to end. The paper gives
only the names of its seven perturbation families, so the exact operators here are my own
approximations. Replace sample_clean with a real data loader for experiments.
"""
import torch

from ..grid import R_DRY
from ..physics.thermodynamics import saturation_humidity

PERTURBATIONS = (
    "mass",
    "hydrostatic",
    "geostrophic",
    "spectral",
    "moisture",
    "temporal",
    "cross_variable",
)

# Residual branch each perturbation is designed to excite (used by tests and reporting).
TARGET_BRANCH = {
    "mass": "conservation",
    "hydrostatic": "dynamics",
    "geostrophic": "dynamics",
    "spectral": "spectral",
    "moisture": "thermodynamic",
    "temporal": "spatiotemporal",
    "cross_variable": "thermodynamic",
}


def smooth_field(batch_shape, H, W, gen, slope=-5.0 / 3.0):
    """Zero-mean, unit-std random fields (*batch_shape, H, W) with an approximate power-law spectrum."""
    noise = torch.randn(*batch_shape, H, W, generator=gen, dtype=torch.float64)
    ky = torch.fft.fftfreq(H, d=1.0 / H, dtype=torch.float64)
    kx = torch.fft.fftfreq(W, d=1.0 / W, dtype=torch.float64)
    k = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    amp = torch.where(k > 0, k.clamp(min=1.0) ** ((slope - 1.0) / 2.0), torch.zeros_like(k))
    f = torch.fft.ifft2(torch.fft.fft2(noise) * amp).real
    f = f - f.mean(dim=(-2, -1), keepdim=True)
    f = f / f.std(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    return f.float()


def _radial_k(H, W):
    ky = torch.fft.fftfreq(H, d=1.0 / H)
    kx = torch.fft.fftfreq(W, d=1.0 / W)
    return torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)


def sample_clean(n, grid, gen):
    """Physically plausible clean states, (n, 5, P, H, W) ordered as residuals.VARIABLES = (u, v, T, q, Phi).

    Geopotential is built hypsometrically from T, winds are near geostrophic at |lat| >= 5 deg,
    and humidity sits below saturation.
    """
    P, H, W = grid.n_levels, grid.n_lat, grid.n_lon
    T = 250.0 + 3.0 * smooth_field((n, P), H, W, gen) + torch.linspace(5.0, -20.0, P).view(1, P, 1, 1)
    Phi = torch.zeros(n, P, H, W)
    Phi[:, 0] = 1.5e4
    log_ratio = torch.log(grid.p_pa[:-1] / grid.p_pa[1:])
    for i in range(P - 1):
        Phi[:, i + 1] = Phi[:, i] + R_DRY * 0.5 * (T[:, i] + T[:, i + 1]) * log_ratio[i]
    f = grid.f.view(-1, 1)
    f_safe = torch.where(f.abs() < 1e-6, torch.full_like(f, 1e-6), f)
    mask = grid.latitude_mask(5.0)
    u = -grid.ddy(Phi) / f_safe * mask + 0.2 * smooth_field((n, P), H, W, gen)
    v = grid.ddx(Phi) / f_safe * mask + 0.2 * smooth_field((n, P), H, W, gen)
    rh = (0.55 + 0.2 * smooth_field((n, P), H, W, gen)).clamp(0.2, 0.95)
    q = rh * saturation_humidity(T, grid.pressures_hpa.view(1, P, 1, 1))
    return torch.stack([u, v, T, q, Phi], dim=1)


def perturb(x, x_other, kind, gen):
    """Apply one perturbation family to a batch x of shape (n, 5, P, H, W).

    x_other supplies material for the temporal family (interpolation towards another state).
    Magnitudes are drawn per sample from [0.1, 0.3).
    """
    n, _, P, H, W = x.shape
    eps = 0.1 + 0.2 * torch.rand(n, generator=gen)
    e = eps.view(n, 1, 1, 1)
    out = x.clone()
    u, v, T, q, Phi = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
    if kind == "mass":
        out[:, 3] = q * (1.0 + e)
    elif kind == "hydrostatic":
        out[:, 4] = Phi + e * 3e4 * smooth_field((n, P), H, W, gen)
    elif kind == "geostrophic":
        c, s = torch.cos(eps).view(n, 1, 1, 1), torch.sin(eps).view(n, 1, 1, 1)
        new_u = c * u - s * v
        new_v = s * u + c * v
        out[:, 0] = new_u
        out[:, 1] = new_v
    elif kind == "spectral":
        mask = (_radial_k(H, W) <= min(H, W) // 6).to(x.dtype)
        out[:, :3] = torch.fft.ifft2(torch.fft.fft2(x[:, :3]) * mask).real
    elif kind == "moisture":
        out[:, 3] = torch.roll(q, shifts=W // 4, dims=-1)
    elif kind == "temporal":
        e5 = eps.view(n, 1, 1, 1, 1)
        out = (1.0 - e5) * x + e5 * x_other
    elif kind == "cross_variable":
        out[:, 2] = torch.roll(T, shifts=1, dims=-3)
    else:
        raise ValueError(f"unknown perturbation: {kind}")
    return out


def make_training_batch(n, grid, gen, p_perturbed=0.5, dt=3600.0):
    """Half clean (y=0), half perturbed (y=1) batch with family labels and a changed-cell mask.

    'prev' is a shifted copy of the clean state. It is a placeholder for real previous analysis
    states, so the spatiotemporal branch receives non-zero input.
    """
    x = sample_clean(n, grid, gen)
    is_pert = torch.rand(n, generator=gen) < p_perturbed
    family = torch.randint(0, len(PERTURBATIONS), (n,), generator=gen)
    other = torch.roll(x, shifts=1, dims=0)
    out = x.clone()
    for k, kind in enumerate(PERTURBATIONS):
        sel = is_pert & (family == k)
        if bool(sel.any()):
            out[sel] = perturb(x[sel], other[sel], kind, gen)
    changed = ((out - x).abs().sum(dim=(1, 2)) > 1e-6).float()
    return {
        "x": out,
        "prev": torch.roll(x, shifts=1, dims=-1),
        "dt": dt,
        "y": is_pert.float(),
        "family": torch.where(is_pert, family, torch.full_like(family, -1)),
        "mask": changed,
    }
