"""Spectral branch: radially binned kinetic energy spectrum, slope deviation and spectral distance.

Wavenumbers are grid-index wavenumbers (cycles per domain), not physical units. The -5/3 target is
therefore a convention for this implementation, not a physical calibration.
"""
import torch


def radial_kinetic_spectrum(u, v, n_bins):
    """(B, P, H, W) winds -> (B, P, n_bins) radially binned KE spectrum for wavenumbers 1..n_bins."""
    B, P, H, W = u.shape
    U = torch.fft.fft2(u)
    V = torch.fft.fft2(v)
    energy = 0.5 * (U.abs() ** 2 + V.abs() ** 2) / float(H * W) ** 2
    ky = torch.fft.fftfreq(H, d=1.0 / H, device=u.device)
    kx = torch.fft.fftfreq(W, d=1.0 / W, device=u.device)
    kr = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).round().long().clamp(max=n_bins + 1)
    index = kr.reshape(1, 1, -1).expand(B, P, -1)
    spec = torch.zeros(B, P, n_bins + 2, dtype=energy.dtype, device=u.device)
    spec.scatter_add_(2, index, energy.reshape(B, P, -1))
    return spec[..., 1 : n_bins + 1]


def spectral_slope(spec, k_min=2, k_max=None):
    """Least-squares slope of log E(k) vs log k over k_min..k_max. (B, P, K) -> (B, P)."""
    n = spec.shape[-1]
    k_max = k_max or n // 2
    k = torch.arange(1, n + 1, dtype=spec.dtype, device=spec.device)
    sel = (k >= k_min) & (k <= k_max)
    x = torch.log(k[sel])
    y = torch.log(spec[..., sel] + 1e-12)
    xc = x - x.mean()
    yc = y - y.mean(dim=-1, keepdim=True)
    return (xc * yc).sum(dim=-1) / (xc ** 2).sum()


def spectral_residuals(u, v, grid, ref_log_spec=None, slope_target=-5.0 / 3.0):
    """Returns (B, 2P, H, W): |slope + 5/3| per level, then L2 distance to a reference log-spectrum."""
    B, P, H, W = u.shape
    n_bins = min(H, W) // 2
    spec = radial_kinetic_spectrum(u, v, n_bins)
    slope_dev = (spectral_slope(spec) - slope_target).abs()
    if ref_log_spec is None:
        distance = torch.zeros_like(slope_dev)
    else:
        log_spec = torch.log(spec + 1e-12)
        distance = torch.sqrt(((log_spec - ref_log_spec) ** 2).sum(dim=-1) + 1e-12)
    per_level = torch.cat([slope_dev, distance], dim=1)  # (B, 2P)
    return per_level[:, :, None, None].expand(B, 2 * P, H, W)
