"""Regular lat-lon grid, pressure levels and finite-difference operators.

Fields are tensors shaped (..., P, H, W): P pressure levels, H latitudes, W longitudes.
Pressure levels must be strictly descending (bottom of the atmosphere first).
"""
import math

import torch

R_EARTH = 6.371e6   # m
OMEGA = 7.292e-5    # rad/s
R_DRY = 287.05      # J/(kg K)
CP = 1004.0         # J/(kg K)
LV = 2.5e6          # J/kg
G = 9.81            # m/s^2


class Grid:
    def __init__(self, n_lat, n_lon, pressures_hpa, lat_max=80.0):
        if n_lat < 4 or n_lon < 4 or n_lat % 2 or n_lon % 2:
            raise ValueError("n_lat and n_lon must be even and >= 4")
        p = torch.as_tensor(pressures_hpa, dtype=torch.float64)
        if p.numel() < 2 or not bool(torch.all(p[:-1] > p[1:])):
            raise ValueError("pressures_hpa must hold at least two strictly descending values")
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.n_levels = int(p.numel())
        self.pressures_hpa = p.float()
        self.p_pa = (p * 100.0).float()
        self.lat_deg = torch.linspace(-lat_max, lat_max, n_lat)
        dphi = math.radians(2 * lat_max / (n_lat - 1))
        dlam = 2 * math.pi / n_lon
        self.dy = R_EARTH * dphi                                          # scalar, m
        self.dx = R_EARTH * torch.cos(torch.deg2rad(self.lat_deg)) * dlam  # (H,), m
        self.f = 2 * OMEGA * torch.sin(torch.deg2rad(self.lat_deg))        # (H,), 1/s
        w = torch.cos(torch.deg2rad(self.lat_deg))
        self.area_w = w / w.sum()                                          # (H,)
        self.dp = torch.gradient(self.p_pa)[0].abs()                       # (P,) level weights, Pa

    def to(self, device):
        for name in ("pressures_hpa", "p_pa", "lat_deg", "dx", "f", "area_w", "dp"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def ddx(self, f):
        """Zonal derivative per metre, applied to (..., H, W)."""
        return torch.gradient(f, dim=-1)[0] / self.dx.view(-1, 1)

    def ddy(self, f):
        """Meridional derivative per metre, applied to (..., H, W)."""
        return torch.gradient(f, dim=-2)[0] / self.dy

    def ddp(self, f):
        """Vertical derivative with respect to pressure (Pa), applied to (..., P, H, W)."""
        return torch.gradient(f, dim=-3, spacing=(self.p_pa,))[0]

    def column_integral(self, f):
        """Integral over pressure levels divided by g: (..., P, H, W) -> (..., H, W)."""
        return (f * self.dp.view(-1, 1, 1)).sum(dim=-3) / G

    def area_mean(self, f):
        """Cos-latitude weighted horizontal mean: (..., H, W) -> (...)."""
        return (f.mean(dim=-1) * self.area_w).sum(dim=-1)

    def latitude_mask(self, min_abs_lat):
        """(H, 1) mask that is 1 where |lat| >= min_abs_lat."""
        return (self.lat_deg.abs() >= min_abs_lat).float().view(-1, 1)
