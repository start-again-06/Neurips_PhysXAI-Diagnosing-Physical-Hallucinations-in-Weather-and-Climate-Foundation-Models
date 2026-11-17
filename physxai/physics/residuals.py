"""Assembles the five physical residual branches into one map per branch."""
from .conservation import conservation_residuals
from .consistency import spatiotemporal_residuals
from .dynamics import dynamics_residuals
from .spectra import spectral_residuals
from .thermodynamics import thermodynamic_residuals

VARIABLES = ("u", "v", "T", "q", "Phi")
BRANCHES = ("conservation", "dynamics", "spectral", "thermodynamic", "spatiotemporal")


def branch_channels(n_levels):
    """Number of residual-map channels each branch produces."""
    return {
        "conservation": 3,
        "dynamics": 2 * n_levels,
        "spectral": 2 * n_levels,
        "thermodynamic": 3 * n_levels,
        "spatiotemporal": 2 * n_levels,
    }


def unpack(x):
    """(B, V, P, H, W) -> dict of (B, P, H, W) tensors keyed by VARIABLES."""
    return {name: x[:, i] for i, name in enumerate(VARIABLES)}


def compute_residual_maps(x, grid, ref_stats=None, ref_log_spec=None, prev=None, dt=None):
    """Dict branch name -> (B, C_branch, H, W) raw residual maps."""
    f = unpack(x)
    return {
        "conservation": conservation_residuals(f, grid, ref_stats),
        "dynamics": dynamics_residuals(f, grid),
        "spectral": spectral_residuals(f["u"], f["v"], grid, ref_log_spec),
        "thermodynamic": thermodynamic_residuals(f, grid),
        "spatiotemporal": spatiotemporal_residuals(f, grid, unpack(prev) if prev is not None else None, dt),
    }
