"""PhysXAI: physics-aware multi-branch diagnostic model (paper Sec. 3)."""
import torch
from torch import nn

from ..data.dataset import branch_scores, calibrate_thresholds
from ..grid import Grid
from ..physics.conservation import column_energy, column_mass
from ..physics.residuals import BRANCHES, VARIABLES, branch_channels, compute_residual_maps, unpack
from ..physics.spectra import radial_kinetic_spectrum
from .contrastive import ProjectionHead
from .detector import DetectionHead, DiagnosisHead, LocalizationHead, UncertaintyHead
from .encoder import UNetEncoder
from .fusion import CrossPhysicsAttention
from .physics_branches import BranchEncoder


class PhysXAI(nn.Module):
    """Forward pass on a state x of shape (B, V, P, H, W).

    prev and dt (optional) enable the spatiotemporal residual branch.
    """

    def __init__(self, cfg, grid: Grid):
        super().__init__()
        self.grid = grid
        n_v, n_p = len(VARIABLES), grid.n_levels
        n_bins = min(grid.n_lat, grid.n_lon) // 2
        self.channels = branch_channels(n_p)
        self.register_buffer("in_mean", torch.zeros(n_v, n_p))
        self.register_buffer("in_std", torch.ones(n_v, n_p))
        self.register_buffer("res_scale", torch.ones(sum(self.channels.values())))
        self.register_buffer("ref_mass", torch.zeros(()))
        self.register_buffer("ref_energy", torch.zeros(()))
        self.register_buffer("ref_log_spec", torch.zeros(n_p, n_bins))
        self.register_buffer("has_ref", torch.zeros(()))
        self.register_buffer("theta", torch.zeros(len(BRANCHES)))
        self.register_buffer("Theta", torch.zeros(len(BRANCHES)))

        d, dm = cfg.physics_dim, cfg.encoder_channels
        self.encoder = UNetEncoder(n_v * n_p, cfg.base_channels, dm)
        self.branch_nets = nn.ModuleList([BranchEncoder(self.channels[b], d) for b in BRANCHES])
        self.fusion = CrossPhysicsAttention(dm, d, dm, len(BRANCHES), cfg.n_heads)
        self.proj = ProjectionHead(dm, cfg.contrastive_dim)
        self.det = DetectionHead(cfg.contrastive_dim)
        self.loc = LocalizationHead(dm)
        self.diag = DiagnosisHead(dm, cfg.n_classes)
        self.unc = UncertaintyHead(dm)

    def _raw_dict(self, x, prev=None, dt=None):
        ref = {"mass": self.ref_mass, "energy": self.ref_energy} if bool(self.has_ref) else None
        spec = self.ref_log_spec if bool(self.has_ref) else None
        return compute_residual_maps(x, self.grid, ref_stats=ref, ref_log_spec=spec, prev=prev, dt=dt)

    def _raw_cat(self, x, prev=None, dt=None):
        maps = self._raw_dict(x, prev, dt)
        return torch.cat([maps[b] for b in BRANCHES], dim=1)

    def _split(self, cat):
        sizes = [self.channels[b] for b in BRANCHES]
        return dict(zip(BRANCHES, torch.split(cat, sizes, dim=1)))

    @torch.no_grad()
    def fit_statistics(self, clean_x):
        """Estimate normalisation, climatological references and thresholds from clean states.

        Mirrors the paper's threshold selection (Sec. 2.3): theta_k at the 95th percentile and
        Theta_k at the 99.9th percentile, computed on physically consistent data only.
        """
        f = unpack(clean_x)
        self.in_mean.copy_(clean_x.mean(dim=(0, 3, 4)))
        self.in_std.copy_(clean_x.std(dim=(0, 3, 4)).clamp(min=1e-6))
        self.ref_mass.fill_(self.grid.area_mean(column_mass(f["q"], self.grid)).mean().item())
        energy = column_energy(f["T"], f["Phi"], f["q"], self.grid)
        self.ref_energy.fill_(self.grid.area_mean(energy).mean().item())
        spec = radial_kinetic_spectrum(f["u"], f["v"], self.ref_log_spec.shape[1])
        self.ref_log_spec.copy_(torch.log(spec + 1e-12).mean(dim=0))
        self.has_ref.fill_(1.0)
        raw = self._raw_cat(clean_x)
        self.res_scale.copy_(raw.std(dim=(0, 2, 3)).clamp(min=1e-8))
        scaled = self._split(raw / self.res_scale.view(1, -1, 1, 1))
        theta, Theta = calibrate_thresholds(branch_scores(scaled))
        self.theta.copy_(theta)
        self.Theta.copy_(Theta)

    def forward(self, x, prev=None, dt=None):
        n_v, n_p = self.in_mean.shape
        x_in = ((x - self.in_mean.view(1, n_v, n_p, 1, 1)) / self.in_std.view(1, n_v, n_p, 1, 1))
        E = self.encoder(x_in.flatten(1, 2))
        scaled = self._raw_cat(x, prev, dt) / self.res_scale.view(1, -1, 1, 1)
        parts = self._split(scaled)
        feats = [net(parts[b]) for b, net in zip(BRANCHES, self.branch_nets)]
        Z = self.fusion(E, feats)
        z = self.proj(Z)
        pooled = Z.mean(dim=(2, 3))
        h_logit = self.det(z).squeeze(-1)
        alpha, beta = self.unc(pooled)
        return {
            "h_logit": h_logit,
            "h": torch.sigmoid(h_logit),
            "m": torch.sigmoid(self.loc(Z)).squeeze(1),
            "diag_logits": self.diag(pooled),
            "alpha": alpha,
            "beta": beta,
            "z": z,
            "branch_scores": branch_scores(parts),
        }
