"""Configuration dataclasses. Defaults follow Table 4 of the paper where it gives values."""
from dataclasses import dataclass, field, fields


@dataclass
class GridConfig:
    n_lat: int = 32
    n_lon: int = 64
    pressures_hpa: list = field(default_factory=lambda: [850, 500, 250, 100])


@dataclass
class ModelConfig:
    base_channels: int = 64
    encoder_channels: int = 128
    physics_dim: int = 64
    contrastive_dim: int = 128
    n_heads: int = 8
    n_classes: int = 7


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    steps_per_epoch: int = 100  # not given in the paper; one "epoch" here is this many steps
    tau: float = 0.1
    seed: int = 0
    eval_batches: int = 8
    log_every: int = 20


@dataclass
class Config:
    grid: GridConfig = field(default_factory=GridConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path=None):
    cfg = Config()
    if path is None:
        return cfg
    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    for section in ("grid", "model", "train"):
        target = getattr(cfg, section)
        valid = {f.name for f in fields(target)}
        for key, value in (raw.get(section) or {}).items():
            if key not in valid:
                raise KeyError(f"unknown {section} option: {key}")
            setattr(target, key, value)
    return cfg
