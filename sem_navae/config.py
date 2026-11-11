"""Typed configuration for Sem-NaVAE.

Configs are plain dataclasses loaded from YAML. Unknown keys raise an error so that
typos never silently fall back to defaults. The full resolved config is stored inside
every checkpoint, so inference never has to guess the architecture.

Defaults follow the paper (Sec. IV-A) unless noted otherwise; see ``docs/paper_to_code.md``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class HeatmapSpec:
    """Bird's-eye-view traversability heatmap I_seg (Eq. 2).

    A single spec is shared by the heatmap builder and the decoder's constrained
    sampling, so both always agree on the metric extent of a pixel.
    """

    size_m: float = 18.0  # side length of the square map [m] (paper: 18 m x 18 m)
    resolution: float = 0.2  # [m/px] (paper: 0.2 m/px)
    sigma_px: float = 5.0  # Gaussian smoothing std [px] (paper: sigma = 5)
    max_obstacle_height: float = 1.0  # non-traversable points above this z [m] are ignored

    @property
    def size_px(self) -> int:
        return int(round(self.size_m / self.resolution))

    @property
    def origin_px(self) -> float:
        """Pixel coordinate of the robot (the map is centred on it)."""
        return self.size_px / 2.0


@dataclass
class ModelConfig:
    # --- LiDAR / PointNet ---
    lidar_num_scans: int = 3  # N_l
    lidar_max_points: int = 2560  # N_p per scan
    lidar_norm_dist: float = 50.0  # xyz are divided by this [m]
    lidar_voxel_size: float = 0.08  # [m]
    pointnet_weights: Optional[str] = None  # pre-trained traversability PointNet
    freeze_pointnet: bool = True
    # --- dimensions ---
    feature_dim: int = 256  # per-stream feature size (paper: 256)
    context_dim: int = 256  # h_x
    latent_dim: int = 512  # N_z (paper: 512)
    posterior_hidden_dim: int = 128  # LSTM size of q(z|x,y)
    posterior_mlp_dim: int = 256
    prior_hidden_dim: int = 512
    use_feature_attention: bool = True
    # --- trajectories ---
    past_traj_len: int = 10  # N_v (5 s at 0.5 s)
    num_waypoints: int = 12  # N_w
    dt: float = 1.0  # time between predicted waypoints [s]
    max_vel: float = 2.0  # velocity normalisation [m/s]
    # --- decoder constraints (Sec. III-A3) ---
    use_heatmap: bool = True
    use_map_features: bool = True
    heatmap: HeatmapSpec = field(default_factory=HeatmapSpec)
    max_accel: float = 0.5  # [m/s^2] (paper: 0.5)
    min_speed: float = 0.1  # [m/s]
    max_speed: float = 2.0  # [m/s]
    max_heading_change_deg: float = 45.0  # per step
    num_proposals: int = 20000  # rejection-sampling proposals per step


@dataclass
class DataConfig:
    root: str = "data"
    index_file: str = "index.pkl"
    split_file: Optional[str] = None  # default <output_dir>/<name>/split.json; created if missing
    val_fraction: float = 0.2
    batch_size: int = 16
    num_workers: int = 4
    preload: bool = False
    goal_norm_dist: float = 300.0  # rho is divided by this [m]
    semantic_map_resolution: float = 0.1  # [m/px] of the ground-truth semantic map
    semantic_map_half_size_px: int = 300  # crop half-size around the robot (30 m)


@dataclass
class LossConfig:
    beta_max: float = 1.0  # final KL weight
    kl_anneal_epochs: int = 300  # linear KL annealing
    free_bits: float = 0.0  # 0 disables free bits
    collision_weight: float = 10.0  # lambda (paper: 10)
    collision_sigma_px: float = 4.0  # GK(.) smoothing of the semantic map
    num_prior_samples: int = 6  # z_p samples per observation in the prior LME term (1 = Eq. 4)


@dataclass
class TrainingConfig:
    max_epochs: int = 1000
    max_iterations_per_epoch: Optional[int] = None
    lr: float = 1e-3  # paper: 1e-3
    weight_decay: float = 1e-2  # AdamW
    lr_decay: float = 0.95
    lr_decay_every: int = 10  # epochs (paper: exponential schedule every 10 epochs)
    grad_clip_norm: Optional[float] = 10.0
    eval_every: int = 10
    save_every: int = 40
    eval_num_samples: int = 20
    output_dir: str = "results"
    logger: str = "tensorboard"  # tensorboard | none


@dataclass
class Config:
    name: str = "navae"
    seed: int = 0
    device: str = "cuda"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls, data: Dict[str, Any], path: str = ""):
    """Recursively build dataclass ``cls`` from a dict, rejecting unknown keys."""
    if data is None:
        data = {}
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"Unknown config key(s) at '{path or '<root>'}': {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        default = getattr(cls(), name)
        if dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), value, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _set_by_path(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def config_from_dict(data: Dict[str, Any]) -> Config:
    return _build(Config, data)


def load_config(
    path: Optional[Union[str, Path]] = None, overrides: Optional[List[str]] = None
) -> Config:
    """Load a YAML config and apply ``key.sub=value`` overrides (values parsed as YAML)."""
    data: Dict[str, Any] = {}
    if path is not None:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got '{item}'")
        key, raw = item.split("=", 1)
        _set_by_path(data, key.strip(), yaml.safe_load(raw))
    return config_from_dict(data)


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
