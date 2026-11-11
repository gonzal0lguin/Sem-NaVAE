"""Shared helpers for the legacy-parity ("golden") tests.

Parameters are filled from a generator seeded by the *new* parameter name, so the legacy
model and the port get identical weights without storing any weights in the repository.
"""

from __future__ import annotations

import zlib

import numpy as np
import torch

# legacy parameter name fragment -> new name fragment (applied in order)
LEGACY_TO_NEW = [
    ("perception.dist_encoder.", "perception.prior_encoder."),
    (".ln_obs.", ".ln_traj."),
    (".ln_pcl.", ".ln_lidar."),
    (".fc_logv.", ".fc_logvar."),
    (".hx_proj.", ".context_proj."),
    ("perception.past_tr_enc.rnn_encoder.", "perception.past_traj_enc.lstm."),
    ("perception.past_tr_enc.fc1.", "perception.past_traj_enc.fc."),
    ("gt_traj_encoder.initial_h_model.", "posterior_encoder.init_h."),
    ("gt_traj_encoder.initial_c_model.", "posterior_encoder.init_c."),
    ("gt_traj_encoder.rnn_encoder.", "posterior_encoder.lstm."),
    ("gt_traj_encoder.", "posterior_encoder."),
    ("decoder.rnn_encoder.", "decoder.gru."),
    ("decoder.fc_std.", "decoder.fc_logvar."),
]
SEG_BIAS = 0.7
LEGACY_UNUSED = (".ln_vel.",)  # legacy parameters never used in the forward pass


def map_legacy_name(name: str) -> str:
    for old, new in LEGACY_TO_NEW:
        name = name.replace(old, new)
    return name


@torch.no_grad()
def deterministic_init(model: torch.nn.Module, legacy: bool = False) -> list:
    """Fill every parameter deterministically from its (new) name. Returns the new names."""
    names = []
    for name, p in model.named_parameters():
        if legacy:
            if any(s in name for s in LEGACY_UNUSED):
                continue
            name = map_legacy_name(name)
        g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
        noise = torch.randn(p.shape, generator=g)
        if p.dim() >= 2:
            fan_in = int(np.prod(p.shape[1:]))
            p.copy_(noise / np.sqrt(fan_in))
        elif name.endswith("weight"):  # norm-layer scales
            p.copy_(1.0 + 0.05 * noise)
        else:
            p.copy_(0.05 * noise)
        if name == "perception.lidar_enc.encoder.conv4.bias":
            # bias the random segmentation so both classes occur and the heatmap has
            # traversable cells (otherwise rejection sampling never accepts a proposal)
            p.copy_(torch.tensor([SEG_BIAS, -SEG_BIAS]))
        names.append(name)
    return sorted(names)


def synthetic_batch(seed: int = 0) -> dict:
    """Small deterministic batch in the dataset format (NumPy arrays)."""
    rng = np.random.default_rng(seed)
    B, n_scans, n_pts = 2, 3, 2560
    lidar = np.concatenate(
        [
            rng.uniform(-0.3, 0.3, (B, n_scans, n_pts, 2)),  # +-15 m once de-normalised
            rng.uniform(-0.05, 0.05, (B, n_scans, n_pts, 1)),
            rng.normal(0, 1, (B, n_scans, n_pts, 1)),
        ],
        axis=-1,
    ).astype(np.float32)
    paths = []
    for b in range(B):
        m = 3 if b == 0 else 2
        vel = rng.uniform(-1, 1, (m, 1, 2)) * 0.5 + np.array([1.0, 0.0])
        vel = np.repeat(vel, 12, axis=1) + rng.normal(0, 0.05, (m, 12, 2))
        pos = np.cumsum(vel, axis=1)
        paths.append(np.concatenate([pos, vel], axis=-1).astype(np.float32))
    semantic_map = (rng.uniform(size=(B, 600, 600)) > 0.97).astype(np.float32)
    past = np.concatenate([np.cumsum(np.full((B, 10, 2), 0.5), 1), np.full((B, 10, 2), 0.5)], -1)
    return {
        "previous_path": past.astype(np.float32)
        + rng.normal(0, 0.02, (B, 10, 4)).astype(np.float32),
        "lidar": lidar,
        "goal": rng.uniform(-1, 1, (B, 2)).astype(np.float32),
        "path": np.stack([p[0] for p in paths]),
        "all_paths_0": paths[0],
        "all_paths_1": paths[1],
        "semantic_map": semantic_map,
    }


def to_torch_batch(arrays: dict, map_key: str) -> dict:
    batch = {
        k: torch.from_numpy(np.asarray(v)) for k, v in arrays.items() if not k.startswith("all_")
    }
    batch["all_paths"] = [list(torch.from_numpy(arrays[f"all_paths_{b}"])) for b in range(2)]
    batch[map_key] = batch.pop("semantic_map")
    return batch


PATHS_FOR_GOAL = np.array(
    [
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
        [[0.0, 0.0], [-1.0, 0.5], [-2.0, 1.0]],
    ]
)
GOAL = np.array([10.0, 1.0])
