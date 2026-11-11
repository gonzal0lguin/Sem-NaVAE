"""Training dataset (format documented in ``docs/dataset_format.md``)."""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sem_navae.config import Config
from sem_navae.data.collate import collate
from sem_navae.data.transforms import normalize_lidar, voxelize_with_intensity
from sem_navae.keys import Keys
from sem_navae.utils.geometry import normalize_angle
from sem_navae.utils.io import read_pkl


def split_ids(ids: List[str], val_fraction: float, seed: int) -> Dict[str, List[str]]:
    """Deterministic train/val split of sample ids (a seeded permutation of the sorted ids)."""
    ordered = sorted(ids)
    perm = np.random.default_rng(seed).permutation(len(ordered))
    n_val = int(round(len(ordered) * val_fraction))
    val = sorted(ordered[i] for i in perm[:n_val])
    train = sorted(ordered[i] for i in perm[n_val:])
    return {"seed": seed, "val_fraction": val_fraction, "train": train, "val": val}


def default_split_file(cfg: Config) -> str:
    return os.path.join(cfg.training.output_dir, cfg.name, "split.json")


def load_or_create_split(cfg: Config, ids: List[str]) -> Dict[str, List[str]]:
    """Read ``data.split_file`` (default ``<output_dir>/<name>/split.json``) or create it.

    Train and val datasets read the same file, so the splits never overlap, and a resumed
    or repeated run reuses exactly the same validation set.
    """
    path = cfg.data.split_file or default_split_file(cfg)
    if os.path.exists(path):
        with open(path) as f:
            split = json.load(f)
        missing = set(split["train"]) | set(split["val"])
        missing -= set(ids)
        if missing:
            raise ValueError(f"{path} lists {len(missing)} samples that are not in the index")
        return split
    split = split_ids(ids, cfg.data.val_fraction, cfg.seed)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=1)
    return split


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)


class NaVAEDataset(Dataset):
    """One sample = one observation with its ground-truth future trajectories.

    The index pickle holds ``root = [maps_dir, samples_dir]`` and
    ``ids = [(sample_file, map_file), ...]``.
    """

    def __init__(self, cfg: Config, train: bool):
        data = cfg.data
        index = read_pkl(os.path.join(data.root, data.index_file))
        self.root = data.root
        self.train = train
        self.samples_dir = index["root"][1]
        self.maps_dir = index["root"][0]
        self.cfg = cfg

        by_sample = {sample: (sample, map_file) for sample, map_file in index["ids"]}
        split = load_or_create_split(cfg, list(by_sample))
        self.ids = [by_sample[s] for s in split["train" if train else "val"]]

        self.cache = None
        if data.preload:
            self.cache = [self._load(i) for i in range(len(self.ids))]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        return self.cache[i] if self.cache is not None else self._load(i)

    def _load(self, i: int) -> dict:
        return self._make_sample(
            read_pkl(os.path.join(self.root, self.samples_dir, self.ids[i][0])), i
        )

    def _local_goal(self, goal: np.ndarray, pose: np.ndarray) -> np.ndarray:
        rho = np.linalg.norm(goal) / self.cfg.data.goal_norm_dist
        # TODO(T8b): the goal is stored in the robot frame, yet the pose yaw is subtracted.
        theta = normalize_angle(np.arctan2(goal[1], goal[0]) - pose[-1]) / np.pi
        return np.array([rho, theta])

    def _crop_map(self, semantic_map: np.ndarray) -> np.ndarray:
        h = self.cfg.data.semantic_map_half_size_px
        cy, cx = semantic_map.shape[0] / 2, semantic_map.shape[1] / 2
        return semantic_map[int(cy - h) : int(cy + h), int(cx - h) : int(cx + h)]

    def _make_sample(self, meta: dict, i: int) -> dict:
        m = self.cfg.model
        path = meta["path"][: min(m.num_waypoints, len(meta["path"]))][:, :4]
        sample = {Keys.path: path, Keys.pose: meta["pose"]}
        if "all_paths" in meta:
            sample[Keys.all_paths] = meta["all_paths"]

        if meta.get("lidar_dn") is not None:
            scans = np.array(meta["lidar_dn"], dtype=np.float32)
        else:
            scans = np.stack(
                [
                    voxelize_with_intensity(s, m.lidar_voxel_size, m.lidar_max_points)
                    for s in meta["lidar"]
                ]
            )
        sample[Keys.lidar] = normalize_lidar(scans, m.lidar_norm_dist)
        sample[Keys.goal] = self._local_goal(meta["goal"], meta["pose"])

        # TODO(T8b): past_traj_len is ignored; only past velocities are normalised by max_vel.
        past = np.array(meta["previous_path"][:, :4], dtype=np.float64)
        past[:, 2:] = past[:, 2:] / m.max_vel
        sample[Keys.past_traj] = past
        sample[Keys.semantic_map] = self._crop_map(meta["local_map"])
        if not self.train:
            sample[Keys.sample_id] = np.asarray([int(self.ids[i][1][:-4])])
        return sample


def make_loaders(cfg: Config, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    def loader(train: bool) -> DataLoader:
        return DataLoader(
            NaVAEDataset(cfg, train=train),
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=shuffle,
            collate_fn=collate,
            worker_init_fn=_worker_init,
            drop_last=False,
        )

    return loader(True), loader(False)
