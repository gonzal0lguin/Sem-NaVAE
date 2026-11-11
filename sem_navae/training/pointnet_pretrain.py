"""Pre-training of the traversability PointNet (paper Sec. III-A2).

Per-point binary labels (0 = traversable, 1 = non-traversable) come from the robot-centred
ground-truth semantic map ``local_map`` of each sample: every point is projected onto the
map, and points higher than ``height_factor`` x the robot height (tree canopies, ...) are
ignored. The network is trained with a class-weighted cross-entropy. Samples whose
``lidar_dn`` already carries a fifth column use it as the label instead.

The checkpoint holds ``model_state_dict`` and is what ``model.pointnet_weights`` expects.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sem_navae.config import Config
from sem_navae.data.dataset import load_or_create_split
from sem_navae.data.transforms import normalize_lidar, voxelize_with_intensity
from sem_navae.models.pointnet import PointNetSemSeg, feature_transform_regularizer
from sem_navae.utils.io import get_device, read_pkl

log = logging.getLogger(__name__)

IGNORE_INDEX = -100
NUM_CLASSES = 2
CLASS_NAMES = ("traversable", "non-traversable")


@dataclass
class PointNetPretrainConfig:
    robot_height: float  # [m]
    sensor_height: float  # height of the LiDAR origin above the ground [m]
    height_factor: float = 1.5  # points above height_factor * robot_height are ignored
    epochs: int = 32
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_decay: float = 0.7
    lr_decay_every: int = 10  # epochs
    feature_transform_weight: float = 1e-3

    @property
    def max_z(self) -> float:
        """Highest kept point in the LiDAR frame [m]."""
        return self.height_factor * self.robot_height - self.sensor_height


def label_points(
    points: np.ndarray, local_map: np.ndarray, resolution: float, max_z: float
) -> np.ndarray:
    """Binary traversability labels of ``(N, >=3)`` robot-frame points [m].

    ``local_map[i, j]`` covers ``x in [(i - H/2) res, (i - H/2 + 1) res)`` and likewise
    ``j`` for ``y`` (``docs/dataset_format.md``); non-zero cells are non-traversable.
    Zero-padding points, points above ``max_z`` and points outside the map get
    ``IGNORE_INDEX``.
    """
    h, w = local_map.shape
    i = np.floor(points[:, 0] / resolution + h / 2).astype(np.int64)
    j = np.floor(points[:, 1] / resolution + w / 2).astype(np.int64)
    valid = np.any(points[:, :3] != 0, axis=1) & (points[:, 2] <= max_z)
    valid &= (i >= 0) & (i < h) & (j >= 0) & (j < w)
    labels = np.full(len(points), IGNORE_INDEX, dtype=np.int64)
    labels[valid] = (local_map[i[valid], j[valid]] != 0).astype(np.int64)
    return labels


class PointNetDataset(Dataset):
    """Stacked, normalised scans ``(N_l * N_p, 4)`` with per-point labels ``(N_l * N_p,)``.

    Uses the same index and train/val split file as NaVAE training, so the PointNet never
    sees NaVAE's validation observations.
    """

    def __init__(self, cfg: Config, pcfg: PointNetPretrainConfig, train: bool):
        index = read_pkl(os.path.join(cfg.data.root, cfg.data.index_file))
        self.samples_dir = os.path.join(cfg.data.root, index["root"][1])
        split = load_or_create_split(cfg, [sample for sample, _ in index["ids"]])
        self.ids: List[str] = split["train" if train else "val"]
        self.cfg, self.pcfg = cfg, pcfg

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        points, labels = self.load(i)
        return torch.from_numpy(points), torch.from_numpy(labels)

    def load(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        m = self.cfg.model
        meta = read_pkl(os.path.join(self.samples_dir, self.ids[i]))
        if meta.get("lidar_dn") is not None:
            scans = np.array(meta["lidar_dn"], dtype=np.float32)
        else:
            scans = np.stack(
                [
                    voxelize_with_intensity(s, m.lidar_voxel_size, m.lidar_max_points)
                    for s in meta["lidar"]
                ]
            )
        points = scans.reshape(-1, scans.shape[-1])
        if points.shape[1] >= 5:
            labels = points[:, 4].astype(np.int64)
            labels[~np.any(points[:, :3] != 0, axis=1)] = IGNORE_INDEX
        else:
            labels = label_points(
                points,
                np.asarray(meta["local_map"]),
                self.cfg.data.semantic_map_resolution,
                self.pcfg.max_z,
            )
        xyzi = normalize_lidar(scans[..., :4].copy(), m.lidar_norm_dist).reshape(-1, 4)
        return xyzi.astype(np.float32), labels


def class_weights(dataset: PointNetDataset) -> np.ndarray:
    """Inverse-frequency weights ``(max_freq / freq) ** (1/3)`` over the labelled points."""
    counts = np.zeros(NUM_CLASSES)
    for i in range(len(dataset)):
        labels = dataset.load(i)[1]
        counts += np.bincount(labels[labels != IGNORE_INDEX], minlength=NUM_CLASSES)
    if np.any(counts == 0):
        raise ValueError(f"A class has no labelled points in the training set: {counts}")
    freq = counts / counts.sum()
    return np.power(freq.max() / freq, 1.0 / 3.0)


def evaluate(model: PointNetSemSeg, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Point accuracy and per-class / mean IoU over the labelled points."""
    model.eval()
    inter, union = np.zeros(NUM_CLASSES), np.zeros(NUM_CLASSES)
    correct = seen = 0
    with torch.no_grad():
        for points, labels in loader:
            logprobs, _ = model(points.to(device).permute(0, 2, 1))
            pred = logprobs.argmax(-1).cpu().numpy().ravel()
            target = labels.numpy().ravel()
            keep = target != IGNORE_INDEX
            pred, target = pred[keep], target[keep]
            correct += int((pred == target).sum())
            seen += len(target)
            for c in range(NUM_CLASSES):
                inter[c] += np.sum((pred == c) & (target == c))
                union[c] += np.sum((pred == c) | (target == c))
    iou = inter / np.maximum(union, 1)
    out = {"accuracy": correct / max(seen, 1), "miou": float(iou.mean())}
    out.update({f"iou_{name}": float(v) for name, v in zip(CLASS_NAMES, iou)})
    return out


def pretrain_pointnet(
    cfg: Config, pcfg: PointNetPretrainConfig, out_dir: Optional[str] = None
) -> str:
    """Train the PointNet and return the path of the best checkpoint (by validation mIoU)."""
    out_dir = out_dir or os.path.join(cfg.training.output_dir, cfg.name, "pointnet")
    os.makedirs(out_dir, exist_ok=True)
    device = get_device(cfg.device)

    train_set = PointNetDataset(cfg, pcfg, train=True)
    val_set = PointNetDataset(cfg, pcfg, train=False)
    weights = class_weights(train_set)
    log.info("train %d / val %d samples, class weights %s", len(train_set), len(val_set), weights)
    loader_args = {"batch_size": pcfg.batch_size, "num_workers": cfg.data.num_workers}
    drop_last = len(train_set) > pcfg.batch_size  # BatchNorm needs batches of 2 or more
    train_loader = DataLoader(train_set, shuffle=True, drop_last=drop_last, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = PointNetSemSeg(num_classes=NUM_CLASSES, channel=4).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=pcfg.lr, weight_decay=pcfg.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(optim, pcfg.lr_decay_every, gamma=pcfg.lr_decay)
    weight_t = torch.as_tensor(weights, dtype=torch.float32, device=device)

    best_path, best_miou = os.path.join(out_dir, "pointnet_best.pth"), -1.0
    for epoch in range(pcfg.epochs):
        model.train()
        total, n = 0.0, 0
        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)
            logprobs, trans_feat = model(points.permute(0, 2, 1))
            loss = F.nll_loss(
                logprobs.reshape(-1, NUM_CLASSES),
                labels.reshape(-1),
                weight=weight_t,
                ignore_index=IGNORE_INDEX,
            )
            loss = loss + pcfg.feature_transform_weight * feature_transform_regularizer(trans_feat)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total, n = total + loss.item(), n + 1
        sched.step()

        metrics = evaluate(model, val_loader, device) if len(val_set) else {"miou": 0.0}
        log.info("epoch %d  loss %.4f  %s", epoch, total / max(n, 1), metrics)
        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "class_weights": weights.tolist(),
                    "pretrain_config": vars(pcfg),
                },
                best_path,
            )
    return best_path
