"""PointNet semantic-segmentation network (vendored).

Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch (MIT License,
Copyright (c) 2019 benny), via the fork https://github.com/gonzal0lguin/Pointnet_Pointnet2_Pytorch.

Changes w.r.t. upstream ``pointnet_sem_seg.get_model``:
  * 4 input channels (x, y, z, intensity) and 2 classes (traversable / non-traversable);
  * ``forward`` returns the 64x64 feature-transform matrix as the second output, which
    Sem-NaVAE flattens and uses as the global LiDAR feature;
  * debug prints removed, device handling cleaned up.

State-dict keys are identical to the fork, so its checkpoints load unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class STN3d(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(3, dtype=x.dtype, device=x.device).view(1, 9).repeat(batch_size, 1)
        return (x + iden).view(-1, 3, 3)


class STNkd(nn.Module):
    def __init__(self, k: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, -1).repeat(batch_size, 1)
        return (x + iden).view(-1, self.k, self.k)


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat: bool = True, feature_transform: bool = False, channel: int = 3):
        super().__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x: torch.Tensor):
        _, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)
        else:
            trans_feat = None

        point_feat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        x = x.view(-1, 1024, 1).repeat(1, 1, N)
        return torch.cat([x, point_feat], 1), trans, trans_feat


class PointNetSemSeg(nn.Module):
    """Per-point traversability segmentation.

    Input ``(B, C, N)``; returns per-point log-probabilities ``(B, N, num_classes)`` and the
    ``(B, 64, 64)`` feature-transform matrix.
    """

    def __init__(self, num_classes: int = 2, channel: int = 4):
        super().__init__()
        self.k = num_classes
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=channel)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor):
        batch_size, _, n_pts = x.size()
        x, _, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x).transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1).view(batch_size, n_pts, self.k)
        return x, trans_feat


def feature_transform_regularizer(trans: torch.Tensor) -> torch.Tensor:
    """Orthogonality penalty ||A A^T - I|| on the feature transform (PointNet paper)."""
    d = trans.size(1)
    eye = torch.eye(d, device=trans.device, dtype=trans.dtype)[None]
    return torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - eye, dim=(1, 2)))
