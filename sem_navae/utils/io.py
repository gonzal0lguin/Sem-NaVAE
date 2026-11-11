"""Device and file helpers."""

from __future__ import annotations

import pickle
from typing import Any, Union

import torch


def get_device(device: Union[torch.device, str] = "cuda") -> torch.device:
    """Return ``device``, falling back to CPU when CUDA is unavailable."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def to_device(x: Any, device: Union[torch.device, str]) -> Any:
    """Recursively move tensors in lists / tuples / dicts to ``device``."""
    if isinstance(x, list):
        return [to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


def release_cuda(x: Any) -> Any:
    """Recursively convert tensors to Python scalars (single element) or NumPy arrays."""
    if isinstance(x, list):
        return [release_cuda(v) for v in x]
    if isinstance(x, tuple):
        return tuple(release_cuda(v) for v in x)
    if isinstance(x, dict):
        return {k: release_cuda(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.detach().cpu().numpy()
    return x


def read_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
