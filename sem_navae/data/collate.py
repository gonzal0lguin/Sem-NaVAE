"""Batch collation."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from sem_navae.keys import Keys


def collate(samples: List[Dict]) -> Dict:
    """Stack every key into a float tensor, except ``all_paths`` (a list of per-sample lists)."""
    batch: Dict[str, list] = {}
    for sample in samples:
        for key, value in sample.items():
            if key == Keys.all_paths:
                value = [torch.from_numpy(np.asarray(p)).to(torch.float) for p in value]
            else:
                value = torch.from_numpy(np.asarray(value)).to(torch.float)
            if key == Keys.path:
                batch.setdefault(Keys.last_poses, []).append(value[-1, :])
            batch.setdefault(key, []).append(value)
    return {k: v if k == Keys.all_paths else torch.stack(v, dim=0) for k, v in batch.items()}
