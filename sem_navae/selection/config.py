"""Selection parameters (Sec. III-B, values from Sec. IV-A)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from sem_navae.config import _build
from sem_navae.selection.costmap import DEFAULT_PROMPT_TEMPLATE


def _paper_classes() -> Dict[str, float]:
    return {
        "pavement": 0.0,
        "tree": 3.0,
        "grass": 2.0,
        "wall": 3.0,
        "stairs": 3.0,
        "person": 3.0,
        "hole": 3.0,
        "sky": 4.0,
    }


@dataclass
class SelectionConfig:
    classes: Dict[str, float] = field(default_factory=_paper_classes)  # class -> cost (C)
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    unknown_cost: float = 2.0  # C_u
    occlusion_threshold: float = 2.0  # T_occ; pixel costs above it count as unknown
    gamma: float = 0.8  # discount of Eq. 8
    alpha1: float = 2.0  # goal distance weight (Eq. 9)
    alpha2: float = 0.2  # goal heading weight (Eq. 9)
    epsilon: float = 0.15  # switching hysteresis (Sec. III-B4; value not given in the paper)
    mode: str = "cost_to_go"  # cost_to_go | paper | periodic (see update_policy.py)
    min_depth: float = 0.1  # waypoints closer to the camera plane are not projected [m]
    clipseg_model: str = "CIDAS/clipseg-rd64-refined"
    clip_rate_hz: float = 2.5  # f_clip
    camera_optical_frame: bool = False  # True if the camera tf frame is an optical frame


def selection_config_from_dict(data: Optional[Dict[str, Any]]) -> SelectionConfig:
    return _build(SelectionConfig, dict(data or {}), "selection")


def load_selection_config(path: Union[str, Path]) -> SelectionConfig:
    with open(path) as f:
        return selection_config_from_dict(yaml.safe_load(f) or {})
