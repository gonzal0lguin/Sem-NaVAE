"""Semantic cost image I_c from per-class segmentation maps (Eq. 7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

DEFAULT_PROMPT_TEMPLATE = "a photo of a {}"


def build_prompts(classes: Sequence[str], template: str = DEFAULT_PROMPT_TEMPLATE):
    """Fill the paper's prompt template ("a photo of a [class]") for every class."""
    return [template.format(c) for c in classes]


def resize_nearest(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a 2-D array to ``(height, width)``.

    Labels and class costs must never be blended, so linear interpolation is not used.
    """
    h, w = image.shape[:2]
    out_h, out_w = size
    rows = np.minimum((np.arange(out_h) + 0.5) * h / out_h, h - 1).astype(int)
    cols = np.minimum((np.arange(out_w) + 0.5) * w / out_w, w - 1).astype(int)
    return image[rows[:, None], cols[None, :]]


@dataclass
class SemanticCostmap:
    """Dense per-pixel cost ``costs`` and the winning class index ``labels``, both (H, W)."""

    costs: np.ndarray
    labels: np.ndarray

    @classmethod
    def from_scores(
        cls,
        scores: np.ndarray,
        class_costs: Sequence[float],
        image_size: Tuple[int, int],
    ) -> SemanticCostmap:
        """Eq. 7: ``I_c(u, v) = C[argmax_i S_i(u, v)]``.

        Args:
            scores: ``(m, h, w)`` per-class probabilities or logits (argmax is the same).
            class_costs: ``m`` costs, in the same order as ``scores``.
            image_size: ``(width, height)`` of the camera image the paths project onto.
        """
        scores = np.asarray(scores)
        if scores.ndim != 3 or scores.shape[0] != len(class_costs):
            raise ValueError(
                f"scores must be (m, h, w) with m={len(class_costs)}, got {scores.shape}"
            )
        width, height = image_size
        labels = resize_nearest(np.argmax(scores, axis=0), (height, width))
        costs = np.asarray(class_costs, dtype=np.float32)[labels]
        return cls(costs=costs, labels=labels)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.costs.shape
