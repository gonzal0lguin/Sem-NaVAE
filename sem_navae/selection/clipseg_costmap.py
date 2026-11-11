"""CLIPSeg wrapper producing the semantic cost map (Sec. III-B2). Needs ``transformers``."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from sem_navae.selection.costmap import DEFAULT_PROMPT_TEMPLATE, SemanticCostmap, build_prompts


class ClipSegCostmapper:
    """Zero-shot segmentation with the off-the-shelf CLIPSeg model (no fine-tuning)."""

    def __init__(
        self,
        classes: Mapping[str, float],
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        model_name: str = "CIDAS/clipseg-rd64-refined",
        device: Optional[str] = None,
    ):
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self._torch = torch
        self.class_names = list(classes.keys())
        self.class_costs = [float(classes[c]) for c in self.class_names]
        self.prompts = build_prompts(self.class_names, prompt_template)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def __call__(self, image_rgb: np.ndarray) -> SemanticCostmap:
        """``(H, W, 3)`` uint8 RGB image -> cost map at the image resolution (Eq. 7)."""
        inputs = self.processor(
            text=self.prompts,
            images=[image_rgb] * len(self.prompts),
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self.model(**inputs).logits  # (m, h, w), or (h, w) when m == 1
        logits = logits.reshape(len(self.prompts), *logits.shape[-2:]).cpu().numpy()
        height, width = image_rgb.shape[:2]
        return SemanticCostmap.from_scores(logits, self.class_costs, (width, height))
