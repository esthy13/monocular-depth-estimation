"""Pluggable depth model interface.

Add new models by subclassing DepthModel and registering them in build_depth_model().
Currently supported:
  - 'depth_anything_v2': Depth Anything V2 via HuggingFace transformers (relative depth).

Planned / easy to add:
  - Metric3Dv2
  - UniDepthV2
  - DAC
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DepthModel(ABC):
    """Abstract interface for monocular depth estimation models."""

    @abstractmethod
    def load(self) -> None:
        """Download weights (if needed) and load them into memory."""
        ...

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth for a single BGR uint8 image (H, W, 3).

        Returns a float32 depth array (H, W).
        Values are relative (arbitrary scale) unless the model provides metric depth.
        """
        ...


# ---------------------------------------------------------------------------
# Depth Anything V2
# ---------------------------------------------------------------------------

class DepthAnythingV2(DepthModel):
    """Depth Anything V2 loaded via the HuggingFace transformers library.

    Model page: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf

    Args:
        encoder: one of 'small', 'base', 'large'.
        device: 'cpu', 'cuda', or 'mps'. Auto-detected if None.
    """

    _MODEL_IDS = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base":  "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    def __init__(self, encoder: str = "small", device: str | None = None) -> None:
        if encoder not in self._MODEL_IDS:
            raise ValueError(
                f"encoder must be one of {list(self._MODEL_IDS)}; got {encoder!r}"
            )
        self.encoder = encoder
        self.model_id = self._MODEL_IDS[encoder]
        self._model = None
        self._processor = None

        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        print(f"Loading {self.model_id} on {self.device} ...")
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self._model.to(self.device).eval()
        print("Model loaded.")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict depth for a BGR uint8 image. Returns float32 (H, W)."""
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load() before predict().")

        import torch
        import torch.nn.functional as F
        import cv2
        from PIL import Image

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            raw_depth = outputs.predicted_depth  # shape: (1, H', W')

        # Resize back to original image resolution
        depth = F.interpolate(
            raw_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy().astype(np.float32)

        return depth


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_depth_model(name: str = "depth_anything_v2", **kwargs) -> DepthModel:
    """Instantiate a depth model by name.

    Args:
        name: model identifier string.
        **kwargs: forwarded to the model's __init__.

    Supported names:
        'depth_anything_v2'  — Depth Anything V2 (HuggingFace, relative depth)
    """
    registry: dict[str, type[DepthModel]] = {
        "depth_anything_v2": DepthAnythingV2,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown depth model: {name!r}. Available: {list(registry)}"
        )
    return registry[name](**kwargs)
