"""Ultralytics YOLO person segmentation with persistent track IDs."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PersonDetection:
    """One tracked person instance in original-image pixel coordinates."""

    track_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray
    tracker_assigned: bool


class YOLOPersonTracker:
    """Run a COCO person segmenter and ByteTrack on consecutive RGB frames."""

    def __init__(
        self,
        weights: str = "yolo26n-seg.pt",
        confidence: float = 0.25,
        iou: float = 0.7,
        image_size: int = 640,
        tracker: str = "bytetrack.yaml",
        device: str | int | None = None,
    ) -> None:
        try:
            import ultralytics
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Person tracking requires ultralytics. Install the project "
                "dependencies or run the updated Colab setup cell."
            ) from error

        self.weights = weights
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self.tracker = tracker
        self.device = device
        self.ultralytics_version = ultralytics.__version__
        self._model = YOLO(weights)
        self._fallback_track_id = 1_000_000

    @staticmethod
    def _instance_mask(result, index: int, image_shape: tuple[int, int]) -> np.ndarray:
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        if result.masks is not None and index < len(result.masks.xy):
            polygon = np.asarray(result.masks.xy[index], dtype=np.float32)
            if polygon.ndim == 2 and len(polygon) >= 3:
                cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        if mask.any():
            return mask.astype(bool)

        x1, y1, x2, y2 = result.boxes.xyxy[index].detach().cpu().numpy()
        x1 = int(np.clip(np.floor(x1), 0, width - 1))
        x2 = int(np.clip(np.ceil(x2), x1 + 1, width))
        y1 = int(np.clip(np.floor(y1), 0, height - 1))
        y2 = int(np.clip(np.ceil(y2), y1 + 1, height))
        mask[y1:y2, x1:x2] = 1
        return mask.astype(bool)

    def track(self, image_bgr: np.ndarray) -> tuple[list[PersonDetection], dict]:
        """Track people in the next consecutive frame."""
        arguments = {
            "source": image_bgr,
            "persist": True,
            "tracker": self.tracker,
            "classes": [0],  # COCO person
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device is not None:
            arguments["device"] = self.device
        result = self._model.track(**arguments)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return [], dict(result.speed or {})

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        track_ids = (
            result.boxes.id.detach().cpu().numpy().astype(int)
            if result.boxes.id is not None
            else None
        )

        detections: list[PersonDetection] = []
        for index, (box, confidence) in enumerate(zip(boxes, confidences)):
            if track_ids is None:
                track_id = self._fallback_track_id
                self._fallback_track_id += 1
                tracker_assigned = False
            else:
                track_id = int(track_ids[index])
                tracker_assigned = True
            detections.append(
                PersonDetection(
                    track_id=track_id,
                    confidence=float(confidence),
                    bbox_xyxy=tuple(float(value) for value in box),
                    mask=self._instance_mask(result, index, image_bgr.shape[:2]),
                    tracker_assigned=tracker_assigned,
                )
            )
        return detections, dict(result.speed or {})
