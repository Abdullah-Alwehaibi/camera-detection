"""ai/detector/base.py — Detector abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class Detector(ABC):
    """Common interface for all detector + tracker backends.

    Every backend must return the same detection dict shape so the rule
    engine, best-shot selector, and output writers are backend-agnostic.

    Detection dict keys
    -------------------
    track_id : int          — persistent tracker ID
    bbox     : tuple        — (x1, y1, x2, y2) in AI-frame pixel coords
    cls      : int          — COCO (or model-specific) class index
    cls_name : str          — human-readable class name (e.g. "car")
    conf     : float        — detector confidence [0, 1]
    """

    @abstractmethod
    def detect(self, frame) -> List[Dict]:
        """Run detection on one frame; return raw detections without track IDs.

        frame: FramePacket (preferred — carries GPU BGR tensor) or a
               numpy uint8 BGR array (H, W, 3).

        Output dict keys: bbox (x1,y1,x2,y2 px), cls (int),
                          cls_name (str), conf (float).
        Track IDs are assigned by a Tracker (ai/tracker/) in the pipeline.
        """

    @abstractmethod
    def warmup(self, n_iters: int = 3) -> float:
        """Run n_iters synthetic inferences; return mean latency in ms."""

    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]:
        """Mapping of class_id → class_name for all model outputs."""

    @property
    @abstractmethod
    def input_size(self) -> int:
        """Square network input dimension (e.g. 640 or 1280)."""

    def close(self) -> None:
        """Release GPU/memory resources. Called on pipeline shutdown."""
