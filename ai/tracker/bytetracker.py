"""
ai/tracker/bytetracker.py

BYTETracker backend for the Tracker abstraction.

BYTETracker (ultralytics built-in) uses a two-stage matching strategy:
  1. High-confidence detections matched first (IoU + Kalman prediction).
  2. Low-confidence detections matched against remaining tracks (rescues
     briefly occluded vehicles from being dropped).

This backend was extracted from pipeline/inference.py (VehicleDetector) and
from the Slice 5 Yolo11TrtDetector — the tracker now lives here independently
of any detector implementation.

Frame rate matters: BYTETracker uses it to set the Kalman velocity prior and
the maximum age (frames before a lost track is removed). Pass the source FPS
from pipeline.yaml (not the detection FPS, which may be lower).
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from ultralytics.trackers import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, ops, yaml_load
from ultralytics.utils.checks import check_yaml

from ai.tracker.base import Tracker

log = logging.getLogger(__name__)


class _DetInput:
    """Minimal result-like shim for BYTETracker.update()."""
    __slots__ = ("xywh", "conf", "cls")

    def __init__(self, xywh, conf, cls):
        self.xywh = xywh
        self.conf = conf
        self.cls  = cls


class BYTETrackerBackend(Tracker):
    """BYTETracker wrapper that conforms to the Tracker interface.

    Input/output format follows Tracker base class; see its docstring.
    The det_idx field returned by BYTETracker (column 7 of each tracked row)
    is used to recover the original cls_name from the input dets list without
    requiring the tracker to know about class names itself.
    """

    def __init__(self, fps: int = 15) -> None:
        cfg = IterableSimpleNamespace(**yaml_load(check_yaml("bytetrack.yaml")))
        self._cfg     = cfg
        self._fps     = fps
        self._tracker = BYTETracker(args=cfg, frame_rate=fps)
        log.info("BYTETrackerBackend: fps=%d", fps)

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Assign track IDs. Returns empty list when no tracks survive matching."""
        if not detections:
            # Advance Kalman predictions for existing tracks even with no dets
            self._tracker.update(_DetInput(
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,),   dtype=np.float32),
                np.zeros((0,),   dtype=np.float32),
            ))
            return []

        boxes_xyxy = np.array([d["bbox"] for d in detections], dtype=np.float32)
        confs      = np.array([d["conf"] for d in detections], dtype=np.float32)
        clss       = np.array([d["cls"]  for d in detections], dtype=np.float32)
        boxes_xywh = ops.xyxy2xywh(boxes_xyxy)

        tracked = self._tracker.update(_DetInput(boxes_xywh, confs, clss))

        result: List[Dict] = []
        for row in tracked:
            x1, y1, x2, y2, track_id, score, cls_idx, det_idx = row
            det_idx = int(det_idx)
            orig    = detections[det_idx] if det_idx < len(detections) else {}
            result.append({
                "track_id": int(track_id),
                "bbox":     (float(x1), float(y1), float(x2), float(y2)),
                "cls":      int(cls_idx),
                "cls_name": orig.get("cls_name", str(int(cls_idx))),
                "conf":     float(score),
            })
        return result

    def reset(self) -> None:
        """Re-create the tracker to clear all Kalman state and track IDs."""
        self._tracker = BYTETracker(args=self._cfg, frame_rate=self._fps)
        log.info("BYTETrackerBackend: state reset")


def build_bytetracker(config: dict) -> BYTETrackerBackend:
    """Build from tracker config dict (pipeline.yaml tracker section)."""
    return BYTETrackerBackend(fps=int(config.get("fps", 15)))
