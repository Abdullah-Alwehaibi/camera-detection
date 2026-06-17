"""ai/tracker/base.py — Tracker abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class Tracker(ABC):
    """Common interface for all tracking backends.

    The tracker receives raw per-frame detections from the detector (no IDs)
    and returns the same dicts enriched with a persistent track_id.  The
    detector and tracker are composed at the pipeline level so either can be
    swapped independently.

    Input detection dict keys (from Detector.detect())
    ---------------------------------------------------
    bbox     : tuple  — (x1, y1, x2, y2) in pixel coords
    cls      : int    — class index
    cls_name : str    — class name
    conf     : float  — detection confidence [0, 1]

    Output dict keys (from Tracker.update())
    ----------------------------------------
    All input keys, plus:
    track_id : int    — persistent tracker ID across frames

    Note: the tracker may return fewer entries than the input if some
    detections fall below the tracker's internal confidence thresholds.
    """

    @abstractmethod
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Assign persistent track IDs to one frame of raw detections.

        Must be called once per frame in order, even when detections is empty,
        so the Kalman filter predictions advance correctly.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal track state (e.g. on source/scene switch)."""
