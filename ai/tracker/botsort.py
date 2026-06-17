"""
ai/tracker/botsort.py

BoT-SORT backend stub — placeholder for appearance-feature-augmented tracking.

BoT-SORT (ByteTrack + Re-ID features + camera motion compensation) improves
ID stability under occlusion vs. vanilla BYTETracker, at the cost of a
ReID model forward pass per frame.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Option A — ultralytics built-in BoT-SORT (simplest; .pt backend only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ultralytics 8.3.0 ships botsort.yaml; simply pass `tracker="botsort.yaml"`
to YOLO.track(). No additional install needed.

Limitation: only available via Yolo11PtDetector (CPU), not the TRT engine
path, because ultralytics' BoT-SORT is embedded in the YOLO inference graph
and can't be called with our custom TRT runner.

To prototype:
  model = YOLO("models/yolo11n.pt")
  results = model.track(frame, tracker="botsort.yaml", persist=True)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Option B — Standalone BoT-SORT with fastreid (TRT compatible)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Full BoT-SORT with appearance (fastreid) + CMC (GSI/ECC).

aarch64 blockers:
  - fastreid: no wheel; must build from source (detectron2 + faiss-gpu deps)
  - faiss-gpu: no aarch64 wheel; requires CUDA-enabled torch (absent here;
    see CLAUDE.md — torch 2.4.1 on this box is CPU-only)

If a CUDA-enabled torch wheel is ever available for cp38-aarch64, the install
sequence would be:
  1. pip install faiss-cpu   (CPU faiss suffices for ReID indexing)
  2. git clone https://github.com/JDAI-CV/fast-reid && pip install -e .
  3. pip install gdown  (to download BoT-SORT ReID weights from Google Drive)
  4. Then implement BotSortBackend using the standalone `botsort` package
     (https://github.com/NirAharon/BoT-SORT) or ultralytics' BotSort class.

Practical recommendation: use BYTETrackerBackend for this deployment. The
5 fps source rate means occlusion events are rare and BYTETracker performs
comparably to BoT-SORT at low frame rates.
"""

from __future__ import annotations

from typing import Dict, List

from ai.tracker.base import Tracker


class BotSortBackend(Tracker):
    """Stub — not yet implemented. Raises at construction time.

    See module docstring for installation instructions.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "BotSortBackend is not yet implemented. "
            "See ai/tracker/botsort.py for installation instructions. "
            "Use BYTETrackerBackend for production."
        )

    def update(self, detections: List[Dict]) -> List[Dict]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
