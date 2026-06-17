"""ai/tracker — tracker backends."""

from ai.tracker.base import Tracker
from ai.tracker.bytetracker import BYTETrackerBackend
from ai.tracker.botsort import BotSortBackend


def build_tracker(config: dict) -> Tracker:
    """Build tracker from pipeline.yaml tracker section.

    config keys: backend ("bytetracker" | "botsort"), fps
    """
    backend = config.get("backend", "bytetracker")
    fps     = int(config.get("fps", 15))

    if backend == "bytetracker":
        return BYTETrackerBackend(fps=fps)
    if backend == "botsort":
        return BotSortBackend(fps=fps)
    raise ValueError(f"Unknown tracker backend: {backend!r}")


__all__ = ["Tracker", "BYTETrackerBackend", "BotSortBackend", "build_tracker"]
