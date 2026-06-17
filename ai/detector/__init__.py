"""ai/detector — detector + tracker backends."""

# torch must be the first real import in any process entrypoint (aarch64
# import-order workaround; see CLAUDE.md). Importing this package is safe
# because yolo11_trt.py begins with `import torch`.
from ai.detector.base import Detector
from ai.detector.yolo11_trt import build_yolo11_detector
from ai.detector.trafficcamnet import TrafficCamNetDetector


def build_detector(config: dict, cuda_ctx=None) -> Detector:
    """Factory: returns the right Detector for pipeline.yaml detector.backend."""
    backend = config.get("backend", "yolo11")
    if backend == "yolo11":
        return build_yolo11_detector(config, cuda_ctx=cuda_ctx)
    if backend == "trafficcamnet":
        return TrafficCamNetDetector(config, cuda_ctx=cuda_ctx)
    raise ValueError(f"Unknown detector backend: {backend!r}")


__all__ = ["Detector", "build_detector", "build_yolo11_detector", "TrafficCamNetDetector"]
