"""
ai/detector/trafficcamnet.py

TrafficCamNet detector stub — placeholder for NVIDIA TAO / DetectNet_v2 backend.

TrafficCamNet is an NVIDIA-pretrained model specifically for roadway vehicle
detection (car, truck, person, two-wheeler — 4 classes). It ships as a TAO
encrypted engine (.etlt) or a pre-built TensorRT engine (.engine) through the
NVIDIA NGC catalogue.

TAO model reference:
  https://catalog.ngc.nvidia.com/orgs/nvidia/models/tao_trafficcamnet

How to obtain + build on this device:
  1. Install tao-converter (part of DeepStream SDK or standalone NGC download).
     On JetPack 5.1.1 (TRT 8.5.x):
       wget https://developer.nvidia.com/.../tao-converter-jp51.zip
       unzip → tao-converter
  2. Download the .etlt weights:
       ngc registry model download-version nvidia/tao/trafficcamnet:pruned_v1.0.1
  3. Convert to TRT engine:
       tao-converter -k nvidia_tlt -d 3,544,960 \\
           -e models/trafficcamnet.engine \\
           -t fp16 trafficcamnet_pruned_v1.0.1/resnet18_trafficcamnet.etlt
       (544×960 is TrafficCamNet's canonical input; use --batch-size 1)
  4. Point pipeline.yaml detector.engine_path → models/trafficcamnet.engine
     and set detector.backend = "trafficcamnet"

Class mapping (TrafficCamNet output indices → labels):
  0: car
  1: truck
  2: person
  3: two-wheeler

Post-processing differences from YOLO11:
  - Output tensor: (1, 4+num_classes, H/stride, W/stride) — SSD-like anchors
  - NMS: DeepStream-style cluster NMS or cv2.dnn.NMSBoxes
  - BYTETracker input format identical → tracker code unchanged

TODO: implement _TrafficCamNetEngine + TrafficCamNetDetector once:
  (a) tao-converter is available on this device, AND
  (b) the .etlt file is licensed and downloaded from NGC.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ai.detector.base import Detector

TRAFFICCAMNET_CLASSES: Dict[int, str] = {
    0: "car",
    1: "truck",
    2: "person",
    3: "two-wheeler",
}

_VEHICLE_CLASSES = {0, 1, 3}   # car, truck, two-wheeler (exclude person)


class TrafficCamNetDetector(Detector):
    """Stub — not yet implemented. Raises at construction time.

    The interface is declared here so service/main.py and the factory can
    reference it, and so the build_detector() factory error message is clear.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "TrafficCamNetDetector is not yet implemented. "
            "See ai/detector/trafficcamnet.py for build instructions."
        )

    @property
    def class_names(self) -> Dict[int, str]:
        return TRAFFICCAMNET_CLASSES

    @property
    def input_size(self) -> int:
        return 960   # canonical 544×960; width reported as the net dimension

    def track(self, frame) -> List[Dict]:
        raise NotImplementedError

    def warmup(self, n_iters: int = 3) -> float:
        raise NotImplementedError
