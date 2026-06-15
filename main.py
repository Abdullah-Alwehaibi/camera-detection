#!/usr/bin/env python3
"""Edge trigger-zone pipeline.

FrameSource -> detection/tracking -> trigger-zone check -> evidence capture + output sink.
"""

import torch  # noqa: F401  -- import-order workaround, see pipeline/inference.py

import datetime
import json
import logging
import time
from pathlib import Path

from pipeline.evidence_capture import draw_databar, draw_detections, save_evidence
from pipeline.gst_pipeline import FrameSource
from pipeline.inference import VehicleDetector
from pipeline.output_sink import OutputSink
from pipeline.trigger_line import load_zones

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config" / "pipeline.json"

LOG_INTERVAL_SEC = 5.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("main")


def bbox_bottom_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def bbox_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def main():
    config = json.loads(CONFIG_PATH.read_text())

    source_cfg = config["source"]
    model_cfg = config["model"]
    zones = load_zones(REPO_ROOT / config["trigger_zones_config"])
    evidence_dir = REPO_ROOT / config["evidence_dir"]
    location = config.get("location", "")

    output_cfg = config.get("output_sink", {})
    events_socket = output_cfg.get("events_socket", "/tmp/ai_events.sock")
    fallback_log = REPO_ROOT / output_cfg.get("fallback_log", "logs/output_events.jsonl")

    log.info(
        "source mode=%s %dx%d @ %dfps",
        source_cfg["mode"], source_cfg["width"], source_cfg["height"], source_cfg.get("fps", 30),
    )
    log.info("trigger zones: %s", [(z.name, z.type, z.points) for z in zones])

    detector = VehicleDetector(
        model_path=str(REPO_ROOT / model_cfg["path"]),
        vehicle_classes=model_cfg["vehicle_classes"],
        confidence=model_cfg.get("confidence", 0.4),
    )

    source = FrameSource(source_cfg)
    sink = OutputSink(socket_path=events_socket, fallback_log=fallback_log)

    frame_count = 0
    start_time = time.monotonic()
    last_log_time = start_time

    with source, sink:
        for frame in source.frames():
            frame_count += 1
            capture_time = datetime.datetime.now()

            detections = detector.track(frame)
            for det in detections:
                point = bbox_bottom_center(det["bbox"])
                for zone in zones:
                    if zone.evaluate(point, det["track_id"]):
                        annotated = draw_detections(frame, detections, highlight_track_id=det["track_id"])
                        draw_databar(annotated, capture_time, location, det["cls_name"])
                        metadata = {
                            "timestamp": capture_time,
                            "track_id": det["track_id"],
                            "bbox": det["bbox"],
                            "class": det["cls_name"],
                            "image": annotated,
                        }
                        path = save_evidence(metadata, output_dir=evidence_dir)

                        sink.send_event({
                            "type": "vehicle",
                            "confidence": det["conf"],
                            "bbox": bbox_to_xywh(det["bbox"]),
                            "track_id": det["track_id"],
                            "timestamp_ns": time.monotonic_ns(),
                            "class": det["cls_name"],
                            "zone_id": zone.name,
                            "crop_path": str(path),
                            "timestamp": capture_time.isoformat(),
                        })

                        log.info(
                            "CROSSING zone=%s track_id=%d class=%s conf=%.2f -> %s",
                            zone.name, det["track_id"], det["cls_name"], det["conf"], path,
                        )

            now = time.monotonic()
            if now - last_log_time >= LOG_INTERVAL_SEC:
                elapsed = now - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                log.info("frames=%d elapsed=%.1fs fps=%.2f", frame_count, elapsed, fps)
                last_log_time = now


if __name__ == "__main__":
    main()
