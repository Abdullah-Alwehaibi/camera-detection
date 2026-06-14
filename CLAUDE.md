# Tahakom Camera Detection — Edge Trigger-Line Pipeline

## Hardware
- NVIDIA Jetson Orin NX 16GB, JetPack 5.1.1 (L4T R35.3.1), Python 3.8
- Camera: Basler ace 2 Pro (Sony IMX545)

## Environment
- Python venv at .venv, created with --system-site-packages
  (GStreamer/PyGObject, TensorRT, and OpenCV-with-GStreamer come from
  system packages — do not pip-install replacements for gi, tensorrt,
  or opencv)
- Run `source .venv/bin/activate` before any python/pip commands

## Architecture
- pipeline/gst_pipeline.py     — GStreamer hardware-accelerated ingestion
- pipeline/trigger_line.py     — trigger-line crossing detection
- pipeline/inference.py        — YOLO/TensorRT inference
- pipeline/evidence_capture.py — per-violation evidence capture

## Conventions
- TensorRT .engine files are built ON THIS DEVICE ONLY — not portable
- Test footage/output goes in data/ and logs/ — gitignored
