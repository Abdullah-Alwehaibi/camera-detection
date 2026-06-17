"""
service/main.py — Production orchestration loop.

Reads NV12 frames from the grabber (or a synthetic source), runs
detection + tracking, evaluates rules, and writes events to both the
boolean AF_UNIX socket (ONVIF path) and the rich sidecar.

Import order: torch MUST come before cv2/gi (aarch64 libgomp constraint).
See CLAUDE.md — "Import-order gotcha (aarch64)".
"""

import torch   # ← MUST BE FIRST on aarch64

import argparse
import datetime
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import yaml

# Repo root on sys.path so that running from anywhere works
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ai.bestshot import BestShotSelector
from ai.rules.engine import RuleEngine, RuleEvent
from ai.rules.loader import RulesLoader
from integration.contract import DEFAULT_SCALE_4K_X, DEFAULT_SCALE_4K_Y
from integration.frame_reader import FrameReader
from integration.outputs import build_outputs
from service import metrics


# ── Logging ───────────────────────────────────────────────────────────────────

log = logging.getLogger("camera-detection")


def _setup_logging(level_str: str, log_dir: Path) -> None:
    level = getattr(logging, level_str.upper(), logging.INFO)
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        handlers.append(fh)
    except OSError:
        pass
    logging.basicConfig(
        level   = level,
        format  = "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S",
        handlers= handlers,
    )


# ── Synthetic frame packet for --fake-detections mode ────────────────────────

@dataclass
class _FakeFramePacket:
    """FramePacket-compatible object for --fake-detections scenarios.

    Holds correct mono/real timestamps but a blank BGR frame (no CUDA).
    The rule engine and best-shot selector both work on detections + timestamps,
    so the blank frame only affects evidence JPEG quality (irrelevant for tests).
    """
    frame_id:   int
    ts_mono_ns: int
    ts_real_ns: int
    width:      int = 1920
    height:     int = 1080
    d_bgr:      object = None

    @property
    def cpu_bgr(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


def _fake_frame_gen(n_frames: int, fps: int = 5) -> Iterator[_FakeFramePacket]:
    """Yield exactly n_frames synthetic packets paced to fps."""
    mono_start  = time.monotonic_ns()
    real_start  = time.time_ns()
    offset_ns   = real_start - mono_start
    interval_ns = int(1e9 / fps)

    for i in range(n_frames):
        ts_mono = mono_start + i * interval_ns
        yield _FakeFramePacket(
            frame_id   = i + 1,
            ts_mono_ns = ts_mono,
            ts_real_ns = ts_mono + offset_ns,
        )
        elapsed = time.monotonic_ns() - mono_start
        target  = (i + 1) * interval_ns
        if target > elapsed:
            time.sleep((target - elapsed) / 1e9)


# ── Event builders ────────────────────────────────────────────────────────────

def _boolean_event(evt: RuleEvent, capture_meta: dict) -> dict:
    """Build a §3.2 boolean event dict from a RuleEvent + bestshot metadata."""
    return {
        "type":           evt.event_type,
        "confidence":     evt.confidence,
        "track_id":       evt.track_id,
        "class":          evt.cls_name,
        "zone_id":        evt.rule_id,
        "bbox":           evt.bbox_1080,
        "bbox_4k":        capture_meta.get("bbox_4k", evt.bbox_4k),
        "ts_monotonic_ns": evt.ts_mono_ns,
        "ts_realtime_ns":  evt.ts_real_ns,
        "timestamp":       datetime.datetime.fromtimestamp(
                               evt.ts_real_ns / 1e9
                           ).isoformat(),
        "crop_path":       capture_meta.get("crop_path"),
    }


def _sidecar_record(evt: RuleEvent, capture_meta: dict) -> dict:
    """Build a rich sidecar record from a RuleEvent + bestshot metadata."""
    return {
        "schema_version":  "1.0",
        "event":           evt.event_type,
        "rule_id":         evt.rule_id,
        "rule_type":       evt.rule_type,
        "track_id":        evt.track_id,
        "class":           "vehicle",
        "subclass":        evt.cls_name,
        "confidence":      evt.confidence,
        "bbox_1080":       evt.bbox_1080,
        "bbox_4k":         capture_meta.get("bbox_4k", evt.bbox_4k),
        "direction":       evt.direction_sign,
        "lane":            None,
        "in_capture_zone": evt.in_capture_zone,
        "snapshot": {
            "mode":      capture_meta.get("source", "backend_pull"),
            "ref":       capture_meta.get("ts_real_ns"),
            "jpeg_path": capture_meta.get("crop_path") or capture_meta.get("jpeg_path"),
        },
        "ts_monotonic_ns": evt.ts_mono_ns,
        "ts_realtime_ns":  evt.ts_real_ns,
    }


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run(cfg: dict, args: argparse.Namespace) -> int:
    """Build all components and run the main frame loop. Returns exit code."""
    dry_run  = args.dry_run or cfg.get("dry_run", False)
    fw       = cfg.get("expected_caps", {}).get("width",  1920)
    fh       = cfg.get("expected_caps", {}).get("height", 1080)
    sx       = cfg.get("scale_4k", {}).get("x", DEFAULT_SCALE_4K_X)
    sy       = cfg.get("scale_4k", {}).get("y", DEFAULT_SCALE_4K_Y)
    log_dir  = Path(cfg.get("log_dir", "logs"))

    if dry_run:
        log.warning("DRY-RUN mode: events will be logged but NOT sent to sockets")

    # ── Rules + capture zones (with hot-reload) ───────────────────────────────
    rules_path = cfg.get("rules_file", "config/rules.json")
    loader     = RulesLoader(rules_path)
    loader.load()
    loader.start_watching()
    log.info("Rules loaded from %s — watching for hot-reload", rules_path)

    engine = RuleEngine(
        rules_source         = lambda: loader.rules,
        capture_zones_source = lambda: loader.capture_zones,
        frame_w    = fw,
        frame_h    = fh,
        scale_4k_x = sx,
        scale_4k_y = sy,
    )

    # ── Best-shot selector ────────────────────────────────────────────────────
    snap_cfg = cfg.get("snapshot", {})
    bestshot = BestShotSelector(
        mode            = snap_cfg.get("mode", "backend_pull"),
        snapshot_socket = snap_cfg.get("snapshot_socket", "/tmp/ai_snapshot.sock"),
        evidence_dir    = cfg.get("evidence_dir", "data/snapshots"),
        scale_4k_x      = sx,
        scale_4k_y      = sy,
        location        = cfg.get("location", "Unknown"),
        dry_run         = dry_run,
    )

    # ── CUDA context (skip in fake-detections mode — no GPU inference needed) ─
    cuda_ctx = None
    if not args.fake_detections:
        import pycuda.driver as cuda
        cuda.init()
        cuda_ctx = cuda.Device(0).retain_primary_context()
        log.info("CUDA context: %s", cuda.Device(0).name())

    # ── Detector + tracker OR fake detection source ───────────────────────────
    fake_src = None
    detector = None
    tracker  = None

    if args.fake_detections:
        from tools.fake_grabber import FakeDetectionSource
        fake_src = FakeDetectionSource(args.fake_detections, loop=args.loop)
        log.info("FAKE DETECTIONS: scenario=%r  n_frames=%d",
                 args.fake_detections, fake_src.n_frames)
    else:
        from ai.detector import build_detector
        from ai.tracker  import build_tracker
        detector = build_detector(cfg.get("detector", {}), cuda_ctx=cuda_ctx)
        tracker  = build_tracker(cfg.get("tracker",  {}))
        log.info("Detector + tracker built")

    # ── Output writers ────────────────────────────────────────────────────────
    boolean_writer, sidecar_writer = build_outputs(cfg, dry_run=dry_run)

    # ── Prometheus metrics ────────────────────────────────────────────────────
    metrics_port = cfg.get("metrics_port", 9108)
    try:
        metrics.start(metrics_port)
        log.info("Prometheus metrics at http://localhost:%d/metrics", metrics_port)
    except OSError as exc:
        log.warning("Could not start metrics server on port %d: %s", metrics_port, exc)

    # ── Frame source ──────────────────────────────────────────────────────────
    frame_reader: Optional[FrameReader] = None

    if args.fake_detections:
        frame_gen = _fake_frame_gen(fake_src.n_frames, fps=5)
    else:
        frame_reader = FrameReader(cfg, cuda_ctx=cuda_ctx)
        frame_gen    = frame_reader.frames()

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    _stop = threading.Event()

    def _on_signal(signum, _):
        log.info("Signal %d received — requesting stop", signum)
        _stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT,  _on_signal)

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_count   = 0
    _loop_state   = argparse.Namespace()   # mutable namespace for per-loop bookkeeping
    log.info("Pipeline started (dry_run=%s)", dry_run)

    try:
        for fp in frame_gen:
            if _stop.is_set():
                break

            t0 = time.perf_counter()

            if fake_src is not None:
                tracked = fake_src.next_frame()
                if tracked is None:
                    log.info("Scenario exhausted — stopping")
                    break
            else:
                raw_dets = detector.detect(fp)
                tracked  = tracker.update(raw_dets)

            inf_sec = time.perf_counter() - t0
            frame_count += 1

            # Metrics
            metrics.frames_total.inc()
            metrics.inference_duration.observe(inf_sec)
            metrics.active_tracks.set(len(tracked))
            if frame_reader:
                fr_metrics = frame_reader.metrics
                metrics.pipeline_fps.set(fr_metrics.fps_measured)
                # Mirror FrameReader drop counter into the Prometheus counter.
                # frames_dropped tracks cumulative drops; only increment the delta.
                _new_drops = fr_metrics.frames_dropped
                if not hasattr(_loop_state, "last_drops"):
                    _loop_state.last_drops = 0
                delta = _new_drops - _loop_state.last_drops
                if delta > 0:
                    metrics.frames_dropped.inc(delta)
                _loop_state.last_drops = _new_drops

            # Best-shot update (every frame, before rule evaluation)
            bestshot.update(
                tracked_dets  = tracked,
                frame_bgr     = fp.cpu_bgr,
                capture_zones = loader.capture_zones,
                ts_mono_ns    = fp.ts_mono_ns,
                ts_real_ns    = fp.ts_real_ns,
                frame_w       = fw,
                frame_h       = fh,
            )

            # Rule evaluation
            rule_events = engine.evaluate(tracked, fp.ts_mono_ns, fp.ts_real_ns)

            for evt in rule_events:
                metrics.rule_events_total.labels(
                    zone_id    = evt.rule_id,
                    event_type = evt.event_type,
                ).inc()

                capture_meta = bestshot.dispatch(evt)
                metrics.bestshot_dispatches_total.labels(
                    source = capture_meta.get("source", "unknown"),
                ).inc()

                bool_ev = _boolean_event(evt, capture_meta)
                try:
                    boolean_writer.send(bool_ev)
                except Exception as exc:
                    metrics.event_send_errors_total.inc()
                    log.error("BooleanEventWriter.send error: %s", exc)

                sidecar_writer.send(_sidecar_record(evt, capture_meta))

                log.info(
                    "EVENT rule=%s type=%s track=%d class=%s conf=%.2f",
                    evt.rule_id, evt.event_type, evt.track_id,
                    evt.cls_name, evt.confidence,
                )

            # Periodic heartbeat log
            if frame_count % 100 == 0:
                fps = frame_reader.metrics.fps_measured if frame_reader else 0.0
                log.info(
                    "heartbeat frame=%d fps=%.1f tracks=%d inf=%.1fms",
                    frame_count, fps, len(tracked), inf_sec * 1000,
                )

    finally:
        log.info("Cleanup (frame_count=%d)", frame_count)
        boolean_writer.close()
        sidecar_writer.close()
        if detector is not None:
            try:
                detector.close()
            except Exception:
                pass
        if frame_reader is not None:
            frame_reader.stop()

    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tahakom camera-detection production pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--config", default="config/pipeline.yaml",
        help="Path to pipeline.yaml (default: config/pipeline.yaml)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Log events but do not send to sockets (overrides pipeline.yaml dry_run)",
    )
    p.add_argument(
        "--fake-detections", metavar="SCENARIO",
        choices=["normal", "line_jump", "occlusion", "id_reuse"],
        help=(
            "Bypass the real detector/tracker and use scripted trajectories.\n"
            "Useful for testing rule engine + output writers without a camera.\n"
            "Scenarios: normal, line_jump, occlusion, id_reuse"
        ),
    )
    p.add_argument(
        "--loop", action="store_true",
        help="Loop the fake-detections scenario (only with --fake-detections)",
    )
    p.add_argument(
        "--log-level", default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log_level from pipeline.yaml",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load config first to get log_level + log_dir before _setup_logging
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        sys.exit(f"Config not found: {args.config}")
    except yaml.YAMLError as exc:
        sys.exit(f"Config parse error: {exc}")

    log_level = args.log_level or cfg.get("log_level", "INFO")
    log_dir   = Path(cfg.get("log_dir", "logs"))
    _setup_logging(log_level, log_dir)

    log.info("Camera-detection service starting  config=%s", args.config)
    log.info(
        "torch=%s  device=%s",
        torch.__version__,
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    sys.exit(run(cfg, args))


if __name__ == "__main__":
    main()
