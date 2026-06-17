#!/usr/bin/env python3
"""
tools/fake_grabber.py — Synthetic frame + detection source for pipeline testing.

Two roles:

  1. FakeDetectionSource — yields scripted detection dicts (same format as
     Tracker.update() output) per frame. Used by service/main.py via the
     --fake-detections flag to test the rule engine, best-shot selector, and
     output writers without a camera or YOLO running.

  2. Visual frame renderer + optional GStreamer shmsink publisher — renders
     the scenario as a BGR frame (colored rectangles on gray background) and
     optionally publishes NV12 frames to /tmp/ai_frames.sock so the frame
     reader receives real (if synthetic) video.

Scenarios
---------
  normal      3 vehicles cross the trigger line smoothly at different times.
              Tests the basic line-crossing event chain end-to-end.

  line_jump   1 vehicle jumps from above to below the line in a SINGLE frame
              (skipping the line geometrically). A per-frame point-side test
              would MISS this; the segment-intersection test in RuleEngine
              CATCHES it. Key correctness test for Slice 4.

  occlusion   1 vehicle disappears for 8 frames (2 track-ID-reuse-safe
              seconds at 5fps < _TRACK_EXPIRE_SEC=5s), then reappears on the
              far side of the line. The rule engine retains prev_pos across
              the gap, so the crossing IS detected during the occlusion window.

  id_reuse    Vehicle A crosses the line then leaves. After 26 frames (5.2s
              > _TRACK_EXPIRE_SEC=5s), Vehicle B arrives at the same position
              and is assigned the same track_id=1 by BYTETracker. The rule
              engine detects the stale state (last_seen > expire_sec), resets
              it, and allows Vehicle B to trigger a fresh crossing.

Usage
-----
  # Just print scenario frames (no camera / GStreamer needed):
  python tools/fake_grabber.py --scenario line_jump --list

  # Save rendered frames to disk for visual inspection:
  python tools/fake_grabber.py --scenario normal --save-frames /tmp/frames/

  # Publish NV12 frames via GStreamer shmsink (requires gi + GStreamer):
  python tools/fake_grabber.py --scenario normal --publish --socket /tmp/ai_frames.sock

  # Use FakeDetectionSource in a test pipeline (see service/main.py --help):
  python service/main.py --fake-detections line_jump --dry-run
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

FRAME_W = 1920
FRAME_H = 1080
FPS_DEFAULT = 5

# Trigger line at 50% of frame height (matches config/trigger_zones.json default)
TRIGGER_Y = FRAME_H // 2   # 540

VEHICLE_CLS = 2   # COCO car
VEHICLE_CLS_NAME = "car"
VEHICLE_CONF = 0.90

# Visual appearance
BG_COLOR  = (80, 80, 80)       # dark gray background
VEH_COLOR = (0, 160, 255)      # BGR orange — visible vehicle box
LINE_COLOR = (0, 0, 255)       # red trigger line
TEXT_COLOR = (220, 220, 220)   # light gray text


# ── Scripted trajectory helpers ───────────────────────────────────────────────

def _bbox_at(x1: int, x2: int, y_bottom: int, height: int = 80) -> Tuple:
    """Return (x1, y_top, x2, y_bottom) for a vehicle at the given bottom-y."""
    return (x1, y_bottom - height, x2, y_bottom)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smooth_path(x1: int, x2: int, y_start: int, y_end: int,
                 n_frames: int, height: int = 80) -> List[Optional[Tuple]]:
    """Interpolate bbox bottom-y from y_start to y_end over n_frames."""
    out = []
    for i in range(n_frames):
        t   = i / max(n_frames - 1, 1)
        y_b = round(_lerp(y_start, y_end, t))
        out.append(_bbox_at(x1, x2, y_b, height))
    return out


# ── Scenario definitions ──────────────────────────────────────────────────────

@dataclass
class _Vehicle:
    """A single vehicle's scripted trajectory."""
    track_id: int
    frames: List[Optional[Tuple]]  # None = absent; tuple = (x1,y1,x2,y2)
    color: Tuple = (0, 160, 255)


def _scenario_normal(n_frames: int = 80) -> List[_Vehicle]:
    """Three vehicles crossing the trigger line at staggered frames."""
    v1_path = _smooth_path(100, 300, 60, 1030, n_frames, 80)
    v2_path = (
        [None] * 10
        + _smooth_path(800, 1000, 60, 1030, n_frames - 10, 80)
    )
    v3_path = (
        [None] * 20
        + _smooth_path(1500, 1700, 60, 1030, n_frames - 20, 80)
    )
    return [
        _Vehicle(1, v1_path, (0, 160, 255)),
        _Vehicle(2, v2_path, (0, 210, 0)),
        _Vehicle(3, v3_path, (255, 80, 0)),
    ]


def _scenario_line_jump(n_frames: int = 35) -> List[_Vehicle]:
    """Vehicle approaches the line then JUMPS over it in a single frame.

    Frames 0–14 : bottom-y rises from 60 to 490 (always ABOVE y=540).
    Frame 15    : bottom-y teleports to 690 (well BELOW y=540).
    Frames 16–34: continues down to 1050.

    The segment from frame 14's position to frame 15's position crosses
    the trigger line even though neither frame's position is AT the line.
    """
    approach = _smooth_path(600, 900, 60, 490, 15, 100)  # frames 0–14
    jump_pos = [_bbox_at(600, 900, 690, 100)]             # frame 15 — the jump
    depart   = _smooth_path(600, 900, 690, 1050, 19, 100) # frames 16–34

    path = approach + jump_pos + depart
    return [_Vehicle(1, path, (0, 80, 255))]


def _scenario_occlusion(n_frames: int = 50) -> List[_Vehicle]:
    """Vehicle vanishes for 8 frames, then reappears on the other side.

    Gap = 8 frames × 0.2 s/frame = 1.6 s < _TRACK_EXPIRE_SEC = 5 s,
    so the rule engine RETAINS prev_pos and detects the crossing across the gap.
    """
    approach = _smooth_path(550, 850, 60, 520, 13, 100)    # frames 0–12 (above line)
    gap      = [None] * 8                                    # frames 13–20 (absent)
    depart   = _smooth_path(550, 850, 570, 1020, 22, 100)   # frames 21–42 (below)
    rest     = [None] * (n_frames - len(approach) - len(gap) - len(depart))

    return [_Vehicle(1, approach + gap + depart + rest, (0, 200, 160))]


def _scenario_id_reuse() -> List[_Vehicle]:
    """Vehicle A leaves; after 5.2 s, Vehicle B appears at the same position.

    Vehicle A: frames 0–20  (crosses at ~frame 12, exits bottom at frame 20).
    Gap      : frames 21–45 = 25 frames × 0.2 s = 5.0 s = _TRACK_EXPIRE_SEC.
               At 5.0 s the rule-engine state expires → fresh state on next frame.
    Vehicle B: frames 46–70 (same x-strip as A; same track_id=1 from BYTETracker).
               The rule engine detects ID reuse (last_seen > expire_sec) and fires
               a NEW crossing event even though track_id=1 was already seen.

    Note: BYTETracker's ID reassignment to 1 is likely but not guaranteed; the
    scenario also works if B gets track_id=2 (still fires, no reuse needed).
    """
    a_path = _smooth_path(700, 1000, 60, 1050, 21, 100)    # frames 0–20
    gap    = [None] * 25                                     # frames 21–45 (5.0 s gap)
    b_path = _smooth_path(700, 1000, 60, 1050, 25, 100)    # frames 46–70

    # Assign track_id=1 to both vehicles to force a reuse situation
    # (in production the tracker assigns IDs; here we pre-script them)
    return [
        _Vehicle(1, a_path + gap + [None] * 25),
        _Vehicle(1, [None] * 46 + b_path),
    ]


_SCENARIOS = {
    "normal":    _scenario_normal,
    "line_jump": _scenario_line_jump,
    "occlusion": _scenario_occlusion,
    "id_reuse":  _scenario_id_reuse,
}

ALL_SCENARIOS = list(_SCENARIOS.keys())


# ── Convert vehicles → detection dicts ───────────────────────────────────────

def _frame_detections(vehicles: List[_Vehicle], frame_idx: int) -> List[Dict]:
    """Return detection dicts for all visible vehicles at frame_idx."""
    seen_ids = set()
    dets = []
    for v in vehicles:
        if frame_idx >= len(v.frames):
            continue
        bbox = v.frames[frame_idx]
        if bbox is None:
            continue
        # In the id_reuse scenario both vehicles may share track_id=1; deduplicate
        # by taking the first occurrence (whichever vehicle is visible).
        if v.track_id in seen_ids:
            continue
        seen_ids.add(v.track_id)
        x1, y1, x2, y2 = bbox
        dets.append({
            "track_id": v.track_id,
            "bbox":     (float(x1), float(y1), float(x2), float(y2)),
            "cls":      VEHICLE_CLS,
            "cls_name": VEHICLE_CLS_NAME,
            "conf":     VEHICLE_CONF,
        })
    return dets


def build_scenario(name: str) -> Tuple[List[_Vehicle], List[List[Dict]]]:
    """Return (vehicles, frames) for the named scenario.

    frames[i] is the list of detection dicts active at frame i.
    """
    if name not in _SCENARIOS:
        raise ValueError(f"Unknown scenario {name!r}; available: {ALL_SCENARIOS}")

    vehicles = _SCENARIOS[name]()
    n_frames = max(len(v.frames) for v in vehicles)
    frames   = [_frame_detections(vehicles, i) for i in range(n_frames)]
    return vehicles, frames


# ── FakeDetectionSource ───────────────────────────────────────────────────────

class FakeDetectionSource:
    """Yields pre-scripted detection dicts, bypassing the real Detector + Tracker.

    Designed to be used by service/main.py when --fake-detections is set:

        det_source = FakeDetectionSource(scenario)
        while True:
            dets = det_source.next_frame()
            if dets is None:
                break
            rule_events = engine.evaluate(dets, ts_mono_ns, ts_real_ns)
            ...

    Detection format: {track_id, bbox (x1,y1,x2,y2), cls, cls_name, conf}
    Same as Tracker.update() output — compatible with RuleEngine.evaluate().
    """

    def __init__(self, scenario: str, loop: bool = False) -> None:
        self._scenario = scenario
        self._loop     = loop
        self._vehicles, self._frames = build_scenario(scenario)
        self._idx      = 0
        print(f"[FakeDetectionSource] scenario={scenario!r} "
              f"frames={len(self._frames)} loop={loop}")

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    @property
    def scenario(self) -> str:
        return self._scenario

    def next_frame(self) -> Optional[List[Dict]]:
        """Return next frame's detection list, or None when exhausted."""
        if self._idx >= len(self._frames):
            if self._loop:
                self._idx = 0
            else:
                return None
        dets = self._frames[self._idx]
        self._idx += 1
        return dets

    def __iter__(self) -> Iterator[List[Dict]]:
        self._idx = 0
        while True:
            dets = self.next_frame()
            if dets is None:
                return
            yield dets

    def reset(self) -> None:
        self._idx = 0


# ── Visual renderer ───────────────────────────────────────────────────────────

def render_frame(
    vehicles: List[_Vehicle],
    frame_idx: int,
    scenario_name: str,
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
) -> np.ndarray:
    """Render one scenario frame as a BGR numpy array.

    Draws: gray background, colored vehicle rectangles, red trigger line,
    frame counter + scenario label. Useful for visual inspection
    (--save-frames) and for the GStreamer publisher.
    """
    img = np.full((frame_h, frame_w, 3), BG_COLOR, dtype=np.uint8)

    # Trigger line
    trigger_y = frame_h // 2
    cv2.line(img, (0, trigger_y), (frame_w, trigger_y), LINE_COLOR, 2)
    cv2.putText(img, f"trigger y={trigger_y}",
                (10, trigger_y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, LINE_COLOR, 1, cv2.LINE_AA)

    # Vehicles
    for v in vehicles:
        if frame_idx >= len(v.frames) or v.frames[frame_idx] is None:
            continue
        x1, y1, x2, y2 = (int(c) for c in v.frames[frame_idx])
        color = v.color
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Bottom-center marker
        bc_x = (x1 + x2) // 2
        cv2.circle(img, (bc_x, y2), 5, color, -1)
        cv2.putText(img, f"id={v.track_id}",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Frame counter + scenario
    cv2.putText(img, f"scenario={scenario_name}  frame={frame_idx:04d}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)

    return img


# ── NV12 conversion (for GStreamer publisher) ─────────────────────────────────

def bgr_to_nv12(bgr: np.ndarray) -> bytes:
    """Convert a BGR uint8 frame to a NV12 byte string.

    OpenCV's COLOR_BGR2YUV_I420 gives Y + planar U + planar V.
    NV12 requires Y + interleaved UV, so we rearrange the chroma planes.
    """
    h, w = bgr.shape[:2]
    yuv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)

    y    = yuv[:h]                                        # (h, w)
    u    = yuv[h : h + h // 4].reshape(h // 2, w // 2)  # (h/2, w/2)
    v    = yuv[h + h // 4 :].reshape(h // 2, w // 2)    # (h/2, w/2)

    uv   = np.empty((h // 2, w), dtype=np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v

    return np.vstack([y, uv]).tobytes()


# ── GStreamer publisher ───────────────────────────────────────────────────────

def publish_via_gstreamer(
    vehicles:     List[_Vehicle],
    scenario:     str,
    fps:          int    = FPS_DEFAULT,
    socket_path:  str    = "/tmp/ai_frames.sock",
    loop:         bool   = False,
) -> None:
    """Publish scenario frames as NV12 via GStreamer shmsink.

    Requires gi (system package — do NOT pip-install PyGObject here).
    """
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)

    pipe_str = (
        f"appsrc name=src is-live=true block=true format=time "
        f"caps=video/x-raw,format=NV12,width={FRAME_W},height={FRAME_H},"
        f"framerate={fps}/1 "
        f"! shmsink socket-path={socket_path} wait-for-connection=false sync=false"
    )
    pipeline = Gst.parse_launch(pipe_str)
    appsrc   = pipeline.get_by_name("src")

    pipeline.set_state(Gst.State.PLAYING)
    print(f"[fake_grabber] Publishing scenario={scenario!r} → {socket_path}  fps={fps}")

    n_frames = max(len(v.frames) for v in vehicles)
    frame_ns = int(1e9 / fps)
    iteration = 0

    try:
        while True:
            for i in range(n_frames):
                bgr  = render_frame(vehicles, i, scenario)
                data = bgr_to_nv12(bgr)
                buf  = Gst.Buffer.new_wrapped(data)
                buf.pts      = (iteration * n_frames + i) * frame_ns
                buf.duration = frame_ns
                ret = appsrc.emit("push-buffer", buf)
                if ret.value_nick != "ok":
                    print(f"[fake_grabber] push-buffer error: {ret}")
                    return
                time.sleep(1.0 / fps)
            if not loop:
                break
            iteration += 1
    except KeyboardInterrupt:
        pass
    finally:
        appsrc.emit("end-of-stream")
        pipeline.set_state(Gst.State.NULL)
        print("[fake_grabber] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--scenario", default="normal", choices=ALL_SCENARIOS,
                        help="Scripted trajectory scenario (default: normal)")
    parser.add_argument("--fps", type=int, default=FPS_DEFAULT,
                        help="Frame rate (default: 5)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the scenario indefinitely")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true",
                      help="Print frame-by-frame detection dicts and exit")
    mode.add_argument("--save-frames", metavar="DIR",
                      help="Save rendered frames as JPEG to DIR and exit")
    mode.add_argument("--publish", action="store_true",
                      help="Publish NV12 frames via GStreamer shmsink")

    parser.add_argument("--socket", default="/tmp/ai_frames.sock",
                        help="shmsink socket path (with --publish)")
    args = parser.parse_args()

    vehicles, frames = build_scenario(args.scenario)
    n = len(frames)

    if args.list:
        print(f"Scenario: {args.scenario!r}  ({n} frames @ {args.fps} fps = "
              f"{n/args.fps:.1f}s)")
        for i, dets in enumerate(frames):
            if dets:
                print(f"  frame {i:04d}: {dets}")
        return

    if args.save_frames:
        out_dir = Path(args.save_frames)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img  = render_frame(vehicles, i, args.scenario)
            path = out_dir / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(path), img)
        print(f"Saved {n} frames to {out_dir}")
        return

    if args.publish:
        publish_via_gstreamer(vehicles, args.scenario,
                              fps=args.fps, socket_path=args.socket,
                              loop=args.loop)
        return

    # Default: print scenario summary
    parser.print_help()
    print(f"\nScenario {args.scenario!r}: {n} frames, {n/args.fps:.1f}s")
    crossings = sum(
        1 for i, dets in enumerate(frames)
        if dets and i > 0 and any(
            d["bbox"][3] / FRAME_H > 0.5 for d in dets  # bbox bottom below 50%
        ) and any(
            d["track_id"] in {d2["track_id"] for d2 in frames[i-1]}
            and frames[i-1][0]["bbox"][3] / FRAME_H <= 0.5
            for d in dets
        )
    )
    print(f"Approx line crossings (50% threshold): {crossings}")


if __name__ == "__main__":
    main()
