"""
ai/bestshot.py

Per-track best-shot selector.

Maintains the highest-scoring detection frame for every active track and
dispatches it (annotated to JPEG + capture metadata) when a rule event fires.

Score formula
-------------
  score = bbox_area_px × detection_conf × zone_factor
  zone_factor = 1.0 if bbox bottom-center is inside any capture zone
                0.3 otherwise (outside-zone detections are still kept as
                fallback in case the vehicle is never in-zone)

Dispatch modes (config/pipeline.yaml snapshot.mode)
----------------------------------------------------
backend_pull (default):
  Returns bbox_4k + ts_real_ns in the capture metadata. The backend
  retrieves the 4K frame from the recording archive by timestamp. No
  socket or grabber change required.

roadside_snapshot:
  Sends a JSON request to snapshot_socket (AF_UNIX SOCK_STREAM, defined
  in docs/recommendation_snapshot_service.md). On success the response
  carries a JPEG path. Auto-falls back to backend_pull on timeout or
  service absence — caller never needs to handle this.

In both modes, an annotated local evidence JPEG is also saved to
evidence_dir (unless dry_run=True).
"""

from __future__ import annotations

import json
import logging
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ai.rules.engine import CaptureZone, RuleEvent
from integration.contract import DEFAULT_SCALE_4K_X, DEFAULT_SCALE_4K_Y, scale_bbox_to_4k
from pipeline.evidence_capture import draw_databar, draw_detections, save_evidence

log = logging.getLogger(__name__)

_EXPIRE_SEC      = 5.0   # clear per-track state this long after last detection
_SNAPSHOT_TIMEOUT = 5.0  # seconds to wait for roadside_snapshot service response


# ── Internal per-track state ──────────────────────────────────────────────────

@dataclass
class _BestFrame:
    score:      float
    frame:      np.ndarray   # raw (unannotated) BGR copy at best-score moment
    bbox_1080:  List[float]  # [x, y, w, h] pixels
    bbox_4k:    List[float]  # scaled to 4K
    ts_mono_ns: int
    ts_real_ns: int
    cls_name:   str
    conf:       float
    last_seen:  float        # monotonic seconds, for expiry


# ── Zone membership (normalized coords) ──────────────────────────────────────

def _in_zones(bc_norm, zones: List[CaptureZone]) -> bool:
    """True if the normalized point falls in any capture zone."""
    if not zones:
        return True   # no capture zones → treat every position as in-zone
    from shapely.geometry import Point, Polygon
    p = Point(bc_norm)
    return any(Polygon(z.points).contains(p) for z in zones)


# ── BestShotSelector ─────────────────────────────────────────────────────────

class BestShotSelector:
    """Maintains per-track best frames; dispatches on rule events.

    Pipeline usage (service/main.py):

        # Update after every frame
        bestshot.update(tracked_dets, frame_packet.cpu_bgr, capture_zones,
                        frame_packet.ts_mono_ns, frame_packet.ts_real_ns,
                        frame_w=1920, frame_h=1080)

        # Dispatch for each rule event
        for event in rule_events:
            capture_meta = bestshot.dispatch(event)
            sidecar.send({...event_fields..., "capture": capture_meta})
    """

    def __init__(
        self,
        mode: str            = "backend_pull",
        snapshot_socket: str = "/tmp/ai_snapshot.sock",
        evidence_dir: str    = "data/snapshots",
        scale_4k_x: float   = DEFAULT_SCALE_4K_X,
        scale_4k_y: float   = DEFAULT_SCALE_4K_Y,
        location: str        = "Unknown",
        dry_run: bool        = False,
    ) -> None:
        if mode not in ("backend_pull", "roadside_snapshot"):
            raise ValueError(f"Unknown snapshot mode: {mode!r}")

        self._mode          = mode
        self._snap_sock     = snapshot_socket
        self._evidence_dir  = Path(evidence_dir)
        self._sx            = scale_4k_x
        self._sy            = scale_4k_y
        self._location      = location
        self._dry_run       = dry_run
        self._best: Dict[int, _BestFrame] = {}

        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "BestShotSelector: mode=%s evidence_dir=%s dry_run=%s",
            mode, evidence_dir, dry_run,
        )

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        tracked_dets:  List[Dict],
        frame_bgr:     np.ndarray,
        capture_zones: List[CaptureZone],
        ts_mono_ns:    int,
        ts_real_ns:    int,
        frame_w:       int = 1920,
        frame_h:       int = 1080,
    ) -> None:
        """Score each tracked detection and update the per-track best frame.

        frame_bgr is the raw (unannotated) BGR frame. A copy is stored when
        the score improves so it is isolated from the next frame's buffer.
        """
        now = ts_mono_ns / 1e9

        for det in tracked_dets:
            track_id = int(det["track_id"])
            bbox     = det["bbox"]   # (x1, y1, x2, y2) px
            conf     = float(det["conf"])

            x1, y1, x2, y2 = bbox
            bc_norm  = ((x1 + x2) / 2.0 / frame_w, y2 / frame_h)
            in_zone  = _in_zones(bc_norm, capture_zones)

            area        = (x2 - x1) * (y2 - y1)
            zone_factor = 1.0 if in_zone else 0.3
            score       = area * conf * zone_factor

            existing = self._best.get(track_id)
            if existing is None or score > existing.score:
                bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                self._best[track_id] = _BestFrame(
                    score      = score,
                    frame      = frame_bgr.copy(),
                    bbox_1080  = bbox_xywh,
                    bbox_4k    = scale_bbox_to_4k(bbox_xywh, self._sx, self._sy),
                    ts_mono_ns = ts_mono_ns,
                    ts_real_ns = ts_real_ns,
                    cls_name   = str(det.get("cls_name", "")),
                    conf       = conf,
                    last_seen  = now,
                )
            else:
                self._best[track_id].last_seen = now

        self._expire_stale(now)

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def dispatch(self, event: RuleEvent) -> dict:
        """Dispatch the best-shot capture for a rule event.

        Saves an annotated evidence JPEG locally (unless dry_run=True) and
        returns a capture metadata dict for inclusion in the sidecar record.

        In roadside_snapshot mode, requests a 4K crop from the snapshot
        service; auto-falls back to backend_pull silently on any failure.
        """
        best = self._best.get(event.track_id)

        # Use the best-scored frame if available; fall back to event metadata
        if best is not None:
            bbox_1080  = best.bbox_1080
            bbox_4k    = best.bbox_4k
            ts_real_ns = best.ts_real_ns
            ts_mono_ns = best.ts_mono_ns
            frame_bgr  = best.frame
            conf       = best.conf
        else:
            log.debug("BestShot: no best frame for track=%d; using event bbox",
                      event.track_id)
            bbox_1080  = event.bbox_1080
            bbox_4k    = event.bbox_4k
            ts_real_ns = event.ts_real_ns
            ts_mono_ns = event.ts_mono_ns
            frame_bgr  = None
            conf       = event.confidence

        # ── Local annotated evidence JPEG ─────────────────────────────────────
        crop_path: Optional[str] = None
        if frame_bgr is not None and not self._dry_run:
            crop_path = self._save_local(event, frame_bgr, bbox_1080, ts_real_ns)

        # ── roadside_snapshot (4K crop from snapshot service) ─────────────────
        if self._mode == "roadside_snapshot":
            snap = self._request_snapshot(event.track_id, ts_mono_ns, bbox_4k)
            if snap:
                log.info("BestShot: roadside_snapshot track=%d path=%s",
                         event.track_id, snap)
                return {
                    "source":    "roadside_snapshot",
                    "jpeg_path": snap,
                    "ts_real_ns": ts_real_ns,
                    "bbox_4k":   bbox_4k,
                    "bbox_1080": bbox_1080,
                    "conf":      conf,
                    "crop_path": crop_path,
                }
            log.warning(
                "BestShot: roadside_snapshot failed for track=%d; "
                "falling back to backend_pull", event.track_id,
            )

        # ── backend_pull (or roadside_snapshot fallback) ──────────────────────
        return {
            "source":     "backend_pull",
            "ts_real_ns": ts_real_ns,
            "ts_mono_ns": ts_mono_ns,
            "bbox_4k":    bbox_4k,
            "bbox_1080":  bbox_1080,
            "conf":       conf,
            "crop_path":  crop_path,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _save_local(
        self,
        event:      RuleEvent,
        frame_bgr:  np.ndarray,
        bbox_1080:  List[float],
        ts_real_ns: int,
    ) -> Optional[str]:
        """Annotate + save JPEG evidence to evidence_dir. Returns path or None."""
        try:
            ts = datetime.fromtimestamp(ts_real_ns / 1e9)

            x, y, w, h = bbox_1080
            det_for_draw = [{
                "track_id": event.track_id,
                "bbox":     (x, y, x + w, y + h),
                "cls_name": event.cls_name,
            }]
            # draw_detections returns a copy with boxes drawn
            annotated = draw_detections(frame_bgr, det_for_draw,
                                        highlight_track_id=event.track_id)
            # draw_databar mutates annotated in place
            draw_databar(annotated, ts, self._location, event.cls_name)

            metadata = {
                "timestamp": ts,
                "track_id":  event.track_id,
                "bbox":      (x, y, x + w, y + h),
                "class":     event.cls_name,
                "image":     annotated,
            }
            path = save_evidence(metadata, str(self._evidence_dir))
            log.info("BestShot: evidence saved track=%d → %s",
                     event.track_id, path)
            return str(path)
        except Exception as exc:
            log.error("BestShot: evidence save failed for track=%d: %s",
                      event.track_id, exc)
            return None

    def _request_snapshot(
        self,
        track_id:   int,
        ts_mono_ns: int,
        bbox_4k:    List[float],
    ) -> Optional[str]:
        """Send JSON request to roadside_snapshot service. Returns JPEG path or None.

        Request  format: {"track_id", "ts_monotonic_ns", "bbox_4k"}
        Response format: {"status": "ok", "jpeg_path", "ts_monotonic_ns"}
                      OR {"status": "error", "error": "..."}

        See docs/recommendation_snapshot_service.md for the full contract.
        """
        req = json.dumps({
            "track_id":        track_id,
            "ts_monotonic_ns": ts_mono_ns,
            "bbox_4k":         bbox_4k,
        }).encode() + b"\n"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(_SNAPSHOT_TIMEOUT)
                s.connect(self._snap_sock)
                s.sendall(req)
                buf = b""
                while b"\n" not in buf:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
            resp = json.loads(buf.split(b"\n")[0])
            if resp.get("status") == "ok":
                return resp["jpeg_path"]
            log.warning("BestShot: snapshot service returned error: %s",
                        resp.get("error", "unknown"))
            return None
        except FileNotFoundError:
            log.debug("BestShot: snapshot socket %s not found", self._snap_sock)
            return None
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            log.debug("BestShot: snapshot request failed: %s", exc)
            return None

    def _expire_stale(self, now: float) -> None:
        cutoff = now - _EXPIRE_SEC
        stale  = [tid for tid, bf in self._best.items() if bf.last_seen < cutoff]
        for tid in stale:
            del self._best[tid]

    # ── Inspection ────────────────────────────────────────────────────────────

    def active_track_count(self) -> int:
        """Number of tracks with stored best frames (for metrics)."""
        return len(self._best)

    def reset(self) -> None:
        """Clear all per-track state (e.g. on source switch)."""
        self._best.clear()
        log.info("BestShotSelector: state reset")
