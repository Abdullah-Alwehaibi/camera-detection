"""
ai/rules/engine.py

Pure rule engine: geometry evaluation, state machines, event emission.

No hardware dependency — runs identically on Jetson and any dev machine.
All geometry is in normalized [0, 1] coordinates (resolution-independent).

Rule types
----------
line_crossing   — segment-intersection test between consecutive track positions
                  and the rule line. Catches fast vehicles that jump the line
                  in a single frame (unlike per-frame point-side tests).
polygon_intrusion — point-in-polygon on the bbox bottom-center; fires on the
                  first outside→inside transition per track. Also tracks dwell
                  (frames-in-zone) for downstream use.
direction       — smoothed heading over the last N track positions vs an allowed
                  heading ± tolerance. Fires when the vehicle is going the
                  wrong way for long enough to be stable.

State machine
-------------
Per (rule_id, track_id): first-seen monotonic time (used to detect track-ID
reuse), last-seen time (used to expire stale entries), cooldown_until, and
rule-specific state (prev_pos, frames_in_zone, dir_history).

Cooldowns are taken from the rule config and MUST stay aligned with
events_adapter.py's own cooldown values so there is no double-suppression.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

from shapely.geometry import LineString, Point, Polygon

from integration.contract import DEFAULT_SCALE_4K_X, DEFAULT_SCALE_4K_Y, scale_bbox_to_4k

# Track state is discarded after this many seconds without a detection,
# which also resets the state when a new vehicle reuses the same track ID.
_TRACK_EXPIRE_SEC = 5.0

# Minimum positions in the direction history before firing a wrong-way event.
_DIR_MIN_SAMPLES = 5


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Rule:
    id: str
    type: str                           # "line_crossing" | "polygon_intrusion" | "direction"
    enabled: bool
    points: List[Tuple[float, float]]   # normalized [0,1] geometry points
    direction: str                      # "any" | "positive" | "negative"
    classes: frozenset                  # class names that trigger this rule
    event_type: str                     # EventType enum value (for boolean writer)
    cooldown_sec: float
    # direction rule only
    allowed_heading_deg: float = 0.0
    tolerance_deg: float       = 45.0


@dataclass
class CaptureZone:
    id: str
    points: List[Tuple[float, float]]   # normalized polygon


@dataclass
class RuleEvent:
    """Output of RuleEngine.evaluate() — ready for both output writers."""
    rule_id:        str
    rule_type:      str
    event_type:     str             # boolean event type (EventType enum)
    track_id:       int
    cls_name:       str
    confidence:     float
    bbox_1080:      List[float]     # [x, y, w, h] pixels, 1080p frame
    bbox_4k:        List[float]     # [x, y, w, h] pixels, 4K frame
    ts_mono_ns:     int
    ts_real_ns:     int
    direction_sign: Optional[int]   # +1 / -1 for line crossing; None otherwise
    frames_in_zone: int             # dwell counter (polygon rules)
    in_capture_zone: bool           # whether the vehicle is in a capture zone


# ── Per-track state ───────────────────────────────────────────────────────────

@dataclass
class _TrackState:
    first_seen:     float                           # monotonic (detect ID reuse)
    last_seen:      float                           # monotonic (expire stale)
    prev_pos:       Optional[Tuple[float, float]]   # previous bottom-center (normalized)
    cooldown_until: float                           # monotonic; 0 = not in cooldown
    frames_in_zone: int
    last_in_zone:   bool
    dir_history:    deque = field(default_factory=lambda: deque(maxlen=20))

    @staticmethod
    def create(now: float) -> "_TrackState":
        return _TrackState(
            first_seen     = now,
            last_seen      = now,
            prev_pos       = None,
            cooldown_until = 0.0,
            frames_in_zone = 0,
            last_in_zone   = False,
        )


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _bottom_center_norm(bbox_xyxy_px: tuple, w: int, h: int) -> Tuple[float, float]:
    """Return normalized bottom-center of a (x1, y1, x2, y2) pixel bbox."""
    x1, y1, x2, y2 = bbox_xyxy_px
    return ((x1 + x2) / 2.0 / w, y2 / h)


def _bbox_to_xywh_px(bbox_xyxy_px: tuple) -> List[float]:
    x1, y1, x2, y2 = bbox_xyxy_px
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _crossing_direction_sign(p1, p2, prev, curr) -> int:
    """Sign of (p2-p1) × (curr-prev) cross product.

    +1: vehicle moves to the left of the line (when facing A→B).
    -1: vehicle moves to the right.
     0: parallel / degenerate (treat as no crossing).
    """
    lx, ly = p2[0] - p1[0], p2[1] - p1[1]
    mx, my = curr[0] - prev[0], curr[1] - prev[1]
    cross = lx * my - ly * mx
    if cross > 1e-12:
        return 1
    if cross < -1e-12:
        return -1
    return 0


def _smoothed_heading_deg(positions: deque) -> Optional[float]:
    """Heading in degrees from the oldest to the newest position in the deque.

    Returns None if the deque has fewer than 2 entries or the vehicle is
    stationary (moved less than 1e-4 in normalized coords).
    """
    if len(positions) < 2:
        return None
    dx = positions[-1][0] - positions[0][0]
    dy = positions[-1][1] - positions[0][1]
    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return None
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _angular_distance(a: float, b: float) -> float:
    """Smallest angular distance between two headings in [0, 360)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _point_in_capture_zones(pos: Tuple[float, float], zones: List[CaptureZone]) -> bool:
    if not zones:
        return True     # no capture zones configured → every position qualifies
    p = Point(pos)
    return any(Polygon(z.points).contains(p) for z in zones)


# ── Rule engine ───────────────────────────────────────────────────────────────

class RuleEngine:
    """Evaluates declarative rules against a stream of detections.

    Args:
        rules_source: either a List[Rule] or a zero-argument callable that
                      returns List[Rule]. The callable form integrates with
                      RulesLoader (hot-reload takes effect on the next call).
        capture_zones_source: same pattern for CaptureZone lists.
        frame_w, frame_h: pixel dimensions of the AI frame (1080p default).
        scale_4k_x/y: per-axis 4K upscale factors (from pipeline.yaml).
        track_expire_sec: how long to retain state for a track that
                          disappears; also the window for detecting ID reuse.
    """

    def __init__(
        self,
        rules_source: Union[List[Rule], Callable[[], List[Rule]]],
        capture_zones_source: Union[List[CaptureZone], Callable[[], List[CaptureZone]], None] = None,
        frame_w: int = 1920,
        frame_h: int = 1080,
        scale_4k_x: float = DEFAULT_SCALE_4K_X,
        scale_4k_y: float = DEFAULT_SCALE_4K_Y,
        track_expire_sec: float = _TRACK_EXPIRE_SEC,
    ) -> None:
        self._rules_src  = rules_source
        self._zones_src  = capture_zones_source
        self._fw         = frame_w
        self._fh         = frame_h
        self._sx         = scale_4k_x
        self._sy         = scale_4k_y
        self._expire_sec = track_expire_sec

        # _states[(rule_id, track_id)] = _TrackState
        self._states: Dict[Tuple[str, int], _TrackState] = {}

    # ── Internal accessors ────────────────────────────────────────────────────

    def _rules(self) -> List[Rule]:
        return self._rules_src() if callable(self._rules_src) else self._rules_src

    def _capture_zones(self) -> List[CaptureZone]:
        if self._zones_src is None:
            return []
        return self._zones_src() if callable(self._zones_src) else self._zones_src

    def _state(self, rule_id: str, track_id: int, now: float) -> _TrackState:
        key = (rule_id, track_id)
        st  = self._states.get(key)
        if st is None:
            st = _TrackState.create(now)
            self._states[key] = st
            return st
        # Track-ID reuse: if the track was absent for longer than expire_sec,
        # a new vehicle has the same ID — reset state completely.
        if now - st.last_seen > self._expire_sec:
            st = _TrackState.create(now)
            self._states[key] = st
        return st

    # ── Rule evaluators ───────────────────────────────────────────────────────

    def _eval_line_crossing(
        self,
        rule: Rule,
        state: _TrackState,
        curr_pos: Tuple[float, float],
        now: float,
    ) -> Optional[int]:
        """Return crossing direction sign (+1/-1) or None if not triggered."""
        if state.prev_pos is None:
            return None
        if now < state.cooldown_until:
            return None

        traj = LineString([state.prev_pos, curr_pos])
        line = LineString([rule.points[0], rule.points[1]])
        if not traj.intersects(line):
            return None

        sign = _crossing_direction_sign(
            rule.points[0], rule.points[1], state.prev_pos, curr_pos
        )

        if rule.direction == "positive" and sign != 1:
            return None
        if rule.direction == "negative" and sign != -1:
            return None

        state.cooldown_until = now + rule.cooldown_sec
        return sign

    def _eval_polygon_intrusion(
        self,
        rule: Rule,
        state: _TrackState,
        curr_pos: Tuple[float, float],
        now: float,
    ) -> bool:
        """Returns True on first outside→inside transition (per cooldown)."""
        poly   = Polygon(rule.points)
        inside = poly.contains(Point(curr_pos))

        if inside:
            state.frames_in_zone += 1
        else:
            state.frames_in_zone = 0

        was_inside = state.last_in_zone
        state.last_in_zone = inside

        # Fire on entry (outside→inside) and not in cooldown
        if inside and not was_inside and now >= state.cooldown_until:
            state.cooldown_until = now + rule.cooldown_sec
            return True
        return False

    def _eval_direction(
        self,
        rule: Rule,
        state: _TrackState,
        curr_pos: Tuple[float, float],
        now: float,
    ) -> bool:
        """Returns True when the smoothed heading deviates beyond tolerance."""
        state.dir_history.append(curr_pos)
        if len(state.dir_history) < _DIR_MIN_SAMPLES:
            return False
        if now < state.cooldown_until:
            return False

        heading = _smoothed_heading_deg(state.dir_history)
        if heading is None:
            return False

        delta = _angular_distance(heading, rule.allowed_heading_deg)
        if delta <= rule.tolerance_deg:
            return False

        state.cooldown_until = now + rule.cooldown_sec
        return True

    # ── Main entry point ──────────────────────────────────────────────────────

    def evaluate(
        self,
        detections: List[dict],
        ts_mono_ns: int,
        ts_real_ns: int,
    ) -> List[RuleEvent]:
        """Evaluate all enabled rules against one frame's detections.

        detections: list of dicts with keys: track_id (int), bbox (x1,y1,x2,y2)
                    in pixels, cls_name (str), conf (float).
        Returns a (possibly empty) list of RuleEvent.
        Must be called in frame order (monotonic ts_mono_ns).
        """
        now    = ts_mono_ns / 1e9       # seconds (monotonic)
        events = []
        zones  = self._capture_zones()

        for det in detections:
            track_id = int(det["track_id"])
            cls_name = str(det.get("cls_name", det.get("class", "")))
            conf     = float(det.get("conf", det.get("confidence", 0.0)))
            bbox     = det["bbox"]       # (x1, y1, x2, y2) pixels

            curr_pos   = _bottom_center_norm(bbox, self._fw, self._fh)
            bbox_xywh  = _bbox_to_xywh_px(bbox)
            bbox_4k    = scale_bbox_to_4k(bbox_xywh, self._sx, self._sy)
            in_cap_zone = _point_in_capture_zones(curr_pos, zones)

            for rule in self._rules():
                if not rule.enabled:
                    continue
                if cls_name not in rule.classes:
                    continue

                state = self._state(rule.id, track_id, now)
                fired = False
                dir_sign = None
                frames_in = 0

                if rule.type == "line_crossing":
                    dir_sign = self._eval_line_crossing(rule, state, curr_pos, now)
                    fired = dir_sign is not None

                elif rule.type == "polygon_intrusion":
                    fired = self._eval_polygon_intrusion(rule, state, curr_pos, now)
                    frames_in = state.frames_in_zone

                elif rule.type == "direction":
                    fired = self._eval_direction(rule, state, curr_pos, now)

                if fired:
                    events.append(RuleEvent(
                        rule_id        = rule.id,
                        rule_type      = rule.type,
                        event_type     = rule.event_type,
                        track_id       = track_id,
                        cls_name       = cls_name,
                        confidence     = conf,
                        bbox_1080      = bbox_xywh,
                        bbox_4k        = bbox_4k,
                        ts_mono_ns     = ts_mono_ns,
                        ts_real_ns     = ts_real_ns,
                        direction_sign = dir_sign,
                        frames_in_zone = frames_in,
                        in_capture_zone = in_cap_zone,
                    ))

                # Update position history after evaluation
                state.prev_pos  = curr_pos
                state.last_seen = now

        self._expire_stale(now)
        return events

    def _expire_stale(self, now: float) -> None:
        """Remove state for tracks not seen recently (O(n) but infrequent)."""
        cutoff = now - self._expire_sec
        stale  = [k for k, st in self._states.items() if st.last_seen < cutoff]
        for k in stale:
            del self._states[k]

    def active_track_count(self) -> int:
        return len(self._states)

    def reset(self) -> None:
        """Clear all track states (e.g. on pipeline restart)."""
        self._states.clear()
