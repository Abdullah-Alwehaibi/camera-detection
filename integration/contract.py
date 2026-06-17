"""
Versioned interface contract — frame grabber ↔ AI pipeline.

Single source of truth for socket paths, GStreamer caps, boolean event
schema, rich sidecar schema, 4K scale factors, and cooldown values.
Every other module imports from here; never hard-code these values
elsewhere.

Design references
-----------------
§3   Frozen interface contract (from the grabber)
§4.3 Dual output — boolean path + rich sidecar
§3.3 ANPR snapshot mapping (1080p → 4K, per-axis factors)
"""

from __future__ import annotations

CONTRACT_VERSION = "1.0"

# ── Socket paths (defaults; overridden by pipeline.yaml) ──────────────────────
DEFAULT_FRAME_SOCKET    = "/tmp/ai_frames.sock"
DEFAULT_EVENTS_SOCKET   = "/tmp/ai_events.sock"
DEFAULT_SIDECAR_SOCKET  = "/tmp/ai_sidecar.sock"
DEFAULT_SNAPSHOT_SOCKET = "/tmp/ai_snapshot.sock"   # roadside_snapshot mode only

# ── Expected GStreamer caps from the grabber (§3.1) ───────────────────────────
# Validate inbound frames against these on connect; fail loudly on drift so
# a grabber resolution change doesn't silently corrupt detection coordinates.
EXPECTED_CAPS = {
    "format": "NV12",
    "width":  1920,
    "height": 1080,
    "fps":    15,
}


# ── Boolean event output (§3.2) ───────────────────────────────────────────────

class EventType:
    """Fixed enum of event types recognised by events_adapter.py.

    Do NOT add values here without coordinating with the events_adapter.py
    maintainer — the adapter maps these to ONVIF topics and has no catch-all.
    Rule engine results that have no native type (e.g. zone_intrusion) are
    mapped to the closest enum and carry real semantics in the sidecar.
    """
    MOTION        = "motion"
    PERSON        = "person"
    VEHICLE       = "vehicle"
    TAMPER        = "tamper"
    LINE_CROSSING = "line_crossing"
    ALL = frozenset({"motion", "person", "vehicle", "tamper", "line_crossing"})


# Cooldown durations — stay aligned with events_adapter.py. Do NOT add a
# second layer of debounce inside this pipeline; the adapter already handles it.
EVENT_COOLDOWN_SEC = {
    EventType.MOTION:        5.0,
    EventType.PERSON:        8.0,
    EventType.VEHICLE:       8.0,
    EventType.LINE_CROSSING: 8.0,
    EventType.TAMPER:       10.0,
}

# Required fields for a valid boolean event (§3.2 schema table)
REQUIRED_EVENT_FIELDS = frozenset({"type", "confidence"})


# ── 4K scale factors (§3.3) ───────────────────────────────────────────────────
# The AI 1080p frame is a straight anamorphic downscale of the 4K source
# (same tee, no temporal skew). Use SEPARATE per-axis factors — a single
# scalar misplaces crops horizontally because 4096/1920 ≠ 2160/1080.
# These are config values (pipeline.yaml scale_4k.*) so they survive a
# grabber resolution change without a code edit.
DEFAULT_SCALE_4K_X = 4096 / 1920   # ≈ 2.13333
DEFAULT_SCALE_4K_Y = 2160 / 1080   # = 2.00000


# ── Snapshot modes (§5.5) ─────────────────────────────────────────────────────
class SnapshotMode:
    BACKEND_PULL      = "backend_pull"
    ROADSIDE_SNAPSHOT = "roadside_snapshot"
    ALL = frozenset({"backend_pull", "roadside_snapshot"})


# ── Rich sidecar (§4.3) ───────────────────────────────────────────────────────
SIDECAR_SCHEMA_VERSION = "1.0"

class SidecarBackend:
    UNIX      = "unix"
    MQTT      = "mqtt"
    WEBSOCKET = "websocket"
    ALL = frozenset({"unix", "mqtt", "websocket"})


# ── Validation helpers ────────────────────────────────────────────────────────

def validate_event(event: dict) -> None:
    """Raise ValueError if a boolean event is malformed.

    Checks: required fields present, type in allowed enum,
    confidence in [0.0, 1.0].
    """
    missing = REQUIRED_EVENT_FIELDS - event.keys()
    if missing:
        raise ValueError(f"Event missing required fields: {missing!r}")
    if event["type"] not in EventType.ALL:
        raise ValueError(
            f"Invalid event type {event['type']!r}; "
            f"must be one of {sorted(EventType.ALL)}"
        )
    conf = event["confidence"]
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        raise ValueError(f"confidence {conf!r} must be a float in [0.0, 1.0]")


def scale_bbox_to_4k(
    bbox_1080: list,
    scale_x: float = DEFAULT_SCALE_4K_X,
    scale_y: float = DEFAULT_SCALE_4K_Y,
) -> list:
    """Convert [x, y, w, h] from 1080p pixel coords to 4K pixel coords.

    Uses separate per-axis factors (§3.3). Pass config values for
    scale_x/scale_y rather than the defaults if the grabber resolution
    has been updated.
    """
    x, y, w, h = bbox_1080
    return [x * scale_x, y * scale_y, w * scale_x, h * scale_y]


def normalize_bbox(bbox_px: list, width: int, height: int) -> list:
    """Convert [x, y, w, h] in pixels to normalized [0, 1] coordinates."""
    x, y, w, h = bbox_px
    return [x / width, y / height, w / width, h / height]


def denormalize_bbox(bbox_norm: list, width: int, height: int) -> list:
    """Convert normalized [0, 1] [x, y, w, h] to pixel coordinates."""
    x, y, w, h = bbox_norm
    return [x * width, y * height, w * width, h * height]


def validate_normalized_points(points: list, context: str = "") -> None:
    """Raise ValueError if any point is outside [0, 1] x [0, 1]."""
    for i, pt in enumerate(points):
        if len(pt) != 2:
            raise ValueError(f"{context}: point[{i}] must be [x, y], got {pt!r}")
        x, y = pt
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                f"{context}: point[{i}] {pt!r} is outside normalized [0,1] range"
            )
