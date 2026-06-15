"""Per-vehicle evidence capture for trigger-zone crossing events.

On a trigger event, the full frame is annotated with a bounding box +
"<class> #<track_id>" label for every currently-tracked vehicle (the one
that triggered the event highlighted in a different color) plus a databar
across the top showing the capture date/time, camera location, and the
triggering vehicle's class, then saved as a JPEG. save_evidence() takes a
metadata dict (timestamp, track_id, bbox, class, image) that maps directly
onto a future evidence record / Violation Queue.
"""

from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "data" / "snapshots"

HIGHLIGHT_COLOR = (0, 0, 255)  # BGR red -- the vehicle that triggered this event
DETECTION_COLOR = (0, 255, 0)  # BGR green -- other vehicles in frame, for context

BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.6
DATABAR_HEIGHT = 40
DATABAR_SCALE = 0.7


def draw_detections(frame, detections, highlight_track_id=None):
    """Return a copy of frame with a bounding box + "<class> #<track_id>"
    label for each detection. The detection matching highlight_track_id
    (the vehicle that triggered this evidence capture) is drawn in
    HIGHLIGHT_COLOR; all others in DETECTION_COLOR, for scene context."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        color = HIGHLIGHT_COLOR if det["track_id"] == highlight_track_id else DETECTION_COLOR
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        label = f'{det["cls_name"]} #{det["track_id"]}'
        cv2.putText(annotated, label, (x1, max(0, y1 - 6)), FONT, LABEL_SCALE, color, BOX_THICKNESS, cv2.LINE_AA)
    return annotated


def draw_databar(frame, timestamp, location, vehicle_class):
    """Draw a solid databar across the top of frame with the capture
    date/time, camera location, and vehicle class. Modifies frame in
    place and returns it."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, DATABAR_HEIGHT), (0, 0, 0), -1)
    text = (
        f"{timestamp.strftime('%Y-%m-%d')}  {timestamp.strftime('%H:%M:%S')}"
        f"  |  {location}  |  {vehicle_class.upper()}"
    )
    cv2.putText(frame, text, (10, DATABAR_HEIGHT - 12), FONT, DATABAR_SCALE, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def save_evidence(metadata, output_dir=DEFAULT_EVIDENCE_DIR):
    """Persist a crossing-event evidence record.

    metadata: dict with timestamp (datetime), track_id (int),
    bbox (x1, y1, x2, y2), class (str), image (annotated BGR ndarray).

    Returns the path to the saved JPEG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = metadata["timestamp"].strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"vehicle_{metadata['track_id']}_{ts}.jpg"

    cv2.imwrite(str(output_path), metadata["image"])
    return output_path
