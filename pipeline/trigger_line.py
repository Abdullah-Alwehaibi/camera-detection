"""Generalized trigger-zone evaluation: line or polygon.

A zone is configured as either:
  {"type": "line",    "points": [[x1,y1],[x2,y2]]}
  {"type": "polygon", "points": [[x1,y1],[x2,y2],[x3,y3],...]}  (3+ points)

Both types share one interface, Zone.evaluate(point, track_id) -> bool,
so the rest of the pipeline doesn't care which kind is configured.

For each tracked object, feed its bbox bottom-center point frame to frame.
evaluate() returns True exactly once per track ID:
  - "line":    the first time the point crosses from one side of the
               (infinite) line to the other.
  - "polygon": the first time the point transitions from outside the
               polygon to inside it.
"""

import json
from pathlib import Path


def _side(p1, p2, point):
    """Sign of the cross product of (p2 - p1) and (point - p1)."""
    (x1, y1), (x2, y2) = p1, p2
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def _point_in_polygon(point, polygon):
    """Ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        if (y1 > y) != (y2 > y):
            x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at_y:
                inside = not inside
        x1, y1 = x2, y2
    return inside


class Zone:
    def __init__(self, zone_type, points, name="zone"):
        if zone_type == "line":
            if len(points) != 2:
                raise ValueError(f"Zone {name!r}: line requires exactly 2 points")
        elif zone_type == "polygon":
            if len(points) < 3:
                raise ValueError(f"Zone {name!r}: polygon requires 3+ points")
        else:
            raise ValueError(f"Zone {name!r}: unknown type {zone_type!r}")

        self.type = zone_type
        self.points = [tuple(p) for p in points]
        self.name = name
        self._last_side = {}
        self._last_inside = {}
        self._triggered = set()

    def evaluate(self, point, track_id):
        """Return True exactly once per track_id, on entry into the zone."""
        if track_id in self._triggered:
            return False

        if self.type == "line":
            crossed = self._evaluate_line(point, track_id)
        else:
            crossed = self._evaluate_polygon(point, track_id)

        if crossed:
            self._triggered.add(track_id)
        return crossed

    def _evaluate_line(self, point, track_id):
        side = _side(self.points[0], self.points[1], point)
        sign = (side > 0) - (side < 0)

        last = self._last_side.get(track_id)
        if sign != 0:
            self._last_side[track_id] = sign

        return last is not None and sign != 0 and sign != last

    def _evaluate_polygon(self, point, track_id):
        inside = _point_in_polygon(point, self.points)
        last = self._last_inside.get(track_id)
        self._last_inside[track_id] = inside

        return last is False and inside is True


def load_zones(config_path):
    data = json.loads(Path(config_path).read_text())
    return [
        Zone(entry["type"], entry["points"], name=entry.get("id", "zone"))
        for entry in data["zones"]
    ]
