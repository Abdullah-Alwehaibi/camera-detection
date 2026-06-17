"""
ai/rules/loader.py

Loads config/rules.json and provides hot-reload via watchdog.

On a bad edit (invalid JSON, out-of-range geometry, unknown event type,
missing required fields), the error is logged and the last-good config
is kept — the pipeline never crashes on a rules file edit.

Thread safety: a threading.RLock protects _rules and _capture_zones.
The watchdog callback fires on the watchdog thread; readers (RuleEngine)
access the rules list on the main thread via the rules/capture_zones
properties.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

from ai.rules.engine import CaptureZone, Rule
from integration.contract import EventType, validate_normalized_points

log = logging.getLogger(__name__)

# ── Pure parsing / validation ─────────────────────────────────────────────────

def parse_rules(
    data: dict,
    valid_classes: Optional[Set[str]] = None,
) -> Tuple[List[Rule], List[CaptureZone]]:
    """Parse and validate rules.json payload.

    Raises ValueError if any rule is invalid (geometry out of [0,1],
    unknown event_type, unsupported rule type, wrong point count, etc.)
    so the loader can fall back to the last-good config.

    valid_classes: if provided, rule.classes must be a subset.
    """
    schema_ver = data.get("schema_version", "")
    if schema_ver not in ("", "1.0"):
        raise ValueError(f"Unsupported rules schema_version: {schema_ver!r}")

    rules: List[Rule] = []
    for i, r in enumerate(data.get("rules", [])):
        rule_id = r.get("id") or f"rule_{i}"
        ctx     = f"rule {rule_id!r}"

        rule_type = r.get("type", "")
        if rule_type not in ("line_crossing", "polygon_intrusion", "direction"):
            raise ValueError(f"{ctx}: unsupported type {rule_type!r}")

        pts = [tuple(p) for p in r.get("geometry", {}).get("points", [])]
        validate_normalized_points(pts, context=ctx)

        if rule_type == "line_crossing" and len(pts) != 2:
            raise ValueError(f"{ctx}: line_crossing requires exactly 2 points, got {len(pts)}")
        if rule_type == "polygon_intrusion" and len(pts) < 3:
            raise ValueError(f"{ctx}: polygon_intrusion requires ≥ 3 points, got {len(pts)}")
        if rule_type == "direction" and len(pts) != 0:
            pass  # direction rules carry heading params, not points
        if rule_type == "direction":
            pts = []   # direction rules use heading params, not points

        event_type = r.get("event_type", "vehicle")
        if event_type not in EventType.ALL:
            raise ValueError(
                f"{ctx}: event_type {event_type!r} not in {sorted(EventType.ALL)}"
            )

        cooldown = float(r.get("cooldown_sec", 8.0))
        if cooldown <= 0:
            raise ValueError(f"{ctx}: cooldown_sec must be > 0, got {cooldown}")

        direction = r.get("direction", "any")
        if direction not in ("any", "positive", "negative"):
            raise ValueError(f"{ctx}: direction must be any/positive/negative")

        cls_list = r.get("classes", [])
        if not cls_list:
            raise ValueError(f"{ctx}: classes list must not be empty")
        cls_set = frozenset(cls_list)
        if valid_classes and not cls_set.issubset(valid_classes):
            unknown = cls_set - valid_classes
            raise ValueError(f"{ctx}: unknown classes {unknown!r}; valid: {valid_classes!r}")

        enabled = bool(r.get("enabled", True))

        geom_cfg = r.get("geometry", {})
        rules.append(Rule(
            id                  = rule_id,
            type                = rule_type,
            enabled             = enabled,
            points              = pts,
            direction           = direction,
            classes             = cls_set,
            event_type          = event_type,
            cooldown_sec        = cooldown,
            allowed_heading_deg = float(geom_cfg.get("allowed_heading_deg", 0.0)),
            tolerance_deg       = float(geom_cfg.get("tolerance_deg", 45.0)),
        ))

    zones: List[CaptureZone] = []
    for j, z in enumerate(data.get("capture_zones", [])):
        zone_id = z.get("id") or f"zone_{j}"
        pts     = [tuple(p) for p in z.get("geometry", {}).get("points", [])]
        validate_normalized_points(pts, context=f"capture_zone {zone_id!r}")
        if len(pts) < 3:
            raise ValueError(f"capture_zone {zone_id!r}: polygon requires ≥ 3 points")
        zones.append(CaptureZone(id=zone_id, points=pts))

    return rules, zones


# ── File loader with hot-reload ───────────────────────────────────────────────

class RulesLoader:
    """Loads rules.json and hot-reloads on file change.

    Usage:
        loader = RulesLoader("config/rules.json")
        loader.load()               # initial load (raises on failure)
        loader.start_watching()     # background watchdog (optional)

        engine = RuleEngine(
            rules_source         = loader.rules,
            capture_zones_source = loader.capture_zones,
        )

    Hot-reload: on any file modification, the loader re-parses and re-validates.
    On error, the last-good config is kept and the error is logged.
    """

    # Brief delay after a filesystem event before reading the file —
    # some editors (vim, nano) write atomically via rename and the event
    # may fire before the rename completes.
    _RELOAD_DELAY_SEC = 0.15

    def __init__(
        self,
        rules_path: str,
        valid_classes: Optional[Set[str]] = None,
    ) -> None:
        self._path          = Path(rules_path)
        self._valid_classes = valid_classes
        self._lock          = threading.RLock()
        self._rules:  List[Rule]         = []
        self._zones:  List[CaptureZone]  = []
        self._observer = None

    # ── Properties (callable by RuleEngine) ──────────────────────────────────

    @property
    def rules(self) -> List[Rule]:
        with self._lock:
            return list(self._rules)

    @property
    def capture_zones(self) -> List[CaptureZone]:
        with self._lock:
            return list(self._zones)

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load and validate from disk. Raises on failure (use at startup)."""
        rules, zones = self._parse_file()
        with self._lock:
            self._rules = rules
            self._zones = zones
        log.info(
            "RulesLoader: loaded %d rule(s), %d capture zone(s) from %s",
            len(rules), len(zones), self._path,
        )

    def _parse_file(self) -> Tuple[List[Rule], List[CaptureZone]]:
        raw  = self._path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return parse_rules(data, self._valid_classes)

    def _try_reload(self) -> None:
        """Hot-reload: keep last-good config on any error."""
        try:
            rules, zones = self._parse_file()
            with self._lock:
                self._rules = rules
                self._zones = zones
            log.info(
                "RulesLoader: hot-reloaded %d rule(s), %d capture zone(s)",
                len(rules), len(zones),
            )
        except Exception as exc:
            log.error(
                "RulesLoader: invalid config — keeping last-good rules. Error: %s", exc
            )

    # ── Watchdog ──────────────────────────────────────────────────────────────

    def start_watching(self) -> None:
        """Start a background watchdog thread that hot-reloads on file change."""
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            log.warning("RulesLoader: watchdog not installed — hot-reload disabled")
            return

        loader = self

        class _Handler(FileSystemEventHandler):
            def _on_rules_event(self, path: str) -> None:
                if Path(path).resolve() == loader._path.resolve():
                    time.sleep(loader._RELOAD_DELAY_SEC)
                    loader._try_reload()

            def on_modified(self, event) -> None:
                if not event.is_directory:
                    self._on_rules_event(event.src_path)

            def on_created(self, event) -> None:
                # Catches atomic-rename saves (vim, nano, etc.)
                if not event.is_directory:
                    self._on_rules_event(event.src_path)

        self._observer = Observer()
        self._observer.schedule(
            _Handler(), str(self._path.parent), recursive=False
        )
        self._observer.start()
        log.info("RulesLoader: watching %s for changes", self._path)

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=3.0)
            self._observer = None
