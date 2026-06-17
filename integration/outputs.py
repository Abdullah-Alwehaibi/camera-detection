"""
integration/outputs.py

Dual output channels (§4.3):

  1. Boolean path  — AF_UNIX SOCK_STREAM to AI_EVENTS_SOCKET.
     One JSON line per event, schema per §3.2. Lossy/best-effort;
     events_adapter.py handles cooldown and ONVIF mapping.
     Reconnects on each send attempt; no buffering (lossy by design).

  2. Rich sidecar  — pluggable backend (unix | mqtt | websocket).
     One JSON record per event with full metadata (bbox_4k, ts_realtime_ns,
     rule_id, class, subclass, direction, lane, snapshot ref …).
     Never routes through events_adapter.py.

Both channels respect dry_run=True (log only, no socket I/O) so preflight
and tests never emit real events to the NVR (constraint §5).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Optional

from integration.contract import (
    SIDECAR_SCHEMA_VERSION,
    SidecarBackend,
    validate_event,
)

log = logging.getLogger(__name__)

# ── Boolean event output ──────────────────────────────────────────────────────

class BooleanEventWriter:
    """Writes §3.2 events to AI_EVENTS_SOCKET (AF_UNIX SOCK_STREAM).

    Reconnects on every send if the socket is absent or the connection drops.
    Falls back to a local JSONL log on failure.
    Validates every event against the contract schema before sending.
    """

    def __init__(
        self,
        socket_path: str = "/tmp/ai_events.sock",
        fallback_log: Optional[Path] = None,
        dry_run: bool = False,
    ) -> None:
        self._path       = socket_path
        self._fallback   = Path(fallback_log) if fallback_log else None
        self._dry_run    = dry_run
        self._sock: Optional[socket.socket] = None
        self._lock       = threading.Lock()

    def _connect(self) -> bool:
        if self._sock is not None:
            return True
        if not os.path.exists(self._path):
            return False
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(self._path)
            self._sock = s
            log.info("BooleanEventWriter: connected to %s", self._path)
            return True
        except OSError as e:
            log.warning("BooleanEventWriter: connect failed: %s", e)
            return False

    def _write_fallback(self, line: str) -> None:
        if self._fallback is None:
            return
        self._fallback.parent.mkdir(parents=True, exist_ok=True)
        with open(self._fallback, "a") as f:
            f.write(line + "\n")

    def send(self, event: dict) -> None:
        """Validate and send one boolean event. Thread-safe."""
        validate_event(event)
        line = json.dumps(event)

        if self._dry_run:
            log.info("[DRY-RUN] boolean event: %s", line)
            return

        with self._lock:
            if self._connect():
                try:
                    self._sock.sendall((line + "\n").encode())
                    return
                except OSError as e:
                    log.warning("BooleanEventWriter: send failed (%s), using fallback", e)
                    self._sock.close()
                    self._sock = None
            self._write_fallback(line)

    def close(self) -> None:
        with self._lock:
            if self._sock:
                self._sock.close()
                self._sock = None

    def __enter__(self) -> "BooleanEventWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Rich sidecar backends ─────────────────────────────────────────────────────

class _UnixSidecarBackend:
    """AF_UNIX SOCK_STREAM sidecar; reconnects per send."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._sock: Optional[socket.socket] = None

    def connect(self) -> bool:
        if self._sock is not None:
            return True
        if not os.path.exists(self._path):
            return False
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(self._path)
            self._sock = s
            log.info("Sidecar(unix): connected to %s", self._path)
            return True
        except OSError as e:
            log.warning("Sidecar(unix): connect failed: %s", e)
            return False

    def send(self, line: str) -> bool:
        if self.connect():
            try:
                self._sock.sendall((line + "\n").encode())
                return True
            except OSError as e:
                log.warning("Sidecar(unix): send failed: %s", e)
                self._sock.close()
                self._sock = None
        return False

    def close(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None


class _MqttSidecarBackend:
    """paho-mqtt sidecar backend. Connects lazily on first send."""

    def __init__(self, broker: str, port: int, topic: str, qos: int = 1) -> None:
        self._broker = broker
        self._port   = port
        self._topic  = topic
        self._qos    = qos
        self._client = None

    def _ensure_connected(self) -> bool:
        if self._client is not None:
            return True
        try:
            import paho.mqtt.client as mqtt
            c = mqtt.Client()
            c.connect(self._broker, self._port, keepalive=60)
            c.loop_start()
            self._client = c
            log.info("Sidecar(mqtt): connected to %s:%d", self._broker, self._port)
            return True
        except Exception as e:
            log.warning("Sidecar(mqtt): connect failed: %s", e)
            return False

    def send(self, line: str) -> bool:
        if self._ensure_connected():
            try:
                self._client.publish(self._topic, line, qos=self._qos)
                return True
            except Exception as e:
                log.warning("Sidecar(mqtt): publish failed: %s", e)
                self._client = None
        return False

    def close(self) -> None:
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._client = None


class _WebSocketSidecarBackend:
    """Asyncio websocket sidecar — runs its own event loop in a daemon thread."""

    def __init__(self, host: str, port: int) -> None:
        import asyncio
        import queue as _q

        self._host  = host
        self._port  = port
        self._queue: "_q.Queue[Optional[str]]" = _q.Queue()
        self._loop  = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="ws-sidecar", daemon=True
        )
        self._thread.start()
        self._send_failures = 0

    def _run_loop(self) -> None:
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        try:
            import websockets
            import asyncio

            clients: set = set()

            async def handler(ws, _path=""):
                clients.add(ws)
                try:
                    await ws.wait_closed()
                finally:
                    clients.discard(ws)

            server = await websockets.serve(handler, self._host, self._port)
            log.info("Sidecar(ws): listening on %s:%d", self._host, self._port)

            while True:
                try:
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._queue.get(timeout=1.0)
                    )
                except Exception:
                    continue
                if line is None:
                    break
                dead = set()
                for ws in list(clients):
                    try:
                        await ws.send(line)
                    except Exception:
                        dead.add(ws)
                clients -= dead

            server.close()
        except Exception as e:
            log.error("Sidecar(ws): server error: %s", e)

    def send(self, line: str) -> bool:
        try:
            self._queue.put_nowait(line)
            return True
        except Exception:
            self._send_failures += 1
            return False

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=3.0)


def _build_sidecar_backend(cfg: dict):
    backend = cfg.get("backend", SidecarBackend.UNIX)
    if backend == SidecarBackend.UNIX:
        return _UnixSidecarBackend(cfg.get("path", "/tmp/ai_sidecar.sock"))
    elif backend == SidecarBackend.MQTT:
        return _MqttSidecarBackend(
            broker=cfg.get("mqtt_broker", "localhost"),
            port=int(cfg.get("mqtt_port", 1883)),
            topic=cfg.get("mqtt_topic", "camera/ai/sidecar"),
            qos=int(cfg.get("mqtt_qos", 1)),
        )
    elif backend == SidecarBackend.WEBSOCKET:
        return _WebSocketSidecarBackend(
            host=cfg.get("ws_host", "0.0.0.0"),
            port=int(cfg.get("ws_port", 8765)),
        )
    else:
        raise ValueError(f"Unknown sidecar backend: {backend!r}")


# ── Rich sidecar writer ───────────────────────────────────────────────────────

class SidecarWriter:
    """Sends rich per-event JSON records through the configured backend.

    Record schema (§4.3):
    {
      "schema_version": "1.0",
      "event":          "line_crossing",
      "track_id":       42,
      "class":          "vehicle",
      "subclass":       "truck",
      "confidence":     0.93,
      "bbox_1080":      [x, y, w, h],
      "bbox_4k":        [x, y, w, h],
      "rule_id":        "zone_abc",
      "direction":      "northbound",
      "lane":           2,
      "snapshot":       {"mode": "backend_pull", "ref": null, "jpeg_path": null},
      "ts_monotonic_ns": 0,
      "ts_realtime_ns":  0
    }
    """

    def __init__(
        self,
        sidecar_cfg: dict,
        fallback_log: Optional[Path] = None,
        dry_run: bool = False,
    ) -> None:
        self._backend   = _build_sidecar_backend(sidecar_cfg)
        self._fallback  = Path(fallback_log) if fallback_log else None
        self._dry_run   = dry_run
        self._lock      = threading.Lock()
        self.send_failures = 0

    def send(self, record: dict) -> None:
        """Send one sidecar record. Thread-safe."""
        record.setdefault("schema_version", SIDECAR_SCHEMA_VERSION)
        line = json.dumps(record)

        if self._dry_run:
            log.info("[DRY-RUN] sidecar: %s", line)
            return

        with self._lock:
            ok = self._backend.send(line)
            if not ok:
                self.send_failures += 1
                if self._fallback:
                    self._fallback.parent.mkdir(parents=True, exist_ok=True)
                    with open(self._fallback, "a") as f:
                        f.write(line + "\n")

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "SidecarWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Convenience factory ───────────────────────────────────────────────────────

def build_outputs(config: dict, dry_run: bool = False):
    """Build BooleanEventWriter + SidecarWriter from pipeline.yaml config dict.

    Returns (boolean_writer, sidecar_writer). Both are context managers.
    dry_run overrides config['dry_run'] when True.
    """
    dry = dry_run or config.get("dry_run", False)
    log_dir = Path(config.get("log_dir", "logs"))

    boolean = BooleanEventWriter(
        socket_path  = config.get("events_socket", "/tmp/ai_events.sock"),
        fallback_log = log_dir / "output_events.jsonl",
        dry_run      = dry,
    )
    sidecar = SidecarWriter(
        sidecar_cfg  = config.get("sidecar", {"backend": "unix"}),
        fallback_log = log_dir / "sidecar_events.jsonl",
        dry_run      = dry,
    )
    return boolean, sidecar
