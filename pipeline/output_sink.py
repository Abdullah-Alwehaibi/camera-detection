"""Detection event output -- sends to jetson-gstreamer-testing's AI_EVENTS_SOCKET.

Protocol (see ai-interface/protocol.h in jetson-gstreamer-testing):
  AI_EVENTS_SOCKET (default /tmp/ai_events.sock), Unix SOCK_STREAM,
  newline-delimited JSON, one detection event per line. events_adapter.py
  reads the "type"/"confidence" fields to drive ONVIF event state; extra
  fields are ignored by it but kept here for our own evidence trail.

Event schema sent by OutputSink.send_event():
  {
    "type":         "vehicle",
    "confidence":   <float 0-1>,            # detector confidence
    "bbox":         [x, y, w, h],           # pixels, AI-frame coordinates
    "track_id":     <int>,                  # tracker ID
    "timestamp_ns": <int>,                  # CLOCK_MONOTONIC ns
    "class":        <str>,                  # COCO class name, e.g. "car"
    "zone_id":      <str>,                  # trigger zone that fired
    "crop_path":    <str>,                  # path to saved evidence JPEG
    "timestamp":    <str>                   # ISO-8601 wall-clock time
  }

If AI_EVENTS_SOCKET does not exist (adapter not running) or the connection
drops, events are appended as JSON lines to a local fallback log instead.
Reconnection is retried on each subsequent send.
"""

import json
import logging
import os
import socket
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVENTS_SOCKET = "/tmp/ai_events.sock"
DEFAULT_FALLBACK_LOG = REPO_ROOT / "logs" / "output_events.jsonl"


class OutputSink:
    def __init__(self, socket_path=DEFAULT_EVENTS_SOCKET, fallback_log=DEFAULT_FALLBACK_LOG):
        self.socket_path = socket_path
        self.fallback_log = Path(fallback_log)
        self._sock = None

    def _connect(self):
        if self._sock is not None:
            return True
        if not os.path.exists(self.socket_path):
            return False
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self.socket_path)
        except OSError as e:
            log.warning("OutputSink: connect to %s failed: %s", self.socket_path, e)
            return False
        self._sock = sock
        log.info("OutputSink: connected to %s", self.socket_path)
        return True

    def _write_fallback(self, line):
        self.fallback_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.fallback_log, "a") as f:
            f.write(line + "\n")

    def send_event(self, event):
        """Send one detection event (dict). Falls back to local JSONL log
        if AI_EVENTS_SOCKET is unavailable."""
        line = json.dumps(event)
        if self._connect():
            try:
                self._sock.sendall((line + "\n").encode("utf-8"))
                return
            except OSError as e:
                log.warning("OutputSink: send failed (%s), falling back to %s", e, self.fallback_log)
                self._sock.close()
                self._sock = None
        self._write_fallback(line)

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
