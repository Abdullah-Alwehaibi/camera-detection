"""Frame source abstraction over GStreamer.

Wraps a GStreamer pipeline (built as a gst-launch-style string and opened
via OpenCV's GStreamer backend) so that swapping the upstream source is a
config change, not a rewrite of anything downstream. frames() always
yields BGR numpy arrays at config width x height, regardless of mode.

Three modes are supported:

- "videotestsrc": synthetic test pattern, used when no real frame source
  is available (or not yet accessible -- see below).

- "shmsrc": reads raw frames from a GStreamer shmsink over a Unix socket --
  this is the AI-tap branch of jetson-gstreamer-testing's send_stream.sh
  (see ai-interface/protocol.h in that repo). Frames arrive as NV12 and are
  converted to BGR via videoconvert.

- "file": reads from a pre-recorded video, or every video file in a
  directory (sorted by filename, optionally looping) -- for testing
  detection/tracking/trigger-zones against real footage with vehicles, the
  same way the live "shmsrc" AI tap feeds frames. Each file is decoded via
  decodebin and scaled/converted to the configured width x height BGR, so
  config/trigger_zones.json's coordinates (calibrated for 1920x1080) still
  line up if the test footage is also 1920x1080.

## Real source status

jetson-gstreamer-testing's AI tap is live at /tmp/ai_frames.sock
(NV12, 1920x1080 @ 5fps, per stream.conf AI_FRAME_* defaults -- confirmed
from the running gst-launch command line). However the socket is
`srw-r----- aaeon:aaeon` and this user (abdullah) is not in the `aaeon`
group, so connecting currently fails with EACCES. Once that's resolved
(e.g. `sudo usermod -aG aaeon abdullah` + new session), flip
config/pipeline.json source.mode to "shmsrc" -- caps already match.
"""

import logging
from pathlib import Path

import cv2

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


class FrameSource:
    def __init__(self, config):
        self.config = config
        self.width = int(config["width"])
        self.height = int(config["height"])
        self.fps = int(config.get("fps", 30))
        self.mode = config.get("mode", "videotestsrc")
        self._cap = None

        if self.mode == "file":
            self._file_paths = self._resolve_file_paths(config["path"])
            self._loop = bool(config.get("loop", False))
            self._file_idx = 0
        else:
            self._pipeline_str = self._build_pipeline(config)

    def _resolve_file_paths(self, path):
        p = Path(path)
        if p.is_dir():
            paths = sorted(q for q in p.iterdir() if q.suffix.lower() in VIDEO_EXTENSIONS)
            if not paths:
                raise ValueError(f"No video files found in {p}")
            return paths
        if not p.is_file():
            raise ValueError(f"Video file not found: {p}")
        return [p]

    def _build_pipeline(self, config):
        mode = config.get("mode", "videotestsrc")
        out_caps = (
            f"video/x-raw,format=BGR,width={self.width},height={self.height},"
            f"framerate={self.fps}/1"
        )

        if mode == "videotestsrc":
            return f"videotestsrc is-live=true ! {out_caps} ! appsink drop=true max-buffers=1 sync=false"
        elif mode == "shmsrc":
            socket_path = config["socket_path"]
            src_format = config.get("source_format", "NV12")
            in_caps = (
                f"video/x-raw,format={src_format},width={self.width},height={self.height},"
                f"framerate={self.fps}/1"
            )
            return (
                f"shmsrc socket-path={socket_path} is-live=true "
                f"! {in_caps} ! videoconvert ! {out_caps} ! appsink drop=true max-buffers=1 sync=false"
            )
        else:
            raise ValueError(f"Unknown FrameSource mode: {mode!r}")

    def _build_file_pipeline(self, path):
        # decodebin picks nvv4l2decoder for H.264/H.265 on Jetson, which
        # outputs NVMM (hardware) buffers that plain videoconvert can't
        # negotiate -- nvvidconv both moves them to system memory and does
        # the resize to the configured width x height in one step.
        # No framerate constraint here -- decodebin's output framerate
        # depends on the source file, and with sync=false frames are
        # delivered as fast as they can be decoded.
        return (
            f'filesrc location="{path}" ! decodebin ! nvvidconv '
            f"! video/x-raw,format=BGRx,width={self.width},height={self.height} "
            f"! videoconvert ! video/x-raw,format=BGR "
            f"! appsink drop=true max-buffers=1 sync=false"
        )

    def open(self):
        if self.mode == "file":
            if not self._open_next_file():
                raise RuntimeError("FrameSource: no video files to play")
        else:
            self._cap = cv2.VideoCapture(self._pipeline_str, cv2.CAP_GSTREAMER)
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open GStreamer pipeline: {self._pipeline_str}")
        return self

    def _open_next_file(self):
        """Open the next file in self._file_paths. Returns False if the
        list is exhausted and not looping."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        if self._file_idx >= len(self._file_paths):
            if not self._loop:
                return False
            self._file_idx = 0

        path = self._file_paths[self._file_idx]
        self._file_idx += 1

        pipeline_str = self._build_file_pipeline(path)
        log.info("FrameSource: playing %s", path)
        cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {path}")
        self._cap = cap
        return True

    def frames(self):
        """Yield BGR numpy frames until the source ends or errors.

        In "file" mode, advances to the next file (or loops back to the
        first) when the current one ends; in other modes, ends when the
        pipeline does.
        """
        if self._cap is None:
            self.open()

        while True:
            ret, frame = self._cap.read()
            if not ret:
                if self.mode == "file" and self._open_next_file():
                    continue
                break
            yield frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
