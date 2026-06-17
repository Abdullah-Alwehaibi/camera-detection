"""
integration/frame_reader.py

GStreamer shmsrc → GPU NV12→BGR frame reader with reconnect and metrics.

Production path (shmsrc):
  shmsrc (NV12) → appsink → pycuda H2D → CUDA NV12→BGR kernel → FramePacket
  No videoconvert element; no CPU color-convert (satisfies §1.6).

videotestsrc path (testing only):
  videotestsrc → videoconvert → NV12 → appsink → same GPU path.
  CPU color-convert inside GStreamer is acceptable for synthetic frames.

Threading:
  _GstThread  — calls cap.read() in a loop; reconnects on failure; puts raw
                NV12 arrays into Queue(1) (discards stale frames if main is slow).
  Main thread — pulls from queue, runs CUDA kernel, yields FramePackets.
  CUDA calls stay on the main thread → no cross-thread context push/pop.

Clock timestamps:
  CLOCK_MONOTONIC and CLOCK_REALTIME are sampled back-to-back at startup
  (3-sample median). Every FramePacket carries both so downstream events
  can carry wall-clock time the backend can align to the 4K recording.
"""

from __future__ import annotations

import ctypes
import datetime
import logging
import queue
import threading
import time
from collections import deque
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from integration.contract import EXPECTED_CAPS

log = logging.getLogger(__name__)

# ── POSIX clock helpers ───────────────────────────────────────────────────────

_CLOCK_REALTIME  = 0
_CLOCK_MONOTONIC = 1


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


_librt = ctypes.CDLL("librt.so.1", use_errno=True)


def _clock_ns(clock_id: int) -> int:
    ts = _Timespec()
    if _librt.clock_gettime(clock_id, ctypes.byref(ts)) != 0:
        raise OSError(ctypes.get_errno(), "clock_gettime failed")
    return ts.tv_sec * 1_000_000_000 + ts.tv_nsec


def measure_clock_offset() -> Tuple[int, int]:
    """Read both clocks back-to-back three times; return (mono_ns, offset_ns).

    offset_ns = realtime_ns - mono_ns. Add to any monotonic timestamp to get
    wall-clock nanoseconds. Three samples, median taken to reduce jitter.
    """
    samples = []
    for _ in range(3):
        m = _clock_ns(_CLOCK_MONOTONIC)
        r = _clock_ns(_CLOCK_REALTIME)
        samples.append((m, r - m))
    samples.sort(key=lambda s: s[0])
    mono_ns, offset_ns = samples[1]
    log.info(
        "clock offset: REALTIME = MONOTONIC + %d ns  (wall %s)",
        offset_ns,
        datetime.datetime.fromtimestamp((mono_ns + offset_ns) / 1e9).isoformat(),
    )
    return mono_ns, offset_ns


# ── CUDA NV12→BGR kernel ──────────────────────────────────────────────────────
# Single contiguous NV12 buffer layout:
#   src[0 .. W*H-1]         Y plane  (one byte per pixel)
#   src[W*H .. W*H*3/2-1]   UV plane (interleaved Cb/Cr, half height, full width)
#
# BT.601 limited range (Y: 16-235, UV: 16-240) → full-range BGR uint8.
# __ldg() routes through the texture cache (read-only, benefits strided UV access).

_NV12_BGR_CUDA = r"""
__global__ void nv12_to_bgr(
    const unsigned char* __restrict__ src,
    unsigned char*       __restrict__ bgr,
    const int width,
    const int height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float Y = (float)__ldg(&src[y * width + x]) - 16.0f;

    const int uv_base = width * height;
    const int uv_col  = (x & ~1);
    const int uv_off  = uv_base + (y >> 1) * width + uv_col;
    const float U = (float)__ldg(&src[uv_off    ]) - 128.0f;
    const float V = (float)__ldg(&src[uv_off + 1]) - 128.0f;

    const float s = 1.164383f;
    const float R = s * Y                  + 1.596027f * V;
    const float G = s * Y - 0.391762f * U  - 0.812969f * V;
    const float B = s * Y + 2.017232f * U;

    const float inv255 = 0.003921569f;
    const int out = (y * width + x) * 3;
    bgr[out + 0] = (unsigned char)(__saturatef(B * inv255) * 255.0f);
    bgr[out + 1] = (unsigned char)(__saturatef(G * inv255) * 255.0f);
    bgr[out + 2] = (unsigned char)(__saturatef(R * inv255) * 255.0f);
}
"""


class _GpuConverter:
    """Compiles the NV12→BGR kernel; owns all pinned and device memory.

    Ping-pong pinned host buffers:
      After the kernel runs, d_bgr is async-copied to h_bgr[ping].
      The caller receives a numpy view of h_bgr[ping] (valid until two more
      convert() calls cycle back to the same buffer — safe in a sequential loop).
    """

    _BLOCK = (32, 8, 1)

    def __init__(self, width: int, height: int, cuda_ctx) -> None:
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule

        self._cuda = cuda
        self._ctx  = cuda_ctx
        self._w, self._h = width, height

        nv12_bytes = width * height * 3 // 2
        bgr_bytes  = width * height * 3

        self._ctx.push()
        try:
            mod = SourceModule(_NV12_BGR_CUDA, no_extern_c=True)
            self._fn = mod.get_function("nv12_to_bgr")

            # Pinned input: fast write-combined H2D path
            self._h_nv12 = cuda.pagelocked_empty(nv12_bytes, dtype=np.uint8)
            self._d_nv12 = cuda.mem_alloc(nv12_bytes)
            self._d_bgr  = cuda.mem_alloc(bgr_bytes)

            # Ping-pong pinned output buffers (async D2H)
            self._h_bgr = [
                cuda.pagelocked_empty(bgr_bytes, dtype=np.uint8),
                cuda.pagelocked_empty(bgr_bytes, dtype=np.uint8),
            ]
            self._ping = 0

            self._stream = cuda.Stream()

            gx = (width  + self._BLOCK[0] - 1) // self._BLOCK[0]
            gy = (height + self._BLOCK[1] - 1) // self._BLOCK[1]
            self._grid = (gx, gy, 1)

            log.info(
                "NV12→BGR kernel compiled. grid=%s block=%s  "
                "device buffers: NV12=%d KB, BGR=%d KB",
                self._grid, self._BLOCK,
                nv12_bytes // 1024, bgr_bytes // 1024,
            )
        finally:
            self._ctx.pop()

    def convert(self, nv12: np.ndarray) -> Tuple:
        """H2D upload + kernel + async D2H. Returns (d_bgr, cpu_bgr_view).

        cpu_bgr_view is a (H, W, 3) uint8 numpy view into pinned memory.
        It is valid until the same ping-pong slot is reused (2 frames later).
        Call .copy() if you need to hold it across frames.
        """
        self._ctx.push()
        try:
            np.copyto(self._h_nv12, nv12.ravel())
            self._cuda.memcpy_htod_async(self._d_nv12, self._h_nv12, self._stream)

            self._fn(
                self._d_nv12,
                self._d_bgr,
                np.int32(self._w),
                np.int32(self._h),
                block=self._BLOCK,
                grid=self._grid,
                stream=self._stream,
            )

            buf = self._h_bgr[self._ping]
            self._cuda.memcpy_dtoh_async(buf, self._d_bgr, self._stream)
            self._stream.synchronize()

            view = buf.reshape((self._h, self._w, 3))
            self._ping ^= 1
            return self._d_bgr, view
        finally:
            self._ctx.pop()


# ── FramePacket ───────────────────────────────────────────────────────────────

class FramePacket:
    """One decoded frame with GPU tensor, timestamps, and lazy CPU access.

    d_bgr       — device pointer to BGR uint8 (H, W, 3). Reused next frame;
                  read it or pass it to the detector before calling frames() again.
    cpu_bgr     — property; returns a .copy() of the current pinned buffer.
                  Safe to hold across frames. Call only within the current
                  loop iteration (before the next packet is yielded).
    ts_mono_ns  — CLOCK_MONOTONIC nanoseconds at frame capture.
    ts_real_ns  — CLOCK_REALTIME nanoseconds (mono + startup offset).
    capture_time— datetime.datetime equivalent of ts_real_ns (local tz).
    """

    __slots__ = (
        "frame_id", "capture_time", "ts_mono_ns", "ts_real_ns",
        "width", "height", "d_bgr", "_pinned_view",
    )

    def __init__(
        self,
        frame_id: int,
        capture_time: datetime.datetime,
        ts_mono_ns: int,
        ts_real_ns: int,
        width: int,
        height: int,
        d_bgr,
        pinned_view: np.ndarray,
    ) -> None:
        self.frame_id     = frame_id
        self.capture_time = capture_time
        self.ts_mono_ns   = ts_mono_ns
        self.ts_real_ns   = ts_real_ns
        self.width        = width
        self.height       = height
        self.d_bgr        = d_bgr
        self._pinned_view = pinned_view

    @property
    def cpu_bgr(self) -> np.ndarray:
        """BGR frame as (H, W, 3) uint8 ndarray. Returns a copy."""
        return self._pinned_view.copy()


# ── Metrics ───────────────────────────────────────────────────────────────────

class ReaderMetrics:
    """Lock-free (GIL-protected) counters updated from two threads."""

    def __init__(self, fps_window_sec: float = 5.0) -> None:
        self.frames_in      = 0
        self.frames_dropped = 0
        self.reconnects     = 0
        self._fps_window    = deque()
        self._fps_win_sec   = fps_window_sec
        self.fps_measured   = 0.0

    def record_frame(self) -> None:
        self.frames_in += 1
        now = time.monotonic()
        self._fps_window.append(now)
        cutoff = now - self._fps_win_sec
        while self._fps_window and self._fps_window[0] < cutoff:
            self._fps_window.popleft()
        n = len(self._fps_window)
        if n > 1:
            span = self._fps_window[-1] - self._fps_window[0]
            self.fps_measured = (n - 1) / span if span > 0 else 0.0

    def record_drop(self) -> None:
        self.frames_dropped += 1

    def record_reconnect(self) -> None:
        self.reconnects += 1

    def snapshot(self) -> dict:
        return {
            "frames_in":      self.frames_in,
            "frames_dropped": self.frames_dropped,
            "reconnects":     self.reconnects,
            "fps_measured":   round(self.fps_measured, 2),
        }


# ── FrameReader ───────────────────────────────────────────────────────────────

class FrameReader:
    """Reads NV12 frames from the grabber via GStreamer, converts to GPU BGR.

    Args:
        config: full pipeline.yaml dict (uses frame_socket, expected_caps,
                and source.mode / source.path for file mode).
        cuda_ctx: pycuda primary context (retain_primary_context()).
                  If None, GPU conversion is skipped and cpu_bgr falls back
                  to cv2.cvtColor on CPU (dev/test use only).
    """

    _BACKOFF_SEC  = [1, 2, 4, 8, 16, 30]
    _QUEUE_WARN   = 5.0   # seconds without a frame before warning

    def __init__(self, config: dict, cuda_ctx=None) -> None:
        self._socket  = config.get("frame_socket", EXPECTED_CAPS.get("socket", "/tmp/ai_frames.sock"))
        self._caps    = {**EXPECTED_CAPS, **config.get("expected_caps", {})}
        self._src_cfg = config.get("source", {})
        self._mode    = self._src_cfg.get("mode", "shmsrc")
        self._cuda_ctx = cuda_ctx

        self.metrics = ReaderMetrics()
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._converter: Optional[_GpuConverter] = None
        self._mono_offset = 0  # realtime_ns - monotonic_ns

        self._thread = threading.Thread(
            target=self._gst_loop, name="gst-reader", daemon=True
        )

    # ── Pipeline builders ────────────────────────────────────────────────────

    def _build_pipeline(self) -> str:
        w   = self._caps["width"]
        h   = self._caps["height"]
        fps = self._caps["fps"]
        fmt = self._caps["format"]          # NV12
        nv12_caps = f"video/x-raw,format={fmt},width={w},height={h},framerate={fps}/1"
        sink = "appsink drop=true max-buffers=1 sync=false"

        if self._mode == "shmsrc":
            # Production: no CPU conversion element
            return (
                f"shmsrc socket-path={self._socket} is-live=true "
                f"! {nv12_caps} ! {sink}"
            )
        elif self._mode == "videotestsrc":
            # Synthetic test; videoconvert inside GStreamer is acceptable here
            return (
                f"videotestsrc is-live=true "
                f"! videoconvert ! {nv12_caps} ! {sink}"
            )
        elif self._mode == "file":
            path = self._src_cfg.get("path", "")
            loop = self._src_cfg.get("loop", False)
            loop_el = "! videoflip method=identity loop=true " if loop else ""
            # nvvidconv: hardware decode NVMM → NV12 system memory
            return (
                f'filesrc location="{path}" ! decodebin ! nvvidconv '
                f"! {nv12_caps} {loop_el}! {sink}"
            )
        else:
            raise ValueError(f"Unknown source mode: {self._mode!r}")

    # ── Caps validation ──────────────────────────────────────────────────────

    def _validate_frame_shape(self, frame: np.ndarray) -> bool:
        """Return True if frame looks like raw NV12, False if cv2 pre-converted it."""
        w, h = self._caps["width"], self._caps["height"]
        if frame.shape == (h * 3 // 2, w):
            return True          # raw NV12 — GPU path
        if frame.shape == (h, w, 3):
            log.warning(
                "GStreamer backend returned BGR (shape %s) instead of NV12 %s. "
                "GPU NV12→BGR kernel disabled; using cv2.cvtColor fallback.",
                frame.shape, (h * 3 // 2, w),
            )
            return False         # cv2 already converted — CPU path
        raise RuntimeError(
            f"Unexpected frame shape {frame.shape}. "
            f"Expected NV12 {(h * 3 // 2, w)} or BGR {(h, w, 3)}. "
            "Grabber may have drifted from expected caps."
        )

    # ── Background GStreamer thread ──────────────────────────────────────────

    def _gst_loop(self) -> None:
        backoff_idx = 0
        cap = None

        while not self._stop.is_set():
            try:
                pipeline = self._build_pipeline()
                log.info("FrameReader: opening pipeline [%s]", pipeline)
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if not cap.isOpened():
                    raise RuntimeError("cv2.VideoCapture: pipeline failed to open")

                self.metrics.record_reconnect()
                backoff_idx = 0
                log.info("FrameReader: pipeline open (reconnects=%d)", self.metrics.reconnects)

                while not self._stop.is_set():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        log.warning("FrameReader: cap.read() returned no frame — EOS/error")
                        break

                    try:
                        self._q.put_nowait(frame)
                    except queue.Full:
                        self.metrics.record_drop()

            except Exception as exc:
                log.error("FrameReader: %s", exc)
            finally:
                if cap is not None:
                    cap.release()
                    cap = None

            if self._stop.is_set():
                break

            delay = self._BACKOFF_SEC[min(backoff_idx, len(self._BACKOFF_SEC) - 1)]
            backoff_idx += 1
            log.info("FrameReader: reconnecting in %d s …", delay)
            self._stop.wait(delay)

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self) -> "FrameReader":
        _, self._mono_offset = measure_clock_offset()

        if self._cuda_ctx is not None:
            self._converter = _GpuConverter(
                self._caps["width"], self._caps["height"], self._cuda_ctx
            )

        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def frames(self) -> Iterator[FramePacket]:
        """Yield FramePackets. Blocks until each frame arrives.

        The main thread owns all CUDA calls here; the background thread
        only does cap.read() (CPU).
        """
        if not self._thread.is_alive():
            self.start()

        nv12_mode: Optional[bool] = None   # True=raw NV12, False=BGR from cv2
        frame_id = 0

        while not self._stop.is_set():
            try:
                raw = self._q.get(timeout=self._QUEUE_WARN)
            except queue.Empty:
                log.warning("FrameReader: no frame for %.1f s", self._QUEUE_WARN)
                continue

            # Detect caps format on first frame
            if nv12_mode is None:
                nv12_mode = self._validate_frame_shape(raw)
                if not nv12_mode and self._converter is not None:
                    self._converter = None   # disable GPU path if cv2 pre-converted

            ts_mono = _clock_ns(_CLOCK_MONOTONIC)
            ts_real = ts_mono + self._mono_offset
            capture_time = datetime.datetime.fromtimestamp(ts_real / 1e9)

            frame_id += 1
            self.metrics.record_frame()

            if self._converter is not None and nv12_mode:
                d_bgr, pinned_view = self._converter.convert(raw)
            else:
                # CPU fallback: NV12→BGR via cv2 (or passthrough if already BGR)
                if nv12_mode:
                    cpu = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_NV12)
                else:
                    cpu = raw
                d_bgr = None
                pinned_view = cpu

            yield FramePacket(
                frame_id     = frame_id,
                capture_time = capture_time,
                ts_mono_ns   = ts_mono,
                ts_real_ns   = ts_real,
                width        = self._caps["width"],
                height       = self._caps["height"],
                d_bgr        = d_bgr,
                pinned_view  = pinned_view,
            )

    def __enter__(self) -> "FrameReader":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()
