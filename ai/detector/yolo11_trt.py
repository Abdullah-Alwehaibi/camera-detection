"""
ai/detector/yolo11_trt.py

YOLO11 detector backends for the AI pipeline.

Two classes:
  Yolo11TrtDetector  — TensorRT FP16 engine (.engine file); GPU inference.
  Yolo11PtDetector   — ultralytics .pt model (CPU); dev/fallback only.

Both implement Detector.detect() — raw detections only, no track IDs.
Tracking is handled by a Tracker (ai/tracker/) composed at the pipeline level.

GPU inference path (Yolo11TrtDetector + FramePacket with d_bgr):
  d_bgr (GPU BGR uint8) → _GpuPreprocessor CUDA kernel → d_input (float32 CHW)
  → TRT execute_async_v2 → d_output → h_output (CPU)
  → NMS → scale_boxes
  No CPU color-convert; no H2D for the input tensor.

CPU fallback (FramePacket without d_bgr, or raw numpy BGR):
  cpu_bgr → _CpuPreprocessor (letterbox + normalize) → h_input
  → H2D → TRT → D2H → NMS → scale_boxes

aarch64 note: `import torch` must be the first import in any entrypoint
that eventually touches cv2 or gi, or torch's bundled libgomp fails with
"cannot allocate memory in static TLS block" (see CLAUDE.md). This module
is imported early via ai/detector/__init__.py so torch is always first.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch  # noqa: F401 — aarch64 import-order workaround, see module docstring
import cv2
import numpy as np
from ultralytics.utils import ops

from ai.detector.base import Detector

log = logging.getLogger(__name__)

# ── COCO-80 class names ───────────────────────────────────────────────────────
# Hard-coded here because the .engine backend has no YOLO object to read .names
# from. Matches ultralytics' YOLO11 output order exactly.
COCO_NAMES: Dict[int, str] = {
    0: "person",       1: "bicycle",      2: "car",           3: "motorcycle",
    4: "airplane",     5: "bus",          6: "train",         7: "truck",
    8: "boat",         9: "traffic light",10: "fire hydrant", 11: "stop sign",
    12: "parking meter",13: "bench",      14: "bird",         15: "cat",
    16: "dog",         17: "horse",       18: "sheep",        19: "cow",
    20: "elephant",    21: "bear",        22: "zebra",        23: "giraffe",
    24: "backpack",    25: "umbrella",    26: "handbag",      27: "tie",
    28: "suitcase",    29: "frisbee",     30: "skis",         31: "snowboard",
    32: "sports ball", 33: "kite",        34: "baseball bat", 35: "baseball glove",
    36: "skateboard",  37: "surfboard",   38: "tennis racket",39: "bottle",
    40: "wine glass",  41: "cup",         42: "fork",         43: "knife",
    44: "spoon",       45: "bowl",        46: "banana",       47: "apple",
    48: "sandwich",    49: "orange",      50: "broccoli",     51: "carrot",
    52: "hot dog",     53: "pizza",       54: "donut",        55: "cake",
    56: "chair",       57: "couch",       58: "potted plant", 59: "bed",
    60: "dining table",61: "toilet",      62: "tv",           63: "laptop",
    64: "mouse",       65: "remote",      66: "keyboard",     67: "cell phone",
    68: "microwave",   69: "oven",        70: "toaster",      71: "sink",
    72: "refrigerator",73: "book",        74: "clock",        75: "vase",
    76: "scissors",    77: "teddy bear",  78: "hair drier",   79: "toothbrush",
}

# Vehicle COCO class IDs (default; overridden by config)
DEFAULT_VEHICLE_CLASSES = [2, 3, 5, 7]   # car, motorcycle, bus, truck


# ── GPU letterbox + normalize kernel ─────────────────────────────────────────
# Fuses: bilinear letterbox resize + BGR→RGB + /255 + CHW transpose into one
# CUDA kernel that reads d_bgr (uint8) and writes directly to d_input (float32).
# Eliminates the CPU letterbox pass AND the H2D upload for the TRT input tensor.
#
# Letterbox: scale = min(H_out/H_in, W_out/W_in); pad gray (114/255) around
# the scaled image to reach H_out × W_out. Same logic as ultralytics LetterBox.

_LETTERBOX_KERNEL = r"""
__global__ void bgr_letterbox_rgb_f32_chw(
    const unsigned char* __restrict__ bgr_in,   /* (H_in, W_in, 3) BGR uint8 */
    float*               __restrict__ chw_out,  /* (1, 3, H_out, W_out) float32 */
    const int H_in,  const int W_in,
    const int H_out, const int W_out,
    const float inv_scale,   /* 1 / scale — source pixels per output pixel */
    const int pad_top,
    const int pad_left,
    const int scaled_h,      /* H_in * scale, rounded */
    const int scaled_w       /* W_in * scale, rounded */
) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= W_out || oy >= H_out) return;

    const float PAD = 0.44706f;   /* 114 / 255 — YOLO letterbox gray */
    float r, g, b;

    const int img_x = ox - pad_left;
    const int img_y = oy - pad_top;

    if (img_x < 0 || img_x >= scaled_w || img_y < 0 || img_y >= scaled_h) {
        r = g = b = PAD;
    } else {
        /* Bilinear sample from source, using inv_scale (avoids per-thread div) */
        const float sx = img_x * inv_scale;
        const float sy = img_y * inv_scale;

        const int x0 = (int)sx, y0 = (int)sy;
        const int x1 = min(x0 + 1, W_in - 1);
        const int y1 = min(y0 + 1, H_in - 1);
        const float fx = sx - x0, fy = sy - y0;
        const float ax = 1.0f - fx, ay = 1.0f - fy;

        /* 4-tap bilinear: read BGR, blend, convert to RGB, normalise */
        const unsigned char* p00 = bgr_in + (y0 * W_in + x0) * 3;
        const unsigned char* p01 = bgr_in + (y0 * W_in + x1) * 3;
        const unsigned char* p10 = bgr_in + (y1 * W_in + x0) * 3;
        const unsigned char* p11 = bgr_in + (y1 * W_in + x1) * 3;

        const float inv255 = 0.003921569f;
        b = (ax*ay*(float)p00[0] + fx*ay*(float)p01[0]
           + ax*fy*(float)p10[0] + fx*fy*(float)p11[0]) * inv255;
        g = (ax*ay*(float)p00[1] + fx*ay*(float)p01[1]
           + ax*fy*(float)p10[1] + fx*fy*(float)p11[1]) * inv255;
        r = (ax*ay*(float)p00[2] + fx*ay*(float)p01[2]
           + ax*fy*(float)p10[2] + fx*fy*(float)p11[2]) * inv255;
    }

    /* Write CHW: R plane, G plane, B plane */
    const int plane = H_out * W_out;
    const int idx   = oy * W_out + ox;
    chw_out[          idx] = r;
    chw_out[  plane + idx] = g;
    chw_out[2*plane + idx] = b;
}
"""


class _GpuPreprocessor:
    """Fused letterbox + BGR→RGB + normalize CUDA kernel.

    Reads from d_bgr (DeviceAllocation, owned by FrameReader._GpuConverter)
    and writes float32 CHW directly into d_input (DeviceAllocation, owned by
    _TensorRTEngine). No intermediate buffer needed; no H2D transfer.
    """

    _BLOCK = (32, 8, 1)

    def __init__(
        self,
        H_in: int, W_in: int,
        imgsz: int,
        d_input,            # _TensorRTEngine.d_input — write target
        cuda_ctx,
        stream,             # shared CUDA stream (same as TRT engine's)
    ) -> None:
        import os
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule

        self._cuda   = cuda
        self._ctx    = cuda_ctx
        self._stream = stream
        self._d_in   = d_input   # TRT input buffer (written by this kernel)

        # Letterbox geometry (precomputed; fixed for this H_in/W_in/imgsz)
        scale       = min(imgsz / H_in, imgsz / W_in)
        scaled_h    = round(H_in * scale)
        scaled_w    = round(W_in * scale)
        pad_top     = (imgsz - scaled_h) // 2
        pad_left    = (imgsz - scaled_w) // 2

        self._inv_scale  = np.float32(1.0 / scale)
        self._pad_top    = np.int32(pad_top)
        self._pad_left   = np.int32(pad_left)
        self._scaled_h   = np.int32(scaled_h)
        self._scaled_w   = np.int32(scaled_w)
        self._H_in       = np.int32(H_in)
        self._W_in       = np.int32(W_in)
        self._H_out      = np.int32(imgsz)
        self._W_out      = np.int32(imgsz)

        gx = (imgsz + self._BLOCK[0] - 1) // self._BLOCK[0]
        gy = (imgsz + self._BLOCK[1] - 1) // self._BLOCK[1]
        self._grid = (gx, gy, 1)

        # nvcc path: CUDA_ROOT env → /usr/local/cuda-11.4 fallback (JetPack 5.1.1)
        cuda_root = os.environ.get("CUDA_ROOT", "/usr/local/cuda-11.4")
        nvcc_path = os.path.join(cuda_root, "bin", "nvcc")

        self._ctx.push()
        try:
            mod = SourceModule(_LETTERBOX_KERNEL, nvcc=nvcc_path)
            self._fn = mod.get_function("bgr_letterbox_rgb_f32_chw")
        finally:
            self._ctx.pop()

        log.info(
            "GPU preprocessor: %dx%d → letterbox %dx%d "
            "(scale=%.4f pad_top=%d pad_left=%d) grid=%s",
            W_in, H_in, imgsz, imgsz, scale, pad_top, pad_left, self._grid,
        )

    def process(self, d_bgr) -> None:
        """Run the fused kernel. d_bgr → d_input (TRT buffer). Async on stream."""
        self._ctx.push()
        try:
            self._fn(
                d_bgr,
                self._d_in,
                self._H_in,  self._W_in,
                self._H_out, self._W_out,
                self._inv_scale,
                self._pad_top,
                self._pad_left,
                self._scaled_h,
                self._scaled_w,
                block  = self._BLOCK,
                grid   = self._grid,
                stream = self._stream,
            )
        finally:
            self._ctx.pop()


# ── CPU preprocessor (fallback) ───────────────────────────────────────────────

class _CpuPreprocessor:
    """CPU letterbox + normalize. Used when d_bgr is unavailable."""

    def __init__(self, H_in: int, W_in: int, imgsz: int) -> None:
        self._imgsz = imgsz
        scale       = min(imgsz / H_in, imgsz / W_in)
        self._new_w = round(W_in * scale)
        self._new_h = round(H_in * scale)
        self._pad_w = (imgsz - self._new_w) / 2
        self._pad_h = (imgsz - self._new_h) / 2

    def process(self, bgr: np.ndarray) -> np.ndarray:
        """Return float32 NCHW array ready for TRT inference."""
        # Resize
        resized = cv2.resize(bgr, (self._new_w, self._new_h),
                             interpolation=cv2.INTER_LINEAR)
        # Pad
        top    = round(self._pad_h - 0.1)
        bottom = round(self._pad_h + 0.1)
        left   = round(self._pad_w - 0.1)
        right  = round(self._pad_w + 0.1)
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        # BGR→RGB, HWC→CHW, /255, add batch dim
        img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return img[None]   # (1, 3, imgsz, imgsz)


# ── TensorRT engine wrapper ───────────────────────────────────────────────────

class _TensorRTEngine:
    """Thin pycuda wrapper around a single-input/single-output TRT engine.

    Uses retain_primary_context() to coexist with GStreamer NVMEDIA contexts
    (see CLAUDE.md — pycuda.autoinit would be torn down by nvv4l2decoder).
    """

    def __init__(self, engine_path: str, cuda_ctx=None) -> None:
        import tensorrt as trt
        import pycuda.driver as cuda

        if cuda_ctx is None:
            cuda.init()
            cuda_ctx = cuda.Device(0).retain_primary_context()

        self._cuda     = cuda
        self._ctx      = cuda_ctx

        self._ctx.push()
        try:
            logger  = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f:
                engine_bytes = f.read()
            runtime         = trt.Runtime(logger)
            self.engine     = runtime.deserialize_cuda_engine(engine_bytes)
            self.context    = self.engine.create_execution_context()

            in_idx  = self.engine.get_binding_index("images")
            out_idx = self.engine.get_binding_index("output0")
            self.input_shape  = tuple(self.engine.get_binding_shape(in_idx))
            self.output_shape = tuple(self.engine.get_binding_shape(out_idx))

            import tensorrt as trt2
            self.h_input  = cuda.pagelocked_empty(trt2.volume(self.input_shape),
                                                   dtype=np.float32)
            self.h_output = cuda.pagelocked_empty(trt2.volume(self.output_shape),
                                                   dtype=np.float32)
            self.d_input  = cuda.mem_alloc(self.h_input.nbytes)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)
            self.bindings = [int(self.d_input), int(self.d_output)]
            self.stream   = cuda.Stream()

            log.info(
                "TRT engine loaded: input=%s output=%s  engine=%s",
                self.input_shape, self.output_shape, engine_path,
            )
        finally:
            self._ctx.pop()

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        """CPU-path inference: H2D upload + execute + D2H."""
        self._ctx.push()
        try:
            np.copyto(self.h_input, input_array.ravel())
            self._cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            self.context.execute_async_v2(bindings=self.bindings,
                                          stream_handle=self.stream.handle)
            self._cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
        finally:
            self._ctx.pop()
        return self.h_output.reshape(self.output_shape)

    def infer_from_device(self) -> np.ndarray:
        """GPU-path inference: d_input already populated; skip H2D upload."""
        self._ctx.push()
        try:
            self.context.execute_async_v2(bindings=self.bindings,
                                          stream_handle=self.stream.handle)
            self._cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
        finally:
            self._ctx.pop()
        return self.h_output.reshape(self.output_shape)


# ── TRT detector ─────────────────────────────────────────────────────────────

class Yolo11TrtDetector(Detector):
    """YOLO11 TensorRT FP16 detector with GPU preprocessing.

    If `cuda_ctx` is supplied and FramePackets carry `d_bgr`, the full GPU
    path is used (no CPU color-convert, no H2D for the input tensor).
    Falls back to the CPU path transparently when d_bgr is None.

    Engine rebuild note:
      The existing models/yolo11n.engine was built for 640-input. This class
      reads the actual input shape from the engine and configures itself
      accordingly. To use 1280-input, rebuild via:
        model.export(format='onnx', imgsz=1280, ...)
        trtexec --onnx=... --saveEngine=... --fp16 --memPoolSize=workspace:2048
      (See CLAUDE.md "Model + TensorRT".)
    """

    def __init__(
        self,
        engine_path: str,
        vehicle_classes: List[int] = None,
        confidence: float = 0.4,
        iou_thres: float  = 0.45,
        frame_h: int      = 1080,
        frame_w: int      = 1920,
        cuda_ctx          = None,
    ) -> None:
        self._vehicle_classes = list(vehicle_classes or DEFAULT_VEHICLE_CLASSES)
        self._conf    = confidence
        self._iou     = iou_thres
        self._frame_h = frame_h
        self._frame_w = frame_w

        # Initialise CUDA context (shared primary; safe to call multiple times)
        if cuda_ctx is None:
            import pycuda.driver as cuda
            cuda.init()
            cuda_ctx = cuda.Device(0).retain_primary_context()
        self._cuda_ctx = cuda_ctx

        self._engine = _TensorRTEngine(engine_path, cuda_ctx)
        imgsz = self._engine.input_shape[-1]   # e.g. 1280 or 640
        self._imgsz = imgsz

        # GPU preprocessor writes directly to engine.d_input
        try:
            self._gpu_prep: Optional[_GpuPreprocessor] = _GpuPreprocessor(
                H_in     = frame_h,
                W_in     = frame_w,
                imgsz    = imgsz,
                d_input  = self._engine.d_input,
                cuda_ctx = cuda_ctx,
                stream   = self._engine.stream,
            )
        except Exception as e:
            log.warning("GPU preprocessor unavailable (%s); using CPU fallback", e)
            self._gpu_prep = None

        self._cpu_prep = _CpuPreprocessor(frame_h, frame_w, imgsz)

        log.info(
            "Yolo11TrtDetector: imgsz=%d classes=%s conf=%.2f gpu_prep=%s",
            imgsz, self._vehicle_classes,
            confidence, self._gpu_prep is not None,
        )

    # ── Detector interface ────────────────────────────────────────────────────

    @property
    def class_names(self) -> Dict[int, str]:
        return COCO_NAMES

    @property
    def input_size(self) -> int:
        return self._imgsz

    def detect(self, frame) -> List[Dict]:
        """Run TRT inference + NMS. Returns raw detections without track IDs."""
        is_packet = hasattr(frame, "d_bgr")
        h0 = frame.height if is_packet else frame.shape[0]
        w0 = frame.width  if is_packet else frame.shape[1]

        # ── Preprocess ───────────────────────────────────────────────────────
        use_gpu = (is_packet and frame.d_bgr is not None
                   and self._gpu_prep is not None)

        if use_gpu:
            self._gpu_prep.process(frame.d_bgr)   # kernel: d_bgr → engine.d_input
            raw = self._engine.infer_from_device() # TRT: d_input → h_output (no H2D)
        else:
            bgr = frame.cpu_bgr if is_packet else frame
            inp = self._cpu_prep.process(bgr)
            raw = self._engine.infer(inp)

        # ── Post-process (CPU) ────────────────────────────────────────────────
        pred   = torch.from_numpy(raw)
        result = ops.non_max_suppression(
            pred,
            conf_thres = self._conf,
            iou_thres  = self._iou,
            classes    = self._vehicle_classes,
        )[0]

        if not result.shape[0]:
            return []

        boxes_xyxy = ops.scale_boxes(
            (self._imgsz, self._imgsz), result[:, :4], (h0, w0)
        ).numpy()
        confs = result[:, 4].numpy()
        clss  = result[:, 5].numpy().astype(int)

        return [
            {
                "bbox":     (float(x1), float(y1), float(x2), float(y2)),
                "cls":      int(cls),
                "cls_name": COCO_NAMES.get(int(cls), str(int(cls))),
                "conf":     float(conf),
            }
            for (x1, y1, x2, y2), cls, conf in zip(boxes_xyxy, clss, confs)
        ]

    def warmup(self, n_iters: int = 3) -> float:
        """Run detect() on a synthetic black frame; return mean latency ms."""
        synthetic = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
        latencies = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            self.detect(synthetic)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = sum(latencies) / len(latencies)
        log.info("Warmup: n=%d mean=%.1f ms", n_iters, mean_ms)
        return mean_ms

    def close(self) -> None:
        pass   # pycuda buffers freed when GC collects _TensorRTEngine


# ── .pt CPU backend (dev / fallback) ─────────────────────────────────────────

class Yolo11PtDetector(Detector):
    """ultralytics .pt backend — CPU inference (~2 fps). Dev/fallback only."""

    def __init__(
        self,
        model_path: str,
        vehicle_classes: List[int] = None,
        confidence: float = 0.4,
    ) -> None:
        from ultralytics import YOLO

        self._vehicle_classes = list(vehicle_classes or DEFAULT_VEHICLE_CLASSES)
        self._conf   = confidence
        self._model  = YOLO(model_path)
        self._imgsz  = 640   # .pt models default to 640
        self._names  = self._model.names

        log.warning(
            "Yolo11PtDetector: CPU-only path (~2 fps). "
            "Use Yolo11TrtDetector for production."
        )

    @property
    def class_names(self) -> Dict[int, str]:
        return self._names

    @property
    def input_size(self) -> int:
        return self._imgsz

    def detect(self, frame) -> List[Dict]:
        """Run predict() (no tracking). Track IDs assigned by Tracker."""
        bgr = frame.cpu_bgr if hasattr(frame, "cpu_bgr") else frame
        results = self._model.predict(
            bgr,
            classes = self._vehicle_classes,
            conf    = self._conf,
            verbose = False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy  = boxes.xyxy.cpu().numpy()
        clss  = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        return [
            {
                "bbox":     tuple(float(v) for v in box),
                "cls":      int(cls),
                "cls_name": self._names.get(int(cls), str(int(cls))),
                "conf":     float(conf),
            }
            for box, cls, conf in zip(xyxy, clss, confs)
        ]

    def warmup(self, n_iters: int = 3) -> float:
        synthetic = np.zeros((640, 640, 3), dtype=np.uint8)
        latencies = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            self.detect(synthetic)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        return sum(latencies) / len(latencies)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_yolo11_detector(config: dict, cuda_ctx=None) -> Detector:
    """Build the appropriate backend from detector config dict.

    config keys: engine_path, vehicle_classes, confidence, input_size, fps
    (all from pipeline.yaml detector section).
    """
    path    = config.get("engine_path", "models/yolo11n.engine")
    classes = config.get("vehicle_classes", DEFAULT_VEHICLE_CLASSES)
    conf    = float(config.get("confidence", 0.4))

    if path.endswith(".engine"):
        return Yolo11TrtDetector(
            engine_path     = path,
            vehicle_classes = classes,
            confidence      = conf,
            cuda_ctx        = cuda_ctx,
        )
    elif path.endswith(".pt"):
        return Yolo11PtDetector(
            model_path      = path,
            vehicle_classes = classes,
            confidence      = conf,
        )
    else:
        raise ValueError(f"Unknown model extension: {path!r} (expected .engine or .pt)")
