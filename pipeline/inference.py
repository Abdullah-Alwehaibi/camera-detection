"""Vehicle detection + tracking.

Two backends, selected by the `model_path` extension:
  - "*.pt"     -> ultralytics YOLO model.track() (CPU, torch 2.4.1).
  - "*.engine" -> custom TensorRT runner (pycuda) for GPU inference,
    since ultralytics' own .engine AutoBackend and TensorRT export both
    require a CUDA-enabled torch (`torch.cuda.is_available()`), which this
    aarch64 build doesn't have. Pre/post-processing (letterbox, NMS,
    BYTETracker) reuse ultralytics' CPU-only utilities so the custom path
    only has to handle the engine's raw input/output tensors.

aarch64 note: importing `torch` (a transitive dependency of ultralytics)
must happen before `cv2` or `gi` are imported anywhere in the process --
otherwise torch's bundled libgomp fails with "cannot allocate memory in
static TLS block". See CLAUDE.md. main.py imports this module first for
that reason.
"""

import torch  # noqa: F401  -- import-order workaround, see module docstring
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, ops, yaml_load
from ultralytics.utils.checks import check_yaml

# COCO-80 class names, in the order ultralytics' YOLO models output them.
# The .engine backend has no YOLO object to read .names from, so this is
# hardcoded -- vehicle_classes (car/motorcycle/bus/truck = 2/3/5/7) are all
# within this set.
COCO_CLASS_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


def _letterbox(frame, new_shape=640, color=(114, 114, 114)):
    """Resize+pad frame to a new_shape x new_shape square, preserving aspect
    ratio (matches ultralytics' LetterBox, which ops.scale_boxes assumes when
    no explicit ratio_pad is given)."""
    h, w = frame.shape[:2]
    gain = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * gain)), int(round(h * gain)))
    dw, dh = (new_shape - new_unpad[0]) / 2, (new_shape - new_unpad[1]) / 2

    resized = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def _preprocess(frame, imgsz):
    """BGR HWC frame -> normalized float32 NCHW batch-of-1, RGB."""
    img = _letterbox(frame, imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img[None]


class _TensorRTEngine:
    """Thin pycuda wrapper around a single-input/single-output TensorRT engine."""

    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda

        cuda.init()
        self._cuda = cuda
        # Use the device's primary context (shared/refcounted with other CUDA
        # consumers in the process) instead of pycuda.autoinit's standalone
        # context -- on Jetson, GStreamer's nvv4l2decoder/nvvidconv (used by
        # FrameSource's "file" mode) create their own NVMEDIA/CUDA context,
        # which tears down an autoinit context out from under pycuda
        # ("cuMemcpyHtoDAsync failed: context is destroyed"). The primary
        # context coexists safely; push/pop around each use keeps it current
        # regardless of what GStreamer does on the same thread in between.
        self._cuda_context = cuda.Device(0).retain_primary_context()

        self._cuda_context.push()
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f:
                engine_bytes = f.read()
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            self.context = self.engine.create_execution_context()

            in_idx = self.engine.get_binding_index("images")
            out_idx = self.engine.get_binding_index("output0")
            self.input_shape = tuple(self.engine.get_binding_shape(in_idx))
            self.output_shape = tuple(self.engine.get_binding_shape(out_idx))

            self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
            self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)
            self.d_input = cuda.mem_alloc(self.h_input.nbytes)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)
            self.bindings = [int(self.d_input), int(self.d_output)]
            self.stream = cuda.Stream()
        finally:
            self._cuda_context.pop()

    def infer(self, input_array):
        """input_array: float32 ndarray matching self.input_shape.
        Returns a float32 ndarray of self.output_shape."""
        self._cuda_context.push()
        try:
            np.copyto(self.h_input, input_array.ravel())
            self._cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            self._cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
        finally:
            self._cuda_context.pop()
        return self.h_output.reshape(self.output_shape)


class _TrackerInput:
    """Minimal results-like object for BYTETracker.update()."""

    def __init__(self, xywh, conf, cls):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls


class VehicleDetector:
    def __init__(self, model_path, vehicle_classes, confidence=0.4, imgsz=640):
        self.vehicle_classes = list(vehicle_classes)
        self.confidence = confidence
        self.imgsz = imgsz

        if str(model_path).endswith(".engine"):
            self.engine = _TensorRTEngine(model_path)
            self.model = None
            self.names = COCO_CLASS_NAMES
            tracker_cfg = IterableSimpleNamespace(**yaml_load(check_yaml("bytetrack.yaml")))
            self.tracker = BYTETracker(args=tracker_cfg, frame_rate=30)
        else:
            self.engine = None
            self.model = YOLO(model_path)
            self.names = self.model.names

    def track(self, frame):
        """Run detection + tracking on a single BGR frame.

        Returns a list of dicts with keys: track_id, bbox (x1, y1, x2, y2),
        cls, cls_name, conf. Detections the tracker hasn't assigned an ID
        to yet are skipped.
        """
        if self.engine is not None:
            return self._track_trt(frame)
        return self._track_pt(frame)

    def _track_pt(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            classes=self.vehicle_classes,
            conf=self.confidence,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        detections = []
        for box, track_id, cls, conf in zip(xyxy, ids, clss, confs):
            detections.append({
                "track_id": int(track_id),
                "bbox": tuple(float(v) for v in box),
                "cls": int(cls),
                "cls_name": self.names[int(cls)],
                "conf": float(conf),
            })
        return detections

    def _track_trt(self, frame):
        h0, w0 = frame.shape[:2]

        inp = _preprocess(frame, self.imgsz)
        raw = self.engine.infer(inp)
        pred = torch.from_numpy(raw)

        result = ops.non_max_suppression(
            pred,
            conf_thres=self.confidence,
            iou_thres=0.45,
            classes=self.vehicle_classes,
        )[0]

        if result.shape[0]:
            boxes_xyxy = ops.scale_boxes((self.imgsz, self.imgsz), result[:, :4], (h0, w0)).numpy()
            confs = result[:, 4].numpy()
            clss = result[:, 5].numpy()
            boxes_xywh = ops.xyxy2xywh(boxes_xyxy)
        else:
            boxes_xywh = np.zeros((0, 4), dtype=np.float32)
            confs = np.zeros((0,), dtype=np.float32)
            clss = np.zeros((0,), dtype=np.float32)

        tracked = self.tracker.update(_TrackerInput(boxes_xywh, confs, clss))

        detections = []
        for x1, y1, x2, y2, track_id, score, cls, _idx in tracked:
            detections.append({
                "track_id": int(track_id),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "cls": int(cls),
                "cls_name": self.names[int(cls)],
                "conf": float(score),
            })
        return detections
