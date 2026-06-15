# Tahakom Camera Detection — Edge Trigger-Zone Pipeline

## Hardware
- NVIDIA Jetson Orin NX 16GB, JetPack 5.1.1 (L4T R35.3.1), Python 3.8, aarch64
- Camera: Basler ace 2 Pro (Sony IMX545)

## Environment
- Python venv at .venv, created with --system-site-packages
  (GStreamer/PyGObject, TensorRT, and OpenCV-with-GStreamer come from
  system packages — do not pip-install replacements for gi, tensorrt,
  or opencv)
- Run `source .venv/bin/activate` before any python/pip commands
- Install Python deps with `pip install --no-deps -r requirements.txt`
  (see requirements.txt header — ultralytics' opencv-python dependency is
  deliberately excluded so the system cv2 4.5.4, GStreamer-enabled, is
  used instead)
- **Import-order gotcha (aarch64)**: `import torch` must be the first
  import in any entrypoint, before `cv2` or `gi`. If cv2 (or anything that
  pulls in cv2) is imported first, torch's bundled libgomp fails with
  `ImportError: ... cannot allocate memory in static TLS block`. main.py
  and pipeline/inference.py both do `import torch` first — preserve that
  ordering in any new entrypoint.
- ultralytics is pinned to **8.3.0** in requirements.txt. 8.4.x added a
  hard `polars` dependency with no prebuilt wheel for cp38-aarch64 (would
  try to build from source and fail). torch==2.4.1 / torchvision==0.19.1
  do have aarch64/py3.8 wheels on PyPI but are **CPU-only**
  (`torch.cuda.is_available()` is False). `lapx` (ByteTrack dep) has no
  aarch64 wheel either and gets built from source automatically on first
  `model.track()` call (~90s, one-time).
- CPU-only torch means ultralytics' own `.engine` export/AutoBackend path
  doesn't work on this box (both require a CUDA-enabled torch). TensorRT
  is still used, via a **custom pycuda runner** — see "Model + TensorRT"
  below. requirements.txt has two extra dependency groups for this:
  - `protobuf>=3.20.2` (venv copy shadows the system 3.6.1, which is too
    old for onnx 1.17's generated `_pb2` modules) + `onnx`/`onnxslim`/
    `onnxruntime`/etc. for the ONNX export step.
  - `pycuda==2024.1`, pinned — newer pycuda's `compyte/dtypes.py` uses
    PEP 604 `X | Y` union syntax in a class body, which breaks on
    Python 3.8. Builds from source (~10min) against
    `/usr/local/cuda-11.4` — needs `CUDA_ROOT=/usr/local/cuda-11.4` and
    that dir's `bin/` on `PATH` during `pip install`.

## Architecture
- pipeline/gst_pipeline.py     — FrameSource: GStreamer source -> BGR numpy frames
- pipeline/inference.py        — VehicleDetector: detect + track, `.pt` (CPU) or `.engine` (custom TensorRT/pycuda) backend
- pipeline/trigger_line.py     — Zone: generic line/polygon crossing detection
- pipeline/evidence_capture.py — crop_with_margin / save_evidence
- pipeline/output_sink.py       — OutputSink: AI_EVENTS_SOCKET protocol + JSONL fallback
- main.py                      — orchestration loop
- config/pipeline.json         — source caps, model path/classes, paths
- config/trigger_zones.json    — list of {id, type: line|polygon, points}
- models/                       — yolo11n.{pt,onnx,engine} (.onnx/.engine gitignored, built on-device)

## Model + TensorRT

**Model**: `models/yolo11n.pt` (YOLO11n, ultralytics' current-generation
nano model, COCO-pretrained — 2,616,248 params, 6.5 GFLOPs, 238 fused
layers). Vehicle classes filtered to car/motorcycle/bus/truck = `[2,3,5,7]`
(`config/pipeline.json` -> `model.vehicle_classes`). Chosen as the smallest
current-gen model with all required COCO vehicle classes and a clean
ultralytics 8.3.0 export path.

**Two backends**, selected by `model_path`'s extension in
`pipeline/inference.py`'s `VehicleDetector`:
- `*.pt` — `ultralytics.YOLO(...).track()` on CPU, with ultralytics' built-in
  BYTETracker for persistent IDs.
- `*.engine` — a custom TensorRT runner (`_TensorRTEngine`, pycuda):
  manual `cuda.mem_alloc` + `execute_async_v2` + async host/device memcpy.
  Pre/post-processing reuse ultralytics' CPU-only utilities (letterbox
  resize in `_preprocess`, `ops.non_max_suppression`, `ops.scale_boxes`,
  `ops.xyxy2xywh`) and a standalone `ultralytics.trackers.BYTETracker`
  instance for tracking. Both backends return the identical
  `track()` result shape, so `main.py` doesn't care which is active.

**Building the `.engine` (must be redone on every new device — see
Conventions)**:
1. ONNX export on CPU (no CUDA needed):
   `model.export(format="onnx", imgsz=640, device="cpu", dynamic=False, simplify=True, opset=17)`
   -> `models/yolo11n.onnx` (input `images` (1,3,640,640) FLOAT, output
   `output0` (1,84,8400) FLOAT).
2. Build the engine with the standalone CLI (no torch/CUDA-torch needed):
   `/usr/src/tensorrt/bin/trtexec --onnx=models/yolo11n.onnx --saveEngine=models/yolo11n.engine --fp16 --memPoolSize=workspace:1024`
   (~16min on this device — `logs/trtexec_build.log`).
3. `pycuda==2024.1` must be installed (see Environment) for
   `pipeline/inference.py`'s `.engine` path to import.

**pycuda context vs. GStreamer hardware decode (NVMEDIA)**: `_TensorRTEngine`
uses `cuda.Device(0).retain_primary_context()` with explicit `push()`/`pop()`
around engine setup and around each `infer()` call — **not**
`pycuda.autoinit`. With `FrameSource` `"file"` mode (`nvv4l2decoder` +
`nvvidconv` for hardware H.264 decode), NVMEDIA creates/manages its own CUDA
context in-process; an `autoinit` context gets torn down by this and the next
`infer()` fails with `pycuda._driver.LogicError: cuMemcpyHtoDAsync failed:
context is destroyed`. The retained primary context coexists with NVMEDIA's
context as long as it's pushed current before each CUDA call. (Not hit by
`videotestsrc`/`shmsrc`, which don't use hardware decode — but keep the
push/pop if touching this class.)

**DLA compatibility (investigated, not used)**: probed via
`trtexec --onnx=models/yolo11n.onnx --useDLACore=0 --allowGPUFallback --fp16`
(`logs/trtexec_dla_probe.log`). Only 6 `ForeignNode` blocks (early backbone
convs through `model.10`'s attention QKV projection) ran on DLA; 26 layers
fell back to GPU — the whole C2PSA attention block (`Concat` not on the
channel dim, `MatMul`, `Div` all unsupported by DLA) and the entire
`model.23` detection head (`Slice`/`Split` need 4D, DLA only supports up to
the layers it was given). Net effect: **throughput dropped from 101.3 qps
(GPU-only) to 33.0 qps with DLA+fallback — ~3x slower**, due to DLA<->GPU
handoff overhead for a model this small. DLA offload is **not** used; the
shipped `.engine` is GPU-only FP16. Purely informational, per the brief.

## Frame source

`config/pipeline.json` `source.mode` is `"videotestsrc"`, 1920x1080 BGR
@ 5fps. **This is a fallback, not the real camera feed** — but the caps
match the real source exactly, so flipping `source.mode` is the only
change needed (see `pipeline/gst_pipeline.py` docstring).

The real source is the AI tap of the sibling repo **jetson-gstreamer-testing**
(`send-stream.service`, runs as user `aaeon` — do not restart it): it
publishes raw frames over a GStreamer shmsink at `/tmp/ai_frames.sock`,
NV12, **1920x1080 @ 5fps** (confirmed from the running gst-launch command
line / `stream.conf` `AI_FRAME_*` defaults — this is where the
1920x1080@5fps fallback caps above came from). As of this session that
socket exists but is `srw-r----- aaeon:aaeon`, and user `abdullah` is not
in the `aaeon` group, so `shmsrc` connects with `EACCES`. Once
`sudo usermod -aG aaeon abdullah` + a new session fixes that, set
`source.mode = "shmsrc"` — no caps changes needed.

The same repo's `ai-interface/protocol.h` documents an **output** socket,
`/tmp/ai_events.sock` — newline-delimited JSON over `AF_UNIX SOCK_STREAM`,
consumed by `events_adapter.py` to drive ONVIF events. `pipeline/output_sink.py`'s
`OutputSink` implements this protocol (see "Output events" below). As of
this session the adapter isn't running, so `OutputSink` always falls back
to `logs/output_events.jsonl`.

Note: a separate, unrelated production stack also lives on this box at
`/opt/enforcement-camera` (runs as the `camera` user, own `ai_socket` at
`/tmp/enforcement_ai.sock`, 1280x720@30fps — do not restart its services
without checking with the team). It is not the current integration target;
`/opt/enforcement-camera/detection_pipeline/` may still be a useful
reference for v2 (plates/OCR, NTP-gated evidence, audit log, web UI).

## Testing with recorded footage

A third `FrameSource` mode, `"file"`, plays back pre-recorded video instead
of `videotestsrc`/`shmsrc` — same `frames()` BGR-numpy interface, so nothing
downstream changes. Drop `.mp4`/`.mkv`/`.avi`/`.mov`/`.webm`/`.m4v` clips
with real vehicle traffic into `data/test_videos/` (gitignored, currently
empty), then point `config/pipeline.json`'s `source` at it:

```json
"source": {
  "mode": "file",
  "width": 1920,
  "height": 1080,
  "fps": 5,
  "path": "data/test_videos",
  "loop": true
}
```

- `path` can be a single file or a directory — if a directory, every video
  file in it is played in sorted-filename order.
- `loop: true` restarts from the first file once the last one ends (good
  for longer FPS-benchmark runs); default `false` stops the pipeline when
  the files are exhausted.
- Each file is decoded via `decodebin` (hardware `nvv4l2decoder` +
  `nvvidconv` on this device) and scaled to the configured width/height —
  for `config/trigger_zones.json`'s `y=540` line to land in the right place,
  footage should ideally already be 1920x1080 (matching the real source),
  but any resolution is scaled to fit.
- `sync=false`, so frames are delivered as fast as they can be decoded —
  useful for quickly measuring detection FPS against real content rather
  than synthetic `videotestsrc` frames.

## Trigger zone config

`config/trigger_zones.json` is `{"zones": [{"id", "type": "line"|"polygon", "points": [[x,y], ...]}]}`.
Default: one horizontal `line` at `y=540` (50% of the 1080px frame height),
spanning the full 1920px width. `Zone.evaluate(point, track_id)` is called
once per tracked object per frame with its bbox bottom-center point, and
returns `True` exactly once per track ID:
- `"line"`: on the first sign-change of the cross-product side (infinite
  line, so angled/multiple lines are just more entries — no logic changes).
- `"polygon"` (3+ points): on the first outside->inside transition
  (ray-casting point-in-polygon).

## Evidence capture

On a trigger-zone crossing, `main.py` saves the **full frame** (not a crop),
annotated via `pipeline/evidence_capture`:
- `draw_detections(frame, detections, highlight_track_id)` draws a box +
  `"<class> #<track_id>"` label for every currently-tracked vehicle (scene
  context), with the crossing vehicle highlighted in red and others in
  green.
- `draw_databar(frame, timestamp, location, vehicle_class)` draws a black
  bar across the top with capture date/time, `config/pipeline.json`'s
  `location` string, and the crossing vehicle's class
  (car/motorcycle/bus/truck).

`timestamp` is `datetime.now()` captured immediately when the frame is
pulled from `source.frames()` in `main.py` -- i.e. actual capture time, not
the time evidence-saving runs (which is after detection/tracking).

`save_evidence(metadata, output_dir)` expects `metadata = {"timestamp":
datetime, "track_id": int, "bbox": (x1,y1,x2,y2), "class": str, "image":
<annotated BGR ndarray>}` and writes
`data/snapshots/vehicle_<id>_<timestamp>.jpg`.

## Output events

`pipeline/output_sink.OutputSink.send_event()` is called once per
trigger-zone crossing (i.e. once per passing vehicle) and sends one JSON
object per line — `{"type": "vehicle", "confidence", "bbox": [x,y,w,h],
"track_id", "timestamp_ns", "class", "zone_id", "crop_path", "timestamp"}` —
to `/tmp/ai_events.sock` (jetson-gstreamer-testing's `AI_EVENTS_SOCKET`
protocol, `ai-interface/protocol.h`), falling back to appending the same
JSON line to `logs/output_events.jsonl` if the socket doesn't exist or the
connection drops. Reconnection is retried on each send. Once
`events_adapter.py` is running and consuming `/tmp/ai_events.sock`, these
become ONVIF `ObjectDetection`/`VehicleDetection` events with no code change
needed here.

## Benchmark results

**Pure inference-loop FPS** (no source pacing — `.track()` looped on a
fixed 1920x1080 frame, imgsz=640, `logs/bench_pt.log` / `logs/bench_engine.log`):
- `.pt` (CPU): **2.218 fps**
- `.engine` (FP16, pycuda/TensorRT): **44.357 fps** (65s, 2884 frames)
- **~20x speedup**. trtexec's own GPU-compute-only number for this engine
  is 101.3 qps (9.84ms/inference, `logs/trtexec_build.log`) — the gap to
  44.4 fps is letterbox/NMS/BYTETracker pre/post-processing, still on CPU.

**In-pipeline FPS** (videotestsrc source capped at 5fps):
- `.pt`: 2.44 fps sustained (`logs/run_pt_baseline.log`) — **detection was
  the bottleneck**, source was not yet saturated.
- `.engine`: ~4.6 fps and still climbing toward the 5fps source cap after
  15s (`logs/run_engine_smoke.log`) — **detection is no longer the
  bottleneck; the 5fps source now is**.

**GPU utilization** (`tegrastats`, 1s interval, over the 65s `.engine`
benchmark, `logs/tegrastats_engine.log`): `GR3D_FREQ` min=0%, max=88%,
mean≈53% across 77 samples.

## Conventions
- TensorRT `.engine` files are built ON THIS DEVICE ONLY — not portable.
  If `models/yolo11n.engine` is missing (fresh clone / new device), either
  point `config/pipeline.json`'s `model.path` back at `models/yolo11n.pt`
  or rebuild via the steps in "Model + TensorRT" above.
- Test footage/output goes in data/ and logs/ — gitignored
