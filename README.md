# Tahakom Camera Detection

AI pipeline for the Basler ace 2 Pro camera on NVIDIA Jetson Orin NX.
Detects and tracks vehicles, evaluates trigger-zone rules, and emits
ONVIF-compatible events to the jetson-gstreamer-testing event adapter.

**Target device:** Jetson Orin NX 16 GB · JetPack 5.1.1 (L4T R35.3.1) · Python 3.8 · aarch64

---

## Quick start

```bash
git clone <repo>
cd tahakom-camera-detection
python3.8 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --no-deps -r requirements.txt
# Build the TRT engine (first time only — ~16 min):
python -c "from ultralytics import YOLO; YOLO('models/yolo11n.pt').export(format='onnx', imgsz=640, device='cpu', dynamic=False, simplify=True, opset=17)"
sudo /usr/src/tensorrt/bin/trtexec \
  --onnx=models/yolo11n.onnx --saveEngine=models/yolo11n.engine \
  --fp16 --memPoolSize=workspace:1024 2>&1 | tee logs/trtexec_build.log
bash tools/preflight.sh
sudo systemctl start camera-detection
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| JetPack | 5.1.1 (L4T R35.3.1) |
| Python | 3.8 (system) |
| CUDA | 11.4 |
| TensorRT | 8.5.x |
| GStreamer with OpenCV | system cv2 4.5.4 |

### System packages

```bash
# Already present on JetPack 5.1.1:
#   python3.8, libopencv-dev (4.5.4 with GStreamer), gstreamer1.0-*, tensorrt
# Extra tools used by pycuda and the GStreamer pipeline:
sudo apt-get install -y python3.8-venv python3.8-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

---

## Setup

### 1. Python environment

```bash
python3.8 -m venv --system-site-packages .venv
source .venv/bin/activate
```

The `--system-site-packages` flag is required so that the venv can see the
system-installed `gi` (PyGObject/GStreamer), `cv2` (OpenCV 4.5.4 with
GStreamer support), and `tensorrt`. **Do not** pip-install replacements for
any of these.

### 2. Python dependencies

```bash
pip install --no-deps -r requirements.txt
```

`--no-deps` prevents ultralytics from pulling in `opencv-python`, which would
shadow the system cv2 (GStreamer would silently break).

`pycuda==2024.1` builds from source (~10 min). It requires `nvcc` from CUDA 11.4:

```bash
export CUDA_ROOT=/usr/local/cuda-11.4
export PATH=$CUDA_ROOT/bin:$PATH
pip install --no-deps -r requirements.txt
```

### 3. Build the TensorRT engine

The `.engine` file is **device-specific** — it must be rebuilt on every Jetson,
even if it looks like the same hardware.  A pre-built engine from another box
will fail silently or produce wrong results.

**Step 1 — Export ONNX** (CPU only, fast, ~30 s):

```bash
source .venv/bin/activate
python - <<'EOF'
from ultralytics import YOLO
YOLO("models/yolo11n.pt").export(
    format="onnx", imgsz=640, device="cpu",
    dynamic=False, simplify=True, opset=17
)
EOF
```

**Step 2 — Build the engine** (~16 min on Orin NX):

```bash
mkdir -p logs
sudo /usr/src/tensorrt/bin/trtexec \
  --onnx=models/yolo11n.onnx \
  --saveEngine=models/yolo11n.engine \
  --fp16 \
  --memPoolSize=workspace:1024 \
  2>&1 | tee logs/trtexec_build.log
```

Expected output includes `[I] Throughput: 101.xxx qps` near the end.
If `models/yolo11n.engine` is missing, the pipeline falls back to the
`.pt` CPU model (~2 fps).

### 4. Configuration

**`config/pipeline.yaml`** — main runtime config (sockets, detector, tracker,
rules path, evidence directory, metrics port).

**`config/rules.json`** — trigger-zone rules loaded by the rule engine with
hot-reload (edit while the service is running; invalid edits are rejected and
the last-good config is kept).

Rules use normalized [0, 1] coordinates, so the same file works regardless of
resolution. The default rule (`zone_1781628896114`) is a line crossing at the
camera's calibrated position.  To add or edit rules, modify `config/rules.json`
directly — no service restart required.

### 5. Frame source

The pipeline reads NV12 frames from the grabber service
(`send-stream.service`, user `aaeon`) via a GStreamer shared-memory socket.

Grant the `abdullah` user access to the socket (once):

```bash
sudo usermod -aG aaeon abdullah
# Log out and back in (or: exec su -l $USER) to apply the new group
```

Then set `source.mode = "shmsrc"` in `config/pipeline.yaml`.

Until the group change is applied (or for testing), `source.mode =
"videotestsrc"` uses synthetic frames at 5 fps — detection works but there
are no real vehicles.

---

## Preflight

Run the preflight gate before starting the service for the first time on a
device, and after any major change (new engine, updated config, dependency
upgrade):

```bash
bash tools/preflight.sh
# Or, for CI / strict gate (any warning = failure):
bash tools/preflight.sh --strict
```

Sections checked:

| Section | Checks |
|---|---|
| A Hardware | linux/aarch64, L4T R35, CUDA GPU, TensorRT version |
| B Deps | 12 Python packages, cv2 GStreamer flag |
| C Interfaces | `/tmp/ai_frames.sock` readable, `/tmp/ai_events.sock` writable |
| D Model | `.engine` file size, load time, warmup latency |
| E Config | `pipeline.yaml` + `rules.json` parse and validate |
| F Clocks | CLOCK_MONOTONIC advancing, CLOCK_REALTIME year plausible |
| G E2E | `line_jump` scenario → rule engine → ≥ 1 event (dry run) |

Exit code `0` = deploy-ready.  `WARN` lines are informational; only `FAIL`
lines block deployment (unless `--strict` is passed).

---

## Deploy

### Install the systemd service (once)

```bash
sudo cp service/camera-detection.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable camera-detection
sudo systemctl start camera-detection
```

### View logs

```bash
# Live tail
journalctl -u camera-detection -f

# Last 200 lines
journalctl -u camera-detection -n 200 --no-pager

# Since last boot
journalctl -u camera-detection -b
```

### Status + control

```bash
sudo systemctl status camera-detection
sudo systemctl restart camera-detection   # applies config changes
sudo systemctl stop camera-detection
```

### Metrics

Prometheus text metrics are available at `http://localhost:9108/metrics`
while the service is running.

Key metrics:

| Metric | Description |
|---|---|
| `camera_pipeline_fps` | Live throughput (5-second window) |
| `camera_frames_total` | Total frames processed |
| `camera_frames_dropped_total` | Frames lost to queue backpressure |
| `camera_inference_seconds` | Per-frame detect + track latency (histogram) |
| `camera_active_tracks` | Track IDs alive in the last frame |
| `camera_rule_events_total` | Events fired, by zone and type |
| `camera_event_send_errors_total` | Boolean socket write failures |

---

## Testing without a camera

### Fake-detections mode

Bypass the detector and tracker entirely with scripted vehicle trajectories:

```bash
source .venv/bin/activate
# line_jump: vehicle teleports across trigger line in one frame (tests segment-intersection)
python service/main.py --fake-detections line_jump --dry-run

# normal: 3 vehicles, smooth crossing, fires real rule events
python service/main.py --fake-detections normal --dry-run

# id_reuse: 5-second gap causes state expiry, same track_id fires twice
python service/main.py --fake-detections id_reuse --dry-run
```

Available scenarios: `normal`, `line_jump`, `occlusion`, `id_reuse`.

### Fake events sink

Capture and validate events without the real `events_adapter.py`:

```bash
# Terminal 1 — start the sink before the pipeline:
python tools/fake_events_sink.py --socket /tmp/ai_events.sock

# Terminal 2 — run with real detector (no --dry-run):
python service/main.py --fake-detections normal
```

Add `--strict` to the sink to exit non-zero if any invalid event is received.

### Visual scenario frames

Save rendered frames to disk to visually inspect a scenario:

```bash
python tools/fake_grabber.py --scenario line_jump --save-frames /tmp/frames/
```

### GStreamer shmsink publisher

Publish synthetic NV12 frames on the real grabber socket:

```bash
python tools/fake_grabber.py --scenario normal --publish --fps 5
```

The real `FrameReader` can connect to this and receive frames while YOLO runs
detection on the synthetic (gray + colored rectangle) video.

---

## Troubleshooting

### `ImportError: cannot allocate memory in static TLS block`

`cv2` or `gi` was imported before `torch` in the process.  Every entrypoint
(`service/main.py`, tools, new scripts) must have `import torch` as its
**first import**, before any cv2/gi import.

### `cuMemcpyHtoDAsync failed: context is destroyed`

The TensorRT engine was initialized with `pycuda.autoinit`.  Use
`cuda.Device(0).retain_primary_context()` instead (see `pipeline/inference.py`
`_TensorRTEngine.__init__`).  The retained primary context coexists with
NVMEDIA's context; an autoinit context is destroyed by the hardware decoder.

### Engine runs slowly (< 10 fps) or gives wrong results

The `.engine` was built on a different device.  Rebuild with `trtexec` on this
device — see [Build the TensorRT engine](#3-build-the-tensorrt-engine).

### `cv2 GStreamer backend enabled: FAIL` in preflight

The venv's `cv2` is not the system GStreamer-enabled build.  The most common
cause is `pip install opencv-python` having shadowed the system package.
Fix: `pip uninstall opencv-python` and restart the preflight.

### `Frame socket missing: /tmp/ai_frames.sock`

`send-stream.service` (user `aaeon`) is not running, or `abdullah` is not in
the `aaeon` group.  Check: `groups` (after re-login).  The service runs in
`videotestsrc` mode if the socket is absent — update `source.mode` in
`config/pipeline.yaml` accordingly.

### Events not appearing in `/tmp/ai_events.sock`

1. Check that `events_adapter.py` (jetson-gstreamer-testing) is running and
   listening on the socket.
2. Or start `tools/fake_events_sink.py` to capture them yourself.
3. If the pipeline is in dry-run mode (`dry_run: true` in `pipeline.yaml` or
   `--dry-run` flag), events are logged but never sent.

### Rules are not hot-reloading

The watchdog observer requires `inotify` kernel support (present on JetPack).
If edits to `config/rules.json` are not picked up, check the service log for
`RulesLoader: invalid config — keeping last-good rules` (your edit has a
syntax error) or `RulesLoader: hot-reloaded` (it worked).

---

## Architecture

See `CLAUDE.md` for detailed architecture notes, benchmark results, DLA
compatibility findings, hardware decoder / CUDA context interaction, and
pycuda build requirements.

Pipeline data flow:

```
shmsrc (NV12, 1920×1080, 15 fps)
  └─ FrameReader (NV12 → GPU BGR kernel, FramePacket)
       └─ Yolo11TrtDetector  (TRT FP16, GPU path, ~44 fps headroom)
            └─ BYTETrackerBackend  (persistent track IDs)
                 └─ RuleEngine  (line crossing / polygon / direction, hot-reload)
                      ├─ BestShotSelector  (best frame per track)
                      ├─ BooleanEventWriter → /tmp/ai_events.sock → events_adapter.py → ONVIF
                      └─ SidecarWriter     → /tmp/ai_sidecar.sock (unix | mqtt | websocket)
```
