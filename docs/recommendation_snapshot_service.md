# Recommendation: 4K Roadside Snapshot Service
**For: jetson-gstreamer-testing team (aaeon)**
**From: camera-detection AI pipeline (abdullah)**
**Status: Required to enable `snapshot.mode: roadside_snapshot` in pipeline.yaml**

---

## Background

The AI pipeline detects vehicles at 1080p and needs a 4K crop of the exact
frame when a vehicle is selected for ANPR. Currently `snapshot.mode` is set
to `backend_pull` (the AI embeds `bbox_4k` + `ts_realtime_ns` in the sidecar
and the backend retrieves the 4K frame from the recording). This recommendation
describes what the aaeon-side service must add to enable on-device 4K crops
without recording access.

**Nothing in this document requires changes to the AI pipeline code.**
The pipeline auto-detects whether the snapshot socket exists on each request
and falls back to `backend_pull` transparently if the service is absent.

---

## What to add to jetson-gstreamer-testing

### 1. 4K frame tap (if not already present)

In `send_stream.sh` (or equivalent), tee the raw 4K source into a ring buffer:

```gst
pylonsrc ... ! tee name=t
t. ! queue ! <existing 1080p AI tap path>
t. ! queue ! appsink name=snap_sink emit-signals=true max-buffers=30 drop=true sync=false
```

`snap_sink` keeps the last 30 4K frames (~6 s at 5 fps) in a drop-first ring
so snapshots can be pulled slightly after the detection event without stalling
the camera loop.

### 2. Snapshot server process

Add a new process/thread (e.g. `snapshot_server.py`) that:

1. Attaches to `snap_sink` (via PyGObject signal or a shared queue).
2. Indexes frames by `CLOCK_MONOTONIC` timestamp (read via
   `Gst.Clock.get_time()` or `time.clock_gettime(CLOCK_MONOTONIC)`).
3. Listens on `AF_UNIX SOCK_STREAM` at `/tmp/ai_snapshot.sock`.
4. Accepts newline-delimited JSON requests and sends newline-delimited JSON
   responses (same framing as `ai_events.sock`).

#### Request schema (AI → snapshot service)

```json
{
  "track_id":       42,
  "ts_monotonic_ns": 1234567890123456789,
  "bbox_4k":        [860, 410, 512, 300]
}
```

| field | type | notes |
|---|---|---|
| `track_id` | int | used to name the output file |
| `ts_monotonic_ns` | int | `CLOCK_MONOTONIC` ns of the AI detection frame |
| `bbox_4k` | [x, y, w, h] | pixels in 4K frame coordinates |

#### Response schema (snapshot service → AI)

**Success:**
```json
{
  "status":         "ok",
  "jpeg_path":      "/data/snapshots/4k_track42_20260617_153200_123456.jpg",
  "ts_monotonic_ns": 1234567890123456789
}
```

**Error (frame expired, crop OOB, etc.):**
```json
{
  "status":  "error",
  "message": "frame not found: ts 1234567890 expired from ring buffer"
}
```

Frame matching: find the buffered 4K frame whose timestamp is closest to
`ts_monotonic_ns`; accept if `|delta| < 500 ms`, otherwise return error.
On error the AI pipeline automatically falls back to `backend_pull`.

#### Crop and save

```python
crop = frame_bgr[y:y+h, x:x+w]
cv2.imwrite(jpeg_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
```

Save to a directory readable by `abdullah` (e.g. `/data/snapshots/` or
the AI pipeline's `evidence_dir`). The path in the response must be an
absolute path.

### 3. Socket permissions (required for `abdullah` to connect)

The socket must be created with mode `0660` and group `aaeon` so the
`abdullah` user (who is in the `aaeon` group) can connect.

In the server script, set the umask before `bind()`:

```python
import os, socket
old_umask = os.umask(0o117)   # results in 0660 for the socket file
sock.bind("/tmp/ai_snapshot.sock")
os.umask(old_umask)
```

Or add to the systemd unit (if the server runs as a service):

```ini
[Service]
UMask=0117
```

Verify after start:
```bash
ls -la /tmp/ai_snapshot.sock
# expected: srw-rw---- aaeon aaeon  (0660)
```

### 4. Systemd integration (optional but recommended)

If running as a service alongside `send-stream.service`:

```ini
[Unit]
Description=4K Snapshot Service
After=send-stream.service
BindsTo=send-stream.service

[Service]
User=aaeon
Group=aaeon
UMask=0117
ExecStart=/path/to/snapshot_server.py
Restart=on-failure
```

---

## Checklist before enabling `roadside_snapshot` on the AI side

- [ ] `snap_sink` tee is live in `send_stream.sh`
- [ ] `snapshot_server.py` is running and `/tmp/ai_snapshot.sock` exists
- [ ] `ls -la /tmp/ai_snapshot.sock` shows `srw-rw----` (mode 0660, group aaeon)
- [ ] `ss -xlp | grep ai_snapshot` shows `LISTEN`
- [ ] Send a test request manually and get a valid JPEG path back
- [ ] Update `pipeline.yaml`: `snapshot.mode: roadside_snapshot`

The AI pipeline will auto-detect the socket on the next restart and
switch to `roadside_snapshot` mode. If the socket disappears at runtime
(service crash, restart), the pipeline logs a warning and falls back to
`backend_pull` for that event — no pipeline restart needed.

---

## Coordinate reference

The AI pipeline sends `bbox_4k` in these coordinates:

| source | width | height |
|---|---|---|
| AI detection frame | 1920 | 1080 |
| 4K source frame | 4096 | 2160 |
| Scale x | 4096/1920 = **2.13333** | — |
| Scale y | — | 2160/1080 = **2.0** |

These factors are in `pipeline.yaml` (`scale_4k.x`, `scale_4k.y`) and
`integration/contract.py` (`DEFAULT_SCALE_4K_X`, `DEFAULT_SCALE_4K_Y`).
