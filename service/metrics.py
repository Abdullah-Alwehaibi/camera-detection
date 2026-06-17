"""service/metrics.py — Prometheus metrics for the detection pipeline.

Exposed at http://localhost:<metrics_port>/metrics in Prometheus text format.
All labels use the rule_id / event_type values from RuleEvent so dashboards
can filter by zone and type without code changes.
"""

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# ── Frame reader ───────────────────────────────────────────────────────────────

frames_total = Counter(
    "camera_frames_total",
    "Total frames processed by the pipeline",
)
frames_dropped = Counter(
    "camera_frames_dropped_total",
    "Frames dropped due to the shmsrc queue being full (GStreamer thread falls behind)",
)

# ── Inference ─────────────────────────────────────────────────────────────────

inference_duration = Histogram(
    "camera_inference_seconds",
    "Per-frame detection + tracking latency (wall time, not GPU compute)",
    buckets=[0.002, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.000],
)
active_tracks = Gauge(
    "camera_active_tracks",
    "Active track IDs produced by the tracker in the last frame",
)

# ── Rule events + dispatch ────────────────────────────────────────────────────

rule_events_total = Counter(
    "camera_rule_events_total",
    "Rule events fired, by rule_id and event_type",
    labelnames=["zone_id", "event_type"],
)
bestshot_dispatches_total = Counter(
    "camera_bestshot_dispatches_total",
    "BestShot dispatches, by capture source (backend_pull / roadside_snapshot)",
    labelnames=["source"],
)
event_send_errors_total = Counter(
    "camera_event_send_errors_total",
    "Boolean event write failures (AF_UNIX send error or schema validation failure)",
)

# ── Pipeline throughput ───────────────────────────────────────────────────────

pipeline_fps = Gauge(
    "camera_pipeline_fps",
    "Current pipeline throughput (fps, 5-second rolling window from FrameReader)",
)


def start(port: int) -> None:
    """Start the Prometheus HTTP server.  Call once at pipeline startup."""
    start_http_server(port)
