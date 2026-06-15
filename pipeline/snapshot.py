"""Single-frame snapshot capture from the Basler ace 2 Pro camera.

Uses the gst-plugin-pylon GStreamer element (pylonsrc), which is the
only camera access path available on this device (pypylon is not
installed for this Python 3.8 venv and has no compatible wheel).
"""

import datetime
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots"


def capture_snapshot(output_dir: Path = SNAPSHOT_DIR, timeout_sec: float = 10.0, image_format: str = "jpeg") -> Path:
    """Grab a single frame from the camera and save it as an image.

    image_format: "jpeg" or "png" ("png" is lossless/uncompressed).

    Returns the path to the saved snapshot.
    """
    if image_format == "jpeg":
        encoder_name, ext = "jpegenc", "jpg"
    elif image_format == "png":
        encoder_name, ext = "pngenc", "png"
    else:
        raise ValueError(f"Unsupported image_format: {image_format!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"snapshot_{timestamp}.{ext}"

    Gst.init(None)

    pipeline = Gst.Pipeline.new("snapshot")
    src = Gst.ElementFactory.make("pylonsrc", "src")
    convert = Gst.ElementFactory.make("videoconvert", "convert")
    encoder = Gst.ElementFactory.make(encoder_name, "encoder")
    sink = Gst.ElementFactory.make("filesink", "sink")

    if not all([src, convert, encoder, sink]):
        raise RuntimeError("Failed to create GStreamer elements (is gst-plugin-pylon installed?)")

    src.set_property("num-buffers", 1)
    sink.set_property("location", str(output_path))

    for element in (src, convert, encoder, sink):
        pipeline.add(element)

    src.link(convert)
    convert.link(encoder)
    encoder.link(sink)

    pipeline.set_state(Gst.State.PLAYING)
    bus = pipeline.get_bus()

    try:
        msg = bus.timed_pop_filtered(
            int(timeout_sec * Gst.SECOND),
            Gst.MessageType.EOS | Gst.MessageType.ERROR,
        )
        if msg is None:
            raise TimeoutError("Timed out waiting for camera frame")
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            raise RuntimeError(f"GStreamer error: {err.message} ({debug})")
    finally:
        pipeline.set_state(Gst.State.NULL)

    return output_path


if __name__ == "__main__":
    path = capture_snapshot(image_format="png")
    print(f"Saved snapshot to {path}")
