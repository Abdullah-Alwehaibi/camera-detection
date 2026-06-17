#!/usr/bin/env python3
"""
tools/fake_events_sink.py — Fake AF_UNIX boolean-events socket consumer.

Listens on /tmp/ai_events.sock (or --socket), accepts connections from the
pipeline's BooleanEventWriter, reads newline-delimited JSON events, validates
them against integration.contract.EventType, and prints each event with a
timestamp to stdout.

Run this BEFORE the main pipeline so the socket file exists when the
pipeline's BooleanEventWriter attempts its first connection:

    # Terminal 1
    python tools/fake_events_sink.py --socket /tmp/ai_events.sock

    # Terminal 2
    python service/main.py --dry-run=false     (or main.py for the legacy entry)

On Ctrl-C the sink prints a summary (total events, per-type counts) and exits.

Options
-------
--socket PATH   AF_UNIX socket path (default: /tmp/ai_events.sock)
--output FILE   Also append received JSONL events to FILE
--strict        Exit with code 1 if any invalid event is received
--quiet         Suppress per-event output; show only the summary
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import threading
import time
from collections import Counter
from pathlib import Path

# integration.contract must be importable; add repo root to sys.path if needed
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from integration.contract import EventType, validate_event


# ── FakeEventsSink ────────────────────────────────────────────────────────────

class FakeEventsSink:
    """AF_UNIX SOCK_STREAM server that consumes and validates JSONL events.

    Thread-safe: each accepted connection is handled in its own daemon thread.
    """

    def __init__(
        self,
        socket_path: str,
        output_path: Optional[str] = None,  # type: ignore[name-defined]
        strict: bool = False,
        quiet: bool  = False,
    ) -> None:
        self._path      = socket_path
        self._strict    = strict
        self._quiet     = quiet
        self._counts:   Counter = Counter()
        self._errors:   int     = 0
        self._lock      = threading.Lock()
        self._server:   Optional[socket.socket] = None  # type: ignore[name-defined]
        self._running   = False

        self._outfile = None
        if output_path:
            self._outfile = open(output_path, "a", encoding="utf-8", buffering=1)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Block and serve until stop() or KeyboardInterrupt."""
        sock_path = Path(self._path)
        sock_path.unlink(missing_ok=True)

        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(sock_path))
        self._server.listen(8)
        self._server.settimeout(1.0)   # allows checking _running periodically
        self._running = True

        print(f"[fake_events_sink] Listening on {self._path}", flush=True)
        print(f"[fake_events_sink] Ctrl-C to stop and show summary", flush=True)

        try:
            while self._running:
                try:
                    conn, _ = self._server.accept()
                except socket.timeout:
                    continue
                t = threading.Thread(target=self._handle_conn, args=(conn,),
                                     daemon=True)
                t.start()
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

        self._print_summary()

        if self._strict and self._errors > 0:
            sys.exit(1)

    def stop(self) -> None:
        self._running = False

    # ── Private ───────────────────────────────────────────────────────────────

    def _handle_conn(self, conn: socket.socket) -> None:
        buf = b""
        try:
            while self._running:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if line:
                        self._on_line(line.decode(errors="replace"))
        except (OSError, ConnectionResetError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _on_line(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            with self._lock:
                self._errors += 1
            print(f"[{ts}] BAD JSON ({exc}): {line!r}", flush=True)
            return

        try:
            validate_event(event)
            valid = True
        except ValueError as exc:
            valid = False
            with self._lock:
                self._errors += 1
            print(f"[{ts}] INVALID EVENT ({exc}): {event!r}", flush=True)

        if valid:
            evt_type = event.get("type", "?")
            with self._lock:
                self._counts[evt_type] += 1
            if not self._quiet:
                zone = event.get("zone_id", "?")
                tid  = event.get("track_id", "?")
                cls  = event.get("class", event.get("cls_name", "?"))
                conf = event.get("confidence", 0.0)
                print(
                    f"[{ts}] {evt_type:<20s}  track={tid:<4}  zone={zone:<12}  "
                    f"class={cls:<12}  conf={conf:.2f}",
                    flush=True,
                )

        if self._outfile is not None:
            try:
                self._outfile.write(line + "\n")
                self._outfile.flush()
            except OSError:
                pass

    def _cleanup(self) -> None:
        if self._server:
            try:
                self._server.close()
            except OSError:
                pass
        Path(self._path).unlink(missing_ok=True)
        if self._outfile:
            try:
                self._outfile.close()
            except OSError:
                pass

    def _print_summary(self) -> None:
        total  = sum(self._counts.values())
        print("\n" + "─" * 60, flush=True)
        print(f"[fake_events_sink] Summary — {total} event(s) received, "
              f"{self._errors} error(s)", flush=True)
        for evt_type, count in sorted(self._counts.items()):
            print(f"  {evt_type:<20s}: {count}", flush=True)
        if self._errors > 0:
            print(f"  {'INVALID/BAD JSON':<20s}: {self._errors}", flush=True)
        print("─" * 60, flush=True)

    @property
    def total_events(self) -> int:
        return sum(self._counts.values())

    @property
    def error_count(self) -> int:
        return self._errors


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--socket", default="/tmp/ai_events.sock",
                        help="AF_UNIX socket path to listen on")
    parser.add_argument("--output", metavar="FILE",
                        help="Append received events (JSONL) to FILE")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if any invalid event received")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-event output; show summary only")
    args = parser.parse_args()

    sink = FakeEventsSink(
        socket_path = args.socket,
        output_path = args.output,
        strict      = args.strict,
        quiet       = args.quiet,
    )
    sink.run()


if __name__ == "__main__":
    main()
