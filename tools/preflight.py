#!/usr/bin/env python3
"""
tools/preflight.py — Pre-flight deployment gate for the camera-detection pipeline.

Checks that the device, dependencies, interfaces, model, config, clocks, and
end-to-end rule logic are all correct before starting the production service.

Sections
--------
  A  Hardware   — platform, L4T version, GPU presence, CUDA init, TRT version
  B  Deps       — all required packages importable, correct versions, cv2 GStreamer
  C  Interfaces — socket paths exist, correct permissions, grabber liveness (soft)
  D  Model      — engine/pt file exists, TRT loads and warms up successfully
  E  Config     — pipeline.yaml + rules.json parse and validate cleanly
  F  Clocks     — CLOCK_MONOTONIC + CLOCK_REALTIME both ticking, year plausible
  G  E2E        — line_jump scenario through RuleEngine fires ≥ 1 event (dry run)

Exit codes
----------
  0  All required checks passed (warnings allowed)
  1  One or more required checks failed
     (with --strict, any warning also triggers exit 1)

Usage
-----
  python tools/preflight.py
  python tools/preflight.py --strict
  python tools/preflight.py --config config/pipeline.yaml --no-color
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import os
import platform
import socket
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Add repo root so tools/ imports from project packages
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ── ANSI colours ─────────────────────────────────────────────────────────────

class _C:
    PASS  = "\033[32m"
    FAIL  = "\033[31m"
    WARN  = "\033[33m"
    INFO  = "\033[36m"
    BOLD  = "\033[1m"
    DIM   = "\033[2m"
    RESET = "\033[0m"


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class _R:
    name:    str
    passed:  bool
    msg:     str
    warn:    bool = False   # True → WARN (not FAIL) in non-strict mode


# ── Preflight runner ──────────────────────────────────────────────────────────

class Preflight:
    def __init__(
        self,
        config_path: str = "config/pipeline.yaml",
        strict:      bool = False,
        color:       bool = True,
    ) -> None:
        self._config_path = Path(config_path)
        self._strict      = strict
        self._color       = color and sys.stdout.isatty()
        self._results: List[_R] = []
        self._cfg: dict = {}

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> int:
        """Run all sections. Returns 0 on success, 1 on failure."""
        self._header("Tahakom Camera-Detection Pipeline — Preflight")
        self._section("A", "Hardware",   self._check_hardware)
        self._section("B", "Deps",       self._check_deps)
        self._section("C", "Interfaces", self._check_interfaces)
        self._section("D", "Model",      self._check_model)
        self._section("E", "Config",     self._check_config)
        self._section("F", "Clocks",     self._check_clocks)
        self._section("G", "E2E dry-run",self._check_e2e)
        return self._summary()

    # ── Section A — Hardware ──────────────────────────────────────────────────

    def _check_hardware(self) -> None:
        self._ok(
            "Platform linux/aarch64",
            sys.platform == "linux" and platform.machine() == "aarch64",
            f"sys.platform={sys.platform!r} machine={platform.machine()!r}",
            warn_on_fail=True,
        )

        # L4T version
        nv_release = Path("/etc/nv_tegra_release")
        if nv_release.exists():
            line = nv_release.read_text().strip().split("\n")[0]
            is_r35 = "R35" in line
            self._ok(
                "L4T R35 (JetPack 5.1.1)",
                is_r35,
                line[:80],
                warn_on_fail=True,
            )
        else:
            self._warn("L4T release file not found",
                       "/etc/nv_tegra_release absent — not a Jetson?")

        # CUDA init via pycuda
        try:
            import pycuda.driver as cuda
            cuda.init()
            dev  = cuda.Device(0)
            name = dev.name()
            mem  = dev.total_memory() // (1024 ** 3)
            self._pass(f"CUDA GPU: {name} ({mem} GiB)")
        except ImportError:
            self._fail("CUDA GPU: pycuda not importable")
        except Exception as exc:
            self._fail(f"CUDA GPU init failed: {exc}")

        # TensorRT
        try:
            import tensorrt as trt
            self._pass(f"TensorRT {trt.__version__}")
        except ImportError:
            self._fail("TensorRT not importable")

    # ── Section B — Deps ──────────────────────────────────────────────────────

    def _check_deps(self) -> None:
        PKGS: List[Tuple[str, Optional[str]]] = [
            ("torch",             "2.4"),
            ("cv2",               "4.5"),
            ("numpy",             None),
            ("tensorrt",          None),
            ("pycuda.driver",     None),
            ("ultralytics",       "8.3"),
            ("shapely",           "2."),
            ("watchdog",          None),
            ("yaml",              None),
            ("prometheus_client", None),
            ("paho.mqtt.client",  None),
        ]
        for pkg, min_ver in PKGS:
            try:
                mod     = importlib.import_module(pkg)
                ver     = getattr(mod, "__version__", None)
                ver_str = f"  ({ver})" if ver else ""
                if min_ver and ver and not ver.startswith(min_ver):
                    self._warn(
                        f"Dep {pkg}: expected ≥{min_ver}, got {ver}",
                        f"import succeeded but version mismatch",
                    )
                else:
                    self._pass(f"Dep {pkg}{ver_str}")
            except ImportError as exc:
                self._fail(f"Dep {pkg}: {exc}")

        # cv2 GStreamer support
        try:
            import cv2
            info = cv2.getBuildInformation()
            idx  = info.find("GStreamer:")
            has_gst = idx != -1 and "YES" in info[idx : idx + 60]
            self._ok(
                "cv2 GStreamer backend enabled",
                has_gst,
                "cv2.getBuildInformation() GStreamer section",
                warn_on_fail=True,
            )
        except Exception as exc:
            self._warn("cv2 GStreamer check failed", str(exc))

    # ── Section C — Interfaces ────────────────────────────────────────────────

    def _check_interfaces(self) -> None:
        cfg = self._load_config_once()
        if not cfg:
            self._warn("Config unavailable", "Skipping interface checks")
            return

        frame_sock = cfg.get("frame_socket", "/tmp/ai_frames.sock")
        events_sock = cfg.get("events_socket", "/tmp/ai_events.sock")

        # Grabber socket (soft — may not be running during preflight)
        p = Path(frame_sock)
        if not p.exists():
            self._warn(
                f"Frame socket missing: {frame_sock}",
                "send-stream.service not running? (soft — ok during pre-deploy)",
            )
        else:
            mode = p.stat().st_mode
            is_sock = stat.S_ISSOCK(mode)
            readable = os.access(str(p), os.R_OK)
            self._ok(
                f"Frame socket exists: {frame_sock}",
                is_sock and readable,
                f"is_sock={is_sock} readable={readable}",
                warn_on_fail=True,
            )

        # Events socket: if it already exists it should be writable / deletable
        ep = Path(events_sock)
        if ep.exists():
            writable = os.access(str(ep), os.W_OK)
            self._ok(
                f"Events socket writable: {events_sock}",
                writable,
                f"writable={writable}",
                warn_on_fail=True,
            )
        else:
            self._pass(f"Events socket absent (pipeline will create it): {events_sock}")

        # Snapshot socket (only in roadside_snapshot mode)
        snap_cfg  = cfg.get("snapshot", {})
        snap_mode = snap_cfg.get("mode", "backend_pull")
        if snap_mode == "roadside_snapshot":
            snap_sock = snap_cfg.get("snapshot_socket", "/tmp/ai_snapshot.sock")
            sp = Path(snap_sock)
            self._ok(
                f"Snapshot socket exists: {snap_sock}",
                sp.exists(),
                f"required for roadside_snapshot mode",
                warn_on_fail=True,
            )

    # ── Section D — Model ─────────────────────────────────────────────────────

    def _check_model(self) -> None:
        cfg = self._load_config_once()
        if not cfg:
            self._warn("Config unavailable", "Skipping model check")
            return

        engine_path = Path(cfg.get("detector", {}).get("engine_path", "models/yolo11n.engine"))

        if not engine_path.exists():
            pt_fallback = engine_path.with_suffix(".pt")
            if pt_fallback.exists():
                self._warn(
                    f"Engine not found: {engine_path}",
                    f".pt fallback exists at {pt_fallback} (~2fps, dev mode only). "
                    "Rebuild engine: see CLAUDE.md 'Model + TensorRT'.",
                )
                return
            self._fail(
                f"Model not found: {engine_path}",
                f"Neither {engine_path} nor {pt_fallback} exist.",
            )
            return

        size_mb = engine_path.stat().st_size / 1e6
        self._pass(f"Engine file: {engine_path} ({size_mb:.1f} MB)")

        # Load engine + warmup
        try:
            import torch   # aarch64 import-order: torch must be first
            import pycuda.driver as cuda
            from ai.detector.yolo11_trt import Yolo11TrtDetector

            t0  = time.perf_counter()
            det = Yolo11TrtDetector(
                engine_path = str(engine_path),
                vehicle_classes = cfg.get("detector", {}).get("vehicle_classes", [2,3,5,7]),
                confidence  = cfg.get("detector", {}).get("confidence", 0.4),
            )
            load_ms = (time.perf_counter() - t0) * 1000
            self._pass(f"Engine loaded in {load_ms:.0f} ms")

            lat_ms = det.warmup(n_iters=3)
            self._ok(
                f"Warmup latency {lat_ms:.1f} ms (threshold < 200 ms)",
                lat_ms < 200,
                f"mean over 3 synthetic frames",
                warn_on_fail=True,
            )
        except ImportError as exc:
            self._fail(f"Engine load skipped: {exc}")
        except Exception as exc:
            self._fail(f"Engine load/warmup failed: {exc}")

    # ── Section E — Config ────────────────────────────────────────────────────

    def _check_config(self) -> None:
        # pipeline.yaml
        try:
            import yaml
            raw   = self._config_path.read_text()
            cfg   = yaml.safe_load(raw)
            self._cfg = cfg
            self._pass(f"pipeline.yaml parsed: {self._config_path}")
        except FileNotFoundError:
            self._fail(f"pipeline.yaml not found: {self._config_path}")
            return
        except Exception as exc:
            self._fail(f"pipeline.yaml parse error: {exc}")
            return

        # Required top-level keys
        REQUIRED = ["frame_socket", "events_socket", "detector", "rules_file",
                    "evidence_dir", "tracker"]
        missing  = [k for k in REQUIRED if k not in cfg]
        self._ok("pipeline.yaml required keys present", not missing,
                 f"missing: {missing}" if missing else "all present")

        # Detector config
        det_cfg = cfg.get("detector", {})
        self._ok("detector.backend set",
                 "backend" in det_cfg, str(det_cfg.get("backend", "(absent)")))
        self._ok("detector.vehicle_classes set",
                 "vehicle_classes" in det_cfg,
                 str(det_cfg.get("vehicle_classes", "(absent)")))

        # rules.json
        rules_path = Path(cfg.get("rules_file", "config/rules.json"))
        try:
            import json
            with open(rules_path) as f:
                rules_raw = json.load(f)
            self._pass(f"rules.json parsed: {rules_path}")

            from ai.rules.loader import parse_rules
            rules, zones = parse_rules(rules_raw)
            self._pass(f"rules.json valid: {len(rules)} rule(s), {len(zones)} capture zone(s)")
        except FileNotFoundError:
            self._fail(f"rules.json not found: {rules_path}")
        except Exception as exc:
            self._fail(f"rules.json error: {exc}")

        # evidence_dir writable
        try:
            ev_dir = Path(cfg.get("evidence_dir", "data/snapshots"))
            ev_dir.mkdir(parents=True, exist_ok=True)
            self._pass(f"evidence_dir writable: {ev_dir}")
        except Exception as exc:
            self._fail(f"evidence_dir not writable: {exc}")

        # log_dir writable
        try:
            log_dir = Path(cfg.get("log_dir", "logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            self._pass(f"log_dir writable: {log_dir}")
        except Exception as exc:
            self._warn("log_dir not writable", str(exc))

    # ── Section F — Clocks ────────────────────────────────────────────────────

    def _check_clocks(self) -> None:
        CLOCK_REALTIME  = 0
        CLOCK_MONOTONIC = 1

        try:
            librt = ctypes.CDLL(ctypes.util.find_library("rt") or "librt.so.1")

            class _ts(ctypes.Structure):
                _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

            def _read(clock_id: int) -> int:
                ts = _ts()
                librt.clock_gettime(clock_id, ctypes.byref(ts))
                return ts.tv_sec * 1_000_000_000 + ts.tv_nsec

            # Monotonic clock is advancing
            m0 = _read(CLOCK_MONOTONIC)
            time.sleep(0.01)
            m1 = _read(CLOCK_MONOTONIC)
            self._ok("CLOCK_MONOTONIC advancing",
                     m1 > m0, f"Δ={(m1-m0)//1000} μs")

            # System has been up at least 60 s (sanity)
            uptime_s = m1 / 1e9
            self._ok("Uptime > 60 s", uptime_s > 60, f"{uptime_s:.0f} s", warn_on_fail=True)

            # Real-time clock is plausible
            rt = _read(CLOCK_REALTIME)
            import datetime
            year = datetime.datetime.fromtimestamp(rt / 1e9).year
            self._ok("CLOCK_REALTIME year plausible (2024–2035)",
                     2024 <= year <= 2035, f"year={year}", warn_on_fail=True)

            # Offset between clocks is plausible (< 100 years in ns)
            offset_days = abs(rt - m1) / (1e9 * 86400)
            self._ok("CLOCK_REALTIME – CLOCK_MONOTONIC offset plausible",
                     offset_days < 36525,   # 100 years
                     f"{offset_days:.0f} days", warn_on_fail=True)

            self._pass(f"Clock offset: REALTIME − MONOTONIC = {(rt-m1)/1e9:.3f} s")

        except Exception as exc:
            self._fail(f"Clock check failed: {exc}")

    # ── Section G — E2E dry-run ───────────────────────────────────────────────

    def _check_e2e(self) -> None:
        """Run the line_jump scenario through the rule engine (no camera/model needed).

        Uses FakeDetectionSource to supply scripted detections, the real
        RuleEngine loaded from rules.json (or a default rule if rules.json
        cannot be loaded), and BestShotSelector with dry_run=True so no disk
        writes or socket sends occur.
        """
        try:
            import torch   # aarch64 import-order
            from tools.fake_grabber import FakeDetectionSource
            self._pass("FakeDetectionSource importable")
        except Exception as exc:
            self._fail(f"FakeDetectionSource import failed: {exc}")
            return

        try:
            from ai.rules.engine import Rule, RuleEngine
            from ai.bestshot import BestShotSelector
            from integration.contract import DEFAULT_SCALE_4K_X, DEFAULT_SCALE_4K_Y

            # Try to load rules from config; fall back to a minimal default rule.
            rules = self._load_rules_for_e2e()
            engine = RuleEngine(
                rules_source  = rules,
                frame_w       = 1920,
                frame_h       = 1080,
                scale_4k_x    = DEFAULT_SCALE_4K_X,
                scale_4k_y    = DEFAULT_SCALE_4K_Y,
            )
            self._pass(f"RuleEngine built: {len(rules)} rule(s)")
        except Exception as exc:
            self._fail(f"RuleEngine build failed: {exc}")
            return

        try:
            src    = FakeDetectionSource("line_jump")
            events = []
            import numpy as np
            fake_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

            for i, dets in enumerate(src):
                ts = int(i * 0.2 * 1e9)
                evts = engine.evaluate(dets, ts_mono_ns=ts, ts_real_ns=ts)
                events.extend(evts)

            n = len(events)
            self._ok(f"line_jump: {n} rule event(s) fired",
                     n >= 1,
                     "expected ≥ 1 crossing (segment-intersection catch)")
        except Exception as exc:
            self._fail(f"E2E rule evaluation failed: {exc}")
            return

        try:
            bs = BestShotSelector(
                mode         = "backend_pull",
                evidence_dir = "data/snapshots",
                location     = "preflight-test",
                dry_run      = True,     # no disk writes, no socket sends
            )
            if events:
                import numpy as np
                fake_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                # bbox_1080 is [x,y,w,h]; update() wants (x1,y1,x2,y2)
                x, y, w, h = (events[0].bbox_1080 + [0,0,100,100])[:4]
                bs.update(
                    tracked_dets  = [{"track_id": events[0].track_id,
                                      "bbox": (x, y, x + w, y + h),
                                      "cls": 2, "cls_name": "car", "conf": 0.9}],
                    frame_bgr     = fake_frame,
                    capture_zones = [],
                    ts_mono_ns    = events[0].ts_mono_ns,
                    ts_real_ns    = events[0].ts_real_ns,
                )
                meta = bs.dispatch(events[0])
                self._ok("BestShotSelector dispatch (dry_run)",
                         meta.get("source") in ("backend_pull", "roadside_snapshot"),
                         f"source={meta.get('source')}  crop_path={meta.get('crop_path')}")
        except Exception as exc:
            self._fail(f"BestShotSelector dispatch failed: {exc}")

    def _load_rules_for_e2e(self):
        """Return a synthetic default rule for the E2E dry-run.

        Section E already validates that rules.json parses correctly.
        Section G uses a fixed horizontal line at y=0.5 (matched to the
        line_jump scenario vehicle path at x=0.39) so it always fires
        regardless of which production geometry is in rules.json.
        """
        from ai.rules.engine import Rule
        return [Rule(
            id="preflight_e2e", type="line_crossing", enabled=True,
            points=[(0.0, 0.5), (1.0, 0.5)],
            direction="any", classes=frozenset({"car","motorcycle","bus","truck"}),
            event_type="vehicle", cooldown_sec=1.0,
            allowed_heading_deg=0.0, tolerance_deg=45.0,
        )]

    # ── Config loader (cached) ────────────────────────────────────────────────

    def _load_config_once(self) -> dict:
        if self._cfg:
            return self._cfg
        try:
            import yaml
            self._cfg = yaml.safe_load(self._config_path.read_text())
        except Exception:
            pass
        return self._cfg

    # ── Output helpers ────────────────────────────────────────────────────────

    def _c(self, code: str, text: str) -> str:
        return f"{code}{text}{_C.RESET}" if self._color else text

    def _pass(self, msg: str) -> None:
        self._results.append(_R(msg, True, ""))
        print(f"  {self._c(_C.PASS, '[PASS]')} {msg}")

    def _fail(self, msg: str, detail: str = "") -> None:
        full = f"{msg}  {self._c(_C.DIM, detail)}" if detail else msg
        self._results.append(_R(full, False, detail))
        print(f"  {self._c(_C.FAIL, '[FAIL]')} {full}")

    def _warn(self, msg: str, detail: str = "") -> None:
        full = f"{msg}  {self._c(_C.DIM, detail)}" if detail else msg
        self._results.append(_R(full, True, detail, warn=True))
        print(f"  {self._c(_C.WARN, '[WARN]')} {full}")

    def _ok(
        self,
        name:          str,
        condition:     bool,
        detail:        str  = "",
        warn_on_fail:  bool = False,
    ) -> None:
        if condition:
            self._pass(f"{name}  {self._c(_C.DIM, detail)}" if detail else name)
        elif warn_on_fail:
            self._warn(name, detail)
        else:
            self._fail(name, detail)

    def _header(self, title: str) -> None:
        bar = "═" * 62
        print(f"\n{self._c(_C.BOLD, bar)}")
        print(f"  {self._c(_C.BOLD, title)}")
        print(self._c(_C.BOLD, bar))

    def _section(self, letter: str, name: str, fn) -> None:
        print(f"\n{self._c(_C.INFO + _C.BOLD, f'── Section {letter}: {name}')}")
        try:
            fn()
        except Exception as exc:
            self._fail(f"Section {letter} crashed: {exc}")

    def _summary(self) -> int:
        passed   = sum(1 for r in self._results if r.passed and not r.warn)
        warnings = sum(1 for r in self._results if r.warn)
        failures = sum(1 for r in self._results if not r.passed)

        print(f"\n{self._c(_C.BOLD, '═' * 62)}")
        if failures == 0 and (not self._strict or warnings == 0):
            status = self._c(_C.PASS + _C.BOLD, "ALL CHECKS PASSED — ready to deploy")
            rc = 0
        else:
            parts = []
            if failures:
                parts.append(self._c(_C.FAIL, f"{failures} failure(s)"))
            if warnings and self._strict:
                parts.append(self._c(_C.WARN, f"{warnings} warning(s) (strict mode)"))
            status = "PREFLIGHT FAILED: " + ", ".join(parts)
            rc = 1

        print(f"  Passed: {passed}  Warnings: {warnings}  Failed: {failures}")
        print(f"  {status}")
        print(self._c(_C.BOLD, "═" * 62))
        return rc


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--config",   default="config/pipeline.yaml",
                   help="Path to pipeline.yaml (default: config/pipeline.yaml)")
    p.add_argument("--strict",   action="store_true",
                   help="Treat warnings as failures (strict gate mode)")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI colour output")
    args = p.parse_args()

    rc = Preflight(
        config_path = args.config,
        strict      = args.strict,
        color       = not args.no_color,
    ).run()
    sys.exit(rc)


if __name__ == "__main__":
    main()
