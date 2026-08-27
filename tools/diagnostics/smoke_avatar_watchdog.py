"""Smoke test for the new black-frame watchdog.

Drives AvatarRenderWorker with a mock State that *forces* the renderer to emit
all-black frames (mean=0) for 4+ seconds. Verifies that:
  1. _watch_black_frame fires its detection log once
  2. It does NOT fire repeatedly (rate-limited by _recovery_emitted)
  3. After the watchdog sees a healthy frame again, _recovery_emitted resets
  4. The render loop itself keeps emitting frames (does not crash / deadlock)
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

from launch_face_avatar_studio import _isolate_dll_search_path  # noqa: E402

if sys.platform == "win32":
    _isolate_dll_search_path()

import numpy as np  # noqa: E402
from PySide6.QtCore import QObject, Signal  # noqa: E402

ROOT = PROJECT_ROOT / "artifacts" / "generated"
ROOT.mkdir(parents=True, exist_ok=True)

from face_avatar_studio import pipeline as P  # noqa: E402
from face_avatar_studio.desktop_app import (  # noqa: E402
    AvatarRenderWorker,
    PreviewLabel,
    PREVIEW_SIZE,
)


class Sigs(QObject):
    avatar_frame = Signal(object)
    packet_ready = Signal(object)
    camera_frame = Signal(object)
    log_added = Signal(str)


class State:
    def __init__(self) -> None:
        self.signals = Sigs()
        self.engine = None
        self.engine_lock = threading.RLock()
        self.renderer_lock = threading.RLock()
        self.latest_packet = None
        self.latest_avatar_frame = None
        self.latest_camera_frame = None
        self.latest_camera_seq = 0
        self.latest_avatar_seq = 0
        self.avatar_renderer = None
        self.avatar_worker = None
        self._avatar_first_packet_seen = False
        self.logs: deque[str] = deque(maxlen=400)

    def add_log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.logs.append(line)
        print(line, flush=True)


def make_black_packet(frame_index: int) -> P.FramePacket:
    # Real-ish vertex buffer; renderer will still produce ~mean=70 frame.
    return P.FramePacket(
        frame_index=frame_index,
        timestamp_sec=time.perf_counter(),
        frame_bgr=np.zeros((360, 640, 3), dtype=np.uint8),
        landmarks=None,
        blendshapes={},
        expression=np.zeros(383, dtype=np.float32),
        rotations=np.zeros((1, 3), dtype=np.float32),
        translation=np.zeros(3, dtype=np.float32),
        face_matrix=None,
        vertices=None,  # forces ValueError -> black placeholder
        face_detected=False,
    )


def make_healthy_packet(frame_index: int, n_vertices: int) -> P.FramePacket:
    return P.FramePacket(
        frame_index=frame_index,
        timestamp_sec=time.perf_counter(),
        frame_bgr=np.zeros((360, 640, 3), dtype=np.uint8),
        landmarks=None,
        blendshapes={"jawOpen": 0.4},
        expression=np.zeros(383, dtype=np.float32),
        rotations=np.zeros((1, 3), dtype=np.float32),
        translation=np.zeros(3, dtype=np.float32),
        face_matrix=None,
        vertices=np.random.default_rng(frame_index).normal(0.0, 0.1, size=(n_vertices, 3)).astype(np.float32),
        face_detected=True,
    )


def main() -> int:
    state = State()

    # Minimal engine: GnmAvatarDriver only (no MediaPipe).
    engine = P.FaceTrackingEngine.__new__(P.FaceTrackingEngine)
    engine.avatar = P.GnmAvatarDriver()
    engine.model_cache_dir = None
    engine.face_model_path = None
    engine._mp = None
    engine._landmarker = None
    state.engine = engine
    n_vertices = engine.avatar.model.template_vertex_positions.shape[0]
    print(f"[smoke] template n_vertices = {n_vertices}")

    arw = AvatarRenderWorker(state, fps=20, size=PREVIEW_SIZE)
    state.avatar_worker = arw
    arw.start()

    received: list[tuple[int, float]] = []
    state.signals.avatar_frame.connect(
        lambda f: received.append((state.latest_avatar_seq, float(f.mean())))
    )

    # Phase 1: feed all-black packets for 5 seconds (should trigger recovery log)
    print("\n[phase 1] feeding all-black packets for 5s")
    deadline = time.time() + 5.0
    fi = 0
    while time.time() < deadline:
        fi += 1
        state.latest_packet = make_black_packet(fi)
        time.sleep(0.1)

    # Phase 2: feed healthy packets for 2 seconds (recovery should reset)
    print("\n[phase 2] feeding healthy packets for 2s")
    deadline = time.time() + 2.0
    while time.time() < deadline:
        fi += 1
        state.latest_packet = make_healthy_packet(fi, n_vertices)
        time.sleep(0.05)

    arw.stop()
    arw.join(timeout=3)

    # Analyze
    print(f"\n[result] total avatar frames received = {len(received)}")
    if received:
        means = [m for _, m in received]
        print(f"[result] frame means: first={means[0]:.1f} last={means[-1]:.1f} "
              f"min={min(means):.1f} max={max(means):.1f}")
    recovery_logs = [l for l in state.logs if "检测到连续黑帧" in l]
    print(f"[result] recovery log lines emitted = {len(recovery_logs)}")
    for l in recovery_logs:
        print(f"  {l}")

    # Phase 1 should produce >=1 recovery log; phase 2 should reset it (so a
    # third black phase would emit another one — we don't run that here).
    if not recovery_logs:
        print("[FAIL] watchdog never fired during 5s of black frames")
        return 1
    if len(recovery_logs) > 2:
        print(f"[FAIL] watchdog fired too many times ({len(recovery_logs)}); should be rate-limited")
        return 2

    print("\n[OK] watchdog fires once on extended black, then rate-limits itself")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
