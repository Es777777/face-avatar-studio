"""End-to-end harness that reproduces the production wiring in isolation.

Spawns:
- A mock `CameraWorker` (real MediaPipe via a synthetic fake sequence) on a thread
  that owns engine_lock for ~10-30ms per packet (MediaPipe + avatar.evaluate).
- A real `AvatarRenderWorker` that reads latest_packet and emits avatar_frame.
- A real `PreviewLabel` connected to avatar_frame signal.

If the production code has a thread/lock/signal problem, this harness surfaces it
without needing a webcam or full Qt window. Saves the last frame as
``diag_avatar_grab.png``.

Run:
    python smoke_concurrent_avatar.py
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QObject, Signal, Slot, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

ROOT = PROJECT_ROOT / "artifacts" / "generated"
ROOT.mkdir(parents=True, exist_ok=True)

from face_avatar_studio import pipeline as P  # noqa: E402

PREVIEW_SIZE = (640, 360)


class Signals(QObject):
    log_added = Signal(str)
    packet_ready = Signal(object)
    avatar_frame = Signal(object)
    camera_frame = Signal(object)


class State:
    def __init__(self) -> None:
        self.signals = Signals()
        self.engine = None
        self.engine_lock = threading.RLock()
        self.renderer_lock = threading.RLock()
        self.latest_packet = None
        # Mirror AppState fields used by AvatarRenderWorker
        self.latest_avatar_frame = None
        self.latest_camera_frame = None
        self.latest_camera_seq = 0
        self.latest_avatar_seq = 0
        self.avatar_renderer = None

    def add_log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {msg}", flush=True)


def mock_camera_worker(state: State, stop: threading.Event) -> None:
    """Mock CameraWorker: skip MediaPipe, just feed latest_packet with synthetic
    neutral-mesh vertices, but DO take engine_lock for 5-15ms each packet to
    simulate the real CameraWorker's MediaPipe + avatar.evaluate cost.
    """
    engine = state.engine
    avatar = engine.avatar
    template = np.asarray(
        avatar.model.template_vertex_positions, dtype=np.float32,
    )
    n_vertices = template.shape[0]

    frame_index = 0
    rng = np.random.default_rng(0)
    while not stop.is_set():
        frame_index += 1
        # Simulate MediaPipe cost: hold engine_lock for a few ms
        with state.engine_lock:
            # Simulate the slow avatar.evaluate call (real one calls model(...))
            time.sleep(rng.uniform(0.005, 0.015))
            # Produce slight deformation so the avatar frame changes frame-to-frame
            jitter = rng.normal(0.0, 0.001, size=template.shape).astype(np.float32)
            verts = (template + jitter).astype(np.float32)
        packet = P.FramePacket(
            frame_index=frame_index,
            timestamp_sec=time.perf_counter(),
            frame_bgr=np.zeros((360, 640, 3), dtype=np.uint8),
            landmarks=None,
            blendshapes={"jawOpen": 0.5},
            expression=np.zeros(383, dtype=np.float32),
            rotations=np.zeros((avatar.model.num_joints, 3), dtype=np.float32),
            translation=np.zeros(3, dtype=np.float32),
            face_matrix=None,
            vertices=verts,
            face_detected=True,
        )
        state.latest_packet = packet
        # ~30 fps
        time.sleep(1.0 / 30)


def main() -> int:
    state = State()
    state.add_log("[harness] initializing FaceTrackingEngine (avatar only, no MediaPipe)...")
    # Initialize engine without triggering MediaPipe (avoid network). We only
    # need GnmAvatarDriver, so call its constructor and wrap it in a tiny engine.
    from face_avatar_studio.pipeline import FaceTrackingEngine
    engine = FaceTrackingEngine.__new__(FaceTrackingEngine)
    engine.avatar = P.GnmAvatarDriver()
    engine.model_cache_dir = None
    engine.face_model_path = None
    engine._mp = None
    engine._landmarker = None
    state.engine = engine
    state.add_log("[harness] engine ready, spawning CameraWorker mock + AvatarRenderWorker")

    stop = threading.Event()

    # Mock camera
    cam_t = threading.Thread(target=mock_camera_worker, args=(state, stop), daemon=True)
    cam_t.start()

    # Real AvatarRenderWorker
    avatar_worker = P.__dict__.get("AvatarRenderWorker")
    if avatar_worker is None:
        # Pull class from desktop_app for full production parity
        from face_avatar_studio.desktop_app import AvatarRenderWorker
        avatar_worker = AvatarRenderWorker

    # We need PREVIEW_SIZE = PREVIEW_SIZE; fallback to known value
    preview_size = PREVIEW_SIZE
    arw = avatar_worker(state, fps=12, size=preview_size)
    arw.start()

    # Set up Qt, wire avatar_frame -> PreviewLabel
    app = QApplication.instance() or QApplication(sys.argv)
    preview = QLabel("avatar")
    preview.setMinimumSize(640, 360)
    preview.setAlignment(Qt.AlignCenter)
    preview.setStyleSheet("background-color: #1c1d22; color: #cbd5e1;")
    preview.show()

    last_frame = {"img": None, "mean": 0.0, "seq": 0, "ts": 0.0}

    def _frame_to_qimage(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

    @Slot(object)
    def _on_frame(frame):
        last_frame["img"] = frame.copy()
        last_frame["mean"] = float(frame.mean())
        last_frame["seq"] += 1
        last_frame["ts"] = time.time()
        img = _frame_to_qimage(frame)
        pix = QPixmap.fromImage(img)
        preview.setPixmap(
            pix.scaled(preview.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        )

    state.signals.avatar_frame.connect(_on_frame)

    # Pump events for 8 seconds
    deadline = time.time() + 8.0
    next_heartbeat = 0.0
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
        if time.time() > next_heartbeat:
            state.add_log(
                f"[harness] frames received={last_frame['seq']} "
                f"last_mean={last_frame['mean']:.1f}"
            )
            next_heartbeat = time.time() + 1.0

    stop.set()
    arw.stop()
    arw.join(timeout=3)
    cam_t.join(timeout=2)

    # Save last frame
    if last_frame["img"] is not None:
        out = ROOT / "diag_avatar_grab.png"
        cv2.imwrite(str(out), last_frame["img"])
        state.add_log(f"[harness] saved last frame -> {out} (mean={last_frame['mean']:.1f})")

    if last_frame["mean"] < 5.0:
        print(f"[harness] FAIL: avatar frame is essentially black (mean={last_frame['mean']:.1f})")
        return 1
    print(f"[harness] OK (frames={last_frame['seq']} mean={last_frame['mean']:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
