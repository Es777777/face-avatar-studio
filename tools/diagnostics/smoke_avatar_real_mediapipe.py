"""Production-faithful avatar harness — uses REAL FaceTrackingEngine with MediaPipe.

This drives the *exact* code path the production app runs, but without a webcam:
- Real ``FaceTrackingEngine`` (so MediaPipe + GNM are loaded the same way as
  production — face_landmarker.task download, XNNPACK init, etc.)
- Synthetic BGR frames fed into ``engine.process_frame`` (so MediaPipe always
  reports ``face_detected=False`` on our fake input)
- Real ``AvatarRenderWorker`` (same class as production)
- Real ``PreviewLabel`` (same class as production)

Goal: confirm whether MediaPipe's "no face detected" branch produces a vertex
buffer that the avatar renderer can still draw (mean > 5), or whether it
crashes/silently emits zeros. The user's report says avatar stays all-black
after MediaPipe initializes successfully.
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
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

ROOT = PROJECT_ROOT / "artifacts" / "generated"
ROOT.mkdir(parents=True, exist_ok=True)

from face_avatar_studio import pipeline as P  # noqa: E402
from face_avatar_studio.desktop_app import (  # noqa: E402
    AvatarRenderWorker,
    PreviewLabel,
    PREVIEW_SIZE,
    frame_to_qimage,
)


class _Signals:
    """Stand-in for the Qt signal hub — only used by AvatarRenderWorker for
    avatar_frame emit. We don't care about the others.
    """

    def __init__(self) -> None:
        from PySide6.QtCore import QObject, Signal

        class Sigs(QObject):
            avatar_frame = Signal(object)

        self.sigs = Sigs()

    def __getattr__(self, name):
        return getattr(self.sigs, name)


class State:
    def __init__(self) -> None:
        from PySide6.QtCore import QObject, Signal

        class Sigs(QObject):
            avatar_frame = Signal(object)
            packet_ready = Signal(object)
            camera_frame = Signal(object)
            log_added = Signal(str)

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
        self.logs: deque[str] = deque(maxlen=400)

    def add_log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        self.logs.append(line)
        print(line, flush=True)


def mediapipe_packet_producer(state: State, stop: threading.Event) -> None:
    """Feed synthetic BGR frames through the REAL ``engine.process_frame`` path.

    This exercises:
    - MediaPipe lazy init (download_file + create_from_options)
    - detect_for_video with synthetic input → face_detected=False
    - avatar.evaluate({}, None) → neutral vertices fallback

    If MediaPipe throws, the except branch also falls back to evaluate({}, None)
    with face_detected=False — same end state. We log every packet's
    face_detected + vertices stats so we can see what the renderer actually got.
    """
    engine = state.engine
    analysis_size = (256, 144)
    frame_index = 0
    while not stop.is_set():
        frame_index += 1
        # Synthetic face-ish frame: gray with a darker circle, just enough that
        # MediaPipe doesn't outright error.
        frame = np.full((analysis_size[1], analysis_size[0], 3), 80, dtype=np.uint8)
        cv2.circle(frame, (analysis_size[0] // 2, analysis_size[1] // 2), 30, (60, 60, 60), -1)
        ts = time.perf_counter()
        with state.engine_lock:
            try:
                packet = engine.process_frame(frame, frame_index, ts)
            except Exception as exc:
                state.add_log(f"[producer] process_frame 抛异常：{exc!r}")
                time.sleep(0.05)
                continue
        v = packet.vertices
        if v is None:
            state.add_log(f"[producer] frame={frame_index} vertices=None face_detected={packet.face_detected}")
        else:
            state.add_log(
                f"[producer] frame={frame_index} face_detected={packet.face_detected} "
                f"verts.shape={v.shape} verts.mean={v.mean():.4f} "
                f"verts.max_abs={float(np.abs(v).max()):.4f}"
            )
        state.latest_packet = packet
        time.sleep(1.0 / 12)  # slower than camera so we can observe each frame


def main() -> int:
    state = State()
    state.add_log("[harness] 初始化真实 FaceTrackingEngine（会触发 MediaPipe + GNM head v3 加载）...")
    engine = P.FaceTrackingEngine()
    state.engine = engine
    state.add_log("[harness] engine 构造完成，spawn workers")

    stop = threading.Event()
    prod = threading.Thread(target=mediapipe_packet_producer, args=(state, stop), daemon=True)
    prod.start()

    arw = AvatarRenderWorker(state, fps=12, size=PREVIEW_SIZE)
    state.avatar_worker = arw
    arw.start()

    app = QApplication.instance() or QApplication(sys.argv)
    preview = PreviewLabel("avatar")
    preview.resize(640, 360)
    preview.show()
    state.add_log("[harness] PreviewLabel 显示中，开始接收 avatar_frame")

    last = {"img": None, "mean": 0.0, "n": 0}

    def _on_avatar_frame(frame):
        last["img"] = frame.copy()
        last["mean"] = float(frame.mean())
        last["n"] += 1
        try:
            preview.set_frame(frame)
        except Exception as exc:
            print(f"set_frame exc: {exc!r}")

    state.signals.avatar_frame.connect(_on_avatar_frame)

    deadline = time.time() + 25.0
    next_hb = 0.0
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
        if time.time() > next_hb:
            state.add_log(
                f"[harness] frames={last['n']} last_mean={last['mean']:.1f}"
            )
            next_hb = time.time() + 2.0

    stop.set()
    arw.stop()
    arw.join(timeout=3)
    prod.join(timeout=2)

    if last["img"] is not None:
        out = ROOT / "diag_avatar_real_mp.png"
        cv2.imwrite(str(out), last["img"])
        state.add_log(f"[harness] saved last frame -> {out} (mean={last['mean']:.1f})")

    if last["mean"] < 5.0:
        state.add_log(f"[harness] FAIL: avatar frame is essentially black (mean={last['mean']:.1f})")
        return 1
    state.add_log(f"[harness] OK (frames={last['n']} mean={last['mean']:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
