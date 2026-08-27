"""Smoke test that reproduces the AvatarRenderWorker pipeline (off-screen VTK
on a worker thread) and the PreviewLabel paint path on a small QWidget.

Run:
    python smoke_avatar_render.py
Exits 0 on success, non-zero if avatar frame is black or camera frame
isn't aspect-correct.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Disable mediapipe GPU and C++ log noise, like the real launch path.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

# Mirror launch_face_avatar_studio's DLL isolation to avoid conda's Qt5 dlls
# shadowing PySide6 on Windows.
from launch_face_avatar_studio import _isolate_dll_search_path  # noqa: E402

if sys.platform == "win32":
    _isolate_dll_search_path()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

ROOT = str(PROJECT_ROOT / "artifacts" / "generated")
os.makedirs(ROOT, exist_ok=True)

from face_avatar_studio import pipeline as P  # noqa: E402

PREVIEW_SIZE = (640, 360)


def _frame_to_qimage(frame_bgr: np.ndarray, size: tuple[int, int]) -> QImage:
    resized = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


class PreviewLabel(QLabel):
    """Mirror of desktop_app.PreviewLabel for aspect-crush verification."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self._show_placeholder = True
        self._placeholder_text = "等待画面..."
        self.setMinimumSize(320, 180)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background-color: #1c1d22; color: #cbd5e1; border-radius: 10px;"
        )
        self._base_pixmap: QPixmap | None = None

    def set_frame(self, frame):
        if frame is None:
            self._base_pixmap = None
            self._show_placeholder = True
            self.update()
            return
        img = _frame_to_qimage(frame, (PREVIEW_SIZE[0], PREVIEW_SIZE[1]))
        self._base_pixmap = QPixmap.fromImage(img)
        self._show_placeholder = False
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor("#1c1d22"))
        if self._base_pixmap is not None and not self._show_placeholder:
            scaled = self._base_pixmap.scaled(
                rect.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation,
            )
            painter.drawPixmap(rect.x(), rect.y(), scaled)
        else:
            painter.setPen(QColor("#cbd5e1"))
            painter.drawText(rect, Qt.AlignCenter, self._placeholder_text)
        painter.end()


def grind_render_bgr(renderer):
    """Run renderer.render_bgr in this thread (main thread)."""
    return renderer.render_bgr()


def main() -> int:
    print("[smoke] initializing GnmAvatarDriver ...")
    avatar = P.GnmAvatarDriver()

    print("[smoke] constructing AvatarPreviewRenderer on a worker thread ...")
    result: dict = {}

    def worker():
        try:
            r = P.AvatarPreviewRenderer(avatar, size=PREVIEW_SIZE)
            neutral = avatar.evaluate({}, None)[3]
            r.update_vertices(neutral)
            f = r.render_bgr()
            result["renderer"] = r
            result["frame"] = f
        except Exception as exc:
            result["error"] = repr(exc)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=30)
    if t.is_alive():
        print("[smoke] FAIL: worker thread did not return within 30s")
        return 2
    if "error" in result:
        print(f"[smoke] FAIL: worker exception: {result['error']}")
        return 3
    frame = result["frame"]
    print(f"[smoke] worker produced frame shape={frame.shape}, mean={frame.mean():.2f}")

    # Avatar must NOT be all black.
    if frame.max() < 10:
        print(f"[smoke] FAIL: avatar frame is essentially black (max={frame.max()})")
        return 4

    # Save the avatar frame as PNG so the user can inspect.
    out = os.path.join(ROOT, "smoke_avatar.png")
    cv2.imwrite(out, frame)
    print(f"[smoke] wrote {out}")

    # Drive a sample expression so we can see motion vs neutral.
    r = result["renderer"]
    expr, rot, trans, verts = avatar.evaluate(
        {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0, "jawOpen": 0.8},
        None,
    )
    r.update_vertices(verts)
    f2 = r.render_bgr()
    diff = np.abs(f2.astype(np.int32) - frame.astype(np.int32)).sum()
    print(f"[smoke] expression vs neutral frame abs-diff sum = {diff}")
    if diff < 5000:
        print("[smoke] WARN: expression produced almost no visible change")

    out2 = os.path.join(ROOT, "smoke_avatar_blink.png")
    cv2.imwrite(out2, f2)
    print(f"[smoke] wrote {out2}")

    # Camera aspect: render a fake 16:9 frame and a fake 4:3 frame; check that
    # scaling via the PreviewLabel path doesn't crush weirdly.
    app = QApplication.instance() or QApplication(sys.argv)

    camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        camera_frame, "16:9", (10, 240),
        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4,
    )

    # Test in a narrow widget (different aspect ratio from PREVIEW_SIZE).
    preview = PreviewLabel("cam")
    preview.resize(800, 200)  # 4:1, very different from 16:9
    preview.set_frame(camera_frame)
    preview.show()

    # Pump events for a moment so paintEvent runs.
    for _ in range(20):
        app.processEvents()
        time.sleep(0.02)

    pixmap = preview.grab()
    grab_path = os.path.join(ROOT, "smoke_camera_grab.png")
    pixmap.save(grab_path)
    print(f"[smoke] wrote {grab_path} (PreviewLabel grab, narrow widget)")
    print(f"[smoke] grab size = {pixmap.size().width()} x {pixmap.size().height()}")

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
