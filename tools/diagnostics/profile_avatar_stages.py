from __future__ import annotations

import multiprocessing
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from face_avatar_studio.pipeline import FaceTrackingEngine


def main() -> None:
    capture = cv2.VideoCapture(str(PROJECT_ROOT / "artifacts/references/xiaohongshu_reference.mp4"))
    capture.set(cv2.CAP_PROP_POS_MSEC, 3400.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("reference frame unavailable")
    engine = FaceTrackingEngine()
    packet = engine.process_frame(frame, 0, 0.0)
    samples = 12
    timings = {"evaluate": [], "render": []}
    renderer = None
    try:
        from face_avatar_studio.tk_ui import AvatarRasterizer, OpenGLAvatarRenderer

        try:
            renderer = OpenGLAvatarRenderer(engine.avatar.model)
        except Exception:
            renderer = AvatarRasterizer(engine.avatar.model)
        for index in range(samples):
            started = time.perf_counter()
            expression, rotations, translation, vertices = engine.avatar.evaluate(
                packet.blendshapes, packet.face_matrix, packet.landmarks
            )
            timings["evaluate"].append(time.perf_counter() - started)
            started = time.perf_counter()
            renderer.render(vertices, packet.blendshapes)
            timings["render"].append(time.perf_counter() - started)
        print(
            {
                key: {
                    "mean_ms": round(sum(values) / len(values) * 1000.0, 2),
                    "max_ms": round(max(values) * 1000.0, 2),
                }
                for key, values in timings.items()
            }
        )
    finally:
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
