"""Smoke-test the optional FaceVerse V2 52D real-time backend."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from face_avatar_studio.pipeline import (  # noqa: E402
    FaceVerseV2AvatarDriver,
    create_tracking_engine,
    faceverse_v2_backend_status,
)


def main() -> int:
    available, message = faceverse_v2_backend_status()
    print(f"[faceverse-v2] available={available}: {message}")
    if not available:
        return 2

    started = time.perf_counter()
    driver = FaceVerseV2AvatarDriver()
    load_seconds = time.perf_counter() - started
    neutral = driver.neutral_state()[3]
    if driver.model.expression_base.shape[1] != 52:
        print("[faceverse-v2] FAIL: expression basis is not 52-dimensional")
        return 3
    if len(neutral) != driver.model.num_vertices or not np.isfinite(neutral).all():
        print("[faceverse-v2] FAIL: neutral mesh is invalid")
        return 4

    action_vertices: dict[str, np.ndarray] = {}
    for name in ("jawOpen", "mouthSmileLeft", "eyeBlinkLeft", "eyeBlinkRight"):
        driver._smooth_expression.fill(0.0)
        values = {name: 0.8}
        if name == "jawOpen":
            values["geometryJawOpen"] = 0.8
        vertices = driver.evaluate(values, None)[3]
        delta = np.linalg.norm(vertices - neutral, axis=1)
        if int(np.count_nonzero(delta > 1e-6)) < 100 or float(delta.max()) < 1e-3:
            print(f"[faceverse-v2] FAIL: {name} did not deform the mesh")
            return 5
        action_vertices[name] = vertices

    blink_difference = float(
        np.abs(action_vertices["eyeBlinkLeft"] - action_vertices["eyeBlinkRight"]).sum()
    )
    if blink_difference < 1e-3:
        print("[faceverse-v2] FAIL: left and right blink are not independent")
        return 6

    engine = create_tracking_engine("faceverse_v2")
    if engine.avatar.source_label != driver.source_label:
        print("[faceverse-v2] FAIL: engine selected a different model source")
        return 7

    frames = 200
    started = time.perf_counter()
    for index in range(frames):
        phase = index / 12.0
        driver.evaluate(
            {
                "jawOpen": 0.3 + 0.2 * max(0.0, float(np.sin(phase))),
                "mouthSmileLeft": 0.2,
                "mouthSmileRight": 0.2,
            },
            None,
        )
    average_ms = (time.perf_counter() - started) * 1000.0 / frames
    print(
        f"[faceverse-v2] PASS: {driver.source_label}; "
        f"load={load_seconds:.3f}s, mesh_eval={average_ms:.3f}ms/frame"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
