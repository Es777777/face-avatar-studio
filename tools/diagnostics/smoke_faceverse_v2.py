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
    dental_ranges = (
        driver.model.lower_teeth_vertex_range,
        driver.model.upper_teeth_vertex_range,
    )
    if any(value is None for value in dental_ranges):
        print("[faceverse-v2] FAIL: compatibility mesh has no separate dental ranges")
        return 5

    action_vertices: dict[str, np.ndarray] = {}
    for name in (
        "jawOpen", "mouthSmileLeft", "eyeBlinkLeft", "eyeBlinkRight",
        "eyeSquintLeft", "eyeSquintRight",
    ):
        driver._smooth_expression.fill(0.0)
        values = {name: 0.8}
        if name == "jawOpen":
            values["geometryJawOpen"] = 0.8
        vertices = driver.evaluate(values, None)[3]
        delta = np.linalg.norm(vertices - neutral, axis=1)
        if int(np.count_nonzero(delta > 1e-6)) < 100 or float(delta.max()) < 1e-3:
            print(f"[faceverse-v2] FAIL: {name} did not deform the mesh")
            return 6
        action_vertices[name] = vertices

    blink_difference = float(
        np.abs(action_vertices["eyeBlinkLeft"] - action_vertices["eyeBlinkRight"]).sum()
    )
    if blink_difference < 1e-3:
        print("[faceverse-v2] FAIL: left and right blink are not independent")
        return 7
    squint_difference = float(
        np.abs(action_vertices["eyeSquintLeft"] - action_vertices["eyeSquintRight"]).sum()
    )
    if squint_difference < 1e-3:
        print("[faceverse-v2] FAIL: left and right squint are not independent")
        return 8

    driver._smooth_expression.fill(0.0)
    dental_values = {
        "mouthSmileLeft": 0.8,
        "mouthSmileRight": 0.2,
        "jawOpen": 0.25,
        "geometryJawOpen": 0.25,
        "geometryLipParted": 0.3,
    }
    expression, _, _, corrected = driver.evaluate(dental_values, None)
    uncorrected = (
        driver.model.template_vertex_positions.reshape(-1)
        + driver.model.expression_base @ expression
    ).reshape(-1, 3)
    lower_start, lower_end = driver.model.lower_teeth_vertex_range
    upper_start, upper_end = driver.model.upper_teeth_vertex_range
    lower_delta = corrected[lower_start:lower_end] - uncorrected[lower_start:lower_end]
    upper_delta = corrected[upper_start:upper_end] - uncorrected[upper_start:upper_end]
    untouched_delta = corrected[:lower_start] - uncorrected[:lower_start]
    if float(np.max(np.abs(untouched_delta))) > 1e-6:
        print("[faceverse-v2] FAIL: dental correction changed the face mesh")
        return 9
    if float(np.mean(lower_delta[:, 2])) >= -1e-4:
        print("[faceverse-v2] FAIL: lower teeth were not brought forward")
        return 10
    if float(np.mean(lower_delta[:, 1])) >= -1e-4:
        print("[faceverse-v2] FAIL: lower teeth were not exposed behind the lower lip")
        return 11
    if float(np.mean(upper_delta[:, 2])) <= 1e-4:
        print("[faceverse-v2] FAIL: upper teeth were not retracted")
        return 12
    for start, end in ((lower_start, lower_end), (upper_start, upper_end)):
        corrected_width = float(np.ptp(corrected[start:end, 0]))
        original_width = float(np.ptp(uncorrected[start:end, 0]))
        if corrected_width >= original_width:
            print("[faceverse-v2] FAIL: smile did not narrow a dental row")
            return 13

    lateral_centers: dict[str, float] = {}
    for side in ("mouthLeft", "mouthRight"):
        driver._smooth_expression.fill(0.0)
        lateral_expression, _, _, lateral_vertices = driver.evaluate(
            {side: 0.9, "mouthPucker": 0.8}, None
        )
        lateral_uncorrected = (
            driver.model.template_vertex_positions.reshape(-1)
            + driver.model.expression_base @ lateral_expression
        ).reshape(-1, 3)
        lateral_centers[side] = float(
            np.mean(lateral_vertices[upper_start:upper_end, 0])
            - np.mean(lateral_uncorrected[upper_start:upper_end, 0])
        )
        if float(np.ptp(lateral_vertices[upper_start:upper_end, 0])) >= float(
            np.ptp(lateral_uncorrected[upper_start:upper_end, 0])
        ):
            print(f"[faceverse-v2] FAIL: {side} did not constrain the dental row")
            return 14
    if lateral_centers["mouthLeft"] >= -1e-4:
        print("[faceverse-v2] FAIL: mouthLeft dental row moved toward the cheek")
        return 15
    if lateral_centers["mouthRight"] <= 1e-4:
        print("[faceverse-v2] FAIL: mouthRight dental row moved toward the cheek")
        return 16

    engine = create_tracking_engine("faceverse_v2")
    if engine.avatar.source_label != driver.source_label:
        print("[faceverse-v2] FAIL: engine selected a different model source")
        return 17

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
