"""Measure load-time hotspots during Face Avatar Studio launch.

Splits the launch into discrete phases, timing each one. This tells us where
the seconds go:

  1. import face_avatar_studio.pipeline
  2. construct GnmAvatarDriver (loads GNM head v3 + parses numpy arrays)
  3. import mediapipe (lazy)
  4. download face_landmarker.task (cached check)
  5. create_from_options (XNNPACK delegate init)
  6. cv2.VideoCapture ctor timing on this box (we know ~12s on MSMF)
  7. enumerate_cameras (DSHOW scan across indices 0..9)

Run:
    python smoke_init_breakdown.py
"""
from __future__ import annotations

import os
import sys
import time
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

ROOT = PROJECT_ROOT / "artifacts" / "generated"
ROOT.mkdir(parents=True, exist_ok=True)


def phase(name: str) -> float:
    return time.perf_counter()


def main() -> int:
    t_total_start = time.perf_counter()

    # Phase 0: full pipeline import (top-level imports inside pipeline.py)
    t = phase("0")
    from face_avatar_studio import pipeline as P
    dt_import = time.perf_counter() - t
    print(f"[0] import face_avatar_studio.pipeline: {dt_import:.2f}s")

    # Phase 1: GnmAvatarDriver construction (loads GNM head v3 from disk + parses)
    t = phase("1")
    avatar = P.GnmAvatarDriver()
    dt_gnm = time.perf_counter() - t
    print(f"[1] GnmAvatarDriver.__init__: {dt_gnm:.2f}s")
    print(f"    model.num_vertices = {avatar.model.num_vertices}")
    print(f"    expression_dim = {avatar.model.expression_dim}")
    print(f"    num_joints = {avatar.model.num_joints}")

    # Phase 2: enumerate_cameras (this is what AppState triggers on first UI show)
    t = phase("2")
    cams = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cams.append(idx)
        cap.release()
    dt_enum = time.perf_counter() - t
    print(f"[2] enumerate_cameras (DSHOW, 10 indices): {dt_enum:.2f}s, found {cams}")

    # Phase 3: lazy import mediapipe (only loads if MediaPipe tasks touched)
    t = phase("3")
    mp = P._lazy_import_mediapipe()
    dt_mp_import = time.perf_counter() - t
    print(f"[3] lazy import mediapipe: {dt_mp_import:.2f}s")

    # Phase 4: model file already cached?
    from face_avatar_studio.pipeline import USER_CACHE_DIR, FACE_LANDMARKER_URL, download_file
    cache_dir = USER_CACHE_DIR / "mediapipe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "face_landmarker.task"
    cached = model_path.exists() and model_path.stat().st_size > 0
    print(f"[4] face_landmarker.task cached: {cached}, "
          f"size={model_path.stat().st_size if cached else 0}")

    t = phase("4")
    if not cached:
        download_file(FACE_LANDMARKER_URL, model_path)
    dt_dl = time.perf_counter() - t
    print(f"[4] download_file: {dt_dl:.2f}s (cached={cached})")

    # Phase 5: create_from_options (XNNPACK delegate init — historically 5-15s)
    t = phase("5")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    dt_landmarker = time.perf_counter() - t
    print(f"[5] FaceLandmarker.create_from_options: {dt_landmarker:.2f}s")

    # Phase 6: one detect_for_video call to warm inference
    t = phase("6")
    rgb = np.zeros((144, 256, 3), dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect_for_video(mp_image, 0)
    dt_detect = time.perf_counter() - t
    print(f"[6] first detect_for_video: {dt_detect:.2f}s")

    total = time.perf_counter() - t_total_start
    print()
    print("=" * 60)
    print(f"TOTAL = {total:.2f}s")
    print("=" * 60)
    print("Breakdown:")
    print(f"  pipeline import: {dt_import:6.2f}s  ({100*dt_import/total:5.1f}%)")
    print(f"  GNM head load  : {dt_gnm:6.2f}s  ({100*dt_gnm/total:5.1f}%)")
    print(f"  camera enum    : {dt_enum:6.2f}s  ({100*dt_enum/total:5.1f}%)")
    print(f"  mp import      : {dt_mp_import:6.2f}s  ({100*dt_mp_import/total:5.1f}%)")
    print(f"  model download : {dt_dl:6.2f}s  ({100*dt_dl/total:5.1f}%)")
    print(f"  landmarker init: {dt_landmarker:6.2f}s  ({100*dt_landmarker/total:5.1f}%)")
    print(f"  first detect   : {dt_detect:6.2f}s  ({100*dt_detect/total:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
