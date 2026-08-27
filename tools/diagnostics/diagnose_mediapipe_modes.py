from __future__ import annotations

import multiprocessing
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from face_avatar_studio.pipeline import FACE_LANDMARKER_URL, USER_CACHE_DIR, download_file


def load_frames() -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(PROJECT_ROOT / "artifacts/references/xiaohongshu_reference.mp4"))
    capture.set(cv2.CAP_PROP_POS_MSEC, 3400.0)
    frames: list[np.ndarray] = []
    while len(frames) < 20:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError("reference video has no readable frames")
    return frames


def run_mode(mode_name: str, frames: list[np.ndarray]) -> None:
    import mediapipe as mp

    model_path = download_file(
        FACE_LANDMARKER_URL,
        USER_CACHE_DIR / "mediapipe" / "face_landmarker.task",
    )
    mode = getattr(mp.tasks.vision.RunningMode, mode_name)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mode,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    started = time.perf_counter()
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        print(f"{mode_name} init={time.perf_counter() - started:.3f}s")
        for label, source, timestamp_base in (
            ("moving", frames, 1000),
            ("same", [frames[0]] * 10, 3000),
        ):
            hits: list[int] = []
            durations: list[float] = []
            for index, frame in enumerate(source):
                height, width = frame.shape[:2]
                scale = min(640.0 / max(width, height), 1.0)
                analysis = cv2.resize(
                    frame,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(analysis, cv2.COLOR_BGR2RGB),
                )
                before = time.perf_counter()
                if mode_name == "VIDEO":
                    result = landmarker.detect_for_video(image, timestamp_base + index * 60)
                else:
                    result = landmarker.detect(image)
                durations.append(time.perf_counter() - before)
                hits.append(int(bool(result.face_landmarks)))
            print(
                f"{mode_name} {label}: hits={hits} "
                f"mean={np.mean(durations):.3f}s max={np.max(durations):.3f}s"
            )


def main() -> None:
    frames = load_frames()
    run_mode(sys.argv[1].upper(), frames)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
