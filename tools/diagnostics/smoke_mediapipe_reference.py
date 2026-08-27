from __future__ import annotations

import multiprocessing
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from face_avatar_studio.pipeline import FaceTrackingEngine


def main() -> int:
    capture = cv2.VideoCapture(str(PROJECT_ROOT / "artifacts/references/xiaohongshu_reference.mp4"))
    capture.set(cv2.CAP_PROP_POS_MSEC, 3400.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        print("failed to read reference")
        return 2
    engine = FaceTrackingEngine()
    results = []
    for index in range(5):
        started = time.perf_counter()
        packet = engine.process_frame(frame, index, index / 18.0)
        results.append({
            "index": index,
            "seconds": round(time.perf_counter() - started, 3),
            "face": packet.face_detected,
            "landmarks": 0 if packet.landmarks is None else len(packet.landmarks),
            "blendshapes": len(packet.blendshapes),
            "error": packet.tracking_error,
            "held": packet.face_tracking_hold,
        })
    black = np.zeros_like(frame)
    held_packet = engine.process_frame(black, 5, 5 / 18.0)
    time.sleep(0.3)
    expired_packet = engine.process_frame(black, 6, 6 / 18.0)
    results.extend(
        [
            {
                "index": 5,
                "face": held_packet.face_detected,
                "held": held_packet.face_tracking_hold,
                "landmarks": 0 if held_packet.landmarks is None else len(held_packet.landmarks),
                "error": held_packet.tracking_error,
            },
            {
                "index": 6,
                "face": expired_packet.face_detected,
                "held": expired_packet.face_tracking_hold,
                "landmarks": 0 if expired_packet.landmarks is None else len(expired_packet.landmarks),
                "error": expired_packet.tracking_error,
            },
        ]
    )
    print(results)
    stable = all(result["face"] for result in results[:5])
    hold_ok = held_packet.face_detected and held_packet.face_tracking_hold
    expiry_ok = not expired_packet.face_detected and not expired_packet.face_tracking_hold
    return 0 if stable and hold_ok and expiry_ok else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
