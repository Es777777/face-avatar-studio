from __future__ import annotations

import multiprocessing
import queue
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from face_avatar_studio.tk_ui import CaptureWorker


def main() -> int:
    worker = CaptureWorker(0, "mediapipe")
    worker.start()
    previous_index = -10
    previous_change = time.perf_counter()
    stalls = 0
    try:
        for second in range(18):
            time.sleep(1.0)
            packet = worker.latest_packet()
            index = -99 if packet is None else packet.frame_index
            if index != previous_index:
                previous_index = index
                previous_change = time.perf_counter()
            stall = time.perf_counter() - previous_change
            if stall > 1.0:
                stalls += 1
            statuses: list[str] = []
            while True:
                try:
                    statuses.append(worker.status_queue.get_nowait())
                except queue.Empty:
                    break
            print(
                {
                    "second": second + 1,
                    "alive": worker.is_alive(),
                    "capture_fps": round(worker.capture_fps, 1),
                    "packet": index,
                    "avatar_fps": round(worker.avatar_render_fps, 1),
                    "face": None if packet is None else packet.face_detected,
                    "error": None if packet is None else packet.tracking_error,
                    "stall_sec": round(stall, 2),
                    "status": statuses,
                },
                flush=True,
            )
    finally:
        worker.stop()
        worker.join(timeout=5.0)
    return 0 if stalls == 0 else 2


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
