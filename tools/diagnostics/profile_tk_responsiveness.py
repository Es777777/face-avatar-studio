from __future__ import annotations

import multiprocessing
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from face_avatar_studio.tk_ui import FaceAvatarStudioApp


def main() -> int:
    app = FaceAvatarStudioApp()
    delays: list[float] = []
    delayed_events: list[tuple[float, float]] = []
    test_started = time.perf_counter()
    previous = time.perf_counter()

    def heartbeat() -> None:
        nonlocal previous
        now = time.perf_counter()
        delay = max(0.0, now - previous - 0.05)
        delays.append(delay)
        if delay > 0.20:
            delayed_events.append((now - test_started, delay))
        previous = now
        app.root.after(50, heartbeat)

    app.root.after(50, heartbeat)
    app.root.after(250, app.start_camera)
    app.root.after(16000, app.close)
    app.run()
    ordered = sorted(delays)
    p95 = ordered[min(len(ordered) - 1, round(len(ordered) * 0.95))]
    result = {
        "samples": len(delays),
        "mean_delay_ms": round(statistics.mean(delays) * 1000.0, 2),
        "p95_delay_ms": round(p95 * 1000.0, 2),
        "max_delay_ms": round(max(delays) * 1000.0, 2),
        "over_250ms": sum(delay > 0.25 for delay in delays),
        "delayed_events": [
            (round(elapsed, 2), round(delay * 1000.0, 2))
            for elapsed, delay in delayed_events
        ],
    }
    print(result)
    return 0 if result["over_250ms"] == 0 else 2


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
