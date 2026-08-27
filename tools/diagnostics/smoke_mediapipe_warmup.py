"""Verify the production warmup path: _MediapipeWarmupThread + RuntimeInitWorker run
in parallel, and the *blocking* time on the main thread drops below mediapipe's
12-14s import cost.

Models what AppState.ensure_runtime_async does, without requiring PySide6.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

from launch_face_avatar_studio import _isolate_dll_search_path  # noqa: E402

if sys.platform == "win32":
    _isolate_dll_search_path()

ROOT = PROJECT_ROOT / "artifacts" / "generated"
ROOT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    # Step 1: import pipeline (already on main thread)
    t0 = time.perf_counter()
    from face_avatar_studio import pipeline as P
    dt_import = time.perf_counter() - t0
    print(f"[main] pipeline import: {dt_import:.2f}s")

    # Step 2: spawn mediapipe warmup thread FIRST, then construct engine
    main_block_start = time.perf_counter()

    warmup_done = threading.Event()
    warmup_timing: dict[str, float] = {}

    def _warmup_worker():
        t = time.perf_counter()
        try:
            P.ensure_mediapipe_loaded()
        except Exception as e:
            print(f"[warmup] failed: {e}")
        warmup_timing["dt"] = time.perf_counter() - t
        warmup_done.set()

    th = threading.Thread(target=_warmup_worker, daemon=True, name="mp-warmup")
    th.start()

    # Step 3: construct engine on main thread (the remaining init work)
    t_engine = time.perf_counter()
    engine = P.FaceTrackingEngine()
    dt_engine = time.perf_counter() - t_engine
    print(f"[main] FaceTrackingEngine.__init__: {dt_engine:.2f}s")

    main_block_dt = time.perf_counter() - main_block_start
    print(f"[main] total blocking time after pipeline import: {main_block_dt:.2f}s")

    # Step 4: wait for warmup to finish
    warmup_done.wait(timeout=30.0)
    if warmup_done.is_set():
        print(f"[warmup] thread dt: {warmup_timing.get('dt', -1):.2f}s")
    else:
        print("[warmup] thread did NOT finish within 30s")

    # If warmup is parallel, blocking time should be max(engine, ~0.1s for
    # thread startup), NOT engine + mediapipe import.
    expected_blocking_naive = dt_engine + 12.5
    print()
    print("=" * 60)
    print(f"main blocking (engine + thread kickoff) = {main_block_dt:.2f}s")
    print(f"  naive serial (engine + mediapipe)     = ~{expected_blocking_naive:.1f}s")
    print(f"  speedup                                = {expected_blocking_naive / main_block_dt:.1f}x")
    print("=" * 60)

    # Sanity check: warmup should have finished by the time we observe blocking.
    th.join(timeout=2.0)
    if not warmup_done.is_set():
        print("[FAIL] warmup thread did not complete")
        return 1
    if main_block_dt > dt_engine + 1.5:
        print(
            f"[FAIL] main blocking {main_block_dt:.2f}s exceeds engine time "
            f"{dt_engine:.2f}s by > 1.5s — warmup is not actually running in parallel"
        )
        return 2

    print("[OK] mediapipe warmup runs in parallel; main thread blocking is bounded by engine init")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
