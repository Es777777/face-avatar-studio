"""Smoke test for CameraWorker backend handshake timing.

Reproduces the **new** subprocess-based backend-iteration logic from
``CameraWorker.run`` (in ``desktop_app.py``):

  1. for each backend in (CAP_MSMF, CAP_DSHOW, CAP_ANY):
       run cv2.VideoCapture + isOpened + read in a SUBPROCESS with timeout
       (per_backend_cap_sec)
       if subprocess returns "read_ok": true → use this backend
  2. Once a backend wins, reopen in parent process (≈0.5-1s)
  3. With no real webcam, all three subprocess probes should hit their
     timeout (≤ per_backend_cap_sec each) and we should fall through cleanly.

The OLD logic (inline ctor in CameraWorker thread) held the GIL on hostile
drivers and could not be time-bounded (single MSMF ctor blocked 12+ seconds
on this box). The new logic should cap total handshake at:
    3 backends × 2.5s = 7.5s worst case
    + parent reopen ≈ 1s
  → ≤ ~8.5s, vs. 30s+ on hostile drivers with the OLD code.

Run:
    python smoke_camera_open.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

from launch_face_avatar_studio import _isolate_dll_search_path  # noqa: E402

if sys.platform == "win32":
    _isolate_dll_search_path()

import cv2  # noqa: E402

ROOT = PROJECT_ROOT / "artifacts" / "generated"
ROOT.mkdir(parents=True, exist_ok=True)


class FakeState:
    def __init__(self) -> None:
        self.logs: deque[str] = deque(maxlen=200)

    def add_log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.logs.append(line)
        print(line, flush=True)


def main() -> int:
    state = FakeState()
    camera_index = 0
    per_backend_cap_sec = 2.5

    # Use the real helper from production code so this test exercises the
    # actual subprocess logic.
    from face_avatar_studio.desktop_app import _probe_camera_backend

    backend_timings: dict[int, float] = {}
    candidate_backends = (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY)
    overall_t0 = time.perf_counter()
    opened_backend: int | None = None
    for backend in candidate_backends:
        t0 = time.perf_counter()
        try:
            result = _probe_camera_backend(
                camera_index, int(backend), timeout_sec=per_backend_cap_sec,
            )
        except subprocess.TimeoutExpired:
            backend_timings[backend] = time.perf_counter() - t0
            state.add_log(
                f"摄像头后端 {backend} {per_backend_cap_sec:.1f}s 内没拿到首帧 "
                f"(subprocess 超时)，跳到下一个后端"
            )
            continue
        except Exception as exc:
            backend_timings[backend] = time.perf_counter() - t0
            state.add_log(f"打开摄像头后端 {backend} 失败：{exc}")
            continue
        rc = result.returncode
        stdout = result.stdout or ""
        backend_timings[backend] = time.perf_counter() - t0
        if rc == 0 and '"read_ok": true' in stdout:
            opened_backend = backend
            break
        state.add_log(
            f"摄像头后端 {backend} {per_backend_cap_sec:.1f}s 内没拿到首帧 "
            f"(rc={rc})，跳到下一个后端"
        )
    probe_elapsed = time.perf_counter() - overall_t0

    # If a backend won, also try the parent-process reopen path.
    reopen_elapsed = 0.0
    parent_cap = None
    if opened_backend is not None:
        reopen_t0 = time.perf_counter()
        parent_cap = cv2.VideoCapture(camera_index, opened_backend)
        reopen_deadline = time.perf_counter() + per_backend_cap_sec
        while not parent_cap.isOpened() and time.perf_counter() < reopen_deadline:
            time.sleep(0.05)
        read_deadline = time.perf_counter() + per_backend_cap_sec
        while time.perf_counter() < read_deadline:
            ok, _ = parent_cap.read()
            if ok:
                break
        reopen_elapsed = time.perf_counter() - reopen_t0
        parent_cap.release()

    total_elapsed = time.perf_counter() - overall_t0
    print()
    print("=" * 60)
    print(f"subprocess probe elapsed = {probe_elapsed:.2f}s")
    print(f"parent reopen elapsed   = {reopen_elapsed:.2f}s")
    print(f"total elapsed           = {total_elapsed:.2f}s")
    print(f"backend timings         = {backend_timings}")
    print(f"opened_backend          = {opened_backend}")
    print("=" * 60)

    # Worst-case bound: 3 backends × 2.5s subprocess probe + ~2.5s parent
    # reopen (reopen also has 2.5s internal deadline). Add some slack for
    # subprocess startup/teardown. Total ≤ ~12s is fine; the OLD code could
    # block 30s+ on a single hostile backend.
    if total_elapsed > 12.0:
        print(
            f"[FAIL] handshake took {total_elapsed:.2f}s (expected ≤ 12s). "
            f"subprocess-based deadline is not enforcing."
        )
        return 1
    print("[OK] subprocess-based deadline enforces handshake budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
