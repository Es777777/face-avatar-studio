"""Face Avatar Studio 桌面应用（PySide6 版本）。

替换原本的 Dear PyGui 实现，使用 Qt 提供更精致的 UI、稳定的实时刷新、
以及修复原版的多处功能缺陷：

- ``_ui_refresh_loop`` 改用 ``QTimer``，保证每秒 30 次同步
- 录制时显示 REC 计时与麦克风电平
- Whisper 转写通过回调显示进度
- 设备/运行时空状态使用占位卡片
- 退出时若正在录制/导出，先询问
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

# 必须在 import cv2 / mediapipe / tensorflow 之前关掉它们的 C++ 端噪声，
# 否则控制台会被 absl / oneDNN / XNNPACK 刷屏，UI 主线程的 print 输出也会被打断。
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

# absl 没初始化前就 print 的 warning，会写到真 stderr 上很刺眼，
# 把它和 python warnings 都路由到 DEVNULL。
try:
    import absl.logging as _absl_logging  # type: ignore
    _absl_logging.set_verbosity(_absl_logging.ERROR)
    _absl_logging.use_python_logging()
except Exception:
    pass
try:
    import warnings as _warnings
    _warnings.filterwarnings("ignore")
except Exception:
    pass
try:
    # 一旦 mediapipe 被 import 进来，它内部的日志句柄已经被初始化；
    # 把 root logger 设到 WARNING，并把 mediapipe/tensorflow/absl 三个模块的 logger
    # 全降到 ERROR。
    import logging as _logging
    _logging.getLogger().setLevel(_logging.WARNING)
    for _name in ("mediapipe", "tensorflow", "absl", "absl.logging", "tflite"):
        _logging.getLogger(_name).setLevel(_logging.ERROR)
except Exception:
    pass

import cv2
import numpy as np

from . import pipeline as _pipeline

try:
    from PySide6.QtCore import (
        QEasingCurve,
        QObject,
        QPropertyAnimation,
        QSize,
        Qt,
        QTimer,
        Signal,
        Slot,
    )
    from PySide6.QtGui import (
        QAction,
        QCloseEvent,
        QColor,
        QFont,
        QImage,
        QPainter,
        QPalette,
        QPixmap,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFrame,
        QGraphicsDropShadowEffect,
        QGridLayout,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpacerItem,
        QStatusBar,
        QStyle,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover - 让错误冒到 main 弹窗
    raise RuntimeError(
        "无法导入 PySide6，请先执行 `python -m pip install -r requirements.txt`"
    ) from exc


# ---------------------------------------------------------------------------
# 常量 / 主题
# ---------------------------------------------------------------------------

DEFAULT_FPS = 30
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_STT_MODEL = "small"

PREVIEW_SIZE = (640, 360)
PERFORMANCE_PROFILES: dict[str, dict[str, Any]] = {
    "流畅": {
        "capture_size": (640, 360),
        "analysis_size": (256, 144),
        "analysis_stride": 2,
        "preview_stride": 1,
        "avatar_fps": 12,
        "camera_fps": 30,
    },
    "平衡": {
        "capture_size": (960, 540),
        "analysis_size": (320, 180),
        "analysis_stride": 2,
        "preview_stride": 1,
        "avatar_fps": 18,
        "camera_fps": 30,
    },
    "高质量": {
        "capture_size": (1280, 720),
        "analysis_size": (480, 270),
        "analysis_stride": 1,
        "preview_stride": 1,
        "avatar_fps": 24,
        "camera_fps": 30,
    },
}

SPEECH_LANGUAGE_OPTIONS = {
    "自动": "auto",
    "中文": "zh",
    "英文": "en",
}
SPEECH_LANGUAGE_LABELS = {value: key for key, value in SPEECH_LANGUAGE_OPTIONS.items()}


# 暖色调色板
COLOR_BG = "#F7F4EC"
COLOR_CARD = "#FFFFFF"
COLOR_CARD_ALT = "#FBF8F1"
COLOR_BORDER = "#E6DFD2"
COLOR_BORDER_SOFT = "#EFE9DC"
COLOR_TEXT = "#1F2937"
COLOR_TEXT_MUTED = "#6B6B6B"
COLOR_TEXT_DISABLED = "#9CA3AF"
COLOR_PRIMARY = "#2D8B6E"
COLOR_PRIMARY_DARK = "#1F6A53"
COLOR_PRIMARY_SOFT = "#DCEFE6"
COLOR_RECORD = "#C9593D"
COLOR_RECORD_SOFT = "#F8DCD3"
COLOR_WARN = "#A66A2D"
COLOR_WARN_SOFT = "#F4E6CC"
COLOR_INFO = "#3F5F95"
COLOR_INFO_SOFT = "#DDE6F5"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def enumerate_cameras(max_index: int = 10) -> list[dict[str, Any]]:
    cameras: list[dict[str, Any]] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if cap.isOpened():
                cameras.append({"index": index, "label": f"摄像头 {index}"})
        finally:
            cap.release()
    return cameras


def enumerate_microphones() -> list[dict[str, Any]]:
    import sounddevice as sd

    devices: list[dict[str, Any]] = []
    for index, device in enumerate(sd.query_devices()):
        if device.get("max_input_channels", 0) > 0:
            devices.append({"index": index, "label": str(device["name"])})
    return devices


def choose_directory(parent: QWidget | None, initial: str) -> str:
    selected = QFileDialog.getExistingDirectory(parent, "选择保存文件夹", initial)
    return selected or ""


def open_path(path: str | os.PathLike[str]) -> None:
    if not path:
        return
    p = os.fspath(path)
    if not os.path.exists(p):
        return
    if sys.platform == "win32":
        os.startfile(p)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        os.system(f'open "{p}"')
    else:
        os.system(f'xdg-open "{p}"')


def frame_to_qimage(frame_bgr: np.ndarray, size: tuple[int, int]) -> QImage:
    resized = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def level_to_db(level: float) -> float:
    if level <= 1e-4:
        return -60.0
    return float(20.0 * np.log10(level))


# ---------------------------------------------------------------------------
# AppState - 复用现有 worker 设计
# ---------------------------------------------------------------------------

class AppState:
    """状态容器 + 后台线程协调。"""

    def __init__(self, parent: QObject | None = None) -> None:
        from PySide6.QtCore import QObject as _QObject
        # 兼容 PySide6 6.5+ 不允许 None parent 的情况
        self._signals = _QtSignals()
        self.engine: Any = None
        self.avatar_renderer: Any = None
        self.capture_worker: CameraWorker | None = None
        self.avatar_worker: AvatarRenderWorker | None = None
        self.export_worker: ExportWorker | None = None
        self.runtime_worker: RuntimeInitWorker | None = None
        self.recorder: _pipeline.SessionRecorder | None = None
        # AvatarRenderWorker 用这个 flag 决定 _on_avatar_frame chip 文案：True 时
        # chip 显示"已驱动"，False 时显示"等待摄像头..."（warm-up 期间中性 mesh 占位）。
        self._avatar_first_packet_seen: bool = False
        self.transcriber: _pipeline.WhisperTranscriber | None = None
        # mediapipe-warmup 后台线程（见 ensure_runtime_async）。它和
        # RuntimeInitWorker 并行跑 mediapipe import，目的是把 14s 启动延迟从
        # 主线程上挪开，避免 UI 黑屏这么久。
        self.mediapipe_warmup_thread: _MediapipeWarmupThread | None = None
        self.latest_packet: Any = None
        self.latest_camera_frame: np.ndarray | None = None
        self.latest_avatar_frame: np.ndarray | None = None
        self.latest_camera_seq = 0
        self.latest_avatar_seq = 0
        self.logs: deque[str] = deque(maxlen=400)
        self.output_root = ensure_directory(Path.home() / "FaceAvatarStudio")
        self.recording = False
        self.exporting = False
        self.runtime_loading = False
        self.runtime_ready = False
        self.runtime_error = ""
        self.devices_loading = False
        self.devices_ready = False
        self.devices_error = ""
        self.pending_camera_items: list[str] = ["正在扫描摄像头..."]
        self.pending_mic_items: list[str] = ["正在扫描麦克风..."]
        self.pending_camera_index: int | None = None
        self.performance_profile_name = "平衡"
        self.current_camera_index: int | None = None
        self.current_mic_index: int | None = None
        self.current_mic_name: str = ""
        self.current_speech_language = "auto"
        self.current_stt_model = DEFAULT_STT_MODEL
        self.last_outputs: _pipeline.RecordingOutputs | None = None
        self.export_progress: tuple[float, str] = (0.0, "")
        # 保护 engine.avatar / 渲染器 / latest_packet 在 CameraWorker 与 AvatarRenderWorker
        # 之间互斥访问。VTK 渲染器不是线程安全的；avatar.evaluate / numpy 数组跨线程也
        # 容易触发访问冲突，所以两边都必须在锁内操作。
        self.engine_lock = threading.RLock()
        self.renderer_lock = threading.RLock()

    # Qt signals -------------------------------------------------
    @property
    def signals(self) -> "_QtSignals":
        return self._signals

    def add_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        with self._signals.lock:
            self.logs.append(line)
        self._signals.log_added.emit(line)

    def consume_packet(self, packet: Any) -> None:
        self._signals.packet_ready.emit(packet)

    def set_last_outputs(self, outputs: _pipeline.RecordingOutputs) -> None:
        self.last_outputs = outputs
        self._signals.outputs_changed.emit(outputs)

    def set_export_progress(self, ratio: float, message: str) -> None:
        self.export_progress = (max(0.0, min(1.0, ratio)), message)
        self._signals.export_progress.emit(ratio, message)

    def performance_settings(self) -> dict[str, Any]:
        profile = PERFORMANCE_PROFILES.get(self.performance_profile_name, PERFORMANCE_PROFILES["平衡"])
        return dict(profile)

    def set_performance_profile(self, profile_name: str) -> None:
        if profile_name in PERFORMANCE_PROFILES:
            self.performance_profile_name = profile_name

    def ensure_runtime_async(self) -> None:
        if self.runtime_ready or self.runtime_loading:
            return
        self.runtime_loading = True
        self._signals.runtime_state_changed.emit()
        # 关键：mediapipe 首次 import 在 Windows 上要 ~14s（TF Lite / absl /
        # XNNPACK），如果让 RuntimeInitWorker 主线程导入，会把 UI 卡 14s。
        # 所以在这里先 spawn 一个纯 daemon 线程把 mediapipe import warmup
        # 起来，与 GnmAvatarDriver 构造并行。这样 RuntimeInitWorker 主线程
        # 进 _ensure_landmarker 时，_lazy_import_mediapipe() 已经是 fast path
        # —— 整体启动从 14s+ 降到 ~1-2s（mediapipe 不阻塞主线程）。
        self.mediapipe_warmup_thread = _MediapipeWarmupThread()
        self.mediapipe_warmup_thread.start()
        self.runtime_worker = RuntimeInitWorker(self)
        self.runtime_worker.start()

    def ensure_device_scan_async(self) -> None:
        if self.devices_loading or self.devices_ready:
            return
        self.devices_loading = True
        self._signals.devices_state_changed.emit()
        self.device_scan_worker = DeviceScanWorker(self)
        self.device_scan_worker.start()

    def request_camera_start(self, camera_index: int, profile_name: str | None = None) -> None:
        if profile_name is not None:
            self.set_performance_profile(profile_name)
        if not self.runtime_ready:
            self.pending_camera_index = camera_index
            self.ensure_runtime_async()
            self.add_log(f"运行时仍在加载，摄像头 {camera_index} 稍后启动。")
            return
        self.start_camera(camera_index)

    def _stop_stream_workers(self) -> None:
        capture_worker = self.capture_worker
        avatar_worker = self.avatar_worker
        self.capture_worker = None
        self.avatar_worker = None
        if capture_worker is not None:
            capture_worker.stop()
            capture_worker.join(timeout=2)
        if avatar_worker is not None:
            avatar_worker.stop()
            avatar_worker.join(timeout=2)

    def start_camera(self, camera_index: int) -> None:
        settings = self.performance_settings()
        if not self.runtime_ready or self.engine is None:
            self.pending_camera_index = camera_index
            self.ensure_runtime_async()
            self.add_log(f"摄像头 {camera_index} 已排队，等待运行时就绪。")
            return
        # 先把 avatar worker 启动起来（在摄像头握手前出中性 mesh 占位），
        # 用户立刻看到头像预览"已就绪"而不是全黑；再启动 camera worker。
        self._start_avatar_preview_if_needed()
        self._stop_stream_workers()
        self.current_camera_index = camera_index
        self.latest_camera_frame = None
        self.latest_avatar_frame = None
        self.latest_camera_seq = 0
        self.latest_avatar_seq = 0
        self._avatar_first_packet_seen = False
        self.capture_worker = CameraWorker(
            self,
            camera_index,
            frame_size=tuple(settings["capture_size"]),
            analysis_size=tuple(settings["analysis_size"]),
            analysis_stride=int(settings["analysis_stride"]),
            preview_stride=int(settings["preview_stride"]),
            fps=int(settings["camera_fps"]),
        )
        self.avatar_worker = AvatarRenderWorker(
            self,
            fps=int(settings["avatar_fps"]),
            size=PREVIEW_SIZE,
        )
        self.capture_worker.start()
        self.avatar_worker.start()
        self.add_log(
            f"已切换到摄像头 {camera_index}，性能档位：{self.performance_profile_name}。"
        )

    def _start_avatar_preview_if_needed(self) -> None:
        """Start the AvatarRenderWorker if it isn't already running.

        Called from RuntimeInitWorker once the engine is ready, and from
        start_camera. The avatar preview is independent of the camera — it
        just needs the engine — so starting it before the camera handshake
        finishes means the user sees the neutral mesh within a second of
        "runtime ready" instead of staring at a black box while OpenCV
        negotiates with the camera driver.
        """
        if self.avatar_worker is not None and self.avatar_worker.is_alive():
            return
        if self.engine is None:
            return
        settings = self.performance_settings()
        self.avatar_worker = AvatarRenderWorker(
            self,
            fps=int(settings["avatar_fps"]),
            size=PREVIEW_SIZE,
        )
        self.avatar_worker.start()

    def start_recording(
        self,
        output_root: str,
        mic_index: int,
        mic_name: str,
        language: str,
        stt_model: str,
    ) -> dict[str, Any]:
        if self.recording or self.exporting:
            raise RuntimeError("录制已在进行中。")
        if self.engine is None:
            raise RuntimeError("人脸跟踪运行时还未就绪。")
        self.output_root = ensure_directory(Path(output_root).expanduser())
        settings = self.performance_settings()
        self.recorder = _pipeline.SessionRecorder(
            output_root=self.output_root,
            frame_size=tuple(settings["capture_size"]),
            fps=int(settings["camera_fps"]),
            sample_rate=DEFAULT_SAMPLE_RATE,
            stt_language=language,
        )
        self.transcriber = _pipeline.WhisperTranscriber(model_name=stt_model)
        self.transcriber.set_progress_callback(self._on_transcribe_progress)
        self.recorder.set_audio_level_callback(self._on_audio_level)
        self.current_mic_index = mic_index
        self.current_mic_name = mic_name
        self.current_speech_language = language
        self.current_stt_model = stt_model
        session_dir = self.recorder.start(
            camera_name=f"摄像头 {self.current_camera_index}",
            microphone_name=mic_name,
            mic_device_index=mic_index,
        )
        self.recording = True
        self.add_log(f"开始录制：{session_dir}")
        return {"session_dir": str(session_dir)}

    def _on_audio_level(self, level: float) -> None:
        self._signals.audio_level.emit(level)

    def _on_transcribe_progress(self, ratio: float, message: str) -> None:
        self.set_export_progress(ratio, message)

    def stop_recording(self) -> None:
        if self.recorder is None and self.capture_worker is None:
            return
        if self.recorder is not None and self.recorder.active:
            self.recorder.stop_capture()
        self.recording = False
        self.exporting = True
        self.set_export_progress(0.0, "正在合并音视频...")
        self.export_worker = ExportWorker(self)
        self.export_worker.start()
        self.add_log("正在导出录制结果并转写语音...")


# ---------------------------------------------------------------------------
# Qt Signals（集中定义便于 worker 触发）
# ---------------------------------------------------------------------------

class _QtSignals(QObject):
    log_added = Signal(str)
    packet_ready = Signal(object)
    outputs_changed = Signal(object)
    runtime_state_changed = Signal()
    devices_state_changed = Signal()
    audio_level = Signal(float)
    export_progress = Signal(float, str)
    camera_seq = Signal(int)
    avatar_seq = Signal(int)
    camera_frame = Signal(object)
    avatar_frame = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.RLock()


# ---------------------------------------------------------------------------
# 后台 Worker
# ---------------------------------------------------------------------------

import threading  # noqa: E402  (放在 signals 之后以避免循环)


def _probe_camera_backend(camera_index: int, backend: int, timeout_sec: float):
    """Probe a single OpenCV camera backend in a subprocess.

    cv2.VideoCapture(...) 的 ctor 在 hostile UVC 驱动下会**持有 GIL** 阻塞
    调用线程 10-30s（与 OpenCV 5.0 + MSMF 已知 issue 一致），单纯 try/except
    或子线程 join 都拦不住。所以我们把 ctor + isOpened + 第一次 read() 关进
    一个独立 Python 子进程，subprocess.run(..., timeout=timeout_sec) 在
    超时后会主动 kill，子进程死了之后 cv2 释放文件句柄，驱动回到可重试状态。

    Returns ``(rc, stdout, stderr)``。stdout 形如
    ``{"opened": true, "read_ok": true}`` 时表示该后端可用。
    """
    probe_script = (
        "import sys, json\n"
        "import cv2\n"
        "backend = int(sys.argv[1])\n"
        "idx = int(sys.argv[2])\n"
        "out = {}\n"
        "try:\n"
        "    cap = cv2.VideoCapture(idx, backend)\n"
        "    out['opened'] = bool(cap.isOpened())\n"
        "    if cap.isOpened():\n"
        "        ok, _ = cap.read()\n"
        "        out['read_ok'] = bool(ok)\n"
        "    cap.release()\n"
        "except Exception as e:\n"
        "    out['exc'] = repr(e)\n"
        "print(json.dumps(out), flush=True)\n"
    )
    cmd = [sys.executable, "-c", probe_script, str(int(backend)), str(int(camera_index))]
    # 把子进程的 stderr 也收回来，避免 Windows 下子进程卡死时父进程控制台被
    # mediapipe/absl 的 banner 刷屏；stdout 单独收，方便我们 parse JSON。
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


class CameraWorker(threading.Thread):
    def __init__(
        self,
        state: AppState,
        camera_index: int,
        frame_size: tuple[int, int],
        analysis_size: tuple[int, int],
        analysis_stride: int,
        preview_stride: int,
        fps: int = DEFAULT_FPS,
    ) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.camera_index = camera_index
        self.frame_size = frame_size
        self.analysis_size = analysis_size
        self.analysis_stride = max(1, int(analysis_stride))
        self.preview_stride = max(1, int(preview_stride))
        self.fps = fps
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        engine = self.state.engine
        if engine is None:
            self.state.add_log("摄像头线程跳过：运行时还没准备好。")
            return
        # CAP_DSHOW 在某些 Windows 摄像头驱动握手时会出现 access violation；优先试
        # CAP_MSMF（Media Foundation），再退回 DSHOW，最后才用默认后端。
        #
        # 关键：cv2.VideoCapture(...) 的构造器在 hostile UVC 驱动（MSMF 常见）下
        # 会**持有 GIL 阻塞线程** 10-30 秒，单纯 try/timeout 或子线程 join 都拦
        # 不住 —— 必须把它关进 subprocess，超时就 kill。所以每个后端的握手分两
        # 段：subprocess_probe(≤per_backend_cap_sec) 探测能用的后端，再在主
        # 进程里用赢到的后端重新 cv2.VideoCapture(...)，这一步 stream 已经被子
        # 进程协商过一次，主进程 ctor 复用驱动内核态对象，~0.5-1s。
        cap = None
        opened_backend: int | None = None
        backend_timings: dict[int, float] = {}
        per_backend_cap_sec = 2.5
        # 先列出所有备选后端，让子进程按顺序探一遍；任一探成功就停。
        candidate_backends = (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY)
        probe_overall_t0 = time.perf_counter()
        for backend in candidate_backends:
            t0 = time.perf_counter()
            try:
                result = _probe_camera_backend(
                    self.camera_index, int(backend),
                    timeout_sec=per_backend_cap_sec,
                )
            except subprocess.TimeoutExpired:
                # subprocess.run 在 timeout 之后会主动 kill 子进程，但同时抛
                # TimeoutExpired —— CompletedProcess 不会返回。我们认为该后端
                # 没在 deadline 内出首帧，跳到下一个。
                backend_timings[backend] = time.perf_counter() - t0
                self.state.add_log(
                    f"摄像头后端 {backend} {per_backend_cap_sec:.1f}s 内没拿到首帧 "
                    f"(subprocess 超时)，跳到下一个后端"
                )
                continue
            except Exception as exc:
                backend_timings[backend] = time.perf_counter() - t0
                self.state.add_log(f"打开摄像头后端 {backend} 失败：{exc}")
                continue
            rc = result.returncode
            stdout = result.stdout or ""
            backend_timings[backend] = time.perf_counter() - t0
            if rc == 0 and "\"read_ok\": true" in stdout:
                opened_backend = backend
                break
            self.state.add_log(
                f"摄像头后端 {backend} {per_backend_cap_sec:.1f}s 内没拿到首帧 "
                f"(rc={rc})，跳到下一个后端"
            )
        if opened_backend is None:
            self.state.add_log(
                f"无法打开摄像头 {self.camera_index}。各后端耗时：{backend_timings} "
                f"总探针={time.perf_counter() - probe_overall_t0:.2f}s"
            )
            return
        # 主进程以赢到的后端重开。子进程已经把驱动内核态对象协商过一遍，
        # 主进程这次 ctor + isOpened 一般 <1s；如果仍卡则再用子进程探一次
        # 兜底，绝不让 CameraWorker 卡死超过 ~3 倍 per_backend_cap_sec。
        reopen_t0 = time.perf_counter()
        cap = cv2.VideoCapture(self.camera_index, opened_backend)
        # 兜底：万一主进程 ctor 又卡死（驱动被多进程争用），强制设上限。
        reopen_deadline = time.perf_counter() + per_backend_cap_sec
        while not cap.isOpened() and time.perf_counter() < reopen_deadline:
            time.sleep(0.05)
        # 首帧 read() 也兜底限时。
        read_deadline = time.perf_counter() + per_backend_cap_sec
        first_ok = False
        while time.perf_counter() < read_deadline:
            ok, _ = cap.read()
            if ok:
                first_ok = True
                break
        reopen_elapsed = time.perf_counter() - reopen_t0
        if not first_ok:
            cap.release()
            self.state.add_log(
                f"主进程复开后端 {opened_backend} 仍拿不到首帧 "
                f"({reopen_elapsed:.2f}s)。各后端耗时：{backend_timings}"
            )
            return
        self.state.add_log(
            f"摄像头后端握手成功 backend={opened_backend} "
            f"subprocess探测={backend_timings.get(opened_backend, 0.0):.2f}s "
            f"主进程复开={reopen_elapsed:.2f}s"
        )
        width, height = self.frame_size
        # 关键：CAP_PROP_FRAME_WIDTH/HEIGHT/FPS/BUFFERSIZE 在 MSMF 后端下会触发
        # 驱动协商（重新打开 stream category），单次就要 5-15 秒。某些国产 UVC 摄
        # 像头驱动协商 4 次接近 30 秒。对生产 30fps 来说协商后的实际分辨率 / FPS
        # 跟默认值差异不大，所以**先不协商**，用驱动默认值启动；等首帧到了再
        # 在 run 循环里异步设置一次，让后续 frame 走协商后的分辨率。
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # 首次 grab() 会触发硬件协商，耗时数秒。先把前几帧"烧掉"，避免运行时加载
        # 完成后 UI 仍然长时间黑屏。同时记录每次 read() 耗时，方便看摄像头是真的慢
        # 还是某次握手卡住。
        warm_deadline = time.perf_counter() + 3.0
        warm_ok = False
        warm_attempts = 0
        warm_first_read_ms = 0.0
        while time.perf_counter() < warm_deadline:
            warm_attempts += 1
            t0 = time.perf_counter()
            ok, _ = cap.read()
            read_ms = (time.perf_counter() - t0) * 1000.0
            if warm_first_read_ms == 0.0 and ok:
                warm_first_read_ms = read_ms
            if ok:
                warm_ok = True
                break
            time.sleep(0.05)
        if not warm_ok:
            self.state.add_log(
                f"摄像头 {self.camera_index} 预热超时 ({warm_attempts} 次尝试)，"
                f"first_ok_read_ms={warm_first_read_ms:.1f}，仍尝试读帧。"
            )
        else:
            self.state.add_log(
                f"摄像头 {self.camera_index} 预热 {warm_attempts} 次后拿到首帧 "
                f"({warm_first_read_ms:.1f}ms)"
            )
        # 拿到首帧之后再异步协商一次分辨率 / FPS；这次只设一次，多数驱动
        # 已经在 stream 里了所以不会重新协商 stream category。
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        except Exception:
            pass
        start_time = time.perf_counter()
        frame_index = 0
        last_packet: Any = None
        signals = self.state.signals
        self.state.add_log(f"摄像头 {self.camera_index} 已启动。")
        try:
            # 第一帧：不等 stride，先把画面刷到 UI
            last_read_ms = 0.0
            last_engine_ms = 0.0
            slow_read_log_tick = 0.0
            slow_engine_log_tick = 0.0
            while not self.stop_event.is_set():
                t0 = time.perf_counter()
                ok, frame = cap.read()
                last_read_ms = (time.perf_counter() - t0) * 1000.0
                if not ok:
                    # read 卡住的话用户既看不到画面也拿不到 packet —— 打心跳让日志
                    # 留证，并把 heartbeat 周期和真实耗时都打出来，方便定位是驱动卡
                    # 还是 OpenCV 后端排队。
                    now = time.perf_counter()
                    if now - slow_read_log_tick >= 1.0:
                        self.state.add_log(
                            f"[CameraWorker] cap.read 失败 last_read_ms={last_read_ms:.1f}"
                        )
                        slow_read_log_tick = now
                    time.sleep(0.05)
                    continue
                # read 偶尔 >120ms 的话，几乎肯定 OpenCV / 驱动在排队 —— 这种状态
                # 用户看到的画面也是"卡"的。每秒最多打一次心跳。
                if last_read_ms > 120.0:
                    now = time.perf_counter()
                    if now - slow_read_log_tick >= 1.0:
                        self.state.add_log(
                            f"[CameraWorker] cap.read 偏慢 {last_read_ms:.1f}ms "
                            f"(frame_index={frame_index})"
                        )
                        slow_read_log_tick = now
                frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
                timestamp_sec = time.perf_counter() - start_time
                packet: Any = None
                if frame_index % self.analysis_stride == 0 or last_packet is None:
                    analysis_frame = cv2.resize(
                        frame, self.analysis_size, interpolation=cv2.INTER_AREA,
                    )
                    # engine.process_frame 会触发 avatar.evaluate，avatar 的内部
                    # 状态会随后被 AvatarRenderWorker 读取；这里用 engine_lock
                    # 把"评估 -> 拷贝 vertices"原子化，避免渲染器读到半更新数据。
                    engine_t0 = time.perf_counter()
                    with self.state.engine_lock:
                        packet = engine.process_frame(analysis_frame, frame_index, timestamp_sec)
                    last_engine_ms = (time.perf_counter() - engine_t0) * 1000.0
                    if last_engine_ms > 200.0:
                        now = time.perf_counter()
                        if now - slow_engine_log_tick >= 1.0:
                            self.state.add_log(
                                f"[CameraWorker] engine.process_frame 偏慢 {last_engine_ms:.1f}ms "
                                f"face_detected={packet.face_detected} "
                                f"(frame_index={frame_index})"
                            )
                            slow_engine_log_tick = now
                    # 拷贝 numpy 缓冲，避免渲染器与 CameraWorker 共享同一段内存
                    safe_vertices = (
                        packet.vertices.copy() if packet.vertices is not None else None
                    )
                    safe_frame_bgr = (
                        packet.frame_bgr.copy() if packet.frame_bgr is not None else None
                    )
                    safe_landmarks = (
                        packet.landmarks.copy() if packet.landmarks is not None else None
                    )
                    safe_packet = _pipeline.FramePacket(
                        frame_index=packet.frame_index,
                        timestamp_sec=packet.timestamp_sec,
                        frame_bgr=safe_frame_bgr if safe_frame_bgr is not None else np.zeros(
                            (self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8,
                        ),
                        landmarks=safe_landmarks,
                        blendshapes=dict(packet.blendshapes),
                        expression=packet.expression.copy() if packet.expression is not None else None,
                        rotations=packet.rotations.copy() if packet.rotations is not None else None,
                        translation=packet.translation.copy() if packet.translation is not None else None,
                        face_matrix=(
                            packet.face_matrix.copy() if packet.face_matrix is not None else None
                        ),
                        vertices=safe_vertices if safe_vertices is not None else np.zeros(
                            (0, 3), dtype=np.float32,
                        ),
                        face_detected=packet.face_detected,
                        face_tracking_hold=packet.face_tracking_hold,
                    )
                    last_packet = safe_packet
                    self.state.latest_packet = safe_packet
                    signals.packet_ready.emit(safe_packet)
                    if frame_index == 0:
                        self.state.add_log(
                            f"[CameraWorker] 首个 packet 已发出 "
                            f"frame_index={safe_packet.frame_index} "
                            f"face_detected={safe_packet.face_detected} "
                            f"vertices.shape="
                            f"{None if safe_packet.vertices is None else safe_packet.vertices.shape}"
                        )
                # 第一帧强制入队（preview_stride 一般是 1，所以这里只是兜底）
                if frame_index == 0 or frame_index % self.preview_stride == 0:
                    self.state.latest_camera_frame = frame.copy()
                    self.state.latest_camera_seq += 1
                    signals.camera_seq.emit(self.state.latest_camera_seq)
                    signals.camera_frame.emit(frame)
                recorder = self.state.recorder
                if recorder is not None and recorder.active and last_packet is not None:
                    recorder.append_frame(
                        frame,
                        last_packet,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                    )
                frame_index += 1
        except Exception as exc:
            self.state.add_log(f"摄像头线程异常：{exc}")
        finally:
            cap.release()
            self.state.add_log("摄像头流已停止。")


class RuntimeInitWorker(threading.Thread):
    def __init__(self, state: AppState) -> None:
        super().__init__(daemon=True)
        self.state = state

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            self.state.add_log("正在加载人脸跟踪运行时...")
            # ``FaceTrackingEngine.__init__`` 同步构造 GnmAvatarDriver（~1.2s）
            # 并装好 mediapipe 的 lazy 模块缓存。mediapipe 本身的 ~14s 已经在
            # AppState.ensure_runtime_async 阶段由 ``_MediapipeWarmupThread`` 提
            # 前导入；这里只剩 fast path（不到 0.1s）。
            engine = _pipeline.FaceTrackingEngine()
            dt_engine = time.perf_counter() - t0
            self.state.add_log(
                f"[runtime] GnmAvatarDriver + mediapipe-warm-path "
                f"完成，耗时 {dt_engine:.2f}s"
            )
            self.state.engine = engine
            self.state.runtime_ready = True
            self.state.runtime_loading = False
            self.state.runtime_error = ""
            self.state.runtime_worker = None
            pending_camera = self.state.pending_camera_index
            self.state.pending_camera_index = None
            self.state.signals.runtime_state_changed.emit()
            self.state.add_log("人脸跟踪运行时已就绪。")
            # 关键：engine 就绪后，先把 AvatarRenderWorker 启动起来 —— 它不依赖
            # 摄像头，能立即画出 neutral mesh 占位，等 CameraWorker 第一个 packet
            # 来了再切到真实驱动。如果不这样做，AvatarWorker 就要等用户点
            # "启动摄像头"，而用户看到的"启动以后就啥都没了"就是这段空窗期。
            self.state._start_avatar_preview_if_needed()
            if pending_camera is not None:
                self.state.start_camera(pending_camera)
            elif self.state.current_camera_index is None and self.state.devices_ready and self.state.pending_camera_items:
                first = self.state.pending_camera_items[0]
                if first and ":" in first:
                    try:
                        idx = int(first.split(":", 1)[0])
                    except ValueError:
                        idx = None
                    if idx is not None:
                        self.state.start_camera(idx)
        except Exception as exc:
            self.state.runtime_error = str(exc)
            self.state.runtime_loading = False
            self.state.runtime_ready = False
            self.state.runtime_worker = None
            self.state.signals.runtime_state_changed.emit()
            self.state.add_log(f"运行时加载失败：{exc}")


class _MediapipeWarmupThread(threading.Thread):
    """Pure background thread that just imports mediapipe.

    Why: mediapipe 首次 import 在 Windows 上要 ~14s（TF Lite / absl / XNNPACK
    的 C++ DLL 加载 + 一连串 banner 输出）。如果让 RuntimeInitWorker 主线程
    同步 import，整个启动 UI 会黑屏 14s，且这段时间内用户既看不到 loading
    也无法取消。我们把它放进 daemon 线程让它和 GnmAvatarDriver 构造并行
    跑：RuntimeInitWorker 主线程进 _ensure_landmarker 时大概率命中 fast path。
    """

    def __init__(self) -> None:
        super().__init__(daemon=True, name="mediapipe-warmup")

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            _pipeline.ensure_mediapipe_loaded()
            dt = time.perf_counter() - t0
            # 注意：这条日志可能比 AppState 初始化还早（spawn 顺序敏感），用
            # 单独 try/except 包起来，logging 出错也不影响整体逻辑。
            print(f"[mediapipe-warmup] import mediapipe 用了 {dt:.2f}s", flush=True)
        except Exception as exc:
            print(f"[mediapipe-warmup] 失败：{exc}", flush=True)


class DeviceScanWorker(threading.Thread):
    def __init__(self, state: AppState) -> None:
        super().__init__(daemon=True)
        self.state = state

    def run(self) -> None:
        try:
            cameras = enumerate_cameras()
            microphones = enumerate_microphones()
            cam_items = [f"{item['index']}: {item['label']}" for item in cameras] or ["未找到摄像头"]
            mic_items = [f"{item['index']}: {item['label']}" for item in microphones] or ["未找到麦克风"]
            self.state.pending_camera_items = cam_items
            self.state.pending_mic_items = mic_items
            self.state.devices_loading = False
            self.state.devices_ready = True
            self.state.devices_error = ""
            self.state.device_scan_worker = None
            self.state.signals.devices_state_changed.emit()
            self.state.add_log("设备扫描完成。")
            # 设备到位后，若运行时也已就绪且还没启动摄像头，自动启动第一个
            if cameras and self.state.runtime_ready and self.state.current_camera_index is None:
                self.state.start_camera(cameras[0]["index"])
        except Exception as exc:
            self.state.pending_camera_items = ["未找到摄像头"]
            self.state.pending_mic_items = ["未找到麦克风"]
            self.state.devices_loading = False
            self.state.devices_ready = False
            self.state.devices_error = str(exc)
            self.state.device_scan_worker = None
            self.state.signals.devices_state_changed.emit()
            self.state.add_log(f"设备扫描失败：{exc}")


class AvatarRenderWorker(threading.Thread):
    def __init__(self, state: AppState, fps: int = 12, size: tuple[int, int] = PREVIEW_SIZE) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.fps = fps
        self.size = size
        self.stop_event = threading.Event()
        # 黑帧哨兵：每帧调一次 _watch_black_frame。连续 >3 秒画面偏黑就触发
        # 恢复渲染日志 + 把窗口里的中性 mesh 重新刷出来。包装在 list 里方便
        # 内层 helper 修改外层引用（run 循环里的本地变量）。
        self._last_non_black_time = [time.perf_counter()]
        self._recovery_emitted = [False]

    def _watch_black_frame(self, frame: np.ndarray) -> None:
        """Monitor rendered avatar frames and recover from prolonged black output.

        Both our production harnesses (smoke_concurrent_avatar.py and
        smoke_avatar_real_mediapipe.py) PASS — they emit frames with mean > 60.
        Yet the user reports the avatar stays all-black in production. To make
        the bug self-healing regardless of which production-only layer (VTK
        offscreen, GPU, full PySide6 GUI contention) is the culprit, we track
        when we last emitted a non-black frame. If 3+ seconds elapse with
        everything darker than mean < 5, we *actively* force a camera reset +
        re-render of the neutral mesh and emit the result. The user sees the
        avatar pop back into view instead of staring at a black box.
        """
        try:
            mean_b = float(frame.mean()) if frame.size else 0.0
        except Exception:
            return
        now = time.perf_counter()
        if mean_b >= 5.0:
            self._last_non_black_time[0] = now
            self._recovery_emitted[0] = False
            return
        if now - self._last_non_black_time[0] < 3.0:
            return
        if self._recovery_emitted[0]:
            return
        self._recovery_emitted[0] = True
        self.state.add_log(
            f"[AvatarWorker] 检测到连续黑帧 >3s (mean={mean_b:.1f})，"
            f"强制 ResetCamera + 重渲中性 mesh"
        )
        # 主动恢复：相机重置 + 中性 mesh 重渲。renderer/neutral 由 run() 在
        # 构造成功后写到这里；这一段不能获取 renderer_lock（run 主循环持有），
        # 但 ResetCamera + render_bgr 在我们已经错过 3 秒"画面全黑"的状态下
        # 抢锁 5-10ms 是值得的——用户多等 1 秒也比永远黑屏好。
        renderer = getattr(self, "_renderer", None)
        neutral = getattr(self, "_neutral", None)
        if renderer is None or neutral is None:
            return
        try:
            with self.state.renderer_lock:
                cam = renderer._renderer.GetActiveCamera()
                cam.ParallelProjectionOn()
                cam.SetParallelScale(0.18)
                cam.Azimuth(180)
                cam.Elevation(10)
                renderer._renderer.ResetCamera()
                renderer._renderer.ResetCameraClippingRange()
                renderer._render_window.Render()
                renderer.update_vertices(neutral)
                frame = renderer.render_bgr()
            mean_b2 = float(frame.mean()) if frame.size else 0.0
            self.state.add_log(
                f"[AvatarWorker] 恢复重渲完成 mean={mean_b2:.1f}"
            )
            # 把恢复帧 emit 到 UI —— 这是真正让用户看到头像回来的关键步骤。
            self.state.latest_avatar_frame = frame
            self.state.latest_avatar_seq += 1
            self.state.signals.avatar_frame.emit(frame)
        except Exception as exc:
            self.state.add_log(f"[AvatarWorker] 恢复重渲失败：{exc}")
        self._last_non_black_time[0] = now

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        self.state.add_log(f"[AvatarWorker] 线程启动 (tid={threading.get_ident()})")
        engine = self.state.engine
        if engine is None:
            # runtime 还没好时，主动等最多 60 秒；首启动需要从 Google 拉
            # face_landmarker.task + 加载 MediaPipe + 初始化 GNM head v3，
            # 总耗时 8 秒很容易不够。中间每秒打一次心跳，避免用户在 UI 上
            # 以为头像预览"启动以后就啥都没了"。
            deadline = time.perf_counter() + 60.0
            last_heartbeat = 0.0
            while engine is None and time.perf_counter() < deadline:
                if self.stop_event.is_set():
                    self.state.add_log("[AvatarWorker] stop_event 在等 engine 阶段触发，退出。")
                    return
                engine = self.state.engine
                now = time.perf_counter()
                if engine is None and now - last_heartbeat >= 2.0:
                    self.state.add_log("头像预览等待运行时加载（>2s）...")
                    last_heartbeat = now
                time.sleep(0.1)
            if engine is None:
                self.state.add_log("头像预览放弃：运行时未在 60 秒内就绪。")
                return
        self.state.add_log(
            f"[AvatarWorker] 拿到 engine，开始构造 AvatarPreviewRenderer"
        )
        # VTK 渲染器必须在唯一线程内构造并使用；构造 + evaluate 都用 engine_lock 串起来，
        # 避免和 CameraWorker 同时访问 engine.avatar。
        try:
            with self.state.engine_lock:
                renderer = _pipeline.AvatarPreviewRenderer(engine.avatar, size=self.size)
                try:
                    neutral = engine.avatar.evaluate({}, None)[3]
                except Exception as exc:
                    self.state.add_log(f"头像初始 evaluate 失败：{exc}")
                    neutral = None
            self.state.add_log(
                f"[AvatarWorker] AvatarPreviewRenderer 已构造，neutral.shape={None if neutral is None else neutral.shape}"
            )
        except Exception as exc:
            self.state.add_log(f"头像预览已禁用：{exc}")
            return
        self.state.avatar_renderer = renderer
        # neutral mesh 单独持有，渲染时不需要再 evaluate
        if neutral is None:
            neutral = np.zeros((renderer.avatar.model.template_vertex_positions.shape[0], 3),
                               dtype=np.float32)
        if neutral.shape[0] != renderer.avatar.model.template_vertex_positions.shape[0]:
            neutral = np.zeros(
                (renderer.avatar.model.template_vertex_positions.shape[0], 3),
                dtype=np.float32,
            )
        try:
            # 渲染属于"独占 VTK"动作，串行锁避免和别的线程同时动 VTK 对象。
            # 首帧我们强制做一次 ResetCamera + ResetCameraClippingRange，再 ResetCamera
            # 让 vtk 把 mesh 重新放到 viewport 中央 —— head v3 的 bbox 很小，
            # 没有这一步时偶发会把 mesh 放到 viewport 外面，导致首帧全黑。
            with self.state.renderer_lock:
                try:
                    renderer._renderer.ResetCamera()
                    renderer._renderer.GetActiveCamera().ParallelProjectionOn()
                    renderer._renderer.GetActiveCamera().SetParallelScale(0.18)
                    renderer._renderer.GetActiveCamera().Azimuth(180)
                    renderer._renderer.GetActiveCamera().Elevation(10)
                    renderer._renderer.ResetCameraClippingRange()
                    renderer._render_window.Render()
                except Exception as exc:
                    self.state.add_log(f"头像首帧 ResetCamera 失败：{exc}")
                renderer.update_vertices(neutral)
                frame = renderer.render_bgr()
        except Exception as exc:
            self.state.add_log(f"头像初始帧失败：{exc}")
            frame = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        # 兜底：若首帧仍偏暗，再尝试以 evaluation 结果的 bbox 重新对齐相机。
        # 这是 head v3 这种小 bbox 在某些 VTK 版本上 viewport 错位的最常见原因。
        first_mean = float(frame.mean()) if frame.size else 0.0
        if first_mean < 5.0:
            self.state.add_log(
                f"[AvatarWorker] 首帧过暗 (mean={first_mean:.1f})，强制 ResetCameraClippingRange 重渲"
            )
            try:
                with self.state.renderer_lock:
                    cam = renderer._renderer.GetActiveCamera()
                    cam.ParallelProjectionOn()
                    cam.SetParallelScale(0.18)
                    cam.Azimuth(180)
                    cam.Elevation(10)
                    renderer._renderer.ResetCamera()
                    renderer._renderer.ResetCameraClippingRange()
                    renderer._render_window.Render()
                    renderer.update_vertices(neutral)
                    frame = renderer.render_bgr()
                first_mean = float(frame.mean()) if frame.size else 0.0
                self.state.add_log(
                    f"[AvatarWorker] 强制对齐后重渲 mean={first_mean:.1f}"
                )
            except Exception as exc:
                self.state.add_log(f"[AvatarWorker] 强制对齐失败：{exc}")
        self.state.add_log(
            f"[AvatarWorker] 首帧渲染完成 shape={frame.shape} mean={first_mean:.1f}"
        )
        # 即便首帧还偏暗也照样 emit + 进 render 循环 —— 中性 mesh 路径
        # + watchdog 会在 3 秒内再渲一张 mean>5 的出来，把画面救回。
        self.state.latest_avatar_frame = frame
        self.state.latest_avatar_seq += 1
        self.state.signals.avatar_frame.emit(frame)
        self.state.add_log("头像预览已就绪，等待摄像头驱动...")
        # 把 renderer + neutral 暴露给 watchdog（_watch_black_frame），
        # 让它能在检测到持续黑帧时主动 ResetCamera + 重渲。
        self._renderer = renderer
        self._neutral = neutral

        last_frame_index: int | None = None
        min_interval = 1.0 / max(1, self.fps)
        last_render_time = 0.0
        signals = self.state.signals
        # 摄像头 thread 一启动就会有 packet。packet 没到前，主动自旋短时间，
        # 然后用中性 mesh 先画一帧占位；之后只要 packet 一直不来，就每 ~0.5s
        # 重画一次中性 mesh，让头像不会"消失"。
        spin_deadline = time.perf_counter() + 0.5
        neutral_render_deadline = time.perf_counter() + 0.5
        last_log_tick = 0.0
        try:
            while not self.stop_event.is_set():
                packet = self.state.latest_packet
                if packet is None:
                    now = time.perf_counter()
                    if now < spin_deadline:
                        time.sleep(0.01)
                        continue
                    # 心跳：等 packet 时每 2 秒打一条，避免用户看到黑屏误以为挂死
                    if now - last_log_tick >= 2.0:
                        self.state.add_log(
                            f"[AvatarWorker] 等待 latest_packet（engine_lock 排队中？），"
                            f"latest_avatar_seq={self.state.latest_avatar_seq}"
                        )
                        last_log_tick = now
                    # 兜底：保持最新一次的中性 mesh（仅第一次记录日志）
                    if neutral_render_deadline > 0:
                        self.state.add_log("头像预览等待摄像头数据（>0.5s），先用中性 mesh 渲染。")
                        neutral_render_deadline = 0.0
                    try:
                        with self.state.renderer_lock:
                            renderer.update_vertices(neutral)
                            frame = renderer.render_bgr()
                    except Exception as exc:
                        self.state.add_log(f"头像中性 mesh 渲染失败：{exc}")
                        frame = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
                    self._watch_black_frame(frame)
                    self.state.latest_avatar_frame = frame
                    self.state.latest_avatar_seq += 1
                    signals.avatar_frame.emit(frame)
                    last_frame_index = -1
                    time.sleep(0.05)
                    continue
                elif packet.frame_index == last_frame_index:
                    time.sleep(0.02)
                    continue
                else:
                    now = time.perf_counter()
                    if now - last_render_time < min_interval:
                        time.sleep(0.005)
                        continue
                    # 记录"已经驱动过一帧"——从此 chip 切到"已驱动"，
                    # 之前的中性 mesh 占位阶段会显示"等待摄像头..."。
                    self.state._avatar_first_packet_seen = True
                    try:
                        # 拷贝一份 vertices，避免 CameraWorker 后续 evaluate 把同一段
                        # 内存里的数据覆盖掉。renderer.update_vertices 内部会 deep-copy
                        # 到 vtk，但和原 numpy 数组之间存在微弱的竞态窗口。
                        verts = packet.vertices
                        safe_verts = verts.copy() if verts is not None else None
                        if safe_verts is None or safe_verts.size == 0:
                            raise ValueError("vertices 为空，跳过渲染")
                        t0 = time.perf_counter()
                        with self.state.renderer_lock:
                            renderer.update_vertices(safe_verts)
                            frame = renderer.render_bgr()
                        render_ms = (time.perf_counter() - t0) * 1000.0
                        # 仅第一次 packet 渲染打一条"成功建立"日志，方便诊断黑屏
                        if last_frame_index is None:
                            self.state.add_log(
                                f"[AvatarWorker] 首个 packet 渲染 OK "
                                f"shape={frame.shape} mean={frame.mean():.1f} "
                                f"render={render_ms:.1f}ms face_detected={packet.face_detected}"
                            )
                    except Exception as exc:
                        self.state.add_log(f"头像预览渲染失败：{exc}")
                        frame = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
                    self._watch_black_frame(frame)
                    self.state.latest_avatar_frame = frame
                    self.state.latest_avatar_seq += 1
                    signals.avatar_frame.emit(frame)
                    last_frame_index = packet.frame_index
                    last_render_time = now
        finally:
            avatar_worker = getattr(self.state, "avatar_worker", None)
            if avatar_worker is self:
                self.state.avatar_worker = None


class ExportWorker(threading.Thread):
    def __init__(self, state: AppState) -> None:
        super().__init__(daemon=True)
        self.state = state

    def run(self) -> None:
        try:
            recorder = self.state.recorder
            if recorder is None:
                return
            self.state.set_export_progress(0.0, "正在合并音视频...")
            outputs = recorder.finalize(transcriber=self.state.transcriber)
            self.state.set_last_outputs(outputs)
            self.state.set_export_progress(1.0, "导出完成。")
            self.state.add_log(f"已保存会话：{outputs.session_dir}")
            if outputs.merged_video_path is not None:
                self.state.add_log(f"视频：{outputs.merged_video_path}")
            else:
                self.state.add_log("视频：未生成合并音视频文件。")
            if outputs.timeline_csv_path is not None:
                self.state.add_log(f"时间轴 CSV：{outputs.timeline_csv_path}")
            if outputs.transcript_csv_path is not None:
                self.state.add_log(f"转写 CSV：{outputs.transcript_csv_path}")
        except Exception as exc:
            self.state.add_log(f"导出失败：{exc}")
            self.state.set_export_progress(0.0, f"导出失败：{exc}")
        finally:
            self.state.export_worker = None
            self.state.recorder = None
            self.state.transcriber = None
            self.state.recording = False
            self.state.exporting = False


# ---------------------------------------------------------------------------
# 自定义控件
# ---------------------------------------------------------------------------

class Card(QFrame):
    """圆角卡片容器，带轻阴影。"""

    def __init__(self, parent: QWidget | None = None, *, alt: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("cardAlt" if alt else "card")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(0, 0, 0, 22))
        self.setGraphicsEffect(shadow)


class ChipLabel(QLabel):
    """状态徽章 - 圆角胶囊。"""

    def __init__(self, text: str = "-", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("chip")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(24)
        self.setMaximumHeight(24)
        self.setMinimumWidth(0)
        self._set_color(QColor(COLOR_TEXT_MUTED), QColor("#FFFFFF"))

    def set_state(self, state: str) -> None:
        palette = {
            "ok": (QColor(COLOR_PRIMARY), QColor(COLOR_PRIMARY_SOFT)),
            "recording": (QColor(COLOR_RECORD), QColor(COLOR_RECORD_SOFT)),
            "warn": (QColor(COLOR_WARN), QColor(COLOR_WARN_SOFT)),
            "info": (QColor(COLOR_INFO), QColor(COLOR_INFO_SOFT)),
            "muted": (QColor(COLOR_TEXT_MUTED), QColor("#EFEAE0")),
        }
        fg, bg = palette.get(state, palette["muted"])
        self._set_color(fg, bg)

    def _set_color(self, fg: QColor, bg: QColor) -> None:
        # 直接给当前 label 设置样式，避开 objectName 选择器在某些 Qt 风格下解析失败的问题
        self.setStyleSheet(
            f"color: {fg.name()}; background-color: {bg.name()};"
            " border-radius: 12px; padding: 2px 10px; font-size: 12px;"
            " border: 1px solid transparent;"
        )


class PreviewLabel(QLabel):
    """显示摄像头/头像预览，带 REC 角标。"""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._recording = False
        self._show_placeholder = True
        self._placeholder_text = "等待画面..."
        self.setMinimumSize(320, 180)
        # 16:9 强制布局：摄像头和头像的源 frame 都是 640x360，控件宽度变化时按
        # 16:9 等比例伸缩，避免出现非 16:9 的窄高 / 扁宽控件让画面被压扁。
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background-color: #1c1d22; color: #cbd5e1; border-radius: 10px;"
        )
        self._base_pixmap: QPixmap | None = None

    def hasHeightForWidth(self) -> bool:  # noqa: D401
        return True

    def heightForWidth(self, w: int) -> int:  # noqa: D401
        # 保持 16:9 比例
        return int(round(w * 9.0 / 16.0))

    def sizeHint(self):  # noqa: D401
        return QSize(640, 360)

    def set_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            self._show_placeholder = True
            self._base_pixmap = None
            self.update()
            return
        try:
            img = frame_to_qimage(frame, (PREVIEW_SIZE[0], PREVIEW_SIZE[1]))
            self._base_pixmap = QPixmap.fromImage(img)
            # 全黑帧（mean < 2）说明渲染线程还没真正出图（例如 avatar 渲染超时
            # / 失败兜底时，AvatarRenderWorker 塞了个 np.zeros）。此时继续显示
            # 上一帧或占位文本，否则看到的就是一块黑屏。
            mean_brightness = float(frame.mean()) if frame.size else 0.0
            self._show_placeholder = mean_brightness < 2.0
        except Exception:
            self._show_placeholder = True
            self._base_pixmap = None
        self.update()

    def set_placeholder(self, text: str) -> None:
        self._placeholder_text = text
        self._show_placeholder = True
        self._base_pixmap = None
        self.update()

    def set_recording(self, recording: bool) -> None:
        self._recording = recording
        self.update()

    def paintEvent(self, event) -> None:  # noqa: D401
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, QColor("#1c1d22"))
        if self._base_pixmap is not None and not self._show_placeholder:
            # 用 KeepAspectRatioByExpanding，让 16:9 的源画面等比放大到刚好能
            # 撑满整个控件的最小尺寸，长边多出来的部分在画的时候被 clip 掉。
            # 完全不修改像素纵横比 —— 16:9 camera frame 在 16:9 控件里 1:1 显示，
            # 在 16:10 / 4:3 控件里也保持 16:9 不被压扁。
            scaled = self._base_pixmap.scaled(
                rect.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation,
            )
            x_off = (rect.width() - scaled.width()) // 2
            y_off = (rect.height() - scaled.height()) // 2
            painter.drawPixmap(rect.x() + x_off, rect.y() + y_off, scaled)
        else:
            painter.setPen(QColor("#cbd5e1"))
            font = painter.font()
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignCenter, self._placeholder_text)
        # 标题条
        title_rect = QRect(rect.x() + 12, rect.y() + 12, 180, 24)
        painter.setBrush(QColor(0, 0, 0, 140))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(title_rect, 8, 8)
        painter.setPen(QColor("#FFFFFF"))
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(title_rect, Qt.AlignCenter, self._title)
        # REC 角标
        if self._recording:
            rec_rect = QRect(rect.right() - 96, rect.y() + 12, 84, 24)
            painter.setBrush(QColor(COLOR_RECORD))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rec_rect, 8, 8)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(rec_rect, Qt.AlignCenter, "● REC")
        painter.end()


from PySide6.QtCore import QRect  # noqa: E402


class LevelMeter(QWidget):
    """麦克风电平条。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._level = 0.0
        self._display = 0.0
        self.setMinimumHeight(28)
        self.setMinimumWidth(180)
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    @Slot(float)
    def set_level(self, level: float) -> None:
        self._level = max(0.0, min(1.0, level))

    def _tick(self) -> None:
        # 平滑过渡
        self._display += (self._level - self._display) * 0.35
        self.update()

    def paintEvent(self, event) -> None:  # noqa: D401
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        # 背景
        painter.setBrush(QColor("#EFEAE0"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 6, 6)
        # 电平
        bar_rect = rect.adjusted(2, 2, -2, -2)
        width = int(bar_rect.width() * self._display)
        if width > 0:
            seg_rect = QRect(bar_rect.x(), bar_rect.y(), width, bar_rect.height())
            # 渐变色：绿→橙→红
            grad = QLinearGradient_(bar_rect.left(), 0, bar_rect.right(), 0)
            grad.setColorAt(0.0, QColor(COLOR_PRIMARY))
            grad.setColorAt(0.7, QColor("#E0A03B"))
            grad.setColorAt(1.0, QColor(COLOR_RECORD))
            painter.setBrush(grad)
            painter.drawRoundedRect(seg_rect, 5, 5)
        # dB 数字
        painter.setPen(QColor(COLOR_TEXT))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        db = level_to_db(self._display)
        text = f"{db:+.0f} dB"
        painter.drawText(rect.adjusted(8, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
        painter.end()


from PySide6.QtGui import QLinearGradient as QLinearGradient_  # noqa: E402


class BlendShapeBar(QWidget):
    """单个 blendshape 的进度条。"""

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._name = name
        self._value = 0.0
        self.setMinimumHeight(22)
        self.setMinimumWidth(180)

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(1.0, float(value)))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: D401
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        name = self._name
        val = self._value
        painter.setPen(QColor(COLOR_TEXT))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        name_rect = QRect(rect.x(), rect.y(), int(rect.width() * 0.55), rect.height())
        painter.drawText(name_rect, Qt.AlignVCenter | Qt.AlignLeft, name)
        bar_rect = QRect(
            name_rect.right() + 8, rect.y() + 6, rect.width() - name_rect.width() - 70, rect.height() - 12,
        )
        painter.setBrush(QColor("#F0E9DA"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(bar_rect, 4, 4)
        if val > 0:
            seg_rect = QRect(bar_rect.x(), bar_rect.y(), int(bar_rect.width() * val), bar_rect.height())
            painter.setBrush(QColor(COLOR_PRIMARY))
            painter.drawRoundedRect(seg_rect, 4, 4)
        val_rect = QRect(bar_rect.right() + 6, rect.y(), 56, rect.height())
        painter.setPen(QColor(COLOR_TEXT_MUTED))
        painter.drawText(val_rect, Qt.AlignVCenter | Qt.AlignRight, f"{val:.2f}")
        painter.end()


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------

class FaceAvatarWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.state = AppState()
        self._build_ui()
        self._wire_signals()

        # 30ms ≈ 33fps
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(33)
        self._ui_timer.timeout.connect(self._on_ui_tick)
        self._ui_timer.start()

        # 启动后台任务
        self.state.ensure_device_scan_async()
        self.state.ensure_runtime_async()

    # ----------------------- UI 构建 -----------------------

    def _build_ui(self) -> None:
        self.setWindowTitle("Face Avatar Studio")
        self.resize(1600, 980)
        self.setMinimumSize(1280, 820)

        # 全局 stylesheet
        self.setStyleSheet(self._global_stylesheet())

        central = QWidget()
        central.setObjectName("central")
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(central)

        # 顶部 header
        root_layout.addWidget(self._build_header())

        # 中部：左 sidebar + 右 main
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(20, 16, 20, 16)
        body_layout.setSpacing(16)
        root_layout.addWidget(body, 1)

        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setObjectName("sidebarScroll")
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setFixedWidth(380)
        self.sidebar_scroll.setFrameShape(QFrame.NoFrame)
        sidebar_inner = QWidget()
        sb_layout = QVBoxLayout(sidebar_inner)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_layout.setSpacing(12)
        sb_layout.addWidget(self._build_device_card())
        sb_layout.addWidget(self._build_record_card())
        sb_layout.addWidget(self._build_status_card())
        sb_layout.addWidget(self._build_log_card(), 1)
        sb_layout.addStretch(1)
        self.sidebar_scroll.setWidget(sidebar_inner)
        body_layout.addWidget(self.sidebar_scroll)

        main_panel = QVBoxLayout()
        main_panel.setSpacing(12)
        main_panel.addWidget(self._build_preview_row())
        main_panel.addWidget(self._build_info_row(), 1)
        body_layout.addLayout(main_panel, 1)

        # 底部状态栏
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self._status_progress = QProgressBar()
        self._status_progress.setMaximumWidth(220)
        self._status_progress.setRange(0, 100)
        self._status_progress.setVisible(False)
        self.statusbar.addPermanentWidget(self._status_progress)
        self._status_message = QLabel("就绪")
        self.statusbar.addWidget(self._status_message, 1)

        # 菜单
        menu = self.menuBar()
        file_menu = menu.addMenu("文件")
        self.action_open_output = QAction("打开输出目录", self)
        self.action_open_output.triggered.connect(self._on_open_output)
        file_menu.addAction(self.action_open_output)
        self.action_open_latest = QAction("打开最近会话", self)
        self.action_open_latest.triggered.connect(self._on_open_session)
        file_menu.addAction(self.action_open_latest)
        file_menu.addSeparator()
        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        help_menu = menu.addMenu("帮助")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _global_stylesheet(self) -> str:
        return f"""
        QMainWindow, #central, QWidget {{
            background-color: {COLOR_BG};
            color: {COLOR_TEXT};
            font-family: 'Microsoft YaHei UI', 'Segoe UI', 'PingFang SC', 'Noto Sans CJK SC';
            font-size: 12pt;
        }}
        #card, #cardAlt {{
            background-color: {COLOR_CARD};
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
        }}
        #cardAlt {{
            background-color: {COLOR_CARD_ALT};
        }}
        QLabel#title {{
            font-size: 18pt; font-weight: 600; color: {COLOR_TEXT};
        }}
        QLabel#subtitle {{
            color: {COLOR_TEXT_MUTED}; font-size: 10pt;
        }}
        QLabel#cardTitle {{
            font-size: 11pt; font-weight: 600; color: {COLOR_TEXT};
        }}
        QLabel#sectionTitle {{
            font-size: 13pt; font-weight: 600; color: {COLOR_TEXT};
        }}
        QLabel#hint {{
            color: {COLOR_TEXT_MUTED}; font-size: 9pt;
        }}
        QLineEdit, QComboBox {{
            background-color: {COLOR_CARD_ALT};
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            padding: 6px 10px;
            color: {COLOR_TEXT};
            selection-background-color: {COLOR_PRIMARY_SOFT};
            min-height: 22px;
        }}
        QLineEdit:focus, QComboBox:focus {{
            border-color: {COLOR_PRIMARY};
        }}
        QComboBox::drop-down {{
            border: none; width: 22px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {COLOR_CARD};
            border: 1px solid {COLOR_BORDER};
            selection-background-color: {COLOR_PRIMARY_SOFT};
            selection-color: {COLOR_TEXT};
        }}
        QPushButton {{
            background-color: {COLOR_CARD};
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            padding: 7px 14px;
            color: {COLOR_TEXT};
            min-height: 22px;
        }}
        QPushButton:hover {{
            background-color: #FBF6EA;
            border-color: #D7CDB6;
        }}
        QPushButton:pressed {{
            background-color: #F1E9D5;
        }}
        QPushButton:disabled {{
            color: {COLOR_TEXT_DISABLED};
            background-color: {COLOR_CARD_ALT};
            border-color: {COLOR_BORDER_SOFT};
        }}
        QPushButton#primaryButton {{
            background-color: {COLOR_PRIMARY};
            color: white;
            border: 1px solid {COLOR_PRIMARY_DARK};
        }}
        QPushButton#primaryButton:hover {{
            background-color: #27957B;
        }}
        QPushButton#primaryButton:pressed {{
            background-color: {COLOR_PRIMARY_DARK};
        }}
        QPushButton#dangerButton {{
            background-color: {COLOR_RECORD};
            color: white;
            border: 1px solid #A6442E;
        }}
        QPushButton#dangerButton:hover {{
            background-color: #D86953;
        }}
        QPlainTextEdit, QTextEdit {{
            background-color: {COLOR_CARD_ALT};
            border: 1px solid {COLOR_BORDER_SOFT};
            border-radius: 8px;
            color: {COLOR_TEXT};
            selection-background-color: {COLOR_PRIMARY_SOFT};
        }}
        QStatusBar {{
            background-color: {COLOR_CARD};
            border-top: 1px solid {COLOR_BORDER};
            color: {COLOR_TEXT_MUTED};
        }}
        QStatusBar QLabel {{
            color: {COLOR_TEXT_MUTED};
        }}
        QProgressBar {{
            background-color: {COLOR_CARD_ALT};
            border: 1px solid {COLOR_BORDER};
            border-radius: 6px;
            text-align: center;
            color: {COLOR_TEXT};
            min-height: 16px;
        }}
        QProgressBar::chunk {{
            background-color: {COLOR_PRIMARY};
            border-radius: 5px;
        }}
        QScrollArea#sidebarScroll {{
            background: transparent;
            border: none;
        }}
        QScrollArea#sidebarScroll > QWidget > QWidget {{
            background: transparent;
        }}
        QMenuBar {{
            background-color: {COLOR_CARD};
            border-bottom: 1px solid {COLOR_BORDER};
            padding: 4px 8px;
        }}
        QMenuBar::item:selected {{
            background-color: {COLOR_PRIMARY_SOFT};
            border-radius: 6px;
        }}
        QMenu {{
            background-color: {COLOR_CARD};
            border: 1px solid {COLOR_BORDER};
        }}
        QMenu::item:selected {{
            background-color: {COLOR_PRIMARY_SOFT};
        }}
        QHeaderView::section {{
            background-color: {COLOR_CARD_ALT};
            border: none;
            border-bottom: 1px solid {COLOR_BORDER};
            padding: 6px 10px;
        }}
        """

    # ----------------------- 头部 -----------------------

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("cardAlt")
        # 直接给当前 widget 设样式，不再依赖 #cardAlt 选择器
        header.setStyleSheet(
            f"background-color: {COLOR_CARD};"
            f" border-bottom: 1px solid {COLOR_BORDER};"
            " border-top-left-radius: 0px; border-top-right-radius: 0px;"
            " border-bottom-left-radius: 0px; border-bottom-right-radius: 0px;"
        )
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 14, 24, 14)
        layout.setSpacing(20)
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("脸部头像工作室")
        title.setObjectName("title")
        subtitle = QLabel("本地桌面采集 · 头像驱动 · 录制转写一站式")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box)
        layout.addStretch(1)
        info_box = QVBoxLayout()
        info_box.setSpacing(2)
        for text, color in [
            ("桌面软件 / 中文界面", COLOR_TEXT_MUTED),
            ("摄像头 + 头像实时驱动", COLOR_PRIMARY_DARK),
            ("CSV · 视频 · 音频一体化导出", COLOR_INFO),
        ]:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {color}; font-size: 10pt;")
            info_box.addWidget(lbl)
        layout.addLayout(info_box)
        return header

    # ----------------------- 侧栏卡片 -----------------------

    def _build_device_card(self) -> QWidget:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        title = QLabel("设备")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.camera_combo = QComboBox()
        self.camera_combo.setObjectName("cameraCombo")
        self.camera_combo.addItems(self.state.pending_camera_items)
        self.mic_combo = QComboBox()
        self.mic_combo.addItems(self.state.pending_mic_items)
        self.perf_combo = QComboBox()
        self.perf_combo.addItems(list(PERFORMANCE_PROFILES.keys()))
        self.perf_combo.setCurrentText(self.state.performance_profile_name)

        layout.addWidget(self._labeled("摄像头", self.camera_combo))
        layout.addWidget(self._labeled("麦克风", self.mic_combo))
        layout.addWidget(self._labeled("性能档位", self.perf_combo))

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.apply_camera_btn = QPushButton("应用摄像头")
        self.apply_camera_btn.setObjectName("primaryButton")
        self.apply_camera_btn.clicked.connect(self._on_apply_camera)
        self.reload_btn = QPushButton("刷新设备")
        self.reload_btn.clicked.connect(self._on_reload_devices)
        self.restart_btn = QPushButton("重连摄像头")
        self.restart_btn.clicked.connect(self._on_restart_camera)
        btn_row.addWidget(self.apply_camera_btn)
        btn_row.addWidget(self.reload_btn)
        btn_row.addWidget(self.restart_btn)
        layout.addLayout(btn_row)
        return card

    def _build_record_card(self) -> QWidget:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        title = QLabel("录制")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.output_edit = QLineEdit(str(self.state.output_root))
        self.output_edit.setReadOnly(True)
        output_row = QHBoxLayout()
        output_row.setSpacing(8)
        self.browse_btn = QPushButton("浏览")
        self.browse_btn.clicked.connect(self._on_browse_output)
        self.open_output_btn = QPushButton("打开")
        self.open_output_btn.clicked.connect(self._on_open_output)
        output_row.addWidget(self.browse_btn)
        output_row.addWidget(self.open_output_btn)
        layout.addWidget(self._labeled("保存到", self.output_edit))
        layout.addLayout(output_row)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(list(SPEECH_LANGUAGE_OPTIONS.keys()))
        self.lang_combo.setCurrentText("自动")
        self.stt_combo = QComboBox()
        self.stt_combo.addItems(["tiny", "base", "small", "medium"])
        self.stt_combo.setCurrentText(DEFAULT_STT_MODEL)

        layout.addWidget(self._labeled("语音语言", self.lang_combo))
        layout.addWidget(self._labeled("Whisper 模型", self.stt_combo))

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.start_btn = QPushButton("开始录制")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.clicked.connect(self._on_start_recording)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.clicked.connect(self._on_stop_recording)
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        self.record_hint = QLabel("准备好后点击「开始录制」")
        self.record_hint.setObjectName("hint")
        layout.addWidget(self.record_hint)
        return card

    def _build_status_card(self) -> QWidget:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)
        title = QLabel("状态")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.chip_runtime = ChipLabel("运行时：加载中")
        self.chip_runtime.set_state("warn")
        self.chip_perf = ChipLabel("档位：-")
        self.chip_record = ChipLabel("空闲")
        self.chip_record.set_state("muted")
        self.chip_camera = ChipLabel("摄像头：-")
        self.chip_mic = ChipLabel("麦克风：-")
        self.chip_face = ChipLabel("人脸：-")
        self.chip_face.set_state("ok")
        self.chip_session = ChipLabel("会话：-")
        self.chip_output = ChipLabel("输出：-")

        for chip in (
            self.chip_runtime,
            self.chip_perf,
            self.chip_record,
            self.chip_camera,
            self.chip_mic,
            self.chip_face,
            self.chip_session,
            self.chip_output,
        ):
            chip.setMinimumWidth(0)
            chip.setMaximumHeight(28)
            chip.setMinimumHeight(28)
            layout.addWidget(chip)
        layout.addSpacing(4)
        return card

    def _build_log_card(self) -> QWidget:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)
        header_row = QHBoxLayout()
        title = QLabel("日志")
        title.setObjectName("sectionTitle")
        header_row.addWidget(title)
        header_row.addStretch(1)
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self._on_clear_log)
        header_row.addWidget(clear_btn)
        layout.addLayout(header_row)

        self.log_filter = QLineEdit()
        self.log_filter.setPlaceholderText("筛选日志...")
        self.log_filter.textChanged.connect(self._on_filter_log)
        layout.addWidget(self.log_filter)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(400)
        font = QFont("Consolas", 9)
        self.log_box.setFont(font)
        layout.addWidget(self.log_box, 1)
        return card

    # ----------------------- 主面板 -----------------------

    def _build_preview_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(self._build_preview_card("摄像头", is_camera=True), 1)
        layout.addWidget(self._build_preview_card("头像驱动", is_camera=False), 1)
        return row

    def _build_preview_card(self, title: str, *, is_camera: bool) -> QWidget:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        header = QHBoxLayout()
        t = QLabel(title)
        t.setObjectName("sectionTitle")
        header.addWidget(t)
        header.addStretch(1)
        info_chip = ChipLabel("等待画面")
        info_chip.set_state("warn")
        if is_camera:
            self.camera_chip = info_chip
        else:
            self.avatar_chip = info_chip
        header.addWidget(info_chip)
        layout.addLayout(header)

        preview = PreviewLabel(title)
        if is_camera:
            self.camera_preview = preview
            preview.set_placeholder("等待摄像头...")
        else:
            self.avatar_preview = preview
            preview.set_placeholder("等待头像驱动...")
        layout.addWidget(preview, 1)
        return card

    def _build_info_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(self._build_frame_card(), 1)
        layout.addWidget(self._build_output_card(), 1)
        layout.addWidget(self._build_audio_card())
        return row

    def _build_frame_card(self) -> QWidget:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)
        title = QLabel("当前帧")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.frame_info = QLabel("帧号：-")
        self.time_info = QLabel("时间：-")
        self.state_info = QLabel("状态：就绪")
        self.fps_info = QLabel("STT 模型：small | 语言：自动")
        for lbl in (self.frame_info, self.time_info, self.state_info, self.fps_info):
            lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 10pt;")
            layout.addWidget(lbl)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {COLOR_BORDER_SOFT};")
        layout.addWidget(sep)

        bs_title = QLabel("主要表情参数")
        bs_title.setObjectName("cardTitle")
        layout.addWidget(bs_title)

        self.blendshape_container = QWidget()
        self.blendshape_layout = QVBoxLayout(self.blendshape_container)
        self.blendshape_layout.setContentsMargins(0, 0, 0, 0)
        self.blendshape_layout.setSpacing(4)
        self._blendshape_widgets: list[BlendShapeBar] = []
        for _ in range(8):
            bar = BlendShapeBar("-")
            self.blendshape_layout.addWidget(bar)
            self._blendshape_widgets.append(bar)
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.blendshape_container)
        scroll.setStyleSheet("background: transparent;")
        layout.addWidget(scroll, 1)
        return card

    def _build_output_card(self) -> QWidget:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)
        title = QLabel("导出文件")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.files_box = QPlainTextEdit()
        self.files_box.setReadOnly(True)
        self.files_box.setMaximumBlockCount(20)
        font = QFont("Consolas", 9)
        self.files_box.setFont(font)
        self.files_box.setPlainText("还没有导出会话。")
        layout.addWidget(self.files_box, 1)

        btn_grid = QGridLayout()
        btn_grid.setSpacing(8)
        self.btn_open_session = QPushButton("打开会话")
        self.btn_open_session.clicked.connect(self._on_open_session)
        self.btn_open_video = QPushButton("打开视频")
        self.btn_open_video.clicked.connect(self._on_open_video)
        self.btn_open_audio = QPushButton("打开音频")
        self.btn_open_audio.clicked.connect(self._on_open_audio)
        self.btn_open_timeline = QPushButton("打开时间轴")
        self.btn_open_timeline.clicked.connect(self._on_open_timeline)
        self.btn_open_transcript = QPushButton("打开转写")
        self.btn_open_transcript.clicked.connect(self._on_open_transcript)
        self.btn_open_manifest = QPushButton("打开清单")
        self.btn_open_manifest.clicked.connect(self._on_open_manifest)
        for i, btn in enumerate([
            self.btn_open_session, self.btn_open_video, self.btn_open_audio,
            self.btn_open_timeline, self.btn_open_transcript, self.btn_open_manifest,
        ]):
            btn.setEnabled(False)
            btn_grid.addWidget(btn, i // 2, i % 2)
        layout.addLayout(btn_grid)
        self._file_buttons = [
            self.btn_open_session, self.btn_open_video, self.btn_open_audio,
            self.btn_open_timeline, self.btn_open_transcript, self.btn_open_manifest,
        ]
        return card

    def _build_audio_card(self) -> QWidget:
        card = Card()
        card.setMinimumWidth(260)
        card.setMaximumWidth(320)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)
        title = QLabel("麦克风")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.audio_status = QLabel("未录制")
        self.audio_status.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 10pt;")
        layout.addWidget(self.audio_status)

        self.level_meter = LevelMeter()
        layout.addWidget(self.level_meter)

        self.record_timer_label = QLabel("00:00")
        self.record_timer_label.setAlignment(Qt.AlignCenter)
        font = self.record_timer_label.font()
        font.setPointSize(20)
        font.setBold(True)
        self.record_timer_label.setFont(font)
        self.record_timer_label.setStyleSheet(f"color: {COLOR_RECORD};")
        layout.addWidget(self.record_timer_label)

        self.export_progress_label = QLabel("")
        self.export_progress_label.setWordWrap(True)
        self.export_progress_label.setStyleSheet(f"color: {COLOR_INFO}; font-size: 9pt;")
        layout.addWidget(self.export_progress_label)

        layout.addStretch(1)
        return card

    # ----------------------- 辅助 -----------------------

    def _labeled(self, label_text: str, widget: QWidget) -> QWidget:
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        lbl = QLabel(label_text)
        lbl.setObjectName("cardTitle")
        v.addWidget(lbl)
        v.addWidget(widget)
        return wrap

    # ----------------------- 信号绑定 -----------------------

    def _wire_signals(self) -> None:
        s = self.state.signals
        s.log_added.connect(self._on_log_added)
        s.camera_frame.connect(self._on_camera_frame)
        s.avatar_frame.connect(self._on_avatar_frame)
        s.outputs_changed.connect(self._on_outputs_changed)
        s.runtime_state_changed.connect(self._on_runtime_state_changed)
        s.devices_state_changed.connect(self._on_devices_state_changed)
        s.audio_level.connect(self.level_meter.set_level)

    # ----------------------- Slots -----------------------

    @Slot(str)
    def _on_log_added(self, line: str) -> None:
        # textChanged filter 处理显示
        self._refresh_log_view()

    @Slot(object)
    def _on_camera_frame(self, frame: np.ndarray) -> None:
        self.camera_preview.set_frame(frame)
        self.camera_chip.set_state("ok")
        self.camera_chip.setText("实时画面")

    @Slot(object)
    def _on_avatar_frame(self, frame: np.ndarray) -> None:
        try:
            mean_b = float(frame.mean()) if frame.size else 0.0
            self.state.add_log(
                f"[UI] _on_avatar_frame 收到 frame shape={frame.shape} mean={mean_b:.1f}"
            )
        except Exception:
            mean_b = 0.0
        # 区分"驱动中（拿到一帧）"和"还在 warm-up（中性 mesh 占位）"。
        # warm-up 期间 CameraWorker 还没送出第一帧，PreviewLabel 上看到的
        # 是 AvatarRenderWorker 在 0.5s 内拿中性 mesh 先画的占位图，那张图
        # 通常 mean 不为 0 但也不算"动"起来。把 chip 文案做成阶段提示，避免
        # 用户看到全黑误判头像驱动坏了。
        if mean_b < 5.0:
            self.avatar_preview.set_frame(frame)
            self.avatar_chip.set_state("warn")
            self.avatar_chip.setText("渲染异常")
        elif getattr(self.state, "_avatar_first_packet_seen", False):
            self.avatar_preview.set_frame(frame)
            self.avatar_chip.set_state("ok")
            self.avatar_chip.setText("已驱动")
        else:
            self.avatar_preview.set_frame(frame)
            self.avatar_chip.set_state("warn")
            self.avatar_chip.setText("等待摄像头...")

    @Slot(object)
    def _on_outputs_changed(self, outputs: _pipeline.RecordingOutputs) -> None:
        for btn in self._file_buttons:
            btn.setEnabled(True)
        lines = []
        if outputs.session_dir:
            lines.append(f"会话：{outputs.session_dir}")
        if outputs.video_path:
            lines.append(f"视频：{outputs.video_path}")
        if outputs.audio_path:
            lines.append(f"音频：{outputs.audio_path}")
        if outputs.merged_video_path:
            lines.append(f"合并：{outputs.merged_video_path}")
        if outputs.timeline_csv_path:
            lines.append(f"时间轴：{outputs.timeline_csv_path}")
        if outputs.transcript_csv_path:
            lines.append(f"转写：{outputs.transcript_csv_path}")
        if outputs.manifest_path:
            lines.append(f"清单：{outputs.manifest_path}")
        self.files_box.setPlainText("\n".join(lines) if lines else "未生成任何文件。")
        self.action_open_latest.setEnabled(True)

    @Slot()
    def _on_runtime_state_changed(self) -> None:
        if self.state.runtime_error:
            self.chip_runtime.set_state("warn")
            self.chip_runtime.setText(f"运行时：{self.state.runtime_error}")
        elif self.state.runtime_ready:
            self.chip_runtime.set_state("ok")
            self.chip_runtime.setText("运行时：已就绪")
        elif self.state.runtime_loading:
            self.chip_runtime.set_state("warn")
            self.chip_runtime.setText("运行时：加载中...")
        else:
            self.chip_runtime.set_state("muted")
            self.chip_runtime.setText("运行时：未启动")

    @Slot()
    def _on_devices_state_changed(self) -> None:
        cam_items = self.state.pending_camera_items
        mic_items = self.state.pending_mic_items
        self.camera_combo.clear()
        self.camera_combo.addItems(cam_items)
        self.mic_combo.clear()
        self.mic_combo.addItems(mic_items)
        if cam_items:
            self.camera_combo.setCurrentIndex(0)
        if mic_items:
            self.mic_combo.setCurrentIndex(0)
        self._refresh_log_view()

    # ----------------------- 按钮回调 -----------------------

    def _on_apply_camera(self) -> None:
        idx = self._selected_camera()
        if idx < 0:
            self.state.add_log("未选择摄像头。")
            return
        profile = self.perf_combo.currentText()
        self.state.request_camera_start(idx, profile)

    def _on_reload_devices(self) -> None:
        self.state.devices_loading = False
        self.state.devices_ready = False
        self.state.ensure_device_scan_async()

    def _on_restart_camera(self) -> None:
        idx = self._selected_camera()
        if idx < 0:
            # 选中摄像头失败时，回退到当前已启动的摄像头
            idx = self.state.current_camera_index if self.state.current_camera_index is not None else 0
        self.state.add_log(f"手动重连摄像头 {idx} ...")
        # 先强制清掉当前 worker，避免和现有线程争抢设备
        self.state._stop_stream_workers()
        self.state.latest_camera_frame = None
        self.state.latest_avatar_frame = None
        self.camera_preview.set_placeholder("正在重连摄像头...")
        self.avatar_preview.set_placeholder("等待头像驱动...")
        self.state.request_camera_start(idx, self.perf_combo.currentText())

    def _on_browse_output(self) -> None:
        selected = choose_directory(self, self.output_edit.text())
        if selected:
            self.output_edit.setText(selected)
            self.state.output_root = ensure_directory(Path(selected))

    def _on_open_output(self) -> None:
        path = self.output_edit.text()
        if not os.path.exists(path):
            self.state.add_log(f"输出目录不存在：{path}")
            return
        open_path(path)

    def _on_start_recording(self) -> None:
        if not self.state.runtime_ready:
            QMessageBox.warning(self, "提示", "运行时还没准备好，请稍候。")
            return
        cam_idx = self._selected_camera()
        mic_idx, mic_name = self._selected_microphone()
        if cam_idx < 0:
            QMessageBox.warning(self, "提示", "请先选择摄像头。")
            return
        if mic_idx is None:
            QMessageBox.warning(self, "提示", "请先选择麦克风。")
            return
        try:
            self.state.set_performance_profile(self.perf_combo.currentText())
            self.state.start_recording(
                output_root=self.output_edit.text(),
                mic_index=mic_idx,
                mic_name=mic_name,
                language=self._selected_language(),
                stt_model=self.stt_combo.currentText(),
            )
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.record_hint.setText("录制中...")
            self.audio_status.setText("正在录音")
            self.chip_record.set_state("recording")
            self.chip_record.setText("录制中")
            self.camera_preview.set_recording(True)
        except Exception as exc:
            self.state.add_log(f"无法开始录制：{exc}")
            QMessageBox.critical(self, "错误", f"无法开始录制：{exc}")

    def _on_stop_recording(self) -> None:
        try:
            self.state.stop_recording()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.audio_status.setText("正在导出")
            self.chip_record.set_state("info")
            self.chip_record.setText("导出中")
            self.record_hint.setText("正在合并音视频 + 转写...")
            self.camera_preview.set_recording(False)
            self._status_progress.setVisible(True)
            self._status_progress.setValue(0)
        except Exception as exc:
            self.state.add_log(f"停止录制失败：{exc}")
            QMessageBox.critical(self, "错误", f"停止录制失败：{exc}")

    def _on_open_session(self) -> None:
        outputs = self.state.last_outputs
        if outputs is not None:
            open_path(outputs.session_dir)

    def _on_open_video(self) -> None:
        outputs = self.state.last_outputs
        if outputs is not None and outputs.merged_video_path is not None:
            open_path(outputs.merged_video_path)
        elif outputs is not None and outputs.video_path is not None:
            open_path(outputs.video_path)

    def _on_open_audio(self) -> None:
        outputs = self.state.last_outputs
        if outputs is not None and outputs.audio_path is not None:
            open_path(outputs.audio_path)

    def _on_open_timeline(self) -> None:
        outputs = self.state.last_outputs
        if outputs is not None and outputs.timeline_csv_path is not None:
            open_path(outputs.timeline_csv_path)

    def _on_open_transcript(self) -> None:
        outputs = self.state.last_outputs
        if outputs is not None and outputs.transcript_csv_path is not None:
            open_path(outputs.transcript_csv_path)

    def _on_open_manifest(self) -> None:
        outputs = self.state.last_outputs
        if outputs is not None and outputs.manifest_path is not None:
            open_path(outputs.manifest_path)

    def _on_clear_log(self) -> None:
        self.state.logs.clear()
        self.log_box.clear()

    def _on_filter_log(self, _text: str) -> None:
        self._refresh_log_view()

    def _refresh_log_view(self) -> None:
        keyword = self.log_filter.text().strip().lower()
        if keyword:
            lines = [ln for ln in self.state.logs if keyword in ln.lower()]
        else:
            lines = list(self.state.logs)
        self.log_box.setPlainText("\n".join(lines))

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "关于",
            "<b>Face Avatar Studio</b><br>"
            "本地桌面采集 / 头像驱动 / 录制转写<br><br>"
            "依赖：mediapipe · faster-whisper · vtk · PySide6",
        )

    # ----------------------- 选中辅助 -----------------------

    def _selected_camera(self) -> int:
        value = self.camera_combo.currentText()
        try:
            return int(value.split(":", 1)[0].strip())
        except Exception:
            return -1

    def _selected_microphone(self) -> tuple[int | None, str]:
        value = self.mic_combo.currentText()
        try:
            return int(value.split(":", 1)[0].strip()), value
        except Exception:
            return None, value

    def _selected_language(self) -> str:
        return SPEECH_LANGUAGE_OPTIONS.get(self.lang_combo.currentText(), "auto")

    # ----------------------- 主循环 -----------------------

    def _on_ui_tick(self) -> None:
        s = self.state.signals
        runtime_ready = self.state.runtime_ready
        recording = self.state.recording
        exporting = self.state.exporting

        # 录制时长
        if recording and self.state.recorder is not None:
            secs = self.state.recorder.elapsed_seconds
            self.record_timer_label.setText(format_duration(secs))
            self.record_timer_label.setStyleSheet(f"color: {COLOR_RECORD};")
        elif exporting:
            self.record_timer_label.setText("导出")
            self.record_timer_label.setStyleSheet(f"color: {COLOR_INFO};")
        else:
            self.record_timer_label.setText("00:00")
            self.record_timer_label.setStyleSheet(f"color: {COLOR_TEXT_DISABLED};")

        # blendshape
        packet = self.state.latest_packet
        if packet is not None and packet.blendshapes:
            top = sorted(packet.blendshapes.items(), key=lambda x: x[1], reverse=True)[:8]
            for bar, (name, value) in zip(self._blendshape_widgets, top):
                bar._name = name
                bar.set_value(value)
                bar.update()

        # frame info
        if packet is not None:
            self.frame_info.setText(f"帧号：{packet.frame_index}")
            self.time_info.setText(f"时间：{packet.timestamp_sec:.2f}s")
            self.state_info.setText(
                "状态：录制" if recording else "状态：导出" if exporting else "状态：就绪"
            )
            self.fps_info.setText(
                f"STT 模型：{self.state.current_stt_model} | 语言：{SPEECH_LANGUAGE_LABELS.get(self.state.current_speech_language, self.state.current_speech_language)}"
            )
            if packet.face_detected:
                self.chip_face.set_state("ok")
                self.chip_face.setText("人脸：已检测到")
            else:
                self.chip_face.set_state("warn")
                self.chip_face.setText("人脸：未检测到")

        # 状态徽章
        self.chip_perf.setText(
            f"档位：{self.state.performance_profile_name} | 采集 {self.state.performance_settings()['capture_size'][0]}x{self.state.performance_settings()['capture_size'][1]}"
        )
        cam_idx = self.state.current_camera_index
        self.chip_camera.setText(f"摄像头：{cam_idx if cam_idx is not None else '-'}")
        self.chip_mic.setText(f"麦克风：{self.state.current_mic_name or '-'}")
        session_dir = self.state.recorder.session_dir if self.state.recorder is not None else None
        self.chip_session.setText(f"会话：{session_dir or '-'}")
        self.chip_output.setText(f"输出：{self.state.output_root}")

        # 导出进度
        ratio, msg = self.state.export_progress
        if exporting:
            self._status_progress.setVisible(True)
            self._status_progress.setValue(int(ratio * 100))
            self.export_progress_label.setText(msg or "导出中...")
            self._status_message.setText(msg or "导出中...")
        else:
            if self._status_progress.isVisible():
                self._status_progress.setVisible(False)
            self._status_message.setText("就绪")
            self.export_progress_label.setText("")

        # 按钮可用性
        busy = recording or exporting
        self.start_btn.setEnabled(runtime_ready and not busy)
        self.stop_btn.setEnabled(busy)
        self.apply_camera_btn.setEnabled(runtime_ready and not busy)
        self.restart_btn.setEnabled(runtime_ready and not busy)
        self.camera_combo.setEnabled(not busy)
        self.mic_combo.setEnabled(not busy)
        self.perf_combo.setEnabled(not busy)
        self.lang_combo.setEnabled(not busy)
        self.stt_combo.setEnabled(not busy)
        self.browse_btn.setEnabled(not busy)
        self.reload_btn.setEnabled(not busy)

        # 菜单可用性
        self.action_open_latest.setEnabled(self.state.last_outputs is not None)

    # ----------------------- 退出 -----------------------

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: D401
        if self.state.recording or self.state.exporting:
            choice = QMessageBox.question(
                self,
                "确认退出",
                "录制或导出仍在进行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                event.ignore()
                return
        self._shutdown()
        event.accept()

    def _shutdown(self) -> None:
        try:
            self._ui_timer.stop()
        except Exception:
            pass
        try:
            self.state._stop_stream_workers()
        except Exception:
            pass
        try:
            if self.state.recorder is not None and self.state.recorder.active:
                self.state.recorder.stop_capture()
        except Exception:
            pass
        try:
            if self.state.export_worker is not None:
                self.state.export_worker.join(timeout=2)
        except Exception:
            pass
        try:
            if self.state.runtime_worker is not None:
                self.state.runtime_worker.join(timeout=2)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def run_desktop_app(argv: list[str] | None = None) -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Face Avatar Studio")
    app.setOrganizationName("FaceAvatarStudio")
    # 暖色调
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLOR_BG))
    palette.setColor(QPalette.WindowText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.Base, QColor(COLOR_CARD_ALT))
    palette.setColor(QPalette.AlternateBase, QColor(COLOR_CARD))
    palette.setColor(QPalette.Text, QColor(COLOR_TEXT))
    palette.setColor(QPalette.Button, QColor(COLOR_CARD))
    palette.setColor(QPalette.ButtonText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.Highlight, QColor(COLOR_PRIMARY))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)

    window = FaceAvatarWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_desktop_app(sys.argv))
