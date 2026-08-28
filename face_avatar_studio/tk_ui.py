from __future__ import annotations

import os
import multiprocessing
import queue
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .pipeline import (
    FACEVERSE_DATA_DIR,
    FACEVERSE_V2_DATA_DIR,
    FaceTrackingEngine,
    FramePacket,
    RealtimeVisemeAnalyzer,
    RecordingOutputs,
    SessionRecorder,
    WhisperTranscriber,
    _draw_landmarks,
    create_tracking_engine,
    ensure_directory,
    faceverse_backend_status,
    faceverse_v2_backend_status,
    prewarm_face_landmarker,
)

CAPTURE_SIZE = (960, 540)
CAMERA_SIZE = (640, 480)
CAPTURE_FPS = 30
TRACKING_FPS = 18

BACKEND_LABELS = {
    "mediapipe": "MediaPipe",
    "faceverse": "FaceVerse v4",
    "faceverse_v2": "FaceVerse V2（52维）",
}


def _backend_display_name(backend: str) -> str:
    return BACKEND_LABELS.get(str(backend), str(backend))


def _sensitive_level(linear_level: float, sensitivity: float = 1.35) -> float:
    """Convert a small linear microphone level into a useful dBFS meter."""
    dbfs = 20.0 * np.log10(max(float(linear_level), 1e-7))
    return float(np.clip(((dbfs + 58.0) / 42.0) * sensitivity, 0.0, 1.0))


def enumerate_cameras(max_index: int = 5) -> list[tuple[int, str]]:
    found = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if cap.isOpened() and cap.read()[0]:
                found.append((index, f"摄像头 {index}"))
        finally:
            cap.release()
    return found


def enumerate_microphones() -> list[tuple[int, str]]:
    import sounddevice as sd
    return [(i, str(d["name"])) for i, d in enumerate(sd.query_devices()) if d.get("max_input_channels", 0) > 0]


def _avatar_render_process_main(requests, responses) -> None:
    """Render GNM in a child process so driver stalls cannot freeze Tk."""
    renderer = None
    try:
        from .pipeline import GnmAvatarDriver

        avatar = GnmAvatarDriver()
        try:
            renderer = OpenGLAvatarRenderer(avatar.model)
        except Exception as exc:
            renderer = AvatarRasterizer(avatar.model)
            responses.put(("ready", -1, f"软件渲染：{exc}", 0.0))
        else:
            responses.put(("ready", -1, "GPU", 0.0))
        while True:
            request = requests.get()
            if request is None:
                return
            frame_index, vertices, blendshapes = request
            started = time.perf_counter()
            frame = renderer.render(vertices, blendshapes)
            responses.put(
                ("frame", frame_index, frame, time.perf_counter() - started)
            )
    except Exception as exc:
        try:
            responses.put(("error", -1, repr(exc), 0.0))
        except Exception:
            pass
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass


class AudioMonitor:
    """Non-recording microphone monitor used while the camera preview is open."""

    def __init__(self) -> None:
        self.stream = None
        self.level = 0.0
        self.error: str | None = None
        self.lock = threading.Lock()
        self.sensitivity = 1.35
        self._envelope = 0.0
        self.viseme_analyzer = RealtimeVisemeAnalyzer(16000)

    @property
    def active(self) -> bool:
        return self.stream is not None

    def start(self, device_index: int) -> None:
        import sounddevice as sd

        self.stop()
        self.error = None

        def callback(indata, _frames, _time_info, status) -> None:
            if status:
                self.error = str(status)
            samples = np.asarray(indata, dtype=np.float32)
            rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
            peak = float(np.max(np.abs(samples))) if samples.size else 0.0
            # A linear meter makes normal speech look almost silent. Map the
            # microphone through dBFS, retain a little peak sensitivity for
            # consonants, and use fast attack / slower release for a stable
            # speech envelope.
            rms_level = _sensitive_level(rms, 1.0)
            target = min(1.0, max(rms_level, peak * 1.8) * self.sensitivity)
            self.viseme_analyzer.process(samples)
            with self.lock:
                alpha = 0.72 if target > self._envelope else 0.18
                self._envelope += alpha * (target - self._envelope)
                self.level = self._envelope

        self.stream = sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=16000,
            blocksize=512,
            dtype="float32",
            callback=callback,
        )
        self.stream.start()

    def latest_level(self) -> float:
        with self.lock:
            return float(self.level)

    def set_sensitivity(self, value: float) -> None:
        self.sensitivity = float(np.clip(value, 0.7, 2.5))

    def latest_visemes(self) -> dict[str, float]:
        return self.viseme_analyzer.values()

    def stop(self) -> None:
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        with self.lock:
            self.level = 0.0
            self._envelope = 0.0
        self.viseme_analyzer.reset()


class CaptureWorker(threading.Thread):
    """The capture and inference work stays outside Tk's event loop."""
    def __init__(self, camera_index: int, tracking_backend: str = "mediapipe") -> None:
        super().__init__(daemon=True, name="face-avatar-camera")
        self.camera_index = camera_index
        self.tracking_backend = tracking_backend
        self.stop_event = threading.Event()
        self.status_queue: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.packet: FramePacket | None = None
        self.recorder: SessionRecorder | None = None
        self.engine: FaceTrackingEngine | None = None
        self.tracking_thread: threading.Thread | None = None
        self.render_thread: threading.Thread | None = None
        self.avatar_frame: np.ndarray | None = None
        self.avatar_render_fps = 0.0
        self.capture_fps = 0.0
        self.audio_level = 0.0
        self.audio_visemes: dict[str, float] = {}
        self.started = 0.0

    def stop(self) -> None:
        self.stop_event.set()

    def set_recorder(self, recorder: SessionRecorder | None) -> None:
        with self.lock:
            self.recorder = recorder

    def set_audio_features(
        self, level: float, visemes: dict[str, float] | None = None
    ) -> None:
        with self.lock:
            self.audio_level = float(np.clip(level, 0.0, 1.0))
            self.audio_visemes = dict(visemes or {})

    def latest_frame(self) -> np.ndarray | None:
        with self.lock:
            return self.frame

    def latest_packet(self) -> FramePacket | None:
        with self.lock:
            return self.packet

    def latest_avatar_frame(self) -> np.ndarray | None:
        with self.lock:
            return self.avatar_frame

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.status_queue.put("无法打开摄像头，请检查设备是否被占用。")
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # 640x480 is supported by more Windows camera drivers at 30 FPS than
        # the uncommon 960x540 mode. Frames remain native and unstretched.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.started = time.perf_counter()
        self.status_queue.put("摄像头已打开，正在后台加载表情模型...")
        self.tracking_thread = threading.Thread(
            target=self._tracking_loop, daemon=True, name="face-avatar-tracking"
        )
        self.tracking_thread.start()
        self.render_thread = threading.Thread(
            target=self._render_loop, daemon=True, name="face-avatar-render"
        )
        self.render_thread.start()
        capture_count = 0
        capture_clock = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                ok, raw = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                # Keep the camera's native aspect ratio. Some webcams ignore
                # the requested 16:9 mode and return 4:3; stretching that
                # frame makes MediaPipe's face geometry unstable.
                with self.lock:
                    self.frame = raw.copy()
                capture_count += 1
                now = time.perf_counter()
                if now - capture_clock >= 1.0:
                    measured = capture_count / max(now - capture_clock, 1e-6)
                    self.capture_fps = measured if self.capture_fps == 0 else (
                        0.35 * measured + 0.65 * self.capture_fps
                    )
                    capture_count = 0
                    capture_clock = now
        except Exception as exc:
            self.status_queue.put(f"摄像头线程异常: {exc}")
        finally:
            self.stop_event.set()
            cap.release()
            self.status_queue.put("camera_stopped")

    def _tracking_loop(self) -> None:
        """Heavy GNM/MediaPipe work; capture continues while this warms up."""
        try:
            self.engine = create_tracking_engine(self.tracking_backend)
            backend_name = _backend_display_name(self.tracking_backend)
            self.status_queue.put(f"{backend_name} 表情追踪已就绪。")
            source_label = getattr(self.engine.avatar, "source_label", "")
            if source_label:
                self.status_queue.put(f"头像模型：{source_label}")
        except Exception as exc:
            self.status_queue.put(f"表情模型加载失败: {exc}")
            return
        # GNM is ready before MediaPipe's first landmarker call. Publish a
        # neutral packet immediately so the avatar is visible while the
        # heavier MediaPipe import/model warm-up continues in this worker.
        with self.lock:
            first_frame = None if self.frame is None else self.frame.copy()
        if first_frame is not None:
            try:
                expression, rotations, translation, vertices = (
                    self.engine.neutral_state()
                )
                neutral_packet = FramePacket(
                    frame_index=-2,
                    timestamp_sec=0.0,
                    frame_bgr=first_frame,
                    landmarks=None,
                    blendshapes={},
                    expression=expression,
                    rotations=rotations,
                    translation=translation,
                    face_matrix=None,
                    vertices=vertices,
                    face_detected=False,
                )
                with self.lock:
                    self.packet = neutral_packet
                self.status_queue.put("中性头像已显示，正在后台预热人脸识别...")
            except Exception as exc:
                self.status_queue.put(f"GNM 中性头像初始化失败: {exc}")
        try:
            # Spawn the MediaPipe owner after publishing the neutral head.
            # Camera capture and avatar rendering remain responsive while the
            # child imports TFLite and performs its one-time XNNPACK warm-up.
            self.engine._ensure_landmarker()
        except Exception as exc:
            self.status_queue.put(f"人脸识别预热失败: {exc}")
        index = 0
        while not self.stop_event.is_set():
            with self.lock:
                frame = None if self.frame is None else self.frame.copy()
                audio_level = self.audio_level
                audio_visemes = dict(self.audio_visemes)
            if frame is None:
                time.sleep(0.01)
                continue
            started = time.perf_counter()
            try:
                packet = self.engine.process_frame(
                    frame,
                    index,
                    started - self.started,
                    audio_level=audio_level,
                    audio_visemes=audio_visemes,
                )
            except Exception as exc:
                self.status_queue.put(f"表情追踪异常: {exc}")
                time.sleep(0.1)
                continue
            index += 1
            with self.lock:
                self.packet = packet
                recorder = self.recorder
            if recorder is not None and recorder.active:
                recorder.append_packet(packet)
            remaining = 1.0 / TRACKING_FPS - (time.perf_counter() - started)
            if remaining > 0:
                self.stop_event.wait(remaining)

    def _render_loop(self) -> None:
        """Render dense GNM frames without ever blocking camera or inference."""
        if self.tracking_backend == "mediapipe":
            self._render_process_loop()
            return
        renderer = None
        last_index = -1
        while not self.stop_event.is_set():
            engine = self.engine
            packet = self.latest_packet()
            if engine is None or packet is None:
                self.stop_event.wait(0.03)
                continue
            if renderer is None:
                try:
                    renderer = OpenGLAvatarRenderer(engine.avatar.model)
                    self.status_queue.put("GPU 参数化头像渲染已启用。")
                except Exception as gpu_exc:
                    renderer = AvatarRasterizer(engine.avatar.model)
                    self.status_queue.put(f"GPU 渲染不可用，已切换软件渲染: {gpu_exc}")
            if packet.frame_index == last_index:
                self.stop_event.wait(0.02)
                continue
            last_index = packet.frame_index
            try:
                render_started = time.perf_counter()
                frame = renderer.render(packet.vertices, packet.blendshapes)
                render_elapsed = max(time.perf_counter() - render_started, 1e-6)
                current_fps = 1.0 / render_elapsed
                self.avatar_render_fps = current_fps if self.avatar_render_fps == 0 else (0.2 * current_fps + 0.8 * self.avatar_render_fps)
                with self.lock:
                    self.avatar_frame = frame
            except Exception as exc:
                self.status_queue.put(f"头像渲染异常: {exc}")
                try:
                    renderer.close()
                except Exception:
                    pass
                renderer = AvatarRasterizer(engine.avatar.model)
                self.status_queue.put("已切换到软件头像渲染。")
                self.stop_event.wait(0.2)
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass

    def _render_process_loop(self) -> None:
        """Keep OpenGL imports, context stalls and readback outside Tk's process."""
        context = multiprocessing.get_context("spawn")
        requests = context.Queue(maxsize=1)
        responses = context.Queue(maxsize=2)
        process = context.Process(
            target=_avatar_render_process_main,
            args=(requests, responses),
            daemon=True,
            name="face-avatar-opengl",
        )
        process.start()
        ready = False
        last_sent_index = -1
        try:
            while not self.stop_event.is_set():
                try:
                    kind, frame_index, payload, elapsed = responses.get(timeout=0.02)
                    if kind == "ready":
                        ready = True
                        self.status_queue.put("独立 GPU 头像渲染进程已启用。")
                    elif kind == "frame":
                        current_fps = 1.0 / max(float(elapsed), 1e-6)
                        self.avatar_render_fps = (
                            current_fps
                            if self.avatar_render_fps == 0
                            else 0.2 * current_fps + 0.8 * self.avatar_render_fps
                        )
                        with self.lock:
                            self.avatar_frame = payload
                    elif kind == "error":
                        self.status_queue.put(f"GPU 头像渲染进程异常: {payload}")
                        break
                except queue.Empty:
                    pass
                if not process.is_alive():
                    self.status_queue.put("GPU 头像渲染进程已停止。")
                    break
                if not ready:
                    continue
                packet = self.latest_packet()
                if packet is None or packet.frame_index == last_sent_index:
                    continue
                try:
                    requests.put_nowait(
                        (packet.frame_index, packet.vertices, packet.blendshapes)
                    )
                    last_sent_index = packet.frame_index
                except queue.Full:
                    # The renderer is still consuming an older request. Keep
                    # the newest packet available in CaptureWorker and retry.
                    pass
        finally:
            try:
                requests.put_nowait(None)
            except Exception:
                pass
            process.join(timeout=1.5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)


class ExportWorker(threading.Thread):
    def __init__(self, recorder: SessionRecorder, transcriber: WhisperTranscriber) -> None:
        super().__init__(daemon=True, name="face-avatar-export")
        self.recorder, self.transcriber = recorder, transcriber
        self.result_queue: queue.Queue[RecordingOutputs] = queue.Queue(maxsize=1)
        self.error_queue: queue.Queue[str] = queue.Queue(maxsize=1)

    def run(self) -> None:
        try:
            self.result_queue.put(self.recorder.finalize(self.transcriber))
        except Exception as exc:
            self.error_queue.put(str(exc))


class AvatarRasterizer:
    """Dense painter-style renderer for the complete upstream GNM head mesh."""
    def __init__(self, model) -> None:
        self.model = model
        self._is_faceverse = callable(getattr(model, "render_vertices", None))
        is_gnm = hasattr(model, "quads") and hasattr(model, "quads_group")
        all_faces = np.asarray(
            model.quads if is_gnm else model.triangles, dtype=np.int32
        )
        eye_exteriors = set()
        if is_gnm:
            eye_exteriors = {
                tuple(face)
                for face in np.asarray(
                    model.quads_group("eye_exteriors"), dtype=np.int32
                )
            }
        # GNM carries an opaque eyeball exterior used by some render pipelines.
        # Google's own mesh viewer excludes it; otherwise it sits in front of
        # the sclera/iris/pupil interior and makes both eyes look blank.
        self.faces = np.asarray(
            [face for face in all_faces if tuple(face) not in eye_exteriors],
            dtype=np.int32,
        )
        neutral = self._display_vertices(model.template_vertex_positions)
        x0, x1 = np.percentile(neutral[:, 0], [1, 99])
        y0, y1 = np.percentile(neutral[:, 1], [1, 99])
        self.view_center = np.asarray(((x0 + x1) * 0.5, (y0 + y1) * 0.5), dtype=np.float32)
        self.view_scale = min(468 / max(x1 - x0, 1e-6), 388 / max(y1 - y0, 1e-6))
        if hasattr(model, "vertex_colors"):
            vertex_colors = np.asarray(model.vertex_colors, dtype=np.float32)
            self.colors = vertex_colors[self.faces].mean(axis=1)[:, ::-1]
        else:
            self.colors = np.tile(
                np.array([198, 126, 64], dtype=np.float32),
                (len(self.faces), 1),
            )
        lookup = {tuple(row): index for index, row in enumerate(self.faces.tolist())}
        self.dental_faces = np.zeros(len(self.faces), dtype=bool)
        self.interior_faces = np.zeros(len(self.faces), dtype=bool)
        material_colors = {
            "scleras": (240, 240, 238), "irises": (55, 105, 150),
            "pupils": (18, 18, 20), "teeth": (220, 225, 229),
            "tongue": (104, 76, 158), "gums": (96, 76, 172),
            "mouth_sock": (38, 30, 42),
        }
        if is_gnm:
            for group, color in material_colors.items():
                for face in np.asarray(model.quads_group(group), dtype=np.int32):
                    index = lookup.get(tuple(face))
                    if index is not None:
                        self.colors[index] = color
                        if group in ("teeth", "gums"):
                            self.dental_faces[index] = True
                        elif group in ("tongue", "mouth_sock"):
                            self.interior_faces[index] = True

    def _display_vertices(self, vertices: np.ndarray) -> np.ndarray:
        if self._is_faceverse:
            return self.model.render_vertices(vertices)
        return np.asarray(vertices, dtype=np.float32)

    def render(self, vertices: np.ndarray, blendshapes: dict[str, float] | None = None) -> np.ndarray:
        verts = self._display_vertices(vertices)
        points = np.empty((len(verts), 2), dtype=np.int32)
        points[:, 0] = (320 + (verts[:, 0] - self.view_center[0]) * self.view_scale).astype(np.int32)
        points[:, 1] = (252 - (verts[:, 1] - self.view_center[1]) * self.view_scale).astype(np.int32)
        face_verts = verts[self.faces]
        normals = np.cross(face_verts[:, 1] - face_verts[:, 0], face_verts[:, 2] - face_verts[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-7)
        light = np.array([-.35, .45, .82], dtype=np.float32)
        colors = np.clip(self.colors * (.42 + .58 * np.abs(normals @ light))[:, None], 0, 255).astype(np.uint8)
        image = np.full((480, 640, 3), (30, 31, 36), dtype=np.uint8)
        order = np.argsort(face_verts[:, :, 2].mean(axis=1))
        for index in order:
            cv2.fillConvexPoly(image, points[self.faces[index]], tuple(int(v) for v in colors[index]), cv2.LINE_AA)
        if getattr(self.model, "is_faceverse_v2", False):
            renderer_label = "FACEVERSE V2 / 52D LIVE"
        elif self._is_faceverse:
            renderer_label = "FACEVERSE V4 / LIVE EXPRESSION"
        else:
            renderer_label = "GNM HEAD / LIVE EXPRESSION"
        cv2.putText(image, renderer_label, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, .55, (213, 220, 232), 1, cv2.LINE_AA)
        return image


class OpenGLAvatarRenderer:
    """Hidden SDL/OpenGL renderer isolated inside the avatar worker thread."""
    def __init__(self, model) -> None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        from OpenGL import GL

        self.pygame, self.gl, self.model = pygame, GL, model
        self._is_faceverse = callable(getattr(model, "render_vertices", None))
        pygame.init()
        pygame.display.set_mode((640, 480), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
        neutral = self._display_vertices(model.template_vertex_positions)
        x0, x1 = np.percentile(neutral[:, 0], [1, 99])
        y0, y1 = np.percentile(neutral[:, 1], [1, 99])
        self.view_center = np.asarray(((x0 + x1) * 0.5, (y0 + y1) * 0.5), dtype=np.float32)
        self.view_scale = 1.62 / max(y1 - y0, (x1 - x0) / 1.25, 1e-6)
        self.is_gnm = hasattr(model, "quads") and hasattr(model, "quads_group")
        self.faces = np.ascontiguousarray(
            model.quads if self.is_gnm else model.triangles, dtype=np.uint32
        )
        self.face_primitive = GL.GL_QUADS if self.is_gnm else GL.GL_TRIANGLES
        face_lookup = {tuple(face): index for index, face in enumerate(self.faces.tolist())}
        dental_mask = np.zeros(len(self.faces), dtype=bool)
        interior_mask = np.zeros(len(self.faces), dtype=bool)
        eye_exterior_mask = np.zeros(len(self.faces), dtype=bool)
        if self.is_gnm:
            for face in np.asarray(
                model.quads_group("eye_exteriors"), dtype=np.uint32
            ):
                index = face_lookup.get(tuple(face))
                if index is not None:
                    eye_exterior_mask[index] = True
            for group in ("teeth", "gums"):
                for face in np.asarray(model.quads_group(group), dtype=np.uint32):
                    index = face_lookup.get(tuple(face))
                    if index is not None:
                        dental_mask[index] = True
            for group in ("tongue", "mouth_sock"):
                for face in np.asarray(model.quads_group(group), dtype=np.uint32):
                    index = face_lookup.get(tuple(face))
                    if index is not None:
                        interior_mask[index] = True
        self.base_faces = np.ascontiguousarray(
            self.faces[~(dental_mask | interior_mask | eye_exterior_mask)].ravel(),
            dtype=np.uint32,
        )
        self.dental_faces = np.ascontiguousarray(
            self.faces[dental_mask].ravel(), dtype=np.uint32
        )
        self.interior_faces = np.ascontiguousarray(
            self.faces[interior_mask].ravel(), dtype=np.uint32
        )
        if hasattr(model, "vertex_colors"):
            self.vertex_colors = np.asarray(model.vertex_colors, dtype=np.uint8).copy()
        else:
            self.vertex_colors = np.tile(
                np.array([64, 126, 198], dtype=np.uint8),
                (len(model.template_vertex_positions), 1),
            )
        group_colors = {
            "scleras": (238, 240, 240), "irises": (150, 105, 55),
            "pupils": (20, 18, 18), "teeth": (229, 225, 220),
            "tongue": (158, 76, 104), "gums": (172, 76, 96),
            "mouth_sock": (42, 30, 38),
        }
        if self.is_gnm:
            for group, color in group_colors.items():
                self.vertex_colors[
                    np.asarray(model.vertex_group_indices(group), dtype=np.int32)
                ] = color
        self.vertex_colors = np.ascontiguousarray(self.vertex_colors)
        GL.glViewport(0, 0, 640, 480)
        GL.glClearColor(0.118, 0.122, 0.141, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [-1.2, 1.4, 3.0, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, [0.28, 0.28, 0.30, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [0.88, 0.88, 0.92, 1.0])
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-1.333, 1.333, -1.0, 1.0, 0.1, 10.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def render(self, vertices: np.ndarray, blendshapes: dict[str, float] | None = None) -> np.ndarray:
        GL = self.gl
        verts = self._display_vertices(vertices)
        transformed = np.ascontiguousarray(verts.copy(), dtype=np.float32)
        transformed[:, 0] = (verts[:, 0] - self.view_center[0]) * self.view_scale
        transformed[:, 1] = (verts[:, 1] - self.view_center[1]) * self.view_scale
        transformed[:, 2] = (verts[:, 2] - np.median(verts[:, 2])) * self.view_scale
        # Normals must be derived after the same display-space transform as
        # vertices; otherwise FaceVerse's two-axis rotation lights the head
        # from the wrong side and can make the upright face look inverted.
        normals = np.ascontiguousarray(
            self.model.compute_vertex_normals(transformed), dtype=np.float32
        )
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        GL.glTranslatef(0.0, 0.0, -3.0)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, transformed)
        GL.glNormalPointer(GL.GL_FLOAT, 0, normals)
        GL.glColorPointer(3, GL.GL_UNSIGNED_BYTE, 0, self.vertex_colors)
        GL.glDrawElements(
            self.face_primitive,
            self.base_faces.size,
            GL.GL_UNSIGNED_INT,
            self.base_faces,
        )
        if self.interior_faces.size:
            GL.glDrawElements(
                self.face_primitive,
                self.interior_faces.size,
                GL.GL_UNSIGNED_INT,
                self.interior_faces,
            )
        if self.dental_faces.size:
            GL.glDrawElements(
                self.face_primitive,
                self.dental_faces.size,
                GL.GL_UNSIGNED_INT,
                self.dental_faces,
            )
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        # glReadPixels below already synchronizes the readback. An explicit
        # glFinish stalls the render worker twice on several Windows drivers.
        pixels = GL.glReadPixels(0, 0, 640, 480, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(480, 640, 3)
        return cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)

    def _display_vertices(self, vertices: np.ndarray) -> np.ndarray:
        if self._is_faceverse:
            return self.model.render_vertices(vertices)
        return np.asarray(vertices, dtype=np.float32)

    def close(self) -> None:
        self.pygame.display.quit()


class AvatarPreview(ttk.Frame):
    """Software projection of the live GNM vertices, with no VTK/OpenGL path."""
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.label = tk.Label(self, bg="#111827", fg="#d1d5db")
        self.label.pack(fill="both", expand=True)
        self.photo: ImageTk.PhotoImage | None = None
        self.show_message("等待表情追踪模型")

    def show_message(self, text: str) -> None:
        self.photo = None
        self.label.configure(image="", text=text)

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        source_h, source_w = frame_bgr.shape[:2]
        target_w = max(320, self.label.winfo_width())
        target_h = max(240, self.label.winfo_height())
        scale = min(target_w / max(source_w, 1), target_h / max(source_h, 1), 1.0)
        if scale < 0.995:
            frame_bgr = cv2.resize(
                frame_bgr,
                (max(1, round(source_w * scale)), max(1, round(source_h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.label.configure(image=self.photo, text="")

class FaceAvatarStudioApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Face Avatar Studio")
        self.root.geometry("1360x820")
        self.root.minsize(1100, 700)
        self.worker: CaptureWorker | None = None
        self.recorder: SessionRecorder | None = None
        self.exporter: ExportWorker | None = None
        self.transcriber: WhisperTranscriber | None = None
        self.audio_monitor = AudioMonitor()
        self.stopping_worker: CaptureWorker | None = None
        self.last_packet_index = -1
        self.last_avatar_frame: np.ndarray | None = None
        self.last_avatar_preview_time = 0.0
        self.last_tracking_wall = time.perf_counter()
        self.last_tracking_index = 0
        self.tracking_clock_started = False
        self.last_preview_time = 0.0
        self.last_diagnostic_time = 0.0
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.output_root = ensure_directory(Path.home() / "FaceAvatarStudio")
        self._build_ui()
        self._load_mics()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(150, self._prewarm_tracking_model)
        self.root.after(33, self.poll)

    def _prewarm_tracking_model(self) -> None:
        def warmup() -> None:
            try:
                prewarm_face_landmarker()
            except Exception as exc:
                try:
                    self.root.after(
                        0,
                        lambda message=str(exc): self.log(
                            f"人脸模型预热失败: {message}"
                        ),
                    )
                except tk.TclError:
                    pass

        threading.Thread(
            target=warmup,
            daemon=True,
            name="face-avatar-model-prewarm",
        ).start()

    def _build_ui(self) -> None:
        s = ttk.Style(self.root)
        try: s.theme_use("clam")
        except tk.TclError: pass
        tkfont.nametofont("TkDefaultFont").configure(family="Microsoft YaHei UI", size=10)
        title_font = tkfont.Font(self.root, family="Microsoft YaHei UI", size=19, weight="bold")
        heading_font = tkfont.Font(self.root, family="Microsoft YaHei UI", size=12, weight="bold")
        s.configure("TFrame", background="#f5f7fa"); s.configure("Panel.TFrame", background="#ffffff")
        s.configure("TLabel", background="#f5f7fa", foreground="#1f2937")
        s.configure("Panel.TLabel", background="#ffffff", foreground="#1f2937")
        s.configure("Title.TLabel", background="#f5f7fa", font=title_font, foreground="#111827")
        s.configure("Heading.TLabel", background="#ffffff", font=heading_font, foreground="#1f2937")
        s.configure("Sub.TLabel", background="#f5f7fa", foreground="#64748b")
        s.configure("Accent.TButton", background="#1677ff", foreground="#ffffff", padding=(14, 9))
        s.map("Accent.TButton", background=[("active", "#0958d9"), ("disabled", "#b9d6ff")])
        s.configure("TButton", padding=(10, 7))
        shell = ttk.Frame(self.root, padding=18); shell.pack(fill="both", expand=True)
        shell.columnconfigure(1, weight=1); shell.rowconfigure(1, weight=1)
        head = ttk.Frame(shell); head.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        ttk.Label(head, text="Face Avatar Studio", style="Title.TLabel").pack(anchor="w")
        self.status = tk.StringVar(value="就绪：选择摄像头后打开实时预览")
        ttk.Label(head, textvariable=self.status, style="Sub.TLabel").pack(anchor="w", pady=(3, 0))
        side = ttk.Frame(shell, style="Panel.TFrame", padding=16); side.grid(row=1, column=0, sticky="nsw", padx=(0, 14))
        main = ttk.Frame(shell); main.grid(row=1, column=1, sticky="nsew"); main.columnconfigure((0, 1), weight=1); main.rowconfigure(1, weight=1)
        ttk.Label(side, text="采集设置", style="Heading.TLabel").pack(anchor="w", pady=(0, 10))
        self.camera = self.field(side, "摄像头", ["0: 默认摄像头"])
        self.tracking_backend = self.field(
            side,
            "表情追踪后端",
            ["MediaPipe（默认）", "FaceVerse v4", "FaceVerse V2（52维）"],
        )
        self.tracking_backend.bind("<<ComboboxSelected>>", self._backend_changed)
        self.mic = self.field(side, "麦克风", ["正在读取设备..."])
        self.language = self.field(side, "语音识别", ["自动识别（中文 / English）", "中文", "English"]); self.language.current(0)
        ttk.Button(side, text="刷新设备列表", command=self.reload_devices).pack(fill="x", pady=(2, 12))
        ttk.Label(side, text="保存位置", style="Panel.TLabel").pack(anchor="w")
        save = ttk.Frame(side, style="Panel.TFrame"); save.pack(fill="x", pady=(5, 12)); save.columnconfigure(0, weight=1)
        self.output = ttk.Entry(save); self.output.insert(0, str(self.output_root)); self.output.grid(row=0, column=0, sticky="ew")
        ttk.Button(save, text="选择", command=self.choose_directory).grid(row=0, column=1, padx=(6, 0))
        self.open_button = ttk.Button(side, text="打开实时预览", style="Accent.TButton", command=self.start_camera); self.open_button.pack(fill="x", pady=3)
        self.close_button = ttk.Button(side, text="关闭摄像头", command=self.stop_camera, state="disabled"); self.close_button.pack(fill="x", pady=3)
        self.audio_button = ttk.Button(side, text="打开音频预览", command=self.toggle_audio_monitor); self.audio_button.pack(fill="x", pady=(3, 3))
        ttk.Label(side, text="音频灵敏度", style="Panel.TLabel").pack(anchor="w", pady=(8, 2))
        self.audio_sensitivity = tk.DoubleVar(value=1.35)
        self.audio_sensitivity_slider = ttk.Scale(
            side, from_=0.7, to=2.5, variable=self.audio_sensitivity,
            command=self._set_audio_sensitivity,
        )
        self.audio_sensitivity_slider.pack(fill="x")
        self.record_button = ttk.Button(side, text="开始录制", style="Accent.TButton", command=self.start_recording, state="disabled"); self.record_button.pack(fill="x", pady=(15, 3))
        self.stop_button = ttk.Button(side, text="停止并导出", command=self.stop_recording, state="disabled"); self.stop_button.pack(fill="x", pady=3)
        self.level = tk.StringVar(value="音频电平：--"); ttk.Label(side, textvariable=self.level, style="Panel.TLabel").pack(anchor="w", pady=(12, 5))
        ttk.Label(side, text="运行日志", style="Panel.TLabel").pack(anchor="w")
        self.logbox = tk.Text(side, width=34, height=17, relief="flat", bg="#f8fafc", fg="#334155"); self.logbox.pack(fill="both", expand=True, pady=(5, 0))
        ttk.Label(main, text="实时摄像头", style="Sub.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Label(main, text="实时参数化头像", style="Sub.TLabel").grid(row=0, column=1, sticky="w", padx=14, pady=(0, 6))
        self.preview = tk.Label(main, bg="#111827", fg="#d1d5db", text="摄像头未打开"); self.preview.grid(row=1, column=0, sticky="nsew", padx=(0, 7))
        self.avatar = AvatarPreview(main); self.avatar.grid(row=1, column=1, sticky="nsew", padx=(7, 0))
        self.metrics = tk.StringVar(value="帧号：--   人脸：--   录制：未开始")
        ttk.Label(
            main, textvariable=self.metrics, style="Sub.TLabel", wraplength=610
        ).grid(row=2, column=0, sticky="w", pady=(9, 0))
        audio_panel = ttk.Frame(main, style="Panel.TFrame", padding=(10, 5))
        audio_panel.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        audio_panel.columnconfigure(1, weight=1)
        self.audio_status = tk.StringVar(value="音频预览：未打开")
        ttk.Label(audio_panel, textvariable=self.audio_status, style="Panel.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.audio_meter = ttk.Progressbar(audio_panel, orient="horizontal", mode="determinate", maximum=100, length=180)
        self.audio_meter.grid(row=0, column=1, sticky="ew")
        self.viseme_status = tk.StringVar(value="口型：--")
        ttk.Label(
            audio_panel, textvariable=self.viseme_status, style="Panel.TLabel"
        ).grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.raw_diagnostics = tk.StringVar(value="MediaPipe 原始分数：等待人脸...")
        self.gnm_diagnostics = tk.StringVar(value="GNM 映射权重：等待人脸...")
        diagnostic = ttk.Frame(main, style="Panel.TFrame", padding=10)
        diagnostic.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(diagnostic, textvariable=self.raw_diagnostics, style="Panel.TLabel", wraplength=940).pack(anchor="w")
        ttk.Label(diagnostic, textvariable=self.gnm_diagnostics, style="Panel.TLabel", wraplength=940).pack(anchor="w", pady=(4, 0))

    def field(self, parent, text, values):
        ttk.Label(parent, text=text, style="Panel.TLabel").pack(anchor="w")
        box = ttk.Combobox(parent, state="readonly", values=values, width=32); box.pack(fill="x", pady=(5, 11)); box.current(0)
        return box

    def log(self, text: str) -> None:
        self.status.set(text); self.logbox.insert("end", f"[{time.strftime('%H:%M:%S')}] {text}\n"); self.logbox.see("end")

    def _load_mics(self) -> None:
        try:
            devices = enumerate_microphones(); values = [f"{i}: {n}" for i, n in devices] or ["未找到麦克风"]
            self.mic.configure(values=values); self.mic.current(0)
        except Exception as exc: self.log(f"读取麦克风失败: {exc}")

    def reload_devices(self) -> None:
        self.status.set("正在后台扫描摄像头..."); self._load_mics()
        def scan():
            try: result = enumerate_cameras()
            except Exception as exc: self.root.after(0, lambda: self.log(f"扫描摄像头失败: {exc}")); return
            self.root.after(0, lambda: self.set_cameras(result))
        threading.Thread(target=scan, daemon=True).start()

    def set_cameras(self, devices):
        values = [f"{i}: {n}" for i, n in devices] or ["0: 默认摄像头"]
        self.camera.configure(values=values); self.camera.current(0); self.log("设备列表已刷新。")

    def choose_directory(self) -> None:
        path = filedialog.askdirectory(initialdir=self.output.get(), title="选择保存目录")
        if path: self.output.delete(0, "end"); self.output.insert(0, path)

    def camera_index(self) -> int:
        try: return int(self.camera.get().split(":", 1)[0])
        except Exception: return -1

    def tracking_backend_id(self) -> str:
        value = self.tracking_backend.get()
        if value.startswith("FaceVerse V2"):
            return "faceverse_v2"
        if value.startswith("FaceVerse v4"):
            return "faceverse"
        return "mediapipe"

    def _backend_changed(self, _event=None) -> None:
        if self.tracking_backend_id() == "mediapipe":
            self.status.set("追踪后端：MediaPipe（默认，低延迟）")
            return
        if self.tracking_backend_id() == "faceverse_v2":
            _available, message = faceverse_v2_backend_status()
            self.status.set(f"追踪后端：FaceVerse V2（52维）；{message}")
            return
        _available, message = faceverse_backend_status()
        self.status.set(f"追踪后端：FaceVerse v4；{message}")

    def _offer_faceverse_setup(self, message: str) -> None:
        prompt = (
            f"FaceVerse v4 暂时不可用：{message}\n\n"
            f"请将模型文件放入：\n{FACEVERSE_DATA_DIR}\n\n"
            "是否打开模型目录和官方两个下载页面？"
        )
        if not messagebox.askyesno("FaceVerse v4", prompt):
            return
        ensure_directory(FACEVERSE_DATA_DIR)
        try:
            os.startfile(str(FACEVERSE_DATA_DIR))
        except Exception:
            pass
        webbrowser.open(
            "https://1drv.ms/u/c/b8eab7b1820a6fa4/"
            "EWJOsgGxPMZDkl8xJ_QZB30BpcjNoMVGK9mnUPq5n9-lyw?e=4GvEs9"
        )
        webbrowser.open(
            "https://1drv.ms/u/c/b8eab7b1820a6fa4/"
            "ETfT_C9Oz1FFlykJdtj3h6MBR1KvQb5BYwesxFykH-7BZA?e=7ti1yj"
        )

    def _offer_faceverse_v2_setup(self, message: str) -> None:
        prompt = (
            f"FaceVerse V2 暂时不可用：{message}\n\n"
            f"请将 faceverse_simple_v2.npy 放入：\n{FACEVERSE_V2_DATA_DIR}\n\n"
            "是否打开模型目录和官方 FaceVerse V2 下载页面？"
        )
        if not messagebox.askyesno("FaceVerse V2", prompt):
            return
        ensure_directory(FACEVERSE_V2_DATA_DIR)
        try:
            os.startfile(str(FACEVERSE_V2_DATA_DIR))
        except Exception:
            pass
        webbrowser.open("https://github.com/LizhenWangT/FaceVerse#faceverse-version-2")

    def mic_choice(self):
        value = self.mic.get()
        try: return int(value.split(":", 1)[0]), value
        except Exception: return None, value

    def start_camera(self) -> None:
        if self.worker and self.worker.is_alive(): return
        if self.stopping_worker and self.stopping_worker.is_alive():
            self.log("正在释放上一台摄像头，请稍候...")
            return
        self.stopping_worker = None
        index = self.camera_index()
        if index < 0: messagebox.showerror("Face Avatar Studio", "请先选择摄像头。"); return
        backend = self.tracking_backend_id()
        if backend == "faceverse":
            available, message = faceverse_backend_status()
            if not available:
                self._offer_faceverse_setup(message)
                return
        elif backend == "faceverse_v2":
            available, message = faceverse_v2_backend_status()
            if not available:
                self._offer_faceverse_v2_setup(message)
                return
        self.worker = CaptureWorker(index, backend); self.worker.start(); self.last_packet_index = -1
        self.tracking_clock_started = False
        self.last_tracking_wall = time.perf_counter()
        self.last_tracking_index = 0
        self.open_button.configure(state="disabled"); self.close_button.configure(state="normal")
        self.tracking_backend.configure(state="disabled")
        self.preview.configure(image="", text="正在打开摄像头..."); self.avatar.show_message("正在加载头像和表情追踪模型")
        backend_name = _backend_display_name(backend)
        self.metrics.set(
            f"后端：{backend_name}   人脸：等待模型   摄像头：正在打开   "
            "追踪：预热中   头像：正在加载   录制：未开始"
        )
        self.log(f"已请求打开摄像头，正在后台加载 {backend_name}。")

    def stop_camera(self) -> None:
        worker = self.worker
        if worker:
            worker.stop()
            self.stopping_worker = worker
        self.worker = None
        self.open_button.configure(state="disabled")
        self.close_button.configure(state="disabled")
        self.record_button.configure(state="disabled")
        self.preview.configure(image="", text="摄像头已关闭"); self.avatar.show_message("等待表情追踪模型"); self.log("摄像头已关闭。")
        if worker:
            self.root.after(50, self._finish_camera_stop)

    def _finish_camera_stop(self) -> None:
        worker = self.stopping_worker
        if worker is not None and worker.is_alive():
            self.root.after(50, self._finish_camera_stop)
            return
        self.stopping_worker = None
        if self.worker is None:
            self.open_button.configure(state="normal")
            self.tracking_backend.configure(state="readonly")

    def toggle_audio_monitor(self) -> None:
        if self.audio_monitor.active:
            self.audio_monitor.stop()
            self.audio_button.configure(text="打开音频预览")
            self.audio_status.set("音频预览：未打开")
            self.log("音频预览已关闭。")
            return
        device, name = self.mic_choice()
        if device is None:
            messagebox.showerror("Face Avatar Studio", "请先选择一个麦克风。")
            return
        try:
            self.audio_monitor.start(device)
        except Exception as exc:
            self.audio_monitor.stop()
            messagebox.showerror("Face Avatar Studio", f"无法打开麦克风：{exc}")
            return
        self.audio_button.configure(text="关闭音频预览")
        self.audio_status.set(f"音频预览：{name[:18]}")
        self.log(f"音频预览已打开：{name}")

    def _set_audio_sensitivity(self, value: str) -> None:
        try:
            self.audio_monitor.set_sensitivity(float(value))
        except (TypeError, ValueError):
            pass

    def start_recording(self) -> None:
        mic, name = self.mic_choice()
        packet = self.worker.latest_packet() if self.worker else None
        if not self.worker or packet is None or not packet.face_detected:
            messagebox.showwarning("Face Avatar Studio", "请等待摄像头检测到人脸后再开始录制。")
            return
        if mic is None: messagebox.showerror("Face Avatar Studio", "请先选择麦克风。"); return
        language = {"中文": "zh", "English": "en"}.get(self.language.get(), "auto")
        try:
            # The preview stream and the recording stream cannot reliably
            # share the same Windows input device. Hand the device to the
            # recorder when recording starts.
            if self.audio_monitor.active:
                self.audio_monitor.stop()
                self.audio_button.configure(text="打开音频预览")
                self.audio_status.set("音频预览：录制占用设备")
            rec = SessionRecorder(
                ensure_directory(Path(self.output.get()).expanduser()),
                CAPTURE_SIZE,
                TRACKING_FPS,
                stt_language=language,
                tracking_backend=self.worker.tracking_backend,
            )
            session = rec.start(self.camera.get(), name, mic); self.recorder = rec; self.transcriber = WhisperTranscriber(); self.worker.set_recorder(rec)
            self.record_button.configure(state="disabled"); self.stop_button.configure(state="normal"); self.log(f"录制已开始: {session}")
        except Exception as exc: messagebox.showerror("Face Avatar Studio", f"无法开始录制: {exc}")

    def stop_recording(self) -> None:
        if not self.recorder: return
        if self.worker: self.worker.set_recorder(None)
        rec, tr = self.recorder, self.transcriber or WhisperTranscriber(); rec.stop_capture(); self.recorder = None; self.transcriber = None
        self.exporter = ExportWorker(rec, tr); self.exporter.start(); self.stop_button.configure(state="disabled"); self.log("正在导出音画同步视频、表情参数和语音时间轴...")

    def poll(self) -> None:
        worker = self.worker
        if worker:
            while True:
                try: message = worker.status_queue.get_nowait()
                except queue.Empty: break
                if message != "camera_stopped": self.log(message)
            packet = worker.latest_packet()
            # packet.frame_bgr stops changing while MediaPipe warms up. The
            # capture thread remains live, so always prefer its newest frame
            # for the camera pane and use the packet only as a fallback.
            frame = worker.latest_frame()
            if frame is None and packet is not None:
                frame = packet.frame_bgr
            if frame is not None and time.perf_counter() - self.last_preview_time > .067:
                self.last_preview_time = time.perf_counter()
                if packet is not None and packet.landmarks is not None:
                    frame = _draw_landmarks(frame, packet.landmarks)
                self.show_camera(frame)
            if packet and packet.frame_index != self.last_packet_index:
                self.last_packet_index = packet.frame_index
                tracking_ready = packet.frame_index >= 0 and packet.face_detected
                self.record_button.configure(
                    state="disabled" if self.recorder or not tracking_ready else "normal"
                )
                if packet.face_tracking_hold:
                    face = "保持上一帧"
                else:
                    face = "已检测到" if packet.face_detected else "未检测到"
                recording = "进行中" if self.recorder else "未开始"
                now = time.perf_counter()
                if packet.frame_index < 0:
                    tracking_fps = 0.0
                    tracking_status = "预热中"
                elif not self.tracking_clock_started:
                    # Do not include child-process/model startup in the FPS.
                    self.tracking_clock_started = True
                    self.last_tracking_wall, self.last_tracking_index = now, packet.frame_index
                    tracking_fps = 0.0
                    tracking_status = "已启动"
                else:
                    elapsed = max(now - self.last_tracking_wall, 1e-6)
                    tracking_fps = (packet.frame_index - self.last_tracking_index) / elapsed
                    self.last_tracking_wall, self.last_tracking_index = now, packet.frame_index
                    tracking_status = f"{tracking_fps:.1f} FPS"
                backend_name = _backend_display_name(worker.tracking_backend)
                self.metrics.set(f"后端：{backend_name}   帧号：{packet.frame_index}   人脸：{face}   摄像头：{worker.capture_fps:.1f} FPS   追踪：{tracking_status}   头像：{worker.avatar_render_fps:.1f} FPS   录制：{recording}")
                # These long diagnostic strings force ttk to recalculate
                # wrapping/layout. Keep the live meter and preview fast, and
                # refresh diagnostics at a human-readable rate.
                if now - self.last_diagnostic_time >= 0.20:
                    self.last_diagnostic_time = now
                    keys = (
                        "jawOpen", "geometryJawOpen", "mouthPucker", "mouthFunnel",
                        "geometryLipParted", "geometryMouthWide",
                        "mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft",
                        "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
                        "geometryBlinkLeft", "geometryBlinkRight",
                        "geometryEyeWideLeft", "geometryEyeWideRight",
                        "audioVoiceActivity", "audioVisemeA", "audioVisemeE",
                        "audioVisemeI", "audioVisemeO", "audioVisemeU",
                        "audioVisemeBMP", "audioVisemeFV", "audioVisemeCH",
                    )
                    if packet.tracking_error:
                        self.raw_diagnostics.set(f"MediaPipe 追踪错误：{packet.tracking_error}")
                    else:
                        self.raw_diagnostics.set(
                            "MediaPipe 原始分数："
                            + "  ".join(
                                f"{key} {packet.blendshapes.get(key, 0.0):.2f}"
                                for key in keys
                            )
                        )
                    mapped = sorted(
                        packet.semantic_weights.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:18]
                    if worker.tracking_backend == "faceverse_v2":
                        parameter_label = "FaceVerse V2 52维参数"
                    elif worker.tracking_backend == "faceverse":
                        parameter_label = "FaceVerse v4 表情参数"
                    else:
                        parameter_label = "GNM 映射权重"
                    self.gnm_diagnostics.set(
                        parameter_label
                        + "："
                        + "  ".join(f"{key} {value:.2f}" for key, value in mapped)
                    )
        if self.recorder:
            level = _sensitive_level(
                self.recorder.latest_level, self.audio_sensitivity.get()
            )
            visemes = self.recorder.latest_audio_visemes()
            self.level.set(f"录音电平：{level:.2f}")
        elif self.audio_monitor.active:
            level = self.audio_monitor.latest_level()
            visemes = self.audio_monitor.latest_visemes()
            self.level.set(f"监听电平：{level:.2f}")
        else:
            level = 0.0
            visemes = {}
            self.level.set("音频电平：--")
        self.audio_meter.configure(value=level * 100.0)
        viseme_names = {
            "audioVisemeA": "A",
            "audioVisemeE": "E",
            "audioVisemeI": "I",
            "audioVisemeO": "O",
            "audioVisemeU": "U",
            "audioVisemeClosed": "闭唇",
            "audioVisemeBMP": "B/P/M",
            "audioVisemeFV": "F/V",
            "audioVisemeTH": "TH",
            "audioVisemeTD": "T/D/N/L",
            "audioVisemeKG": "K/G/NG",
            "audioVisemeCH": "CH/SH/J",
            "audioVisemeSZ": "S/Z",
            "audioVisemeR": "R",
        }
        active_viseme = max(
            viseme_names,
            key=lambda name: float(visemes.get(name, 0.0)),
        )
        active_value = float(visemes.get(active_viseme, 0.0))
        self.viseme_status.set(
            f"口型：{viseme_names[active_viseme]}"
            if active_value >= 0.12 else "口型：--"
        )
        if worker:
            worker.set_audio_features(level, visemes)
        if worker:
            avatar_frame = worker.latest_avatar_frame()
            if (
                avatar_frame is not None
                and avatar_frame is not self.last_avatar_frame
                and time.perf_counter() - self.last_avatar_preview_time >= 0.08
            ):
                self.last_avatar_preview_time = time.perf_counter()
                self.last_avatar_frame = avatar_frame
                self.avatar.set_frame(avatar_frame)
        if self.exporter:
            try: result = self.exporter.result_queue.get_nowait()
            except queue.Empty:
                try: error = self.exporter.error_queue.get_nowait()
                except queue.Empty: error = None
                if error: self.log(f"导出失败: {error}"); self.exporter = None
            else: self.log(f"导出完成: {result.session_dir}"); self.exporter = None
        try: self.root.after(33, self.poll)
        except tk.TclError: pass

    def show_camera(self, frame):
        source_h, source_w = frame.shape[:2]
        target_w = max(420, self.preview.winfo_width())
        target_h = max(236, self.preview.winfo_height())
        scale = min(target_w / source_w, target_h / source_h, 1.0)
        if scale < 0.995:
            frame = cv2.resize(
                frame,
                (max(1, int(source_w * scale)), max(1, int(source_h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_photo, text="")

    def close(self) -> None:
        if self.recorder and self.recorder.active:
            try: self.recorder.stop_capture()
            except Exception: pass
        self.audio_monitor.stop()
        if self.worker: self.worker.stop()
        if self.stopping_worker: self.stopping_worker.stop()
        self.root.destroy()

    def run(self) -> int:
        self.root.mainloop(); return 0


def main() -> int:
    return FaceAvatarStudioApp().run()


if __name__ == "__main__":
    raise SystemExit(main())
