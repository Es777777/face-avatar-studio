from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from vtk import vtkActor, vtkCellArray, vtkNamedColors, vtkPoints, vtkPolyData, vtkPolyDataNormals, vtkPolyDataMapper, vtkRenderer, vtkTriangleFilter
from vtk.util import numpy_support
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from .pipeline import (
    DEFAULT_FRAME_SIZE,
    DEFAULT_FPS,
    DEFAULT_SAMPLE_RATE,
    FaceTrackingEngine,
    FramePacket,
    RecordingOutputs,
    SessionRecorder,
    WhisperTranscriber,
    ensure_directory,
)


def enumerate_cameras(max_index: int = 10) -> list[tuple[int, str]]:
    cameras: list[tuple[int, str]] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if cap.isOpened():
                cameras.append((index, f"Camera {index}"))
        finally:
            cap.release()
    return cameras


def enumerate_microphones() -> list[tuple[int, str]]:
    import sounddevice as sd

    devices = []
    for index, device in enumerate(sd.query_devices()):
        if device.get("max_input_channels", 0) > 0:
            devices.append((index, str(device["name"])))
    return devices


class CaptureThread(QtCore.QThread):
    frame_ready = QtCore.Signal(object)
    status = QtCore.Signal(str)
    stopped = QtCore.Signal()

    def __init__(
        self,
        camera_index: int,
        recorder: SessionRecorder | None,
        engine: FaceTrackingEngine,
        frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
        fps: int = DEFAULT_FPS,
    ) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.recorder = recorder
        self.frame_size = frame_size
        self.fps = fps
        self._stop_requested = threading.Event()
        self._engine = engine

    def stop(self) -> None:
        self._stop_requested.set()

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.status.emit(f"Unable to open camera {self.camera_index}.")
            self.stopped.emit()
            return
        width, height = self.frame_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        start_time = time.perf_counter()
        frame_index = 0
        self.status.emit(f"Camera {self.camera_index} started.")
        try:
            while not self._stop_requested.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
                timestamp_sec = time.perf_counter() - start_time
                packet = self._engine.process_frame(frame, frame_index, timestamp_sec)
                if self.recorder is not None and self.recorder.active:
                    self.recorder.append_packet(packet)
                self.frame_ready.emit(packet)
                frame_index += 1
        finally:
            cap.release()
            self.stopped.emit()


class ExportThread(QtCore.QThread):
    done = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(self, recorder: SessionRecorder, transcriber: WhisperTranscriber | None) -> None:
        super().__init__()
        self.recorder = recorder
        self.transcriber = transcriber

    def run(self) -> None:
        try:
            outputs = self.recorder.finalize(transcriber=self.transcriber)
            self.done.emit(outputs)
        except Exception as exc:
            self.error.emit(str(exc))


class AvatarViewport(QtWidgets.QWidget):
    def __init__(
        self,
        avatar_driver,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._driver = avatar_driver
        self._points = vtkPoints()
        self._polydata = vtkPolyData()
        self._normals = vtkPolyDataNormals()
        self._mapper = vtkPolyDataMapper()
        self._actor = vtkActor()
        self._renderer = vtkRenderer()
        self._render_widget = QVTKRenderWindowInteractor(self)
        self._render_widget.GetRenderWindow().AddRenderer(self._renderer)
        self._renderer.SetBackground(0.08, 0.09, 0.11)
        self._build_pipeline()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._render_widget)
        self._render_widget.Initialize()
        self._render_widget.Start()

    def _build_pipeline(self) -> None:
        vertices = np.asarray(self._driver.model.template_vertex_positions, dtype=np.float32)
        triangles = np.asarray(self._driver.model.triangles, dtype=np.int64)
        self._points.SetData(numpy_support.numpy_to_vtk(vertices, deep=True))
        self._polydata.SetPoints(self._points)
        faces = np.hstack(
            [
                np.full((triangles.shape[0], 1), 3, dtype=np.int64),
                triangles,
            ]
        ).ravel()
        cell_array = vtkCellArray()
        cell_array.SetCells(
            int(triangles.shape[0]), numpy_support.numpy_to_vtkIdTypeArray(faces, deep=True)
        )
        self._polydata.SetPolys(cell_array)
        self._normals.SetInputData(self._polydata)
        self._normals.ComputePointNormalsOn()
        self._normals.ComputeCellNormalsOn()
        self._normals.ConsistencyOn()
        self._normals.SplittingOff()
        self._normals.AutoOrientNormalsOn()
        self._normals.Update()
        self._mapper.SetInputConnection(self._normals.GetOutputPort())
        self._actor.SetMapper(self._mapper)
        prop = self._actor.GetProperty()
        prop.SetColor(0.85, 0.78, 0.70)
        prop.SetSpecular(0.2)
        prop.SetSpecularPower(20)
        prop.SetAmbient(0.25)
        self._renderer.AddActor(self._actor)
        self._renderer.ResetCamera()
        camera = self._renderer.GetActiveCamera()
        camera.Azimuth(180)
        camera.Elevation(10)
        camera.Dolly(1.7)
        self._renderer.ResetCameraClippingRange()

    def set_vertices(self, vertices: np.ndarray) -> None:
        vtk_array = numpy_support.numpy_to_vtk(np.asarray(vertices, dtype=np.float32), deep=True)
        self._points.SetData(vtk_array)
        self._points.Modified()
        self._polydata.Modified()
        self._normals.Update()
        self._render_widget.GetRenderWindow().Render()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Face Avatar Studio")
        self.resize(1560, 920)
        self._capture_thread: CaptureThread | None = None
        self._export_thread: ExportThread | None = None
        self._recorder: SessionRecorder | None = None
        self._transcriber: WhisperTranscriber | None = None
        self._last_packet: FramePacket | None = None
        self._engine = FaceTrackingEngine()
        self._output_root = ensure_directory(Path.home() / "FaceAvatarStudio")
        self._build_ui()
        self._reload_devices()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self.control_panel = QtWidgets.QFrame()
        self.control_panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        controls = QtWidgets.QVBoxLayout(self.control_panel)
        controls.setContentsMargins(12, 12, 12, 12)
        controls.setSpacing(10)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        self.camera_combo = QtWidgets.QComboBox()
        self.mic_combo = QtWidgets.QComboBox()
        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.addItems(["auto", "zh", "en"])
        self.output_edit = QtWidgets.QLineEdit(str(self._output_root))
        browse = QtWidgets.QToolButton()
        browse.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        browse.clicked.connect(self._choose_output_dir)
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(browse, 0)

        form.addRow("Camera", self.camera_combo)
        form.addRow("Microphone", self.mic_combo)
        form.addRow("Speech", self.language_combo)
        form.addRow("Save to", self._wrap_layout(output_row))
        controls.addLayout(form)

        self.refresh_button = QtWidgets.QPushButton("Reload Devices")
        self.refresh_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        self.refresh_button.clicked.connect(self._reload_devices)

        self.record_button = QtWidgets.QPushButton("Start Recording")
        self.record_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.record_button.clicked.connect(self._toggle_recording)

        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop))
        self.stop_button.clicked.connect(self._stop_recording)
        self.stop_button.setEnabled(False)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.refresh_button)
        button_row.addWidget(self.record_button)
        button_row.addWidget(self.stop_button)
        controls.addLayout(button_row)

        self.status_label = QtWidgets.QLabel("Idle")
        self.status_label.setWordWrap(True)
        controls.addWidget(self.status_label)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Session log will appear here.")
        self.log_view.setMinimumHeight(220)
        controls.addWidget(self.log_view, 1)

        right_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumSize(720, 405)
        self.preview_label.setStyleSheet("background: #0d0f12; border: 1px solid #20242c;")
        self.preview_label.setText("Camera preview")
        self.avatar_view = AvatarViewport(self._engine.avatar)
        right_split.addWidget(self.preview_label)
        right_split.addWidget(self.avatar_view)
        right_split.setStretchFactor(0, 1)
        right_split.setStretchFactor(1, 1)

        root.addWidget(self.control_panel, 0)
        root.addWidget(right_split, 1)
        self.setCentralWidget(central)

    def _wrap_layout(self, layout: QtWidgets.QHBoxLayout) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        return widget

    def _log(self, message: str) -> None:
        self.log_view.appendPlainText(message)
        self.status_label.setText(message)

    def _choose_output_dir(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose save folder", self.output_edit.text())
        if selected:
            self.output_edit.setText(selected)

    def _reload_devices(self) -> None:
        self.camera_combo.clear()
        self.mic_combo.clear()
        for index, name in enumerate_cameras():
            self.camera_combo.addItem(f"{index}: {name}", index)
        for index, name in enumerate_microphones():
            self.mic_combo.addItem(f"{index}: {name}", index)
        if self.camera_combo.count() == 0:
            self.camera_combo.addItem("No camera found", -1)
        if self.mic_combo.count() == 0:
            self.mic_combo.addItem("No microphone found", -1)
        self._log("Device list refreshed.")

    def _selected_camera(self) -> int:
        return int(self.camera_combo.currentData())

    def _selected_mic(self) -> tuple[int | None, str]:
        device_index = int(self.mic_combo.currentData())
        if device_index < 0:
            return None, ""
        return device_index, self.mic_combo.currentText()

    def _toggle_recording(self) -> None:
        if self._capture_thread is not None:
            self._stop_recording()
            return
        self._start_recording()

    def _start_recording(self) -> None:
        camera_index = self._selected_camera()
        mic_index, mic_name = self._selected_mic()
        if camera_index < 0:
            self._log("No camera selected.")
            return
        if mic_index is None:
            self._log("No microphone selected.")
            return

        output_root = Path(self.output_edit.text()).expanduser()
        ensure_directory(output_root)
        self._recorder = SessionRecorder(
            output_root=output_root,
            frame_size=DEFAULT_FRAME_SIZE,
            fps=DEFAULT_FPS,
            sample_rate=DEFAULT_SAMPLE_RATE,
            stt_language=self.language_combo.currentText(),
        )
        try:
            session_dir = self._recorder.start(
                camera_name=self.camera_combo.currentText(),
                microphone_name=mic_name,
                mic_device_index=mic_index,
            )
        except Exception as exc:
            self._recorder = None
            self._log(f"Recording could not start: {exc}")
            return

        self._transcriber = WhisperTranscriber()
        self._capture_thread = CaptureThread(camera_index, self._recorder, self._engine)
        self._capture_thread.frame_ready.connect(self._on_frame_ready)
        self._capture_thread.status.connect(self._log)
        self._capture_thread.stopped.connect(self._on_capture_stopped)
        self._capture_thread.start()
        self.record_button.setText("Recording")
        self.record_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop))
        self.stop_button.setEnabled(True)
        self.refresh_button.setEnabled(False)
        self._log(f"Recording started in {session_dir}")

    def _stop_recording(self) -> None:
        if self._capture_thread is None:
            return
        self._capture_thread.stop()
        self._capture_thread.wait()
        if self._recorder is not None:
            self._recorder.stop_capture()
        self.stop_button.setEnabled(False)
        self._start_export()

    def _on_capture_stopped(self) -> None:
        pass

    def _on_frame_ready(self, packet: FramePacket) -> None:
        self._last_packet = packet
        self._show_frame(packet.frame_bgr)
        self.avatar_view.set_vertices(packet.vertices)
        if packet.face_detected:
            self.statusBar().showMessage(
                f"Face detected | frame {packet.frame_index} | {packet.timestamp_sec:.2f}s"
            )
        else:
            self.statusBar().showMessage(f"Tracking | frame {packet.frame_index} | {packet.timestamp_sec:.2f}s")

    def _show_frame(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(image)
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._last_packet is not None:
            self._show_frame(self._last_packet.frame_bgr)

    def _start_export(self) -> None:
        if self._recorder is None:
            self._cleanup_after_recording()
            return
        self._export_thread = ExportThread(self._recorder, self._transcriber)
        self._export_thread.done.connect(self._on_export_done)
        self._export_thread.error.connect(self._on_export_error)
        self._export_thread.start()
        self._log("Exporting recording and transcribing speech...")

    def _on_export_done(self, outputs: RecordingOutputs) -> None:
        self._log(f"Saved session: {outputs.session_dir}")
        if outputs.merged_video_path is not None:
            self._log(f"Video: {outputs.merged_video_path}")
        if outputs.timeline_csv_path is not None:
            self._log(f"Timeline CSV: {outputs.timeline_csv_path}")
        if outputs.transcript_csv_path is not None:
            self._log(f"Transcript CSV: {outputs.transcript_csv_path}")
        self._cleanup_after_recording()

    def _on_export_error(self, message: str) -> None:
        self._log(f"Export failed: {message}")
        self._cleanup_after_recording()

    def _cleanup_after_recording(self) -> None:
        self._capture_thread = None
        self._export_thread = None
        self._recorder = None
        self._transcriber = None
        self.record_button.setText("Start Recording")
        self.record_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.stop_button.setEnabled(False)
        self.refresh_button.setEnabled(True)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._capture_thread is not None:
            self._capture_thread.stop()
            self._capture_thread.wait(3000)
        if self._recorder is not None and self._recorder.active:
            try:
                self._recorder.stop_capture()
            except Exception:
                pass
        if self._export_thread is not None:
            self._export_thread.wait(5000)
        super().closeEvent(event)
