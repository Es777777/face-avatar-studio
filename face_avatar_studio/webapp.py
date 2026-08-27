from __future__ import annotations

import json
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from .pipeline import (
    AvatarPreviewRenderer,
    DEFAULT_FPS,
    DEFAULT_FRAME_SIZE,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_STT_MODEL,
    FaceTrackingEngine,
    FramePacket,
    RecordingOutputs,
    SessionRecorder,
    WhisperTranscriber,
    ensure_directory,
)


def enumerate_cameras(max_index: int = 10) -> list[dict[str, Any]]:
    cameras: list[dict[str, Any]] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if cap.isOpened():
                cameras.append({"index": index, "label": f"Camera {index}"})
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


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 85) -> bytes:
    success, encoded = cv2.imencode(
        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not success:
        return b""
    return encoded.tobytes()


class CameraWorker(threading.Thread):
    def __init__(
        self,
        state: "AppState",
        camera_index: int,
        frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
        fps: int = DEFAULT_FPS,
    ) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.camera_index = camera_index
        self.frame_size = frame_size
        self.fps = fps
        self.stop_event = threading.Event()
        self.preview_stride = 2

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.state.add_log(f"Unable to open camera {self.camera_index}.")
            return

        width, height = self.frame_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        start_time = time.perf_counter()
        frame_index = 0
        self.state.add_log(f"Camera {self.camera_index} started.")

        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
                timestamp_sec = time.perf_counter() - start_time
                packet = self.state.engine.process_frame(frame, frame_index, timestamp_sec)
                self.state.consume_packet(packet)

                if frame_index % self.preview_stride == 0:
                    self.state.latest_camera_jpeg = encode_jpeg(packet.frame_bgr)
                    self.state.avatar_renderer.update_vertices(packet.vertices)
                    self.state.latest_avatar_jpeg = self.state.avatar_renderer.render_jpeg()

                if self.state.recorder is not None and self.state.recorder.active:
                    self.state.recorder.append_packet(packet)

                self.state.latest_packet = packet
                frame_index += 1
        finally:
            cap.release()
            self.state.add_log("Camera stream stopped.")


class ExportWorker(threading.Thread):
    def __init__(self, state: "AppState") -> None:
        super().__init__(daemon=True)
        self.state = state

    def run(self) -> None:
        try:
            recorder = self.state.recorder
            if recorder is None:
                return
            outputs = recorder.finalize(transcriber=self.state.transcriber)
            self.state.set_last_outputs(outputs)
            self.state.add_log(f"Saved session: {outputs.session_dir}")
            if outputs.merged_video_path is not None:
                self.state.add_log(f"Video: {outputs.merged_video_path}")
            if outputs.timeline_csv_path is not None:
                self.state.add_log(f"Timeline CSV: {outputs.timeline_csv_path}")
            if outputs.transcript_csv_path is not None:
                self.state.add_log(f"Transcript CSV: {outputs.transcript_csv_path}")
        except Exception as exc:
            self.state.add_log(f"Export failed: {exc}")
        finally:
            with self.state.lock:
                self.state.export_worker = None
                self.state.recorder = None
                self.state.transcriber = None
                self.state.recording = False
                self.state.exporting = False


class AppState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.engine = FaceTrackingEngine()
        self.avatar_renderer = AvatarPreviewRenderer(self.engine.avatar, size=(960, 540))
        self.capture_worker: CameraWorker | None = None
        self.export_worker: ExportWorker | None = None
        self.recorder: SessionRecorder | None = None
        self.transcriber: WhisperTranscriber | None = None
        self.latest_packet: FramePacket | None = None
        self.latest_camera_jpeg: bytes | None = None
        self.latest_avatar_jpeg: bytes | None = None
        self.logs: deque[str] = deque(maxlen=200)
        self.output_root = ensure_directory(Path.home() / "FaceAvatarStudio")
        self.recording = False
        self.exporting = False
        self.current_camera_index: int | None = None
        self.current_mic_index: int | None = None
        self.current_mic_name: str = ""
        self.current_speech_language = "auto"
        self.current_stt_model = DEFAULT_STT_MODEL
        self.last_outputs: RecordingOutputs | None = None

    def add_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with self.lock:
            self.logs.append(f"[{stamp}] {message}")

    def consume_packet(self, packet: FramePacket) -> None:
        with self.lock:
            self.latest_packet = packet
            if self.latest_camera_jpeg is None:
                self.latest_camera_jpeg = encode_jpeg(packet.frame_bgr)

    def set_last_outputs(self, outputs: RecordingOutputs) -> None:
        with self.lock:
            self.last_outputs = outputs

    def start_camera(self, camera_index: int) -> None:
        with self.lock:
            if self.capture_worker is not None:
                self.capture_worker.stop()
                self.capture_worker.join(timeout=3)
            self.current_camera_index = camera_index
            self.capture_worker = CameraWorker(self, camera_index)
            self.capture_worker.start()
            self.add_log(f"Switched to camera {camera_index}.")

    def start_recording(
        self,
        output_root: str,
        mic_index: int,
        mic_name: str,
        language: str,
        stt_model: str,
    ) -> dict[str, Any]:
        with self.lock:
            if self.recording or self.exporting:
                raise RuntimeError("Recording already in progress.")

            self.output_root = ensure_directory(Path(output_root).expanduser())
            self.recorder = SessionRecorder(
                output_root=self.output_root,
                frame_size=DEFAULT_FRAME_SIZE,
                fps=DEFAULT_FPS,
                sample_rate=DEFAULT_SAMPLE_RATE,
                stt_language=language,
            )
            self.transcriber = WhisperTranscriber(model_name=stt_model)
            self.current_mic_index = mic_index
            self.current_mic_name = mic_name
            self.current_speech_language = language
            self.current_stt_model = stt_model
            session_dir = self.recorder.start(
                camera_name=f"Camera {self.current_camera_index}",
                microphone_name=mic_name,
                mic_device_index=mic_index,
            )
            self.recording = True
            self.add_log(f"Recording started in {session_dir}")
            return {"session_dir": str(session_dir)}

    def stop_recording(self) -> None:
        with self.lock:
            if self.recorder is None and self.capture_worker is None:
                return
            if self.recorder is not None and self.recorder.active:
                self.recorder.stop_capture()
            self.recording = False
            self.exporting = True
            self.export_worker = ExportWorker(self)
            self.export_worker.start()
            self.add_log("Exporting recording and transcribing speech...")

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            packet = self.latest_packet
            outputs = self.last_outputs
            return {
                "recording": self.recording,
                "exporting": self.exporting,
                "camera_index": self.current_camera_index,
                "mic_index": self.current_mic_index,
                "mic_name": self.current_mic_name,
                "speech_language": self.current_speech_language,
                "stt_model": self.current_stt_model,
                "frame_index": None if packet is None else packet.frame_index,
                "timestamp_sec": None if packet is None else round(packet.timestamp_sec, 3),
                "face_detected": None if packet is None else packet.face_detected,
                "session_dir": None if self.recorder is None else str(self.recorder.session_dir),
                "output_root": str(self.output_root),
                "last_outputs": None
                if outputs is None
                else {
                    "session_dir": str(outputs.session_dir),
                    "video_path": None if outputs.merged_video_path is None else str(outputs.merged_video_path),
                    "timeline_csv_path": None if outputs.timeline_csv_path is None else str(outputs.timeline_csv_path),
                    "transcript_csv_path": None if outputs.transcript_csv_path is None else str(outputs.transcript_csv_path),
                    "audio_path": None if outputs.audio_path is None else str(outputs.audio_path),
                    "manifest_path": None if outputs.manifest_path is None else str(outputs.manifest_path),
                },
                "logs": list(self.logs)[-40:],
            }


def build_html() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Face Avatar Studio</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0c1016;
      --panel: #121823;
      --panel-2: #0f141d;
      --line: #233041;
      --text: #e6edf7;
      --muted: #93a4b7;
      --accent: #52a8ff;
      --accent-2: #68d391;
      --danger: #ff6b6b;
      --shadow: 0 12px 34px rgba(0, 0, 0, 0.28);
      --radius: 12px;
      --radius-sm: 8px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top left, rgba(82, 168, 255, 0.14), transparent 22%),
                  linear-gradient(180deg, #0a0e14 0%, #0c1016 100%);
      color: var(--text);
      font: 14px/1.4 "Segoe UI", system-ui, -apple-system, sans-serif;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      background: rgba(10, 14, 20, 0.72);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .brand-badge {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: 0 0 0 6px rgba(82, 168, 255, 0.13);
    }
    .status-row { display: flex; gap: 8px; flex-wrap: wrap; }
    .chip {
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(18, 24, 35, 0.88);
      color: var(--muted);
      font-size: 12px;
    }
    .chip.ok { color: var(--accent-2); }
    .chip.live { color: var(--accent); }
    .chip.warn { color: #ffd166; }
    .layout {
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      gap: 16px;
      padding: 16px;
    }
    .panel {
      background: linear-gradient(180deg, rgba(18, 24, 35, 0.96), rgba(12, 16, 22, 0.96));
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .sidebar { padding: 16px; display: flex; flex-direction: column; gap: 16px; }
    .section-title { font-weight: 700; margin-bottom: 10px; }
    .field { display: grid; gap: 6px; margin-bottom: 12px; }
    .field label { color: var(--muted); font-size: 12px; }
    select, input, button {
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: var(--panel-2);
      color: var(--text);
      padding: 10px 12px;
      font: inherit;
      outline: none;
    }
    input { width: 100%; }
    button {
      cursor: pointer;
      transition: 0.15s ease;
    }
    button:hover { border-color: var(--accent); transform: translateY(-1px); }
    button.primary { background: linear-gradient(180deg, #3f8cff, #2d74f0); border-color: #3f8cff; }
    button.good { background: linear-gradient(180deg, #3dbd7d, #23965e); border-color: #3dbd7d; }
    button.danger { background: linear-gradient(180deg, #f45d5d, #c63c3c); border-color: #f45d5d; }
    button:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }
    .button-row { display: flex; gap: 8px; flex-wrap: wrap; }
    .button-row button { flex: 1 1 0; min-width: 96px; }
    .main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 16px;
      min-width: 0;
    }
    .preview-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      min-width: 0;
    }
    .view-card {
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-width: 0;
    }
    .view-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
    }
    .preview {
      width: 100%;
      aspect-ratio: 16 / 9;
      border-radius: 10px;
      background: #06080b;
      border: 1px solid var(--line);
      object-fit: cover;
      display: block;
    }
    .info-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .stat {
      padding: 12px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: rgba(18, 24, 35, 0.72);
    }
    .stat .k { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .stat .v { font-size: 15px; font-weight: 700; word-break: break-word; }
    .logs {
      padding: 14px 16px 16px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      min-width: 0;
    }
    .logbox, .filebox {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(8, 11, 16, 0.8);
      padding: 12px;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      color: #cfd9e6;
      max-height: 240px;
      overflow: auto;
    }
    .links { display: grid; gap: 8px; }
    a.download {
      display: inline-block;
      color: var(--text);
      text-decoration: none;
      padding: 9px 11px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(18, 24, 35, 0.86);
    }
    a.download:hover { border-color: var(--accent); }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      .preview-grid, .info-grid, .logs { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand"><span class="brand-badge"></span> Face Avatar Studio</div>
    <div class="status-row">
      <span class="chip" id="chipCamera">Camera: -</span>
      <span class="chip" id="chipMic">Mic: -</span>
      <span class="chip" id="chipRecord">Idle</span>
      <span class="chip" id="chipFace">Face: -</span>
    </div>
  </header>

  <div class="layout">
    <aside class="panel sidebar">
      <div>
        <div class="section-title">Devices</div>
        <div class="field">
          <label for="cameraSelect">Camera</label>
          <select id="cameraSelect"></select>
        </div>
        <div class="field">
          <label for="micSelect">Microphone</label>
          <select id="micSelect"></select>
        </div>
        <div class="button-row">
          <button id="applyCameraBtn">Apply Camera</button>
          <button id="reloadBtn">Reload</button>
        </div>
      </div>

      <div>
        <div class="section-title">Recording</div>
        <div class="field">
          <label for="outputRoot">Save folder</label>
          <input id="outputRoot" type="text" />
        </div>
        <div class="field">
          <label for="speechLang">Speech language</label>
          <select id="speechLang">
            <option value="auto">Auto</option>
            <option value="zh">Chinese</option>
            <option value="en">English</option>
          </select>
        </div>
        <div class="field">
          <label for="sttModel">STT model</label>
          <select id="sttModel">
            <option value="tiny">tiny</option>
            <option value="base">base</option>
            <option value="small" selected>small</option>
            <option value="medium">medium</option>
          </select>
        </div>
        <div class="button-row">
          <button class="primary" id="startBtn">Start Recording</button>
          <button class="danger" id="stopBtn" disabled>Stop</button>
        </div>
      </div>

      <div>
        <div class="section-title">Session</div>
        <div class="stat"><div class="k">Output root</div><div class="v" id="statOutputRoot">-</div></div>
        <div style="height: 10px"></div>
        <div class="stat"><div class="k">Session folder</div><div class="v" id="statSession">-</div></div>
      </div>
    </aside>

    <main class="main">
      <section class="panel view-card">
        <div class="view-head">
          <span>Live Views</span>
          <span id="liveMeta">waiting</span>
        </div>
        <div class="preview-grid">
          <img id="cameraFeed" class="preview" alt="camera feed" src="/stream/camera.mjpg" />
          <img id="avatarFeed" class="preview" alt="avatar feed" src="/stream/avatar.mjpg" />
        </div>
        <div class="info-grid">
          <div class="stat"><div class="k">Frame</div><div class="v" id="statFrame">-</div></div>
          <div class="stat"><div class="k">Time</div><div class="v" id="statTime">-</div></div>
          <div class="stat"><div class="k">Last output</div><div class="v" id="statLastOutput">-</div></div>
          <div class="stat"><div class="k">Status</div><div class="v" id="statState">-</div></div>
        </div>
      </section>

      <section class="panel logs">
        <div class="logbox">
          <div class="section-title">Logs</div>
          <pre id="logBox"></pre>
        </div>
        <div class="filebox">
          <div class="section-title">Files</div>
          <div class="links" id="fileLinks"></div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const state = {
      cameras: [],
      microphones: [],
      lastSession: null,
    };

    const ids = {
      cameraSelect: document.getElementById('cameraSelect'),
      micSelect: document.getElementById('micSelect'),
      outputRoot: document.getElementById('outputRoot'),
      speechLang: document.getElementById('speechLang'),
      sttModel: document.getElementById('sttModel'),
      startBtn: document.getElementById('startBtn'),
      stopBtn: document.getElementById('stopBtn'),
      applyCameraBtn: document.getElementById('applyCameraBtn'),
      reloadBtn: document.getElementById('reloadBtn'),
      chipCamera: document.getElementById('chipCamera'),
      chipMic: document.getElementById('chipMic'),
      chipRecord: document.getElementById('chipRecord'),
      chipFace: document.getElementById('chipFace'),
      liveMeta: document.getElementById('liveMeta'),
      statFrame: document.getElementById('statFrame'),
      statTime: document.getElementById('statTime'),
      statLastOutput: document.getElementById('statLastOutput'),
      statState: document.getElementById('statState'),
      statOutputRoot: document.getElementById('statOutputRoot'),
      statSession: document.getElementById('statSession'),
      logBox: document.getElementById('logBox'),
      fileLinks: document.getElementById('fileLinks'),
    };

    function fillSelect(select, items, currentIndex) {
      select.innerHTML = '';
      if (!items.length) {
        const opt = document.createElement('option');
        opt.value = '-1';
        opt.textContent = 'None available';
        select.appendChild(opt);
        return;
      }
      items.forEach((item) => {
        const opt = document.createElement('option');
        opt.value = item.index;
        opt.textContent = `${item.index}: ${item.label}`;
        if (item.index === currentIndex) opt.selected = true;
        select.appendChild(opt);
      });
    }

    function setText(id, value) {
      ids[id].textContent = value ?? '-';
    }

    function renderFiles(snapshot) {
      const links = [];
      const last = snapshot.last_outputs;
      if (last) {
        if (last.video_path) links.push(['Merged video', '/api/download/video']);
        if (last.audio_path) links.push(['Audio WAV', '/api/download/audio']);
        if (last.timeline_csv_path) links.push(['Timeline CSV', '/api/download/timeline']);
        if (last.transcript_csv_path) links.push(['Transcript CSV', '/api/download/transcript']);
        if (last.manifest_path) links.push(['Manifest JSON', '/api/download/manifest']);
      }
      ids.fileLinks.innerHTML = links.length
        ? links.map(([label, href]) => `<a class="download" href="${href}">${label}</a>`).join('')
        : '<div style="color:#93a4b7;">No exported session yet.</div>';
    }

    async function loadDevices() {
      const res = await fetch('/api/devices');
      const data = await res.json();
      state.cameras = data.cameras;
      state.microphones = data.microphones;
      fillSelect(ids.cameraSelect, state.cameras, data.current_camera_index);
      fillSelect(ids.micSelect, state.microphones, data.current_mic_index);
      ids.outputRoot.value = data.output_root;
      ids.outputRoot.placeholder = data.output_root;
      setText('statOutputRoot', data.output_root);
    }

    async function applyCamera() {
      const cameraIndex = Number(ids.cameraSelect.value);
      const res = await fetch('/api/camera', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({camera_index: cameraIndex}),
      });
      if (!res.ok) {
        alert((await res.json()).detail || 'Failed to switch camera');
      }
    }

    async function startRecording() {
      const payload = {
        output_root: ids.outputRoot.value.trim(),
        mic_index: Number(ids.micSelect.value),
        mic_name: ids.micSelect.selectedOptions[0]?.textContent || '',
        speech_language: ids.speechLang.value,
        stt_model: ids.sttModel.value,
      };
      const res = await fetch('/api/record/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        alert((await res.json()).detail || 'Failed to start recording');
        return;
      }
      await refreshState();
    }

    async function stopRecording() {
      const res = await fetch('/api/record/stop', {method: 'POST'});
      if (!res.ok) {
        alert((await res.json()).detail || 'Failed to stop recording');
      }
      await refreshState();
    }

    async function refreshState() {
      const res = await fetch('/api/state');
      const snapshot = await res.json();
      state.lastSession = snapshot;

      ids.chipCamera.textContent = snapshot.camera_index === null ? 'Camera: -' : `Camera: ${snapshot.camera_index}`;
      ids.chipMic.textContent = snapshot.mic_name ? `Mic: ${snapshot.mic_name}` : 'Mic: -';
      ids.chipRecord.textContent = snapshot.recording ? 'Recording' : snapshot.exporting ? 'Exporting' : 'Idle';
      ids.chipRecord.className = 'chip ' + (snapshot.recording ? 'live' : snapshot.exporting ? 'warn' : '');
      ids.chipFace.textContent = snapshot.face_detected === null ? 'Face: -' : `Face: ${snapshot.face_detected ? 'detected' : 'not found'}`;
      ids.liveMeta.textContent = snapshot.recording
        ? 'recording'
        : snapshot.exporting
          ? 'finalizing'
          : 'preview';

      setText('statFrame', snapshot.frame_index ?? '-');
      setText('statTime', snapshot.timestamp_sec === null ? '-' : `${snapshot.timestamp_sec.toFixed(2)} s`);
      setText('statState', snapshot.exporting ? 'exporting' : snapshot.recording ? 'recording' : 'ready');
      setText('statSession', snapshot.session_dir || '-');
      setText('statLastOutput', snapshot.last_outputs?.session_dir || '-');

      ids.logBox.textContent = snapshot.logs.join('\n');
      renderFiles(snapshot);

      ids.stopBtn.disabled = !(snapshot.recording || snapshot.exporting);
      ids.startBtn.disabled = snapshot.recording || snapshot.exporting;
    }

    ids.applyCameraBtn.addEventListener('click', applyCamera);
    ids.reloadBtn.addEventListener('click', async () => { await loadDevices(); await refreshState(); });
    ids.startBtn.addEventListener('click', startRecording);
    ids.stopBtn.addEventListener('click', stopRecording);

    (async () => {
      await loadDevices();
      await refreshState();
      setInterval(refreshState, 500);
    })();
  </script>
</body>
</html>
    """


def create_app() -> tuple[FastAPI, AppState]:
    state = AppState()
    app = FastAPI(title="Face Avatar Studio")

    @app.on_event("startup")
    def _startup() -> None:
        cameras = enumerate_cameras()
        if cameras:
            state.start_camera(cameras[0]["index"])
        else:
            state.add_log("No camera detected.")

    @app.on_event("shutdown")
    def _shutdown() -> None:
        if state.capture_worker is not None:
            state.capture_worker.stop()
            state.capture_worker.join(timeout=3)
        if state.recorder is not None and state.recorder.active:
            try:
                state.recorder.stop_capture()
            except Exception:
                pass

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(build_html())

    @app.get("/api/devices")
    def api_devices() -> JSONResponse:
        return JSONResponse(
            {
                "cameras": enumerate_cameras(),
                "microphones": enumerate_microphones(),
                "current_camera_index": state.current_camera_index,
                "current_mic_index": state.current_mic_index,
                "output_root": str(state.output_root),
            }
        )

    @app.get("/api/state")
    def api_state() -> JSONResponse:
        return JSONResponse(state.snapshot())

    @app.post("/api/camera")
    async def api_camera(request: Request) -> JSONResponse:
        payload = await request.json()
        camera_index = int(payload.get("camera_index", -1))
        if camera_index < 0:
            raise HTTPException(status_code=400, detail="Invalid camera index.")
        state.start_camera(camera_index)
        return JSONResponse({"ok": True})

    @app.post("/api/record/start")
    async def api_record_start(request: Request) -> JSONResponse:
        payload = await request.json()
        mic_index = int(payload.get("mic_index", -1))
        if mic_index < 0:
            raise HTTPException(status_code=400, detail="Invalid microphone index.")
        output_root = str(payload.get("output_root") or state.output_root)
        speech_language = str(payload.get("speech_language") or "auto")
        stt_model = str(payload.get("stt_model") or DEFAULT_STT_MODEL)
        mic_name = str(payload.get("mic_name") or "")
        return JSONResponse(
            state.start_recording(
                output_root=output_root,
                mic_index=mic_index,
                mic_name=mic_name,
                language=speech_language,
                stt_model=stt_model,
            )
        )

    @app.post("/api/record/stop")
    def api_record_stop() -> JSONResponse:
        state.stop_recording()
        return JSONResponse({"ok": True})

    @app.get("/stream/camera.mjpg")
    def camera_stream() -> StreamingResponse:
        def generate():
            boundary = b"--frame"
            while True:
                frame = state.latest_camera_jpeg
                if frame:
                    yield (
                        boundary
                        + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(frame)).encode("ascii")
                        + b"\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                time.sleep(1 / 15)

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/stream/avatar.mjpg")
    def avatar_stream() -> StreamingResponse:
        def generate():
            boundary = b"--frame"
            while True:
                frame = state.latest_avatar_jpeg
                if frame:
                    yield (
                        boundary
                        + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(frame)).encode("ascii")
                        + b"\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                time.sleep(1 / 12)

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/download/{kind}")
    def api_download(kind: str) -> Response:
        outputs = state.last_outputs
        if outputs is None:
            raise HTTPException(status_code=404, detail="No exported session yet.")

        mapping = {
            "video": outputs.merged_video_path,
            "audio": outputs.audio_path,
            "timeline": outputs.timeline_csv_path,
            "transcript": outputs.transcript_csv_path,
            "manifest": outputs.manifest_path,
        }
        path = mapping.get(kind)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="File not available.")

        return Response(
            content=path.read_bytes(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
        )

    return app, state


def run_server(host: str = "127.0.0.1", port: int = 8765) -> int:
    import uvicorn

    app, _ = create_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=False)
    thread.start()
    time.sleep(1.0)
    webbrowser.open(f"http://{host}:{port}")
    try:
        thread.join()
    except KeyboardInterrupt:
        server.should_exit = True
        thread.join(timeout=5)
    return 0


def main() -> int:
    return run_server()

