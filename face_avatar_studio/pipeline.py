from __future__ import annotations

import csv
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# GNM's frame-sized matrix operations are small. Letting OpenBLAS create 16
# workers for each one starves Tk, camera capture, MediaPipe and OpenGL on the
# same machine, causing periodic "not responding" windows.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import h5py
import numpy as np
import requests
import sounddevice as sd
from imageio_ffmpeg import get_ffmpeg_exe
from scipy.linalg import solve_toeplitz
from scipy.io import wavfile
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciRotation
# VTK is intentionally loaded only by the optional off-screen renderer.  Its
# Windows OpenGL bootstrap can block or crash on some machines, while tracking,
# recording and CSV export do not need it at all.
_vtk_loaded = False
_vtk_error: Exception | None = None


def _load_vtk() -> None:
    global _vtk_loaded, _vtk_error
    if _vtk_loaded:
        return
    if _vtk_error is not None:
        raise RuntimeError("VTK is unavailable") from _vtk_error
    try:
        from vtk import (
            vtkActor,
            vtkCellArray,
            vtkFileOutputWindow,
            vtkPoints,
            vtkPolyData,
            vtkPolyDataMapper,
            vtkOutputWindow,
            vtkRenderWindow,
            vtkRenderer,
            vtkWindowToImageFilter,
        )
        from vtk.util import numpy_support
        from vtk.util.numpy_support import vtk_to_numpy

        globals().update(
            vtkActor=vtkActor,
            vtkCellArray=vtkCellArray,
            vtkFileOutputWindow=vtkFileOutputWindow,
            vtkPoints=vtkPoints,
            vtkPolyData=vtkPolyData,
            vtkPolyDataMapper=vtkPolyDataMapper,
            vtkOutputWindow=vtkOutputWindow,
            vtkRenderWindow=vtkRenderWindow,
            vtkRenderer=vtkRenderer,
            vtkWindowToImageFilter=vtkWindowToImageFilter,
            numpy_support=numpy_support,
            vtk_to_numpy=vtk_to_numpy,
        )
        _vtk_loaded = True
    except Exception as exc:
        _vtk_error = exc
        raise RuntimeError("VTK renderer could not be loaded") from exc

if getattr(sys, "frozen", False):
    APP_ROOT = Path(sys.executable).resolve().parent
else:
    APP_ROOT = Path(__file__).resolve().parent.parent

PROJECT_ROOT = APP_ROOT
EXTERNAL_GNM_DIR = PROJECT_ROOT / "external_GNM"
EXTERNAL_FACEVERSE_DIR = PROJECT_ROOT / "external_FaceVerse_v4"
FACEVERSE_DATA_DIR = EXTERNAL_FACEVERSE_DIR / "data"
FACEVERSE_MODEL_PATH = FACEVERSE_DATA_DIR / "faceverse_v4_2.npy"
FACEVERSE_NETWORK_PATH = FACEVERSE_DATA_DIR / "faceverse_resnet50.pth"
USER_CACHE_DIR = Path.home() / ".face_avatar_studio_cache"

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

DEFAULT_FRAME_SIZE = (960, 540)
DEFAULT_FPS = 30
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_STT_MODEL = "base"
FACE_TRACK_HOLD_SECONDS = 0.25


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resize_for_face_tracking(frame_bgr: np.ndarray) -> np.ndarray:
    """Downscale for inference without changing the camera aspect ratio."""
    height, width = frame_bgr.shape[:2]
    longest_side = max(width, height)
    if longest_side <= 640:
        return frame_bgr
    scale = 640.0 / max(longest_side, 1)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame_bgr, target, interpolation=cv2.INTER_AREA)


def _fit_frame_to_size(frame_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Fit a camera frame into a fixed recording canvas without stretching faces."""
    target_width, target_height = int(size[0]), int(size[1])
    height, width = frame_bgr.shape[:2]
    if width == target_width and height == target_height:
        return frame_bgr
    scale = min(target_width / max(width, 1), target_height / max(height, 1))
    fitted_width = max(1, round(width * scale))
    fitted_height = max(1, round(height * scale))
    fitted = cv2.resize(
        frame_bgr, (fitted_width, fitted_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros((target_height, target_width, 3), dtype=frame_bgr.dtype)
    left = (target_width - fitted_width) // 2
    top = (target_height - fitted_height) // 2
    canvas[top : top + fitted_height, left : left + fitted_width] = fitted
    return canvas


def faceverse_backend_status() -> tuple[bool, str]:
    """Report whether the optional FaceVerse v4 runtime can be selected."""
    missing: list[str] = []
    if not (EXTERNAL_FACEVERSE_DIR / "faceversev4" / "__init__.py").exists():
        missing.append("FaceVerse v4 代码")
    if not FACEVERSE_MODEL_PATH.exists():
        missing.append(FACEVERSE_MODEL_PATH.name)
    if not FACEVERSE_NETWORK_PATH.exists():
        missing.append(FACEVERSE_NETWORK_PATH.name)
    if missing:
        return False, "缺少 " + "、".join(missing)
    try:
        import torch
    except Exception as exc:
        return False, f"PyTorch 不可用：{exc}"
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    return True, f"FaceVerse v4 已就绪（{device}）"


def configure_vtk_output_window() -> None:
    try:
        _load_vtk()
        ensure_directory(USER_CACHE_DIR)
        output_window = vtkFileOutputWindow()
        output_window.SetFileName(str(USER_CACHE_DIR / "vtk_output.log"))
        vtkOutputWindow.SetInstance(output_window)
        vtkOutputWindow.SetGlobalWarningDisplay(False)
    except Exception:
        pass


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0"))
        written = 0
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
    tmp_path.replace(destination)
    return destination


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    if not rows and not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class RealtimeVisemeAnalyzer:
    """Low-latency phoneme-family front end for camera-assisted lip sync.

    Vowels come from LPC formants. Consonant families use closure, burst and
    fricative spectra, which gives the avatar an immediate response while the
    slower STT pass supplies exact Chinese/English phonemes for the CSV.
    """

    _VOWELS = {
        "A": (760.0, 1250.0),
        "E": (520.0, 1850.0),
        "I": (320.0, 2350.0),
        "O": (520.0, 900.0),
        "U": (330.0, 720.0),
    }
    _CONSONANTS = ("BMP", "FV", "TH", "TD", "KG", "CH", "SZ", "R")

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.sample_rate = int(sample_rate)
        self._buffer = np.zeros(2048, dtype=np.float32)
        self._values = {
            "audioVoiceActivity": 0.0,
            "audioVisemeA": 0.0,
            "audioVisemeE": 0.0,
            "audioVisemeI": 0.0,
            "audioVisemeO": 0.0,
            "audioVisemeU": 0.0,
            "audioVisemeClosed": 0.0,
            "audioVisemeFricative": 0.0,
            **{f"audioViseme{name}": 0.0 for name in self._CONSONANTS},
        }
        self._previous_voice = 0.0
        self._previous_rms = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = clamp01(value)
        return value * value * (3.0 - 2.0 * value)

    def _estimate_formants(
        self, frame: np.ndarray
    ) -> tuple[float, float, float | None] | None:
        signal = np.asarray(frame, dtype=np.float64)
        signal -= signal.mean()
        if float(np.sqrt(np.mean(signal * signal))) < 2e-4:
            return None
        signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
        signal *= np.hamming(len(signal))
        order = 12 if self.sample_rate <= 16000 else 16
        fft_size = 1 << int(np.ceil(np.log2(len(signal) * 2 - 1)))
        spectrum = np.fft.rfft(signal, n=fft_size)
        autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)
        autocorrelation = np.real(autocorrelation[: order + 1])
        if autocorrelation[0] <= 1e-9:
            return None
        autocorrelation[0] *= 1.0005
        try:
            coefficients = solve_toeplitz(
                autocorrelation[:-1], -autocorrelation[1:]
            )
        except Exception:
            return None
        roots = np.roots(np.concatenate(([1.0], coefficients)))
        roots = roots[np.imag(roots) >= 0.0]
        angles = np.arctan2(np.imag(roots), np.real(roots))
        frequencies = angles * self.sample_rate / (2.0 * np.pi)
        bandwidths = -0.5 * self.sample_rate / np.pi * np.log(
            np.maximum(np.abs(roots), 1e-8)
        )
        valid = frequencies[
            (frequencies >= 180.0)
            & (frequencies <= 3500.0)
            & (bandwidths <= 750.0)
        ]
        valid.sort()
        if len(valid) < 2:
            return None
        third = float(valid[2]) if len(valid) >= 3 else None
        return float(valid[0]), float(valid[1]), third

    def process(self, samples: np.ndarray) -> None:
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return
        if mono.size >= len(self._buffer):
            self._buffer[:] = mono[-len(self._buffer):]
        else:
            self._buffer[:-mono.size] = self._buffer[mono.size:]
            self._buffer[-mono.size:] = mono

        rms = float(np.sqrt(np.mean(mono * mono)))
        dbfs = 20.0 * np.log10(max(rms, 1e-7))
        voice = self._smoothstep((dbfs + 56.0) / 24.0)
        frame = self._buffer[-1024:]
        formants = self._estimate_formants(frame) if voice > 0.04 else None

        targets = {name: 0.0 for name in self._VOWELS}
        if formants is not None:
            f1, f2, _f3 = formants
            for name, (center_f1, center_f2) in self._VOWELS.items():
                distance = (
                    ((f1 - center_f1) / 260.0) ** 2
                    + ((f2 - center_f2) / 560.0) ** 2
                )
                targets[name] = float(np.exp(-0.5 * distance))
            raw_confidence = max(targets.values(), default=0.0)
            for name in targets:
                targets[name] = targets[name] ** 2.2
            total = sum(targets.values())
            if total > 1e-6:
                confidence = min(1.0, raw_confidence * 1.45)
                for name in targets:
                    targets[name] = targets[name] / total * confidence * voice

        windowed = frame * np.hanning(len(frame))
        magnitude = np.abs(np.fft.rfft(windowed))
        frequencies = np.fft.rfftfreq(len(frame), 1.0 / self.sample_rate)
        total_energy = float(np.sum(magnitude * magnitude)) + 1e-9
        high_energy = float(np.sum(
            (magnitude[(frequencies >= 2800.0) & (frequencies <= 7000.0)]) ** 2
        ))
        high_ratio = high_energy / total_energy
        magnitude_sum = float(np.sum(magnitude)) + 1e-9
        spectral_centroid = float(np.sum(frequencies * magnitude) / magnitude_sum)
        useful = magnitude[1:] + 1e-8
        spectral_flatness = float(
            np.exp(np.mean(np.log(useful))) / max(np.mean(useful), 1e-8)
        )
        zero_crossing_rate = float(np.mean(frame[:-1] * frame[1:] < 0.0))
        vowel_confidence = max(targets.values(), default=0.0)
        fricative = voice * self._smoothstep((high_ratio - 0.08) / 0.34)
        fricative = max(
            fricative,
            voice
            * self._smoothstep((spectral_flatness - 0.06) / 0.30)
            * self._smoothstep((zero_crossing_rate - 0.04) / 0.24),
        )
        fricative *= 1.0 - 0.65 * vowel_confidence
        closure = 0.0
        if voice < 0.16 and self._previous_voice > 0.38:
            closure = min(1.0, (self._previous_voice - voice) * 1.8)
        onset = voice * self._smoothstep(
            (rms - self._previous_rms - 0.0015) / 0.024
        )
        self._previous_rms = 0.72 * rms + 0.28 * self._previous_rms
        self._previous_voice = 0.70 * voice + 0.30 * self._previous_voice

        def spectral_band(center: float, width: float) -> float:
            return float(np.exp(-0.5 * ((spectral_centroid - center) / width) ** 2))

        consonants = {name: 0.0 for name in self._CONSONANTS}
        if fricative > 0.01:
            fricative_bands = {
                "FV": spectral_band(1650.0, 900.0),
                "TH": spectral_band(2850.0, 1050.0),
                "CH": spectral_band(3550.0, 1050.0),
                "SZ": spectral_band(5350.0, 1300.0),
            }
            band_total = sum(value ** 2 for value in fricative_bands.values())
            if band_total > 1e-8:
                for name, value in fricative_bands.items():
                    consonants[name] = fricative * value ** 2 / band_total * 1.55

        # Plosive place is most visible at the release burst. Low-frequency
        # diffuse bursts are bilabial, mid bursts velar, and high bursts
        # alveolar. Closure itself strongly favors B/P/M.
        consonants["BMP"] = max(
            consonants["BMP"],
            0.92 * closure,
            onset * spectral_band(850.0, 850.0),
        )
        consonants["KG"] = max(
            consonants["KG"], onset * spectral_band(2300.0, 950.0)
        )
        consonants["TD"] = max(
            consonants["TD"], onset * spectral_band(4200.0, 1500.0)
        )
        if formants is not None:
            _f1, f2, f3 = formants
            if f3 is not None:
                rhotic = self._smoothstep((2250.0 - f3) / 520.0)
                rhotic *= self._smoothstep((1850.0 - f2) / 700.0)
                consonants["R"] = voice * rhotic * (1.0 - 0.55 * vowel_confidence)

        target_values = {
            "audioVoiceActivity": voice,
            **{f"audioViseme{name}": targets[name] for name in targets},
            "audioVisemeClosed": closure,
            "audioVisemeFricative": fricative,
            **{
                f"audioViseme{name}": clamp01(value)
                for name, value in consonants.items()
            },
        }
        with self._lock:
            for name, target in target_values.items():
                previous = self._values[name]
                alpha = 0.62 if target > previous else 0.28
                self._values[name] = clamp01(previous + alpha * (target - previous))

    def values(self) -> dict[str, float]:
        with self._lock:
            return dict(self._values)

    def reset(self) -> None:
        self._buffer.fill(0.0)
        self._previous_voice = 0.0
        self._previous_rms = 0.0
        with self._lock:
            for name in self._values:
                self._values[name] = 0.0


def _load_gnm_numpy():
    if str(EXTERNAL_GNM_DIR) not in sys.path and EXTERNAL_GNM_DIR.exists():
        sys.path.insert(0, str(EXTERNAL_GNM_DIR))
    from gnm.shape import gnm_numpy

    return gnm_numpy


@dataclass(slots=True)
class FramePacket:
    frame_index: int
    timestamp_sec: float
    frame_bgr: np.ndarray
    landmarks: np.ndarray | None
    blendshapes: dict[str, float]
    expression: np.ndarray
    rotations: np.ndarray
    translation: np.ndarray
    face_matrix: np.ndarray | None
    vertices: np.ndarray
    face_detected: bool = False
    face_tracking_hold: bool = False
    semantic_weights: dict[str, float] = field(default_factory=dict)
    tracking_error: str | None = None


@dataclass(slots=True)
class TranscriptSegment:
    index: int
    start: float
    end: float
    text: str
    words: list[dict[str, Any]] = field(default_factory=list)
    avg_logprob: float | None = None


@dataclass(slots=True)
class TranscriptionResult:
    language: str | None
    language_probability: float | None
    segments: list[TranscriptSegment]


_ARPABET_TO_VISEME = {
    "B": "BMP", "P": "BMP", "M": "BMP",
    "F": "FV", "V": "FV",
    "TH": "TH", "DH": "TH",
    "T": "TD", "D": "TD", "N": "TD", "L": "TD",
    "K": "KG", "G": "KG", "NG": "KG", "HH": "KG",
    "CH": "CH", "JH": "CH", "SH": "CH", "ZH": "CH",
    "S": "SZ", "Z": "SZ",
    "R": "R", "W": "U", "Y": "I",
    "AA": "A", "AE": "A", "AH": "A", "AW": "A", "AY": "A",
    "EH": "E", "EY": "E", "ER": "R",
    "IH": "I", "IY": "I",
    "AO": "O", "OW": "O", "OY": "O",
    "UH": "U", "UW": "U",
}

_PINYIN_INITIAL_TO_VISEME = {
    "b": "BMP", "p": "BMP", "m": "BMP",
    "f": "FV",
    "d": "TD", "t": "TD", "n": "TD", "l": "TD",
    "g": "KG", "k": "KG", "h": "KG",
    "j": "CH", "q": "CH", "x": "CH",
    "zh": "CH", "ch": "CH", "sh": "CH",
    "z": "SZ", "c": "SZ", "s": "SZ", "r": "R",
    "y": "I", "w": "U",
}


def phoneme_to_viseme(phoneme: str) -> str:
    """Map ARPAbet or pinyin-sized phonemes to the 12 visible mouth shapes."""
    phone = re.sub(r"\d", "", str(phoneme)).strip().upper()
    if phone in _ARPABET_TO_VISEME:
        return _ARPABET_TO_VISEME[phone]
    lower = phone.lower()
    if lower in _PINYIN_INITIAL_TO_VISEME:
        return _PINYIN_INITIAL_TO_VISEME[lower]
    if "a" in lower:
        return "A"
    if any(value in lower for value in ("e", "ê")):
        return "E"
    if any(value in lower for value in ("i", "y")):
        return "I"
    if "o" in lower:
        return "O"
    if any(value in lower for value in ("u", "ü", "v")):
        return "U"
    return "Closed"


def _pinyin_final_phonemes(final: str) -> list[str]:
    value = re.sub(r"\d", "", final.lower()).replace("u:", "ü")
    ending = ""
    if value.endswith("ng"):
        value, ending = value[:-2], "ng"
    elif value.endswith("n"):
        value, ending = value[:-1], "n"
    phones: list[str] = []
    for character in value:
        if character in "a":
            phone = "a"
        elif character in "eê":
            phone = "e"
        elif character in "i":
            phone = "i"
        elif character in "o":
            phone = "o"
        elif character in "uüv":
            phone = "u"
        else:
            continue
        if not phones or phones[-1] != phone:
            phones.append(phone)
    if ending:
        phones.append(ending)
    return phones or ([value] if value else [])


def text_to_phonemes(text: str, language: str | None = None) -> list[str]:
    """Phonemize mixed Chinese/English text without a network dependency."""
    output: list[str] = []
    english_dictionary = None
    for token in re.findall(r"[A-Za-z']+|[\u3400-\u9fff]", str(text)):
        if re.fullmatch(r"[A-Za-z']+", token):
            if english_dictionary is None:
                try:
                    import cmudict

                    english_dictionary = cmudict.dict()
                except Exception:
                    english_dictionary = {}
            pronunciations = english_dictionary.get(token.lower(), [])
            if pronunciations:
                output.extend(re.sub(r"\d", "", phone) for phone in pronunciations[0])
            else:
                output.extend(character.upper() for character in token if character.isalpha())
            continue
        try:
            from pypinyin import Style, pinyin

            initial = pinyin(token, style=Style.INITIALS, strict=False)[0][0]
            final = pinyin(token, style=Style.FINALS, strict=False)[0][0]
        except Exception:
            initial, final = "", ""
        if initial:
            output.append(initial)
        output.extend(_pinyin_final_phonemes(final))
    return [phone for phone in output if phone]


def build_phoneme_timeline(
    segments: list[TranscriptSegment], language: str | None = None
) -> list[dict[str, Any]]:
    """Distribute phonemes over Whisper word timestamps for frame CSV export."""
    events: list[dict[str, Any]] = []
    for segment in segments:
        timed_words = [word for word in segment.words if isinstance(word, dict)]
        if not timed_words:
            timed_words = [{
                "word": segment.text,
                "start": segment.start,
                "end": segment.end,
            }]
        for word in timed_words:
            text = str(word.get("word", "")).strip()
            phones = text_to_phonemes(text, language)
            if not phones:
                continue
            start = float(word.get("start", segment.start) or segment.start)
            end = float(word.get("end", segment.end) or segment.end)
            end = max(end, start + 0.04 * len(phones))
            duration = (end - start) / len(phones)
            for index, phone in enumerate(phones):
                phone_start = start + duration * index
                phone_end = start + duration * (index + 1)
                events.append({
                    "word": text,
                    "phoneme": phone,
                    "viseme": phoneme_to_viseme(phone),
                    "start": phone_start,
                    "end": phone_end,
                })
    return sorted(events, key=lambda event: float(event["start"]))


@dataclass(slots=True)
class RecordingOutputs:
    session_dir: Path
    video_path: Path | None
    audio_path: Path | None
    merged_video_path: Path | None
    timeline_csv_path: Path | None
    transcript_csv_path: Path | None
    manifest_path: Path | None


class GnmAvatarDriver:
    """Thin wrapper around the upstream ``gnm_numpy.GNM`` head v3 model.

    把 MediaPipe FaceLandmarker 的 blendshape（52 个分类）+ facial_transformation_matrixes
    直接映射到 GNM 的 383 维 expression 与 4 个 joint 的 rotations/translation。

    **关键事实**：上游 GNM v3 head 模型的 ``model.expression_names`` 不是语义标签，
    而是带前缀的索引 dim：``left_eye_region_000…099``（100 维）、
    ``right_eye_region_000…099``（100 维）、``lower_face_region_000…149``（150 维）、
    ``tongue_mean`` + ``tongue_000…030``（32 维）、``pupils_000``（1 维），总共 383 维。

    上游代码里 *没有* "MediaPipe blendshape → 数字维度" 的官方映射函数。上游的
    ``semantic_sampler.ExpressionSampler`` 是离线 CVAE 解码器，从 latent + 20 类
    表情 one-hot 重建 383 维；它依赖 ``tensorflow.keras`` 与 ``expression_decoder_model.h5``，
    不是实时驱动路径。实时驱动路径上我们做的事就是：拿到 ``expression_names`` 的
    顺序与前缀，把 MediaPipe 52 个 blendshape 的 score 当作这 383 维的占位——

    * 每个 MediaPipe blendshape 关联一个上游前缀（``left_eye_region_*`` /
      ``right_eye_region_*`` / ``lower_face_region_*`` / ``tongue_*`` / ``pupils_*``）；
    * 按前缀分组，每个前缀的 N 个 MediaPipe score 顺序灌入该前缀的 N 个 dim，
      首个 dim 写第一个 score，第二个 dim 写第二个 score …，没填到的保持 0。

    这样既严格走 ``model.expression_names`` 的索引、不引入任何魔数区间，也避开
    了没有官方语义表的问题。``model.__call__`` 的完整管线
    （``vertex_positions_bind_pose`` → ``joint_positions_bind_pose`` →
    ``compute_pose_correctives`` → ``linear_blend_skinning``）全部走
    ``gnm_xnp.GNM.__call__``，不复制数学。
    """

    # (上游 expression_names 前缀, MediaPipe blendshape 名字)。
    # 上游前缀里：tongue_mean 是单数维；其它都是 ``_<N>`` 多个。
    # score 直接灌进对应前缀的下一个未占用 dim；超出的 score 丢弃。
    _BLENDSHAPE_TO_PREFIX: tuple[tuple[str, str], ...] = (
        # 左眼：blink / squint / wide / lookUp/Down/In/Out
        ("left_eye_region_", "eyeBlinkLeft"),
        ("left_eye_region_", "eyeSquintLeft"),
        ("left_eye_region_", "eyeWideLeft"),
        ("left_eye_region_", "eyeLookUpLeft"),
        ("left_eye_region_", "eyeLookDownLeft"),
        ("left_eye_region_", "eyeLookInLeft"),
        ("left_eye_region_", "eyeLookOutLeft"),
        # 右眼
        ("right_eye_region_", "eyeBlinkRight"),
        ("right_eye_region_", "eyeSquintRight"),
        ("right_eye_region_", "eyeWideRight"),
        ("right_eye_region_", "eyeLookUpRight"),
        ("right_eye_region_", "eyeLookDownRight"),
        ("right_eye_region_", "eyeLookInRight"),
        ("right_eye_region_", "eyeLookOutRight"),
        # 下半脸：jaw / mouth / cheek / nose / brow
        ("lower_face_region_", "jawOpen"),
        ("lower_face_region_", "jawLeft"),
        ("lower_face_region_", "jawRight"),
        ("lower_face_region_", "jawForward"),
        ("lower_face_region_", "mouthSmileLeft"),
        ("lower_face_region_", "mouthSmileRight"),
        ("lower_face_region_", "mouthFrownLeft"),
        ("lower_face_region_", "mouthFrownRight"),
        ("lower_face_region_", "mouthPucker"),
        ("lower_face_region_", "mouthFunnel"),
        ("lower_face_region_", "mouthStretchLeft"),
        ("lower_face_region_", "mouthStretchRight"),
        ("lower_face_region_", "mouthPressLeft"),
        ("lower_face_region_", "mouthPressRight"),
        ("lower_face_region_", "mouthUpperUpLeft"),
        ("lower_face_region_", "mouthUpperUpRight"),
        ("lower_face_region_", "mouthLowerDownLeft"),
        ("lower_face_region_", "mouthLowerDownRight"),
        ("lower_face_region_", "cheekSquintLeft"),
        ("lower_face_region_", "cheekSquintRight"),
        ("lower_face_region_", "noseSneerLeft"),
        ("lower_face_region_", "noseSneerRight"),
        ("lower_face_region_", "browInnerUp"),
        ("lower_face_region_", "browDownLeft"),
        ("lower_face_region_", "browDownRight"),
        ("lower_face_region_", "browOuterUpLeft"),
        ("lower_face_region_", "browOuterUpRight"),
        # 舌头
        ("tongue_mean", "tongueOut"),
        ("tongue_", "tongueOut"),
        # 瞳孔（仅 1 维，eyeWideLeft/Right 共用）
        ("pupils_", "eyeWideLeft"),
        ("pupils_", "eyeWideRight"),
    )

    # MediaPipe Face Mesh vertices corresponding to the conventional iBUG 68
    # topology used by GNM's bundled ``head_sparse_68`` landmark definition.
    _MEDIAPIPE_TO_IBUG_68 = np.asarray((
        234, 93, 132, 58, 172, 136, 150, 149, 152, 378, 379, 365, 397, 288,
        361, 323, 454,
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
        168, 6, 197, 195, 98, 97, 2, 326, 327,
        33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
        61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
        78, 82, 13, 312, 308, 317, 14, 87,
    ), dtype=np.int32)

    def __init__(self) -> None:
        gnm_numpy = _load_gnm_numpy()
        # 上游 factory 走的就是 GNM.from_local(...) → load_model_from_runfile →
        # _from_model_data，所有 24 个 GNM_DATA_ATTRIBUTES 直接读进 np。
        self.model = gnm_numpy.GNM.from_local(
            version=gnm_numpy.GNMMajorVersion.V3,
            variant=gnm_numpy.GNMVariant.HEAD,
        )
        self.identity = np.zeros(self.model.identity_dim, dtype=np.float32)

        # 帧间 EMA 平滑状态（见 evaluate 内的 _smooth_* 调用）。
        self._smooth_expr_state: np.ndarray | None = None
        self._smooth_rot_state: np.ndarray | None = None
        self._smooth_transl_state: np.ndarray | None = None

        # 上游 model.expression_names 是 383 个带前缀的索引维。用前缀查询，不依赖
        # 任何硬编码区间偏移。
        self.expression_names: list[str] = [str(n) for n in self.model.expression_names]
        self._indices_by_prefix: dict[str, list[int]] = {}
        for idx, name in enumerate(self.expression_names):
            if name == "tongue_mean":
                self._indices_by_prefix.setdefault("tongue_mean", []).append(idx)
                continue
            for prefix in (
                "left_eye_region_",
                "right_eye_region_",
                "lower_face_region_",
                "tongue_",
                "pupils_",
            ):
                if name.startswith(prefix):
                    self._indices_by_prefix.setdefault(prefix, []).append(idx)
                    break

        # joint 名字 -> 索引。head v3 只有 4 个 joint：neck / head / left_eye / right_eye。
        # 用 joint_names 查询比硬编码 [0]=neck, [1]=head, [2]=left_eye, [3]=right_eye
        # 更接近上游代码风格（gnm_xnp 的 joint_transforms_world 也按 parent index 走
        # LBS，不假设顺序）。
        self._joint_index: dict[str, int] = {
            str(name): idx for idx, name in enumerate(self.model.joint_names)
        }
        self._semantic_regions = {
            name: np.asarray(self.model.vertex_group_indices(name), dtype=np.int32)
            for name in (
                "left_eye", "right_eye", "left_orbital_region", "right_orbital_region",
                "left_brow_region", "middle_brow_region", "right_brow_region",
                "left_zygomatic_region", "right_zygomatic_region",
                "left_infraorbital_region", "right_infraorbital_region",
                "left_cheek_region", "right_cheek_region", "nose_region",
                "upper_lip", "lower_lip", "upper_lip_region", "lower_lip_region",
                "chin_region", "mouth_sock", "tongue", "lower_teeth_and_gums",
            )
        }
        # Smooth spatial falloffs let small asymmetric actions move through
        # the dense surface without the hard seams caused by editing a raw
        # vertex-group membership mask.  These are computed once from the
        # neutral GNM surface and reused for every frame.
        self._continuous_weights = self._build_continuous_weights()
        decoder_path = (
            EXTERNAL_GNM_DIR / "gnm" / "shape" / "data" / "semantic_sampler"
            / "expression_decoder_model.h5"
        )
        self._expression_decoder_weights: list[tuple[np.ndarray, np.ndarray]] = []
        with h5py.File(decoder_path, "r") as decoder_file:
            weights_root = decoder_file["model_weights"]
            for layer_name in ("dense_13", "dense_14", "dense_15", "dense_16", "dense_17"):
                layer = weights_root[layer_name][layer_name]
                self._expression_decoder_weights.append((
                    np.asarray(layer["kernel:0"], dtype=np.float32),
                    np.asarray(layer["bias:0"], dtype=np.float32),
                ))
        self._decoder_neutral = self._decode_semantic_labels(np.zeros(20, dtype=np.float32))
        prototypes = []
        for class_index in range(20):
            one_hot = np.zeros(20, dtype=np.float32)
            one_hot[class_index] = 1.0
            prototypes.append(self._decode_semantic_labels(one_hot) - self._decoder_neutral)
        self._semantic_prototypes = np.asarray(prototypes, dtype=np.float32)
        self._expression_region_masks = {
            "left_eye": np.asarray([name.startswith("left_eye_region_") for name in self.expression_names], dtype=np.float32),
            "right_eye": np.asarray([name.startswith("right_eye_region_") for name in self.expression_names], dtype=np.float32),
            "lower_face": np.asarray([name.startswith("lower_face_region_") for name in self.expression_names], dtype=np.float32),
            "tongue": np.asarray([name == "tongue_mean" or name.startswith("tongue_") for name in self.expression_names], dtype=np.float32),
        }
        self._prepare_landmark_expression_fit()
        self.last_semantic_weights: dict[str, float] = {}

        # expression_basis 是 (E_dim, V, 3) 的 orthonormal basis：能量集中在每个
        # prefix 的前几个 dim（例如 lower_face_region_000 单独一根 dim 就能让 mesh
        # 大幅形变）。我们按 prefix 把 dim 按 basis 能量降序排好，MediaPipe score
        # 优先打到能量最高的几个 dim 上 —— 这是上游 ``semantic_sampler`` 没法
        # 跨帧重做的事，但靠 basis 自身的稀疏性是可行的。
        basis = np.asarray(self.model.expression_basis, dtype=np.float32)
        dim_energy = (basis * basis).sum(axis=(1, 2))
        self._ordered_slots_by_prefix: dict[str, list[int]] = {}
        for prefix, slots in self._indices_by_prefix.items():
            self._ordered_slots_by_prefix[prefix] = sorted(
                slots, key=lambda idx: -float(dim_energy[idx])
            )

        # upstream expression_basis 是 orthonormal basis（dim 之间近似正交）。
        # 实测：MediaPipe score ∈ [0, 1] 时单根 dim 只能贡献
        #   sqrt(dim_energy) 量级的 mesh 形变：
        #     lower_face_region_000 单独 = 1.0  → mesh max_abs 0.0068
        #     left_eye_region_000   单独 = 1.0  → mesh max_abs 0.0027
        # 要让眨眼/张嘴肉眼可见，必须给每个 prefix 配一个标量增益。
        #
        # 校准目标：MediaPipe score=1 → mesh max_abs 落在 0.05–0.10 区间（GNM
        # 官方 demo 在 head v3 上的"中等强度表情"水平）。
        #
        # 注意：增益偏大会让 MediaPipe 的单帧抖动（眨眼噪声、tongue/eye 的亚毫秒
        # 抖动）直接被放大成 mesh 的明显波纹/闪烁 —— 实测 20× 会把眨眼帧的能量
        # 推到 ±0.054 mesh max_abs（一帧一抖），看起来像"水波"。降到 6× / 3×
        # 仍然保留表情幅度（jawOpen 帧 max_abs ≈ 0.04），但抖动被压到肉眼几乎
        # 不可见。EMA 平滑（见 evaluate 里的 _smooth_expression）再削一层。
        #
        #   prefix            top1_energy  recommended_gain
        #   left_eye_region_  0.00251      6.0
        #   right_eye_region_ 0.00251      6.0
        #   lower_face_region_0.08431      3.0   （top1 能量已经较大，少放大）
        #   tongue_mean       0.02169      3.0
        #   tongue_           0.02904      3.0
        #   pupils_           0.00006      1.0   （pupils 几乎不带能量，保留原样）
        self._prefix_gain: dict[str, float] = {
            "left_eye_region_": 6.0,
            "right_eye_region_": 6.0,
            "lower_face_region_": 3.0,
            "tongue_mean": 3.0,
            "tongue_": 3.0,
            "pupils_": 1.0,
        }

        # 缓存每个 prefix 关联多少个 MediaPipe blendshape，build 时读一次。
        self._prefix_blendshape_count: dict[str, int] = {}
        for prefix, _mediapipe_name in self._BLENDSHAPE_TO_PREFIX:
            self._prefix_blendshape_count[prefix] = (
                self._prefix_blendshape_count.get(prefix, 0) + 1
            )

    def _build_continuous_weights(self) -> dict[str, np.ndarray]:
        vertices = np.asarray(self.model.template_vertex_positions, dtype=np.float32)
        specs = {
            "left_orbital": ("left_orbital_region", 0.010),
            "right_orbital": ("right_orbital_region", 0.010),
            "left_cheek": ("left_cheek_region", 0.018),
            "right_cheek": ("right_cheek_region", 0.018),
            "left_zygomatic": ("left_zygomatic_region", 0.013),
            "right_zygomatic": ("right_zygomatic_region", 0.013),
            "upper_lip": ("upper_lip_region", 0.008),
            "lower_lip": ("lower_lip_region", 0.008),
            "chin": ("chin_region", 0.014),
            "mouth": ("mouth_sock", 0.012),
        }
        weights: dict[str, np.ndarray] = {}
        for name, (region_name, sigma) in specs.items():
            points = vertices[self._semantic_regions[region_name]]
            distances, _ = cKDTree(points).query(vertices, workers=1)
            weights[name] = np.exp(
                -0.5 * (np.asarray(distances, dtype=np.float32) / sigma) ** 2
            ).astype(np.float32)
        return weights

    @staticmethod
    def _clamp01(value: float) -> float:
        return clamp01(value)

    def _prepare_landmark_expression_fit(self) -> None:
        """Precompute ridge projectors from iBUG landmarks to GNM coefficients."""
        landmark_path = (
            EXTERNAL_GNM_DIR / "gnm" / "shape" / "data" / "landmarks"
            / "head_sparse_68.txt"
        )
        landmark_data = np.loadtxt(landmark_path)
        triangle_indices = landmark_data[:, ::2].astype(np.int32)
        barycentric_weights = landmark_data[:, 1::2].astype(np.float32)

        basis = np.asarray(self.model.expression_basis, dtype=np.float32)
        landmark_basis = (
            basis[:, triangle_indices, :]
            * barycentric_weights[None, :, :, None]
        ).sum(axis=2)

        neutral_vertices = np.asarray(
            self.model(
                identity=self.identity,
                expression=np.zeros(self.model.expression_dim, dtype=np.float32),
                rotations=np.zeros((self.model.num_joints, 3), dtype=np.float32),
                translation=np.zeros(3, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        neutral_landmarks = (
            neutral_vertices[triangle_indices]
            * barycentric_weights[:, :, None]
        ).sum(axis=1)
        self._gnm_landmark_scale = float(
            np.linalg.norm(neutral_landmarks[36, :2] - neutral_landmarks[45, :2])
        )

        right_eye_points = np.asarray((*range(17, 22), *range(36, 42)), dtype=np.int32)
        left_eye_points = np.asarray((*range(22, 27), *range(42, 48)), dtype=np.int32)
        mouth_points = np.arange(48, 68, dtype=np.int32)
        region_specs = (
            ("right_eye_fit", "right_eye", right_eye_points, 0.0),
            ("left_eye_fit", "left_eye", left_eye_points, 0.0),
            # A 2-D PCA fit cannot distinguish jaw opening from lip rounding
            # when the camera sees a speaking face.  Writing that residual
            # into the whole lower-face basis was the source of the recent
            # false pucker and unstable mouth corners.  Keep the projector
            # available for diagnostics, but let the semantic retargeter own
            # the live mouth until a calibrated 3-D mouth fit is available.
            ("mouth_fit", "lower_face", mouth_points, 0.0),
        )
        self._landmark_fit_regions: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float]] = []
        for name, mask_name, point_indices, gain in region_specs:
            expression_indices = np.flatnonzero(
                self._expression_region_masks[mask_name]
            ).astype(np.int32)
            design = landmark_basis[expression_indices][:, point_indices, :2]
            design = design.reshape(len(expression_indices), -1).T.astype(np.float64)
            # A.T @ inv(A @ A.T + lambda I) is the repeated ridge solve. The
            # regularizer removes low-energy PCA modes that otherwise amplify
            # sub-pixel landmark noise into unstable lips and cheeks.
            regularization = 3e-6 if mask_name == "lower_face" else 1.5e-6
            gram = design @ design.T
            projector = design.T @ np.linalg.inv(
                gram + regularization * np.eye(len(gram), dtype=np.float64)
            )
            self._landmark_fit_regions.append((
                name,
                expression_indices,
                point_indices,
                projector.astype(np.float32),
                gain,
            ))

        self._landmark_neutral_samples: list[np.ndarray] = []
        self._landmark_neutral_shape: np.ndarray | None = None

    def _canonical_capture_shape(self, landmarks: np.ndarray | None) -> np.ndarray | None:
        if landmarks is None or len(landmarks) <= int(self._MEDIAPIPE_TO_IBUG_68.max()):
            return None
        shape = np.asarray(landmarks, dtype=np.float32)[
            self._MEDIAPIPE_TO_IBUG_68, :2
        ].copy()
        eye_a, eye_b = shape[36], shape[45]
        eye_vector = eye_b - eye_a
        eye_distance = float(np.linalg.norm(eye_vector))
        if eye_distance < 1e-5:
            return None
        center = 0.5 * (eye_a + eye_b)
        cosine = float(eye_vector[0] / eye_distance)
        sine = float(eye_vector[1] / eye_distance)
        centered = shape - center
        canonical = np.empty_like(centered)
        canonical[:, 0] = (cosine * centered[:, 0] + sine * centered[:, 1]) / eye_distance
        canonical[:, 1] = (sine * centered[:, 0] - cosine * centered[:, 1]) / eye_distance
        return canonical

    @staticmethod
    def _align_capture_shape(shape: np.ndarray, reference: np.ndarray) -> np.ndarray:
        stable = np.asarray((27, 28, 29, 30, 31, 35, 36, 39, 42, 45), dtype=np.int32)
        source = shape[stable]
        target = reference[stable]
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        source_zero = source - source_center
        target_zero = target - target_center
        u, singular_values, vt = np.linalg.svd(source_zero.T @ target_zero)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        denominator = max(float(np.sum(source_zero * source_zero)), 1e-8)
        scale = float(np.sum(singular_values) / denominator)
        return (shape - source_center) @ rotation * scale + target_center

    def _fit_landmark_expression(
        self,
        landmarks: np.ndarray | None,
        blendshapes: dict[str, float],
    ) -> np.ndarray:
        expression = np.zeros(self.model.expression_dim, dtype=np.float32)
        shape = self._canonical_capture_shape(landmarks)
        if shape is None:
            return expression

        if self._landmark_neutral_shape is None:
            calibration_actions = (
                "jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker",
                "mouthFunnel", "mouthStretchLeft", "mouthStretchRight",
            )
            action_level = max(
                (float(blendshapes.get(name, 0.0)) for name in calibration_actions),
                default=0.0,
            )
            if action_level >= 0.28:
                self.last_semantic_weights["fit_waiting_neutral"] = 1.0
                return expression
            self._landmark_neutral_samples.append(shape)
            self.last_semantic_weights["fit_calibration"] = min(
                1.0, len(self._landmark_neutral_samples) / 15.0
            )
            if len(self._landmark_neutral_samples) < 15:
                return expression
            self._landmark_neutral_shape = np.median(
                np.stack(self._landmark_neutral_samples, axis=0), axis=0
            ).astype(np.float32)
            self._landmark_neutral_samples.clear()

        self.last_semantic_weights["fit_calibration"] = 1.0

        neutral = self._landmark_neutral_shape
        aligned = self._align_capture_shape(shape, neutral)
        delta = (aligned - neutral) * self._gnm_landmark_scale
        # Remove the tiny residual translation measured on the rigid nose area.
        delta -= delta[27:36].mean(axis=0, keepdims=True)
        delta[np.abs(delta) < 0.00018] = 0.0

        for name, expression_indices, point_indices, projector, gain in self._landmark_fit_regions:
            target = delta[point_indices].reshape(-1)
            coefficients = projector @ target
            coefficients = np.clip(coefficients, -1.15, 1.15)
            norm = float(np.linalg.norm(coefficients))
            if norm > 2.7:
                # Large residuals usually mean the 2D landmark fit is seeing a
                # head-pose or correspondence error, not a valid local muscle
                # action. Fall back to the named blendshape mapping this frame.
                self.last_semantic_weights[f"{name}_rejected"] = 1.0
                continue
            expression[expression_indices] += coefficients.astype(np.float32) * gain
            self.last_semantic_weights[name] = min(1.0, norm / 4.0)

        neutral_actions = (
            "jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker",
            "mouthFunnel", "mouthStretchLeft", "mouthStretchRight",
            "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
            "browInnerUp", "browDownLeft", "browDownRight",
        )
        if max((float(blendshapes.get(name, 0.0)) for name in neutral_actions), default=0.0) < 0.10:
            self._landmark_neutral_shape = 0.998 * neutral + 0.002 * aligned
        return expression

    def build_expression_vector(
        self,
        blendshapes: dict[str, float],
        landmarks: np.ndarray | None = None,
    ) -> np.ndarray:
        """按 ``model.expression_names`` 顺序填充 383 维 expression 向量。

        实时驱动场景下没有时间跑 ``semantic_sampler.ExpressionSampler``（那是离线
        CVAE 解码器，从 latent + 20 类表情 one-hot 重建 383 维）。这里直接把
        MediaPipe 的 52 个 blendshape score 当作 GNM 表达向量的子集。

        上游 ``expression_basis``（E_dim, V, 3）是 orthonormal basis，能量非常稀疏
        （每根 dim 单独能量 ≤ 0.012，前缀总和 ≤ 0.17）。所以单个 score 写到单根
        dim 上时 mesh 形变会小到肉眼不可见（basis dim 不是单独原子表情，而是
        orthonormal basis 中的小片段）。

        策略：每个 MediaPipe blendshape 关联到上游 ``expression_names`` 的一个前缀
        （``left_eye_region_*`` / ``right_eye_region_*`` / ``lower_face_region_*`` /
        ``tongue_*`` / ``pupils_*``），按前缀分组后，每个 score 写到自己前缀里
        能量最高的一个空闲 dim 上 —— 这保持了 *每个 score 都对应到独立的 basis
        dim* 的语义，与上游 CVAE 解码后每个 dim 的随机激活更接近，不会把多个独立
        的表情动作压缩到一根 dim 上。
        """
        semantic_expression = self._build_partitioned_expression(blendshapes)
        fitted_expression = self._fit_landmark_expression(landmarks, blendshapes)
        return semantic_expression + fitted_expression

        # Retarget named MediaPipe actions through GNM's official 20-class CVAE
        # decoder. Its output is the model's native 383-dimensional expression.
        def value(name: str) -> float:
            raw = self._clamp01(blendshapes.get(name, 0.0))
            threshold = 0.12 if name.startswith(("mouthSmile", "eyeSquint")) else 0.08
            return max(0.0, (raw - threshold) / (1.0 - threshold))

        labels = np.zeros(20, dtype=np.float32)
        blink_left, blink_right = value("eyeBlinkLeft"), value("eyeBlinkRight")
        smile_left, smile_right = value("mouthSmileLeft"), value("mouthSmileRight")
        labels[0] = max(value("jawOpen"), value("eyeWideLeft"), value("eyeWideRight"))
        labels[1] = max(value("noseSneerLeft"), value("noseSneerRight"))
        labels[2] = value("mouthPucker")
        labels[3] = max(value("mouthPressLeft"), value("mouthPressRight"))
        labels[4] = max(value("mouthStretchLeft"), value("mouthStretchRight"))
        labels[5] = .5 * (smile_left + smile_right)
        labels[6] = max(
            .5 * (value("eyeSquintLeft") + value("eyeSquintRight")),
            min(blink_left, blink_right),
        )
        labels[7] = max(value("mouthLowerDownLeft"), value("mouthLowerDownRight"))
        labels[8] = value("cheekPuff")
        labels[9] = value("mouthFunnel")
        labels[10] = max(labels[5], labels[4])
        labels[11] = .5 * (value("mouthFrownLeft") + value("mouthFrownRight"))
        labels[12] = value("mouthPucker")
        labels[13] = blink_left * (1.0 - blink_right)
        labels[14] = blink_right * (1.0 - blink_left)
        labels[15] = value("mouthLeft")
        labels[16] = value("mouthRight")
        labels[17] = max(value("mouthRollUpper"), value("mouthRollLower"))
        labels[18] = max(value("noseSneerLeft"), value("noseSneerRight"))
        labels[19] = value("tongueOut")
        if float(labels.max()) < 0.025:
            return np.zeros(self.model.expression_dim, dtype=np.float32)
        expression = self._decode_semantic_labels(labels) - self._decoder_neutral
        return np.asarray(expression * 1.44, dtype=np.float32)

    def _build_partitioned_expression(self, blendshapes: dict[str, float]) -> np.ndarray:
        """Compose official GNM semantic prototypes in their matching regions."""
        def raw(name: str) -> float:
            return self._clamp01(blendshapes.get(name, 0.0))

        def active(name: str, threshold: float = 0.06) -> float:
            value = raw(name)
            return max(0.0, (value - threshold) / (1.0 - threshold))

        def eye_blink(raw_name: str, geometry_name: str, ratio_name: str) -> float:
            raw_value = active(raw_name, 0.14)
            ratio = raw(ratio_name)
            # A high MediaPipe blink score is frequently emitted while the eye
            # is visibly open. Let the measured eyelid aspect ratio veto that
            # false positive, especially on the subject's left eye.
            if ratio >= 0.300:
                return raw(geometry_name)
            eye_open = self._clamp01((ratio - 0.220) / 0.080)
            return max(raw(geometry_name), raw_value * (1.0 - eye_open))

        geometry_jaw = raw("geometryJawOpen")
        audio_voice = active("audioVoiceActivity", 0.06)
        viseme_a = raw("audioVisemeA")
        viseme_e = raw("audioVisemeE")
        viseme_i = raw("audioVisemeI")
        viseme_o = raw("audioVisemeO")
        viseme_u = raw("audioVisemeU")
        viseme_closed = raw("audioVisemeClosed")
        viseme_fricative = raw("audioVisemeFricative")
        viseme_bmp = raw("audioVisemeBMP")
        viseme_fv = raw("audioVisemeFV")
        viseme_th = raw("audioVisemeTH")
        viseme_td = raw("audioVisemeTD")
        viseme_kg = raw("audioVisemeKG")
        viseme_ch = raw("audioVisemeCH")
        viseme_sz = raw("audioVisemeSZ")
        viseme_r = raw("audioVisemeR")
        jaw = max(
            active("jawOpen", 0.045),
            0.86 * geometry_jaw,
            0.06 * audio_voice,
            0.72 * viseme_a,
            0.28 * viseme_e,
            0.17 * viseme_i,
            0.38 * viseme_o,
            0.10 * viseme_u,
            0.08 * viseme_fv,
            0.10 * viseme_th,
            0.13 * viseme_td,
            0.18 * viseme_kg,
            0.20 * viseme_ch,
            0.08 * viseme_sz,
            0.12 * viseme_r,
        )
        geometry_lift_gate = 0.0 if (
            raw("mouthPucker") >= 0.45 and raw("jawOpen") < 0.25
        ) else 1.0
        smile_left = max(
            active("mouthSmileLeft", 0.04),
            0.68 * raw("geometryMouthLiftLeft") * geometry_lift_gate,
        )
        smile_right = max(
            active("mouthSmileRight", 0.04),
            0.68 * raw("geometryMouthLiftRight") * geometry_lift_gate,
        )
        smile = 0.5 * (smile_left + smile_right)
        stretch = max(
            active("mouthStretchLeft", 0.08),
            active("mouthStretchRight", 0.08),
            0.48 * raw("geometryMouthWide"),
            0.66 * viseme_e,
            0.88 * viseme_i,
            0.18 * viseme_fricative,
            0.24 * viseme_fv,
            0.18 * viseme_th,
            0.16 * viseme_td,
            0.12 * viseme_ch,
            0.34 * viseme_sz,
        )
        blink_left = eye_blink("eyeBlinkLeft", "geometryBlinkLeft", "geometryEyeRatioLeft")
        blink_right = eye_blink("eyeBlinkRight", "geometryBlinkRight", "geometryEyeRatioRight")
        # Do not let FaceLandmarker report a squint while the measured eye is
        # wide open.  The geometric channel is more reliable for this model
        # because its neutral eyelid aperture is already relatively narrow.
        eye_left_closed = self._clamp01((0.345 - raw("geometryEyeRatioLeft")) / 0.125)
        eye_right_closed = self._clamp01((0.345 - raw("geometryEyeRatioRight")) / 0.125)
        squint_left = max(
            raw("geometrySquintLeft"),
            active("eyeSquintLeft", 0.05) * eye_left_closed,
        )
        squint_right = max(
            raw("geometrySquintRight"),
            active("eyeSquintRight", 0.05) * eye_right_closed,
        )
        # A blink is not a squint. Keep both channels independent so a
        # bilateral blink closes both eyelids instead of driving GNM squint.
        squint_left *= 1.0 - 0.75 * blink_left
        squint_right *= 1.0 - 0.75 * blink_right
        squint = 0.5 * (squint_left + squint_right)
        pucker_raw, funnel_raw = raw("mouthPucker"), raw("mouthFunnel")
        # MediaPipe commonly emits pucker/funnel as secondary speech scores.
        # Only accept an intentional, dominant gesture on a nearly closed mouth.
        pucker = max(0.30 * viseme_u, 0.08 * viseme_r)
        raw_jaw = raw("jawOpen")
        mouth_eye_ratio = raw("geometryMouthEyeRatio")
        lip_gap_ratio = raw("geometryLipGapRatio")
        if (
            audio_voice < 0.10
            and raw_jaw < 0.23
            and jaw < 0.28
            and pucker_raw >= 0.50
            and pucker_raw > raw_jaw + 0.24
            and lip_gap_ratio < 0.18
            and mouth_eye_ratio < 0.56
            and smile < 0.24
            and stretch < 0.24
        ):
            pucker = max(pucker, active("mouthPucker", 0.50))
        funnel = max(0.30 * viseme_o, 0.08 * viseme_ch)
        if (
            audio_voice < 0.10
            and raw_jaw < 0.23
            and jaw < 0.28
            and funnel_raw >= 0.48
            and funnel_raw > pucker_raw + 0.05
            and funnel_raw > raw_jaw + 0.24
            and lip_gap_ratio < 0.18
            and mouth_eye_ratio < 0.60
            and stretch < 0.30
        ):
            funnel = max(funnel, active("mouthFunnel", 0.42))

        cheek_left = max(active("cheekSquintLeft", 0.02), 0.82 * smile_left)
        cheek_right = max(active("cheekSquintRight", 0.02), 0.82 * smile_right)
        cheek_puff = active("cheekPuff", 0.10)
        nose_left = active("noseSneerLeft", 0.06)
        nose_right = active("noseSneerRight", 0.06)
        brow_up = active("browInnerUp", 0.05)
        brow_outer_left = active("browOuterUpLeft", 0.05)
        brow_outer_right = active("browOuterUpRight", 0.05)
        brow_down_left = active("browDownLeft", 0.06)
        brow_down_right = active("browDownRight", 0.06)
        mouth_press = max(
            0.5 * (active("mouthPressLeft", 0.08) + active("mouthPressRight", 0.08)),
            0.72 * viseme_closed,
            0.88 * viseme_bmp,
        )
        upper_lip = max(active("mouthUpperUpLeft", 0.06), active("mouthUpperUpRight", 0.06))
        lower_lip = max(
            active("mouthLowerDownLeft", 0.07),
            active("mouthLowerDownRight", 0.07),
            0.30 * viseme_fricative,
            0.48 * viseme_fv,
            0.20 * viseme_th,
        )
        lip_roll = max(active("mouthRollUpper", 0.10), active("mouthRollLower", 0.10))
        mouth_left = max(active("mouthLeft", 0.12), active("jawLeft", 0.16))
        mouth_right = max(active("mouthRight", 0.12), active("jawRight", 0.16))
        wide_left = max(active("eyeWideLeft", 0.08), raw("geometryEyeWideLeft"))
        wide_right = max(active("eyeWideRight", 0.08), raw("geometryEyeWideRight"))

        left_eye = self._expression_region_masks["left_eye"]
        right_eye = self._expression_region_masks["right_eye"]
        lower = self._expression_region_masks["lower_face"]
        expression = np.zeros(self.model.expression_dim, dtype=np.float32)
        # Blend surprise with smile-wide to get a clear but speech-like aperture.
        # Surprise alone is the rounded O mouth that caused false puckering.
        expression += self._semantic_prototypes[0] * lower * jaw * 1.72
        expression += self._semantic_prototypes[10] * lower * jaw * 0.44
        expression += self._semantic_prototypes[5] * lower * smile * 0.84
        expression += self._semantic_prototypes[10] * lower * stretch * 0.96
        expression += self._semantic_prototypes[6] * left_eye * squint_left
        expression += self._semantic_prototypes[6] * right_eye * squint_right
        expression += self._semantic_prototypes[13] * left_eye * blink_left * 1.18
        expression += self._semantic_prototypes[14] * right_eye * blink_right * 1.18
        expression += self._semantic_prototypes[0] * left_eye * wide_left * 0.62
        expression += self._semantic_prototypes[0] * right_eye * wide_right * 0.62
        # Keep smile-driven eye narrowing modest.  The GNM smile prototype
        # already carries an eyelid component; using it at full cheek gain
        # made a normal smile look like a bilateral squint.
        expression += self._semantic_prototypes[5] * left_eye * cheek_left * 0.38
        expression += self._semantic_prototypes[5] * right_eye * cheek_right * 0.38
        expression += self._semantic_prototypes[8] * cheek_puff * 0.92
        expression += self._semantic_prototypes[18] * max(nose_left, nose_right, upper_lip) * 0.88
        expression += self._semantic_prototypes[0] * left_eye * max(brow_up, brow_outer_left) * 0.88
        expression += self._semantic_prototypes[0] * right_eye * max(brow_up, brow_outer_right) * 0.88
        expression += self._semantic_prototypes[3] * left_eye * brow_down_left * 0.78
        expression += self._semantic_prototypes[3] * right_eye * brow_down_right * 0.78
        expression += self._semantic_prototypes[3] * lower * mouth_press * 0.42
        expression += self._semantic_prototypes[7] * lower * lower_lip * 0.42
        expression += self._semantic_prototypes[17] * lower * lip_roll * 0.54
        expression += self._semantic_prototypes[15] * lower * mouth_left * 0.62
        expression += self._semantic_prototypes[16] * lower * mouth_right * 0.62
        expression += self._semantic_prototypes[12] * lower * pucker
        expression += self._semantic_prototypes[9] * lower * funnel
        expression += self._semantic_prototypes[11] * lower * 0.5 * (
            active("mouthFrownLeft", 0.08) + active("mouthFrownRight", 0.08)
        )
        expression += self._semantic_prototypes[19] * self._expression_region_masks["tongue"] * active("tongueOut", 0.08)
        self.last_semantic_weights = {
            "jaw_open": jaw, "closed_smile": smile, "tooth_smile": stretch,
            "squint": squint, "blink_left": blink_left, "blink_right": blink_right,
            "cheek_left": cheek_left, "cheek_right": cheek_right,
            "cheek_puff": cheek_puff, "brow_up": brow_up,
            "brow_down_left": brow_down_left, "brow_down_right": brow_down_right,
            "nose_sneer": max(nose_left, nose_right),
            "eye_wide_left": wide_left, "eye_wide_right": wide_right,
            "lower_lip": lower_lip, "lip_roll": lip_roll,
            "mouth_left": mouth_left, "mouth_right": mouth_right,
            "pucker": pucker, "funnel": funnel,
            "audio_voice": audio_voice,
            "viseme_a": viseme_a, "viseme_e": viseme_e,
            "viseme_i": viseme_i, "viseme_o": viseme_o,
            "viseme_u": viseme_u, "viseme_closed": viseme_closed,
            "viseme_fricative": viseme_fricative,
            "viseme_bmp": viseme_bmp, "viseme_fv": viseme_fv,
            "viseme_th": viseme_th, "viseme_td": viseme_td,
            "viseme_kg": viseme_kg, "viseme_ch": viseme_ch,
            "viseme_sz": viseme_sz, "viseme_r": viseme_r,
            "mouth_eye_ratio": mouth_eye_ratio, "lip_gap_ratio": lip_gap_ratio,
        }
        return np.asarray(expression * 1.44, dtype=np.float32)

        expression = np.zeros(self.model.expression_dim, dtype=np.float32)

        # 按 prefix 收集 score，每个 prefix 内部用"下一个空闲 dim"指针：
        # 不同 MediaPipe 动作写到不同的 dim 上，避免互相覆盖。
        per_prefix_scores: dict[str, list[float]] = {}
        for prefix, mediapipe_name in self._BLENDSHAPE_TO_PREFIX:
            score = float(blendshapes.get(mediapipe_name, 0.0))
            if score <= 0.0:
                continue
            per_prefix_scores.setdefault(prefix, []).append(self._clamp01(score))

        for prefix, scores in per_prefix_scores.items():
            slots = self._ordered_slots_by_prefix.get(prefix, [])
            if not slots:
                continue
            # 把 prefix 增益 × 每个 score 后，写到该 prefix 能量最高的几个 dim 上
            # （按 prefix 内 score 个数取首段）。这是 *纯标量增益*，不复制上游 basis
            # 数学、不重排 basis 子空间 —— 上游 ``__call__`` 的 LBS / pose correctives
            # 完全不变。
            gain = float(self._prefix_gain.get(prefix, 1.0))
            for cursor, score in enumerate(scores):
                if cursor >= len(slots):
                    break
                expression[slots[cursor]] = self._clamp01(score) * gain
        return expression

    def _decode_semantic_labels(self, labels: np.ndarray) -> np.ndarray:
        """Run Google's bundled semantic GNM decoder using its saved weights."""
        values = np.concatenate((np.zeros(64, dtype=np.float32), labels.astype(np.float32)))
        last = len(self._expression_decoder_weights) - 1
        for index, (kernel, bias) in enumerate(self._expression_decoder_weights):
            values = values @ kernel + bias
            if index != last:
                values = np.maximum(values, 0.0)
        return np.asarray(values, dtype=np.float32)

    def _apply_semantic_deformations(
        self, vertices: np.ndarray, blendshapes: dict[str, float]
    ) -> np.ndarray:
        """Apply MediaPipe's named high-frequency actions to GNM face regions."""
        result = np.asarray(vertices, dtype=np.float32).copy()

        def score(name: str) -> float:
            return self._clamp01(blendshapes.get(name, 0.0))

        def scale_eye(region: str, eye: str, amount: float, gain: float) -> None:
            if amount <= 0:
                return
            eye_indices = self._semantic_regions[eye]
            center_y = float(result[eye_indices, 1].mean())
            factor = 1.0 + gain * amount
            weight = self._continuous_weights[region]
            result[:, 1] += (
                (result[:, 1] - center_y) * (factor - 1.0) * weight
            ).astype(np.float32)

        # Blink closes the eye opening; squint and wide use the same semantic
        # regions with smaller inverse adjustments.
        def eyelid_blink(raw_name: str, geometry_name: str, ratio_name: str) -> float:
            raw_value = max(0.0, (score(raw_name) - 0.14) / 0.86)
            ratio = score(ratio_name)
            if ratio >= 0.340:
                return 0.0
            # Blendshape blink is useful when the landmark contour is noisy,
            # but it must be gated by a visibly narrowing eyelid.
            eye_closed = self._clamp01((0.315 - ratio) / 0.105)
            return max(score(geometry_name), raw_value * eye_closed)

        left_blink = eyelid_blink(
            "eyeBlinkLeft", "geometryBlinkLeft", "geometryEyeRatioLeft"
        )
        right_blink = eyelid_blink(
            "eyeBlinkRight", "geometryBlinkRight", "geometryEyeRatioRight"
        )
        # Use a continuous falloff instead of scaling only the orbital group.
        # The latter leaves a visible ring-shaped depression around the eye.
        scale_eye("left_orbital", "left_eye", left_blink, -0.20)
        scale_eye("right_orbital", "right_eye", right_blink, -0.20)
        for region, eye, wide in (
            (
                "left_orbital", "left_eye",
                max(score("eyeWideLeft"), 0.62 * score("geometryEyeWideLeft")),
            ),
            (
                "right_orbital", "right_eye",
                max(score("eyeWideRight"), 0.62 * score("geometryEyeWideRight")),
            ),
        ):
            scale_eye(region, eye, wide, 0.12)

        # Mouth vertices are intentionally left to GNM's continuous expression
        # basis. Direct lip/chin offsets created the bumps and depressions seen
        # around the vermilion border when several channels overlapped.
        smile_left = max(
            0.0,
            (score("mouthSmileLeft") - 0.04) / 0.96,
            0.68 * score("geometryMouthLiftLeft") * (
                0.0 if score("mouthPucker") >= 0.45 and score("jawOpen") < 0.25 else 1.0
            ),
        )
        smile_right = max(
            0.0,
            (score("mouthSmileRight") - 0.04) / 0.96,
            0.68 * score("geometryMouthLiftRight") * (
                0.0 if score("mouthPucker") >= 0.45 and score("jawOpen") < 0.25 else 1.0
            ),
        )
        cheek_left = max(
            0.0, (score("cheekSquintLeft") - 0.02) / 0.98
        )
        cheek_right = max(
            0.0, (score("cheekSquintRight") - 0.02) / 0.98
        )
        cheek_left = max(cheek_left, 0.72 * smile_left)
        cheek_right = max(cheek_right, 0.72 * smile_right)
        left_cheek = self._continuous_weights["left_cheek"]
        right_cheek = self._continuous_weights["right_cheek"]
        left_zygomatic = self._continuous_weights["left_zygomatic"]
        right_zygomatic = self._continuous_weights["right_zygomatic"]
        # GNM +X is the viewer-left cheek. Lift each side independently and
        # draw it slightly toward its mouth corner for a visible zygomatic
        # response without translating a hard-edged group.
        result[:, 1] += left_cheek * (0.0050 * cheek_left)
        result[:, 1] += right_cheek * (0.0050 * cheek_right)
        result[:, 1] += left_zygomatic * (0.0075 * cheek_left)
        result[:, 1] += right_zygomatic * (0.0075 * cheek_right)
        result[:, 0] -= left_zygomatic * (0.0018 * cheek_left)
        result[:, 0] += right_zygomatic * (0.0018 * cheek_right)
        # A small outward displacement gives the zygomatic/cheek action a
        # readable volume change under the preview light. Keep it much lower
        # than the vertical lift so it reads as muscle, not a balloon.
        result[:, 2] += left_cheek * (0.0028 * cheek_left)
        result[:, 2] += right_cheek * (0.0028 * cheek_right)
        return result

    def pose_from_face_matrix(
        self,
        face_matrix: np.ndarray | None,
        blendshapes: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """MediaPipe facial_transformation_matrixes -> (rotations[J, 3], translation[3]).

        * ``neck`` / ``head`` joint 直接用 face_matrix 的旋转 + 平移驱动（与上游
          ``joint_transforms_world`` 完全一致，rotation 是 axis-angle、translation 是
          根节点平移）。
        * ``left_eye`` / ``right_eye`` joint 用 ``eyeLook*`` blendshape 推断 pitch/yaw。
          GNM 没有官方眼球 retarget，沿用启发式，但用 ``model.joint_names`` 索引。
        """
        rotations = np.zeros((self.model.num_joints, 3), dtype=np.float32)
        translation = np.zeros(3, dtype=np.float32)

        if face_matrix is not None:
            matrix = np.asarray(face_matrix, dtype=np.float32).reshape(4, 4)
            # SciRotation.from_matrix 走的是 SVD，每次都重算一遍；face_matrix 在
            # 没检测到脸 / 检测到但头部稳定时近似单位矩阵（rot_vec≈0），跳过 SVD
            # 既能省 CPU，也能避免 SVD 在接近奇异矩阵上抖出 ±0.001 rad 噪声 —— 这
            # 份噪声被 EMA 放过去之后 head/neck joint 每帧就会抖一下，肉眼看就是
            # "波纹"。低于 1e-3 的旋转角当 0 处理。
            rot_part = matrix[:3, :3]
            trans_part = matrix[:3, 3]
            trace = float(rot_part[0, 0] + rot_part[1, 1] + rot_part[2, 2])
            # trace=3 → 单位矩阵（无旋转）；trace 偏离 3 越远旋转越大。
            cos_angle = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
            angle = float(np.arccos(cos_angle))
            if angle < 1e-3:
                rot_vec = np.zeros(3, dtype=np.float32)
            else:
                rot_vec = SciRotation.from_matrix(rot_part).as_rotvec().astype(np.float32)
            neck_idx = self._joint_index.get("neck")
            head_idx = self._joint_index.get("head")
            if neck_idx is not None:
                rotations[neck_idx] = 0.35 * rot_vec
            if head_idx is not None:
                rotations[head_idx] = 0.65 * rot_vec
            # MediaPipe's canonical face transform translation is in
            # centimeters while GNM v3 uses meters.
            translation = (0.01 * trans_part).astype(np.float32)

        left_eye_idx = self._joint_index.get("left_eye")
        right_eye_idx = self._joint_index.get("right_eye")
        if left_eye_idx is not None:
            pitch = 0.6 * (
                self._clamp01(blendshapes.get("eyeLookUpLeft", 0.0))
                - self._clamp01(blendshapes.get("eyeLookDownLeft", 0.0))
            )
            yaw = 0.6 * (
                self._clamp01(blendshapes.get("eyeLookOutLeft", 0.0))
                - self._clamp01(blendshapes.get("eyeLookInLeft", 0.0))
            )
            rotations[left_eye_idx] = np.array([pitch, yaw, 0.0], dtype=np.float32)
        if right_eye_idx is not None:
            pitch = 0.6 * (
                self._clamp01(blendshapes.get("eyeLookUpRight", 0.0))
                - self._clamp01(blendshapes.get("eyeLookDownRight", 0.0))
            )
            yaw = 0.6 * (
                self._clamp01(blendshapes.get("eyeLookInRight", 0.0))
                - self._clamp01(blendshapes.get("eyeLookOutRight", 0.0))
            )
            rotations[right_eye_idx] = np.array([pitch, yaw, 0.0], dtype=np.float32)

        return rotations, translation

    def evaluate(
        self,
        blendshapes: dict[str, float],
        face_matrix: np.ndarray | None,
        landmarks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        expression = self.build_expression_vector(blendshapes, landmarks)
        rotations, translation = self.pose_from_face_matrix(face_matrix, blendshapes)
        # 帧间 EMA 平滑：MediaPipe 的眨眼/嘴部 blendshape 在每帧之间有亚毫秒级抖
        # 动；不做平滑的话 mesh 顶点的位移在每帧之间都会抖（用户反馈的"波纹/闪
        # 烁"）。α=0.35 让单帧突变压到 35% 以下，几帧内回归稳态。
        expression = self._smooth_expression(expression)
        rotations = self._smooth_rotations(rotations)
        translation = self._smooth_translation(translation)
        # 关键：直接走上游的 __call__；vertex_positions_bind_pose → joint_positions_bind_pose
        # → compute_pose_correctives → linear_blend_skinning 全在 gnm_xnp.GNM.__call__ 里。
        vertices = self.model(
            identity=self.identity,
            expression=expression,
            rotations=rotations,
            translation=translation,
        )
        vertices = self._apply_semantic_deformations(vertices, blendshapes)
        return (
            expression,
            rotations,
            translation,
            np.asarray(vertices, dtype=np.float32),
        )

    # -- 帧间 EMA 平滑 ---------------------------------------------------------
    # MediaPipe 输出的 blendshape / face_matrix 是按 VIDEO 模式逐帧计算的，本身
    # 已经有时间戳稳定化，但在实际驱动场景里（30 fps、CPU XNNPACK、单帧之间
    # 无 MediaPipe 内部 Kalman 接力）眨眼/嘴部 blendshape 仍会有 ±0.02-0.05 的
    # 抖动；这些抖动经过 ``_prefix_gain`` 放大后变成 ±0.1-0.3 mesh max_abs 振
    # 幅，肉眼看就是 mesh 在抖。把这一层用 EMA 压掉 —— 用户反馈的"头像闪
    # 烁/有波纹"基本就是它引起的。
    _SMOOTH_ALPHA_EXPR = 0.74       # Favor responsive speech and one-sided expressions.
    _SMOOTH_ALPHA_ROT = 0.45        # Keep head pose attached without visible lag.
    _SMOOTH_ALPHA_TRANSL = 0.45     # translation 平滑稍弱一点（头部位置不动感）
    _smooth_expr_state: np.ndarray | None = None
    _smooth_rot_state: np.ndarray | None = None
    _smooth_transl_state: np.ndarray | None = None

    def _smooth_expression(self, expression: np.ndarray) -> np.ndarray:
        prev = self._smooth_expr_state
        if prev is None or prev.shape != expression.shape or prev.dtype != expression.dtype:
            self._smooth_expr_state = expression.copy()
            return self._smooth_expr_state
        alpha = float(self._SMOOTH_ALPHA_EXPR)
        # in-place: prev = alpha*cur + (1-alpha)*prev
        np.multiply(expression, alpha, out=expression)
        np.multiply(prev, 1.0 - alpha, out=prev)
        expression += prev
        self._smooth_expr_state = expression
        return expression

    def _smooth_rotations(self, rotations: np.ndarray) -> np.ndarray:
        prev = self._smooth_rot_state
        if prev is None or prev.shape != rotations.shape or prev.dtype != rotations.dtype:
            self._smooth_rot_state = rotations.copy()
            return self._smooth_rot_state
        alpha = float(self._SMOOTH_ALPHA_ROT)
        np.multiply(rotations, alpha, out=rotations)
        np.multiply(prev, 1.0 - alpha, out=prev)
        rotations += prev
        self._smooth_rot_state = rotations
        return rotations

    def _smooth_translation(self, translation: np.ndarray) -> np.ndarray:
        prev = self._smooth_transl_state
        if prev is None or prev.shape != translation.shape or prev.dtype != translation.dtype:
            self._smooth_transl_state = translation.copy()
            return self._smooth_transl_state
        alpha = float(self._SMOOTH_ALPHA_TRANSL)
        np.multiply(translation, alpha, out=translation)
        np.multiply(prev, 1.0 - alpha, out=prev)
        translation += prev
        self._smooth_transl_state = translation
        return translation


class FaceVerseAvatarModelAdapter:
    """Expose FaceVerse's triangular mesh through the local renderer API."""

    def __init__(self, reconstructor: Any) -> None:
        data = reconstructor.fvd
        self.template_vertex_positions = np.asarray(
            data["meanshape"], dtype=np.float32
        ).reshape(-1, 3) / 100.0
        self.triangles = np.asarray(data["tri"], dtype=np.uint32)
        self.num_vertices = len(self.template_vertex_positions)
        texture = np.asarray(data.get("meantex"), dtype=np.float32).reshape(-1, 3)
        self.vertex_colors = np.clip(texture, 0.0, 255.0).astype(np.uint8)
        self.is_faceverse = True
        # FaceVerse stores a camera-facing head with Y/Z opposite to the
        # display convention used by our GNM preview renderers.  Applying two
        # axis flips preserves the winding (determinant +1), while turning the
        # native back/underside view into an upright front view.
        self.render_axis_signs = np.asarray((1.0, -1.0, -1.0), dtype=np.float32)

    def render_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """Convert native FaceVerse coordinates into preview coordinates."""
        values = np.asarray(vertices, dtype=np.float32)
        return np.ascontiguousarray(values * self.render_axis_signs, dtype=np.float32)

    def compute_vertex_normals(self, vertices: np.ndarray) -> np.ndarray:
        values = np.asarray(vertices, dtype=np.float32)
        triangles = self.triangles.astype(np.int32, copy=False)
        corners = values[triangles]
        face_normals = np.cross(
            corners[:, 1] - corners[:, 0],
            corners[:, 2] - corners[:, 0],
        )
        normals = np.zeros_like(values)
        np.add.at(normals, triangles[:, 0], face_normals)
        np.add.at(normals, triangles[:, 1], face_normals)
        np.add.at(normals, triangles[:, 2], face_normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-7)
        return normals


def _render_space_vertices(model: Any, vertices: np.ndarray) -> np.ndarray:
    """Return vertices in the coordinate system used by the preview camera."""
    converter = getattr(model, "render_vertices", None)
    if callable(converter):
        return converter(vertices)
    return np.ascontiguousarray(np.asarray(vertices, dtype=np.float32))


class FaceVerseAvatarDriver:
    """FaceVerse v4 ResNet50 inference and 621-D parameter smoothing."""

    def __init__(self) -> None:
        available, message = faceverse_backend_status()
        if not available:
            raise RuntimeError(
                f"FaceVerse 后端不可用：{message}。模型目录：{FACEVERSE_DATA_DIR}"
            )
        if str(EXTERNAL_FACEVERSE_DIR) not in sys.path:
            sys.path.insert(0, str(EXTERNAL_FACEVERSE_DIR))
        import torch
        from faceversev4 import FaceVerseRecon

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reconstructor = FaceVerseRecon(
            str(FACEVERSE_MODEL_PATH),
            str(FACEVERSE_NETWORK_PATH),
            self.device,
        )
        self.model = FaceVerseAvatarModelAdapter(self.reconstructor)
        self._smooth_coefficients: Any = None
        self._last_vertices = self.model.template_vertex_positions.copy()
        self.last_semantic_weights: dict[str, float] = {}
        names = self.reconstructor.fvd.get("exp_name_list", [])
        self.expression_names = [str(name) for name in names]

    def neutral_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        expression = np.zeros(self.reconstructor.exp_dims, dtype=np.float32)
        rotations = np.zeros((1, 3), dtype=np.float32)
        translation = np.zeros(3, dtype=np.float32)
        return expression, rotations, translation, self._last_vertices.copy()

    @staticmethod
    def _eye_coefficients(pixel_landmarks: np.ndarray) -> np.ndarray | None:
        if len(pixel_landmarks) <= 473:
            return None

        def eye(first: int, second: int, pupil: int) -> tuple[float, float]:
            vector = pixel_landmarks[first, :2] - pixel_landmarks[second, :2]
            distance = max(float(np.linalg.norm(vector)), 1e-6)
            direction = vector / distance
            center = 0.5 * (
                pixel_landmarks[first, :2] + pixel_landmarks[second, :2]
            )
            offset = pixel_landmarks[pupil, :2] - center
            horizontal = float(np.dot(offset, direction) / distance * 3.0)
            vertical = float(
                np.dot(offset, direction[[1, 0]]) / distance * -1.5
            )
            return vertical, horizontal

        left = eye(362, 263, 473)
        right = eye(33, 133, 468)
        return np.asarray((*left, *right), dtype=np.float32)

    def evaluate(
        self,
        frame_rgb: np.ndarray,
        landmarks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        height, width = frame_rgb.shape[:2]
        pixels = np.asarray(landmarks[:, :2], dtype=np.float32).copy()
        pixels[:, 0] *= width
        pixels[:, 1] *= height
        bbox = np.asarray([[
            float(np.min(pixels[:, 0])),
            float(np.min(pixels[:, 1])),
            float(np.max(pixels[:, 0])),
            float(np.max(pixels[:, 1])),
        ]], dtype=np.float32)
        coefficients, _bbox_list = self.reconstructor.process_imgs(frame_rgb, bbox)
        previous = self._smooth_coefficients
        if previous is not None:
            id_end = self.reconstructor.id_dims
            exp_end = id_end + self.reconstructor.exp_dims
            tex_end = exp_end + self.reconstructor.tex_dims
            smoothed = coefficients.clone()
            smoothed[:, :id_end] = 0.08 * coefficients[:, :id_end] + 0.92 * previous[:, :id_end]
            smoothed[:, id_end:exp_end] = 0.68 * coefficients[:, id_end:exp_end] + 0.32 * previous[:, id_end:exp_end]
            smoothed[:, exp_end:tex_end] = 0.05 * coefficients[:, exp_end:tex_end] + 0.95 * previous[:, exp_end:tex_end]
            smoothed[:, tex_end:] = 0.58 * coefficients[:, tex_end:] + 0.42 * previous[:, tex_end:]
            coefficients = smoothed
        eye_values = self._eye_coefficients(pixels)
        if eye_values is not None:
            coefficients[:, -4:] = self.torch.from_numpy(eye_values).to(
                coefficients.device
            )
        self._smooth_coefficients = coefficients.detach().clone()

        parts = self.reconstructor.split_coeffs_dict(coefficients)
        expression_tensor = parts["exp"].clone()
        expression_tensor[:, 14:16] = self.reconstructor.adjust_eyes(
            expression_tensor[:, 14:16]
        )
        expression_tensor[:, 49:50] = self.reconstructor.adjust_mouth(
            expression_tensor[:, 49:50]
        )
        vertices = self.reconstructor.get_vs(
            parts["id"], expression_tensor, parts["eyes"]
        )
        rotation_matrix = self.reconstructor.compute_rotation_matrix(parts["angle"])
        vertices = self.reconstructor.rigid_transform(
            vertices, rotation_matrix, parts["trans"]
        )
        vertices_np = vertices[0].detach().cpu().numpy().astype(np.float32)
        expression = parts["exp"][0].detach().cpu().numpy().astype(np.float32)
        rotations = parts["angle"].detach().cpu().numpy().astype(np.float32)
        translation = parts["trans"][0].detach().cpu().numpy().astype(np.float32)
        self._last_vertices = vertices_np
        strongest = np.argsort(np.abs(expression))[-12:][::-1]
        self.last_semantic_weights = {
            (
                self.expression_names[index]
                if index < len(self.expression_names)
                else f"faceverse_exp_{index:03d}"
            ): float(expression[index])
            for index in strongest
        }
        return expression, rotations, translation, vertices_np


class AvatarPreviewRenderer:
    """Off-screen VTK renderer for the GNM head mesh.

    VTK 这层只用来出图（offscreen 渲染 + WindowToImageFilter），不算 vertex normals。
    法向全部走上游 ``gnm_numpy.GNM.compute_vertex_normals``，避免每次 mesh 更新都重建
    VTK normals filter。
    """

    def __init__(self, avatar: GnmAvatarDriver, size: tuple[int, int] = (960, 540)) -> None:
        _load_vtk()
        configure_vtk_output_window()
        self.avatar = avatar
        self.size = size
        self._points = vtkPoints()
        self._polydata = vtkPolyData()
        self._mapper = vtkPolyDataMapper()
        self._actor = vtkActor()
        self._renderer = vtkRenderer()
        self._render_window = vtkRenderWindow()
        self._render_window.SetSize(int(size[0]), int(size[1]))
        self._render_window.OffScreenRenderingOn()
        self._render_window.AddRenderer(self._renderer)
        self._capture_filter = vtkWindowToImageFilter()
        self._capture_filter.SetInput(self._render_window)
        self._capture_filter.ReadFrontBufferOff()
        # 缓存上游算出的 vertex normals，update_vertices 时复用。
        self._normals: np.ndarray | None = None
        self._setup_pipeline()

    def _setup_pipeline(self) -> None:
        model = self.avatar.model
        vertices = _render_space_vertices(
            model, np.asarray(model.template_vertex_positions, dtype=np.float32)
        )
        triangles = np.asarray(model.triangles, dtype=np.int64)

        # 上游 compute_vertex_normals 一次算出来，比 vtkPolyDataNormals 更稳。
        self._normals = np.asarray(
            model.compute_vertex_normals(vertices), dtype=np.float32
        )

        self._renderer.SetBackground(0.08, 0.09, 0.11)
        self._points.SetData(numpy_support.numpy_to_vtk(vertices, deep=True))
        normal_array = numpy_support.numpy_to_vtk(self._normals, deep=True)
        normal_array.SetName("Normals")
        self._polydata.SetPoints(self._points)
        self._polydata.GetPointData().SetNormals(normal_array)
        faces = np.hstack(
            [np.full((triangles.shape[0], 1), 3, dtype=np.int64), triangles]
        ).ravel()
        cell_array = vtkCellArray()
        cell_array.SetCells(
            int(triangles.shape[0]),
            numpy_support.numpy_to_vtkIdTypeArray(faces, deep=True),
        )
        self._polydata.SetPolys(cell_array)
        self._mapper.SetInputData(self._polydata)
        self._actor.SetMapper(self._mapper)
        prop = self._actor.GetProperty()
        prop.SetColor(0.85, 0.78, 0.70)
        prop.SetSpecular(0.2)
        prop.SetSpecularPower(20)
        prop.SetAmbient(0.25)
        self._renderer.AddActor(self._actor)

        # 参考上游 render_gnm.get_look_at_world_to_camera：
        #   - look_at vertex group 用 ``hockey_mask``（脸主体轮廓）
        #   - forward 用 ``nose_region``，left/right 用 ``ears``
        #   - default camera_distance=2.0，target_fill_factor=0.4
        # 我们这套走 VTK 正交投影 + SetParallelScale，
        # 但 ``look_at`` / 坐标系 / 翻转 严格按上游来——GNM 头 v3 是
        # +Y 朝上的，相机默认放在 +Z 看 -Z（VTK 标准 look down -Z）。
        # 由于 VTK 默认 forward 是 -Z，上游 OpenCV/GL 是 +Z 朝相机，所以
        # 直接用 upstream vertex group centroid 作为 focal point，把相机摆
        # 在 ``centroid + (0, 0, +camera_distance)``。
        verts_for_cam = np.asarray(model.template_vertex_positions, dtype=np.float32)

        def _group_centroid(group_name: str) -> np.ndarray:
            # ``vertex_group_indices`` 接收 group name 列表，返回索引数组
            indices = model.vertex_group_indices(group_name)
            return verts_for_cam[indices].mean(axis=0)

        is_gnm = hasattr(model, "vertex_group_indices") and hasattr(model, "quads")
        if is_gnm:
            # 1. 计算 GNM look-at target（hockey_mask 中心）+ forward 向量
            target = _group_centroid("hockey_mask")
            nose = _group_centroid("nose_region")
            # vertex_group_mask 接收 ('ears', '&left') 这种 operator-then-group
            # 序列：& 把它和 ears 取交集（耳朵在左半边的部分）。
            left_indices = model.vertex_group_indices("ears", "&left")
            right_indices = model.vertex_group_indices("ears", "&right")
            left = verts_for_cam[left_indices].mean(axis=0)
            right = verts_for_cam[right_indices].mean(axis=0)
            back = 0.5 * (left + right)
            fwd = nose - back
            fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
            camera_distance = 2.0
            camera_position = target + camera_distance * fwd
            # 上游用 ``-fwd`` 当 view direction（相机朝目标看），
            # 在 VTK 里就是 SetPosition + SetFocalPoint。
            view_dir = target - camera_position
        else:
            # FaceVerse is already a full head mesh.  Its transformed display
            # coordinates use Y-up and face the VTK camera on +Z; no GNM-only
            # vertex groups are available here.
            mins = verts_for_cam.min(axis=0)
            maxs = verts_for_cam.max(axis=0)
            target = 0.5 * (mins + maxs)
            target[2] = float(np.median(verts_for_cam[:, 2]))
            camera_position = target + np.asarray((0.0, 0.0, 2.0), dtype=np.float32)
            view_dir = target - camera_position

        self._renderer.AddActor(self._actor)

        # 2. 先 AddActor 完，让 renderer 持有 actor —— ResetCamera 才
        #    能用 actor bounds。ResetCamera 在正交投影下默认按 actor 的
        #    bounding sphere 设 parallel scale，所以我们切正交 + ResetCamera
        #    后再调 scale / 朝向。
        camera = self._renderer.GetActiveCamera()
        # 3. 先设好 position / focal point，再 ResetCamera——ResetCamera 会
        #    按当前 position + focal point 的方向重算 bounds，但不会改
        #    position/focal，所以 view direction 仍然指向 hockey_mask 中心。
        camera.SetPosition(*camera_position.tolist())
        camera.SetFocalPoint(*target.tolist())
        # 4. 切到正交投影（这是关键，否则 ResetCamera 会按 perspective 设
        #    view angle，画面就是一张圆盘 + 透视畸变 = 用户看到的"白圆"）。
        camera.ParallelProjectionOn()
        # 5. ResetCameraClippingRange 用当前 position 算 near/far，再
        #    ResetCamera 按当前方向 fit actor 到 viewport。
        self._renderer.ResetCameraClippingRange()
        self._renderer.ResetCamera()
        # 6. 现在 ResetCamera 已经把 parallel scale 设成"刚好包住 actor
        #    bbox"——但 bbox 里包含后脑勺/耳朵，会把脸压到画面 30% 左右。
        #    用 mask 宽度手动算 parallel scale（target_fill_factor=0.4），
        #    让脸主体占 viewport 高度 ~70%（=1/0.4 * 0.244 ≈ 0.6 偏大，
        #    取 mask 高度 ~0.3 即可）。
        if is_gnm:
            mask_indices = model.vertex_group_indices("hockey_mask")
            mask_verts = verts_for_cam[mask_indices]
            x_span = float(np.ptp(mask_verts[:, 0]))
            y_span = float(np.ptp(mask_verts[:, 1]))
            viewport_height_world = max(y_span, x_span) / 0.7
        else:
            x_span = float(np.ptp(verts_for_cam[:, 0]))
            y_span = float(np.ptp(verts_for_cam[:, 1]))
            aspect = max(float(self.size[0]) / max(float(self.size[1]), 1.0), 1e-6)
            # Keep the complete FaceVerse head visible with a small margin.
            viewport_height_world = max(y_span, x_span / aspect) / 0.86
        # 相机并行 scale 直接就是 viewport 半高（world unit）。
        new_scale = viewport_height_world * 0.5
        camera.SetParallelScale(new_scale)
        # 7. 兜底再 ResetCameraClippingRange 一次，确保近/远平面包住
        #    deform 后的 vertices（平移 + rotation 后顶点可能超出原 bounds）。
        self._renderer.ResetCameraClippingRange()
        # 8. 触发一次 Render 把 pipeline 跑通。
        self._render_window.Render()
        self._camera_baseline_set = True
        # 留着下面这串作为调试时取参的入口；生产路径里只用上方的固定相机。
        _debug_camera = {
            "position": camera.GetPosition(),
            "focal": camera.GetFocalPoint(),
            "view_dir": view_dir.tolist(),
            "parallel_scale": camera.GetParallelScale(),
            "fov": camera.GetViewAngle(),
        }

    def update_vertices(self, vertices: np.ndarray) -> None:
        """用上游 ``gnm.compute_vertex_normals`` 一次性算好顶点+法向，再灌给 VTK。

        这样既避免每次更新都重建 vtkPolyDataNormals filter（VTK 这边多线程 +
        RebuildPipeline 容易踩坑），也保证 normals 与顶点来自同一个上游函数。
        """
        verts = _render_space_vertices(self.avatar.model, vertices)
        normals = np.asarray(
            self.avatar.model.compute_vertex_normals(verts), dtype=np.float32
        )
        self._normals = normals
        self._points.SetData(numpy_support.numpy_to_vtk(verts, deep=True))
        normal_array = numpy_support.numpy_to_vtk(normals, deep=True)
        normal_array.SetName("Normals")
        self._polydata.GetPointData().SetNormals(normal_array)
        self._polydata.Modified()
        self._points.Modified()

    def render_bgr(self) -> np.ndarray:
        self._render_window.Render()
        self._capture_filter.Modified()
        self._capture_filter.Update()
        output = self._capture_filter.GetOutput()
        scalars = output.GetPointData().GetScalars()
        if scalars is None:
            return np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        width, height, _ = output.GetDimensions()
        array = vtk_to_numpy(scalars).reshape(height, width, -1)
        rgb = array[:, :, :3]
        rgb = np.flipud(rgb).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def render_jpeg(self, quality: int = 85) -> bytes:
        image = self.render_bgr()
        success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return b""
        return encoded.tobytes()


def _lazy_import_mediapipe():
    """Lazy-import mediapipe; first call costs ~14s on Windows due to TF Lite +
    absl + XNNPACK DLL loads. Cache the module so the cost is paid once per
    process, in the *background* (caller decides when to pay)."""
    global _mediapipe_module
    cached = _mediapipe_module
    if cached is not None:
        return cached
    import mediapipe as mp  # ~14s on Windows; first call only

    _mediapipe_module = mp
    return mp


_mediapipe_module: Any = None
_mediapipe_lock = threading.Lock()


def ensure_mediapipe_loaded():
    """Public hook for warming mediapipe in a background thread.

    Returns immediately if already imported; otherwise imports on the calling
    thread (intended to be a worker). Catches exceptions so a mediapipe import
    failure doesn't kill the worker — the real init path will surface it.
    """
    try:
        _lazy_import_mediapipe()
    except Exception:
        pass


def _extract_blendshapes(result: Any) -> dict[str, float]:
    blendshapes: dict[str, float] = {}
    entries = getattr(result, "face_blendshapes", None) or []
    if not entries:
        return blendshapes
    first = entries[0]
    for category in first:
        name = getattr(category, "category_name", None) or getattr(category, "display_name", None)
        score = getattr(category, "score", None)
        if name is None or score is None:
            continue
        blendshapes[str(name)] = float(score)
    return blendshapes


def _extract_face_matrix(result: Any) -> np.ndarray | None:
    matrices = getattr(result, "facial_transformation_matrixes", None) or []
    if not matrices:
        return None
    matrix = np.asarray(matrices[0], dtype=np.float32)
    if matrix.size == 16:
        return matrix.reshape(4, 4)
    return None


def _extract_landmarks(result: Any) -> np.ndarray | None:
    landmarks = getattr(result, "face_landmarks", None) or []
    if not landmarks:
        return None
    first = landmarks[0]
    values = np.array([[lm.x, lm.y, lm.z] for lm in first], dtype=np.float32)
    return values


def _mediapipe_process_main(
    request_queue: Any,
    response_queue: Any,
    model_path: str,
) -> None:
    """Own MediaPipe in a child process so its first import cannot freeze Tk."""
    try:
        # mediapipe.tasks marks TensorFlow as an optional doc-generation
        # dependency, but importing it on Windows adds many seconds and a
        # large amount of memory. Tasks inference itself does not use it.
        sys.modules.setdefault("tensorflow", None)
        mp = _lazy_import_mediapipe()
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            # Camera exposure changes and small framing shifts are common in
            # desktop capture. Lower thresholds prevent those from becoming
            # an all-or-nothing face result while tracking is warming up.
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        # Pay the graph's one-time inference cost while the desktop window is
        # already usable, instead of charging it to the first camera frame.
        warmup_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        warmup_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=warmup_frame
        )
        landmarker.detect_for_video(warmup_image, 0)
        response_queue.put((-2, None, None, None, "ready"))
    except Exception as exc:
        response_queue.put((-1, None, None, None, str(exc)))
        return

    try:
        last_timestamp_ms = 0
        while True:
            request = request_queue.get()
            if request is None:
                return
            request_id, rgb_frame, timestamp_ms = request
            try:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = max(int(timestamp_ms), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(image, timestamp_ms)
                response_queue.put((
                    request_id,
                    _extract_blendshapes(result),
                    _extract_face_matrix(result),
                    _extract_landmarks(result),
                    None,
                ))
            except Exception as exc:
                response_queue.put((request_id, None, None, None, str(exc)))
    finally:
        try:
            landmarker.close()
        except Exception:
            pass


_MEDIAPIPE_SERVICE_LOCK = threading.Lock()
_MEDIAPIPE_SERVICE: tuple[Any, Any, Any] | None = None


def prewarm_face_landmarker() -> tuple[Any, Any, Any]:
    """Start the shared MediaPipe owner without constructing the GNM avatar."""
    global _MEDIAPIPE_SERVICE
    with _MEDIAPIPE_SERVICE_LOCK:
        if _MEDIAPIPE_SERVICE is not None:
            process, requests_queue, responses_queue = _MEDIAPIPE_SERVICE
            if process.is_alive():
                return process, requests_queue, responses_queue
            try:
                process.close()
            except Exception:
                pass
            _MEDIAPIPE_SERVICE = None

        model_dir = ensure_directory(USER_CACHE_DIR / "mediapipe")
        model_path = download_file(FACE_LANDMARKER_URL, model_dir / "face_landmarker.task")
        context = multiprocessing.get_context("spawn")
        requests_queue = context.Queue(maxsize=1)
        responses_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=_mediapipe_process_main,
            args=(requests_queue, responses_queue, str(model_path)),
            daemon=True,
            name="face-avatar-mediapipe",
        )
        process.start()
        _MEDIAPIPE_SERVICE = (process, requests_queue, responses_queue)
        return _MEDIAPIPE_SERVICE


def _derive_landmark_actions(
    landmarks: np.ndarray | None,
    blendshapes: dict[str, float] | None = None,
    frame_shape: tuple[int, ...] | None = None,
) -> dict[str, float]:
    """Measure high-frequency mouth and eyelid actions from Face Mesh geometry."""
    if landmarks is None or len(landmarks) <= 386:
        return {}

    points = np.asarray(landmarks[:, :2], dtype=np.float32)
    if frame_shape is not None and len(frame_shape) >= 2:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if height > 0:
            # MediaPipe landmarks are normalized independently by image width
            # and height. Convert them to equal physical units before computing
            # eye and mouth aspect ratios on a 16:9 camera frame.
            points = points.copy()
            points[:, 0] *= width / float(height)

    def distance(first: int, second: int) -> float:
        return float(np.linalg.norm(points[first] - points[second]))

    def smoothstep(value: float) -> float:
        value = clamp01(value)
        return value * value * (3.0 - 2.0 * value)

    mouth_width = max(distance(61, 291), 1e-6)
    mouth_ratio = distance(13, 14) / mouth_width
    lip_parted = smoothstep((mouth_ratio - 0.025) / 0.265)
    eye_distance = max(distance(33, 263), 1e-6)
    mouth_eye_ratio = clamp01(mouth_width / eye_distance)
    mouth_wide = smoothstep((mouth_eye_ratio - 0.50) / 0.16)
    mouth_center_y = 0.5 * (points[13, 1] + points[14, 1])
    # MediaPipe y grows downward, so a positive value means the mouth corner
    # is lifted. This recovers one-sided smiles when the classifier score is
    # weak or delayed.
    mouth_lift_left = smoothstep(
        ((mouth_center_y - points[61, 1]) / mouth_width - 0.015) / 0.12
    )
    mouth_lift_right = smoothstep(
        ((mouth_center_y - points[291, 1]) / mouth_width - 0.015) / 0.12
    )
    # A rounded pucker can make one lip corner look artificially high in the
    # 2-D contour. Do not reinterpret that contour artifact as a one-sided
    # smile when the classifier has a strong pucker and a closed jaw.
    if blendshapes:
        raw_pucker = clamp01(float(blendshapes.get("mouthPucker", 0.0)))
        raw_jaw = clamp01(float(blendshapes.get("jawOpen", 0.0)))
        if raw_pucker >= 0.45 and raw_jaw < 0.25:
            mouth_lift_left = 0.0
            mouth_lift_right = 0.0
    # Keep this as a separate named channel so the UI can distinguish a
    # genuine jaw/lip gap from a wide mouth shape.
    lip_gap_ratio = clamp01(mouth_ratio)
    # Lip separation alone is not a jaw opening. Teeth can remain together
    # while the lips part during speech; keep that state visually narrow.
    jaw_hint = clamp01(float(blendshapes.get("jawOpen", 0.0))) if blendshapes else 0.0
    jaw_open = lip_parted * (0.16 + 0.84 * jaw_hint)

    def eye_actions(
        corner_a: int,
        corner_b: int,
        pairs: tuple[tuple[int, int], ...],
    ) -> tuple[float, float, float]:
        width = max(distance(corner_a, corner_b), 1e-6)
        ratio = sum(distance(top, bottom) for top, bottom in pairs) / (len(pairs) * width)
        blink = smoothstep((0.300 - ratio) / 0.150)
        squint = smoothstep((0.340 - ratio) / 0.140) * (1.0 - 0.72 * blink)
        return ratio, blink, squint

    # Face Mesh 362/263 is the subject's left eye; 33/133 is the right eye.
    eye_ratio_left, blink_left, squint_left = eye_actions(
        362, 263, ((386, 374), (385, 380))
    )
    eye_ratio_right, blink_right, squint_right = eye_actions(
        33, 133, ((159, 145), (158, 153))
    )
    eye_wide_left = smoothstep((eye_ratio_left - 0.285) / 0.115)
    eye_wide_right = smoothstep((eye_ratio_right - 0.285) / 0.115)
    return {
        "geometryJawOpen": jaw_open,
        "geometryLipParted": lip_parted,
        "geometryMouthRatio": mouth_ratio,
        "geometryMouthEyeRatio": mouth_eye_ratio,
        "geometryMouthWide": mouth_wide,
        "geometryMouthLiftLeft": mouth_lift_left,
        "geometryMouthLiftRight": mouth_lift_right,
        "geometryLipGapRatio": lip_gap_ratio,
        "geometryBlinkLeft": blink_left,
        "geometryBlinkRight": blink_right,
        "geometrySquintLeft": squint_left,
        "geometrySquintRight": squint_right,
        "geometryEyeRatioLeft": eye_ratio_left,
        "geometryEyeRatioRight": eye_ratio_right,
        "geometryEyeWideLeft": eye_wide_left,
        "geometryEyeWideRight": eye_wide_right,
    }


def _draw_landmarks(frame_bgr: np.ndarray, landmarks: np.ndarray | None) -> np.ndarray:
    output = frame_bgr.copy()
    if landmarks is not None:
        h, w = output.shape[:2]
        pixel_points = np.empty((len(landmarks), 2), dtype=np.int32)
        pixel_points[:, 0] = np.clip(landmarks[:, 0] * w, 0, w - 1).astype(np.int32)
        pixel_points[:, 1] = np.clip(landmarks[:, 1] * h, 0, h - 1).astype(np.int32)
        # Feature contours carry most of the useful tracking information.
        # Drawing every one of 478 points twice per frame puts avoidable load
        # on Tk's preview path, so retain a dense-enough one-in-three sample.
        for point in pixel_points[::3]:
            cv2.circle(output, tuple(point), 1, (64, 196, 255), -1, lineType=cv2.LINE_AA)
        feature_contours = (
            ([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], (80, 235, 130)),
            ([362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398], (80, 235, 130)),
            ([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78], (90, 110, 255)),
            ([10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109], (255, 190, 70)),
        )
        for indices, color in feature_contours:
            valid = [index for index in indices if index < len(pixel_points)]
            if len(valid) > 1:
                cv2.polylines(output, [pixel_points[valid]], True, color, 1, cv2.LINE_AA)
    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (310, 82), (12, 12, 18), -1)
    cv2.addWeighted(overlay, 0.45, output, 0.55, 0, output)
    cv2.putText(
        output,
        "Face Avatar Studio",
        (22, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    status = "face detected" if landmarks is not None else "no face"
    cv2.putText(
        output,
        status,
        (22, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (130, 220, 160) if landmarks is not None else (120, 120, 120),
        1,
        cv2.LINE_AA,
    )
    return output


class FaceTrackingEngine:
    # 类级缓存：确保整个进程里只构造一次 MediaPipe face_landmarker。
    # 用户日志里出现过两次 ``inference_feedback_manager.cc:121`` 警告，说明
    # ``FaceTrackingEngine`` 被构造了两次，XNNPACK delegate 也跟着 init 了两
    # 次，导致摄像头启动慢 + 渲染线程里读不到 ``engine.avatar`` 的稳定状态。
    _instances: dict[int, "FaceTrackingEngine"] = {}
    _instances_lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "FaceTrackingEngine":
        # ``id(cls)`` 保证不同子类各自一份缓存；同一类多次构造返回同一实例。
        key = id(cls)
        with cls._instances_lock:
            inst = cls._instances.get(key)
            if inst is not None:
                # Reusing the process-wide engine is expected when the user
                # stops and reopens the camera. Do not dump a Python stack on
                # every normal camera restart; it looks like a crash and
                # noticeably slows the console on Windows.
                return inst
            inst = super().__new__(cls)
            cls._instances[key] = inst
            return inst

    def __init__(
        self,
        model_cache_dir: Path | None = None,
        avatar: GnmAvatarDriver | None = None,
    ) -> None:
        # 防止 ``__init__`` 多次执行：第二次构造拿到的是 __new__ 缓存的实例，
        # 但 __init__ 仍会被调用 —— 用 _initialized 标记跳过重复初始化。
        if getattr(self, "_initialized", False):
            return
        # 关键：__init__ 只做轻量配置（路径解析 + GnmAvatarDriver 构造，~1.3s）。
        # mediapipe 的 ~14s import / landmarker init 全部推迟到 process_frame 的
        # 第一次调用（即 _ensure_landmarker）。这样：
        # 1. AppState.ensure_runtime_async 的 RuntimeInitWorker 主线程 ~1s 就完成。
        # 2. AvatarRenderWorker 立刻能用中性 mesh 出预览（用户看到 UI）。
        # 3. mediapipe 的成本可以与第一帧摄像头抓取 / DeviceScanWorker 并行发生。
        # 兜底：AppState.ensure_runtime_async 已经先 spawn 了
        # _MediapipeWarmupThread 在后台导入，到 process_frame 真正需要时通常已
        # 完成，命中 fast path。
        self.model_cache_dir = ensure_directory(
            model_cache_dir or (USER_CACHE_DIR / "mediapipe")
        )
        self.face_model_path = self.model_cache_dir / "face_landmarker.task"
        self.avatar = avatar or GnmAvatarDriver()
        self._landmarker_process: Any = None
        self._landmarker_requests: Any = None
        self._landmarker_responses: Any = None
        self._landmarker_request_id = 0
        self._landmarker_ready = False
        self._landmarker_init_lock = threading.Lock()
        # VIDEO mode requires strictly increasing timestamps. The Tk camera
        # worker starts its local clock at zero every time the camera is
        # reopened, while this engine intentionally survives that restart.
        self._last_video_timestamp_ms = 0
        self._last_valid_tracking: tuple[
            dict[str, float], np.ndarray | None, np.ndarray
        ] | None = None
        self._last_face_seen_wall = 0.0
        self._initialized = True

    def _ensure_landmarker(self) -> None:
        process = self._landmarker_process
        if process is not None and process.is_alive():
            if not self._landmarker_ready:
                self._wait_for_landmarker_ready()
            return
        with self._landmarker_init_lock:
            process = self._landmarker_process
            if process is not None and process.is_alive():
                if not self._landmarker_ready:
                    self._wait_for_landmarker_ready()
                return
            (
                self._landmarker_process,
                self._landmarker_requests,
                self._landmarker_responses,
            ) = prewarm_face_landmarker()
            self._landmarker_ready = False
            self._wait_for_landmarker_ready()

    def _wait_for_landmarker_ready(self) -> None:
        """Consume the child warmup marker before camera frames are queued."""
        responses_queue = self._landmarker_responses
        if responses_queue is None:
            raise RuntimeError("MediaPipe response queue is unavailable")
        response_id, blendshapes, face_matrix, landmarks, error = responses_queue.get(
            timeout=35.0
        )
        if response_id == -1:
            raise RuntimeError(f"MediaPipe initialization failed: {error}")
        if response_id != -2:
            raise RuntimeError("MediaPipe warmup response was out of sequence")
        self._landmarker_ready = True

    def neutral_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.avatar.evaluate({}, None)

    def _stabilize_tracking(
        self,
        blendshapes: dict[str, float] | None,
        face_matrix: np.ndarray | None,
        landmarks: np.ndarray | None,
    ) -> tuple[dict[str, float], np.ndarray | None, np.ndarray | None, bool]:
        """Bridge isolated detector misses without freezing a departed face."""
        now = time.perf_counter()
        values = dict(blendshapes or {})
        if landmarks is not None:
            self._last_valid_tracking = (values, face_matrix, landmarks)
            self._last_face_seen_wall = now
            return values, face_matrix, landmarks, False
        cached = self._last_valid_tracking
        if cached is not None and now - self._last_face_seen_wall <= FACE_TRACK_HOLD_SECONDS:
            cached_values, cached_matrix, cached_landmarks = cached
            return dict(cached_values), cached_matrix, cached_landmarks, True
        self._last_valid_tracking = None
        return values, face_matrix, None, False

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
        audio_level: float = 0.0,
        audio_visemes: dict[str, float] | None = None,
    ) -> FramePacket:
        try:
            self._ensure_landmarker()
            process = self._landmarker_process
            requests_queue = self._landmarker_requests
            responses_queue = self._landmarker_responses
            if process is None or requests_queue is None or responses_queue is None:
                raise RuntimeError("MediaPipe process did not start")
            analysis = _resize_for_face_tracking(frame_bgr)
            rgb_frame = cv2.cvtColor(analysis, cv2.COLOR_BGR2RGB)
            requested_timestamp_ms = int(max(0.0, timestamp_sec) * 1000)
            timestamp_ms = max(
                requested_timestamp_ms,
                self._last_video_timestamp_ms + 1,
            )
            self._last_video_timestamp_ms = timestamp_ms
            self._landmarker_request_id += 1
            request_id = self._landmarker_request_id
            requests_queue.put((request_id, rgb_frame, timestamp_ms), timeout=1.0)
            timeout = 35.0 if not self._landmarker_ready else 3.0
            while True:
                response_id, blendshapes, face_matrix, landmarks, error = (
                    responses_queue.get(timeout=timeout)
                )
                if response_id == -2:
                    self._landmarker_ready = True
                    continue
                break
            if response_id == -1:
                raise RuntimeError(f"MediaPipe initialization failed: {error}")
            if response_id != request_id:
                raise RuntimeError("MediaPipe response was out of sequence")
            if error:
                raise RuntimeError(error)
            self._landmarker_ready = True
            blendshapes, face_matrix, landmarks, tracking_hold = (
                self._stabilize_tracking(blendshapes, face_matrix, landmarks)
            )
            blendshapes.update(
                _derive_landmark_actions(landmarks, blendshapes, analysis.shape)
            )
            # Audio supplies only speech activity/opening assistance. It must
            # never invent pucker or phoneme-specific shapes from volume alone.
            blendshapes["audioVoiceActivity"] = clamp01(
                (float(audio_level) - 0.04) / 0.66
            )
            if audio_visemes:
                for name, value in audio_visemes.items():
                    if name.startswith("audioViseme") or name == "audioVoiceActivity":
                        blendshapes[name] = clamp01(value)
            expression, rotations, translation, vertices = self.avatar.evaluate(
                blendshapes, face_matrix, landmarks
            )
            return FramePacket(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                frame_bgr=frame_bgr,
                landmarks=landmarks,
                blendshapes=blendshapes,
                expression=expression,
                rotations=rotations,
                translation=translation,
                face_matrix=face_matrix,
                vertices=vertices,
                face_detected=landmarks is not None,
                face_tracking_hold=tracking_hold,
                semantic_weights=dict(self.avatar.last_semantic_weights),
            )
        except Exception as exc:
            expression, rotations, translation, vertices = self.avatar.evaluate({}, None)
            return FramePacket(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                frame_bgr=frame_bgr,
                landmarks=None,
                blendshapes={},
                expression=expression,
                rotations=rotations,
                translation=translation,
                face_matrix=None,
                vertices=vertices,
                face_detected=False,
                semantic_weights={},
                tracking_error=str(exc),
            )


class FaceVerseTrackingEngine(FaceTrackingEngine):
    """Optional FaceVerse v4 backend with MediaPipe used only for face ROI."""

    def __init__(self, model_cache_dir: Path | None = None) -> None:
        if getattr(self, "_initialized", False):
            return
        self.model_cache_dir = ensure_directory(
            model_cache_dir or (USER_CACHE_DIR / "mediapipe")
        )
        self.face_model_path = self.model_cache_dir / "face_landmarker.task"
        self._landmarker_process = None
        self._landmarker_requests = None
        self._landmarker_responses = None
        self._landmarker_request_id = 0
        self._landmarker_ready = False
        self._landmarker_init_lock = threading.Lock()
        self._last_video_timestamp_ms = 0
        self._last_valid_tracking = None
        self._last_face_seen_wall = 0.0
        self.avatar = FaceVerseAvatarDriver()
        self._initialized = True

    def neutral_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.avatar.neutral_state()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
        audio_level: float = 0.0,
        audio_visemes: dict[str, float] | None = None,
    ) -> FramePacket:
        try:
            self._ensure_landmarker()
            process = self._landmarker_process
            requests_queue = self._landmarker_requests
            responses_queue = self._landmarker_responses
            if process is None or requests_queue is None or responses_queue is None:
                raise RuntimeError("MediaPipe ROI process did not start")
            analysis = _resize_for_face_tracking(frame_bgr)
            rgb_frame = cv2.cvtColor(analysis, cv2.COLOR_BGR2RGB)
            requested_timestamp_ms = int(max(0.0, timestamp_sec) * 1000)
            timestamp_ms = max(
                requested_timestamp_ms, self._last_video_timestamp_ms + 1
            )
            self._last_video_timestamp_ms = timestamp_ms
            self._landmarker_request_id += 1
            request_id = self._landmarker_request_id
            requests_queue.put((request_id, rgb_frame, timestamp_ms), timeout=1.0)
            timeout = 35.0 if not self._landmarker_ready else 3.0
            while True:
                response_id, blendshapes, face_matrix, landmarks, error = (
                    responses_queue.get(timeout=timeout)
                )
                if response_id == -2:
                    self._landmarker_ready = True
                    continue
                break
            if response_id == -1:
                raise RuntimeError(f"MediaPipe ROI initialization failed: {error}")
            if response_id != request_id:
                raise RuntimeError("MediaPipe ROI response was out of sequence")
            if error:
                raise RuntimeError(error)
            self._landmarker_ready = True
            blendshapes, face_matrix, landmarks, tracking_hold = (
                self._stabilize_tracking(blendshapes, face_matrix, landmarks)
            )
            blendshapes.update(
                _derive_landmark_actions(landmarks, blendshapes, analysis.shape)
            )
            blendshapes["trackingBackendFaceVerse"] = 1.0
            blendshapes["audioVoiceActivity"] = clamp01(
                (float(audio_level) - 0.04) / 0.66
            )
            if audio_visemes:
                for name, value in audio_visemes.items():
                    if name.startswith("audioViseme") or name == "audioVoiceActivity":
                        blendshapes[name] = clamp01(value)
            if landmarks is None:
                expression, rotations, translation, vertices = self.neutral_state()
            else:
                expression, rotations, translation, vertices = self.avatar.evaluate(
                    rgb_frame, landmarks
                )
            return FramePacket(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                frame_bgr=frame_bgr,
                landmarks=landmarks,
                blendshapes=blendshapes,
                expression=expression,
                rotations=rotations,
                translation=translation,
                face_matrix=face_matrix,
                vertices=vertices,
                face_detected=landmarks is not None,
                face_tracking_hold=tracking_hold,
                semantic_weights=dict(self.avatar.last_semantic_weights),
            )
        except Exception as exc:
            expression, rotations, translation, vertices = self.neutral_state()
            return FramePacket(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                frame_bgr=frame_bgr,
                landmarks=None,
                blendshapes={},
                expression=expression,
                rotations=rotations,
                translation=translation,
                face_matrix=None,
                vertices=vertices,
                face_detected=False,
                semantic_weights={},
                tracking_error=str(exc),
            )


def create_tracking_engine(backend: str = "mediapipe") -> FaceTrackingEngine:
    normalized = str(backend).strip().lower()
    if normalized == "faceverse":
        return FaceVerseTrackingEngine()
    return FaceTrackingEngine()


class WhisperTranscriber:
    def __init__(
        self,
        model_name: str = DEFAULT_STT_MODEL,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self.compute_type = compute_type or ("float16" if self.device == "cuda" else "int8")
        self.progress_callback = None  # type: Any
        self._model = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        options = {
            "device": self.device,
            "compute_type": self.compute_type,
        }
        try:
            # Prefer the normal Hugging Face cache. This machine already has
            # faster-whisper-base locally, so export must not wait on a failed
            # network probe before phoneme alignment can begin.
            self._model = WhisperModel(
                self.model_name,
                local_files_only=True,
                **options,
            )
        except Exception:
            self._model = WhisperModel(
                self.model_name,
                download_root=str(ensure_directory(USER_CACHE_DIR / "whisper")),
                **options,
            )

    def set_progress_callback(self, callback) -> None:
        self.progress_callback = callback

    def _emit_progress(self, ratio: float, message: str) -> None:
        cb = self.progress_callback
        if cb is None:
            return
        try:
            cb(max(0.0, min(1.0, float(ratio))), message)
        except Exception:
            pass

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        self._ensure_model()
        assert self._model is not None
        self._emit_progress(0.0, "正在分析音频时长...")
        try:
            import wave

            with wave.open(str(audio_path), "rb") as wav:
                frames = wav.getnframes()
                sample_rate = wav.getframerate() or 1
                duration = frames / float(sample_rate)
        except Exception:
            duration = 0.0
        self._emit_progress(0.05, f"音频 {duration:.1f}s，开始转写...")
        segments, info = self._model.transcribe(
            str(audio_path),
            language=None if language in {None, "auto"} else language,
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
        )
        collected: list[TranscriptSegment] = []
        total = max(1, int(getattr(info, "duration", duration) or duration or 1))
        last_progress = 0.0
        for index, segment in enumerate(segments):
            words: list[dict[str, Any]] = []
            if getattr(segment, "words", None):
                words = [
                    {
                        "word": str(getattr(word, "word", "")),
                        "start": float(getattr(word, "start", segment.start)),
                        "end": float(getattr(word, "end", segment.end)),
                        "probability": float(getattr(word, "probability", 0.0)),
                    }
                    for word in segment.words
                    if getattr(word, "word", "")
                ]
            collected.append(
                TranscriptSegment(
                    index=index,
                    start=float(segment.start),
                    end=float(segment.end),
                    text=str(segment.text).strip(),
                    words=words,
                    avg_logprob=getattr(segment, "avg_logprob", None),
                )
            )
            ratio = min(1.0, max(0.05, float(segment.end) / total)) if total else 0.5
            if ratio - last_progress >= 0.05 or index % 4 == 0:
                self._emit_progress(ratio, f"已转写 {index + 1} 段 / {ratio * 100:.0f}%")
                last_progress = ratio
        self._emit_progress(1.0, "转写完成。")
        return TranscriptionResult(
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            segments=collected,
        )


class SessionRecorder:
    def __init__(
        self,
        output_root: Path,
        frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
        fps: int = DEFAULT_FPS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        stt_language: str = "auto",
        tracking_backend: str = "mediapipe",
    ) -> None:
        self.output_root = ensure_directory(output_root)
        self.frame_size = frame_size
        self.fps = fps
        self.sample_rate = sample_rate
        self.stt_language = stt_language
        self.tracking_backend = tracking_backend
        self.session_dir: Path | None = None
        self.video_path: Path | None = None
        self.audio_path: Path | None = None
        self.merged_video_path: Path | None = None
        self.timeline_csv_path: Path | None = None
        self.transcript_csv_path: Path | None = None
        self.manifest_path: Path | None = None
        self.video_writer: cv2.VideoWriter | None = None
        self.audio_stream: sd.InputStream | None = None
        self.audio_chunks: list[np.ndarray] = []
        self.viseme_analyzer = RealtimeVisemeAnalyzer(sample_rate)
        self.audio_level_callback = None  # type: Any
        self.frame_rows: list[dict[str, Any]] = []
        self._audio_lock = threading.Lock()
        self._active = False
        self._start_wall_time = 0.0
        self._mic_device: int | None = None
        self._camera_name: str | None = None
        self._mic_name: str | None = None
        self._last_level: float = 0.0
        self._last_export_error: str | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def elapsed_seconds(self) -> float:
        if not self._active:
            return 0.0
        return max(0.0, time.perf_counter() - self._start_wall_time)

    @property
    def latest_level(self) -> float:
        return float(self._last_level)

    def set_audio_level_callback(self, callback) -> None:
        self.audio_level_callback = callback

    def latest_audio_visemes(self) -> dict[str, float]:
        return self.viseme_analyzer.values()

    def start(
        self,
        camera_name: str,
        microphone_name: str,
        mic_device_index: int | None,
    ) -> Path:
        if self._active:
            raise RuntimeError("Recording already active.")

        session_name = time.strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = ensure_directory(self.output_root / session_name)
        self.video_path = self.session_dir / "video_only.mp4"
        self.audio_path = self.session_dir / "audio.wav"
        self.merged_video_path = self.session_dir / "recording_with_audio.mp4"
        self.timeline_csv_path = self.session_dir / "timeline.csv"
        self.transcript_csv_path = self.session_dir / "transcript_segments.csv"
        self.manifest_path = self.session_dir / "session.json"
        self.frame_rows = []
        self.audio_chunks = []
        self.viseme_analyzer.reset()
        self._camera_name = camera_name
        self._mic_name = microphone_name
        self._mic_device = mic_device_index
        self._start_wall_time = time.perf_counter()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width, height = self.frame_size
        self.video_writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            float(self.fps),
            (int(width), int(height)),
        )
        if not self.video_writer.isOpened():
            raise RuntimeError("Failed to open video writer.")

        try:
            self.audio_stream = sd.InputStream(
                device=mic_device_index,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.audio_stream.start()
        except Exception as exc:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.audio_stream = None
            raise RuntimeError(f"Failed to open microphone: {exc}") from exc

        self._active = True
        return self.session_dir

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            pass
        chunk = np.asarray(indata, dtype=np.float32)
        if chunk.ndim == 2 and chunk.shape[1] > 1:
            chunk = chunk.mean(axis=1, keepdims=True)
        level = float(np.abs(chunk).max()) if chunk.size else 0.0
        self._last_level = level
        self.viseme_analyzer.process(chunk)
        cb = self.audio_level_callback
        if cb is not None:
            try:
                cb(level)
            except Exception:
                pass
        with self._audio_lock:
            self.audio_chunks.append(chunk.copy())

    def append_packet(self, packet: FramePacket) -> None:
        self.append_frame(packet.frame_bgr, packet)

    def append_frame(
        self,
        frame_bgr: np.ndarray,
        packet: FramePacket,
        *,
        frame_index: int | None = None,
        timestamp_sec: float | None = None,
    ) -> None:
        if not self._active or self.video_writer is None:
            return
        frame = _fit_frame_to_size(frame_bgr, self.frame_size)
        self.video_writer.write(frame)
        recorded_frame_index = packet.frame_index if frame_index is None else frame_index
        recorded_timestamp = packet.timestamp_sec if timestamp_sec is None else timestamp_sec
        self.frame_rows.append(
            {
                "frame_index": recorded_frame_index,
                "timestamp_sec": round(recorded_timestamp, 6),
                "face_detected": int(packet.face_detected),
                "face_tracking_hold": int(packet.face_tracking_hold),
                "blendshapes_json": json_dumps(
                    {k: round(float(v), 6) for k, v in sorted(packet.blendshapes.items())}
                ),
                "gnm_semantic_weights_json": json_dumps(
                    {k: round(float(v), 6) for k, v in sorted(packet.semantic_weights.items())}
                ),
                "expression_json": json_dumps(np.round(packet.expression, 6).tolist()),
                "rotations_json": json_dumps(np.round(packet.rotations, 6).tolist()),
                "translation_json": json_dumps(np.round(packet.translation, 6).tolist()),
                "face_matrix_json": json_dumps(
                    None
                    if packet.face_matrix is None
                    else np.round(packet.face_matrix, 6).tolist()
                ),
            }
        )

    def stop_capture(self) -> None:
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
            finally:
                self.audio_stream.close()
                self.audio_stream = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self._active = False
        if self.audio_path is not None:
            self._write_audio_file(self.audio_path)

    def _write_audio_file(self, path: Path) -> None:
        with self._audio_lock:
            chunks = list(self.audio_chunks)
        if not chunks:
            path.write_bytes(b"")
            return
        audio = np.concatenate(chunks, axis=0).squeeze()
        audio = np.asarray(audio, dtype=np.float32)
        # USB/webcam microphones often expose very conservative Windows input
        # gain. Normalize against a robust peak so the exported WAV, muxed
        # video and STT input are audible, while capping gain to avoid turning
        # a silent/noisy device into full-scale noise.
        reference_peak = float(np.percentile(np.abs(audio), 99.5)) if audio.size else 0.0
        if reference_peak >= 5e-4:
            gain = float(np.clip(0.82 / reference_peak, 1.0, 12.0))
            audio *= gain
        audio = np.clip(audio, -1.0, 1.0)
        wavfile.write(path, self.sample_rate, (audio * 32767.0).astype(np.int16))

    def _attach_transcript(self, rows: list[dict[str, Any]], segments: list[TranscriptSegment]) -> None:
        phoneme_events = build_phoneme_timeline(segments, self.stt_language)
        if not segments:
            for row in rows:
                row["speech_text"] = ""
                row["speech_segment_index"] = ""
                row["speech_start_sec"] = ""
                row["speech_end_sec"] = ""
                row["speech_word"] = ""
                row["phoneme"] = ""
                row["phoneme_viseme"] = ""
                row["phoneme_start_sec"] = ""
                row["phoneme_end_sec"] = ""
            return

        seg_index = 0
        phoneme_index = 0
        for row in rows:
            timestamp = float(row["timestamp_sec"])
            while seg_index + 1 < len(segments) and segments[seg_index].end < timestamp:
                seg_index += 1
            active = None
            if seg_index < len(segments):
                candidate = segments[seg_index]
                if candidate.start <= timestamp <= candidate.end:
                    active = candidate
                elif seg_index + 1 < len(segments):
                    next_candidate = segments[seg_index + 1]
                    if next_candidate.start <= timestamp <= next_candidate.end:
                        active = next_candidate
                        seg_index += 1
            if active is None:
                row["speech_text"] = ""
                row["speech_segment_index"] = ""
                row["speech_start_sec"] = ""
                row["speech_end_sec"] = ""
            else:
                row["speech_text"] = active.text
                row["speech_segment_index"] = active.index
                row["speech_start_sec"] = round(active.start, 6)
                row["speech_end_sec"] = round(active.end, 6)
            while (
                phoneme_index + 1 < len(phoneme_events)
                and float(phoneme_events[phoneme_index]["end"]) < timestamp
            ):
                phoneme_index += 1
            phoneme_event = None
            if phoneme_index < len(phoneme_events):
                candidate = phoneme_events[phoneme_index]
                if float(candidate["start"]) <= timestamp <= float(candidate["end"]):
                    phoneme_event = candidate
            if phoneme_event is None:
                row["speech_word"] = ""
                row["phoneme"] = ""
                row["phoneme_viseme"] = ""
                row["phoneme_start_sec"] = ""
                row["phoneme_end_sec"] = ""
            else:
                row["speech_word"] = phoneme_event["word"]
                row["phoneme"] = phoneme_event["phoneme"]
                row["phoneme_viseme"] = phoneme_event["viseme"]
                row["phoneme_start_sec"] = round(float(phoneme_event["start"]), 6)
                row["phoneme_end_sec"] = round(float(phoneme_event["end"]), 6)

    def _mux_audio_video(self) -> Path | None:
        if self.video_path is None or self.audio_path is None:
            return None
        if not self.video_path.exists() or not self.audio_path.exists():
            return None
        try:
            ffmpeg = get_ffmpeg_exe()
            merged = self.merged_video_path or (self.session_dir / "recording_with_audio.mp4")
            command = [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(self.video_path),
                "-i",
                str(self.audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(merged),
            ]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            return merged
        except Exception as exc:
            self._last_export_error = f"ffmpeg 合并失败：{exc}"
            return None

    def finalize(self, transcriber: WhisperTranscriber | None = None) -> RecordingOutputs:
        if self.session_dir is None:
            raise RuntimeError("No recording session available.")

        rows = [dict(row) for row in self.frame_rows]
        segments: list[TranscriptSegment] = []
        if transcriber is not None and self.audio_path and self.audio_path.exists() and self.audio_path.stat().st_size > 0:
            result = transcriber.transcribe(self.audio_path, language=self.stt_language)
            segments = result.segments
            transcript_rows = [
                {
                    "segment_index": seg.index,
                    "start_sec": round(seg.start, 6),
                    "end_sec": round(seg.end, 6),
                    "text": seg.text,
                    "words": json_dumps(seg.words),
                    "phonemes": json_dumps(build_phoneme_timeline([seg], self.stt_language)),
                    "avg_logprob": "" if seg.avg_logprob is None else round(float(seg.avg_logprob), 6),
                }
                for seg in segments
            ]
            write_csv(
                self.transcript_csv_path,
                transcript_rows,
                fieldnames=[
                    "segment_index",
                    "start_sec",
                    "end_sec",
                    "text",
                    "words",
                    "phonemes",
                    "avg_logprob",
                ],
            )
        elif self.transcript_csv_path is not None:
            write_csv(
                self.transcript_csv_path,
                [],
                fieldnames=[
                    "segment_index",
                    "start_sec",
                    "end_sec",
                    "text",
                    "words",
                    "phonemes",
                    "avg_logprob",
                ],
            )

        self._attach_transcript(rows, segments)
        if self.timeline_csv_path is not None:
            write_csv(self.timeline_csv_path, rows)

        merged_video = self._mux_audio_video()
        if merged_video is None:
            merged_video = self.video_path if self.video_path and self.video_path.exists() else None

        warnings: list[str] = []
        if merged_video is None:
            warnings.append("未生成合并音视频文件，仅保留纯视频。")
        if self._last_export_error:
            warnings.append(self._last_export_error)
        if not segments:
            warnings.append("音频无有效语音，未生成转写段。")

        manifest = {
            "session_dir": str(self.session_dir),
            "camera_name": self._camera_name,
            "microphone_name": self._mic_name,
            "frame_size": list(self.frame_size),
            "fps": self.fps,
            "sample_rate": self.sample_rate,
            "stt_language": self.stt_language,
            "tracking_backend": self.tracking_backend,
            "video_path": None if self.video_path is None else str(self.video_path),
            "audio_path": None if self.audio_path is None else str(self.audio_path),
            "merged_video_path": None if merged_video is None else str(merged_video),
            "timeline_csv_path": None if self.timeline_csv_path is None else str(self.timeline_csv_path),
            "transcript_csv_path": None if self.transcript_csv_path is None else str(self.transcript_csv_path),
            "warnings": warnings,
        }
        if self.manifest_path is not None:
            self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        return RecordingOutputs(
            session_dir=self.session_dir,
            video_path=self.video_path,
            audio_path=self.audio_path,
            merged_video_path=merged_video,
            timeline_csv_path=self.timeline_csv_path,
            transcript_csv_path=self.transcript_csv_path,
            manifest_path=self.manifest_path,
        )
