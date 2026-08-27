# Face Avatar Studio

A native Windows desktop application for real-time facial expression capture and 3D avatar driving. It is built around [Google GNM](https://github.com/google/GNM) and [MediaPipe Face Landmarker](https://github.com/google-ai-edge/mediapipe), with an optional [FaceVerse v4](https://github.com/LizhenWangT/FaceVerse_v4) full-head parametric backend.

[中文 README](README.md)

## Features

- Real-time camera capture of face landmarks, expressions, and head pose
- MediaPipe + GNM by default for low-latency full-head avatar driving
- Optional FaceVerse v4 621-dimensional parameter regression, including the head, eyeballs, teeth, and tongue
- Camera and microphone selection from the desktop UI
- Live camera preview, avatar preview, FPS, tracking state, and microphone level
- One-click recording of synchronized video, audio, and expression timelines
- Optional faster-whisper transcription for Chinese or English
- CSV export combining expression parameters, timestamps, audio segments, and transcript text

## Requirements

- Windows 10/11
- Python 3.13 in the current development setup
- A working camera and microphone
- CPU is supported; CUDA is recommended for better FaceVerse performance

The application uses a Tk desktop shell so the main window does not depend on the Qt/VTK DLL startup chain. VTK is loaded on demand in the avatar preview worker, and renderer failures are isolated from the main UI.

## Installation

From the project root:

```powershell
python -m pip install -r requirements.txt
```

If GNM is not present locally:

```powershell
git clone https://github.com/google/GNM.git external_GNM
```

The MediaPipe `face_landmarker.task` model is downloaded and cached according to the project configuration.

## Launch

Double-click:

```text
start_face_avatar_studio.bat
```

Or run:

```powershell
python launch_face_avatar_studio.py
```

After launch, select one of the tracking backends:

- `MediaPipe (default)`: MediaPipe provides expression parameters and GNM drives the avatar
- `FaceVerse v4`: MediaPipe is used only for a stable face ROI; the FaceVerse ResNet50 predicts the 621-dimensional parameters

Close the camera before switching backends. The first FaceVerse startup initializes PyTorch and the model and may take several seconds.

## FaceVerse Models

The FaceVerse weights are not distributed directly in the official source repository. This project uses a separate GitHub LFS model repository:

<https://github.com/Es777777/face-avatar-studio-models>

Required files:

```text
external_FaceVerse_v4/data/faceverse_v4_2.npy
external_FaceVerse_v4/data/faceverse_resnet50.pth
```

Both files are already present in the current project. For a manual deployment, download them from the model repository and place them in the directory above. The application checks these files explicitly and reports a clear error instead of silently pretending that FaceVerse is active.

Refer to the original FaceVerse repository for its code, license, copyright, and research citation:
<https://github.com/LizhenWangT/FaceVerse_v4>

## Recording Outputs

Each recording creates an independent session directory, normally containing:

```text
recording_with_audio.mp4   muxed audio/video recording
video_only.mp4             video without audio
audio.wav                  original audio
timeline.csv               per-frame expressions, timestamps, and tracking state
transcript_segments.csv    STT segment results
session.json               recording configuration and metadata
```

STT uses faster-whisper. The selected model can be downloaded on first use or prepared in the local model cache in advance.

## Repository Layout

```text
face_avatar_studio/       desktop application source
external_GNM/             Google GNM source
external_FaceVerse_v4/    FaceVerse v4 source and model files
tools/diagnostics/        diagnostics and regression scripts
artifacts/generated/      test screenshots and verification results
artifacts/references/     reference assets
artifacts/logs/           historical logs
.cache/                   MediaPipe and development caches
```

## Troubleshooting

### FaceVerse is upside down, backwards, or appears frozen

Confirm that both model files are present in `external_FaceVerse_v4/data/`, then fully exit and restart the application. The adapter converts FaceVerse native coordinates into the desktop preview coordinate system and updates the mesh from live regression results. Check that the status bar reports `FaceVerse v4` and that the frame counter and avatar FPS continue changing.

### First startup is slow

The first run may load MediaPipe, PyTorch, FaceVerse, or VTK. These modules remain cached in the process afterward. Do not run multiple instances or switch backends while the camera is open.

### Black avatar or unresponsive UI

Close the camera and restart the avatar preview. If the issue persists, inspect the launch log and `artifacts/logs/`. The software renderer can be used to distinguish graphics-driver issues from tracking issues.

## Packaging

Run:

```powershell
.\build_windows_app.ps1
```

The output is normally created under `dist/FaceAvatarStudio/`. A distributable build must include the external source trees, the MediaPipe model cache configuration, and the FaceVerse weights.

## Licenses and Acknowledgements

This project integrates several upstream projects. The licenses and usage restrictions for GNM, MediaPipe, FaceVerse, Sim3DR, and their model files are governed by their respective repositories. Keep the upstream license files and citations when redistributing the application.
