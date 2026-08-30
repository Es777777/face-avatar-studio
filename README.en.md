# Face Avatar Studio

A native Windows desktop application for real-time facial expression capture and 3D avatar driving. It is built around [Google GNM](https://github.com/google/GNM) and [MediaPipe Face Landmarker](https://github.com/google-ai-edge/mediapipe), with optional [FaceVerse V2](https://github.com/LizhenWangT/FaceVerse#faceverse-version-2) 52D ARKit driving and a [FaceVerse v4](https://github.com/LizhenWangT/FaceVerse_v4) full-head parametric backend.

[中文 README](README.md)

> This repository contains the Windows desktop application source, not a web project,
> and does not currently publish a prebuilt `.exe`. The default entry point is
> `face_avatar_studio/tk_ui.py`. Large model files are managed with Git LFS, so install
> and initialize Git LFS before cloning.

## Quick Start

```powershell
git lfs install
git clone https://github.com/Es777777/face-avatar-studio.git
cd face-avatar-studio
python -m pip install -r requirements.txt
python launch_face_avatar_studio.py
```

You can also double-click `start_face_avatar_studio.bat`. A ZIP download may contain
small Git LFS pointer files instead of the real models; use a Git clone and run
`git lfs pull`, otherwise FaceVerse and some GNM data cannot load.

## Features

- Real-time camera capture of face landmarks, expressions, and head pose
- MediaPipe + GNM by default for low-latency full-head avatar driving
- Optional FaceVerse V2 with direct MediaPipe-to-ARKit 52D expression driving
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

`requirements.txt` includes both runtime dependencies and PyInstaller for the Windows
build. FaceVerse inference can use substantial memory, and a slow first model load does
not mean that the camera failed to open.

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

The repository already includes the GNM source and data. If you are using an incomplete
ZIP or have removed the external directory, restore it using the upstream GNM repository.
The MediaPipe `face_landmarker.task` model is downloaded and cached according to the
project configuration.

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
- `FaceVerse V2 (52D)`: MediaPipe's 52 ARKit expression scores directly drive the matching FaceVerse bases

Close the camera before switching backends. The first FaceVerse startup initializes PyTorch and the model and may take several seconds.

## FaceVerse Models

The FaceVerse weights are not distributed directly in the official source repository.
This project stores them as GitHub LFS files in this same software repository:

<https://github.com/Es777777/face-avatar-studio>

Required files:

```text
external_FaceVerse_v4/data/faceverse_v4_2.npy
external_FaceVerse_v4/data/faceverse_resnet50.pth
```

With Git LFS enabled, cloning this repository restores both files to the directory above.
For a manual deployment, run `git lfs pull`, or download the LFS files from this repository
and place them there. Keep the exact filenames; the application checks them explicitly and
reports a clear error instead of silently pretending that FaceVerse is active.

Refer to the original FaceVerse repository for its code, license, copyright, and research citation:
<https://github.com/LizhenWangT/FaceVerse_v4>

Optional official FaceVerse V2 model location:

```text
external_FaceVerse_v2/data/faceverse_simple_v2.npy
```

When this file is present, the V2 backend automatically uses the official simplified V2 mesh. Until it is available, the backend remains functional by using the V4 full-head jaw, teeth, and oral-cavity deformation bases with semantic mapping for the 52 ARKit channels. The status bar always identifies which source is active. See the original V2 model and download instructions at:
<https://github.com/LizhenWangT/FaceVerse#faceverse-version-2>

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
external_FaceVerse_v2/    optional FaceVerse V2 model directory
external_FaceVerse_v4/    FaceVerse v4 source and model files
tools/diagnostics/        diagnostics and regression scripts
artifacts/generated/      test screenshots and verification results
artifacts/references/     reference assets
artifacts/logs/           historical logs
.cache/                   MediaPipe and development caches
```

The active launch chain is `start_face_avatar_studio.bat` ->
`python -m face_avatar_studio` -> `face_avatar_studio/tk_ui.py`.
`desktop_app.py`, `ui.py`, and `webapp.py` are earlier Qt/web experiments and are not
used by the default launcher. They remain in the tree for historical reference and
compatibility with older development versions. The scripts under `tools/diagnostics/`
are development checks, not additional application launchers.

## Troubleshooting

### FaceVerse is upside down, backwards, or appears frozen

Confirm that both model files are present in `external_FaceVerse_v4/data/`, then fully exit and restart the application. The adapter converts FaceVerse native coordinates into the desktop preview coordinate system and updates the mesh from live regression results. Check that the status bar reports `FaceVerse v4` and that the frame counter and avatar FPS continue changing.

### FaceVerse V2 reports "52D compatibility mode"

This is an operational fallback: V2's direct ARKit mapping is using the complete V4-compatible head mesh. Place the official `faceverse_simple_v2.npy` under `external_FaceVerse_v2/data/` and restart the application to switch automatically to the official V2 mesh.

### First startup is slow

The first run may load MediaPipe, PyTorch, FaceVerse, or VTK. These modules remain cached in the process afterward. Do not run multiple instances or switch backends while the camera is open.

### Black avatar or unresponsive UI

Close the camera and restart the avatar preview. If the issue persists, inspect the launch log and `artifacts/logs/`. The software renderer can be used to distinguish graphics-driver issues from tracking issues.

If the first run is stuck while downloading a model, check network/proxy access and that
the user cache directory is writable. You can run
`python tools\diagnostics\smoke_mediapipe_warmup.py` to check MediaPipe separately.
For FaceVerse, confirm that Git LFS has restored real model files rather than pointer
files that are only a few hundred bytes long.

## Packaging

Run:

```powershell
.\build_windows_app.ps1
```

Before packaging, run `python -m pip install -r requirements.txt`; this installs
`PyInstaller`. Source-code users do not need to run the packaging step.

The output is normally created under `dist/FaceAvatarStudio/`. A distributable build must include the external source trees, the MediaPipe model cache configuration, and the FaceVerse weights. The V2 model directory is included by the build script.

## Licenses and Acknowledgements

This project integrates several upstream projects. The licenses and usage restrictions for GNM, MediaPipe, FaceVerse, Sim3DR, and their model files are governed by their respective repositories. Keep the upstream license files and citations when redistributing the application.
