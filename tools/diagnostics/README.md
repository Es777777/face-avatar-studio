# Diagnostics

## FaceVerse V2

```powershell
python tools/diagnostics/smoke_faceverse_v2.py
```

Checks the 52D model source, neutral mesh, jaw/smile/blink deformation,
left/right blink independence, backend factory selection, and per-frame mesh
evaluation performance without opening the camera.

These scripts are development and regression checks. They are not needed to
launch the desktop application.

Run them from the project root, for example:

```powershell
python tools\diagnostics\smoke_mediapipe_reference.py
```

Generated screenshots and reports are written to `artifacts/generated/`.
