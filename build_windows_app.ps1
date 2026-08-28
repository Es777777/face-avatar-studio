Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$dist = Join-Path $root "dist"
$build = Join-Path $root "build"

if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
if (Test-Path $build) { Remove-Item -Recurse -Force $build }

python -m PyInstaller `
  --noconsole `
  --onedir `
  --name FaceAvatarStudio `
  --add-data "external_GNM;external_GNM" `
  --add-data "external_FaceVerse_v2;external_FaceVerse_v2" `
  --add-data "external_FaceVerse_v4;external_FaceVerse_v4" `
  --collect-all mediapipe `
  --collect-all torch `
  --collect-all imageio_ffmpeg `
  --hidden-import mediapipe.tasks `
  --hidden-import mediapipe.tasks.vision `
  --hidden-import mediapipe.tasks.python `
  --hidden-import mediapipe.tasks.python.vision `
  face_avatar_studio\main.py

Write-Host ""
Write-Host "Build complete."
Write-Host ("Executable: " + (Join-Path $dist "FaceAvatarStudio\FaceAvatarStudio.exe"))
