@echo off
setlocal
cd /d "%~dp0"

python -m face_avatar_studio
if errorlevel 1 (
  echo.
  echo Program failed to start.
  pause
)
