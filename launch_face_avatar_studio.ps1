$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

& python -X faulthandler "$PSScriptRoot\launch_face_avatar_studio.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Program failed to start."
    pause
}