@echo off
REM One-click desktop entry: starts the Decadic API server + React web UI.
REM Delegates to launch_decadic.ps1 (port checks, no duplicate spawns, opens browser).
setlocal
title Decadic Launcher
cd /d "%~dp0.."
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0launch_decadic.ps1"
if errorlevel 1 (
    echo.
    echo Launcher failed. See messages above.
    pause
    exit /b 1
)
exit /b 0
