@echo off
setlocal
set "REPO=%~dp0.."
cd /d "%REPO%"

for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":8765 .*LISTENING" /C:":5173 .*LISTENING"') do (
  taskkill /PID %%P /F >nul 2>nul
)

ping 127.0.0.1 -n 3 >nul

start "Decadic Server" /min cmd.exe /k "%REPO%\scripts\run_decadic_server.cmd"
start "Decadic Web UI" /min cmd.exe /k "%REPO%\scripts\run_decadic_ui.cmd"
