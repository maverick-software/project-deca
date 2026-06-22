@echo off
setlocal
set "REPO=%~dp0.."
cd /d "%REPO%\dashboard"

npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
