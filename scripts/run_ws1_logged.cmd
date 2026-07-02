@echo off
cd /d "D:\Users\charl\software\Self-Determination Model"
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\run_ws1.ps1" -SkipTests -SkipSmoke > "reports\ws1_launch.log" 2>&1
