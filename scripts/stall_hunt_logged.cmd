@echo off
cd /d "D:\Users\charl\software\Self-Determination Model"
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\stall_hunt.ps1" > "reports\stall_hunt_launch.log" 2>&1
