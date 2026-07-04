# Vast.ai end-to-end deploy test: rent -> deploy -> verify model runs -> destroy.
# SPENDS REAL MONEY (typically well under $0.50 at defaults). The instance is
# auto-destroyed on every exit path; hard wall-clock cap default 40 minutes.
#
# Usage:  .\scripts\run_vast_e2e.ps1                 # interactive rent confirmation
#         .\scripts\run_vast_e2e.ps1 -Yes            # no prompt (for scripted runs)
#         .\scripts\run_vast_e2e.ps1 -MaxDph 0.30 -ObserveMinutes 6

param(
    [switch]$Yes,
    [double]$MaxDph = 0.45,
    [string]$GpuName = "RTX_4090",
    [string]$Preset = "tiny",
    [string]$Encoder = "zeros",
    [string]$Scene = "forage",
    [double]$ObserveMinutes = 4,
    [double]$MaxMinutes = 40,
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"

# Preflight: the local server must already be running (it holds the API key
# and the vast controller). Start it with scripts\launch_decadic.ps1 if not.
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:$Port/vast/settings" -TimeoutSec 5 | Out-Null
} catch {
    Write-Host "Server not reachable on port $Port. Start it first: .\scripts\launch_decadic.ps1" -ForegroundColor Red
    exit 1
}

$TestArgs = @(
    "scripts\vast_e2e_test.py",
    "--base-url", "http://127.0.0.1:$Port",
    "--gpu-name", $GpuName,
    "--max-dph", "$MaxDph",
    "--preset", $Preset,
    "--encoder", $Encoder,
    "--scene", $Scene,
    "--observe-minutes", "$ObserveMinutes",
    "--max-minutes", "$MaxMinutes"
)
if ($Yes) { $TestArgs += "--yes" }

& $Py @TestArgs
exit $LASTEXITCODE
