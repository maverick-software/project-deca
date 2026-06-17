<#
.SYNOPSIS
    One-click launcher for the Decadic stack: starts the FastAPI server and the
    React dashboard (web UI), then opens the dashboard in your browser.

.DESCRIPTION
    Spawns two visible PowerShell windows you can watch and close:
      1. Server  - uvicorn on http://127.0.0.1:8765
      2. Web UI  - Vite dev server on http://localhost:5173
    Each process is only started if its port is free, so re-running this (or the
    desktop shortcut) won't spawn duplicates. First run installs the dashboard's
    npm dependencies if they're missing.

.PARAMETER ServerPort
    Port for the FastAPI server (default 8765).

.PARAMETER UiPort
    Port for the dashboard dev server (default 5173).

.PARAMETER NoBrowser
    Don't auto-open the browser when the UI is ready.
#>
[CmdletBinding()]
param(
    [int]$ServerPort = 8765,
    [int]$UiPort = 5173,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

# Repo root is the parent of this script's folder (scripts/..).
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DashboardDir = Join-Path $RepoRoot "dashboard"

function Test-PortListening([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Decadic launcher" -ForegroundColor Cyan
Write-Host " repo: $RepoRoot" -ForegroundColor DarkGray
Write-Host "==================================================" -ForegroundColor Cyan

# --- 1. Server ------------------------------------------------------------
if (Test-PortListening $ServerPort) {
    Write-Host "[server] already running on port $ServerPort - reusing it." -ForegroundColor Yellow
}
else {
    Write-Host "[server] starting uvicorn on http://127.0.0.1:$ServerPort ..." -ForegroundColor Green
    # Tell the server which port server-managed bodies (MuJoCo adapter) must
    # connect back to. Without this the body defaults can mismatch a custom port
    # and the body crashes with ConnectionRefused.
    $serverCmd = @"
`$host.UI.RawUI.WindowTitle = 'Decadic Server (port $ServerPort)'
Set-Location '$RepoRoot'
`$env:DECADIC_SELF_HOST = '127.0.0.1'
`$env:DECADIC_SELF_PORT = '$ServerPort'
# New-agent defaults: full-size brain with all three neuroplasticity
# subsystems on (optimal config knobs come from decadic/config.py). Each can
# still be turned off per-agent from the dashboard.
`$env:DECADIC_NEURAL_PRESET = 'full'
`$env:DECADIC_PLASTICITY_ENABLED = '1'
`$env:DECADIC_SPARSE_ENABLED = '1'
`$env:DECADIC_GROWTH_ENABLED = '1'
# Perception: frozen CLIP + Whisper encoders so the agent actually sees and
# hears (first run downloads ~1 GB). 'hf' is already the code default; pinned
# here so an inherited shell env can never silently fall back to 'zeros'.
`$env:DECADIC_ENCODER_MODE = 'hf'
# Compute: run cognition on the GPU. Auto-detect already prefers CUDA, but pin it
# so an inherited env can't silently fall back to the (10-20x slower) CPU path.
# The frozen CLIP/Whisper encoders autocast to bf16 on the GPU (Ampere+); the
# trainable stack stays fp32. Falls back to CPU automatically if CUDA is absent.
`$env:DECADIC_DEVICE = 'cuda'
# Persistence: write the per-cycle episodic record on a background worker so the
# SQLite commit never blocks the cognitive lock. Birth default for new agents; also
# a live per-agent toggle in the dashboard (Agent Settings).
`$env:DECADIC_EPISODIC_ASYNC = '1'
# Embodiment: the humanoid has 21 actuators (ankle pitch+roll added). The body
# subprocess inherits this env, so the motor head and body stay in lockstep.
`$env:DECADIC_N_ACTUATORS = '21'
# Locomotion curriculum default for NEW agents: 'legacy' = Standard support with
# the manual training-wheels harness, which the runtime starts at assist level 0
# (no assistance) -- the agent must hold itself up from birth. Switch a given
# agent to 'guided' (assist-as-needed vertical-only support) from the Motor tab.
`$env:DECADIC_CURRICULUM_MODE = 'legacy'
python -m uvicorn decadic.api.app:app --host 127.0.0.1 --port $ServerPort
Write-Host ''
Write-Host 'Server stopped. Press any key to close this window.' -ForegroundColor Yellow
[void][System.Console]::ReadKey(`$true)
"@
    Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-NoProfile", "-Command", $serverCmd)
}

# --- 2. Web UI ------------------------------------------------------------
if (-not (Test-Path (Join-Path $DashboardDir "node_modules"))) {
    Write-Host "[web] installing dashboard dependencies (first run) ..." -ForegroundColor Green
    Push-Location $DashboardDir
    try { npm install } finally { Pop-Location }
}

if (Test-PortListening $UiPort) {
    Write-Host "[web] already running on port $UiPort - reusing it." -ForegroundColor Yellow
}
else {
    Write-Host "[web] starting Vite dev server on http://localhost:$UiPort ..." -ForegroundColor Green
    # Quote the '--' so PowerShell passes it through to npm literally (an
    # unquoted -- is PowerShell's end-of-parameters token and would be dropped,
    # leaving vite to ignore the port and fall back to its config default).
    $uiCmd = @"
`$host.UI.RawUI.WindowTitle = 'Decadic Web UI (port $UiPort)'
Set-Location '$DashboardDir'
npm run dev '--' '--port' $UiPort '--strictPort'
Write-Host ''
Write-Host 'Web UI stopped. Press any key to close this window.' -ForegroundColor Yellow
[void][System.Console]::ReadKey(`$true)
"@
    Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-NoProfile", "-Command", $uiCmd)
}

# --- 3. Open the browser when the UI responds -----------------------------
if (-not $NoBrowser) {
    $uiUrl = "http://localhost:$UiPort"
    Write-Host "[web] waiting for the UI to come up ..." -ForegroundColor Green
    $ready = $false
    foreach ($i in 1..60) {
        if (Test-PortListening $UiPort) { $ready = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if ($ready) {
        Start-Sleep -Seconds 1
        Write-Host "[web] opening $uiUrl" -ForegroundColor Green
        Start-Process $uiUrl
    }
    else {
        Write-Host "[web] UI did not come up in time; open $uiUrl manually." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Done. Server and Web UI run in their own windows; close them to stop." -ForegroundColor Cyan
Start-Sleep -Seconds 3
