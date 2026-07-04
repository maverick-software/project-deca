# WS5-M0.4 smoke: scenario entities must appear as WM slots in a live agent.
# Stub pipeline (no neural) - this tests the perception/WM seam, not cognition.
# Self-terminating (~1 min). Usage: .\scripts\run_binding_smoke.ps1
param(
    [int]$Port = 8766,
    [int]$Steps = 300,
    # NB: NOT under docs\eval_scenarios -- the /eval/scenarios route parses
    # everything there as an EvalSpec (different schema; broke the suite once).
    [string]$Scenario = "docs\binding_scenarios\binding_probe.json"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $Root "reports\bindsmoke_$Stamp"
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$BaseUrl = "http://127.0.0.1:$Port"

if (-not (Test-Path $Scenario)) {
    & $Py "scripts\gen_binding_scenario.py" --out $Scenario
}

# Save/restore every env var this script touches: $env: assignments persist
# in the caller's session after the script exits, and stale DECADIC_* vars
# have now bitten twice (sqlite probe 07-04 morning; USE_NEURAL=0 breaking
# 10 bundle tests 07-04 afternoon).
$TouchedEnv = @(
    "DECADIC_SELF_HOST", "DECADIC_SELF_PORT", "DECADIC_USE_NEURAL",
    "DECADIC_DEVICE", "DECADIC_LOG_DIR", "DECADIC_PERCEPTION_MODE"
)
$SavedEnv = @{}
foreach ($n in $TouchedEnv) { $SavedEnv[$n] = [Environment]::GetEnvironmentVariable($n) }

$env:DECADIC_SELF_HOST = "127.0.0.1"
$env:DECADIC_SELF_PORT = "$Port"
$env:DECADIC_USE_NEURAL = "0"
$env:DECADIC_DEVICE = "cpu"
$env:DECADIC_LOG_DIR = $RunDir
# The binding probe injects entities at the ORACLE seam (PRD ws5 5.5): in the
# server's default discovered mode, world_state.entities is quarantined as
# eval-only truth and WM integrates zero camera proposals per frame
# (diagnosed 2026-07-04: wm cycle=300, slots=0).
$env:DECADIC_PERCEPTION_MODE = "oracle"

$Server = $null
try {
    Write-Host "starting stub server on port $Port..." -ForegroundColor Cyan
    $Server = Start-Process -FilePath $Py `
        -ArgumentList "-m", "uvicorn", "decadic.api.app:app", "--host", "127.0.0.1", "--port", "$Port" `
        -WorkingDirectory $Root -PassThru -NoNewWindow `
        -RedirectStandardOutput "$RunDir\server.out.log" -RedirectStandardError "$RunDir\server.err.log"
    $Ready = $false
    foreach ($i in 1..60) {
        Start-Sleep -Seconds 2
        try { Invoke-RestMethod "$BaseUrl/agents" -TimeoutSec 3 | Out-Null; $Ready = $true; break }
        catch { if ($Server.HasExited) { throw "server exited early - see $RunDir\server.err.log" } }
    }
    if (-not $Ready) { throw "server not ready" }

    $aid = (Invoke-RestMethod -Method Post -Uri "$BaseUrl/agent" -TimeoutSec 180).agent_id
    Write-Host "agent: $aid  scenario: $Scenario"
    & $Py "scripts\synthetic_ws_client.py" --port $Port --agent-id $aid `
        --steps $Steps --rate 0.02 --log-every 0 --binding-scenario $Scenario `
        *> "$RunDir\client.log"

    Write-Host "`n=== binding smoke verdict ===" -ForegroundColor Cyan
    & $Py "scripts\check_binding_smoke.py" $BaseUrl $aid 3 *>&1 |
        Tee-Object -FilePath "$RunDir\smoke_verdict.log"
}
finally {
    if ($Server -and -not $Server.HasExited) {
        Stop-Process -Id $Server.Id -Force -ErrorAction SilentlyContinue
    }
    foreach ($n in $TouchedEnv) {
        if ($null -eq $SavedEnv[$n]) {
            Remove-Item "Env:$n" -ErrorAction SilentlyContinue
        } else {
            Set-Item "Env:$n" $SavedEnv[$n]
        }
    }
}
Write-Host "artifacts in $RunDir" -ForegroundColor Green
