# Stall hunter - reproduce the cycle-loop freeze and capture an asyncio task
# dump at the moment it happens. Self-terminating (default 20 min hard cap).
# Usage: .\scripts\stall_hunt.ps1            # hunt defect 3 (growth off)
#        .\scripts\stall_hunt.ps1 -Growth    # hunt defect 2 (growth on)
param(
    [switch]$Growth,
    [int]$MaxMinutes = 20,
    [int]$StallSeconds = 60,
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $Root "reports\stallhunt_$Stamp"
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$BaseUrl = "http://127.0.0.1:$Port"

$env:DECADIC_SELF_HOST = "127.0.0.1"
$env:DECADIC_SELF_PORT = "$Port"
$env:DECADIC_NEURAL_PRESET = "full"
$env:DECADIC_PLASTICITY_ENABLED = "1"
$env:DECADIC_SPARSE_ENABLED = "1"
$env:DECADIC_GROWTH_ENABLED = $(if ($Growth) { "1" } else { "0" })
$env:DECADIC_ENCODER_MODE = "hf"
$env:DECADIC_DEVICE = "cuda"
$env:DECADIC_EPISODIC_ASYNC = "1"
$env:DECADIC_N_ACTUATORS = "21"
$env:DECADIC_CURRICULUM_MODE = "legacy"
$env:DECADIC_SELF_MODEL_FEEDBACK = "1"
$env:DECADIC_GWT_ENABLED = "1"
$env:DECADIC_INTEGRATION_WINDOW_MS = "200"
$env:DECADIC_PREDICTIVE_AFFECT = "1"
$env:DECADIC_REPRESENTED_SELF = "1"
$env:DECADIC_MEMORY_EFFICIENT_TRAINING = "1"
$env:DECADIC_PREFETCH_OVERLOAD_POLICY = "drop_oldest"
$env:DECADIC_LOG_DIR = $RunDir

$Server = $null
$Client = $null
$Result = "unknown"
try {
    Write-Host "starting server (growth=$($env:DECADIC_GROWTH_ENABLED))..." -ForegroundColor Cyan
    $Server = Start-Process -FilePath $Py `
        -ArgumentList "-m", "uvicorn", "decadic.api.app:app", "--host", "127.0.0.1", "--port", "$Port" `
        -WorkingDirectory $Root -PassThru -NoNewWindow `
        -RedirectStandardOutput "$RunDir\server.out.log" -RedirectStandardError "$RunDir\server.err.log"
    $Ready = $false
    foreach ($i in 1..120) {
        Start-Sleep -Seconds 2
        try { Invoke-RestMethod "$BaseUrl/agents" -TimeoutSec 3 | Out-Null; $Ready = $true; break }
        catch { if ($Server.HasExited) { throw "server exited early" } }
    }
    if (-not $Ready) { throw "server not ready" }

    $aid = (Invoke-RestMethod -Method Post -Uri "$BaseUrl/agent").agent_id
    Write-Host "agent: $aid"
    $Client = Start-Process -FilePath $Py `
        -ArgumentList "scripts\synthetic_ws_client.py", "--port", "$Port", "--agent-id", "$aid", `
            "--steps", "2000000", "--rate", "0.1", "--log-every", "1000" `
        -WorkingDirectory $Root -PassThru -NoNewWindow `
        -RedirectStandardOutput "$RunDir\client.out.log" -RedirectStandardError "$RunDir\client.err.log"

    $deadline = (Get-Date).AddMinutes($MaxMinutes)
    $lastCycle = -1
    $lastChange = Get-Date
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        $agents = (Invoke-RestMethod "$BaseUrl/agents" -TimeoutSec 5).agents
        $cycle = ($agents | Where-Object { $_.agent_id -eq $aid }).cycles_completed
        $ts = Get-Date -Format "HH:mm:ss"
        Write-Host "[$ts] cycle $cycle"
        if ($cycle -ne $lastCycle) {
            $lastCycle = $cycle
            $lastChange = Get-Date
        } elseif ($cycle -gt 0 -and ((Get-Date) - $lastChange).TotalSeconds -ge $StallSeconds) {
            Write-Host "STALL at cycle $cycle - capturing dumps" -ForegroundColor Red
            Invoke-RestMethod "$BaseUrl/debug/tasks" -TimeoutSec 30 |
                ConvertTo-Json -Depth 12 | Set-Content "$RunDir\stall_tasks.json"
            Invoke-RestMethod "$BaseUrl/agent/$aid/metrics" -TimeoutSec 30 |
                ConvertTo-Json -Depth 8 | Set-Content "$RunDir\stall_metrics.json"
            $Result = "stall_captured_at_cycle_$cycle"
            break
        }
    }
    if ($Result -eq "unknown") { $Result = "no_stall_within_${MaxMinutes}min_last_cycle_$lastCycle" }
}
catch { $Result = "error: $($_.Exception.Message)" }
finally {
    foreach ($p in @($Client, $Server)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
}
Set-Content "$RunDir\hunt_result.txt" $Result
Write-Host "RESULT: $Result (artifacts in $RunDir)" -ForegroundColor Green
