# Body-rig A/B diagnostic (2026-07-04): is the kuzu graph write path the
# fsync storm? Runs server+body with a CLEAN explicit env, graph backend
# swappable, samples cycle rate, captures [body] budget lines, writes a
# verdict summary, and tears everything down. Self-terminating.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_body_diag.ps1
#         ... run_body_diag.ps1 -GraphBackend kuzu   (the control arm)
param(
    [string]$GraphBackend = "sqlite",
    [int]$Seconds = 150,
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $Root "reports\bodydiag_${GraphBackend}_$Stamp"
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$BaseUrl = "http://127.0.0.1:$Port"

# Kill anything already on the port (stale servers have haunted this port all day).
try { & powershell -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\kill_8765.ps1") | Out-Null } catch {}
Start-Sleep -Seconds 2

# CLEAN explicit environment -- this process only, nothing inherited matters.
$env:DECADIC_SELF_HOST = "127.0.0.1"
$env:DECADIC_SELF_PORT = "$Port"
$env:DECADIC_NEURAL_PRESET = "full"
$env:DECADIC_ENCODER_MODE = "hf"
$env:DECADIC_DEVICE = "cuda"
$env:DECADIC_EPISODIC_ASYNC = "1"
$env:DECADIC_LTM_ASYNC = "1"
$env:DECADIC_PREFETCH_OVERLOAD_POLICY = "drop_oldest"
$env:DECADIC_N_ACTUATORS = "21"
$env:DECADIC_LOG_DIR = $RunDir
$env:DECADIC_GRAPH_BACKEND = $GraphBackend
$env:DECADIC_BODY_COMMAND_STALE_S = "5"
# All validated faculties ride their new ON defaults; gate log stays off.

$Server = $null
$Body = $null
try {
    Write-Host "[diag] starting server (graph=$GraphBackend)..." -ForegroundColor Cyan
    $Server = Start-Process -FilePath $Py `
        -ArgumentList "-m", "uvicorn", "decadic.api.app:app", "--host", "127.0.0.1", "--port", "$Port" `
        -WorkingDirectory $Root -PassThru -NoNewWindow `
        -RedirectStandardOutput "$RunDir\server.out.log" -RedirectStandardError "$RunDir\server.err.log"
    $Ready = $false
    foreach ($i in 1..90) {
        Start-Sleep -Seconds 2
        try { Invoke-RestMethod "$BaseUrl/agents" -TimeoutSec 3 | Out-Null; $Ready = $true; break }
        catch { if ($Server.HasExited) { throw "server exited early - see $RunDir\server.err.log" } }
    }
    if (-not $Ready) { throw "server not ready" }

    Write-Host "[diag] starting MuJoCo body (headless, runs until teardown)..." -ForegroundColor Cyan
    $Body = Start-Process -FilePath $Py `
        -ArgumentList "scripts\mujoco_decadic_adapter.py", "--port", "$Port" `
        -WorkingDirectory $Root -PassThru -NoNewWindow `
        -RedirectStandardOutput "$RunDir\body.out.log" -RedirectStandardError "$RunDir\body.err.log"

    # Sample the cycle rate from /agents (cheap; no /state perturbation).
    $samples = @()
    $t0 = Get-Date
    $lastCycles = -1
    $AgentId = $null
    while (((Get-Date) - $t0).TotalSeconds -lt $Seconds) {
        Start-Sleep -Seconds 10
        try {
            $ag = (Invoke-RestMethod "$BaseUrl/agents" -TimeoutSec 5).agents | Select-Object -First 1
            if ($null -ne $ag) {
                $samples += [pscustomobject]@{
                    t_s = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
                    cycles = $ag.cycles_completed
                }
                $lastCycles = $ag.cycles_completed
                $AgentId = $ag.agent_id
                Write-Host ("[diag] t={0}s cycles={1}" -f $samples[-1].t_s, $ag.cycles_completed)
            }
        } catch { Write-Host "[diag] sample failed: $($_.Exception.Message)" }
        if ($Body.HasExited) { Write-Host "[diag] body exited early!" -ForegroundColor Red; break }
    }
    # Grab the full metrics dump BEFORE teardown: it carries the graph/store
    # timing telemetry (commit ms, match ms, prefetch/queue stats) that
    # attributes any remaining slowdown.
    if ($AgentId) {
        try {
            Invoke-RestMethod "$BaseUrl/agent/$AgentId/metrics" -TimeoutSec 20 |
                ConvertTo-Json -Depth 6 | Set-Content "$RunDir\metrics.json" -Encoding ascii
        } catch { Write-Host "[diag] metrics fetch failed: $($_.Exception.Message)" }
    }
}
finally {
    foreach ($p in @($Body, $Server)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
}

# ---- verdict summary -------------------------------------------------------
$sum = @()
$sum += "body diag: graph_backend=$GraphBackend seconds=$Seconds stamp=$Stamp"
if ($samples.Count -ge 2) {
    $first = $samples[0]; $last = $samples[-1]
    $rate = ($last.cycles - $first.cycles) / [math]::Max(1.0, ($last.t_s - $first.t_s))
    $sum += ("cycle rate (steady window): {0:N2} cycles/s  ({1} -> {2} cycles over {3}s)" -f `
        $rate, $first.cycles, $last.cycles, ($last.t_s - $first.t_s))
    $sum += ($samples | ForEach-Object { "  t=$($_.t_s)s cycles=$($_.cycles)" })
} else {
    $sum += "insufficient samples - see logs"
}
$budget = Select-String -Path "$RunDir\body.out.log" -Pattern "budget ms/obs" -ErrorAction SilentlyContinue |
    Select-Object -Last 3 -ExpandProperty Line
if ($budget) { $sum += "body budget (last 3):"; $sum += $budget }
if (Test-Path "$RunDir\metrics.json") {
    $gm = Select-String -Path "$RunDir\metrics.json" `
        -Pattern '"(graph_flush_ms|graph_flush_lock_ms|graph_flush_queue_depth|graph_flush_error_batches|sqlite_last_commit_ms|sqlite_commit_count|sqlite_batch_commit_count|kuzu_vector_index_rebuilds|ltm_match_ms|commit_lag_ms|lance_last_commit_ms|match_cache_hits|match_cache_misses)[^,]*' |
        ForEach-Object { $_.Matches.Value.Trim() } | Select-Object -First 16
    if ($gm) { $sum += "store telemetry:"; $sum += ($gm | ForEach-Object { "  $_" }) }
}
$bodyTail = Get-Content "$RunDir\body.err.log" -Tail 5 -ErrorAction SilentlyContinue
if ($bodyTail) { $sum += "body stderr tail:"; $sum += $bodyTail }
$sum | Set-Content "$RunDir\summary.txt" -Encoding ascii
Write-Host "`n=== body diag summary ===" -ForegroundColor Cyan
$sum | ForEach-Object { Write-Host $_ }
Write-Host "artifacts in $RunDir" -ForegroundColor Green
Start-Sleep -Seconds 8
