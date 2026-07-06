# Body-rig A/B diagnostic (2026-07-04): is the kuzu graph write path the
# fsync storm? Runs server+body with a CLEAN explicit env, graph backend
# swappable, samples cycle rate, captures [body] budget lines, writes a
# verdict summary, and tears everything down. Self-terminating.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_body_diag.ps1
#         ... run_body_diag.ps1 -GraphBackend kuzu   (the control arm)
param(
    [string]$GraphBackend = "sqlite",
    [int]$Seconds = 150,
    [int]$Port = 8765,
    # -Watch: open the native MuJoCo viewer + the web dashboard so the run
    # is observable. NB the dashboard's polling adds a small, known observer
    # cost (state snapshots); fine for soaks, avoid for tight A/B numbers.
    [switch]$Watch
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
# Let the dashboard's Give-directly / Place-nearby buttons drive THIS standalone
# body (it's a live control consumer on the cycle ws, just not a supervised
# "scenario"). Lets you feed the agent mid-run to watch viability<->tempo.
$env:DECADIC_ALLOW_EXTERNAL_BODY_PROVISION = "1"
# Gate log: OFF for headless A/B (clean timing), ON when watching so the
# per-cycle deliberation/viability/drive curve is captured to correlate against
# feeding events. It's off-thread now (~microseconds/cycle), so ON barely
# taxes the run. Resource grants + retrievals are logged either way (server log).
$env:DECADIC_GATE_LOG = $(if ($Watch) { "1" } else { "0" })

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

    $BodyArgs = @("scripts\mujoco_decadic_adapter.py", "--port", "$Port")
    if ($Watch) { $BodyArgs += "--view" }
    $mode = if ($Watch) { "native viewer" } else { "headless" }
    Write-Host "[diag] starting MuJoCo body ($mode, runs until teardown)..." -ForegroundColor Cyan
    $Body = Start-Process -FilePath $Py `
        -ArgumentList $BodyArgs `
        -WorkingDirectory $Root -PassThru -NoNewWindow `
        -RedirectStandardOutput "$RunDir\body.out.log" -RedirectStandardError "$RunDir\body.err.log"

    $Dash = $null
    if ($Watch) {
        # Web dashboard: reuse a running dev server on 5173, else start one.
        $dashUrl = "http://localhost:5173"
        $dashUp = $false
        try {
            Invoke-WebRequest $dashUrl -TimeoutSec 2 -UseBasicParsing | Out-Null
            $dashUp = $true
        } catch {}
        if (-not $dashUp) {
            Write-Host "[diag] starting dashboard dev server..." -ForegroundColor Cyan
            $Dash = Start-Process -FilePath "cmd" `
                -ArgumentList "/c", "npm run dev" `
                -WorkingDirectory (Join-Path $Root "dashboard") -PassThru -WindowStyle Minimized
            Start-Sleep -Seconds 6
        }
        Start-Process $dashUrl
    }

    # Sample the cycle rate from /agents (cheap; no /state perturbation).
    $samples = @()
    $t0 = Get-Date
    $lastCycles = -1
    $AgentId = $null
    $EndsAt = $t0.AddSeconds($Seconds)
    Write-Host ("[diag] running {0:N0} min | ends at {1:HH:mm:ss} | graph={2}{3}" -f `
        ($Seconds / 60.0), $EndsAt, $GraphBackend, $(if ($Watch) { " | watching" } else { "" })) `
        -ForegroundColor Green
    while (((Get-Date) - $t0).TotalSeconds -lt $Seconds) {
        Start-Sleep -Seconds 10
        try {
            $agAll = (Invoke-RestMethod "$BaseUrl/agents" -TimeoutSec 5).agents
            # Run hygiene (2026-07-06): a second brain started mid-run (e.g. a
            # dashboard scenario) invalidates every number this diag reports --
            # two 204M-param models share one GPU/server. Warn loudly.
            if ($agAll.Count -gt 1) {
                Write-Host ("[diag] WARNING: {0} agents on this server -- measurements are now confounded (second brain competing for GPU/CPU/IO)" -f $agAll.Count) -ForegroundColor Red
            }
            $ag = $agAll | Select-Object -First 1
            if ($null -ne $ag) {
                $samples += [pscustomobject]@{
                    t_s = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
                    cycles = $ag.cycles_completed
                }
                $lastCycles = $ag.cycles_completed
                $AgentId = $ag.agent_id
                # Informative status line: countdown, progress %, live + avg rate.
                $tNow = $samples[-1].t_s
                $left = [math]::Max(0, $Seconds - $tNow)
                $pct = [math]::Min(100, [math]::Round(100.0 * $tNow / $Seconds))
                $fmt = { param($s) "{0:d2}:{1:d2}" -f [int]([math]::Floor($s / 60)), [int]($s % 60) }
                $instRate = ""
                if ($samples.Count -ge 2) {
                    $prev = $samples[-2]
                    $dt = [math]::Max(0.1, $tNow - $prev.t_s)
                    $instRate = "{0:N1}/s now, " -f (($ag.cycles_completed - $prev.cycles) / $dt)
                }
                $avgRate = "{0:N1}/s avg" -f ($ag.cycles_completed / [math]::Max(0.1, $tNow))
                Write-Host ("[diag] {0} elapsed | {1} left ({2}%) | cycles={3} | {4}{5}" -f `
                    (& $fmt $tNow), (& $fmt $left), $pct, $ag.cycles_completed, $instRate, $avgRate)
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
    # Only stop the dashboard if THIS run started it (leave a pre-existing one).
    # IMPORTANT: `cmd /c npm run dev` spawns cmd -> npm -> node(Vite). Stopping
    # the cmd PID alone ORPHANS the node/Vite child, which then runs for the whole
    # session and bloats (HMR + module graph + the Three.js/graph panels) -- the
    # cause of the RAM creep. taskkill /T kills the entire process tree so the
    # Vite node dies too; the port-based sweep catches any stray that outlived it.
    if ($Dash -and -not $Dash.HasExited) {
        & taskkill /F /T /PID $Dash.Id 2>$null | Out-Null
        Stop-Process -Id $Dash.Id -Force -ErrorAction SilentlyContinue
        # Belt-and-suspenders: kill whatever still holds the Vite port (5173).
        try {
            $viteConns = Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue
            foreach ($c in $viteConns) {
                Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        } catch {}
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
        -Pattern '"(graph_flush_ms|graph_flush_lock_ms|graph_flush_queue_depth|graph_flush_rows|graph_flush_stmts|graph_flush_error_batches|graph_writes_deferred|graph_coalesce_dedup_rows|graph_deferred_depth|sqlite_last_commit_ms|sqlite_commit_count|sqlite_batch_commit_count|kuzu_vector_index_rebuilds|ltm_match_ms|commit_lag_ms|lance_last_commit_ms|match_cache_hits|match_cache_misses|growth_events|awake_neurons|plasticity_pc_ema|growth_blocked_reason)[^,]*' |
        ForEach-Object { $_.Matches.Value.Trim() } | Select-Object -First 28
    if ($gm) { $sum += "store telemetry:"; $sum += ($gm | ForEach-Object { "  $_" }) }
}
# Memory-accumulation trend: is per-cycle recall/match latency creeping up as
# the store grows? First vs last few samples make a monotonic climb obvious.
$memLines = @()
foreach ($log in @("$RunDir\decadic_server.jsonl", "$RunDir\server.err.log")) {
    if (Test-Path $log) {
        $memLines += Select-String -Path $log -Pattern "memory_accumulation" -ErrorAction SilentlyContinue |
            ForEach-Object { ($_.Line -replace '.*memory_accumulation ', '') }
    }
}
if ($memLines.Count -ge 2) {
    $sum += "memory accumulation (first 2 vs last 2 samples -- watch recall_ms/ltm_match_ms/ltm_nodes):"
    $sum += ($memLines | Select-Object -First 2 | ForEach-Object { "  $_" })
    $sum += "  ..."
    $sum += ($memLines | Select-Object -Last 2 | ForEach-Object { "  $_" })
}
# WS-FORAGE: successor value = incentive salience. It must climb off ~0 for the
# agent to ever pursue a resource; a flat ~0 means it hasn't yet lived enough
# earned approach->relief episodes to learn what's worth going for.
if (Test-Path "$RunDir\metrics.json") {
    $svLine = Select-String -Path "$RunDir\metrics.json" -Pattern '"(raw|weighted)"\s*:\s*[-0-9.eE]+' -ErrorAction SilentlyContinue |
        Select-Object -First 2 -ExpandProperty Line | ForEach-Object { $_.Trim() }
    if ($svLine) { $sum += "successor value (must climb off ~0 to forage):"; $sum += ($svLine | ForEach-Object { "  $_" }) }
}
# WS-FORAGE: what is the agent THINKING? Gate-reason breakdown surfaces how often
# it dropped into the deliberate path, and WHY -- including type2_memory_search
# (need + remembered-but-not-here -> pursue from memory).
$gateLog = Get-ChildItem "$RunDir\gate_decisions_*.jsonl" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($gateLog) {
    $reasons = @{}; $total = 0; $esc = 0
    Get-Content $gateLog.FullName | ForEach-Object {
        if ($_ -match '"reason":"([a-z_0-9]+)"') {
            $r = $Matches[1]; $reasons[$r] = 1 + $reasons[$r]; $total++
            if ($r -ne "skip") { $esc++ }
        }
    }
    if ($total -gt 0) {
        $escPct = [math]::Round(100.0 * $esc / $total, 1)
        $sum += "thinking (gate reasons over $total cycles; deliberated ${escPct}%):"
        foreach ($k in ($reasons.Keys | Sort-Object { - $reasons[$_] })) {
            $sum += ("  {0,-22} {1,7}  ({2}%)" -f $k, $reasons[$k], [math]::Round(100.0 * $reasons[$k] / $total, 1))
        }
    }
}
$bodyTail = Get-Content "$RunDir\body.err.log" -Tail 5 -ErrorAction SilentlyContinue
if ($bodyTail) { $sum += "body stderr tail:"; $sum += $bodyTail }
$sum | Set-Content "$RunDir\summary.txt" -Encoding ascii
Write-Host "`n=== body diag summary ===" -ForegroundColor Cyan
$sum | ForEach-Object { Write-Host $_ }
Write-Host "artifacts in $RunDir" -ForegroundColor Green
Start-Sleep -Seconds 8
