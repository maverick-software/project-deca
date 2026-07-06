# Foraging-curriculum harness (WS-FORAGE). Purpose: give the successor-features
# value the ONE thing it needs to climb off zero -- repeated EARNED approach->
# relief episodes -- by keeping a reachable resource in front of a genuinely
# hungry agent that cannot die.
#
# Design note (important): true Immortal mode PINS reservoirs at full, so deficit
# = 0, and the foraging drive (deficit-gated incentive salience) is zero -- an
# immortal agent has no reason to reach for food and learns nothing. So the
# curriculum runs METABOLIC (it gets hungry -> motivated) with a SURVIVAL SAFETY
# NET: the harness rescues a reservoir just before it would be fatal, keeping the
# agent alive but still hungry. -Immortal is available but only builds the
# object->relief association, NOT foraging; it will warn.
#
# The loop, every -PlaceEverySec:
#   1. read the agent's reservoirs,
#   2. place the resource for the MOST-DEPLETED reservoir WITHIN REACH (matches
#      the active need, so drive and opportunity align; consumption is
#      contact-gated so the agent must reach it -- an earned meal),
#   3. if any reservoir is near-fatal, admin-rescue it up to a still-hungry floor.
# Consumption is never hand-delivered (no "give directly"), so every meal the
# agent gets is one it earned -- exactly the episodes the value learner needs.
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run_foraging_curriculum.ps1 -Watch
param(
    [int]$Seconds = 3600,
    [int]$Port = 8765,
    [switch]$Watch,
    [switch]$Immortal,               # honored, but disables the foraging drive (warns)
    [double]$RescueFloor = 12.0,     # rescue a reservoir that falls below this (metabolic net)
    [double]$RescueTo = 35.0,        # ...crediting it up to here (kept hungry, not sated)
    [int]$PlaceEverySec = 8,         # re-place a within-reach resource this often
    [double]$ReachDistance = 0.6,    # within-reach placement distance (m)
    [double]$ContactRadius = 0.35    # body-part must be within this to consume (m); a touch
)                                    #   looser than the 0.30 default to help the first successes

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $Root "reports\forage_$Stamp"
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$BaseUrl = "http://127.0.0.1:$Port"

try { & powershell -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\kill_8765.ps1") | Out-Null } catch {}
Start-Sleep -Seconds 2

# --- clean env -------------------------------------------------------------
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
$env:DECADIC_GRAPH_BACKEND = "kuzu"
$env:DECADIC_BODY_COMMAND_STALE_S = "5"
# Foraging-specific:
$env:DECADIC_VIABILITY_MODE = $(if ($Immortal) { "immortal" } else { "metabolic" })
$env:DECADIC_BODY_CONSUME_MODE = "contact"                # every meal must be reached for
$env:DECADIC_BODY_CONTACT_RADIUS = "$ContactRadius"
$env:DECADIC_BODY_REACH_DISTANCE = "$ReachDistance"
$env:DECADIC_ALLOW_EXTERNAL_BODY_PROVISION = "1"          # let this harness provision the body
$env:DECADIC_GATE_LOG = "1"                               # capture the learning telemetry
# SF value regime rides its (now default-on) settings.

if ($Immortal) {
    Write-Host "[forage] WARNING: -Immortal pins reservoirs at full -> deficit=0 -> the" -ForegroundColor Yellow
    Write-Host "[forage] foraging drive (deficit-gated value) is ZERO. The agent will NOT" -ForegroundColor Yellow
    Write-Host "[forage] learn to forage this way; it only builds the object->relief cue." -ForegroundColor Yellow
    Write-Host "[forage] Use metabolic (default) to actually teach approach." -ForegroundColor Yellow
}

$Server = $null; $Body = $null; $Dash = $null
$rescues = 0
$firstConsume = $null; $lastConsume = 0
$samples = @()

function Get-AgentMetrics($aid) {
    try { return (Invoke-RestMethod "$BaseUrl/agent/$aid/metrics" -TimeoutSec 6).metrics } catch { return $null }
}
function Give($aid, $res, $mode, $amount) {
    $url = "$BaseUrl/agent/$aid/give?resource=$res&mode=$mode"
    if ($amount -ne $null) { $url += "&amount=$amount" }
    try { Invoke-RestMethod $url -Method Post -TimeoutSec 6 | Out-Null; return $true } catch { return $false }
}

try {
    Write-Host "[forage] starting server (metabolic=$(-not $Immortal), contact-consume, reach=$ReachDistance/$ContactRadius m)..." -ForegroundColor Cyan
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
    Write-Host "[forage] starting MuJoCo body..." -ForegroundColor Cyan
    $Body = Start-Process -FilePath $Py -ArgumentList $BodyArgs `
        -WorkingDirectory $Root -PassThru -NoNewWindow `
        -RedirectStandardOutput "$RunDir\body.out.log" -RedirectStandardError "$RunDir\body.err.log"

    if ($Watch) {
        $dashUrl = "http://localhost:5173"; $dashUp = $false
        try { Invoke-WebRequest $dashUrl -TimeoutSec 2 -UseBasicParsing | Out-Null; $dashUp = $true } catch {}
        if (-not $dashUp) {
            $Dash = Start-Process -FilePath "cmd" -ArgumentList "/c", "npm run dev" `
                -WorkingDirectory (Join-Path $Root "dashboard") -PassThru -WindowStyle Minimized
            Start-Sleep -Seconds 6
        }
        Start-Process $dashUrl
    }

    $t0 = Get-Date
    $EndsAt = $t0.AddSeconds($Seconds)
    $AgentId = $null
    Write-Host ("[forage] running {0:N0} min | ends {1:HH:mm:ss} | rescue<{2}->{3} | place every {4}s" -f `
        ($Seconds / 60.0), $EndsAt, $RescueFloor, $RescueTo, $PlaceEverySec) -ForegroundColor Green

    while (((Get-Date) - $t0).TotalSeconds -lt $Seconds) {
        Start-Sleep -Seconds $PlaceEverySec
        if ($Body.HasExited) { Write-Host "[forage] body exited early!" -ForegroundColor Red; break }
        try {
            $ag = (Invoke-RestMethod "$BaseUrl/agents" -TimeoutSec 5).agents | Select-Object -First 1
        } catch { Write-Host "[forage] sample failed: $($_.Exception.Message)"; continue }
        if ($null -eq $ag) { continue }
        $AgentId = $ag.agent_id
        $m = Get-AgentMetrics $AgentId
        if ($null -eq $m) { continue }

        $hyd = [double]$m.hydration; $enr = [double]$m.energy; $intg = [double]$m.integrity
        $via = [double]$m.viability; $cons = [int]$m.consume_events
        if ($null -eq $firstConsume) { $firstConsume = $cons }
        $lastConsume = $cons

        # 1) place the resource for the MOST-DEPLETED reservoir within reach.
        $res = "food"; $lo = $enr
        if ($hyd -lt $lo) { $res = "water"; $lo = $hyd }
        if ($intg -lt $lo) { $res = "medical_kit"; $lo = $intg }
        [void](Give $AgentId $res "within_reach" $null)

        # 2) survival safety net (metabolic only): rescue anything near-fatal.
        if (-not $Immortal) {
            foreach ($pair in @(@("water", $hyd), @("food", $enr), @("medical_kit", $intg))) {
                if ([double]$pair[1] -lt $RescueFloor) {
                    $amt = [math]::Round($RescueTo - [double]$pair[1], 1)
                    if (Give $AgentId $pair[0] "admin" $amt) { $rescues++ }
                }
            }
        }

        $tNow = [int]((Get-Date) - $t0).TotalSeconds
        $samples += [pscustomobject]@{ t = $tNow; cons = $cons; via = $via }
        Write-Host ("[forage] {0,5}s | cyc={1} | viab={2:N1} (H{3:N0} E{4:N0}) | offered={5} within-reach | EARNED meals={6} | rescues={7}" -f `
            $tNow, $ag.cycles_completed, $via, $hyd, $enr, $res, $cons, $rescues)
    }

    if ($AgentId) {
        try {
            Invoke-RestMethod "$BaseUrl/agent/$AgentId/metrics" -TimeoutSec 20 |
                ConvertTo-Json -Depth 8 | Set-Content "$RunDir\metrics.json" -Encoding ascii
        } catch { Write-Host "[forage] metrics fetch failed: $($_.Exception.Message)" }
    }
}
finally {
    foreach ($p in @($Body, $Server)) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
    if ($Dash -and -not $Dash.HasExited) { Stop-Process -Id $Dash.Id -Force -ErrorAction SilentlyContinue }
}

# --- verdict ---------------------------------------------------------------
$sum = @()
$sum += "foraging curriculum: mode=$($env:DECADIC_VIABILITY_MODE) seconds=$Seconds stamp=$Stamp"
$mealsEarned = if ($null -ne $firstConsume) { $lastConsume - $firstConsume } else { 0 }
$sum += "EARNED meals over the run (consume_events delta): $mealsEarned   [rescues: $rescues]"
$sum += "  -> did the agent learn to reach? watch this climb across runs, and successor_value below climb off ~0."
if (Test-Path "$RunDir\metrics.json") {
    $sv = Select-String -Path "$RunDir\metrics.json" -Pattern '"(raw|weighted)":\s*-?[0-9.eE-]+' |
        Select-Object -First 2 -ExpandProperty Line
    if ($sv) { $sum += "successor_value (SF incentive salience; should grow above ~0 as it learns):"; $sum += ($sv | ForEach-Object { "  $($_.Trim())" }) }
    $cg = Select-String -Path "$RunDir\metrics.json" -Pattern '"(consume_events|resource_relief_events|viability|hydration|energy)":\s*[0-9.]+' |
        ForEach-Object { $_.Matches.Value } | Select-Object -First 6
    if ($cg) { $sum += "final state:"; $sum += ($cg | ForEach-Object { "  $_" }) }
}
$sum | Set-Content "$RunDir\summary.txt" -Encoding ascii
Write-Host "`n=== foraging curriculum summary ===" -ForegroundColor Cyan
$sum | ForEach-Object { Write-Host $_ }
Write-Host "artifacts in $RunDir" -ForegroundColor Green
Start-Sleep -Seconds 6
