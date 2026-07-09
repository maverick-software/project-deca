# WS-DEPTH A2 — long-life run (ASCII only). Default 12 h; self-terminating.
# Wraps the body-diag harness and emits TREND verdicts over periodic metric
# snapshots: the trained capacities (schema accuracy, metacog calibration,
# habit trust vs divergence, code utilization, topdown_frac) must MATURE, not
# merely exist. Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_long_life.ps1 [-Hours 12]
param(
    [double]$Hours = 12,
    [string]$GraphBackend = "kuzu",
    [int]$SnapshotEveryS = 600,
    # A1 live probe folded into the soak: at minute N spawn the companion
    # SCRIPTED (the adaptivity gate must stay closed), then 5 min later flip
    # it ADAPTIVE (the gate must open, other_vec populates). 0 = no companion.
    [int]$CompanionAtMin = 0
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Seconds = [int]($Hours * 3600)

# Periodic snapshot job: poll /agent/<id>/metrics into a JSONL trend file
# alongside the harness (which handles start/teardown/summary itself).
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$TrendFile = Join-Path $Root "reports\longlife_trend_$Stamp.jsonl"
$Poller = Start-Job -ScriptBlock {
    param($BaseUrl, $TrendFile, $EveryS, $TotalS)
    # WS4C M4.2: the metrics endpoint NESTS everything under .metrics, so the
    # old property access ($m.schema_accuracy) read $null into every snapshot
    # (the nesting bug). Regex the RAW body instead -- the exact pattern the
    # body-diag probe uses; robust to nesting and JSON depth limits alike.
    $keys = @("schema_accuracy","schema_base_accuracy","metacog_calibration","metacog_err_mae",
              "cached_policy_w","cached_divergence","symbol_utilization","topdown_frac",
              "self_ignition_rate","percept_fwd_loss","motor_corrector_loss","lc_gamma",
              "other_tracks","other_models_active","rests_entered","rests_aborted",
              "viability","graph_write_pressure","graph_deferred_depth","commit_lag_ms")
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalSeconds -lt $TotalS) {
        Start-Sleep -Seconds $EveryS
        try {
            $ags = (Invoke-RestMethod "$BaseUrl/agents" -TimeoutSec 5).agents
            $ag = $ags | Select-Object -First 1
            if ($ag) {
                $raw = (Invoke-WebRequest "$BaseUrl/agent/$($ag.agent_id)/metrics" -TimeoutSec 20 -UseBasicParsing).Content
                $row = [ordered]@{ t = (Get-Date -Format o); cycles = $ag.cycles_completed }
                foreach ($k in $keys) {
                    if ($raw -match ('"' + $k + '"\s*:\s*(-?[0-9.eE+]+)')) { $row[$k] = [double]$Matches[1] }
                }
                # legacy aliases kept for downstream readers of old trend files
                $row["schema_base"] = $row["schema_base_accuracy"]
                $row["metacog_err_mae2"] = $row["metacog_err_mae"]
                $row | ConvertTo-Json -Compress | Add-Content $TrendFile
            }
        } catch {}
    }
} -ArgumentList "http://127.0.0.1:8765", $TrendFile, $SnapshotEveryS, $Seconds

$Comp = $null
if ($CompanionAtMin -gt 0) {
    $Comp = Start-Job -ScriptBlock {
        param($BaseUrl, $AtMin)
        Start-Sleep -Seconds ($AtMin * 60)
        try {
            $ag = ((Invoke-RestMethod "$BaseUrl/agents" -TimeoutSec 5).agents | Select-Object -First 1)
            if ($ag) {
                Invoke-RestMethod -Method Post "$BaseUrl/agent/$($ag.agent_id)/body/companion?action=spawn&mode=scripted" -TimeoutSec 5 | Out-Null
                Start-Sleep -Seconds 300
                Invoke-RestMethod -Method Post "$BaseUrl/agent/$($ag.agent_id)/body/companion?action=mode&mode=adaptive" -TimeoutSec 5 | Out-Null
            }
        } catch {}
    } -ArgumentList "http://127.0.0.1:8765", $CompanionAtMin
}

# WS-SOAK: a long-life run IS a soak test -- revive on death so the brain runs
# past a single ~100-min lifespan (viability=0 was the "freeze" all along).
& powershell -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\run_body_diag.ps1") -GraphBackend $GraphBackend -Seconds $Seconds -SoakRevive
Stop-Job $Poller -ErrorAction SilentlyContinue; Remove-Job $Poller -Force -ErrorAction SilentlyContinue
if ($Comp) { Stop-Job $Comp -ErrorAction SilentlyContinue; Remove-Job $Comp -Force -ErrorAction SilentlyContinue }

# ---- trend verdicts (first third vs last third of snapshots) ----------------
if (-not (Test-Path $TrendFile)) { Write-Host "[longlife] no trend snapshots captured" -ForegroundColor Red; exit 1 }
$rows = Get-Content $TrendFile | ForEach-Object { $_ | ConvertFrom-Json }
if ($rows.Count -lt 6) { Write-Host "[longlife] too few snapshots ($($rows.Count)) for trends" -ForegroundColor Yellow; exit 0 }
$n = $rows.Count; $a = $rows[0..([int]($n/3)-1)]; $b = $rows[([int](2*$n/3))..($n-1)]
function Avg($set, $k) { ($set | ForEach-Object { $_.$k } | Where-Object { $null -ne $_ } | Measure-Object -Average).Average }
function NZ($v) { if ($null -eq $v) { 0 } else { $v } }  # PS 5.1 has no ?? operator
$V = @()
function Trend([string]$name, $early, $late, [bool]$wantUp, [string]$note) {
    $ok = $false
    if ($null -ne $early -and $null -ne $late) { $ok = $(if ($wantUp) { $late -ge $early } else { $late -le $early }) }
    $script:V += ("{0,-6} {1,-24} {2} -> {3}  {4}" -f $(if ($ok) {"PASS"} else {"CHECK"}), $name, [math]::Round([double](NZ $early),4), [math]::Round([double](NZ $late),4), $note)
}
Trend "schema accuracy"      (Avg $a 'schema_accuracy')     (Avg $b 'schema_accuracy')     $true  "(should climb past base rate)"
Trend "metacog calibration"  (Avg $a 'metacog_calibration') (Avg $b 'metacog_calibration') $false "(ECE: lower = better)"
Trend "metacog error MAE"    (Avg $a 'metacog_err_mae')     (Avg $b 'metacog_err_mae')     $false "(self-assessment sharpens)"
Trend "corrector FEL"        (Avg $a 'motor_corrector_loss') (Avg $b 'motor_corrector_loss') $false "(tracking gap learned)"
Trend "percept fwd loss"     (Avg $a 'percept_fwd_loss')    (Avg $b 'percept_fwd_loss')    $false "(refinement earning keep)"
Trend "symbol utilization"   (Avg $a 'symbol_utilization')  (Avg $b 'symbol_utilization')  $true  "(code space filling)"
# WS4C M4.2: write pressure must stay under 1.0 across the whole life -- the
# death-spiral signature is a LATE crossing, invisible to end-only checks.
$wpLate = Avg $b 'graph_write_pressure'
if ($null -ne $wpLate) {
    $wpOk = $wpLate -lt 1.0
    $V += ("{0,-6} {1,-24} {2} -> {3}  {4}" -f $(if ($wpOk) {"PASS"} else {"RED"}), "write pressure", `
        [math]::Round([double](NZ (Avg $a 'graph_write_pressure')),4), [math]::Round([double](NZ $wpLate),4), "(late third must stay < 1.0)")
}
$clLate = Avg $b 'commit_lag_ms'
if ($null -ne $clLate) {
    $clOk = $clLate -le 60000
    $V += ("{0,-6} {1,-24} {2} -> {3}  {4}" -f $(if ($clOk) {"PASS"} else {"RED"}), "commit lag ms", `
        [math]::Round([double](NZ (Avg $a 'commit_lag_ms')),0), [math]::Round([double](NZ $clLate),0), "(deep-process staleness; RED > 60000)")
}
$V += ("INFO   trust w {0} -> {1} | divergence {2} -> {3} (judge together: rising trust with falling divergence is healthy)" -f `
    [math]::Round([double](NZ (Avg $a 'cached_policy_w')),4), [math]::Round([double](NZ (Avg $b 'cached_policy_w')),4), `
    [math]::Round([double](NZ (Avg $a 'cached_divergence')),4), [math]::Round([double](NZ (Avg $b 'cached_divergence')),4))
$V += ("INFO   self-ignition rate {0} -> {1} | topdown_frac {2} -> {3}" -f `
    [math]::Round([double](NZ (Avg $a 'self_ignition_rate')),5), [math]::Round([double](NZ (Avg $b 'self_ignition_rate')),5), `
    [math]::Round([double](NZ (Avg $a 'topdown_frac')),4), [math]::Round([double](NZ (Avg $b 'topdown_frac')),4))
$OutFile = Join-Path $Root "reports\longlife_verdicts_$Stamp.txt"
"WS-DEPTH long-life trends ($($rows.Count) snapshots over $Hours h):" | Set-Content $OutFile -Encoding ascii
$V | Add-Content $OutFile -Encoding ascii
Write-Host "`n=== long-life trend verdicts ===" -ForegroundColor Cyan
Get-Content $OutFile | ForEach-Object { Write-Host $_ }
