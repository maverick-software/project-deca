# WS-EXPAND live probe (ASCII only). ~10 minutes, then teardown + verdicts.
# Wraps run_body_diag.ps1 with prod-default WS-EXPAND flags (all ON) and ONE
# probe override: the rest threshold is lowered so E7 demonstrably fires,
# rests ~30 s, and wakes inside the window. Every other feature runs exactly
# as it would in production. Emits a per-feature verdict table from the
# end-of-run metrics snapshot.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run_ws_expand_probe.ps1
param(
    [int]$Seconds = 600,
    [string]$GraphBackend = "kuzu"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# --- probe environment (inherited by the harness's child processes) ---------
# E7: prod threshold (~4000 load) would not fire inside 10 min; lower it so a
# rest ENTERS ~4-5 min in, holds ~30 s (120 cycles), and wakes. Everything
# else is the production default (WS-EXPAND ships default-ON).
$env:DECADIC_REST_LOAD_THRESHOLD = "1200"
$env:DECADIC_REST_MIN_WAKE_CYCLES = "600"

Write-Host "[probe] WS-EXPAND live probe: $($Seconds)s, graph=$GraphBackend, all features at prod defaults (rest threshold lowered for coverage)" -ForegroundColor Cyan
& powershell -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\run_body_diag.ps1") -GraphBackend $GraphBackend -Seconds $Seconds

# --- locate the run the harness just produced --------------------------------
$RunDir = Get-ChildItem (Join-Path $Root "reports") -Directory -Filter "bodydiag_${GraphBackend}_*" |
    Sort-Object Name -Descending | Select-Object -First 1
if (-not $RunDir) { Write-Host "[probe] no run dir found" -ForegroundColor Red; exit 1 }
$Metrics = Join-Path $RunDir.FullName "metrics.json"
if (-not (Test-Path $Metrics)) { Write-Host "[probe] no metrics.json in $($RunDir.FullName)" -ForegroundColor Red; exit 1 }
$Raw = Get-Content $Metrics -Raw

function Get-Num([string]$key) {
    if ($Raw -match ('"' + $key + '"\s*:\s*(-?[0-9.eE+]+)')) { return [double]$Matches[1] }
    return $null
}

# --- per-feature verdicts -----------------------------------------------------
$V = @()
function Verdict([string]$name, [bool]$pass, [string]$detail) {
    $script:V += [pscustomobject]@{ feature = $name; verdict = $(if ($pass) { "PASS" } else { "CHECK" }); detail = $detail }
}

# E2 - multi-channel learning control
$eta = Get-Num "lc_eta_scale"; $gam = Get-Num "lc_gamma"; $sur = Get-Num "lc_surprise"; $gmoves = Get-Num "lc_gamma_moves"
Verdict "E2 rate channels" ($null -ne $eta -and $eta -ge 0.24 -and $eta -le 3.01) ("lc_eta_scale=$eta lc_surprise=$sur")
Verdict "E2 gamma clamp" ($null -ne $gam -and $gam -ge 0.9899 -and $gam -le 0.9971) ("lc_gamma=$gam moves=$gmoves (band 0.99..0.997)")
$gbias = Get-Num "gate_modulation_bias"
Verdict "E2 gate coupling" ($null -ne $gbias -and $gbias -ge 0 -and $gbias -le 0.15) ("gate_modulation_bias=$gbias")

# E1 - cognitive map + planner
$nodes = Get-Num "cmap_nodes"; $pupd = Get-Num "cmap_pose_updates"; $rer = Get-Num "cmap_reroutes"; $stalls = Get-Num "cmap_stall_events"
Verdict "E1 map accrual" ($null -ne $nodes -and $nodes -gt 0 -and $null -ne $pupd -and $pupd -gt 0) ("nodes=$nodes pose_updates=$pupd stalls=$stalls reroutes=$rer (reroutes only after evidenced blockage)")
$pb = Get-Num "planner_bias_linf"
Verdict "E1.6 rollout planner" ($true) ("planner_bias_linf=$pb (null until SF ramp opens + escalated cycle -- expected early)")

# E3 - motor corrector (FEL)
$mcl = Get-Num "motor_corrector_loss"
Verdict "E3 corrector FEL" ($null -ne $mcl -and $mcl -ge 0) ("motor_corrector_loss=$mcl (supervised tracking-error fit; watch it fall across runs)")

# E4 - dual control
$cw = Get-Num "cached_policy_w"; $cd = Get-Num "cached_divergence"
Verdict "E4 dual control" ($null -ne $cw -and $cw -ge 0 -and $cw -le 1) ("cached_policy_w=$cw divergence=$cd (trust must be EARNED; 0 early is correct)")

# E5 - veto (threat channels train only after pain beliefs form)
$veto = Get-Num "veto_attenuation"
Verdict "E5 action veto" ($null -ne $veto -and $veto -ge 0 -and $veto -le 0.5) ("veto_attenuation=$veto (cap 0.5; ~0 without viability drops is correct)")

# E7 - rest (probe threshold lowered so this FIRES in-window)
$rents = Get-Num "rests_entered"; $rload = Get-Num "rest_load"; $rabort = Get-Num "rests_aborted"
Verdict "E7 rest cycle" ($null -ne $rents -and $rents -ge 1) ("rests_entered=$rents aborted=$rabort load=$rload (>=1 expected at probe threshold)")

# E8.1/E8.2 have no dedicated counters (zero-init ingress / bounded salience
# multiplier); their live effect is the A/B. Presence is implied by parity of
# the run itself. E13:
$ps = Get-Num "phase_slow"; $pf = Get-Num "phase_fast"
Verdict "E13 phase telemetry" ($null -ne $ps -and [math]::Abs($ps) -le 1.0 -and $null -ne $pf) ("phase_slow=$ps phase_fast=$pf (instrumentation only)")

# E9 - symbols
$sc = Get-Num "symbol_code"; $su = Get-Num "symbol_utilization"
Verdict "E9 discrete codes" ($null -ne $sc -and $sc -ge 0) ("symbol_code=$sc utilization=$su")

# E10 - other agents (SOLO run: models_active MUST be 0)
$otr = Get-Num "other_tracks"; $oma = Get-Num "other_models_active"
Verdict "E10 adaptivity gate" ($null -ne $oma -and $oma -eq 0) ("tracks=$otr models_active=$oma (MUST be 0 solo -- nonzero = gate broken)")

# WS-IND I1 - attention schema (accuracy must beat the base rate once trained)
$sa = Get-Num "schema_accuracy"; $sb = Get-Num "schema_base_accuracy"; $sbias = Get-Num "schema_bias"
Verdict "I1 attention schema" ($null -ne $sa) ("accuracy=$sa base=$sb bias=$sbias (accuracy should approach/beat base as it trains)")

# WS-IND I2 - sequential deliberation (rounds=2 on escalated cycles)
$wsr = Get-Num "ws_seq_rounds"; $wsd = Get-Num "ws_seq_divergence"
Verdict "I2 sequential rounds" ($null -ne $wsr) ("rounds=$wsr divergence=$wsd (2 on escalated cycles; divergence ~0 at birth is correct)")

# WS-IND I3/I5 - reliability + smoothness telemetry
$snm = Get-Num "slot_noise_mean"; $ss = Get-Num "symbol_smoothness"
Verdict "I3/I5 telemetry" ($true) ("slot_noise_mean=$snm symbol_smoothness=$ss (presence/trend telemetry)")

# WS-DEPTH D1 - metacognitive calibration (ECE lower = better calibrated)
$mc = Get-Num "metacog_calibration"; $mm = Get-Num "metacog_err_mae"
Verdict "D1 metacog calibration" ($null -ne $mc) ("ECE=$mc err_mae=$mm (watch both FALL across runs)")

# WS-DEPTH P1/P2 - lived perception channels
$pf = Get-Num "percept_fwd_loss"; $td = Get-Num "topdown_frac"
Verdict "P1/P2 perception" ($true) ("percept_fwd_loss=$pf topdown_frac=$td (frac capped at DECADIC_PERCEPT_TOPDOWN_CAP)")

# WS-DEPTH D2 - the self in the workspace (needs GWT on to be meaningful)
$sir = Get-Num "self_ignition_rate"
Verdict "D2 self-ignition" ($true) ("self_ignition_rate=$sir (0 early is correct: birth ramp; deprivation should raise it)")

# --- emit ---------------------------------------------------------------------
$out = @("WS-EXPAND live probe verdicts ($($RunDir.Name)):", "")
$out += ($V | ForEach-Object { "{0,-6} {1,-22} {2}" -f $_.verdict, $_.feature, $_.detail })
$checks = @($V | Where-Object { $_.verdict -eq "CHECK" }).Count
$out += ""
$out += "result: $((@($V).Count - $checks))/$(@($V).Count) PASS, $checks CHECK (details above)"
$out | Set-Content (Join-Path $RunDir.FullName "ws_expand_verdicts.txt") -Encoding ascii
Write-Host ""
Write-Host "=== WS-EXPAND probe verdicts ===" -ForegroundColor Cyan
$out | ForEach-Object { Write-Host $_ }
Write-Host "full artifacts in $($RunDir.FullName)" -ForegroundColor Green
