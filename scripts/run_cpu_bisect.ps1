# CPU-regression bisect (ASCII only). 2026-07-07: host CPU went 30-35% -> ~78%
# with the WS-EXPAND/IND/DEPTH features on. Every feature is env-gated, so
# attribution is five short arms. Each arm is a fresh 3-min body diag with one
# suspect group OFF; the harness now records host CPU per sample.
# Total ~18 min. Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_cpu_bisect.ps1
param([int]$Seconds = 180)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Suspect groups, ranked: (1) sequential deliberation doubles the forward pass
# on escalated cycles; (2) per-cycle distillation backward; (3) slot
# reliability does a GPU->CPU sync + pure-python per-dim loops every cycle;
# (4) the many small per-cycle training heads (schema/metacog/veto/inverse/
# percept-fwd) each cost dispatch + a .item() sync.
$Arms = @(
    @{ name = "all_on";        env = @{} },
    @{ name = "no_ws_seq";     env = @{ DECADIC_WS_SEQ = "0" } },
    @{ name = "no_distill";    env = @{ DECADIC_CACHED_POLICY = "0" } },
    @{ name = "no_slot_rel";   env = @{ DECADIC_SLOT_RELIABILITY = "0"; DECADIC_INPUT_ROUTING = "0" } },
    @{ name = "no_small_heads"; env = @{ DECADIC_ATTENTION_SCHEMA = "0"; DECADIC_METACOG_CALIBRATION = "0"; DECADIC_ACTION_VETO = "0"; DECADIC_INVERSE_MODEL = "0"; DECADIC_PERCEPT_REFINE = "0"; DECADIC_SYMBOLS = "0" } }
)

$Results = @()
foreach ($arm in $Arms) {
    Write-Host ("`n[bisect] arm: {0}" -f $arm.name) -ForegroundColor Cyan
    foreach ($k in $arm.env.Keys) { Set-Item -Path "env:$k" -Value $arm.env[$k] }
    & powershell -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\run_body_diag.ps1") -GraphBackend kuzu -Seconds $Seconds
    foreach ($k in $arm.env.Keys) { Remove-Item -Path "env:$k" -ErrorAction SilentlyContinue }
    $runDir = Get-ChildItem (Join-Path $Root "reports") -Directory -Filter "bodydiag_kuzu_*" |
        Sort-Object Name -Descending | Select-Object -First 1
    $line = Select-String -Path (Join-Path $runDir.FullName "summary.txt") -Pattern "host CPU|cycle rate" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty Line
    $Results += [pscustomobject]@{ arm = $arm.name; summary = ($line -join " | ") }
    Start-Sleep -Seconds 10  # let the port/processes settle between arms
}

Write-Host "`n=== CPU bisect verdicts ===" -ForegroundColor Cyan
$out = @("CPU bisect ($Seconds s/arm; compare mean CPU vs the all_on arm; pre-WS baseline 30-35%):")
foreach ($r in $Results) { $out += ("  {0,-15} {1}" -f $r.arm, $r.summary) }
$out += "Reading: the arm whose CPU drops most below all_on names the dominant cost."
$out | ForEach-Object { Write-Host $_ }
$out | Set-Content (Join-Path $Root "reports\cpu_bisect_$(Get-Date -Format yyyyMMdd_HHmmss).txt") -Encoding ascii
