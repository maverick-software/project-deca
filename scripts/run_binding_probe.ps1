# WS5-M5.2: the binding ablation. Run TWICE: -Binding on (expect BINDING)
# and -Binding off (expect NOT-BINDING -- a pooled system structurally
# cannot represent which-entity-is-adjacent-to-which).
# Self-terminating (~18 min per leg). Gate disabled (no confound); oracle
# perception (WM-seam injection); full preset + learning ON (the relation
# must be LEARNED during the train phases, not scripted).
param(
    [ValidateSet("on", "off")] [string]$Binding = "on",
    [int]$Port = 8767,
    [string]$Scenario = "docs\binding_scenarios\binding_probe.json"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $Root "reports\bindprobe_${Binding}_$Stamp"
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$BaseUrl = "http://127.0.0.1:$Port"

if (-not (Test-Path $Scenario)) { & $Py "scripts\gen_binding_scenario.py" --out $Scenario }

$TouchedEnv = @(
    "DECADIC_SELF_HOST", "DECADIC_SELF_PORT", "DECADIC_NEURAL_PRESET",
    "DECADIC_PLASTICITY_ENABLED", "DECADIC_SPARSE_ENABLED", "DECADIC_GROWTH_ENABLED",
    "DECADIC_ENCODER_MODE", "DECADIC_DEVICE", "DECADIC_EPISODIC_ASYNC",
    "DECADIC_PERCEPTION_MODE", "DECADIC_GATE_ENABLED", "DECADIC_LOG_DIR",
    "DECADIC_PREFETCH_OVERLOAD_POLICY", "DECADIC_MEMORY_EFFICIENT_TRAINING",
    "DECADIC_WM_SLOT_TENSOR", "DECADIC_MEMORY_TOKENS", "DECADIC_RELATIONAL_CORE",
    "DECADIC_USE_NEURAL"
)
$SavedEnv = @{}
foreach ($n in $TouchedEnv) { $SavedEnv[$n] = [Environment]::GetEnvironmentVariable($n) }

$env:DECADIC_SELF_HOST = "127.0.0.1"
$env:DECADIC_SELF_PORT = "$Port"
$env:DECADIC_USE_NEURAL = "1"
$env:DECADIC_NEURAL_PRESET = "full"
$env:DECADIC_PLASTICITY_ENABLED = "1"
$env:DECADIC_SPARSE_ENABLED = "1"
$env:DECADIC_GROWTH_ENABLED = "1"
$env:DECADIC_ENCODER_MODE = "hf"
$env:DECADIC_DEVICE = "cuda"
$env:DECADIC_EPISODIC_ASYNC = "1"
$env:DECADIC_PERCEPTION_MODE = "oracle"
$env:DECADIC_GATE_ENABLED = "0"
$env:DECADIC_PREFETCH_OVERLOAD_POLICY = "drop_oldest"
$env:DECADIC_MEMORY_EFFICIENT_TRAINING = "1"
$env:DECADIC_LOG_DIR = $RunDir
$flagVal = if ($Binding -eq "on") { "1" } else { "0" }
$env:DECADIC_WM_SLOT_TENSOR = $flagVal
$env:DECADIC_MEMORY_TOKENS = $flagVal
$env:DECADIC_RELATIONAL_CORE = $flagVal

$Server = $null
try {
    Write-Host "starting server (binding=$Binding)..." -ForegroundColor Cyan
    $Server = Start-Process -FilePath $Py `
        -ArgumentList "-m", "uvicorn", "decadic.api.app:app", "--host", "127.0.0.1", "--port", "$Port" `
        -WorkingDirectory $Root -PassThru -NoNewWindow `
        -RedirectStandardOutput "$RunDir\server.out.log" -RedirectStandardError "$RunDir\server.err.log"
    $Ready = $false
    foreach ($i in 1..120) {
        Start-Sleep -Seconds 2
        try { Invoke-RestMethod "$BaseUrl/agents" -TimeoutSec 3 | Out-Null; $Ready = $true; break }
        catch { if ($Server.HasExited) { throw "server exited early - see $RunDir\server.err.log" } }
    }
    if (-not $Ready) { throw "server not ready" }

    $aid = (Invoke-RestMethod -Method Post -Uri "$BaseUrl/agent" -TimeoutSec 180).agent_id
    Write-Host "agent: $aid  binding=$Binding  scenario: $Scenario"
    & $Py "scripts\binding_probe_run.py" --port $Port --agent-id $aid `
        --scenario $Scenario --out "$RunDir\samples.jsonl" *>&1 |
        Tee-Object -FilePath "$RunDir\probe.log"

    $Expect = if ($Binding -eq "on") { "pass" } else { "fail" }
    Write-Host "`n=== binding probe verdict (expect $Expect leg) ===" -ForegroundColor Cyan
    & $Py "scripts\check_binding_probe.py" "$RunDir\samples.jsonl" $Scenario --expect $Expect *>&1 |
        Tee-Object -FilePath "$RunDir\verdict.log"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "ABLATION LEG ($Binding): OK" -ForegroundColor Green
    } else {
        Write-Host "ABLATION LEG ($Binding): UNEXPECTED RESULT" -ForegroundColor Red
    }
}
finally {
    if ($Server -and -not $Server.HasExited) {
        Stop-Process -Id $Server.Id -Force -ErrorAction SilentlyContinue
    }
    foreach ($n in $TouchedEnv) {
        if ($null -eq $SavedEnv[$n]) { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
        else { Set-Item "Env:$n" $SavedEnv[$n] }
    }
}
Write-Host "artifacts in $RunDir" -ForegroundColor Green
