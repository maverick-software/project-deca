# WS3-P1 startle test: gate-enabled server, event-injecting client, eval
# collection, then check_gate_probe.py verdict. Self-terminating (~6 min).
# Usage: .\scripts\run_gate_probe.ps1
#        .\scripts\run_gate_probe.ps1 -Threshold 0.5 -Weights "0.5,0.2,0.2,0.1"
param(
    # Operating point tuned offline 2026-07-04 (tune_gate.py on
    # gateprobe_20260704_091426): measured first-exposure spikes are ~0.35,
    # so the old 0.55 threshold was unreachable; 0.30 with default weights
    # answers 2/2 bursts at settled rate 0.040 (target 0.05).
    [double]$Threshold = 0.30,
    [string]$Weights = "0.35,0.3,0.25,0.1",
    # Redesign 2026-07-04: two UNIQUE-target novel events (each must spike on
    # first exposure) + one revisit of the first target (must NOT spike --
    # the agent has been there; habituation-across-events is the assertion).
    [string]$Events = "collision:700,novel:1400,novel:2100,revisit:2800",
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $Root "reports\gateprobe_$Stamp"
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$BaseUrl = "http://127.0.0.1:$Port"

# Server env (WS1/WS2 standard) + the gate.
$env:DECADIC_SELF_HOST = "127.0.0.1"
$env:DECADIC_SELF_PORT = "$Port"
$env:DECADIC_NEURAL_PRESET = "full"
$env:DECADIC_PLASTICITY_ENABLED = "1"
$env:DECADIC_SPARSE_ENABLED = "1"
$env:DECADIC_GROWTH_ENABLED = "1"
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
$env:DECADIC_GATE_ENABLED = "1"
$env:DECADIC_GATE_THRESHOLD = "$Threshold"
if ($Weights -ne "") { $env:DECADIC_GATE_WEIGHTS = $Weights }

$Server = $null
$Client = $null
try {
    Write-Host "starting gate-enabled server (threshold=$Threshold)..." -ForegroundColor Cyan
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

    $aid = (Invoke-RestMethod -Method Post -Uri "$BaseUrl/agent").agent_id
    Write-Host "agent: $aid  events: $Events"
    $Client = Start-Process -FilePath $Py `
        -ArgumentList "scripts\synthetic_ws_client.py", "--port", "$Port", "--agent-id", "$aid", `
            "--steps", "2000000", "--rate", "0.1", "--log-every", "1000", "--events", "$Events" `
        -WorkingDirectory $Root -PassThru -NoNewWindow `
        -RedirectStandardOutput "$RunDir\client.out.log" -RedirectStandardError "$RunDir\client.err.log"

    & $Py "scripts\run_training_eval.py" --scenario gate_probe --agent-id $aid `
        --timeout 900 --base-url $BaseUrl --out-dir $RunDir *>&1 |
        Tee-Object -FilePath "$RunDir\eval.log"

    $Samples = Get-ChildItem "$RunDir\training_eval_gate_probe_*.jsonl" |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if ($Samples) {
        # Exact expected burst count = the novel: entries in the spec.
        # revisit: entries must NOT add bursts (habituation assertion).
        $ExpectNovel = @($Events -split "," | Where-Object { $_.Trim() -like "novel:*" }).Count
        Write-Host "`n=== gate probe verdict ===" -ForegroundColor Cyan
        & $Py "scripts\check_gate_probe.py" $Samples.FullName --expect-novel $ExpectNovel *>&1 |
            Tee-Object -FilePath "$RunDir\probe_verdict.log"
    } else {
        Write-Host "no samples collected - see $RunDir\eval.log" -ForegroundColor Red
    }
}
finally {
    foreach ($p in @($Client, $Server)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
}
Write-Host "artifacts in $RunDir" -ForegroundColor Green
