# WS1 - Neural-path verification post-bf16-fix (one-shot runner).
# Runs: full pytest suite -> server -> health_smoke eval -> 3000-cycle learning
# eval -> PC-loss trend check. All artifacts land in reports\ws1_<timestamp>\.
#
# Usage:  .\scripts\run_ws1.ps1              # everything
#         .\scripts\run_ws1.ps1 -SkipTests   # skip pytest, just the live runs
#         .\scripts\run_ws1.ps1 -Cycles 1500 # shorter learning run

param(
    [switch]$SkipTests,
    [switch]$SkipSmoke,
    [int]$Cycles = 3000,
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $Root "reports\ws1_$Stamp"
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$BaseUrl = "http://127.0.0.1:$Port"
$Summary = [ordered]@{ started = (Get-Date -Format o); git_sha = (git rev-parse --short HEAD) }

# --- Environment (mirrors scripts\run_decadic_server.ps1) ---
$env:DECADIC_SELF_HOST = "127.0.0.1"
$env:DECADIC_SELF_PORT = "$Port"
$env:DECADIC_NEURAL_PRESET = "full"
$env:DECADIC_PLASTICITY_ENABLED = "1"
$env:DECADIC_SPARSE_ENABLED = "1"
# WS1 finding: growth evaluation (every 500 cycles) hangs the cycle loop on the
# 10GB 3080 with the full preset - suspected CUDA alloc stall in grow_step /
# reset_optimizer_state. Disabled for verification; growth needs its own fix.
$env:DECADIC_GROWTH_ENABLED = "0"
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
$env:DECADIC_LOG_DIR = $RunDir
# WS1 finding: default overload policy "block" deadlocks intake when observations
# outpace cycles; drop_oldest is the existing non-blocking path with accounting.
$env:DECADIC_PREFETCH_OVERLOAD_POLICY = "drop_oldest"

# --- Step 1: test suite ---
if (-not $SkipTests) {
    Write-Host "=== [1/4] pytest suite (log: $RunDir\pytest.log) ===" -ForegroundColor Cyan
    & $Py -m pytest -q --durations=25 --junitxml "$RunDir\pytest_junit.xml" *>&1 |
        Tee-Object -FilePath "$RunDir\pytest.log"
    $Summary.pytest_exit = $LASTEXITCODE
    Write-Host "pytest exit code: $LASTEXITCODE"
} else {
    Write-Host "=== [1/4] pytest SKIPPED ===" -ForegroundColor Yellow
    $Summary.pytest_exit = "skipped"
}

$Server = $null
$Client = $null
try {
    # --- Step 2: server ---
    Write-Host "=== [2/4] starting server on port $Port ===" -ForegroundColor Cyan
    $Server = Start-Process -FilePath $Py `
        -ArgumentList "-m", "uvicorn", "decadic.api.app:app", "--host", "127.0.0.1", "--port", "$Port" `
        -WorkingDirectory $Root -PassThru -NoNewWindow `
        -RedirectStandardOutput "$RunDir\server.out.log" -RedirectStandardError "$RunDir\server.err.log"

    $Ready = $false
    foreach ($i in 1..120) {
        Start-Sleep -Seconds 2
        try {
            Invoke-RestMethod -Uri "$BaseUrl/agents" -TimeoutSec 3 | Out-Null
            $Ready = $true; break
        } catch { if ($Server.HasExited) { throw "Server exited early - see $RunDir\server.err.log" } }
    }
    if (-not $Ready) { throw "Server not ready after 240s" }
    Write-Host "server ready."

    function Start-StreamedAgent([string]$Label, [int]$Steps) {
        $resp = Invoke-RestMethod -Method Post -Uri "$BaseUrl/agent"
        $aid = $resp.agent_id
        Write-Host "agent ($Label): $aid"
        $c = Start-Process -FilePath $Py `
            -ArgumentList "scripts\synthetic_ws_client.py", "--port", "$Port", "--agent-id", "$aid", `
                "--steps", "$Steps", "--rate", "0.1", "--log-every", "500" `
            -WorkingDirectory $Root -PassThru -NoNewWindow `
            -RedirectStandardOutput "$RunDir\client_$Label.out.log" -RedirectStandardError "$RunDir\client_$Label.err.log"
        return @{ agent = $aid; proc = $c }
    }

    # --- Step 3: smoke eval (500 cycles) ---
    if (-not $SkipSmoke) {
        Write-Host "=== [3/4] health_smoke eval ===" -ForegroundColor Cyan
        $S = Start-StreamedAgent "smoke" 200000
        & $Py "scripts\run_training_eval.py" --scenario health_smoke --agent-id $S.agent `
            --base-url $BaseUrl --out-dir $RunDir *>&1 | Tee-Object -FilePath "$RunDir\eval_smoke.log"
        $Summary.smoke_exit = $LASTEXITCODE
        Stop-Process -Id $S.proc.Id -Force -ErrorAction SilentlyContinue
        # Remove the smoke agent entirely so it cannot compete with the learning run.
        try { Invoke-RestMethod -Method Delete -Uri "$BaseUrl/agent/$($S.agent)" | Out-Null } catch {}
    } else {
        Write-Host "=== [3/4] smoke SKIPPED ===" -ForegroundColor Yellow
        $Summary.smoke_exit = "skipped"
    }

    # --- Step 4: learning run + trend ---
    Write-Host "=== [4/4] ws1_learning_run eval ($Cycles cycles) ===" -ForegroundColor Cyan
    $L = Start-StreamedAgent "learning" 2000000
    & $Py "scripts\run_training_eval.py" --scenario ws1_learning_run --agent-id $L.agent `
        --cycles $Cycles --timeout 900 --base-url $BaseUrl --out-dir $RunDir *>&1 | Tee-Object -FilePath "$RunDir\eval_learning.log"
    $Summary.learning_exit = $LASTEXITCODE
    Stop-Process -Id $L.proc.Id -Force -ErrorAction SilentlyContinue
    try { Invoke-RestMethod -Method Delete -Uri "$BaseUrl/agent/$($L.agent)" | Out-Null } catch {}

    $SamplesFile = Get-ChildItem "$RunDir\training_eval_ws1_learning_run_*.jsonl" |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if ($SamplesFile) {
        & $Py "scripts\plot_pc_trend.py" $SamplesFile.FullName --out-png "$RunDir\pc_trend.png" *>&1 |
            Tee-Object -FilePath "$RunDir\pc_trend.log"
        $Summary.trend_exit = $LASTEXITCODE
    } else {
        $Summary.trend_exit = "no samples file"
    }
}
finally {
    foreach ($p in @($Client, $Server)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
}

$Summary.finished = (Get-Date -Format o)
$Summary | ConvertTo-Json | Set-Content "$RunDir\ws1_summary.json"
Write-Host "`n=== WS1 complete - artifacts in $RunDir ===" -ForegroundColor Green
Write-Host ($Summary | ConvertTo-Json)
