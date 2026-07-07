# WS6 audio-cognition probe: does live microphone sound reach and move cognition?
# Full preset + hf encoder (real Whisper) + validated gate config + mic intake.
# The synthetic client sends proprio-only observations (NO audio) -- everything
# the agent hears comes from YOUR room via the intake. Interactive (~3 min):
# follow the SPEAK/SILENT prompts. Usage: .\scripts\run_audio_probe.ps1
param(
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $Root "reports\audioprobe_$Stamp"
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$BaseUrl = "http://127.0.0.1:$Port"

# Save/restore every env var we touch (the stale-env lesson, twice paid).
$TouchedEnv = @(
    "DECADIC_SELF_HOST", "DECADIC_SELF_PORT", "DECADIC_NEURAL_PRESET",
    "DECADIC_PLASTICITY_ENABLED", "DECADIC_SPARSE_ENABLED", "DECADIC_GROWTH_ENABLED",
    "DECADIC_ENCODER_MODE", "DECADIC_DEVICE", "DECADIC_EPISODIC_ASYNC",
    "DECADIC_N_ACTUATORS", "DECADIC_PREFETCH_OVERLOAD_POLICY", "DECADIC_LOG_DIR",
    "DECADIC_GATE_ENABLED", "DECADIC_GATE_THRESHOLD", "DECADIC_GATE_WEIGHTS",
    "DECADIC_GATE_NOVELTY_SOURCE", "DECADIC_GATE_LOG",
    "DECADIC_AUDIO_INTAKE", "DECADIC_VOICE"
)
$SavedEnv = @{}
foreach ($n in $TouchedEnv) { $SavedEnv[$n] = [Environment]::GetEnvironmentVariable($n) }

# WS1/WS2 standard server env + validated gate config + the WS6 organs.
$env:DECADIC_SELF_HOST = "127.0.0.1"
$env:DECADIC_SELF_PORT = "$Port"
$env:DECADIC_NEURAL_PRESET = "full"
$env:DECADIC_PLASTICITY_ENABLED = "1"
$env:DECADIC_SPARSE_ENABLED = "1"
$env:DECADIC_GROWTH_ENABLED = "1"
$env:DECADIC_ENCODER_MODE = "hf"          # real Whisper: the ear must parse
$env:DECADIC_DEVICE = "cuda"
$env:DECADIC_EPISODIC_ASYNC = "1"
$env:DECADIC_N_ACTUATORS = "21"
$env:DECADIC_PREFETCH_OVERLOAD_POLICY = "drop_oldest"
$env:DECADIC_LOG_DIR = $RunDir
$env:DECADIC_GATE_ENABLED = "1"           # gate telemetry = the cognition dial
$env:DECADIC_GATE_THRESHOLD = "0.30"
$env:DECADIC_GATE_WEIGHTS = "0.35,0.3,0.25,0.1"
$env:DECADIC_GATE_NOVELTY_SOURCE = "percept"
$env:DECADIC_GATE_LOG = "1"               # per-cycle decision rows for later analysis
$env:DECADIC_AUDIO_INTAKE = "mic"         # THE organ under test
$env:DECADIC_VOICE = "1"                  # mouth on (it may hum; that is fine)

$Server = $null
$Client = $null
try {
    Write-Host "starting full-preset server (mic intake ON)..." -ForegroundColor Cyan
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

    $aid = (Invoke-RestMethod -Method Post -Uri "$BaseUrl/agent" -TimeoutSec 240).agent_id
    Write-Host "agent: $aid"
    # Proprio-only patrol: the client supplies NO audio, so obs.audio can only
    # come from the intake (client-audio-wins precedence never triggers).
    $Client = Start-Process -FilePath $Py `
        -ArgumentList "scripts\synthetic_ws_client.py", "--port", "$Port", "--agent-id", "$aid", `
            "--steps", "2000000", "--rate", "0.1", "--log-every", "0" `
        -WorkingDirectory $Root -PassThru -NoNewWindow `
        -RedirectStandardOutput "$RunDir\client.out.log" -RedirectStandardError "$RunDir\client.err.log"
    Start-Sleep -Seconds 20   # settle past early warmup before baselining

    & $Py "scripts\audio_probe_driver.py" --base $BaseUrl --agent $aid --out $RunDir
}
finally {
    foreach ($p in @($Client, $Server)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
    foreach ($n in $TouchedEnv) {
        if ($null -eq $SavedEnv[$n]) { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
        else { Set-Item "Env:$n" $SavedEnv[$n] }
    }
}
Write-Host "artifacts in $RunDir" -ForegroundColor Green
