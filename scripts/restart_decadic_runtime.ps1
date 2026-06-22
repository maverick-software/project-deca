$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$DashboardDir = Join-Path $RepoRoot "dashboard"

$ports = @(8765, 5173)
$listeners = netstat -ano |
  Select-String -Pattern "LISTENING" |
  Select-String -Pattern ":8765|:5173"

$pids = foreach ($line in $listeners) {
  $parts = ($line.ToString() -split "\s+") | Where-Object { $_ }
  if ($parts.Length -ge 5) { [int]$parts[-1] }
}

$pids | Select-Object -Unique | ForEach-Object {
  Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 1

$serverScript = Join-Path $RepoRoot "scripts\run_decadic_server.ps1"
$uiScript = Join-Path $RepoRoot "scripts\run_decadic_ui.ps1"

@"
Set-Location "$RepoRoot"
`$env:DECADIC_SELF_HOST = "127.0.0.1"
`$env:DECADIC_SELF_PORT = "8765"
`$env:DECADIC_NEURAL_PRESET = "full"
`$env:DECADIC_PLASTICITY_ENABLED = "1"
`$env:DECADIC_SPARSE_ENABLED = "1"
`$env:DECADIC_GROWTH_ENABLED = "1"
`$env:DECADIC_ENCODER_MODE = "hf"
`$env:DECADIC_DEVICE = "cuda"
`$env:DECADIC_EPISODIC_ASYNC = "1"
`$env:DECADIC_N_ACTUATORS = "21"
`$env:DECADIC_CURRICULUM_MODE = "legacy"
`$env:DECADIC_SELF_MODEL_FEEDBACK = "1"
`$env:DECADIC_GWT_ENABLED = "1"
`$env:DECADIC_INTEGRATION_WINDOW_MS = "200"
`$env:DECADIC_PREDICTIVE_AFFECT = "1"
`$env:DECADIC_REPRESENTED_SELF = "1"
`$env:DECADIC_MEMORY_EFFICIENT_TRAINING = "1"
`$env:UV_CACHE_DIR = Join-Path "$RepoRoot" ".uv-cache"
& ".\.venv\Scripts\python.exe" -m uvicorn decadic.api.app:app --host 127.0.0.1 --port 8765
"@ | Set-Content -Path $serverScript -Encoding UTF8

@"
Set-Location "$DashboardDir"
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
"@ | Set-Content -Path $uiScript -Encoding UTF8

$server = Start-Process -FilePath "powershell.exe" `
  -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-NoProfile", "-File", $serverScript) `
  -WindowStyle Minimized `
  -PassThru

$ui = Start-Process -FilePath "powershell.exe" `
  -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-NoProfile", "-File", $uiScript) `
  -WindowStyle Minimized `
  -PassThru

[pscustomobject]@{
  ServerWindowPid = $server.Id
  UiWindowPid = $ui.Id
}
