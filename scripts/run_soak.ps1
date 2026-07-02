# WS2 soak wrapper - visible console, self-terminating.
# Usage: .\scripts\run_soak.ps1 -Hours 1        (shakedown)
#        .\scripts\run_soak.ps1 -Hours 12       (overnight soak)
#        .\scripts\run_soak.ps1 -Hours 1 -ConsolidationOff
param(
    [double]$Hours = 1.0,
    [switch]$ConsolidationOff,
    [string]$StallPolicy = "abort"
)
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$extra = @()
if ($ConsolidationOff) { $extra += "--consolidation-off" }
& $Py "scripts\soak_run.py" --hours $Hours --stall-policy $StallPolicy @extra
exit $LASTEXITCODE
