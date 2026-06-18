<#
.SYNOPSIS
    Create or refresh the Decadic desktop shortcut (server + web UI).

.DESCRIPTION
    Points the shortcut at scripts\Launch Decadic.cmd, which runs
    launch_decadic.ps1: starts uvicorn on :8765, Vite on :5173, opens the browser.
    Re-running is safe (reuses processes already listening on those ports).
#>
[CmdletBinding()]
param(
    [string]$ShortcutName = "Decadic.lnk"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$CmdPath = Join-Path $RepoRoot "scripts\Launch Decadic.cmd"
if (-not (Test-Path $CmdPath)) {
    throw "Missing launcher: $CmdPath"
}

$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop $ShortcutName

$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($ShortcutPath)
$link.TargetPath = $CmdPath
$link.WorkingDirectory = $RepoRoot
$link.WindowStyle = 1
$link.Description = "Start the Decadic API server (port 8765) and web UI (port 5173), then open the dashboard."
$link.IconLocation = "$env:SystemRoot\System32\imageres.dll,76"
$link.Save()

# Retire the old dashboard-only name if we created Decadic.lnk.
$legacy = Join-Path $Desktop "Decadic Dashboard.lnk"
if ($ShortcutName -eq "Decadic.lnk" -and (Test-Path $legacy)) {
    Remove-Item $legacy -Force
    Write-Host "Removed legacy shortcut: Decadic Dashboard.lnk"
}

Write-Host "Desktop shortcut updated: $ShortcutPath"
Write-Host "  Target: $CmdPath"
