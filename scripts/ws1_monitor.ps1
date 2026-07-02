# Live WS1 progress monitor - safe to close anytime (read-only).
$Root = "D:\Users\charl\software\Self-Determination Model"
$Host.UI.RawUI.WindowTitle = "WS1 progress monitor"
Write-Host "WS1 progress monitor - polls the Decadic server every 3s. Ctrl+C or close to exit." -ForegroundColor Cyan
while ($true) {
    $ts = Get-Date -Format "HH:mm:ss"
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:8765/agents" -TimeoutSec 3
        $agents = $r.agents
        if (-not $agents) { $agents = $r }
        $parts = @()
        foreach ($a in $agents) {
            $aid = ($a.agent_id).Substring(0, 8)
            $parts += ("agent " + $aid + ": " + $a.cycles_completed + " cycles (" + $a.status + ")")
        }
        Write-Host ("[" + $ts + "] " + ($parts -join "  |  "))
    } catch {
        Write-Host ("[" + $ts + "] server not reachable: " + $_.Exception.Message) -ForegroundColor Yellow
    }
    $latest = Get-ChildItem "$Root\reports\ws1_2*" -Directory | Sort-Object Name | Select-Object -Last 1
    if ($latest -and (Test-Path "$($latest.FullName)\ws1_summary.json")) {
        Write-Host ("RUN COMPLETE - summary: " + $latest.FullName + "\ws1_summary.json") -ForegroundColor Green
    }
    Start-Sleep -Seconds 3
}
