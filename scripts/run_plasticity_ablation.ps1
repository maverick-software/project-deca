param(
    [int[]]$RunIds = @(0, 1, 2, 3, 4),
    [int]$TargetCycles = 35,
    [int]$TimeoutSeconds = 180,
    [string]$OutDir = "logs\plasticity_ablation"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ".").Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$out = Join-Path $root $OutDir
New-Item -ItemType Directory -Force -Path $out | Out-Null

function Stop-Ports {
    $listeners = netstat -ano | Select-String -Pattern 'LISTENING' | Select-String -Pattern ':8765|:5173'
    $pids = foreach ($line in $listeners) {
        $parts = ($line.ToString() -split '\s+') | Where-Object { $_ }
        if ($parts.Length -ge 5) { [int]$parts[-1] }
    }
    foreach ($pidToStop in ($pids | Select-Object -Unique)) {
        Stop-Process -Id $pidToStop -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

function Wait-Server {
    param([int]$TimeoutSeconds)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-RestMethod "http://127.0.0.1:8765/agents" -TimeoutSec 2 | Out-Null
            return
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "Server did not become ready within $TimeoutSeconds seconds."
}

function Get-FreezeLine {
    param([string]$AgentId)
    $pattern = "plasticity_frozen agent_id=$AgentId"
    $line = Select-String -Path (Join-Path $root "logs\decadic_server.jsonl") -Pattern $pattern -ErrorAction SilentlyContinue | Select-Object -Last 1
    if ($null -eq $line) { return $null }
    return $line.Line
}

function Parse-FreezeCycle {
    param([string]$Line)
    if ([string]::IsNullOrWhiteSpace($Line)) { return $null }
    if ($Line -match 'cycle=(\d+)') { return [int]$Matches[1] }
    return $null
}

$configs = @(
    @{ Id = 0; Name = "baseline"; Env = @{} },
    @{ Id = 1; Name = "self_model_feedback=0"; Env = @{ DECADIC_SELF_MODEL_FEEDBACK = "0" } },
    @{ Id = 2; Name = "predictive_affect=0"; Env = @{ DECADIC_PREDICTIVE_AFFECT = "0" } },
    @{ Id = 3; Name = "represented_self=0"; Env = @{ DECADIC_REPRESENTED_SELF = "0" } },
    @{ Id = 4; Name = "memory_efficient_training=0"; Env = @{ DECADIC_MEMORY_EFFICIENT_TRAINING = "0" } },
    @{ Id = 5; Name = "learning_rate=3e-5"; Env = @{ DECADIC_LEARNING_RATE = "3e-5" } },
    @{ Id = 6; Name = "plasticity_alpha=0"; Env = @{ DECADIC_PLASTICITY_ALPHA = "0" } },
    @{ Id = 7; Name = "plasticity_alpha=0.01"; Env = @{ DECADIC_PLASTICITY_ALPHA = "0.01" } },
    @{ Id = 8; Name = "plasticity_alpha=0.001"; Env = @{ DECADIC_PLASTICITY_ALPHA = "0.001" } }
)
$configs = @($configs | Where-Object { $RunIds -contains [int]$_.Id })

$results = @()

foreach ($cfg in $configs) {
    Write-Host "=== Run $($cfg.Id): $($cfg.Name) ==="
    Stop-Ports

    foreach ($name in @(
        "DECADIC_SELF_MODEL_FEEDBACK",
        "DECADIC_PREDICTIVE_AFFECT",
        "DECADIC_REPRESENTED_SELF",
        "DECADIC_MEMORY_EFFICIENT_TRAINING",
        "DECADIC_LEARNING_RATE",
        "DECADIC_PLASTICITY_ALPHA"
    )) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
    $env:DECADIC_CYCLE_PROFILE = "1"
    $env:DECADIC_REQUIRE_CUDA = "1"
    $env:DECADIC_ENCODER_MODE = "hf"
    $env:DECADIC_PORT = "8765"
    $env:DECADIC_SELF_PORT = "8765"
    foreach ($kv in $cfg.Env.GetEnumerator()) {
        Set-Item "Env:$($kv.Key)" $kv.Value
    }

    $stdout = Join-Path $out "run_$($cfg.Id)_server.out.log"
    $stderr = Join-Path $out "run_$($cfg.Id)_server.err.log"
    $proc = Start-Process -FilePath $python `
        -ArgumentList @("-m", "uvicorn", "decadic.api.app:app", "--host", "127.0.0.1", "--port", "8765") `
        -WorkingDirectory $root `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru `
        -WindowStyle Hidden

    $agentId = $null
    $samples = @()
    $finalMetrics = $null
    $freezeLine = $null
    try {
        Wait-Server -TimeoutSeconds $TimeoutSeconds
        $body = @{
            elements = @("house", "food", "water")
            vision = $true
            audio = $false
            braces = $false
            replace = $true
            preset = "full"
        } | ConvertTo-Json
        $envStatus = Invoke-RestMethod "http://127.0.0.1:8765/environment" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 30
        $agentId = [string]$envStatus.agent_id
        Write-Host "Agent: $agentId"

        $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
        $seenCycles = @{}
        while ((Get-Date) -lt $deadline) {
            try {
                $mResp = Invoke-RestMethod "http://127.0.0.1:8765/agent/$agentId/metrics" -TimeoutSec 5
                $m = $mResp.metrics
                $cyc = [int]($m.cycles_completed)
                if ($cyc -gt 0 -and -not $seenCycles.ContainsKey($cyc)) {
                    $seenCycles[$cyc] = $true
                    $samples += [pscustomobject]@{
                        cycle = $cyc
                        pc_loss = [double]($m.neural_pc_loss_last)
                        plasticity_frozen = [bool]($m.plasticity_frozen)
                        prediction_error_ema = [double]($m.prediction_error_ema)
                    }
                }
                $finalMetrics = $m
                $freezeLine = Get-FreezeLine -AgentId $agentId
                if ($cyc -ge $TargetCycles -or $null -ne $freezeLine) {
                    break
                }
            } catch {
                Start-Sleep -Milliseconds 250
            }
            Start-Sleep -Milliseconds 150
        }

        Start-Sleep -Seconds 1
        $freezeLine = Get-FreezeLine -AgentId $agentId
        if ($null -eq $finalMetrics) {
            $mResp = Invoke-RestMethod "http://127.0.0.1:8765/agent/$agentId/metrics" -TimeoutSec 5
            $finalMetrics = $mResp.metrics
        }
    } finally {
        try { Invoke-RestMethod "http://127.0.0.1:8765/environment/stop" -Method Post -TimeoutSec 5 | Out-Null } catch {}
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }

    $freezeCycle = Parse-FreezeCycle -Line $freezeLine
    $froze = $null -ne $freezeCycle
    $first15 = @($samples | Sort-Object cycle | Select-Object -First 15)
    $finalBeforeFreeze = $null
    if ($froze) {
        $prior = @($samples | Where-Object { $_.cycle -le $freezeCycle } | Sort-Object cycle | Select-Object -Last 1)
        if ($prior.Count -gt 0) { $finalBeforeFreeze = $prior[0].pc_loss }
    } elseif ($null -ne $finalMetrics) {
        $finalBeforeFreeze = [double]($finalMetrics.neural_pc_loss_last)
    }
    $behavior = "unknown"
    if ($first15.Count -ge 2) {
        $vals = @($first15 | ForEach-Object { [double]$_.pc_loss })
        if ($vals | Where-Object { [double]::IsNaN($_) -or [double]::IsInfinity($_) }) {
            $behavior = "sudden-NaN"
        } elseif (($vals[-1] -gt ($vals[0] * 2.0)) -and ($vals[-1] -gt 1.0)) {
            $behavior = "climbing"
        } else {
            $behavior = "flat/stable"
        }
    }

    $result = [pscustomobject]@{
        id = $cfg.Id
        config = $cfg.Name
        agent_id = $agentId
        froze = $froze
        freeze_cycle = $freezeCycle
        pc_loss_behavior = $behavior
        final_pc_loss_before_freeze = $finalBeforeFreeze
        final_cycles = if ($null -ne $finalMetrics) { [int]$finalMetrics.cycles_completed } else { $null }
        final_plasticity_frozen = if ($null -ne $finalMetrics) { [bool]$finalMetrics.plasticity_frozen } else { $null }
        final_neural_pc_loss = if ($null -ne $finalMetrics) { [double]$finalMetrics.neural_pc_loss_last } else { $null }
        freeze_log = $freezeLine
        samples_first15 = $first15
    }
    $results += $result
    $result | ConvertTo-Json -Depth 8 | Set-Content -Path (Join-Path $out "run_$($cfg.Id)_result.json") -Encoding UTF8
}

$jsonPath = Join-Path $out "summary.json"
$results | ConvertTo-Json -Depth 8 | Set-Content -Path $jsonPath -Encoding UTF8

$md = @()
$md += "| # | Config | Froze? | Freeze cycle | pc-loss behavior | Final pc-loss before freeze | Notes |"
$md += "|---|--------|--------|--------------|------------------|-----------------------------|-------|"
foreach ($r in $results) {
    $frozeText = if ($r.froze) { "Y" } else { "N" }
    $cycleText = if ($null -ne $r.freeze_cycle) { "$($r.freeze_cycle)" } else { "" }
    $pcText = if ($null -ne $r.final_pc_loss_before_freeze) { "{0:N6}" -f $r.final_pc_loss_before_freeze } else { "" }
    $notes = "agent=$($r.agent_id); final_cycles=$($r.final_cycles); final_pc={0:N6}" -f $r.final_neural_pc_loss
    $md += "| $($r.id) | $($r.config) | $frozeText | $cycleText | $($r.pc_loss_behavior) | $pcText | $notes |"
}
$mdPath = Join-Path $out "summary.md"
$md | Set-Content -Path $mdPath -Encoding UTF8

Write-Host "Wrote $jsonPath"
Write-Host "Wrote $mdPath"
Get-Content $mdPath
