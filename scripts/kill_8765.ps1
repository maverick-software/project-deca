$out = "D:\Users\charl\software\Self-Determination Model\reports\kill_probe.txt"
try {
    $p = (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction Stop).OwningProcess | Select-Object -Unique
    if ($p) {
        foreach ($procId in $p) { Stop-Process -Id $procId -Force }
        Set-Content $out ("killed " + ($p -join ","))
    } else {
        Set-Content $out "no listener on 8765"
    }
} catch {
    Set-Content $out ("error: " + $_.Exception.Message)
}
