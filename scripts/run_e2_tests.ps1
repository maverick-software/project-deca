# WS-EXPAND E2 test runner. ASCII only (no unicode glyphs).
# Uses the repo venv python (plain "python" resolves to a system install
# without kuzu/pyarrow -> 51 spurious failures, 2026-07-06).
$Root = "D:\Users\charl\software\Self-Determination Model"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
Set-Location $Root
New-Item -ItemType Directory -Force -Path reports | Out-Null

Write-Host "=== Phase 1: WS-EXPAND unit tests (E2 + E1) ==="
& $Py -m pytest tests/test_learning_control.py tests/test_cognitive_map.py tests/test_goal_conditioning.py tests/test_action_planner.py -q 2>&1 | Tee-Object -FilePath "reports\e2_tests_new.txt"

Write-Host ""
Write-Host "=== Phase 2: full suite (flag-off parity) ==="
& $Py -m pytest -q 2>&1 | Tee-Object -FilePath "reports\e2_tests_full.txt"

Write-Host ""
Write-Host "E2 TEST RUN COMPLETE - results in reports\e2_tests_new.txt and reports\e2_tests_full.txt"
