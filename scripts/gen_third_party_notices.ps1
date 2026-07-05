# Regenerate THIRD_PARTY_NOTICES.txt (the verbatim license-text appendix) for
# every package installed in the project venv. Run after any dependency change.
#
#   powershell -ExecutionPolicy Bypass -File scripts\gen_third_party_notices.ps1
#
# The curated summary (obligations, license families) lives in the committed
# THIRD_PARTY_NOTICES.md; this script produces the full machine-generated text
# that the summary points to. Unix equivalent is in the header of that file.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Out = Join-Path $Root "THIRD_PARTY_NOTICES.txt"

# Ensure the generator is available (idempotent).
& $Py -m pip install --quiet --disable-pip-version-check pip-licenses | Out-Null

# Verbatim license file text + copyright + URL for every installed distribution,
# written to the REPO ROOT deterministically (not the caller's cwd).
& $Py -m piplicenses `
    --with-license-file --no-license-path --with-urls --with-authors `
    --format=plain-vertical `
    --output-file $Out

$header = @"
THIRD-PARTY LICENSE NOTICES
Project Deca (Decadic Cycle Cognitive Architecture)
Generated $(Get-Date -Format "yyyy-MM-dd") by scripts/gen_third_party_notices.ps1

This file reproduces the copyright and license text of the third-party packages
installed in the project environment, as required by their (permissive) licenses.
See THIRD_PARTY_NOTICES.md for the curated summary and obligations.
================================================================================

"@
$body = Get-Content $Out -Raw
Set-Content -Path $Out -Value ($header + $body) -Encoding utf8

Write-Host "Wrote $Out" -ForegroundColor Green
