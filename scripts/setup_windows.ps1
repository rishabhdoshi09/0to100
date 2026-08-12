# First-time Windows bootstrap: venv + deps + .env seed.
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_windows_common.ps1"

Write-Host "[SETUP] Checking Python / Node…"
$py = Get-QuantTermPython
Write-Host "[SETUP] Python: $py"
$node = Get-Command node -ErrorAction SilentlyContinue
$npm = Get-Command npm -ErrorAction SilentlyContinue
if (-not $node -or -not $npm) {
    throw "Node.js/npm not found. Install LTS from https://nodejs.org/ then re-open PowerShell."
}
Write-Host "[SETUP] Node: $($node.Source)"

Invoke-QuantTermStack -StackArgs @("setup")
