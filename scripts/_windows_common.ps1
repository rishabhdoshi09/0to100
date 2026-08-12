# QuantTerm Windows stack wrappers — call the cross-platform Python engine.
# Requires: Python 3.11+, Node.js/npm, Git.
# Usage (PowerShell, from repo root):
#   .\scripts\setup_windows.ps1
#   .\scripts\run_quantterm.ps1
#   .\scripts\run_quantterm_low_power.ps1
#   .\scripts\run_quantterm_lean.ps1
#   .\scripts\run_quantterm_complete.ps1
#   .\scripts\stop_quantterm.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Get-QuantTermPython {
    $venvPy = Join-Path $Root "venv\Scripts\python.exe"
    if (Test-Path $venvPy) { return $venvPy }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "Python not found. Install Python 3.11+ from https://www.python.org/downloads/ (check 'Add to PATH')."
}

function Invoke-QuantTermStack {
    param(
        [Parameter(Mandatory = $true)][string[]]$StackArgs
    )
    $py = Get-QuantTermPython
    $script = Join-Path $Root "scripts\quantterm_stack.py"
    & $py $script @StackArgs
    exit $LASTEXITCODE
}
