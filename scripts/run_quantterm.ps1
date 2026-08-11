# Start QuantTerm terminal stack on Windows (API :8765 + Vite :5173).
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_windows_common.ps1"
Invoke-QuantTermStack -StackArgs @("run")
