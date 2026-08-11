# Stop QuantTerm listeners on :8765 / :8766 / :5173 (Windows).
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_windows_common.ps1"
Invoke-QuantTermStack -StackArgs @("stop")
