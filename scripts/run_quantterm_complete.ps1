# Complete stack on Windows — report API :8766 + terminal stack.
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_windows_common.ps1"
Invoke-QuantTermStack -StackArgs @("run", "--complete")
