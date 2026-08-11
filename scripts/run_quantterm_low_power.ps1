# Low-power Windows start — QT_LOW_POWER=1, skips idle backtest watcher.
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_windows_common.ps1"
Invoke-QuantTermStack -StackArgs @("run", "--low-power")
