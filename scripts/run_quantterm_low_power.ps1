# Low-power Windows start — same complete stack as Mac; CPU eco via QT_LOW_POWER.
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_windows_common.ps1"
$env:QT_LOW_POWER = "1"
$env:QT_DISABLE_IDLE_BACKTEST = if ($env:QT_DISABLE_IDLE_BACKTEST) { $env:QT_DISABLE_IDLE_BACKTEST } else { "1" }
$env:QT_DISABLE_US_BOOTSTRAP = if ($env:QT_DISABLE_US_BOOTSTRAP) { $env:QT_DISABLE_US_BOOTSTRAP } else { "1" }
Remove-Item Env:QT_DISABLE_AUTO_MARKET_SCAN -ErrorAction SilentlyContinue
Remove-Item Env:QT_DISABLE_AUTO_LONG_TERM -ErrorAction SilentlyContinue
Invoke-QuantTermStack -StackArgs @("run", "--complete", "--low-power")
