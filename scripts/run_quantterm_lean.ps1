# Lean Windows start for ~3GB RAM PCs.
# Keeps: autonomy, terminal API :8765, Vite :5173, market-scan bootstrap, autopilot feed.
# Skips: research-report API :8766 (Research Data / PDF) to save RAM.
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_windows_common.ps1"
$env:QT_LOW_POWER = "1"
$env:QT_LEAN = "1"
$env:QT_DISABLE_IDLE_BACKTEST = if ($env:QT_DISABLE_IDLE_BACKTEST) { $env:QT_DISABLE_IDLE_BACKTEST } else { "1" }
$env:QT_DISABLE_US_BOOTSTRAP = if ($env:QT_DISABLE_US_BOOTSTRAP) { $env:QT_DISABLE_US_BOOTSTRAP } else { "1" }
Remove-Item Env:QT_DISABLE_AUTO_MARKET_SCAN -ErrorAction SilentlyContinue
Remove-Item Env:QT_DISABLE_AUTO_LONG_TERM -ErrorAction SilentlyContinue
Write-Host "[LEAN] Trading stack only — no report API :8766 (saves RAM on 3GB PCs)."
Write-Host "[LEAN] Market scan + autopilot still auto-start. Open http://127.0.0.1:5173"
Invoke-QuantTermStack -StackArgs @("run", "--lean")
