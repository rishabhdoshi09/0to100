@echo off
REM Lean Windows start for ~3GB RAM — trading stack only, no report API :8766
setlocal
cd /d "%~dp0\.."
set QT_LOW_POWER=1
set QT_LEAN=1
if "%QT_DISABLE_IDLE_BACKTEST%"=="" set QT_DISABLE_IDLE_BACKTEST=1
if "%QT_DISABLE_US_BOOTSTRAP%"=="" set QT_DISABLE_US_BOOTSTRAP=1
set QT_DISABLE_AUTO_MARKET_SCAN=
set QT_DISABLE_AUTO_LONG_TERM=
echo [LEAN] Trading stack only — no report API :8766
if exist "venv\Scripts\python.exe" (
  venv\Scripts\python.exe scripts\quantterm_stack.py run --lean %*
) else (
  python scripts\quantterm_stack.py run --lean %*
)
