@echo off
setlocal
cd /d "%~dp0\.."
set QT_LOW_POWER=1
if "%QT_DISABLE_IDLE_BACKTEST%"=="" set QT_DISABLE_IDLE_BACKTEST=1
if "%QT_DISABLE_US_BOOTSTRAP%"=="" set QT_DISABLE_US_BOOTSTRAP=1
set QT_DISABLE_AUTO_MARKET_SCAN=
set QT_DISABLE_AUTO_LONG_TERM=
if exist "venv\Scripts\python.exe" (
  venv\Scripts\python.exe scripts\quantterm_stack.py run --complete --low-power %*
) else (
  python scripts\quantterm_stack.py run --complete --low-power %*
)
