@echo off
REM Double-click / cmd.exe entrypoint for QuantTerm on Windows.
setlocal
cd /d "%~dp0\.."
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found on PATH. Install Python 3.11+ and re-open the terminal.
  exit /b 1
)
if not exist "venv\Scripts\python.exe" (
  echo Creating venv + installing deps ^(first run^)…
  python scripts\quantterm_stack.py setup
  if errorlevel 1 exit /b 1
)
venv\Scripts\python.exe scripts\quantterm_stack.py run %*
