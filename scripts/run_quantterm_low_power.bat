@echo off
setlocal
cd /d "%~dp0\.."
if exist "venv\Scripts\python.exe" (
  venv\Scripts\python.exe scripts\quantterm_stack.py run --low-power %*
) else (
  python scripts\quantterm_stack.py run --low-power %*
)
