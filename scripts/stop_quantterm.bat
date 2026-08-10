@echo off
setlocal
cd /d "%~dp0\.."
if exist "venv\Scripts\python.exe" (
  venv\Scripts\python.exe scripts\quantterm_stack.py stop %*
) else (
  python scripts\quantterm_stack.py stop %*
)
