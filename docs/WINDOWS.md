# Windows local run (pull & go)

## Prerequisites
1. [Python 3.11+](https://www.python.org/downloads/) — tick **Add python.exe to PATH**
2. [Node.js LTS](https://nodejs.org/) (includes npm)
3. Git for Windows

## First pull
```powershell
git clone <repo-url> 0to100
cd 0to100
git pull
.\scripts\setup_windows.ps1
notepad .env   # fill Kite + Telegram keys from .env.example
```

If PowerShell says scripts are disabled:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Daily start / stop
```powershell
# ~3GB RAM (old PC) — trading only, no report API:
.\scripts\run_quantterm_lean.ps1

# Older PC with a bit more RAM — full stack, lighter CPU:
.\scripts\run_quantterm_low_power.ps1

# Standard:
.\scripts\run_quantterm.ps1

# Complete (+ Research Data / PDF on :8766):
.\scripts\run_quantterm_complete.ps1

# stop
.\scripts\stop_quantterm.ps1
```

| Launcher | Report API :8766 | Market scan / autopilot | Best for |
|----------|------------------|-------------------------|----------|
| `run_quantterm_lean.ps1` | No (saves RAM) | Yes | **~3GB Windows** |
| `run_quantterm_low_power.ps1` | Yes | Yes | 4–8GB older PC |
| `run_quantterm_complete.ps1` | Yes | Yes | Normal daily use |

Open **http://127.0.0.1:5173**

## What works on Windows
- Cross-platform stack engine: `scripts/quantterm_stack.py`
- Autonomy / market-ops / idle-watcher file locks via `utils/process_lock.py` (`msvcrt`)
- Idle detection via Win32 `GetLastInputInfo`
- Same React terminal + FastAPI ports as Mac/Linux (`5173` / `8765` / `8766`)

## Honesty
Paper-first. LIVE orders stay locked unless your safety gates are satisfied.
Missing data stays missing — Windows run does not invent quotes or signals.
