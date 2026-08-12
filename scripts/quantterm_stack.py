"""Cross-platform QuantTerm local stack (Windows / macOS / Linux).

Modes:
  python scripts/quantterm_stack.py run [--low-power] [--complete]
  python scripts/quantterm_stack.py stop
  python scripts/quantterm_stack.py setup

Prefer the thin wrappers:
  Windows:  .\\scripts\\run_quantterm.ps1   or  scripts\\run_quantterm.bat
  Unix:     bash scripts/run_quantterm.sh  (existing) or python scripts/quantterm_stack.py run
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_PORT = 8765
REPORT_PORT = 8766
VITE_PORT = 5173
API_HEALTH = f"http://127.0.0.1:{API_PORT}/api/health"
REPORT_HEALTH = f"http://127.0.0.1:{REPORT_PORT}/health"


def _is_windows() -> bool:
    return os.name == "nt"


def _venv_python() -> Path:
    if _is_windows():
        return ROOT / "venv" / "Scripts" / "python.exe"
    return ROOT / "venv" / "bin" / "python"


def _venv_ok() -> bool:
    return _venv_python().is_file()


def _log(msg: str) -> None:
    print(msg, flush=True)


def _health_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= int(getattr(resp, "status", 200)) < 300
    except Exception:
        return False


def _pids_on_port(port: int) -> list[int]:
    pids: list[int] = []
    if _is_windows():
        try:
            # Prefer PowerShell net object — works without admin.
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    f"(Get-NetTCPConnection -LocalPort {port} -State Listen "
                    f"-ErrorAction SilentlyContinue).OwningProcess"
                ),
            ]
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=15)
            for line in out.splitlines():
                line = line.strip()
                if line.isdigit():
                    pids.append(int(line))
        except Exception:
            try:
                out = subprocess.check_output(
                    ["netstat", "-ano"], text=True, stderr=subprocess.DEVNULL, timeout=15
                )
                needle = f":{port} "
                for line in out.splitlines():
                    if "LISTENING" not in line.upper() and "LISTEN" not in line.upper():
                        continue
                    if needle not in line and f":{port}\n" not in line + "\n":
                        # also match end-of-token :port
                        parts = line.split()
                        local = next((p for p in parts if p.endswith(f":{port}")), "")
                        if not local:
                            continue
                    parts = line.split()
                    if parts and parts[-1].isdigit():
                        pids.append(int(parts[-1]))
            except Exception:
                pass
    else:
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", f":{port}"], text=True, stderr=subprocess.DEVNULL, timeout=10
            )
            for line in out.splitlines():
                line = line.strip()
                if line.isdigit():
                    pids.append(int(line))
        except Exception:
            pass
    # unique preserve order
    seen: set[int] = set()
    out_pids: list[int] = []
    for pid in pids:
        if pid and pid not in seen:
            seen.add(pid)
            out_pids.append(pid)
    return out_pids


def _port_listening(port: int) -> bool:
    return bool(_pids_on_port(port))


def _kill_pid(pid: int, *, force: bool = False) -> None:
    if pid <= 0:
        return
    try:
        if _is_windows():
            args = ["taskkill", "/PID", str(pid)]
            if force:
                args.append("/F")
            subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.kill(pid, sig)
    except Exception:
        pass


def _free_port(port: int, label: str = "service") -> None:
    pids = _pids_on_port(port)
    if not pids:
        return
    _log(f"[STACK] Port {port} still in use by {label} (pid(s): {pids}) — stopping…")
    for pid in pids:
        _kill_pid(pid, force=False)
    time.sleep(1.0)
    pids = _pids_on_port(port)
    if pids:
        for pid in pids:
            _kill_pid(pid, force=True)
        time.sleep(0.5)


def _stop_ports(*ports: int) -> None:
    for port in ports:
        _free_port(port, f"port-{port}")


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        if _is_windows():
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _popen(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.Popen:
    creationflags = 0
    if _is_windows():
        # Detach from console Ctrl-C inheritance quirks; we still track the PID.
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(
        args,
        cwd=str(ROOT),
        env=env,
        stdout=None,
        stderr=None,
        creationflags=creationflags,
    )


def _python() -> str:
    return str(_venv_python())


def _ensure_api_deps(py: str) -> None:
    try:
        subprocess.check_call(
            [py, "-c", "import fastapi, uvicorn"],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        _log("[STACK] Installing local terminal API dependencies…")
        subprocess.check_call(
            [py, "-m", "pip", "install", "fastapi>=0.115.0", "uvicorn>=0.30.0"],
            cwd=str(ROOT),
        )


def _ensure_report_deps(py: str) -> None:
    try:
        subprocess.check_call(
            [py, "-c", "import reportlab, fastapi, uvicorn"],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        _log("[STACK] Installing professional report dependencies…")
        subprocess.check_call(
            [py, "-m", "pip", "install", "reportlab>=4.2.0", "fastapi>=0.115.0", "uvicorn>=0.30.0"],
            cwd=str(ROOT),
        )


def _ensure_frontend() -> None:
    nm = ROOT / "frontend" / "node_modules"
    if nm.is_dir():
        return
    _log("[STACK] Installing terminal frontend dependencies…")
    subprocess.check_call(["npm", "install"], cwd=str(ROOT / "frontend"), shell=_is_windows())


def _wait_health(health_url: str, pid: int | None, label: str, attempts: int = 240) -> bool:
    for _ in range(attempts):
        if pid and not _pid_alive(pid):
            _log(f"[STACK] {label} exited during startup. Review errors above.")
            return False
        if _health_ok(health_url):
            return True
        time.sleep(0.5)
    _log(f"[STACK] {label} did not become ready at {health_url} within {attempts / 2:.0f}s.")
    return False


def _autonomy_running(env: dict[str, str]) -> bool:
    try:
        code = subprocess.call(
            [
                _python(),
                "-c",
                "from product.autonomy_status import read_autonomy_status; "
                "raise SystemExit(0 if read_autonomy_status().get('running') else 1)",
            ],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return code == 0
    except Exception:
        return False


def cmd_setup() -> int:
    _log("[SETUP] QuantTerm Windows/cross-platform bootstrap")
    if not _venv_ok():
        _log("[SETUP] Creating venv…")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"], cwd=str(ROOT))
    py = _python()
    req = ROOT / "requirements.txt"
    if req.exists():
        _log("[SETUP] Installing requirements.txt (this can take a few minutes)…")
        subprocess.check_call([py, "-m", "pip", "install", "-U", "pip"], cwd=str(ROOT))
        subprocess.check_call([py, "-m", "pip", "install", "-r", str(req)], cwd=str(ROOT))
    _ensure_api_deps(py)
    _ensure_frontend()
    env_example = ROOT / ".env.example"
    env_path = ROOT / ".env"
    if env_example.exists() and not env_path.exists():
        env_path.write_text(env_example.read_text(encoding="utf-8"), encoding="utf-8")
        _log("[SETUP] Created .env from .env.example — fill Kite / Telegram keys.")
    _log("[SETUP] Done. Start with: python scripts/quantterm_stack.py run")
    _log("[SETUP] Or Windows: .\\scripts\\run_quantterm.ps1")
    return 0


def cmd_stop() -> int:
    _log(f"[STOP] Stopping QuantTerm local services on ports {API_PORT}, {REPORT_PORT}, {VITE_PORT}…")
    # Best-effort: idle research watcher
    if _is_windows():
        try:
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process | "
                    "Where-Object { $_.CommandLine -match 'idle_full_universe_backtest' } | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    else:
        subprocess.run(
            ["pkill", "-f", "idle_full_universe_backtest.py"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    _stop_ports(API_PORT, REPORT_PORT, VITE_PORT)
    _log("[STOP] Done. Restart with: python scripts/quantterm_stack.py run --complete")
    return 0


def cmd_run(*, complete: bool = False, low_power: bool = False, lean: bool = False) -> int:
    if not _venv_ok():
        _log("Missing venv. Run: python scripts/quantterm_stack.py setup")
        return 1

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if lean:
        # 3GB / very old PCs: trading stack only — no research-report API (:8766).
        # Autopilot + market-scan bootstrap stay on.
        low_power = True
        complete = False
        env["QT_LOW_POWER"] = "1"
        env["QT_LEAN"] = "1"
        env.setdefault("QT_DISABLE_IDLE_BACKTEST", "1")
        env.setdefault("QT_DISABLE_US_BOOTSTRAP", "1")
        env.pop("QT_DISABLE_AUTO_MARKET_SCAN", None)
        env.pop("QT_DISABLE_AUTO_LONG_TERM", None)
        _log("[STACK] Lean mode: autonomy + terminal API + Vite (no report API :8766).")
        _log("[STACK] Market scan + autopilot feed still auto-start. Best for ~3GB RAM.")
    elif low_power:
        # Same service topology as complete (report API + autonomy + scans).
        # Low-power only throttles CPU / skips idle backtest + US bootstrap.
        complete = True
        env["QT_LOW_POWER"] = "1"
        env.setdefault("QT_DISABLE_IDLE_BACKTEST", "1")
        env.setdefault("QT_DISABLE_US_BOOTSTRAP", "1")
        env.pop("QT_DISABLE_AUTO_MARKET_SCAN", None)
        env.pop("QT_DISABLE_AUTO_LONG_TERM", None)
        _log("[STACK] Low-power mode: same stack as complete; lighter CPU / no idle backtest.")
        _log("[STACK] Market scan + autopilot feed still auto-start (not disabled).")

    py = _python()
    _ensure_api_deps(py)
    if complete:
        _ensure_report_deps(py)
    _ensure_frontend()

    children: list[subprocess.Popen] = []
    api_external = False
    report_external = False
    api_pid: int | None = None
    report_pid: int | None = None
    shutting_down = False

    def cleanup(*_args: object) -> None:
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        _log("\n[STACK] Stopping QuantTerm child services…")
        for proc in reversed(children):
            try:
                if proc.poll() is None:
                    _kill_pid(proc.pid, force=_is_windows())
                    if not _is_windows():
                        try:
                            proc.send_signal(signal.SIGTERM)
                        except Exception:
                            pass
            except Exception:
                pass
        time.sleep(0.8)
        for proc in children:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        _free_port(VITE_PORT, "Vite dev server")
        if not api_external:
            _free_port(API_PORT, "Terminal API")
        if complete and not report_external:
            _free_port(REPORT_PORT, "Research-report API")
        _log("[STACK] Child services stopped.")

    try:
        signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    except Exception:
        pass

    try:
        # Research-report API (optional)
        if complete:
            if _health_ok(REPORT_HEALTH):
                report_external = True
                _log(f"[STACK] Research-report API already healthy at {REPORT_HEALTH} — reusing.")
            else:
                _free_port(REPORT_PORT, "Research-report API")
                if _health_ok(REPORT_HEALTH):
                    report_external = True
                    _log("[STACK] Research-report API became healthy after freeing port — reusing.")
                else:
                    _log(f"[STACK] Starting Research-report API at http://127.0.0.1:{REPORT_PORT}…")
                    rep = _popen(
                        [
                            py,
                            "-u",
                            "-m",
                            "uvicorn",
                            "report_api:app",
                            "--host",
                            "127.0.0.1",
                            "--port",
                            str(REPORT_PORT),
                        ],
                        env=env,
                    )
                    children.append(rep)
                    report_pid = rep.pid
                    if not _wait_health(REPORT_HEALTH, report_pid, "Research-report API", attempts=60):
                        cleanup()
                        return 1

        # Autonomy
        if _autonomy_running(env):
            _log("[STACK] A healthy autonomy supervisor is already running; reusing it.")
        else:
            _log("[STACK] Starting autonomy supervisor…")
            auto = _popen([py, "-u", "main.py", "autonomy"], env=env)
            children.append(auto)
            time.sleep(1.0)
            if auto.poll() is not None:
                if _autonomy_running(env):
                    _log("[STACK] Another healthy supervisor acquired the lock; reusing it.")
                else:
                    _log("[STACK] Autonomy failed to stay alive; continuing with API + Vite only.")
                    try:
                        children.remove(auto)
                    except ValueError:
                        pass

        # Terminal API
        if _health_ok(API_HEALTH):
            api_external = True
            _log(f"[STACK] Terminal API already healthy at {API_HEALTH} — reusing.")
        else:
            _free_port(API_PORT, "Terminal API")
            if _health_ok(API_HEALTH):
                api_external = True
                _log("[STACK] Terminal API became healthy after freeing port — reusing.")
            else:
                _log(f"[STACK] Starting Terminal API at http://127.0.0.1:{API_PORT}…")
                api_proc = _popen(
                    [
                        py,
                        "-u",
                        "-m",
                        "uvicorn",
                        "terminal_product_api:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(API_PORT),
                    ],
                    env=env,
                )
                children.append(api_proc)
                api_pid = api_proc.pid

        _log("[STACK] Waiting for terminal API (bhav load can take ~15–45s)…")
        if not _wait_health(API_HEALTH, None if api_external else api_pid, "Terminal API"):
            cleanup()
            return 1
        _log("[STACK] Terminal API ready.")

        # Idle backtest watcher
        if env.get("QT_DISABLE_IDLE_BACKTEST", "").strip() != "1":
            _log("[STACK] Starting idle full-universe backtest watcher…")
            idle = _popen(
                [
                    py,
                    "-u",
                    str(ROOT / "scripts" / "idle_full_universe_backtest.py"),
                    "--idle-seconds",
                    "600",
                ],
                env=env,
            )
            children.append(idle)

        _free_port(VITE_PORT, "Vite dev server")
        _log(f"[STACK] Starting dedicated terminal at http://127.0.0.1:{VITE_PORT} …")
        npm_cmd = [
            "npm",
            "--prefix",
            str(ROOT / "frontend"),
            "run",
            "dev",
            "--",
            "--host",
            "127.0.0.1",
            "--port",
            str(VITE_PORT),
        ]
        if _is_windows():
            front = subprocess.Popen(
                subprocess.list2cmdline(npm_cmd),
                cwd=str(ROOT),
                env=env,
                shell=True,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        else:
            front = _popen(npm_cmd, env=env)
        children.append(front)

        vite_bound = False
        for _ in range(60):
            if _port_listening(VITE_PORT):
                vite_bound = True
                break
            if front.poll() is not None and not _port_listening(VITE_PORT):
                _log("[STACK] Vite exited during startup. Review errors above.")
                cleanup()
                return 1
            time.sleep(0.5)
        if not vite_bound:
            _log(f"[STACK] Vite did not bind :{VITE_PORT} within 30s.")
            cleanup()
            return 1

        mode = "lean" if lean else ("complete" if complete else ("low-power" if low_power else "standard"))
        _log(f"[STACK] QuantTerm is ready ({mode}). Open http://127.0.0.1:{VITE_PORT}")
        _log("[STACK] Keep this window open. Ctrl-C stops services started here.")
        if _is_windows():
            _log("[STACK] Stop later with: .\\scripts\\stop_quantterm.ps1")
            if lean:
                _log("[STACK] Research Data / PDF reports need: .\\scripts\\run_quantterm_complete.ps1")
        else:
            _log("[STACK] Stop later with: bash scripts/stop_quantterm.sh")
            if lean:
                _log("[STACK] Research Data / PDF reports need: bash scripts/run_quantterm_complete.sh")

        api_health_fails = 0
        api_http_fails = 0
        api_dead_limit = 3
        while not shutting_down:
            if not _port_listening(VITE_PORT):
                _log("[STACK] FRONTEND/Vite exited unexpectedly (nothing on :5173).")
                cleanup()
                return 1

            api_process_dead = (not api_external) and api_pid is not None and not _pid_alive(api_pid)
            api_port_down = not _port_listening(API_PORT)
            if api_process_dead or api_port_down:
                api_health_fails += 1
                _log(
                    f"[STACK] Terminal API process/port soft-fail {api_health_fails}/{api_dead_limit} "
                    f"(dead={int(api_process_dead)} port_down={int(api_port_down)})"
                )
                if api_health_fails >= api_dead_limit:
                    _log("[STACK] Terminal API is down on :8765 — restart the stack.")
                    cleanup()
                    return 1
            else:
                api_health_fails = 0
                if not _health_ok(API_HEALTH):
                    api_http_fails += 1
                    if api_http_fails == 1 or api_http_fails % 5 == 0:
                        _log(
                            f"[STACK] Terminal API HTTP health slow/busy at {API_HEALTH} "
                            f"(warn {api_http_fails}; process still up) — not exiting."
                        )
                else:
                    api_http_fails = 0
            time.sleep(3)
    except KeyboardInterrupt:
        cleanup()
        return 0
    finally:
        if not shutting_down:
            cleanup()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QuantTerm cross-platform stack")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup", help="Create venv, install deps, seed .env")
    sub.add_parser("stop", help="Stop API / report / Vite ports")

    run_p = sub.add_parser("run", help="Start terminal stack")
    run_p.add_argument("--complete", action="store_true", help="Also start research-report API :8766")
    run_p.add_argument("--low-power", action="store_true", help="QT_LOW_POWER=1 for older/slower machines")
    run_p.add_argument(
        "--lean",
        action="store_true",
        help="3GB RAM profile: low-power eco + no report API (keeps market scan / autopilot)",
    )

    args = parser.parse_args(argv)
    os.chdir(ROOT)
    if args.cmd == "setup":
        return cmd_setup()
    if args.cmd == "stop":
        return cmd_stop()
    if args.cmd == "run":
        return cmd_run(
            complete=bool(args.complete),
            low_power=bool(args.low_power),
            lean=bool(getattr(args, "lean", False)),
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
