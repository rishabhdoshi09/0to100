"""Offer the existing Kite login once from a ready, interactive operator console.

Never run this from a service worker: the console owns browser/input, while the
already-started stack continues running independently of optional broker auth.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def probe_status() -> int:
    from data.kite_client import _fresh_env
    from research.autonomy.auth import TOKEN_MISSING, SESSION_EXPIRED, probe_auth

    if not _fresh_env("KITE_API_KEY") or not _fresh_env("KITE_API_SECRET"):
        return 2
    health = probe_auth()
    if health.status in {TOKEN_MISSING, SESSION_EXPIRED}:
        return 1
    return 0 if health.valid else 3


def offer_login(*, runner=None) -> None:
    runner = runner or subprocess.run
    # Check before importing runtime/provider modules or touching a credential.
    if (not sys.stdin.isatty() or any(os.getenv(key) == "1" for key in
            ("QT_NONINTERACTIVE", "QT_NO_BROWSER", "QT_NO_AUTO_LOGIN"))):
        return
    try:
        result = runner(
            [sys.executable, "-m", "product.startup_login", "--probe"],
            cwd=ROOT, timeout=20, capture_output=True, check=False,
        )
    except subprocess.TimeoutExpired:
        print("[KITE] Session check timed out; desk continues. Retry with python main.py login.")
        return
    if result.returncode == 0:
        print("[KITE] Existing session is valid; no login needed.")
        return
    if result.returncode != 1:
        print("[KITE] Session could not be verified or credentials are missing; desk continues.")
        return
    print("[KITE] Daily login required. Opening browser; paste the redirect URL below.")
    print("[KITE] The desk is already running. Submit an empty line to skip login.")
    result = runner([sys.executable, str(ROOT / "main.py"), "login"],
                    cwd=ROOT, check=False)
    if result.returncode:
        print("[KITE] Login was not completed; desk continues. Retry with python main.py login.")


def main() -> None:
    if "--probe" not in sys.argv and (not sys.stdin.isatty() or any(
            os.getenv(k) == "1" for k in
            ("QT_NONINTERACTIVE", "QT_NO_BROWSER", "QT_NO_AUTO_LOGIN"))):
        return
    # Match the installed service's credentials, including external env files.
    from product.host_entrypoint import load_env_file
    load_env_file(os.environ.get("QT_HOST_ENV_FILE"))
    if "--probe" in sys.argv:
        try:
            code = probe_status()
        except Exception:
            code = 3  # Unknown/provider errors are not evidence of expired auth.
        raise SystemExit(code)
    if not sys.stdin.isatty() or any(os.getenv(k) == "1" for k in
            ("QT_NONINTERACTIVE", "QT_NO_BROWSER", "QT_NO_AUTO_LOGIN")):
        return
    import fcntl
    from core.runtime_paths import ensure_logs_path
    # Two operator consoles must not launch competing token-exchange prompts.
    with ensure_logs_path("startup_kite_login.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[KITE] Login is already being checked in another console.")
            return
        offer_login()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n[KITE] Login skipped.")
