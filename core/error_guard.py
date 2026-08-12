"""
Error Guard — crashes become diagnosable events, not blank screens.

- guard("page"): context manager around a page render. On exception:
  full traceback to logs/errors.log (structured, timestamped), friendly
  error box in the UI with the traceback in an expander — the rest of
  the app keeps working.
- recent_errors(): last N entries for the diagnostics panel.
- check_config(): startup self-check — which integrations are
  configured, which are missing (no more silent half-configured runs).
"""
from __future__ import annotations

import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

_ERR_LOG = Path(__file__).resolve().parent.parent / "logs" / "errors.log"
_MAX_LOG_BYTES = 512 * 1024      # rotate at 512 KB — never grows unbounded


def log_error(context: str, exc: BaseException) -> None:
    try:
        _ERR_LOG.parent.mkdir(parents=True, exist_ok=True)
        if _ERR_LOG.exists() and _ERR_LOG.stat().st_size > _MAX_LOG_BYTES:
            # keep the tail half on rotation
            data = _ERR_LOG.read_text(errors="replace")
            _ERR_LOG.write_text(data[len(data) // 2:])
        with _ERR_LOG.open("a") as f:
            f.write(f"\n{'='*70}\n[{datetime.now().isoformat(timespec='seconds')}] "
                    f"context={context}\n")
            f.write("".join(traceback.format_exception(exc)))
    except Exception:
        pass  # error logging must never itself crash anything


@contextmanager
def guard(context: str):
    """Wrap a page render: exceptions get logged + shown, app survives."""
    try:
        yield
    except Exception as exc:
        log_error(context, exc)
        try:
            import streamlit as st
            st.error(f"⚠️ **{context}** page mein kuch toot gaya — baaki app "
                     f"theek chal raha hai. Detail neeche, aur logs/errors.log "
                     f"mein bhi save ho gaya.")
            with st.expander("🔍 Technical detail (developer ke liye)"):
                st.code("".join(traceback.format_exception(exc)), language="text")
        except Exception:
            raise exc


def recent_errors(limit: int = 5) -> list[str]:
    """Last N error blocks from the log, newest first."""
    try:
        if not _ERR_LOG.exists():
            return []
        blocks = _ERR_LOG.read_text(errors="replace").split("=" * 70)
        return [b.strip() for b in blocks if b.strip()][-limit:][::-1]
    except Exception:
        return []


def check_config() -> list[tuple[str, bool, str]]:
    """[(name, ok, hint)] — every integration's configured/missing state."""
    out: list[tuple[str, bool, str]] = []
    try:
        from config import settings
        out.append(("Kite API key", bool(settings.kite_api_key),
                    "KITE_API_KEY in .env"))
        out.append(("Kite token (daily)", bool(settings.kite_access_token),
                    "python main.py login — roz subah"))
        out.append(("DeepSeek (JARVIS)", bool(settings.deepseek_api_key),
                    "DEEPSEEK_API_KEY in .env"))
        import os
        out.append(("Telegram alerts",
                    bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")),
                    "TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env"))
        # Market gates IST-explicit hain (galat TZ pe bhi SAHI chalte hain),
        # par non-IST machine pe logs/cron timestamps shift dikhte hain.
        try:
            from core.market_clock import system_tz_is_ist
            out.append(("System clock (IST)", system_tz_is_ist(),
                        "server: timedatectl set-timezone Asia/Kolkata "
                        "(gates phir bhi IST pe hi chalte hain)"))
        except Exception:
            pass
        try:
            from pathlib import Path
            _v = (Path(__file__).resolve().parent.parent / "VERSION") \
                .read_text().strip()
            out.append((f"Build v{_v}", True, ""))
        except Exception:
            pass
    except Exception as exc:
        out.append(("config load", False, str(exc)[:80]))
    return out
