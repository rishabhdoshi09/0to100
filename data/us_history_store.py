"""US EOD history disk cache — Yahoo as the honest free primary source.

Kite has no US equities. This store persists daily OHLCV under ``logs/us_bhav/``
so market-ops and Stock Intelligence do not re-hit Yahoo on every request.
Never invents bars: missing symbols stay missing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from logger import get_logger

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
STORE_DIR = ROOT / "logs" / "us_bhav"
META_PATH = STORE_DIR / "meta.json"
_MIN_BARS = 60


def _symbol_path(symbol: str) -> Path:
    clean = "".join(ch for ch in str(symbol).upper() if ch.isalnum() or ch in ("-", "."))
    return STORE_DIR / f"{clean}.json"


def _frame_to_rows(df) -> list[dict[str, Any]]:
    rows = []
    for index, row in df.iterrows():
        stamp = getattr(index, "date", lambda: index)()
        rows.append({
            "date": str(stamp),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0.0) or 0.0),
        })
    return rows


def _rows_to_frame(rows: list[dict[str, Any]]):
    import pandas as pd

    if not rows:
        return None
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    return frame[["open", "high", "low", "close", "volume"]]


def save_symbol(symbol: str, df) -> bool:
    if df is None or len(df) < _MIN_BARS:
        return False
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    path = _symbol_path(symbol)
    payload = {
        "symbol": str(symbol).upper(),
        "source": "yfinance",
        "saved_at": time.time(),
        "bars": _frame_to_rows(df),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)
    return True


def load_symbol(symbol: str):
    path = _symbol_path(symbol)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = list(payload.get("bars") or [])
        frame = _rows_to_frame(rows)
        if frame is None or len(frame) < _MIN_BARS:
            return None
        return frame
    except Exception:
        return None


def cached_symbols() -> list[str]:
    if not STORE_DIR.exists():
        return []
    out = []
    for path in STORE_DIR.glob("*.json"):
        if path.name == "meta.json":
            continue
        out.append(path.stem.upper())
    return sorted(out)


def status() -> dict[str, Any]:
    symbols = cached_symbols()
    meta: dict[str, Any] = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    latest = ""
    sample = symbols[:5]
    for sym in sample:
        frame = load_symbol(sym)
        if frame is not None and len(frame):
            latest = max(latest, str(frame.index[-1].date()))
    ready = len(symbols) >= 40
    return {
        "ready": ready,
        "symbols": len(symbols),
        "latest_date": latest,
        "source": "yfinance",
        "source_note": (
            "Yahoo Finance is the free primary EOD source for US equities in QuantTerm "
            "(Kite has no US cash market). Incomplete/delayed bars are never invented."
        ),
        "cache_dir": str(STORE_DIR),
        "last_prepare_at": meta.get("last_prepare_at"),
        "last_scope": meta.get("scope"),
        "last_prepared_count": meta.get("prepared_count"),
        "minimum_symbols": 40,
    }


def prepare_history(
    symbols: Iterable[str],
    *,
    lookback_days: int = 400,
    progress: Callable[[int, int], None] | None = None,
    scope: str = "S&P 500",
) -> dict[str, Any]:
    """Fetch Yahoo daily bars for ``symbols`` and persist to disk.

    Returns readiness stats. Partial success is honest — prepared count can be
    less than requested when Yahoo has no usable history.
    """
    from data.us_data import get_us_daily_batch

    wanted = [str(s).strip().upper() for s in symbols if str(s).strip()]
    wanted = list(dict.fromkeys(wanted))
    total = max(1, len(wanted))
    prepared = 0
    batch = 80
    for i in range(0, len(wanted), batch):
        chunk = wanted[i:i + batch]
        frames = get_us_daily_batch(chunk, lookback_days=lookback_days)
        for sym, df in frames.items():
            if save_symbol(sym, df):
                prepared += 1
        if callable(progress):
            try:
                progress(min(i + len(chunk), len(wanted)), total)
            except Exception:
                pass
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(
        json.dumps({
            "last_prepare_at": time.time(),
            "scope": scope,
            "requested": len(wanted),
            "prepared_count": prepared,
            "source": "yfinance",
        }, indent=2),
        encoding="utf-8",
    )
    result = status()
    result["requested"] = len(wanted)
    result["prepared_count"] = prepared
    result["scope"] = scope
    log.info("us_history_prepared", requested=len(wanted), prepared=prepared, scope=scope)
    return result


def get_ohlcv(symbol: str, *, allow_network: bool = True):
    """Disk cache first; optional Yahoo fetch on miss (never invents)."""
    frame = load_symbol(symbol)
    if frame is not None:
        return frame
    if not allow_network:
        return None
    try:
        from data.us_data import get_us_daily

        df = get_us_daily(str(symbol).upper())
        if df is not None and save_symbol(symbol, df):
            return df
        return df
    except Exception:
        return None
