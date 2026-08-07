"""
Stocks hitting upper/lower price circuits on NSE.
A stock hitting upper circuit 3+ days is very different from a pattern breakout alone.
"""

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
    "Accept": "application/json",
}

_DB_PATH = Path("logs/circuit_history.db")
_SESSION: requests.Session | None = None

_CACHE: dict = {}          # {"ts": float, "upper": [...], "lower": [...]}
_CACHE_TTL = 300           # 5 minutes

_UPPER_THRESHOLD = 19.5
_LOWER_THRESHOLD = -19.5


@dataclass
class CircuitInfo:
    symbol: str
    circuit_type: str      # "UPPER" or "LOWER"
    price: float
    consecutive_days: int
    pct_change: float


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _init_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS circuit_history (
                symbol       TEXT NOT NULL,
                date         TEXT NOT NULL,
                circuit_type TEXT NOT NULL,
                PRIMARY KEY (symbol, date)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _record_circuit(symbol: str, date: str, circuit_type: str) -> None:
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO circuit_history (symbol, date, circuit_type)
                VALUES (?, ?, ?)
                """,
                (symbol, date, circuit_type),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def _consecutive_days(symbol: str, circuit_type: str) -> int:
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            rows = conn.execute(
                """
                SELECT date FROM circuit_history
                WHERE symbol = ? AND circuit_type = ?
                ORDER BY date DESC
                LIMIT 30
                """,
                (symbol, circuit_type),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return 1

        dates = sorted([r[0] for r in rows], reverse=True)
        from datetime import date as date_cls, timedelta

        count = 1
        prev = datetime.strptime(dates[0], "%Y-%m-%d").date()
        for d_str in dates[1:]:
            d = datetime.strptime(d_str, "%Y-%m-%d").date()
            if prev - d == timedelta(days=1):
                count += 1
                prev = d
            else:
                break
        return count
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# NSE session
# ---------------------------------------------------------------------------

def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        try:
            _SESSION.get(
                "https://www.nseindia.com/",
                headers=_HEADERS,
                timeout=10,
            )
        except Exception:
            pass
    return _SESSION


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _fetch_nifty500() -> list[dict]:
    session = _get_session()
    try:
        resp = session.get(
            "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500",
            headers=_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception:
        return []


def _build_circuit_list() -> tuple[list[CircuitInfo], list[CircuitInfo]]:
    records = _fetch_nifty500()
    today = _today_str()
    upper: list[CircuitInfo] = []
    lower: list[CircuitInfo] = []

    for rec in records:
        try:
            symbol = str(rec.get("symbol", "")).strip().upper()
            pct = float(rec.get("pChange", 0.0))
            price = float(rec.get("lastPrice", 0.0))

            if pct >= _UPPER_THRESHOLD:
                _record_circuit(symbol, today, "UPPER")
                days = _consecutive_days(symbol, "UPPER")
                upper.append(
                    CircuitInfo(
                        symbol=symbol,
                        circuit_type="UPPER",
                        price=price,
                        consecutive_days=days,
                        pct_change=pct,
                    )
                )
            elif pct <= _LOWER_THRESHOLD:
                _record_circuit(symbol, today, "LOWER")
                days = _consecutive_days(symbol, "LOWER")
                lower.append(
                    CircuitInfo(
                        symbol=symbol,
                        circuit_type="LOWER",
                        price=price,
                        consecutive_days=days,
                        pct_change=pct,
                    )
                )
        except Exception:
            continue

    return upper, lower


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_init_db()


def get_circuit_stocks(circuit_type: str = "both") -> list[CircuitInfo]:
    global _CACHE
    now = time.monotonic()
    if _CACHE and (now - _CACHE.get("ts", 0)) < _CACHE_TTL:
        upper = _CACHE.get("upper", [])
        lower = _CACHE.get("lower", [])
    else:
        try:
            upper, lower = _build_circuit_list()
            _CACHE = {"ts": now, "upper": upper, "lower": lower}
        except Exception:
            return []

    ct = circuit_type.upper()
    if ct == "UPPER":
        return upper
    elif ct == "LOWER":
        return lower
    return upper + lower


def is_in_circuit(symbol: str) -> tuple[bool, str]:
    symbol = symbol.strip().upper()
    try:
        stocks = get_circuit_stocks("both")
        for s in stocks:
            if s.symbol == symbol:
                return True, s.circuit_type
        return False, ""
    except Exception:
        return False, ""
