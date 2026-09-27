"""Forward-only ATM IV history for the NSE F&O paper lane.

No historical option IV is fabricated. The store is populated from observed
near-close option quotes for the mapped stock-F&O universe. Intraday candidates
may compare their current ATM IV only with prior completed-session observations.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import sqlite3
from statistics import median
from typing import Any, Mapping, Sequence

from core.runtime_paths import logs_dir
from data.nfo_market import (
    candidate_option_instruments,
    quote_to_option_contract,
    read_market_quotes,
    read_nfo_quotes,
)

MIN_PRIOR_SESSIONS = 60
LOOKBACK_SESSIONS = 252
SOURCE = "FORWARD_OBSERVED_ATM_IV_CLOSE"


def _f(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def _kind(row: Mapping[str, Any]) -> str:
    return str(row.get("instrument_type") or row.get("option_type") or "").upper()


def _expiry(row: Mapping[str, Any]) -> str:
    return str(row.get("expiry") or "")[:10]


def _nearest_pair(
    option_metas: Sequence[Mapping[str, Any]],
    *,
    spot: float,
) -> list[dict[str, Any]]:
    """Nearest-expiry, nearest-strike CE+PE metadata. Require both sides."""
    spot = _f(spot)
    if spot <= 0:
        return []
    expiries = sorted({_expiry(row) for row in option_metas if _expiry(row)})
    if not expiries:
        return []
    expiry = expiries[0]
    chosen: list[dict[str, Any]] = []
    for side in ("CE", "PE"):
        rows = [
            dict(row)
            for row in option_metas
            if _expiry(row) == expiry and _kind(row) == side and _f(row.get("strike")) > 0
        ]
        if not rows:
            return []
        rows.sort(key=lambda row: abs(_f(row.get("strike")) - spot))
        chosen.append(rows[0])
    return chosen


def representative_atm_iv_pct(
    option_metas: Sequence[Mapping[str, Any]],
    option_quotes: Mapping[str, Mapping[str, Any]],
    *,
    spot: float,
    as_of: date,
) -> dict[str, Any]:
    """Observed near-ATM IV using both CE and PE from the nearest expiry."""
    pair = _nearest_pair(option_metas, spot=spot)
    if len(pair) != 2:
        return {
            "available": False,
            "reason": "ATM_CE_PE_PAIR_UNAVAILABLE",
            "iv_pct": None,
            "contracts": [],
            "source": SOURCE,
        }
    ivs: list[float] = []
    contracts: list[str] = []
    for meta in pair:
        symbol = str(meta.get("tradingsymbol") or "")
        quote = option_quotes.get(symbol)
        if not symbol or not isinstance(quote, Mapping):
            return {
                "available": False,
                "reason": "ATM_OPTION_QUOTE_UNAVAILABLE",
                "iv_pct": None,
                "contracts": contracts,
                "source": SOURCE,
            }
        contract = quote_to_option_contract(meta, quote, spot=spot, as_of=as_of)
        iv = _f(contract.get("iv"))
        if iv <= 0:
            return {
                "available": False,
                "reason": "ATM_OPTION_IV_UNAVAILABLE",
                "iv_pct": None,
                "contracts": contracts,
                "source": SOURCE,
            }
        ivs.append(iv)
        contracts.append(symbol)
    return {
        "available": True,
        "reason": "OBSERVED_NEAREST_EXPIRY_ATM_CE_PE",
        "iv_pct": round(float(median(ivs)), 4),
        "contracts": contracts,
        "source": SOURCE,
    }


class FoIvHistoryStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else logs_dir() / "product" / "fo_iv_history.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fo_iv_history (
                symbol TEXT NOT NULL,
                session TEXT NOT NULL,
                iv_pct REAL NOT NULL,
                source TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(symbol, session)
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "FoIvHistoryStore":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def upsert(self, *, symbol: str, session: str, iv_pct: float, source: str = SOURCE) -> None:
        value = _f(iv_pct)
        if value <= 0:
            return
        self.conn.execute(
            """
            INSERT INTO fo_iv_history(symbol, session, iv_pct, source)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(symbol, session) DO UPDATE SET
                iv_pct=excluded.iv_pct,
                source=excluded.source,
                updated_at=CURRENT_TIMESTAMP
            """,
            (str(symbol or "").upper(), str(session)[:10], value, str(source or SOURCE)),
        )
        self.conn.commit()

    def percentile_before(
        self,
        *,
        symbol: str,
        session: str,
        current_iv_pct: float,
        min_prior_sessions: int = MIN_PRIOR_SESSIONS,
        lookback_sessions: int = LOOKBACK_SESSIONS,
    ) -> dict[str, Any]:
        current = _f(current_iv_pct)
        minimum = max(1, int(min_prior_sessions))
        lookback = max(minimum, int(lookback_sessions))
        rows = self.conn.execute(
            """
            SELECT iv_pct
            FROM fo_iv_history
            WHERE symbol=? AND session<?
            ORDER BY session DESC
            LIMIT ?
            """,
            (str(symbol or "").upper(), str(session)[:10], lookback),
        ).fetchall()
        values = [_f(row[0]) for row in rows if _f(row[0]) > 0]
        n = len(values)
        if current <= 0 or n < minimum:
            return {
                "available": False,
                "percentile_pct": None,
                "prior_sessions": n,
                "minimum_prior_sessions": minimum,
                "lookback_sessions": lookback,
                "current_iv_pct": round(current, 4) if current > 0 else None,
                "source": SOURCE,
                "historical_backfill": False,
            }
        rank = sum(1 for value in values if value <= current) / n * 100.0
        return {
            "available": True,
            "percentile_pct": round(rank, 2),
            "prior_sessions": n,
            "minimum_prior_sessions": minimum,
            "lookback_sessions": lookback,
            "current_iv_pct": round(current, 4),
            "source": SOURCE,
            "historical_backfill": False,
        }

    def status(self) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(session), MAX(session) FROM fo_iv_history"
        ).fetchone()
        return {
            "rows": int(row[0] or 0),
            "symbols": int(row[1] or 0),
            "first_session": str(row[2] or ""),
            "latest_session": str(row[3] or ""),
            "minimum_prior_sessions": MIN_PRIOR_SESSIONS,
            "lookback_sessions": LOOKBACK_SESSIONS,
            "historical_backfill": False,
            "source": SOURCE,
        }


def collect_close_iv_snapshot(
    *,
    symbols: Sequence[str],
    instrument_rows: Sequence[Mapping[str, Any]],
    client,
    as_of: date,
    store: FoIvHistoryStore | None = None,
) -> dict[str, Any]:
    """Persist one unbiased near-close ATM-IV observation per mapped underlying."""
    owned = store is None
    store = store or FoIvHistoryStore()
    try:
        clean = sorted({str(symbol or "").upper() for symbol in symbols if str(symbol or "").strip()})
        spot_quotes = read_market_quotes([f"NSE:{symbol}" for symbol in clean], client=client)

        metas_by_symbol: dict[str, list[dict[str, Any]]] = {}
        quote_symbols: list[str] = []
        spots: dict[str, float] = {}
        for symbol in clean:
            spot = _f((spot_quotes.get(f"NSE:{symbol}") or {}).get("last_price"))
            if spot <= 0:
                continue
            spots[symbol] = spot
            metas = candidate_option_instruments(
                instrument_rows,
                symbol,
                spot=spot,
                as_of=as_of,
                max_expiries=1,
                max_moneyness_pct=8.0,
            )
            pair = _nearest_pair(metas, spot=spot)
            if len(pair) != 2:
                continue
            metas_by_symbol[symbol] = pair
            quote_symbols.extend(str(row.get("tradingsymbol") or "") for row in pair)

        option_quotes = read_nfo_quotes(quote_symbols, client=client)
        persisted = 0
        failures: list[dict[str, str]] = []
        for symbol in clean:
            metas = metas_by_symbol.get(symbol) or []
            if len(metas) != 2:
                failures.append({"symbol": symbol, "reason": "ATM_PAIR_UNAVAILABLE"})
                continue
            observed = representative_atm_iv_pct(
                metas,
                option_quotes,
                spot=spots.get(symbol, 0.0),
                as_of=as_of,
            )
            if not observed.get("available"):
                failures.append({"symbol": symbol, "reason": str(observed.get("reason") or "IV_UNAVAILABLE")})
                continue
            store.upsert(
                symbol=symbol,
                session=as_of.isoformat(),
                iv_pct=float(observed["iv_pct"]),
                source=SOURCE,
            )
            persisted += 1
        return {
            "available": True,
            "session": as_of.isoformat(),
            "requested_symbols": len(clean),
            "persisted_symbols": persisted,
            "failed_symbols": len(failures),
            "failures": failures[:50],
            "store": store.status(),
            "historical_backfill": False,
            "source": SOURCE,
        }
    finally:
        if owned:
            store.close()
