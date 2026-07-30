"""Dynamic F&O underlyings for options-flow scanning.

The operational universe is derived from the current Kite NSE/NFO instrument
master. It includes individual equities and supported index underlyings that
have at least one non-expired futures or options contract. No hand-picked
shortlist is used as a runtime fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from data.fno_universe import DEFAULT_CACHE_PATH, load_current_instruments


_INDEX_ALIASES = {
    "NIFTY": "NIFTY",
    "NIFTY 50": "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "NIFTY BANK": "BANKNIFTY",
    "FINNIFTY": "FINNIFTY",
    "NIFTY FIN SERVICE": "FINNIFTY",
    "MIDCPNIFTY": "MIDCPNIFTY",
    "NIFTY MID SELECT": "MIDCPNIFTY",
    "NIFTYNXT50": "NIFTYNXT50",
    "NIFTY NEXT 50": "NIFTYNXT50",
    "SENSEX": "SENSEX",
    "BANKEX": "BANKEX",
}


@dataclass(frozen=True)
class OptionsUnderlying:
    symbol: str
    kind: str
    contract_count: int
    nearest_expiry: str


@dataclass(frozen=True)
class OptionsUniverseReport:
    underlyings: tuple[OptionsUnderlying, ...]
    exclusions: tuple[str, ...]
    total_rows: int
    derivative_contracts: int
    source: str
    loaded_at: datetime
    cache_modified_at: datetime | None = None

    @property
    def symbols(self) -> list[str]:
        return [item.symbol for item in self.underlyings]

    @property
    def stock_count(self) -> int:
        return sum(item.kind == "stock" for item in self.underlyings)

    @property
    def index_count(self) -> int:
        return sum(item.kind == "index" for item in self.underlyings)


def _norm(value: Any) -> str:
    return str(value or "").strip().upper()


def _expiry(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text[:19], fmt).date()
        except ValueError:
            pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def build_options_underlyings(
    rows: Iterable[Mapping[str, Any]],
    *,
    as_of: date | None = None,
    source: str = "instrument_master",
    loaded_at: datetime | None = None,
    cache_modified_at: datetime | None = None,
) -> OptionsUniverseReport:
    as_of = as_of or date.today()
    loaded_at = loaded_at or datetime.now()
    materialized = [dict(row) for row in rows]

    cash_symbols: set[str] = set()
    cash_names: dict[str, str] = {}
    for row in materialized:
        if _norm(row.get("exchange")) == "NSE" and _norm(row.get("instrument_type")) == "EQ":
            symbol = _norm(row.get("tradingsymbol"))
            name = _norm(row.get("name"))
            if symbol:
                cash_symbols.add(symbol)
                if name:
                    cash_names.setdefault(name, symbol)

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    derivative_contracts = 0
    exclusions: list[str] = []
    for row in materialized:
        exchange = _norm(row.get("exchange"))
        segment = _norm(row.get("segment"))
        instrument_type = _norm(row.get("instrument_type"))
        if exchange not in {"NFO", "BFO"}:
            continue
        if instrument_type not in {"FUT", "CE", "PE"} and segment not in {
            "NFO-FUT", "NFO-OPT", "BFO-FUT", "BFO-OPT"
        }:
            continue
        derivative_contracts += 1
        expiry = _expiry(row.get("expiry"))
        if expiry is not None and expiry < as_of:
            continue

        name = _norm(row.get("name"))
        trading_symbol = _norm(row.get("tradingsymbol"))
        index_symbol = _INDEX_ALIASES.get(name) or _INDEX_ALIASES.get(trading_symbol)
        if index_symbol:
            key = ("index", index_symbol)
        else:
            stock_symbol = name if name in cash_symbols else cash_names.get(name)
            if not stock_symbol and trading_symbol in cash_symbols:
                stock_symbol = trading_symbol
            if not stock_symbol:
                exclusions.append(name or trading_symbol or "unnamed derivative")
                continue
            key = ("stock", stock_symbol)
        grouped.setdefault(key, []).append(row)

    underlyings: list[OptionsUnderlying] = []
    for (kind, symbol), contracts in sorted(grouped.items(), key=lambda item: item[0][1]):
        expiries = sorted(exp for exp in (_expiry(row.get("expiry")) for row in contracts) if exp)
        underlyings.append(
            OptionsUnderlying(
                symbol=symbol,
                kind=kind,
                contract_count=len(contracts),
                nearest_expiry=expiries[0].isoformat() if expiries else "",
            )
        )

    return OptionsUniverseReport(
        underlyings=tuple(underlyings),
        exclusions=tuple(sorted(set(exclusions))),
        total_rows=len(materialized),
        derivative_contracts=derivative_contracts,
        source=source,
        loaded_at=loaded_at,
        cache_modified_at=cache_modified_at,
    )


def current_options_underlyings(
    client: Any | None = None,
    *,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    as_of: date | None = None,
) -> OptionsUniverseReport:
    rows, source = load_current_instruments(client, cache_path=cache_path)
    cache_modified_at = None
    path = Path(cache_path)
    if source == "instrument_cache" and path.exists():
        cache_modified_at = datetime.fromtimestamp(path.stat().st_mtime)
    return build_options_underlyings(
        rows,
        as_of=as_of,
        source=source,
        loaded_at=datetime.now(),
        cache_modified_at=cache_modified_at,
    )
