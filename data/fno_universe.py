"""Current NSE individual-stock F&O universe and transparent evaluation funnel.

The Kite instrument dump contains NSE cash instruments and every NFO contract.
This module collapses all current stock-futures expiries into one underlying,
keeps the nearest listed future as display metadata, and records why any
underlying cannot be evaluated. It deliberately does not place orders.
"""
from __future__ import annotations

import csv
import io
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

DEFAULT_CACHE_PATH = Path("logs/instruments_cache.csv")

INDEX_UNDERLYINGS = frozenset(
    {
        "NIFTY",
        "NIFTY 50",
        "BANKNIFTY",
        "NIFTY BANK",
        "FINNIFTY",
        "NIFTY FIN SERVICE",
        "MIDCPNIFTY",
        "NIFTY MID SELECT",
        "NIFTYNXT50",
        "NIFTY NEXT 50",
        "SENSEX",
        "BANKEX",
    }
)


@dataclass(frozen=True)
class FnoUnderlying:
    symbol: str
    company_name: str
    future_symbol: str
    expiry: str
    lot_size: int
    instrument_token: int
    contract_count: int


@dataclass(frozen=True)
class UniverseExclusion:
    underlying: str
    stage: str
    reason: str


@dataclass(frozen=True)
class FnoUniverseReport:
    underlyings: tuple[FnoUnderlying, ...]
    exclusions: tuple[UniverseExclusion, ...]
    total_instrument_rows: int
    total_future_contracts: int
    index_future_contracts: int
    unique_stock_underlyings: int
    mapped_underlyings: int
    source: str = "unknown"

    @property
    def symbols(self) -> list[str]:
        return [item.symbol for item in self.underlyings]


@dataclass(frozen=True)
class MomentumEvaluation:
    symbol: str
    company_name: str
    future_symbol: str
    expiry: str
    lot_size: int
    qualified: bool
    stage: str
    reason: str
    score: float = 0.0
    verdict: str = ""
    price: float = 0.0
    momentum_5d: float = 0.0
    rsi: float = 0.0
    volume_ratio: float = 0.0
    signals: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MomentumFunnel:
    rows: tuple[MomentumEvaluation, ...]
    total_underlyings: int
    data_ready: int
    evaluated: int
    momentum_qualified: int

    @property
    def qualified(self) -> list[MomentumEvaluation]:
        return [row for row in self.rows if row.qualified]

    @property
    def excluded(self) -> list[MomentumEvaluation]:
        return [row for row in self.rows if not row.qualified]


def _norm(value: Any) -> str:
    return str(value or "").strip().upper()


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _expiry(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text[:19], fmt).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _nearest_contract(rows: Sequence[Mapping[str, Any]], as_of: date) -> Mapping[str, Any]:
    dated = [(row, _expiry(row.get("expiry"))) for row in rows]
    future = [(row, exp) for row, exp in dated if exp is not None and exp >= as_of]
    pool = future or [(row, exp) for row, exp in dated if exp is not None]
    if pool:
        return min(pool, key=lambda item: item[1])[0]
    return rows[0]


def build_fno_universe(
    rows: Iterable[Mapping[str, Any]],
    *,
    as_of: date | None = None,
    source: str = "instrument_master",
) -> FnoUniverseReport:
    """Collapse every current individual-stock future contract to one underlying."""
    as_of = as_of or date.today()
    materialized = [dict(row) for row in rows]

    equities_by_symbol: dict[str, Mapping[str, Any]] = {}
    equities_by_name: dict[str, Mapping[str, Any]] = {}
    futures_by_underlying: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    index_contracts = 0
    future_contracts = 0

    for row in materialized:
        exchange = _norm(row.get("exchange"))
        segment = _norm(row.get("segment"))
        instrument_type = _norm(row.get("instrument_type"))
        symbol = _norm(row.get("tradingsymbol"))
        name = _norm(row.get("name"))

        if exchange == "NSE" and instrument_type == "EQ" and symbol:
            equities_by_symbol[symbol] = row
            if name:
                equities_by_name.setdefault(name, row)
            continue

        is_future = exchange == "NFO" and (
            instrument_type == "FUT" or segment == "NFO-FUT"
        )
        if not is_future:
            continue

        future_contracts += 1
        underlying = name or symbol
        if underlying in INDEX_UNDERLYINGS:
            index_contracts += 1
            continue
        futures_by_underlying[underlying].append(row)

    mapped: list[FnoUnderlying] = []
    exclusions: list[UniverseExclusion] = []

    for underlying in sorted(futures_by_underlying):
        contracts = futures_by_underlying[underlying]
        equity = equities_by_symbol.get(underlying) or equities_by_name.get(underlying)
        if equity is None:
            exclusions.append(
                UniverseExclusion(
                    underlying=underlying,
                    stage="canonical_mapping",
                    reason="Current F&O underlying could not be mapped to an NSE cash equity",
                )
            )
            continue

        contract = _nearest_contract(contracts, as_of)
        symbol = _norm(equity.get("tradingsymbol"))
        company_name = str(equity.get("name") or symbol).strip()
        mapped.append(
            FnoUnderlying(
                symbol=symbol,
                company_name=company_name,
                future_symbol=_norm(contract.get("tradingsymbol")),
                expiry=str(contract.get("expiry") or ""),
                lot_size=_as_int(contract.get("lot_size"), 1),
                instrument_token=_as_int(contract.get("instrument_token"), 0),
                contract_count=len(contracts),
            )
        )

    return FnoUniverseReport(
        underlyings=tuple(mapped),
        exclusions=tuple(exclusions),
        total_instrument_rows=len(materialized),
        total_future_contracts=future_contracts,
        index_future_contracts=index_contracts,
        unique_stock_underlyings=len(futures_by_underlying),
        mapped_underlyings=len(mapped),
        source=source,
    )


def parse_instrument_csv(text: str) -> list[dict[str, str]]:
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]


def load_cached_instruments(path: str | Path = DEFAULT_CACHE_PATH) -> list[dict[str, str]]:
    cache_path = Path(path)
    if not cache_path.exists():
        return []
    return parse_instrument_csv(cache_path.read_text(encoding="utf-8"))


def load_current_instruments(
    client: Any | None = None,
    *,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> tuple[list[dict[str, Any]], str]:
    """Read the current master from a data-only Kite client, then cache fallback."""
    if client is not None:
        nse = list(client.instruments("NSE"))
        nfo = list(client.instruments("NFO"))
        rows = [dict(row) for row in nse + nfo]
        if rows:
            return rows, "zerodha_kite"
    rows = load_cached_instruments(cache_path)
    return rows, "instrument_cache" if rows else "unavailable"


def current_fno_universe(
    client: Any | None = None,
    *,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    as_of: date | None = None,
) -> FnoUniverseReport:
    rows, source = load_current_instruments(client, cache_path=cache_path)
    return build_fno_universe(rows, as_of=as_of, source=source)


def evaluate_all_underlyings(
    universe: FnoUniverseReport,
    *,
    history_getter: Callable[[str], Any],
    analyzer: Callable[[str, Any], Any],
    minimum_sessions: int = 60,
) -> MomentumFunnel:
    """Evaluate every mapped F&O underlying and record one outcome per symbol."""
    rows: list[MomentumEvaluation] = []
    data_ready = 0
    evaluated = 0
    qualified = 0

    for item in universe.underlyings:
        try:
            history = history_getter(item.symbol)
        except Exception as exc:
            rows.append(
                MomentumEvaluation(
                    symbol=item.symbol,
                    company_name=item.company_name,
                    future_symbol=item.future_symbol,
                    expiry=item.expiry,
                    lot_size=item.lot_size,
                    qualified=False,
                    stage="history",
                    reason=f"Historical data could not be read: {exc}",
                )
            )
            continue

        try:
            sessions = len(history) if history is not None else 0
        except Exception:
            sessions = 0
        if sessions < minimum_sessions:
            rows.append(
                MomentumEvaluation(
                    symbol=item.symbol,
                    company_name=item.company_name,
                    future_symbol=item.future_symbol,
                    expiry=item.expiry,
                    lot_size=item.lot_size,
                    qualified=False,
                    stage="history",
                    reason=f"Insufficient history: {sessions} sessions available, {minimum_sessions} required",
                )
            )
            continue

        data_ready += 1
        try:
            result = analyzer(item.symbol, history)
        except Exception as exc:
            rows.append(
                MomentumEvaluation(
                    symbol=item.symbol,
                    company_name=item.company_name,
                    future_symbol=item.future_symbol,
                    expiry=item.expiry,
                    lot_size=item.lot_size,
                    qualified=False,
                    stage="analysis",
                    reason=f"Momentum evaluation failed: {exc}",
                )
            )
            continue

        evaluated += 1
        if result is None:
            rows.append(
                MomentumEvaluation(
                    symbol=item.symbol,
                    company_name=item.company_name,
                    future_symbol=item.future_symbol,
                    expiry=item.expiry,
                    lot_size=item.lot_size,
                    qualified=False,
                    stage="safety_checks",
                    reason="Failed price, liquidity, trend or safety checks",
                )
            )
            continue

        signals = tuple(str(signal) for signal in getattr(result, "signals", ()) or ())
        is_momentum = "MOMENTUM" in signals
        reasons = list(getattr(result, "reasons", ()) or ())
        if is_momentum:
            qualified += 1
            reason = next(
                (str(r) for r in reasons if "day" in str(r).lower() or "momentum" in str(r).lower()),
                str(reasons[0]) if reasons else "Momentum conditions passed",
            )
        else:
            reason = "Momentum conditions not met"
            if reasons:
                reason += f"; strongest setup: {reasons[0]}"

        rows.append(
            MomentumEvaluation(
                symbol=item.symbol,
                company_name=item.company_name,
                future_symbol=item.future_symbol,
                expiry=item.expiry,
                lot_size=item.lot_size,
                qualified=is_momentum,
                stage="qualified" if is_momentum else "momentum",
                reason=reason,
                score=float(getattr(result, "score", 0.0) or 0.0),
                verdict=str(getattr(result, "verdict", "") or ""),
                price=float(getattr(result, "price", 0.0) or 0.0),
                momentum_5d=float(getattr(result, "momentum_5d", 0.0) or 0.0),
                rsi=float(getattr(result, "rsi", 0.0) or 0.0),
                volume_ratio=float(getattr(result, "volume_ratio", 0.0) or 0.0),
                signals=signals,
            )
        )

    return MomentumFunnel(
        rows=tuple(rows),
        total_underlyings=len(universe.underlyings),
        data_ready=data_ready,
        evaluated=evaluated,
        momentum_qualified=qualified,
    )
