"""Canonical, Streamlit-free whole-market scan service.

The autonomy supervisor and the retail Momentum page call this exact service. UI code may pass a
progress callback, but this module never imports Streamlit or any ``ui.*`` module. A provider failure
is reported as failure; a healthy scan with zero setups is a valid result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Any


SUCCEEDED = "SUCCEEDED"
NO_SETUPS = "NO_SETUPS"
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
FAILED = "FAILED"

try:
    from research.feature002.observe import try_observe_production_scan as _feature002_hook
except Exception:
    _feature002_hook = None


@dataclass(frozen=True)
class MarketScanReport:
    status: str
    payload: dict = field(default_factory=dict)
    universe_size: int = 0
    scanned: int = 0
    approved_universe: int = 0
    exclusions: tuple = ()
    source_snapshot_id: str = ""
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (SUCCEEDED, NO_SETUPS)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "payload": self.payload,
            "approved_universe": self.approved_universe,
            "universe_size": self.universe_size,
            "scanned": self.scanned,
            "exclusions": list(self.exclusions),
            "source_snapshot_id": self.source_snapshot_id,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


def _symbol_list_from_records(payload: Mapping[str, Any] | None, *, setups_only: bool = False) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    out: list[str] = []
    for row in payload.get("records") or []:
        if not isinstance(row, Mapping):
            continue
        if setups_only:
            signals = row.get("signals") or row.get("reasons") or []
            if not signals:
                continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol:
            out.append(symbol)
    return out


def _reco_symbols(payload: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    out: list[str] = []
    categories = payload.get("categories")
    if isinstance(categories, Mapping):
        for rows in categories.values():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, Mapping):
                    symbol = str(row.get("symbol") or "").strip().upper()
                    if symbol:
                        out.append(symbol)
    out.extend(_symbol_list_from_records(payload))
    return out


def priority_ordered_symbols(
    symbols: list[str],
    *,
    scan_payload: Mapping[str, Any] | None = None,
    reco_payload: Mapping[str, Any] | None = None,
    long_term_payload: Mapping[str, Any] | None = None,
    fno_symbols: set[str] | list[str] | tuple[str, ...] | None = None,
    watchlist: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Scan prior-interest names first, but still walk the entire approved universe."""
    allowed = {str(s).strip().upper() for s in symbols if str(s).strip()}
    seen: set[str] = set()
    ordered: list[str] = []

    def _add(seq: list[str] | tuple[str, ...] | set[str] | None) -> None:
        if not seq:
            return
        for raw in seq:
            symbol = str(raw).strip().upper()
            if symbol in allowed and symbol not in seen:
                seen.add(symbol)
                ordered.append(symbol)

    _add(_symbol_list_from_records(scan_payload, setups_only=True))
    _add(watchlist)
    _add(_reco_symbols(reco_payload))
    _add(_symbol_list_from_records(long_term_payload))
    _add(list(fno_symbols or []))
    _add(sorted(allowed))
    return ordered


def _saved_priority_inputs() -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, Mapping[str, Any] | None, list[str]]:
    scan_payload = None
    reco_payload = None
    long_term_payload = None
    watchlist: list[str] = []
    try:
        from product.scan_store import load_scan, watchlist_rows
        scan_payload = load_scan()
        watchlist = [
            str(row.get("symbol") or "").strip().upper()
            for row in watchlist_rows(scan_payload) or []
            if str(row.get("symbol") or "").strip()
        ]
    except Exception:
        pass
    try:
        from product.recommendations_store import load_recommendations
        reco_payload = load_recommendations()
    except Exception:
        pass
    try:
        from product.long_term_store import load_long_term_scan
        long_term_payload = load_long_term_scan()
    except Exception:
        pass
    return scan_payload, reco_payload, long_term_payload, watchlist


def _default_universe() -> Mapping[str, str]:
    """Return every current NSE EQ instrument we can prove, never just rows with names.

    The previous implementation used ``get_nse_universe_with_names()`` as the
    universe itself. That meant a valid symbol could disappear simply because its
    company-name field was absent. The Kite instrument master is now preferred and
    filtered by exchange + instrument_type only; names are metadata and fall back to
    the symbol. If Kite is unavailable, the authoritative symbol list is still used
    and the name map is joined onto it instead of defining it.
    """
    try:
        from data.instruments import InstrumentManager
        manager = InstrumentManager()
        rows = getattr(manager, "_meta_map", {}) or {}
        out: dict[str, str] = {}
        for raw_symbol, row in rows.items():
            if not isinstance(row, Mapping):
                continue
            if str(row.get("exchange") or "").strip().upper() != "NSE":
                continue
            if str(row.get("instrument_type") or "").strip().upper() != "EQ":
                continue
            symbol = str(raw_symbol or row.get("tradingsymbol") or "").strip().upper()
            if not symbol:
                continue
            out[symbol] = str(row.get("name") or symbol).strip() or symbol
        if len(out) >= 200:
            return out
    except Exception:
        pass

    from data.nse_universe import get_nse_universe, get_nse_universe_with_names
    symbols = [str(s).strip().upper() for s in (get_nse_universe() or []) if str(s).strip()]
    try:
        names = dict(get_nse_universe_with_names() or {})
    except Exception:
        names = {}
    return {symbol: str(names.get(symbol) or symbol) for symbol in symbols}


def _default_prefetch(symbols, *, progress=None):
    from scan.bulk_fetcher import prefetch
    return prefetch(symbols, progress=progress)


def _default_scanner():
    from core.eco import workers
    from scan.unified_scanner import UnifiedScanner
    return UnifiedScanner(max_workers=workers(12))


def _default_fno_symbols() -> set[str]:
    from data.fno_universe import current_fno_universe
    return set(current_fno_universe().symbols)


def _active_snapshot_id() -> str:
    try:
        from research.intelligence.data.snapshot_store import SnapshotStore
        return str(SnapshotStore().get_active_snapshot() or "")
    except Exception:
        return ""


def _coverage_exclusions(summary: Mapping[str, Any] | None) -> tuple[str, ...]:
    counts = dict((summary or {}).get("reason_counts") or {})
    return tuple(
        f"{key}={int(value or 0)}"
        for key, value in sorted(counts.items())
        if key not in {"QUALIFIED", "NO_SETUP"} and int(value or 0) > 0
    )


def run_whole_market_scan(
    *,
    universe_provider: Callable[[], Mapping[str, str]] | None = None,
    prefetch_fn: Callable[..., Any] | None = None,
    scanner=None,
    fno_provider: Callable[[], set[str] | list[str] | tuple[str, ...]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    save: bool = True,
    snapshot_id: str | None = None,
) -> MarketScanReport:
    """Run and optionally persist one deterministic broad-NSE scan.

    A successful run now also persists a stock-by-stock coverage ledger. A symbol
    may fail to qualify, be excluded by an explicit policy gate, lack enough data,
    or raise an analysis error, but it may no longer silently disappear.
    """
    universe_provider = universe_provider or _default_universe
    prefetch_fn = prefetch_fn or _default_prefetch
    scanner = scanner or _default_scanner()
    fno_provider = fno_provider or _default_fno_symbols

    try:
        names = dict(universe_provider() or {})
    except Exception as exc:
        return MarketScanReport(DATA_UNAVAILABLE, error_code="UNIVERSE_ERROR",
                                error_message=str(exc), source_snapshot_id=snapshot_id or "")
    approved_n = len(names)
    symbols = sorted({str(s).strip().upper() for s in names if str(s).strip()})
    universe_n = len(symbols)
    if not symbols:
        return MarketScanReport(
            DATA_UNAVAILABLE,
            universe_size=0,
            scanned=0,
            approved_universe=approved_n,
            error_code="EMPTY_UNIVERSE",
            error_message="approved NSE universe is empty",
            source_snapshot_id=snapshot_id or "",
        )

    scan_payload, reco_payload, long_term_payload, watchlist = _saved_priority_inputs()
    try:
        fno_for_order = set(fno_provider() or ())
    except Exception:
        fno_for_order = set()
    symbols = priority_ordered_symbols(
        symbols,
        scan_payload=scan_payload,
        reco_payload=reco_payload,
        long_term_payload=long_term_payload,
        fno_symbols=fno_for_order,
        watchlist=watchlist,
    )

    walked_total = 0

    def _on_progress(current, total=0, **kw):
        nonlocal walked_total
        try:
            walked_total = max(walked_total, int(total or 0))
        except (TypeError, ValueError):
            pass
        if progress_callback:
            progress_callback(current, total, **kw)

    audit: dict[str, Any] = {"summary": {}, "ledger": []}
    try:
        # Prefetch warms OHLCV. Do not pass the stock-scan callback — bulk
        # prefetch reports bhavcopy days, not symbols.
        try:
            prefetch_fn(symbols, progress=None)
        except TypeError:
            prefetch_fn(symbols)

        from scan.scan_coverage import observe_scanner
        with observe_scanner(scanner, symbols) as probe:
            try:
                results = list(scanner.scan(symbols, progress=_on_progress, prefetch=False) or [])
            except TypeError:
                results = list(scanner.scan(symbols) or [])

        try:
            from scan.bulk_fetcher import cached_symbols
            cached = list(cached_symbols() or [])
        except Exception:
            cached = []
        audit = probe.finalize(results, cached=cached, walked_total=walked_total)
    except Exception as exc:
        return MarketScanReport(
            FAILED,
            universe_size=universe_n,
            scanned=0,
            approved_universe=approved_n,
            error_code="SCAN_ERROR",
            error_message=str(exc),
            source_snapshot_id=snapshot_id or "",
        )

    coverage = dict(audit.get("summary") or {})
    if not results:
        warm: list[str] = []
        try:
            from scan.bulk_fetcher import cached_symbols
            warm = [str(s).upper() for s in (cached_symbols() or [])]
        except Exception:
            warm = []
        if not warm:
            if save:
                try:
                    from scan.scan_coverage import save_audit
                    save_audit(audit)
                except Exception:
                    pass
            return MarketScanReport(
                DATA_UNAVAILABLE,
                universe_size=universe_n,
                scanned=0,
                approved_universe=approved_n,
                exclusions=_coverage_exclusions(coverage),
                error_code="OHLCV_CACHE_EMPTY",
                error_message="OHLCV cache was empty; the last readable scan was kept.",
                source_snapshot_id=snapshot_id or _active_snapshot_id(),
            )

    fno_symbols = set(fno_for_order)

    from product.scan_store import build_scan_payload, save_scan
    if coverage.get("scanner_instrumented"):
        scanned_n = int(coverage.get("checked") or 0)
    else:
        scanned_n = walked_total or universe_n
    payload = build_scan_payload(
        names,
        results,
        fno_symbols,
        scanned=scanned_n,
        approved_universe=approved_n,
    )
    payload["universe_size"] = universe_n
    payload["coverage"] = coverage
    payload["coverage_state"] = str(coverage.get("state") or "UNKNOWN")
    payload["coverage_warning"] = (
        "Some approved NSE equities did not receive a full technical evaluation. Open Scan Coverage for exact symbols and reasons."
        if coverage.get("state") == "DEGRADED" else ""
    )
    sid = snapshot_id if snapshot_id is not None else _active_snapshot_id()
    payload["source_snapshot_id"] = sid
    payload["scan_status"] = SUCCEEDED
    if save:
        try:
            from scan.scan_coverage import save_audit
            save_audit(audit)
        except Exception:
            pass
        save_scan(payload)
        try:
            from product.sepa_setup import persist_public_best_setups
            persist_public_best_setups(payload)
        except Exception:
            pass
        try:
            from scan.long_term_service import overlay_long_term_from_market_scan
            overlay = overlay_long_term_from_market_scan(
                payload, refresh_fundamentals=False, save=True,
            )
            payload["long_term_overlay"] = {
                "status": overlay.status,
                "records": len((overlay.payload or {}).get("records") or []),
                "error_code": overlay.error_code,
            }
        except Exception as exc:
            payload["long_term_overlay"] = {
                "status": FAILED,
                "records": 0,
                "error_code": type(exc).__name__,
            }
        try:
            from product.desk_scan_overlays import persist_desks_from_market_scan
            payload["desk_overlays"] = persist_desks_from_market_scan(payload)
        except Exception as exc:
            payload["desk_overlays"] = {"error": type(exc).__name__}
        # FEATURE-002 is observe-only and runs AFTER production results are final.
        if _feature002_hook is not None:
            try:
                _feature002_hook(payload.get("records") or [])
            except Exception:
                pass
    summary = dict(payload.get("summary", {}))
    n_setups = int(summary.get("with_any_setup", 0) or 0)
    status = SUCCEEDED if n_setups else NO_SETUPS
    payload["scan_status"] = status
    return MarketScanReport(
        status=status,
        payload=payload,
        approved_universe=approved_n,
        universe_size=universe_n,
        scanned=scanned_n,
        exclusions=_coverage_exclusions(coverage),
        source_snapshot_id=sid,
    )
