"""Durable US whole-market scan for the React retail product.

Reuses ``UnifiedScanner._analyze`` (same edge engine as NSE) with S&P RS
benchmark and US quality floor. Persists to ``logs/product/latest_us_scan.json``.
Never invents setups: empty result with healthy data is ``NO_SETUPS``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from logger import get_logger

log = get_logger(__name__)

US_SCAN_PATH = Path("logs/product/latest_us_scan.json")
SUCCEEDED = "SUCCEEDED"
NO_SETUPS = "NO_SETUPS"
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
FAILED = "FAILED"


@dataclass(frozen=True)
class UsMarketScanReport:
    status: str
    payload: dict = field(default_factory=dict)
    universe_size: int = 0
    scanned: int = 0
    scope: str = "S&P 500"
    source: str = "yfinance"
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (SUCCEEDED, NO_SETUPS)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "payload": self.payload,
            "universe_size": self.universe_size,
            "scanned": self.scanned,
            "scope": self.scope,
            "source": self.source,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "market": "US",
            "places_orders": False,
        }


def _scope_universe(scope: str) -> tuple[dict[str, str], str]:
    from data.us_universe import get_us_universe_with_names

    label = (scope or "S&P 500").strip() or "S&P 500"
    if label.lower() in ("all", "full"):
        names = dict(get_us_universe_with_names() or {})
        return names, "All"
    try:
        from data.us_indices import get_index_members

        members, src = get_index_members(label)
        if members:
            all_names = get_us_universe_with_names()
            names = {s: all_names.get(s, s) for s in sorted(members)}
            return names, f"{label} ({src})"
    except Exception as exc:
        log.debug("us_scope_failed", scope=label, error=str(exc)[:80])
    # Fail closed to curated liquid names — never scan an empty invented book.
    from data.us_universe import _CURATED

    return dict(_CURATED), "Curated liquid (index unavailable)"


def _liquid_first(symbols: list[str]) -> list[str]:
    try:
        from data.us_universe import _CURATED

        head = [s for s in _CURATED if s in symbols]
        seen = set(head)
        return head + [s for s in symbols if s not in seen]
    except Exception:
        return symbols


def _quality_floor(rows: list[dict]) -> list[dict]:
    from scan.us_scanner import _quality_floor as floor

    return floor(rows)


def run_us_market_scan(
    *,
    scope: str = "S&P 500",
    progress_callback: Callable[[int, int], None] | None = None,
    save: bool = True,
    max_workers: int = 8,
    use_disk_cache: bool = True,
) -> UsMarketScanReport:
    """Run one US retail scan and optionally persist product payload."""
    try:
        names, scope_label = _scope_universe(scope)
    except Exception as exc:
        return UsMarketScanReport(
            DATA_UNAVAILABLE,
            error_code="US_UNIVERSE_ERROR",
            error_message=str(exc),
            scope=scope,
        )
    symbols = _liquid_first(sorted({str(s).upper() for s in names if str(s).strip()}))
    if not symbols:
        return UsMarketScanReport(
            DATA_UNAVAILABLE,
            error_code="EMPTY_US_UNIVERSE",
            error_message="US universe is empty",
            scope=scope_label,
        )

    try:
        from data.us_data import get_us_daily_batch, sp500_return_30d
        from data import us_history_store as hist
        from scan.unified_scanner import UnifiedScanner
        from product.scan_store import build_scan_payload, save_scan
    except Exception as exc:
        return UsMarketScanReport(
            FAILED,
            universe_size=len(symbols),
            error_code="US_SCAN_IMPORT",
            error_message=str(exc),
            scope=scope_label,
        )

    scanner = UnifiedScanner(max_workers=max_workers)
    try:
        scanner._nifty_ret30 = sp500_return_30d()
    except Exception:
        scanner._nifty_ret30 = 0.0

    def _get_frame(sym: str):
        if use_disk_cache:
            frame = hist.load_symbol(sym)
            if frame is not None:
                return frame
        batch = get_us_daily_batch([sym])
        frame = batch.get(sym)
        if frame is not None and use_disk_cache:
            try:
                hist.save_symbol(sym, frame)
            except Exception:
                pass
        return frame

    raw_signals: list[Any] = []
    total = len(symbols)
    done = 0
    if callable(progress_callback):
        try:
            progress_callback(0, total)
        except Exception:
            pass

    # Prefetch in batches for speed, analyze with thread pool.
    batch_size = 100
    try:
        for i in range(0, total, batch_size):
            chunk = symbols[i:i + batch_size]
            if use_disk_cache:
                missing = [s for s in chunk if hist.load_symbol(s) is None]
                if missing:
                    frames = get_us_daily_batch(missing)
                    for sym, df in frames.items():
                        hist.save_symbol(sym, df)
            else:
                get_us_daily_batch(chunk)

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = {pool.submit(scanner._analyze, sym, _get_frame(sym)): sym for sym in chunk}
                for fut in as_completed(futs):
                    done += 1
                    if callable(progress_callback) and (done % 10 == 0 or done == total):
                        try:
                            progress_callback(done, total)
                        except Exception:
                            pass
                    try:
                        result = fut.result()
                        if result and getattr(result, "signals", None):
                            raw_signals.append(result)
                    except Exception:
                        continue
    except Exception as exc:
        return UsMarketScanReport(
            FAILED,
            universe_size=len(symbols),
            scanned=done,
            error_code="US_SCAN_ERROR",
            error_message=str(exc),
            scope=scope_label,
        )

    # Apply US liquid-name floor on engine objects (need avg_vol20), then serialize.
    serialized_for_floor = []
    kept_signals = []
    for sig in raw_signals:
        row = {
            "symbol": getattr(sig, "symbol", ""),
            "price": float(getattr(sig, "price", 0) or 0),
            "avg_vol20": float(getattr(sig, "avg_vol20", 0) or 0),
        }
        serialized_for_floor.append((row, sig))
    floor_kept = {
        str(r["symbol"]).upper()
        for r in _quality_floor([r for r, _ in serialized_for_floor])
    }
    kept_signals = [sig for row, sig in serialized_for_floor if str(row["symbol"]).upper() in floor_kept]

    payload = build_scan_payload(names, kept_signals, fno_symbols=())
    # US has no NSE F&O overlay — force honest flags.
    for row in payload.get("records", []):
        row["fno_available"] = False
        row["market"] = "US"
        row["currency"] = "USD"
    records = list(payload.get("records") or [])
    momentum = [r for r in records if "MOMENTUM" in (r.get("signals") or [])]
    near = [
        r for r in records
        if "PRE_BREAKOUT" in (r.get("signals") or []) and "MOMENTUM" not in (r.get("signals") or [])
    ]
    ready = [r for r in records if r.get("status") == "Ready to trade"]
    payload["summary"] = {
        "with_any_setup": len(records),
        "momentum": len(momentum),
        "fno_momentum": 0,
        "near_breakout": len(near),
        "ready_to_trade": len(ready),
        "extended": sum(1 for r in records if r.get("chase_risk")),
        "with_measured_edge": sum(1 for r in records if r.get("edge_r") is not None),
    }
    payload["market"] = "US"
    payload["currency"] = "USD"
    payload["scope"] = scope_label
    payload["source"] = "yfinance"
    payload["places_orders"] = False
    payload["honesty"] = (
        "US scan uses Yahoo Finance daily bars + NASDAQ Trader listings. "
        "No US options desk. Paper autopilot only — never a live US broker order."
    )
    payload["scanned_at"] = datetime.now(timezone.utc).isoformat()
    payload["universe_size"] = len(symbols)

    if save:
        save_scan(payload, US_SCAN_PATH)

    n_setups = int(payload["summary"].get("with_any_setup", 0) or 0)
    status = SUCCEEDED if n_setups else NO_SETUPS
    payload["scan_status"] = status
    if save:
        save_scan(payload, US_SCAN_PATH)

    # Feed paper US autopilot (never live).
    try:
        from execution.us_autopilot import on_setups, review_cycle

        review_cycle()
        on_setups(list(records))
    except Exception as exc:
        log.debug("us_autopilot_feed_skip", error=str(exc)[:80])

    return UsMarketScanReport(
        status=status,
        payload=payload,
        universe_size=len(symbols),
        scanned=done,
        scope=scope_label,
        source="yfinance",
    )


def load_us_scan() -> dict[str, Any] | None:
    from product.scan_store import load_scan

    return load_scan(US_SCAN_PATH)
