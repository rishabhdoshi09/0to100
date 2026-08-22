"""Canonical, Streamlit-free whole-market scan service.

The autonomy supervisor and the retail Momentum page call this exact service.  UI code may pass a
progress callback, but this module never imports Streamlit or any ``ui.*`` module.  A provider failure
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
            "universe_size": self.universe_size,
            "scanned": self.scanned,
            "exclusions": list(self.exclusions),
            "source_snapshot_id": self.source_snapshot_id,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


def _default_universe() -> Mapping[str, str]:
    from data.nse_universe import get_nse_universe_with_names
    return get_nse_universe_with_names()


def _default_prefetch(symbols, *, progress=None):
    from scan.bulk_fetcher import prefetch
    return prefetch(symbols, progress=progress)


def _default_scanner():
    from scan.unified_scanner import UnifiedScanner
    return UnifiedScanner()


def _default_fno_symbols() -> set[str]:
    from data.fno_universe import current_fno_universe
    return set(current_fno_universe().symbols)


def _active_snapshot_id() -> str:
    try:
        from research.intelligence.data.snapshot_store import SnapshotStore
        return str(SnapshotStore().get_active_snapshot() or "")
    except Exception:
        return ""


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

    Dependency injection keeps this function network-free in tests.  Results are always ordered by
    the existing canonical payload builder, and F&O availability is overlaid only after the cash
    universe has been evaluated.
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
    symbols = sorted({str(s).strip().upper() for s in names if str(s).strip()})
    if not symbols:
        return MarketScanReport(DATA_UNAVAILABLE, universe_size=0, error_code="EMPTY_UNIVERSE",
                                error_message="approved NSE universe is empty",
                                source_snapshot_id=snapshot_id or "")

    try:
        prefetch_fn(symbols, progress=progress_callback)
        results = list(scanner.scan(symbols) or [])
    except Exception as exc:
        return MarketScanReport(FAILED, universe_size=len(symbols), error_code="SCAN_ERROR",
                                error_message=str(exc), source_snapshot_id=snapshot_id or "")

    try:
        fno_symbols = set(fno_provider() or ())
    except Exception:
        # F&O is an overlay, not the cash-universe source of truth.  Its failure is represented by
        # an empty overlay rather than invalidating a successfully completed cash scan.
        fno_symbols = set()

    from product.scan_store import build_scan_payload, save_scan
    payload = build_scan_payload(names, results, fno_symbols)
    sid = snapshot_id if snapshot_id is not None else _active_snapshot_id()
    payload["source_snapshot_id"] = sid
    payload["scan_status"] = SUCCEEDED
    if save:
        save_scan(payload)
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
    return MarketScanReport(status=status, payload=payload, universe_size=len(symbols),
                            scanned=int(payload.get("universe_size", len(symbols)) or len(symbols)),
                            source_snapshot_id=sid)
