"""Install the production whole-market scanner's non-blocking pre-loop contract.

The broad NSE scan must be compute-only once official OHLCV is warm. In particular,
a missing/stale index feed must never block thousands of stock evaluations before
the first progress event. Relative-strength and regime context therefore consume
cached official index/regime evidence only; absence means neutral/no demotion, never
a synthetic market view.
"""
from __future__ import annotations

from typing import Any


_INSTALLED = False


def _cached_nifty_return_30d() -> float:
    """Return a cached official Nifty 30-session return without network access."""
    try:
        from data.index_store import recent_index_closes

        closes = list(recent_index_closes("^NSEI", n=31) or [])
        if len(closes) < 31:
            return 0.0
        first = float(closes[-31])
        last = float(closes[-1])
        if first <= 0 or last <= 0:
            return 0.0
        return float((last / first - 1.0) * 100.0)
    except Exception:
        return 0.0


def _scan_nonblocking(self, symbols: list[str], progress=None, *, prefetch: bool = True):
    """UnifiedScanner.scan with cached-only benchmark/regime setup.

    This mirrors the canonical scanner loop but deliberately avoids
    ``get_index_ohlcv``/``compute_regime`` because those may perform bounded network
    builds. The first progress callback is emitted as soon as the cached symbol set
    is known, before optional contextual calibration is read.
    """
    from scan.bulk_fetcher import cached_symbols, get_cached, prefetch as do_prefetch
    from scan import unified_scanner as module

    if prefetch:
        do_prefetch(symbols)
    available_set = set(cached_symbols())
    available = [s for s in symbols if s in available_set]
    if not available:
        try:
            from scan.bulk_fetcher import adopt_ready_store

            adopt_ready_store(overlay_live=False)
        except Exception:
            pass
        available_set = set(cached_symbols())
        available = [s for s in symbols if s in available_set]
    if not available:
        do_prefetch(symbols)
        available_set = set(cached_symbols())
        available = [s for s in symbols if s in available_set]

    total = len(available)
    if progress and total:
        try:
            progress(0, total)
        except Exception:
            pass

    # Context is cached-only. Missing evidence is neutral/no-demotion rather than
    # a fabricated Nifty return or a network wait in the scan lane.
    self._nifty_ret30 = _cached_nifty_return_30d()
    try:
        from core.regime_engine import peek_cached_regime
        from scan.live_edge import regime_calibration

        cached = peek_cached_regime()
        self._regime = str(getattr(cached, "market_regime", "") or "") if cached is not None else ""
        self._regime_calib = regime_calibration(self._regime) if self._regime else {}
        if self._regime_calib:
            module.log.info(
                "scanner_regime_calibrated_cached",
                regime=self._regime,
                signals=len(self._regime_calib),
            )
    except Exception as exc:
        module.log.debug("regime_calib_cached_skip", error=str(exc))
        self._regime, self._regime_calib = "", {}

    module.log.info("unified_scan_start", requested=len(symbols), with_data=total)
    results: list[Any] = []
    done = 0
    with module.ThreadPoolExecutor(max_workers=self._max_workers) as pool:
        futures = {
            pool.submit(self._analyze, sym, get_cached(sym)): sym
            for sym in available
        }
        for fut in module.as_completed(futures):
            done += 1
            try:
                result = fut.result()
                if result and result.signals:
                    results.append(result)
            except Exception as exc:
                module.log.debug(
                    "unified_analyze_failed", symbol=futures[fut], error=str(exc)
                )
            if progress:
                try:
                    progress(done, total)
                except Exception:
                    pass

    results.sort(key=lambda row: row.score, reverse=True)
    module.log.info("unified_scan_done", scanned=total, with_signals=len(results))
    return results


def install() -> None:
    """Patch UnifiedScanner exactly once for production/import-time consistency."""
    global _INSTALLED
    if _INSTALLED:
        return
    from scan.unified_scanner import UnifiedScanner

    if not getattr(UnifiedScanner.scan, "__qt_nonblocking_scan__", False):
        setattr(_scan_nonblocking, "__qt_nonblocking_scan__", True)
        UnifiedScanner.scan = _scan_nonblocking
    _INSTALLED = True
