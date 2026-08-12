"""Resumable fundamentals backfill for the full NSE EQ universe.

Fetches Screener.in deep fundamentals into ``data/fundamentals_cache.db``.
Skips symbols already fresh in cache unless ``force=True``. Records per-symbol
failures without aborting the run.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from logger import get_logger

log = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_STATE_PATH = _ROOT / "logs" / "product" / "fundamentals_backfill.json"
_DELAY_SECONDS = 1.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_state() -> dict[str, Any]:
    try:
        if _STATE_PATH.exists():
            return dict(json.loads(_STATE_PATH.read_text(encoding="utf-8")))
    except Exception:
        pass
    return {}


def _save_state(state: dict[str, Any]) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_universe(scope: str = "nse") -> list[str]:
    """Return sorted symbol list for backfill."""
    if scope == "bhav":
        from data.bhavcopy_store import store_symbols
        return sorted(store_symbols())
    if scope == "nifty500":
        from data.nse_universe import get_nifty500_universe
        return sorted(get_nifty500_universe())
    from data.nse_universe import get_nse_universe
    return sorted(get_nse_universe())


def run_fundamentals_backfill(
    *,
    symbols: Sequence[str] | None = None,
    scope: str = "nse",
    force: bool = False,
    limit: int | None = None,
    resume: bool = True,
    delay_seconds: float = _DELAY_SECONDS,
    fetcher: Callable[[str, bool], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Populate fundamentals cache for many symbols.

    Parameters
    ----------
    scope
        ``nse`` (~2000 EQ), ``nifty500``, or ``bhav`` (all bhavcopy symbols).
    force
        Re-scrape even when cache entry is fresh.
    limit
        Max symbols to process this run (after skip logic).
    resume
        Skip symbols marked succeeded in the last incomplete run when not forcing.
    """
    if fetcher is None:
        from fundamentals.fetcher import get_deep_fundamentals

        def _fetch(sym: str, force_flag: bool) -> dict[str, Any]:
            return get_deep_fundamentals(sym, force_refresh=force_flag)

        fetcher = _fetch

    universe = sorted({str(s).upper().strip() for s in (symbols or resolve_universe(scope)) if str(s).strip()})

    prior = _load_state() if resume else {}
    skip_ok = set(prior.get("succeeded", []) or []) if resume and not force else set()

    from fundamentals.cache import FundamentalsCache
    cache = FundamentalsCache()

    succeeded: list[str] = list(prior.get("succeeded", []) or []) if resume else []
    failed: dict[str, str] = dict(prior.get("failed", {}) or {}) if resume else {}
    succeeded_set = set(succeeded)

    processed = 0
    skipped_fresh = 0
    skipped_resume = 0
    started = _now()

    for symbol in universe:
        if symbol in skip_ok and symbol in succeeded_set:
            skipped_resume += 1
            continue
        if not force and cache.get(symbol) is not None:
            skipped_fresh += 1
            if symbol not in succeeded_set:
                succeeded.append(symbol)
                succeeded_set.add(symbol)
            continue
        if limit is not None and limit > 0 and processed >= limit:
            break
        try:
            fetcher(symbol, force)
            succeeded.append(symbol)
            succeeded_set.add(symbol)
            failed.pop(symbol, None)
            processed += 1
            log.info("fundamentals_backfill_ok", symbol=symbol)
        except Exception as exc:
            failed[symbol] = type(exc).__name__
            log.warning("fundamentals_backfill_fail", symbol=symbol, error=str(exc)[:120])
            processed += 1
        time.sleep(max(0.0, float(delay_seconds)))

    state = {
        "started_at": prior.get("started_at") or started,
        "updated_at": _now(),
        "scope": scope,
        "force": bool(force),
        "universe_size": len(universe),
        "processed_this_run": processed,
        "skipped_fresh": skipped_fresh,
        "skipped_resume": skipped_resume,
        "succeeded_count": len(succeeded_set),
        "failed_count": len(failed),
        "succeeded": sorted(succeeded_set),
        "failed": failed,
        "complete": (len(succeeded_set) + len(failed)) >= len(universe) and all(
            s in succeeded_set or s in failed for s in universe
        ),
        "state_path": str(_STATE_PATH),
    }
    _save_state(state)
    return state


def backfill_status() -> dict[str, Any]:
    state = _load_state()
    if not state:
        return {"available": False, "message": "No fundamentals backfill run recorded yet."}
    return {"available": True, **state}
