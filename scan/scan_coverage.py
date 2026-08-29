"""Truthful stock-by-stock coverage ledger for the canonical market scan.

The scanner intentionally returns only rows that have at least one setup. That is
useful for ranking but terrible for operator trust: a symbol with missing history,
a policy exclusion, a clean "no setup" result, or an analysis exception used to
all disappear from the visible output.

This module instruments the existing UnifiedScanner *without creating a second
scanner*. It records what happened to each requested symbol while the canonical
``_analyze`` method still does the work, then persists a compact audit ledger.
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


AUDIT_PATH = Path(__file__).resolve().parents[1] / "logs" / "product" / "latest_scan_audit.json"

QUALIFIED = "QUALIFIED"
NO_SETUP = "NO_SETUP"
NO_OHLCV = "NO_OHLCV"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
PRICE_FILTER = "PRICE_BELOW_20"
LIQUIDITY_FILTER = "LOW_LIQUIDITY"
FALLING_KNIFE_FILTER = "FALLING_KNIFE"
ANALYSIS_ERROR = "ANALYSIS_ERROR"
ANALYSIS_SKIPPED = "ANALYSIS_SKIPPED"
NOT_OBSERVED = "NOT_OBSERVED"

_POLICY_EXCLUSIONS = {PRICE_FILTER, LIQUIDITY_FILTER, FALLING_KNIFE_FILTER}
_DATA_GAPS = {NO_OHLCV, INSUFFICIENT_HISTORY}


def _symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _precheck_reason(df) -> tuple[str, str]:
    """Mirror only UnifiedScanner's explicit early hard gates.

    The actual decision remains inside UnifiedScanner. These checks exist only to
    explain an observed ``None`` result; they never promote or reject a stock.
    """
    if df is None:
        return NO_OHLCV, "No OHLCV frame was available to the scanner."
    try:
        n = len(df)
    except Exception:
        n = 0
    if n < 60:
        return INSUFFICIENT_HISTORY, f"Only {n} daily bars available; scanner requires at least 60."
    try:
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float) if "high" in df.columns else close
        vol = df["volume"].values.astype(float) if "volume" in df.columns else None
        price = float(close[-1])
        if price < 20:
            return PRICE_FILTER, f"Price ₹{price:.2f} is below the scanner's ₹20 policy floor."
        if vol is not None and len(vol) >= 20:
            turnover = float(np.nanmean(vol[-20:])) * price
            if turnover < 1e7:
                return LIQUIDITY_FILTER, f"20-session average traded value is below ₹1 crore/day."
        from scan.unified_scanner import is_beaten_down_arr
        if is_beaten_down_arr(high[-250:], price):
            return FALLING_KNIFE_FILTER, "Price is more than the configured maximum drop from its 52-week high."
    except Exception:
        return "", ""
    return "", ""


class ScanCoverageProbe:
    def __init__(self, symbols: Iterable[str], *, instrumented: bool):
        self.requested = [_symbol(s) for s in symbols if _symbol(s)]
        self.instrumented = bool(instrumented)
        self._lock = threading.Lock()
        self._rows: dict[str, dict[str, Any]] = {}

    def _record(self, symbol: str, status: str, reason: str = "", *, error: str = "") -> None:
        sym = _symbol(symbol)
        if not sym:
            return
        row = {"symbol": sym, "status": status, "reason": reason}
        if error:
            row["error"] = error
        with self._lock:
            self._rows[sym] = row

    def run_analyze(self, original, symbol: str, df):
        pre_status, pre_reason = _precheck_reason(df)
        try:
            result = original(symbol, df)
        except Exception as exc:
            self._record(
                symbol,
                ANALYSIS_ERROR,
                "Per-stock analysis raised an exception.",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        if result is None:
            if pre_status:
                self._record(symbol, pre_status, pre_reason)
            else:
                self._record(symbol, NO_SETUP, "Fully analyzed; no configured setup qualified on this scan.")
            return result
        signals = list(getattr(result, "signals", ()) or ())
        if signals:
            self._record(symbol, QUALIFIED, f"Qualified with {len(signals)} setup signal(s).")
        else:
            self._record(symbol, NO_SETUP, "Fully analyzed; no configured setup qualified on this scan.")
        return result

    def finalize(self, results: Iterable[Any] = (), *, cached: Iterable[str] = (), walked_total: int = 0) -> dict[str, Any]:
        requested = list(dict.fromkeys(self.requested))
        cached_set = {_symbol(s) for s in cached if _symbol(s)}
        qualified = {_symbol(getattr(row, "symbol", None) if not isinstance(row, dict) else row.get("symbol")) for row in results}
        qualified.discard("")

        with self._lock:
            rows = dict(self._rows)

        for sym in requested:
            if sym in rows:
                continue
            if sym not in cached_set:
                rows[sym] = {
                    "symbol": sym,
                    "status": NO_OHLCV,
                    "reason": "Requested universe symbol had no cached OHLCV after prefetch.",
                }
            elif sym in qualified:
                rows[sym] = {
                    "symbol": sym,
                    "status": QUALIFIED,
                    "reason": "Qualified by the scanner.",
                }
            elif not self.instrumented:
                rows[sym] = {
                    "symbol": sym,
                    "status": NOT_OBSERVED,
                    "reason": "Scanner implementation did not expose per-symbol analysis instrumentation.",
                }
            else:
                rows[sym] = {
                    "symbol": sym,
                    "status": ANALYSIS_SKIPPED,
                    "reason": "Symbol had cached data but did not produce an observed analysis outcome.",
                }

        ledger = [rows[s] for s in sorted(rows)]
        counts = Counter(str(row.get("status") or "UNKNOWN") for row in ledger)
        requested_n = len(requested)
        technical = counts[QUALIFIED] + counts[NO_SETUP]
        policy = sum(counts[key] for key in _POLICY_EXCLUSIONS)
        data_gaps = sum(counts[key] for key in _DATA_GAPS)
        errors = counts[ANALYSIS_ERROR] + counts[ANALYSIS_SKIPPED] + counts[NOT_OBSERVED]
        checked = technical + policy
        accounted = requested_n - counts[NOT_OBSERVED]
        coverage_pct = round((checked / requested_n) * 100.0, 1) if requested_n else 0.0
        history_pct = round(((requested_n - counts[NO_OHLCV]) / requested_n) * 100.0, 1) if requested_n else 0.0
        accounted_pct = round((accounted / requested_n) * 100.0, 1) if requested_n else 0.0
        state = "FULL" if data_gaps == 0 and errors == 0 else "DEGRADED"
        summary = {
            "state": state,
            "requested": requested_n,
            "checked": checked,
            "technical_evaluated": technical,
            "qualified": counts[QUALIFIED],
            "no_setup": counts[NO_SETUP],
            "policy_excluded": policy,
            "data_unavailable": data_gaps,
            "analysis_errors": counts[ANALYSIS_ERROR],
            "analysis_skipped": counts[ANALYSIS_SKIPPED],
            "not_observed": counts[NOT_OBSERVED],
            "history_available": requested_n - counts[NO_OHLCV],
            "coverage_pct": coverage_pct,
            "history_coverage_pct": history_pct,
            "accounted_pct": accounted_pct,
            "scanner_instrumented": self.instrumented,
            "walked_total_reported": int(walked_total or 0),
            "reason_counts": dict(sorted(counts.items())),
        }
        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "ledger": ledger,
        }


@contextmanager
def observe_scanner(scanner, symbols: Iterable[str]):
    """Temporarily wrap the canonical scanner's existing ``_analyze`` method."""
    original = getattr(scanner, "_analyze", None)
    instrumented = callable(original)
    probe = ScanCoverageProbe(symbols, instrumented=instrumented)
    if not instrumented:
        yield probe
        return

    def wrapped(symbol, df):
        return probe.run_analyze(original, symbol, df)

    scanner._analyze = wrapped
    try:
        yield probe
    finally:
        scanner._analyze = original


def save_audit(audit: dict[str, Any], path: str | Path = AUDIT_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_audit(path: str | Path = AUDIT_PATH) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {"schema_version": 1, "summary": {}, "ledger": []}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("scan audit is not an object")
        payload.setdefault("summary", {})
        payload.setdefault("ledger", [])
        return payload
    except Exception:
        return {"schema_version": 1, "summary": {}, "ledger": []}


def lookup_symbol(symbol: str, audit: dict[str, Any] | None = None) -> dict[str, Any] | None:
    clean = _symbol(symbol)
    payload = audit or load_audit()
    for row in payload.get("ledger") or []:
        if _symbol(row.get("symbol")) == clean:
            return dict(row)
    return None
