"""Point-in-time correlation evidence from the official NSE bhavcopy store.

No network calls are made here. The loader uses only already-present official
bhavcopy data, cuts every series at ``as_of`` when supplied, and returns only
pairwise correlations with enough overlapping sessions. Missing stays missing.

Important: without an explicit or row-derived time anchor, correlations may be
computed for diagnostics but are NOT labelled point-in-time and must not be used
as a production hard gate by Portfolio Selection Authority.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

DEFAULT_WINDOW = 60
DEFAULT_MIN_PERIODS = 30


def _key(a: str, b: str) -> str:
    left, right = sorted((str(a).upper(), str(b).upper()))
    return f"{left}|{right}"


def _as_of_from_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    for row in rows:
        value = str(row.get("decision_as_of") or row.get("as_of") or row.get("scan_scanned_at") or "").strip()
        if value:
            return value[:10]
    return ""


def build_pit_correlations(
    symbols: Sequence[str],
    *,
    as_of: str = "",
    window: int = DEFAULT_WINDOW,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> dict[str, Any]:
    """Build pairwise close-return correlations from local official NSE data."""
    wanted = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
    anchor = as_of[:10] if as_of else ""
    out: dict[str, Any] = {
        "source": "official_nse_bhavcopy_local",
        "point_in_time": bool(anchor),
        "as_of": anchor,
        "window": int(window),
        "min_periods": int(min_periods),
        "symbols_requested": wanted,
        "symbols_available": [],
        "correlations": {},
        "pair_samples": {},
        "missing_pairs": [],
        "coverage": 0.0,
        "network_used": False,
        "production_usable": bool(anchor),
        "warning": "" if anchor else "NO_TIME_ANCHOR_DIAGNOSTIC_ONLY",
    }
    if len(wanted) < 2:
        return out
    try:
        import pandas as pd
        from data import bhavcopy_store as store

        if not store.is_ready():
            # Local-only bootstrap. This parses user/worker-supplied bhavcopy CSVs;
            # it deliberately does not call build_store(), which could download.
            try:
                store.build_from_local()
            except Exception:
                pass
        series: dict[str, Any] = {}
        cutoff = pd.Timestamp(anchor) if anchor else None
        for symbol in wanted:
            frame = store.get_ohlcv(symbol)
            if frame is None or len(frame) < int(min_periods) + 1 or "close" not in frame.columns:
                continue
            df = frame.copy()
            if cutoff is not None:
                df = df[df.index <= cutoff]
            if len(df) < int(min_periods) + 1:
                continue
            returns = df["close"].astype(float).pct_change().dropna().tail(int(window))
            if len(returns) >= int(min_periods):
                series[symbol] = returns
        out["symbols_available"] = sorted(series)
        total_pairs = len(wanted) * (len(wanted) - 1) // 2
        for a, b in combinations(wanted, 2):
            key = _key(a, b)
            if a not in series or b not in series:
                out["missing_pairs"].append(key)
                continue
            joined = pd.concat([series[a], series[b]], axis=1, join="inner").dropna()
            n = len(joined)
            if n < int(min_periods):
                out["missing_pairs"].append(key)
                continue
            corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            if corr != corr:
                out["missing_pairs"].append(key)
                continue
            out["correlations"][key] = round(corr, 6)
            out["pair_samples"][key] = n
        if total_pairs:
            out["coverage"] = round(len(out["correlations"]) / total_pairs, 4)
        return out
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}:{str(exc)[:160]}"
        out["production_usable"] = False
        return out


def correlations_for_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    held_symbols: Sequence[str] | None = None,
    as_of: str = "",
    window: int = DEFAULT_WINDOW,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> dict[str, Any]:
    symbols = [str(r.get("symbol") or "").upper() for r in rows]
    symbols.extend(str(s).upper() for s in (held_symbols or []))
    return build_pit_correlations(
        symbols,
        as_of=as_of or _as_of_from_rows(rows),
        window=window,
        min_periods=min_periods,
    )
