"""Multi-source fundamentals resolver that yields status at every step.

Order (official / primary reputed → fallback reputed → local → user):
  1. local fresh cache
  2. Screener.in (primary fundamentals source)
  3. Yahoo Finance via yfinance (reputed public fallback)
  4. local stale cache
  5. user structured evidence uploads

Never invents numbers. Exhausted resolution returns data=None with a full
step trail and next-action hints — the UI stays usable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import time
from typing import Any, Callable, Iterator, Mapping

from logger import get_logger

log = get_logger(__name__)

ProviderFn = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class ResolveStep:
    step: int
    source: str
    status: str  # TRYING | OK | EMPTY | ERROR | SKIPPED | PARTIAL | EXHAUSTED
    message: str
    elapsed_ms: int = 0
    sections: dict[str, Any] = field(default_factory=dict)
    reputed: bool = False
    official: bool = False
    coverage: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def section_counts(data: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(data or {})
    return {
        "about": bool(str(payload.get("about") or "").strip()),
        "key_ratios": len(payload.get("key_ratios") or []),
        "quarterly_results": len(payload.get("quarterly_results") or []),
        "profit_loss": len(payload.get("profit_loss") or []),
        "balance_sheet": len(payload.get("balance_sheet") or []),
        "cash_flow": len(payload.get("cash_flow") or []),
        "shareholding": len(payload.get("shareholding") or []),
        "peer_comparison": len(payload.get("peer_comparison") or []),
    }


def coverage_score(data: Mapping[str, Any] | None) -> int:
    """0–100 usefulness score. Enough for evidence when ≥ 35."""
    counts = section_counts(data)
    weights = {
        "about": 15,
        "key_ratios": 20,
        "profit_loss": 20,
        "quarterly_results": 15,
        "balance_sheet": 10,
        "cash_flow": 10,
        "shareholding": 10,
    }
    score = 0
    for key, weight in weights.items():
        value = counts.get(key)
        if isinstance(value, bool):
            score += weight if value else 0
        else:
            score += weight if int(value or 0) > 0 else 0
    return min(100, score)


def _enough(data: Mapping[str, Any] | None, *, minimum: int = 35) -> bool:
    return coverage_score(data) >= minimum


def _frame_to_rows(frame: Any, *, limit_cols: int = 8) -> list[dict[str, Any]]:
    """Convert a yfinance-style DataFrame into Screener-like row dicts."""
    try:
        import pandas as pd
    except Exception:
        return []
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    cols = list(frame.columns)[:limit_cols]
    rows: list[dict[str, Any]] = []
    for index, series in frame.iterrows():
        row: dict[str, Any] = {"": str(index)}
        for col in cols:
            label = str(getattr(col, "date", lambda: col)()) if hasattr(col, "date") else str(col)[:12]
            try:
                value = series[col]
                if value is None or (isinstance(value, float) and value != value):
                    continue
                # Yahoo reports absolute currency; Screener uses ₹ Cr for India.
                if isinstance(value, (int, float)) and abs(float(value)) >= 1_000_000:
                    row[label] = round(float(value) / 1e7, 2)
                else:
                    row[label] = value
            except Exception:
                continue
        if len(row) > 1:
            rows.append(row)
    return rows


def fetch_yahoo_finance(symbol: str) -> dict[str, Any]:
    """Reputed public fallback. Partial coverage is OK — never pad missing fields."""
    import yfinance as yf

    ticker = yf.Ticker(f"{symbol.upper().strip()}.NS")
    info = {}
    try:
        info = dict(ticker.info or {})
    except Exception:
        info = {}
    about = str(info.get("longBusinessSummary") or "").strip()
    ratios: list[dict[str, str]] = []
    mapping = [
        ("Market Cap (Cr)", info.get("marketCap"), 1e7),
        ("Current Price", info.get("currentPrice") or info.get("regularMarketPrice"), 1),
        ("P/E", info.get("trailingPE") or info.get("forwardPE"), 1),
        ("P/B", info.get("priceToBook"), 1),
        ("ROE %", (info.get("returnOnEquity") or 0) * 100 if info.get("returnOnEquity") is not None else None, 1),
        ("Debt to Equity", info.get("debtToEquity"), 1),
        ("Dividend Yield %", (info.get("dividendYield") or 0) * 100 if info.get("dividendYield") is not None else None, 1),
        ("EPS", info.get("trailingEps"), 1),
        ("Revenue Growth %", (info.get("revenueGrowth") or 0) * 100 if info.get("revenueGrowth") is not None else None, 1),
    ]
    for name, raw, divisor in mapping:
        if raw in (None, "", 0, 0.0):
            continue
        try:
            value = float(raw) / float(divisor)
            ratios.append({"name": name, "value": f"{value:.2f}"})
        except Exception:
            continue

    profit_loss = _frame_to_rows(getattr(ticker, "financials", None))
    quarterly = _frame_to_rows(getattr(ticker, "quarterly_financials", None))
    balance = _frame_to_rows(getattr(ticker, "balance_sheet", None))
    cash = _frame_to_rows(getattr(ticker, "cashflow", None))

    payload = {
        "about": about[:2000],
        "key_ratios": ratios,
        "profit_loss": profit_loss,
        "quarterly_results": quarterly,
        "balance_sheet": balance,
        "cash_flow": cash,
        "shareholding": [],
        "peer_comparison": [],
        "_source": "yahoo_finance",
        "_source_url": f"https://finance.yahoo.com/quote/{symbol.upper().strip()}.NS",
        "_fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    if not _enough(payload, minimum=20):
        raise RuntimeError(f"Yahoo Finance returned insufficient fundamentals for {symbol}")
    return payload


def fetch_screener(symbol: str) -> dict[str, Any]:
    from fundamentals.period_freshness import pack_needs_filings_retry, prefer_fresher_pack
    from fundamentals.screener_deep import ScreenerDeepFetcher

    fetcher = ScreenerDeepFetcher()
    data = dict(fetcher.fetch_all(symbol) or {})
    if pack_needs_filings_retry(data):
        try:
            standalone = dict(fetcher.fetch_all(symbol, consolidated=False) or {})
            data = prefer_fresher_pack(data, standalone)
        except TypeError:
            pass
        except Exception as exc:
            log.warning("screener_standalone_retry_failed", symbol=symbol, error=str(exc))
        data["_filings_refresh_attempted"] = True
    data.setdefault("_source", "screener_in")
    data.setdefault(
        "_source_url",
        data.get("url") or f"https://www.screener.in/company/{symbol}/consolidated/",
    )
    data["_fetched_at"] = datetime.now(timezone.utc).isoformat()
    if not _enough(data, minimum=20):
        raise RuntimeError(f"Screener.in returned insufficient fundamentals for {symbol}")
    return data


def fetch_user_uploads(symbol: str) -> dict[str, Any]:
    """Build a partial deep-fundamentals dict from validated Research Data uploads."""
    from reporting.evidence_intake import structured_rows

    profile = structured_rows(symbol, "business_profile")
    financials = structured_rows(symbol, "financial_history")
    shareholding = structured_rows(symbol, "shareholding_history")
    about = ""
    if profile:
        about = str(profile[0].get("business_summary") or "").strip()
    profit_loss: list[dict[str, Any]] = []
    for row in financials:
        if str(row.get("period_type") or "").lower() not in {"annual", "year", "yearly", ""}:
            continue
        period = str(row.get("period_end") or "")
        if not period:
            continue
        for label, field in (("Sales", "revenue_cr"), ("PAT", "pat_cr"), ("EBITDA", "ebitda_cr")):
            value = row.get(field)
            if value in (None, ""):
                continue
            existing = next((item for item in profit_loss if item.get("") == label), None)
            if existing is None:
                existing = {"": label}
                profit_loss.append(existing)
            existing[period] = value
    payload = {
        "about": about,
        "key_ratios": [],
        "profit_loss": profit_loss,
        "quarterly_results": [],
        "balance_sheet": [],
        "cash_flow": [],
        "shareholding": shareholding,
        "peer_comparison": [],
        "_source": "user_structured_upload",
        "_source_url": "local://research_evidence",
        "_fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    if not _enough(payload, minimum=15):
        raise RuntimeError(f"No usable structured uploads for {symbol}")
    return payload


def next_actions(symbol: str) -> list[dict[str, str]]:
    sym = symbol.upper().strip()
    return [
        {"label": "Open Screener.in", "url": f"https://www.screener.in/company/{sym}/consolidated/", "kind": "reputed"},
        {"label": "Open Yahoo Finance", "url": f"https://finance.yahoo.com/quote/{sym}.NS", "kind": "reputed"},
        {"label": "NSE financial results", "url": "https://www.nseindia.com/companies-listing/corporate-filings-financial-results", "kind": "official"},
        {"label": "BSE financial results", "url": "https://www.bseindia.com/corporates/comp_results.aspx", "kind": "official"},
        {"label": "NSE shareholding", "url": f"https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern?symbol={sym}", "kind": "official"},
        {"label": "Download worked-example CSV", "url": "/evidence/examples/financial_history.csv", "kind": "local_fixture"},
        {"label": "Auto-install worked example", "url": f"/evidence/{sym}/actions/install-worked-example", "kind": "local_fixture"},
    ]


def iter_resolve(
    symbol: str,
    *,
    force_refresh: bool = False,
    write_cache: bool = True,
    providers: Mapping[str, ProviderFn] | None = None,
) -> Iterator[ResolveStep]:
    """Yield one status event per attempt. Stops early when coverage is enough."""
    symbol = symbol.upper().strip()
    if not symbol:
        yield ResolveStep(1, "input", "ERROR", "empty symbol", official=True)
        return

    from fundamentals.cache import FundamentalsCache

    cache = FundamentalsCache()
    provider_map: dict[str, ProviderFn] = {
        "screener_in": fetch_screener,
        "yahoo_finance": fetch_yahoo_finance,
        "user_uploads": fetch_user_uploads,
    }
    if providers:
        provider_map.update(dict(providers))

    step_no = 0
    best: dict[str, Any] | None = None
    best_score = 0

    def _emit(source: str, status: str, message: str, *, data: Mapping[str, Any] | None = None,
              elapsed_ms: int = 0, reputed: bool = False, official: bool = False) -> ResolveStep:
        nonlocal step_no, best, best_score
        step_no += 1
        counts = section_counts(data)
        score = coverage_score(data)
        if data is not None and score > best_score:
            best = dict(data)
            best_score = score
        return ResolveStep(
            step=step_no, source=source, status=status, message=message,
            elapsed_ms=elapsed_ms, sections=counts, reputed=reputed,
            official=official, coverage=score,
        )

    # 1) Fresh local cache
    if not force_refresh:
        yield _emit("local_cache_fresh", "TRYING", "Checking local fundamentals cache")
        cached = cache.get(symbol)
        if cached is not None and _enough(cached):
            from fundamentals.period_freshness import pack_needs_filings_retry
            if pack_needs_filings_retry(cached):
                yield _emit(
                    "local_cache_fresh", "PARTIAL",
                    "Today's cache exists but the latest filings column is behind "
                    "the current reporting season — refetching",
                    data=cached, reputed=True,
                )
            else:
                yield _emit(
                    "local_cache_fresh", "OK",
                    "Fresh cache hit — no network fetch needed",
                    data=cached, reputed=True,
                )
                return
        elif cached is not None:
            yield _emit(
                "local_cache_fresh", "PARTIAL",
                "Cache present but coverage too thin — continuing",
                data=cached,
            )
        else:
            yield _emit("local_cache_fresh", "EMPTY", "No fresh cache entry")

    # 2) Screener.in
    yield _emit("screener_in", "TRYING", "Fetching Screener.in (primary reputed source)", reputed=True)
    started = time.perf_counter()
    try:
        data = provider_map["screener_in"](symbol)
        elapsed = int((time.perf_counter() - started) * 1000)
        if write_cache:
            cache.set(symbol, data)
        yield _emit(
            "screener_in", "OK",
            "Screener.in returned usable fundamentals",
            data=data, elapsed_ms=elapsed, reputed=True,
        )
        if _enough(data):
            return
        yield _emit(
            "screener_in", "PARTIAL",
            "Screener coverage thin — trying Yahoo Finance fallback",
            data=data, elapsed_ms=elapsed, reputed=True,
        )
    except Exception as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        log.warning("resolver_screener_failed", symbol=symbol, error=type(exc).__name__)
        yield _emit(
            "screener_in", "ERROR",
            f"Screener.in failed: {exc}",
            elapsed_ms=elapsed, reputed=True,
        )

    # 3) Yahoo Finance (reputed fallback)
    yield _emit("yahoo_finance", "TRYING", "Fetching Yahoo Finance fallback", reputed=True)
    started = time.perf_counter()
    try:
        data = provider_map["yahoo_finance"](symbol)
        elapsed = int((time.perf_counter() - started) * 1000)
        if write_cache and _enough(data, minimum=35):
            cache.set(symbol, data)
        status = "OK" if _enough(data) else "PARTIAL"
        yield _emit(
            "yahoo_finance", status,
            "Yahoo Finance returned fundamentals" if status == "OK"
            else "Yahoo Finance partial — keeping and continuing",
            data=data, elapsed_ms=elapsed, reputed=True,
        )
        if _enough(data):
            return
    except Exception as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        log.warning("resolver_yahoo_failed", symbol=symbol, error=type(exc).__name__)
        yield _emit(
            "yahoo_finance", "ERROR",
            f"Yahoo Finance failed: {exc}",
            elapsed_ms=elapsed, reputed=True,
        )

    # 4) Stale local cache
    yield _emit("local_cache_stale", "TRYING", "Checking stale local cache")
    stale = cache.get_any(symbol)
    if stale is not None and _enough(stale, minimum=20):
        yield _emit(
            "local_cache_stale", "OK",
            "Serving stale cache after remote failures",
            data=stale,
        )
        return
    if stale is not None:
        yield _emit("local_cache_stale", "PARTIAL", "Stale cache too thin", data=stale)
    else:
        yield _emit("local_cache_stale", "EMPTY", "No stale cache available")

    # 5) User structured uploads
    yield _emit("user_uploads", "TRYING", "Checking Research Data structured uploads")
    started = time.perf_counter()
    try:
        data = provider_map["user_uploads"](symbol)
        elapsed = int((time.perf_counter() - started) * 1000)
        status = "OK" if _enough(data, minimum=20) else "PARTIAL"
        yield _emit(
            "user_uploads", status,
            "Structured uploads available for analysis",
            data=data, elapsed_ms=elapsed,
        )
        if _enough(data, minimum=20):
            return
    except Exception as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        yield _emit(
            "user_uploads", "EMPTY",
            f"No usable uploads ({exc})",
            elapsed_ms=elapsed,
        )

    # Exhausted — still yield best partial if any
    if best is not None and best_score > 0:
        yield _emit(
            "resolver", "PARTIAL",
            f"All primary sources failed or thin; returning best partial coverage={best_score}",
            data=best,
        )
        return

    yield _emit(
        "resolver", "EXHAUSTED",
        "No fundamentals available from cache, Screener.in, Yahoo Finance, or uploads. "
        "Use official NSE/BSE links or install a worked example.",
    )


def resolve(
    symbol: str,
    *,
    force_refresh: bool = False,
    write_cache: bool = True,
    providers: Mapping[str, ProviderFn] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Run the yielding resolver once. Returns (best_data_or_none, step_trail)."""
    steps: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score = 0
    stash: dict[str, Any] = {"payload": None}

    def _capture(fn: ProviderFn) -> ProviderFn:
        def inner(sym: str) -> dict[str, Any]:
            payload = fn(sym)
            stash["payload"] = payload
            return payload
        return inner

    base: dict[str, ProviderFn] = {
        "screener_in": fetch_screener,
        "yahoo_finance": fetch_yahoo_finance,
        "user_uploads": fetch_user_uploads,
    }
    if providers:
        base.update(dict(providers))
    wrapped = {name: _capture(fn) for name, fn in base.items()}

    for event in iter_resolve(
        symbol,
        force_refresh=force_refresh,
        write_cache=write_cache,
        providers=wrapped,
    ):
        steps.append(event.as_dict())
        payload = stash.get("payload")
        if payload is not None:
            score = coverage_score(payload)
            if score >= best_score:
                best = dict(payload)
                best_score = score
        if event.source in {"local_cache_fresh", "local_cache_stale"} and event.status in {"OK", "PARTIAL"}:
            from fundamentals.cache import FundamentalsCache

            cached = FundamentalsCache().get_any(symbol.upper().strip())
            if cached is not None:
                score = coverage_score(cached)
                if score >= best_score:
                    best = dict(cached)
                    best_score = score
        if event.status == "OK" and event.coverage >= 35 and best is not None:
            return best, steps

    if best is not None and best_score > 0:
        return best, steps
    return None, steps
