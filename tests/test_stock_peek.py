"""Snapshot popup numbers — on-file + timed tape, never a scrape gate.

Fixtures are generic scan names (ALPHA / BETA). Nothing is hard-wired to a
listed ticker.
"""
from __future__ import annotations

import time

import pandas as pd

from product.stock_peek import build_stock_peek, fetch_missing_fundamentals_for_peek, _upside, _pack_is_thin


def _frame(periods: int = 280) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="B")
    close = pd.Series([100 + i * 0.4 for i in range(periods)], index=index)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [120000] * periods,
        },
        index=index,
    )


def _scan_row(symbol: str, *, price: float, entry: float, stop: float, target: float, **extra):
    row = {
        "symbol": symbol,
        "company": f"{symbol} Co",
        "price": price,
        "entry": entry,
        "stop": stop,
        "target": target,
        "rsi": 61.0,
        "volume_ratio": 1.8,
        "change_pct": 1.1,
        "sector": "Chemicals",
    }
    row.update(extra)
    return row


def test_upside_from_buy_and_target():
    assert _upside(100.0, 130.0) == 30.0
    assert _upside(0, 100) is None
    assert _upside(100, None) is None


def test_peek_is_generic_for_any_scan_symbol():
    scan = {
        "records": [
            _scan_row("ALPHA", price=110, entry=100, stop=90, target=130, rsi=55.0),
            _scan_row("BETA", price=210, entry=200, stop=180, target=240, rsi=62.0, change_pct=-0.4),
        ]
    }
    lt = {
        "records": [
            {
                "symbol": "ALPHA",
                "classification": "GARP_CANDIDATE",
                "fundamental_coverage": 0.7,
                "fundamentals": {"pe": 18.4, "roe": 22.1, "roce": 19.0},
                "sector": "Chemicals",
            },
            {
                "symbol": "BETA",
                "classification": "QUALITY_COMPOUNDER",
                "fundamental_coverage": 0.8,
                "fundamentals": {"pe": 22.0, "roe": 16.0},
                "sector": "Chemicals",
            },
        ]
    }
    alpha = build_stock_peek(
        "ALPHA",
        scan_payload=scan,
        long_term_payload=lt,
        raw_fundamentals={"data": {}, "cache_status": "TODAY"},
        frame=_frame(),
        quote={"price": 111.5, "chg_pct": 1.25, "source": "nse"},
        load_history=False,
        load_live=False,
    )
    beta = build_stock_peek(
        "BETA",
        scan_payload=scan,
        long_term_payload=lt,
        raw_fundamentals={"data": {}, "cache_status": "TODAY"},
        frame=_frame(),
        quote={"price": 208.0, "chg_pct": -0.4, "source": "nse"},
        load_history=False,
        load_live=False,
    )
    assert alpha["symbol"] == "ALPHA" and beta["symbol"] == "BETA"
    assert alpha["upside_from_buy_pct"] == 30.0
    assert beta["upside_from_buy_pct"] == 20.0
    assert alpha["change_pct"] == 1.25
    assert beta["change_pct"] == -0.4
    assert {m["key"] for m in alpha["fundamentals"]["metrics"]} >= {"pe", "roe"}
    assert {m["key"] for m in beta["fundamentals"]["metrics"]} >= {"pe", "roe"}


def test_peek_does_not_wait_on_hung_history_or_scrape(monkeypatch):
    import product.stock_peek as peek

    def hang():
        time.sleep(4)
        raise AssertionError("history must be timed out")

    def scrape(*_a, **_k):
        raise AssertionError("snapshot must not scrape fundamentals")

    monkeypatch.setattr(peek, "_load_frame", hang)
    monkeypatch.setattr("fundamentals.lazy.ensure_deep_fundamentals", scrape)
    t0 = time.monotonic()
    payload = build_stock_peek(
        "ALPHA",
        scan_payload={"records": [_scan_row("ALPHA", price=110, entry=100, stop=90, target=130, change_pct=0.8)]},
        long_term_payload={"records": []},
        raw_fundamentals={"data": {}},
        load_history=True,
        load_live=False,
    )
    assert time.monotonic() - t0 < 4.0
    assert payload["symbol"] == "ALPHA"
    assert payload["upside_from_buy_pct"] == 30.0
    assert payload["technical"]["change_pct"] == 0.8
    assert "history" in (payload.get("history_note") or "").lower()


def test_fetch_missing_scrapes_when_pack_is_thin():
    scan = {"records": [_scan_row("ALPHA", price=110, entry=100, stop=90, target=130)]}

    def resolve(symbol, force_refresh=False, write_cache=True):
        return (
            {
                "key_ratios": [{"name": "P/E", "value": "18.4"}, {"name": "ROE", "value": "22.1"}],
                "_source": "screener_in",
            },
            [{"source": "screener_in", "status": "OK", "message": "ok"}],
        )

    out = fetch_missing_fundamentals_for_peek(
        "ALPHA",
        resolve_fn=resolve,
        scan_payload=scan,
        long_term_payload={"records": []},
    )
    assert out["scrape"]["outcome"] == "READY"
    assert out["scrape"]["source"] == "screener_in"
    filled = {m["key"]: m["value"] for m in out["fundamentals"]["metrics"] if m.get("value") is not None}
    assert filled.get("pe") == 18.4
    assert filled.get("roe") == 22.1
    assert out["pack_thin"] is False
    ratio_vals = {str(r.get("label") or r.get("key")).lower(): r.get("value") for r in out.get("ratios") or []}
    assert any("p/e" in k or k == "pe" for k in ratio_vals)
    assert any("roe" in k for k in ratio_vals)


def test_fetch_missing_does_not_invent_when_sources_empty():
    scan = {"records": [_scan_row("ALPHA", price=110, entry=100, stop=90, target=130)]}

    def empty(*_a, **_k):
        return (None, [{"source": "screener_in", "status": "EMPTY", "message": "no pack"}])

    out = fetch_missing_fundamentals_for_peek(
        "ALPHA",
        resolve_fn=empty,
        scan_payload=scan,
        long_term_payload={"records": []},
    )
    assert out["scrape"]["outcome"] == "MISSING"
    filled = {m["key"]: m["value"] for m in out["fundamentals"]["metrics"] if m.get("value") is not None}
    assert "pe" not in filled
    assert "roe" not in filled
    assert out["pack_thin"] is True


def test_get_peek_never_calls_resolver(monkeypatch):
    import product.stock_peek as peek

    def boom(*_a, **_k):
        raise AssertionError("GET snapshot must not scrape Screener or Yahoo")

    monkeypatch.setattr("fundamentals.resolver.resolve", boom)
    monkeypatch.setattr(peek, "_cached_raw", lambda _s: {"data": {}})
    payload = build_stock_peek(
        "ALPHA",
        scan_payload={"records": [_scan_row("ALPHA", price=110, entry=100, stop=90, target=130)]},
        long_term_payload={"records": []},
        raw_fundamentals={"data": {}},
        load_history=False,
        load_live=False,
    )
    assert payload["symbol"] == "ALPHA"
    assert payload["pack_thin"] is True
    assert payload.get("scrape") in (None, {})


def test_fetch_missing_does_not_scrape_when_pack_has_numbers():
    scan = {
        "records": [_scan_row("ALPHA", price=110, entry=100, stop=90, target=130)],
    }
    lt = {
        "records": [{
            "symbol": "ALPHA",
            "fundamentals": {"pe": 19.0, "roe": 15.0},
            "classification": "GARP_CANDIDATE",
        }],
    }

    def boom(*_a, **_k):
        raise AssertionError("must not scrape a pack that already has numbers")

    out = fetch_missing_fundamentals_for_peek(
        "ALPHA",
        resolve_fn=boom,
        scan_payload=scan,
        long_term_payload=lt,
    )
    assert out["scrape"]["outcome"] == "CACHE"
    assert out["scrape"]["ran"] is False


def test_fetch_missing_times_out_without_inventing():
    scan = {"records": [_scan_row("ALPHA", price=110, entry=100, stop=90, target=130)]}

    def hang(*_a, **_k):
        time.sleep(4)
        raise AssertionError("scrape must be timed out")

    t0 = time.monotonic()
    out = fetch_missing_fundamentals_for_peek(
        "ALPHA",
        resolve_fn=hang,
        timeout=0.3,
        scan_payload=scan,
        long_term_payload={"records": []},
    )
    assert time.monotonic() - t0 < 2.0
    assert out["scrape"]["outcome"] == "TIMEOUT"
    assert out["upside_from_buy_pct"] == 30.0
    assert _pack_is_thin(out) is True
    filled = {m["key"] for m in out["fundamentals"]["metrics"] if m.get("value") is not None}
    assert "pe" not in filled
    assert "roe" not in filled
