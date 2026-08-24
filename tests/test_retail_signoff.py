"""
Retail runtime acceptance / sign-off tests (deterministic, network-free).

These lock the product-integrity guarantees the retail app must keep: the four user journeys, the
underlying-first F&O funnel and its exact reconciliation, distinct no-data vs no-trade states,
deterministic ranking, rerun stability, no broker reachability, and honest failure surfaces. They
drive the CANONICAL product/ + data/fno_universe modules — no domain logic is reimplemented here.
"""
from __future__ import annotations

import importlib
import inspect
import io
import sys
import tokenize
from dataclasses import dataclass
from datetime import date

import pandas as pd

from product.projection import ProductInputs, build_product_state, TERMINOLOGY
from product.no_trade import build_no_trade_explanation
from product.scan_store import build_scan_payload, scan_age_hours, watchlist_rows
from data.fno_universe import build_fno_universe, evaluate_all_underlyings, current_fno_universe


# ── fixtures ─────────────────────────────────────────────────────────────────────
def _eq(symbol, name=None):
    return {"exchange": "NSE", "segment": "NSE", "instrument_type": "EQ",
            "tradingsymbol": symbol, "name": name or symbol, "instrument_token": 1, "lot_size": 1}


def _fut(symbol, expiry, token, *, name=None):
    return {"exchange": "NFO", "segment": "NFO-FUT", "instrument_type": "FUT",
            "tradingsymbol": f"{symbol}{expiry.replace('-', '')}FUT", "name": name or symbol,
            "expiry": expiry, "instrument_token": token, "lot_size": 500}


@dataclass
class _Sig:
    signals: list
    reasons: list
    score: float = 72.0
    verdict: str = "BUY"
    price: float = 100.0
    momentum_5d: float = 8.0
    rsi: float = 65.0
    volume_ratio: float = 1.8


def _mixed_master():
    # a valid mapped underlying, a missing cash mapping, insufficient history, evaluated-rejected, qualified
    return [
        _eq("AAA"), _eq("BBB"), _eq("CCC"), _eq("DDD"),
        _fut("AAA", "2026-08-27", 11), _fut("AAA", "2026-09-24", 12),   # qualified (2 contracts)
        _fut("BBB", "2026-08-27", 13),                                   # evaluated but rejected
        _fut("CCC", "2026-08-27", 14),                                   # insufficient history
        _fut("MISSING", "2026-08-27", 15),                              # no cash mapping
        _fut("NIFTY", "2026-08-27", 99),                               # index → excluded intentionally
    ]


def _histories():
    long = pd.DataFrame({"close": range(100)}, index=pd.date_range("2026-01-01", periods=100))
    return {"AAA": long, "BBB": long, "CCC": pd.DataFrame({"close": range(20)},
                                                          index=pd.date_range("2026-01-01", periods=20))}


def _analyzer(symbol, history):
    if symbol == "AAA":
        return _Sig(signals=["MOMENTUM"], reasons=["Up 8% in 5 days"], score=80.0)
    return _Sig(signals=["PRE_BREAKOUT"], reasons=["Near breakout, not momentum"], score=40.0)


def _funnel(master=None, minimum=60):
    universe = build_fno_universe(master or _mixed_master(), as_of=date(2026, 7, 31))
    hist = _histories()
    funnel = evaluate_all_underlyings(universe, history_getter=lambda s: hist.get(s),
                                      analyzer=_analyzer, minimum_sessions=minimum)
    return universe, funnel


def _code_only(mod) -> str:
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(inspect.getsource(mod)).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            out.append(tok.string)
    return " ".join(out)


# ── 1: boots without optional feedparser ────────────────────────────────────────
def test_boots_without_feedparser(monkeypatch):
    monkeypatch.setitem(sys.modules, "feedparser", None)   # `import feedparser` now raises
    fetcher = importlib.reload(importlib.import_module("news.fetcher"))
    assert fetcher.feedparser is None
    assert fetcher.NewsFetcher()._fetch_rss("http://example.com/rss", 24) == []   # graceful, no crash
    importlib.reload(fetcher)                                # restore for other tests
    importlib.import_module("ui.retail_pages_v2")           # retail page chain imports cleanly


# ── 2: retail is the default route ───────────────────────────────────────────────
def test_retail_home_is_default_route():
    # read app.py as text — importing it would execute the Streamlit script (navigation.run())
    from pathlib import Path
    src = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    assert "st.Page(render_home" in src and "default=True" in src
    assert 'title="Today"' in src and "DEVBLOOM_CSS" in src
    assert "render_desk_backtest" in src
    assert '"Everyday"' not in src


# ── 3: advanced view remains reachable ───────────────────────────────────────────
def test_advanced_view_reachable():
    from ui.retail_pages import render_advanced
    assert callable(render_advanced)
    assert "legacy_app" in inspect.getsource(render_advanced)
    # check the engineering entrypoint exists by PATH — importing it would run the legacy app
    from pathlib import Path
    assert Path(__file__).resolve().parents[1].joinpath("legacy_app.py").exists()


# ── 4: ready-but-no-trade is DISTINCT from unavailable data ──────────────────────
def test_ready_no_trade_distinct_from_unavailable():
    ready = build_product_state(ProductInputs(kite_connected=True, active_snapshot_id="s1",
                                              data_ready=True, paper_auto_enabled=True,
                                              worker_running=True, market_open=True, open_positions=0))
    unavailable = build_product_state(ProductInputs(kite_connected=True, active_snapshot_id=None,
                                                    data_ready=False))
    assert ready.primary_key == "none" and "running" in ready.headline.lower()
    assert unavailable.primary_key == "update_data" and "data" in unavailable.headline.lower()
    assert ready.headline != unavailable.headline
    for s in (ready.headline, ready.activity, unavailable.headline):
        assert "error" not in s.lower()                    # a healthy no-trade is never an error
    # the funnel explanation also separates "no scan yet" from "scan found nothing ready"
    no_scan = build_no_trade_explanation(None)
    scanned = build_no_trade_explanation({"universe_size": 1800, "summary": {"ready_to_trade": 0,
                                                                             "with_any_setup": 5, "momentum": 3}})
    assert no_scan.headline != scanned.headline
    assert "no entry-ready setup" in scanned.headline.lower()


# ── 5: stale data is visibly labelled ────────────────────────────────────────────
def test_stale_scan_age_is_computed():
    from datetime import datetime, timezone, timedelta
    now = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)
    payload = {"scanned_at": (now - timedelta(hours=30)).isoformat(), "records": []}
    age = scan_age_hours(payload, now=now)
    assert age is not None and 29.9 < age < 30.1
    assert scan_age_hours({"records": []}) is None         # unknown age is None, not a fake 0


# ── 6: a provider error cannot become a false "no opportunities" ─────────────────
def test_provider_error_is_recorded_not_silent():
    universe = build_fno_universe([_eq("AAA"), _fut("AAA", "2026-08-27", 11)], as_of=date(2026, 7, 31))

    def boom(_sym):
        raise RuntimeError("history provider down")

    funnel = evaluate_all_underlyings(universe, history_getter=boom, analyzer=_analyzer,
                                      minimum_sessions=60)
    assert funnel.momentum_qualified == 0
    assert len(funnel.excluded) == 1                        # NOT an empty, ambiguous result
    row = funnel.excluded[0]
    assert row.stage == "history" and "provider down" in row.reason
    # analyzer failure is distinct from a genuine no-momentum result
    hist = {"AAA": pd.DataFrame({"close": range(100)}, index=pd.date_range("2026-01-01", periods=100))}
    def bad_analyzer(_s, _h):
        raise ValueError("scanner exploded")
    f2 = evaluate_all_underlyings(universe, history_getter=lambda s: hist.get(s),
                                  analyzer=bad_analyzer, minimum_sessions=60)
    assert f2.excluded[0].stage == "analysis"


# ── 7: all unique F&O underlyings enter the funnel ───────────────────────────────
def test_all_unique_underlyings_enter_funnel():
    universe, funnel = _funnel()
    assert universe.unique_stock_underlyings == 4          # AAA,BBB,CCC,MISSING (NIFTY is index)
    assert universe.index_future_contracts == 1
    handled = {r.symbol for r in funnel.rows} | {e.underlying for e in universe.exclusions}
    assert handled == {"AAA", "BBB", "CCC", "MISSING"}     # nobody vanished


# ── 8: missing mappings are recorded ─────────────────────────────────────────────
def test_missing_mapping_recorded():
    universe, _ = _funnel()
    miss = [e for e in universe.exclusions if e.underlying == "MISSING"]
    assert miss and miss[0].stage == "canonical_mapping"


# ── 9: insufficient-history exclusions include observed and required sessions ────
def test_insufficient_history_shows_counts():
    _, funnel = _funnel()
    ccc = next(r for r in funnel.rows if r.symbol == "CCC")
    assert ccc.stage == "history" and "20 sessions available" in ccc.reason and "60 required" in ccc.reason


# ── 10: evaluated-but-rejected symbols remain in reconciliation ──────────────────
def test_evaluated_rejected_remains():
    _, funnel = _funnel()
    bbb = next(r for r in funnel.rows if r.symbol == "BBB")
    assert bbb.qualified is False and bbb.stage == "momentum"   # evaluated, then rejected — recorded


# ── 11: qualified + excluded + mapping-exclusions == input universe ──────────────
def test_funnel_reconciles_exactly():
    universe, funnel = _funnel()
    total = len(funnel.qualified) + len(funnel.excluded) + len(universe.exclusions)
    assert total == universe.unique_stock_underlyings
    assert len(funnel.rows) == universe.mapped_underlyings
    assert universe.mapped_underlyings + len(universe.exclusions) == universe.unique_stock_underlyings


# ── 12: duplicate derivative contracts do not duplicate an underlying ────────────
def test_duplicate_contracts_one_underlying():
    universe, funnel = _funnel()
    aaa_rows = [r for r in funnel.rows if r.symbol == "AAA"]
    assert len(aaa_rows) == 1
    aaa = next(u for u in universe.underlyings if u.symbol == "AAA")
    assert aaa.contract_count == 2 and aaa.instrument_token == 11   # nearest contract kept


# ── 13: ranking is deterministic (stable tie-break) ──────────────────────────────
def test_ranking_is_deterministic():
    names = {"AAA": "Aaa", "BBB": "Bbb", "CCC": "Ccc"}
    results = [
        {"symbol": s, "signals": ["MOMENTUM"], "score": 70.0, "verdict": "BUY", "reasons": ["m"]}
        for s in ("CCC", "AAA", "BBB")                      # equal scores, shuffled input
    ]
    p1 = build_scan_payload(names, results)
    p2 = build_scan_payload(names, list(reversed(results)))
    order1 = [r["symbol"] for r in p1["records"]]
    order2 = [r["symbol"] for r in p2["records"]]
    assert order1 == order2 == ["AAA", "BBB", "CCC"]        # symbol tiebreak → identical, sorted


# ── 14: repeated evaluation does not mutate canonical results ────────────────────
def test_repeated_evaluation_is_stable():
    u1, f1 = _funnel()
    u2, f2 = _funnel()
    assert [(r.symbol, r.qualified, r.stage) for r in f1.rows] == \
           [(r.symbol, r.qualified, r.stage) for r in f2.rows]
    assert (f1.total_underlyings, f1.data_ready, f1.evaluated, f1.momentum_qualified) == \
           (f2.total_underlyings, f2.data_ready, f2.evaluated, f2.momentum_qualified)
    # product state is a frozen projection — building twice yields an equal value
    inp = ProductInputs(kite_connected=True, active_snapshot_id="s", data_ready=True)
    assert build_product_state(inp) == build_product_state(inp)


# ── 15: no retail action can reach a broker order ────────────────────────────────
def test_no_broker_order_in_retail_paths():
    mods = ["ui.retail_pages_v2", "ui.retail_home_momentum", "ui.retail_trade_market",
            "ui.retail_backtest_data", "ui.retail_pages", "ui.fno_momentum_page",
            "ui.desk_board", "ui.desk_pages", "product.paper_lessons",
            "product.projection", "product.gather", "product.no_trade", "product.market_view",
            "product.scan_store", "product.retail_backtest", "product.runtime", "data.fno_universe"]
    banned = ("place_order", "place_gtt", "modify_order", "cancel_order", "cancel_gtt")
    for name in mods:
        code = _code_only(importlib.import_module(name))
        for b in banned:
            assert b not in code, f"{b} reachable in {name}"


# ── 16: historical F&O evaluation calls no live provider ─────────────────────────
def test_evaluation_uses_injected_providers_only():
    code = _code_only(evaluate_all_underlyings)
    for live in ("prefetch", "requests", "yf", "download", "current_fno_universe", "date.today"):
        assert live not in code                            # evaluation only touches its injected callables


# ── 17: empty / malformed / unavailable instrument inputs fail safely ────────────
def test_instrument_inputs_fail_safely():
    empty = build_fno_universe([], as_of=date(2026, 7, 31))
    assert empty.unique_stock_underlyings == 0 and empty.underlyings == () and empty.exclusions == ()
    malformed = build_fno_universe([{"exchange": "NFO", "instrument_type": "FUT"},   # no name/symbol
                                    {"garbage": 1}], as_of=date(2026, 7, 31))
    assert isinstance(malformed.unique_stock_underlyings, int)                       # no crash
    unavailable = current_fno_universe(client=None, cache_path="does/not/exist.csv")
    assert unavailable.source == "unavailable" and unavailable.underlyings == ()


# ── 18: displayed results carry an as-of timestamp ───────────────────────────────
def test_results_carry_as_of_timestamp():
    payload = build_scan_payload({"AAA": "Aaa"},
                                 [{"symbol": "AAA", "signals": ["MOMENTUM"], "score": 70.0,
                                   "verdict": "BUY", "reasons": ["m"]}])
    assert payload.get("scanned_at") and scan_age_hours(payload) is not None
    # the F&O universe is built against an explicit as_of (never an implicit hidden 'now')
    sig = inspect.signature(build_fno_universe)
    assert "as_of" in sig.parameters
