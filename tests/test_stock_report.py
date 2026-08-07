"""
Stock Research Report — the search-bar aggregator.

It composes many live resources, so these tests monkeypatch the underlying
feeds: the report must (1) never raise (fail-open), (2) compose a setup with its
plain-English "why" when one exists, and (3) degrade to a partial report when a
symbol has no setup. Network is never touched.
"""
import types

from research import stock_report as SR


class _FakeSignal:
    """Only the attributes the aggregator reads off a StockSignal."""
    def __init__(self):
        self.verdict = "BUY"
        self.score = 78.0
        self.breakout_conviction = 66.0
        self.signal_labels = ["Breakout 52W", "Pocket Pivot"]
        self.signals = ["BREAKOUT_52W", "POCKET_PIVOT"]
        self.reasons = ["clean breakout on volume"]
        self.entry = 100.0
        self.stop = 96.0
        self.target = 110.0
        self.rsi = 62.0

    @property
    def risk_reward(self):
        return (self.target - self.entry) / (self.entry - self.stop)


def _silence_side_feeds(monkeypatch):
    """Stub every network/data feed the report touches beyond quote+setup."""
    import agents.tools as tools
    monkeypatch.setattr(tools, "get_technical_indicators", lambda s: {"rsi": 62})
    monkeypatch.setattr(tools, "get_fundamentals", lambda s: {"pe": 24.5})
    import scan.breadth as breadth
    monkeypatch.setattr(breadth, "breadth_from_cache",
                        lambda: {"verdict": "HEALTHY", "pct_above_50": 61})
    import scan.sector_heat as sh
    monkeypatch.setattr(sh, "sector_of", lambda s: "IT")
    import core.regime_engine as re_
    monkeypatch.setattr(re_, "compute_regime",
                        lambda: types.SimpleNamespace(market_regime="TRENDING_BULL"))
    import risk.position_sizer as ps
    monkeypatch.setattr(ps, "size_position",
                        lambda e, s, capital=0: {"qty": 25, "invested": 2500,
                                                 "max_loss": 100})
    import research.explainability as ex
    monkeypatch.setattr(ex, "row_intelligence", lambda *a, **k: {
        "why_buy": {"summary": "Why buy: strong breakout"},
        "evidence": {"belief": "breakouts work", "summary": "184 obs"},
        "trust": {"summary": "Trust basis: 184 observations"},
        "similar_history": {"found": False}})
    import research.drift as drift
    monkeypatch.setattr(drift, "drift_report", lambda: [])


def test_report_composes_a_full_buy_setup(monkeypatch):
    monkeypatch.setattr(SR, "_quote", lambda sym, us: {
        "price": 101.0, "prev": 99.0, "change_pct": 2.02,
        "week52_high": 120, "week52_low": 70})
    monkeypatch.setattr(SR, "_setup", lambda sym, us: _FakeSignal())
    _silence_side_feeds(monkeypatch)

    rep = SR.research_stock("RELIANCE")
    assert rep["symbol"] == "RELIANCE" and rep["cur"] == "₹"
    assert rep["quote"]["price"] == 101.0
    s = rep["setup"]
    assert s["verdict"] == "BUY" and s["signals"] == ["Breakout 52W", "Pocket Pivot"]
    assert s["rr"] == round((110 - 100) / (100 - 96), 1)   # 2.5×
    assert rep["sizing"]["qty"] == 25
    assert rep["why"]["why_buy"]["summary"].startswith("Why buy")
    assert rep["fundamentals"]["pe"] == 24.5
    assert rep["context"]["market_health"] == "HEALTHY"
    line = SR.research_summary_line(rep)
    assert "RELIANCE" in line and "BUY" in line


def test_report_is_fail_open_when_no_setup(monkeypatch):
    monkeypatch.setattr(SR, "_quote", lambda sym, us: {})
    monkeypatch.setattr(SR, "_setup", lambda sym, us: None)   # no setup / thin data
    _silence_side_feeds(monkeypatch)
    rep = SR.research_stock("XYZUNKNOWN")
    assert rep["setup"]["verdict"] == "NO SETUP"             # graceful, no raise
    assert "symbol" in rep and "technicals" in rep
    assert "poori research" in SR.research_summary_line(rep)


def test_report_never_raises_even_when_everything_breaks(monkeypatch):
    _silence_side_feeds(monkeypatch)                         # then break them:
    def _boom(*a, **k):
        raise RuntimeError("feed down")
    monkeypatch.setattr(SR, "_quote", _boom)
    monkeypatch.setattr(SR, "_setup", _boom)
    import agents.tools as tools
    monkeypatch.setattr(tools, "get_technical_indicators", _boom)
    monkeypatch.setattr(tools, "get_fundamentals", _boom)
    import scan.breadth as breadth
    monkeypatch.setattr(breadth, "breadth_from_cache", _boom)
    import core.regime_engine as re_
    monkeypatch.setattr(re_, "compute_regime", _boom)
    rep = SR.research_stock("ANYTHING")                      # must not raise
    assert rep["symbol"] == "ANYTHING"
    assert isinstance(rep.get("quote"), dict)
