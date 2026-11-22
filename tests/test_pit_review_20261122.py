"""GPT cross-review of 3391c665 (22 Nov 2026), findings QT-01..QT-09 as regression tests.

Each test is GPT's own sketch. Adaptations to the fixed design, so the tests
exercise the same claim:

* QT-02/03/06/09: the "official bar" gate is now the verified archive list
  (``_archive_dates`` / ``archive_fn``), because ``official_dates()`` also lists
  settlement-calendar sessions that have no archive (that was QT-02 itself).
* QT-05: a stored fill carries the provenance ``freeze_decision`` always writes
  (``meta.pit_features.filled``); the live analog query projects exactly those
  fills back out. Card-supplied stored values are live evidence by design.
* QT-06: the stubbed ``market_context`` takes the session (and optional
  index_fn) and returns a complete context once the sources are ready.

QT-08 (an EQ-series fund named only by its instrument name) and QT-09 (a cold
breadth pass inside a WHY request) are recorded as strict xfails: the first
needs a point-in-time instrument master the store does not keep; the second is
accepted as one bounded, cached pass per store generation (journal cycle 3).
"""
import pandas as pd
import pytest

from product import pit_features as PF

def test_review_freeze_requires_bhavcopy_publication(tmp_path, monkeypatch):
    from data import bhavcopy_store as bs
    from product import evidence_intelligence as ei
    from product.decision import Decision
    from research import feature_store as fs

    monkeypatch.setattr(fs, "_DB_PATH", tmp_path / "features.db")
    monkeypatch.setattr(PF, "context_features", lambda at: {})
    idx = pd.bdate_range(end="2026-09-04", periods=51)
    frame = pd.DataFrame(
        {"high": 101., "low": 99., "close": 100., "deliv_per": 40.}, index=idx
    )
    frame.loc[idx[-1], "deliv_per"] = 80.
    # First read after the 18:00 archive arrival, for a decision made at 15:31.
    monkeypatch.setattr(bs, "get_ohlcv", lambda symbol: frame)
    d = Decision(symbol="TCS", state="WATCH",
                 generated_at="2026-09-04T15:31:00+05:30")

    assert ei.freeze_decision(d)["status"] == "frozen"
    row = fs.get_observation(f"decision::{d.decision_id}")
    assert row["features"]["delivery_pct"] in (None, 40.)


@pytest.mark.parametrize("archive_landed", [False, True])
def test_review_breadth_rejects_live_row(monkeypatch, archive_landed):
    from datetime import date
    from core import session_window as sw
    from data import bhavcopy_store as bs
    from research.intelligence.data import nse_calendar as cal

    idx = pd.bdate_range(end="2026-09-04", periods=52)
    cut = idx[-1].date()
    frame = pd.DataFrame({"close": [100.] * 51 + [110.]}, index=idx)
    frame.attrs["quantterm_live_overlay_date"] = cut.isoformat()
    disk = list(idx[:-1].date) + [cut if archive_landed else date(2026, 9, 7)]
    monkeypatch.setattr(bs, "_dates_on_disk", lambda: disk)
    monkeypatch.setattr(bs, "_store", {f"S{i:03d}": frame for i in range(300)})
    monkeypatch.setattr(bs, "_CA_EVENTS", {})
    monkeypatch.setattr(cal, "load_holidays", lambda: set())
    monkeypatch.setattr(cal, "covered_years", lambda holidays: {2026})
    monkeypatch.setattr(sw, "_CACHE", None)
    monkeypatch.setattr(PF, "_BREADTH_CACHE", None)

    assert PF.breadth_pct_above_50dma(cut) in (None, 0.)


def test_review_future_split_cannot_change_breadth(monkeypatch):
    from data import bhavcopy_store as bs

    idx = pd.bdate_range(end="2026-09-04", periods=1765)
    cut = idx[-1].date()
    frame = pd.DataFrame({"close": 101.}, index=idx)
    monkeypatch.setattr(bs, "_store", {f"S{i:03d}": frame for i in range(300)})
    monkeypatch.setattr(bs, "_CA_EVENTS", {})
    monkeypatch.setattr(PF, "_archive_dates", lambda: list(idx.date))
    monkeypatch.setattr(PF, "_BREADTH_CACHE", None)
    assert PF.breadth_pct_above_50dma(cut) == 0.

    monkeypatch.setattr(bs, "_CA_EVENTS", {
        "S000": [{"ex_date": pd.Timestamp("2026-09-07"), "factor": 3.}]
    })
    PF.reset_breadth_cache()
    assert PF.breadth_pct_above_50dma(cut) == 0.  # HEAD yields 0.333333


def test_review_approved_model_keeps_its_validated_inputs(monkeypatch):
    from product import challenger_learning as cl
    from product import trading_thesis as tt
    from product.decision import Decision

    n = len(cl.MODEL_FEATURES)
    i = cl.MODEL_FEATURES.index("vix")
    coef, medians = [0.] * n, [0.] * n
    coef[i], medians[i] = 1., 10.
    monkeypatch.setattr(cl, "load", lambda path=None: {"current": {
        "status": cl.PAPER_ACTIVE, "model_version": "clf_validated_before_fill",
        "coef": coef, "medians": medians, "scales": [1.] * n, "intercept": 0.,
    }})
    monkeypatch.setattr(PF, "technical_features", lambda *a, **kw: {})
    monkeypatch.setattr(PF, "context_features", lambda at: {"vix": 13.})
    at = "2026-09-04T19:00:00+05:30"
    d = Decision(symbol="TCS", state="WATCH", generated_at=at)
    paper = cl.paper_selection_adjustment({"symbol": "TCS", "scan_scanned_at": at})
    board = tt.active_learning_adjustment(d)

    assert (paper["adjustment"], board["adjustment"]) == (0., 0.)


def test_review_learning_fill_preserves_live_analog_veto(monkeypatch):
    from product import evidence_intelligence as ei
    from product.decision import Decision
    from product.decision_service import historical_demotion
    from research import feature_schema as fs

    values = {
        k: f.categories[0] if f.categories else
        next((v for v in (f.soft_lo, f.lo, 0.) if v is not None))
        for k, f in fs.FEATURE_REGISTRY.items()
    }
    for k in ("atr_pct", "adr_pct", "clv", "volume_z", "liquidity_cr",
              "dist_20ema_pct", "dist_50dma_pct", "delivery_pct", "vix"):
        values[k] = None

    def row(i, vix):
        features = dict(values, vix=vix)
        return {
            "observation_id": str(i), "symbol": f"S{i:03d}",
            "ts": "2026-08-03T18:00:00+05:30", "outcome": -1.,
            "features": features, "validation": fs.validate_vector(features).problems,
            "meta": {"setup": "VCP", "pit_features": {"filled": ["vix"] if vix is not None else []}},
            "outcome_meta": {"evidence_class": "HISTORICAL_REPLAY", "not_pnl": True},
        }

    rows = [row(i, None) for i in range(30)]
    monkeypatch.setattr(ei, "_prior_observation_rows", lambda at: rows)
    monkeypatch.setattr(ei, "_policy_prior", lambda setup: {})
    q = Decision(symbol="TCS", state="BUY", setup="VCP",
                 generated_at="2026-09-04T19:00:00+05:30",
                 technical_evidence={k: values[k] for k in ("rsi", "volume_ratio")})

    assert historical_demotion(ei.evidence_read(q, k=30))
    rows[:15] = [row(i, 14.) for i in range(15)]
    assert historical_demotion(ei.evidence_read(q, k=30))


def test_review_refresh_after_store_catches_up(tmp_path, monkeypatch):
    import time
    from product import evidence_intelligence as ei
    from product.decision import Decision
    from research import feature_store as fs

    idx = pd.bdate_range(end="2026-09-04", periods=52)
    frame = pd.DataFrame({"close": [100.] * 51 + [110.]}, index=idx)
    early, late = "2026-09-04T18:58:00+05:30", "2026-09-04T19:00:00+05:30"
    source = {"ready": False, "clock": 100.}
    monkeypatch.setattr(time, "monotonic", lambda: source["clock"])
    monkeypatch.setattr(PF, "_BREADTH_CACHE", None)
    monkeypatch.setattr(PF, "_CONTEXT_CACHE", {})
    monkeypatch.setattr(PF, "_archive_dates", lambda: list(idx.date))
    monkeypatch.setattr(PF, "_store_symbols", lambda: [f"S{i:03d}" for i in range(300)])
    monkeypatch.setattr(PF, "_store_ohlcv",
                        lambda symbol: frame if source["ready"] else frame.iloc[:-1])
    monkeypatch.setattr(PF, "market_context",
                        lambda cut, index_fn=None: {"vix": 14., "index_trend": "UP"} if source["ready"] else {})
    monkeypatch.setattr(PF, "technical_features", lambda *a, **kw: {})
    monkeypatch.setattr(fs, "_DB_PATH", tmp_path / "features.db")
    assert {k: v for k, v in PF.context_features(early).items() if k != "_session"} == {}

    source.update(ready=True, clock=220.)
    d = Decision(symbol="TCS", state="WATCH", generated_at=late)
    assert ei.freeze_decision(d)["status"] == "frozen"
    source["clock"] = 521.
    refreshed = PF.context_features(late)
    stored = fs.get_observation(f"decision::{d.decision_id}")["features"]
    assert (stored["vix"], stored["breadth_pct_above_50dma"],
            refreshed.get("breadth_pct_above_50dma")) == (14., 100., 100.)


def test_review_market_context_uses_latest_and_distinct_sessions():
    idx = pd.bdate_range(end="2026-09-04", periods=199)
    nifty = pd.DataFrame({"Close": range(1000, 1199)}, index=idx)
    nifty = pd.concat([nifty, nifty.iloc[[-1]]])
    vix = pd.DataFrame({"Close": [14., 40.]}, index=idx[[-1, -2]])

    got = PF.market_context(idx[-1].date(), index_fn=lambda t, d:
                            vix if t == "^INDIAVIX" else nifty)
    assert got == {"vix": 14.}

def test_review_context_age_is_relative_to_decision(monkeypatch):
    vix = pd.DataFrame({"Close": [14.]}, index=pd.to_datetime(["2026-09-04"]))
    monkeypatch.setattr(PF, "_CONTEXT_CACHE", {})
    monkeypatch.setattr(PF, "_index_loader",
                        lambda t, d: vix if t == "^INDIAVIX" else None)
    monkeypatch.setattr(PF, "breadth_pct_above_50dma", lambda cut: None)

    assert "vix" not in PF.context_features("2026-09-08T10:00:00+05:30")


@pytest.mark.xfail(strict=True, reason="QT-08 known limit: no point-in-time instrument names in the store")
def test_review_name_classified_etf_cannot_supply_breadth_minimum():
    from data.nse_universe import _is_non_stock_fund

    assert _is_non_stock_fund("MON100", "MOTILAL OSWAL NASDAQ 100 ETF")
    idx = pd.bdate_range(end="2026-09-04", periods=51)
    frame = pd.DataFrame({"close": [100.] * 50 + [110.]}, index=idx)
    frames = {f"S{i:03d}": frame for i in range(299)}
    frames["MON100"] = frame
    got = PF.breadth_pct_above_50dma(
        idx[-1].date(), symbols_fn=lambda: list(frames), ohlcv_fn=frames.get,
        archive_fn=lambda: list(idx.date),
    )
    assert got is None


@pytest.mark.xfail(strict=True, reason="QT-09 accepted: one bounded breadth pass per store generation")
def test_review_why_does_not_build_breadth_synchronously(tmp_path, monkeypatch):
    from product import challenger_learning as cl, decision_service as ds
    from product import decision_ranking as dr, evidence_intelligence as ei
    from product import recommendation_truth as rt, scan_store as ss
    from research import feature_store as fs

    at = "2026-09-04T19:00:00+05:30"
    monkeypatch.setattr(fs, "_DB_PATH", tmp_path / "features.db")
    monkeypatch.setattr(ss, "current_buy_certificate", lambda ts: set())
    monkeypatch.setattr(rt, "project_workspace_truth", lambda ws: {})
    monkeypatch.setattr(dr, "load_evidence_store", lambda path=None: {})
    monkeypatch.setattr(cl, "load", lambda path=None: {})
    monkeypatch.setattr(ei, "_policy_prior", lambda setup: {})
    monkeypatch.setattr(PF, "_BREADTH_CACHE", None)
    monkeypatch.setattr(PF, "_CONTEXT_CACHE", {})
    monkeypatch.setattr(PF, "_archive_dates", lambda: [__import__("datetime").date(2026, 9, 4)])
    monkeypatch.setattr(PF, "_store_symbols", lambda: [f"S{i:03d}" for i in range(300)])
    monkeypatch.setattr(PF, "market_context", lambda cut, index_fn=None: {})
    monkeypatch.setattr(PF, "technical_features", lambda *a, **kw: {})
    passes = []
    monkeypatch.setattr(PF, "_market_pass",
                        lambda *a: passes.append(PF._BREADTH_LOCK.locked()) or PF._MarketPass())
    workspace = {"scan_scanned_at": at, "categories": [{"cards": [
        {"symbol": "TCS", "reco_tier": "watch"},
        {"symbol": "INFY", "reco_tier": "watch"},
    ]}]}

    assert ds.decision_why("TCS", workspace=workspace)["available"]
    assert passes == []  # HEAD records [True]


