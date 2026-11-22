"""GPT cross-review of 9e597998 (cycle 4, 22 Nov 2026): findings RS-01..RS-05 as regression tests.

Each test is GPT's own sketch. One adaptation: ``rs_percentile_on`` no longer
takes a frame (a name's composite is read from the window it is ranked in, the
RS-03 fix), so RS-02/03 call it with the session only.
"""
import pandas as pd
import pytest

from product import pit_features as PF


@pytest.fixture(autouse=True)
def _fresh_cache():
    PF.reset_breadth_cache()
    yield
    PF.reset_breadth_cache()


@pytest.fixture
def flat_rs_store(monkeypatch):
    from data import bhavcopy_store as bs

    idx = pd.bdate_range("2025-01-01", periods=253)
    days = tuple(idx.date)
    frames = {
        f"S{i:03d}": pd.DataFrame({"close": 110.0}, index=idx)
        for i in range(300)
    }
    monkeypatch.setattr(bs, "_store", frames)
    monkeypatch.setattr(bs, "_store_last_day", days[-1])
    monkeypatch.setattr(bs, "_store_sessions", len(days))
    monkeypatch.setattr(bs, "_CA_EVENTS", {})
    monkeypatch.setattr(PF, "_archive_dates", lambda: days)
    monkeypatch.setattr(PF, "market_context", lambda _: {})
    return bs, idx, days, frames


def test_future_bonus_does_not_split_historical_rs_ties(monkeypatch, flat_rs_store):
    bs, idx, days, frames = flat_rs_store
    for frame in frames.values():
        frame.loc[idx[-1], "close"] = 120.0
    at = f"{days[-1]}T19:00:00+05:30"

    def rank():
        return PF.fill_missing({}, "S000", at)[0]["rs_percentile"]

    assert rank() == 0.0
    monkeypatch.setattr(bs, "_CA_EVENTS", {
        "S000": [{
            "ex_date": idx[-1] + pd.Timedelta(days=7),
            "factor": 3.0,
            "type": "bonus",
        }]
    })
    assert rank() == 0.0  # 9e597998 yields 99.666667


def test_old_overlay_cannot_become_an_official_rs_bar(monkeypatch, flat_rs_store):
    from data import kite_client, nse_live, nse_universe

    bs, idx, days, frames = flat_rs_store
    d = days[-1]
    e = (idx[-1] + pd.offsets.BDay()).date()
    peers = list(frames)
    frames["PROBE"] = frames["S000"].copy()
    frames["PROBE"].loc[idx[-1], "close"] = 120.0
    frames["PROBE"].attrs["quantterm_live_overlay_date"] = d.isoformat()
    for symbol in peers:
        frames[symbol] = frames[symbol].iloc[:-1].copy()
    monkeypatch.setattr(bs, "_store_last_day", days[-2])
    monkeypatch.setattr(bs, "_store_sessions", len(days) - 1)
    monkeypatch.setattr(bs, "_save_pkl", lambda: None)
    monkeypatch.setattr(bs, "_read_day", lambda _: pd.DataFrame({
        "symbol": peers, "date": idx[-1], "close": 110.0,
    }))
    bs._incremental_append([d])

    class Clock:
        @staticmethod
        def today():
            return e

    monkeypatch.setattr(nse_live, "date", Clock)
    from core import market_clock
    monkeypatch.setattr(market_clock, "today_ist", lambda: e)      # the producer dates by the NSE clock (C7-08)
    monkeypatch.setattr(nse_live, "_is_trading_now", lambda: True)
    monkeypatch.setattr(nse_universe, "get_nse_universe", lambda: [])
    monkeypatch.setattr(kite_client, "kite_credentials_available", lambda: False)
    monkeypatch.setattr(nse_live, "fetch_live_snapshot",
                        lambda: {"PROBE": {"close": 130.0}})
    assert nse_live.apply_live_to_store() == 1

    rank = PF.rs_percentile_on("PROBE", d)
    assert rank is None  # 9e597998 yields 99.667774


def test_store_replacement_invalidates_rs_peer_scores(monkeypatch, flat_rs_store):
    bs, idx, days, frames = flat_rs_store
    session = days[-1]

    def rank():
        return PF.rs_percentile_on("S000", session)

    assert rank() == 0.0
    corrected = frames["S000"].copy()
    corrected.loc[idx[-1], "close"] = 120.0
    # Same state transition as a same-session, same-size full rebuild.
    monkeypatch.setattr(bs, "_store", {**frames, "S000": corrected})
    assert rank() == pytest.approx(100 * 299 / 300, abs=1e-6)


def test_new_rs_producer_cannot_claim_old_forward_predictions(monkeypatch):
    from product import challenger_learning as cl

    rows = [
        {"observation_id": "train-a", "features": {"rs_percentile": 10.0},
         "outcome": -1.0, "outcome_meta": {"evidence_class": "PAPER_FORWARD"}},
        {"observation_id": "train-b", "features": {"rs_percentile": 90.0},
         "outcome": 1.0, "outcome_meta": {"evidence_class": "PAPER_FORWARD"}},
    ]
    coef, intercept, medians, scales = cl._fit(*cl._matrix(rows))
    args = dict(rows=rows, coef=coef, intercept=intercept, medians=medians,
                scales=scales, validation={}, status=cl.SHADOW_CANDIDATE)
    with monkeypatch.context() as patch:
        patch.setattr(cl, "_current_producer", lambda: PF.CONTEXT_PRODUCER)
        old = cl._serialize_model(**args)
    new = cl._serialize_model(**args)
    later_settlement = {
        "observation_id": "forward-a",
        "outcome": 1.0,
        "outcome_meta": {"evidence_class": "PAPER_FORWARD"},
        "meta": {"challenger_model_version": old["model_version"],
                 "predicted_p": 0.5, "challenger_predicted_p": 0.8},
    }
    assert cl.forward_comparison(old["model_version"], [later_settlement])["n"] == 1
    assert cl.forward_comparison(new["model_version"], [later_settlement])["n"] == 0


def test_v2_breadth_does_not_build_rs_history(monkeypatch, flat_rs_store):
    _, _, days, _ = flat_rs_store
    calls = []
    original = PF._rs_scores

    def record_scores(closes):
        calls.append(len(closes))
        return original(closes)

    monkeypatch.setattr(PF, "_rs_scores", record_scores)
    values, _ = PF.fill_missing(
        {}, "S000", f"{days[-1]}T19:00:00+05:30",
        producer=PF.CONTEXT_PRODUCER,
    )
    assert values["breadth_pct_above_50dma"] == 0.0
    assert calls == []  # 9e597998 calls _rs_scores for all 300 symbols


# ── the store-integrity half of RS-02, and the revision RS-03 keys on ─────────

def test_an_archive_drops_the_overlay_row_of_a_symbol_it_does_not_list(monkeypatch):
    from data import bhavcopy_store as bs

    idx = pd.bdate_range("2026-09-01", periods=5)
    d = idx[-1]
    listed = pd.DataFrame({"close": [100.0] * 4}, index=idx[:-1])
    orphan = pd.DataFrame({"close": [50.0] * 5}, index=idx)          # last row: an intraday quote
    orphan.attrs["quantterm_live_overlay_date"] = d.date().isoformat()
    monkeypatch.setattr(bs, "_store", {"LISTED": listed, "ORPHAN": orphan})
    monkeypatch.setattr(bs, "_store_last_day", idx[-2].date())
    monkeypatch.setattr(bs, "_store_sessions", 4)
    monkeypatch.setattr(bs, "_save_pkl", lambda: None)
    monkeypatch.setattr(bs, "_read_day", lambda day: pd.DataFrame(
        {"symbol": ["LISTED"], "date": d, "close": [101.0]}))
    before = bs.store_revision()[0]
    bs._incremental_append([d.date()])
    assert d not in bs._store["ORPHAN"].index                       # never persisted as history
    assert "quantterm_live_overlay_date" not in bs._store["ORPHAN"].attrs
    assert len(bs._store["ORPHAN"]) == 4 and bs._store["LISTED"].loc[d, "close"] == 101.0
    assert bs.store_revision()[0] == before + 1


def test_only_builders_bump_the_store_revision(monkeypatch):
    from data import bhavcopy_store as bs

    first = bs.store_revision()
    bs.reset_in_memory_store()
    second = bs.store_revision()
    assert second[0] == first[0] + 1
    assert bs.store_revision() == second                            # reads never bump
