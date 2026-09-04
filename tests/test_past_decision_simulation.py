"""Per-decision simulator: PIT integrity, honesty, and journal fixtures."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import terminal_product_api_parallel as api
from product.decision_journal import persist
from product.decision_simulator import (
    AMBIGUOUS_HISTORICAL_DECISION,
    ENTRY_CLOSE_AT_T,
    ENTRY_PERSISTED,
    ENTRY_UNAVAILABLE,
    HISTORICAL_DECISION_UNAVAILABLE,
    NOT_ENTERED,
    PIT_INTEGRITY_FAILED,
    SUCCEEDED,
    UNAVAILABLE,
    simulate_past_decision,
)
from product.forward_evidence import BACKTEST


def _bars(end: str, n: int = 80, *, future_days: int = 8, start: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp(end) + pd.Timedelta(days=future_days), periods=n + future_days)
    close = [start + i * 0.5 for i in range(len(idx))]
    return pd.DataFrame(
        {
            "open": close,
            "high": [c + 2 for c in close],
            "low": [c - 2 for c in close],
            "close": close,
            "volume": [250_000] * len(idx),
        },
        index=idx,
    )


def _journal(tmp_path, monkeypatch, **over):
    monkeypatch.setattr("product.decision_journal.DB_PATH", tmp_path / "decisions.db")
    monkeypatch.setattr("product.decision_journal.JSONL_PATH", tmp_path / "decisions.jsonl")
    monkeypatch.setenv("QT_PAPER_AUTOPILOT_JOURNAL", str(tmp_path / "no-journal.json"))
    monkeypatch.setenv("QT_HISTORICAL_REPLAY_DIR", str(tmp_path / "replay"))
    row = {
        "decision_id": "2025-07-15|RELIANCE|WAIT|ENTRY_TOO_EXTENDED|fixture",
        "symbol": "RELIANCE",
        "decision": "WAIT",
        "decision_time": "2025-07-15T10:00:00+00:00",
        "market_as_of": "2025-07-15",
        "evidence_cutoff": "2025-07-15",
        "reason_code": "ENTRY_TOO_EXTENDED",
        "reason": "Price extended versus the defined entry.",
        "tier": "good_setup",
        "entry": 100.0,
        "stop": 94.0,
        "target": 112.0,
        "methods_buy": ["Tape", "Funds"],
        "evidence_family_votes": {"PRICE_STRUCTURE": True, "FUNDAMENTALS": True},
    }
    row.update(over)
    return persist(row, path=tmp_path / "decisions.db")


def _sim(tmp_path, monkeypatch, frame=None, **kwargs):
    _journal(tmp_path, monkeypatch)
    bars = frame if frame is not None else _bars("2025-07-15")
    kwargs.setdefault("replay_engine", False)
    return simulate_past_decision(
        symbol="RELIANCE",
        as_of="2025-07-15",
        ohlcv_fn=lambda _s: bars,
        **kwargs,
    )


def test_historical_decision_loads_from_journal(tmp_path, monkeypatch):
    out = _sim(tmp_path, monkeypatch, alternative="BUY")
    assert out["status"] == SUCCEEDED
    assert out["available"] is True
    assert out["symbol"] == "RELIANCE"
    assert out["as_of"] == "2025-07-15"
    assert out["original"]["action"] == "WAIT"
    assert out["original"]["reason_code"] == "ENTRY_TOO_EXTENDED"
    assert out["original"]["entry"] == 100.0
    assert out["original"]["entry_source"] == ENTRY_PERSISTED
    assert out["simulated"]["entry_source"] == ENTRY_PERSISTED
    assert out["provenance"] == BACKTEST
    assert out["live_locked"] is True


def test_counterfactual_buy_executes_on_subsequent_bars(tmp_path, monkeypatch):
    out = _sim(tmp_path, monkeypatch, alternative="BUY")
    assert out["simulated"]["action"] == "BUY"
    sim = out["subsequent_outcome"]["simulated"]
    assert sim["status"] == "COMPUTED"
    assert sim["simulated_entry"] == 100.0
    assert sim["mfe_pct"] not in {None, UNAVAILABLE}
    assert sim["mae_pct"] not in {None, UNAVAILABLE}
    assert sim["hypothetical_return_pct"] not in {None, UNAVAILABLE}
    assert float(sim["bars_used"]) >= 1
    actual = out["subsequent_outcome"]["actual"]
    assert actual["status"] == NOT_ENTERED
    assert actual["hypothetical_return_pct"] == UNAVAILABLE


def test_future_bars_cannot_enter_decision_time_features(tmp_path, monkeypatch):
    frame = _bars("2025-07-15", future_days=12)
    seen: list[str] = []

    def analyzer(symbol, bars):
        last = str(bars.index[-1].date())
        seen.append(last)
        if last > "2025-07-15":
            raise AssertionError(f"lookahead bar {last}")
        return None

    out = _sim(
        tmp_path,
        monkeypatch,
        frame=frame,
        alternative="BUY",
        replay_engine=True,
        analyzer=analyzer,
        decide_fn=lambda card, **_k: type("D", (), {"as_dict": lambda self: {
            "symbol": "RELIANCE", "decision": "WAIT", "reason_code": "ENTRY_TOO_EXTENDED",
            "entry": 100, "stop": 94, "target": 112,
        }})(),
    )
    assert out["evidence_at_t"]["future_bars_used_for_decision"] is False
    assert out["evidence_at_t"]["max_bar_date"] <= "2025-07-15"
    assert all(day <= "2025-07-15" for day in seen)
    later = out["subsequent_outcome"]["simulated"]
    assert later["status"] == "COMPUTED"
    assert later["bars_used"] >= 1


def test_post_t_financials_and_news_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "product.pit_query.get_financial_snapshot",
        lambda symbol, as_of, path=None: {
            "available": True,
            "latest_publication": "2025-08-01",
            "roe": 99.0,
            "numbers_parsed": True,
        },
    )
    monkeypatch.setattr(
        "product.pit_query.get_research_snapshot",
        lambda symbol, as_of, path=None: {"available": True, "latest_publication": "2025-07-01"},
    )
    monkeypatch.setattr(
        "product.pit_events.get_events",
        lambda symbol, as_of, path=None, limit=40: [
            {"headline": "Future leak", "available_from": "2025-08-01", "event_class": "RESULTS", "source": "NSE"},
            {"headline": "Known at T", "available_from": "2025-07-10", "event_class": "RESULTS", "source": "NSE"},
        ],
    )
    out = _sim(tmp_path, monkeypatch, alternative="BUY")
    fin = out["evidence_at_t"]["financials"]
    assert fin["available"] is False
    assert fin["status"] == UNAVAILABLE
    assert "after T" in str(fin.get("reason") or fin.get("note") or "")
    headlines = [row.get("headline") for row in out["evidence_at_t"]["news"]]
    assert "Future leak" not in headlines
    assert headlines == ["Known at T"]


def test_missing_journal_does_not_invent_a_decision(tmp_path, monkeypatch):
    monkeypatch.setattr("product.decision_journal.DB_PATH", tmp_path / "empty.db")
    monkeypatch.setattr("product.decision_journal.JSONL_PATH", tmp_path / "empty.jsonl")
    monkeypatch.setenv("QT_PAPER_AUTOPILOT_JOURNAL", str(tmp_path / "no-journal.json"))
    monkeypatch.setenv("QT_HISTORICAL_REPLAY_DIR", str(tmp_path / "replay"))
    out = simulate_past_decision(
        symbol="RELIANCE",
        as_of="2025-07-15",
        alternative="BUY",
        replay_engine=False,
        ohlcv_fn=lambda _s: _bars("2025-07-15"),
    )
    assert out["status"] == HISTORICAL_DECISION_UNAVAILABLE
    assert out["available"] is False
    assert out["original"]["action"] == UNAVAILABLE
    assert out["simulated"]["action"] == UNAVAILABLE
    assert out["comparison"]["return_delta_pct"] == UNAVAILABLE
    blob = json.dumps(out["subsequent_outcome"]).lower()
    assert "invented" in blob or "not invented" in blob
    assert out["original"]["entry"] == UNAVAILABLE
    assert out["original"]["entry_source"] == ENTRY_UNAVAILABLE
    assert out["error"]


def test_subsequent_prices_are_used_only_for_outcome(tmp_path, monkeypatch):
    frame = _bars("2025-07-15", future_days=10, start=100)
    out = _sim(tmp_path, monkeypatch, frame=frame, alternative="BUY")
    assert out["evidence_at_t"]["close"] not in {None, UNAVAILABLE}
    assert out["evidence_at_t"]["max_bar_date"] == "2025-07-15"
    sim = out["subsequent_outcome"]["simulated"]
    assert sim["status"] == "COMPUTED"
    assert "after T" in sim["methodology"]


def test_api_contract_single_decision(tmp_path, monkeypatch):
    _journal(tmp_path, monkeypatch)
    frame = _bars("2025-07-15")
    monkeypatch.setattr(
        "product.decision_simulator.simulate_past_decision",
        lambda **kwargs: simulate_past_decision(
            ohlcv_fn=lambda _s: frame,
            replay_engine=False,
            **{k: v for k, v in kwargs.items() if k not in {"ohlcv_fn", "replay_engine"}},
        ),
    )
    payload = api.decision_simulator_get(symbol="RELIANCE", as_of="2025-07-15", alternative="BUY")
    assert payload["kind"] == "PAST_DECISION_SIMULATION"
    assert payload["original"]["action"] == "WAIT"
    assert payload["simulated"]["action"] == "BUY"
    posted = api.decision_simulator_run(symbol="RELIANCE", as_of="2025-07-15", alternative="BUY")
    assert posted["fingerprint"] == payload["fingerprint"]


def test_simulation_failure_is_honest(tmp_path, monkeypatch):
    out = simulate_past_decision(symbol="", as_of="", alternative="BUY")
    assert out["status"] == "FAILED"
    assert out["available"] is False
    assert out["error"]
    assert out["original"]["action"] == UNAVAILABLE
    bad = simulate_past_decision(symbol="RELIANCE", as_of="2025-07-15", alternative="HOPE")
    assert bad["status"] == "FAILED"
    assert "Unknown alternative" in str(bad["error"])


def test_repeated_simulation_is_deterministic(tmp_path, monkeypatch):
    frame = _bars("2025-07-15")
    first = _sim(tmp_path, monkeypatch, frame=frame, alternative="BUY")
    second = simulate_past_decision(
        symbol="RELIANCE",
        as_of="2025-07-15",
        alternative="BUY",
        replay_engine=False,
        ohlcv_fn=lambda _s: frame,
    )
    assert first["fingerprint"] == second["fingerprint"]
    assert first["comparison"] == second["comparison"]
    assert first["subsequent_outcome"]["simulated"]["mfe_pct"] == second["subsequent_outcome"]["simulated"]["mfe_pct"]


def test_journal_fixture_smoke_reliance_july(tmp_path, monkeypatch):
    persisted = _journal(tmp_path, monkeypatch)
    assert persisted["symbol"] == "RELIANCE"
    assert persisted["market_as_of"] == "2025-07-15"
    out = simulate_past_decision(
        symbol="RELIANCE",
        as_of="2025-07-15",
        alternative="BUY",
        replay_engine=False,
        ohlcv_fn=lambda _s: _bars("2025-07-15", start=100),
    )
    assert out["status"] == SUCCEEDED
    assert out["original"]["action"] == "WAIT"
    assert out["simulated"]["action"] == "BUY"
    assert out["evidence_at_t"]["future_bars_used_for_decision"] is False
    assert out["pit_status"] == "PIT_OK"
    assert datetime.fromisoformat(out["generated_at"].replace("Z", "+00:00")).tzinfo == timezone.utc


def test_missing_persisted_entry_is_not_original_close(tmp_path, monkeypatch):
    _journal(tmp_path, monkeypatch, entry=None, hypothetical_entry=None)
    out = simulate_past_decision(
        symbol="RELIANCE",
        as_of="2025-07-15",
        alternative="BUY",
        replay_engine=False,
        ohlcv_fn=lambda _s: _bars("2025-07-15", start=100),
    )
    assert out["status"] == SUCCEEDED
    assert out["original"]["action"] == "WAIT"
    assert out["original"]["entry"] == UNAVAILABLE
    assert out["original"]["entry_source"] == ENTRY_UNAVAILABLE
    assert out["simulated"]["entry_source"] == ENTRY_CLOSE_AT_T
    assert out["simulated"]["entry"] not in {None, UNAVAILABLE}
    assert out["subsequent_outcome"]["simulated"]["entry_source"] == ENTRY_CLOSE_AT_T
    assert "not a QuantTerm-recorded entry" in " ".join(out["warnings"])


def test_decision_id_binds_to_symbol_and_date(tmp_path, monkeypatch):
    row = _journal(tmp_path, monkeypatch)
    did = row["decision_id"]
    ok = simulate_past_decision(
        symbol="RELIANCE",
        as_of="2025-07-15",
        decision_id=did,
        alternative="BUY",
        replay_engine=False,
        ohlcv_fn=lambda _s: _bars("2025-07-15"),
    )
    assert ok["status"] == SUCCEEDED
    assert ok["decision_id"] == did
    wrong_symbol = simulate_past_decision(
        symbol="TCS",
        as_of="2025-07-15",
        decision_id=did,
        alternative="BUY",
        replay_engine=False,
        ohlcv_fn=lambda _s: _bars("2025-07-15"),
    )
    assert wrong_symbol["status"] == "FAILED"
    assert "identity mismatch" in str(wrong_symbol["error"])
    assert "RELIANCE" in str(wrong_symbol["error"])
    assert wrong_symbol["original"]["entry"] == UNAVAILABLE
    wrong_date = simulate_past_decision(
        symbol="RELIANCE",
        as_of="2025-01-01",
        decision_id=did,
        alternative="BUY",
        replay_engine=False,
        ohlcv_fn=lambda _s: _bars("2025-07-15"),
    )
    assert wrong_date["status"] == "FAILED"
    assert "identity mismatch" in str(wrong_date["error"])
    assert "2025-07-15" in str(wrong_date["error"])


def test_ambiguous_same_day_requires_decision_id(tmp_path, monkeypatch):
    _journal(tmp_path, monkeypatch)
    persist({
        "decision_id": "2025-07-15|RELIANCE|AVOID|LOW_QUALITY_SETUP|fixture-2",
        "symbol": "RELIANCE",
        "decision": "AVOID",
        "decision_time": "2025-07-15T11:00:00+00:00",
        "market_as_of": "2025-07-15",
        "reason_code": "LOW_QUALITY_SETUP",
        "entry": 101.0,
        "stop": 94.0,
        "target": 112.0,
    }, path=tmp_path / "decisions.db")
    ambiguous = simulate_past_decision(
        symbol="RELIANCE",
        as_of="2025-07-15",
        alternative="BUY",
        replay_engine=False,
        ohlcv_fn=lambda _s: _bars("2025-07-15"),
    )
    assert ambiguous["status"] == AMBIGUOUS_HISTORICAL_DECISION
    ids = {row["decision_id"] for row in ambiguous["matches"]}
    assert "2025-07-15|RELIANCE|WAIT|ENTRY_TOO_EXTENDED|fixture" in ids
    assert "2025-07-15|RELIANCE|AVOID|LOW_QUALITY_SETUP|fixture-2" in ids
    chosen = simulate_past_decision(
        symbol="RELIANCE",
        as_of="2025-07-15",
        decision_id="2025-07-15|RELIANCE|AVOID|LOW_QUALITY_SETUP|fixture-2",
        alternative="BUY",
        replay_engine=False,
        ohlcv_fn=lambda _s: _bars("2025-07-15"),
    )
    assert chosen["status"] == SUCCEEDED
    assert chosen["original"]["action"] == "AVOID"
    assert chosen["decision_id"] == "2025-07-15|RELIANCE|AVOID|LOW_QUALITY_SETUP|fixture-2"


def test_lookahead_replay_cannot_report_clean_success(tmp_path, monkeypatch):
    _journal(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "product.decision_simulator._replay_at_t",
        lambda *_a, **_k: {
            "status": SUCCEEDED,
            "decision": "WAIT",
            "reason": "reconstructed",
            "pit": {"future_evidence_used": True, "max_bar_date": "2025-08-01"},
        },
    )
    out = simulate_past_decision(
        symbol="RELIANCE",
        as_of="2025-07-15",
        alternative="BUY",
        replay_engine=True,
        ohlcv_fn=lambda _s: _bars("2025-07-15"),
    )
    assert out["status"] == PIT_INTEGRITY_FAILED
    assert out["status"] != SUCCEEDED
    assert out["available"] is False
    assert out["counterfactual_trustworthy"] is False
    assert out["pit_status"] == PIT_INTEGRITY_FAILED
    assert out["evidence_at_t"]["future_bars_used_for_decision"] is True
    assert out["evidence_at_t"]["pit_status"] == PIT_INTEGRITY_FAILED
    assert out["evidence_at_t"]["reconstructed_engine_decision"]["pit"]["future_evidence_used"] is True
    assert out["subsequent_outcome"]["simulated"]["status"] != "COMPUTED"
    assert out["subsequent_outcome"]["simulated"]["trustworthy"] is False
    api_payload = api.decision_simulator_get(symbol="RELIANCE", as_of="2025-07-15", alternative="BUY")
    assert api_payload["status"] != SUCCEEDED
    assert api_payload["status"] == PIT_INTEGRITY_FAILED
