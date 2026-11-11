"""Per-trade win probability: Beta posterior over replay, tracked signals, forward paper."""
from __future__ import annotations

from datetime import date

from product.win_probability import estimate_win_probability, overlay_cards


def test_no_evidence_is_the_prior_and_says_so():
    out = estimate_win_probability()
    assert out["p_win"] == 0.5 and out["label"] == "UNPROVEN" and "prior only" in out["text"]
    assert out["low_80"] < 0.5 < out["high_80"]


def test_replay_and_decayed_live_signals_combine():
    out = estimate_win_probability(
        historical={"positive_rate": 0.3687, "effective_n": 48.53},
        live={"p_win": 24.4, "ev_n_eff": 1174, "ev_last_outcome": "2026-07-23"},
        today=date(2026, 11, 11))
    assert 0.24 < out["p_win"] < 0.37 and out["label"] == "ESTIMATED"
    live = [s for s in out["sources"] if s["source"] == "LIVE_SIGNALS"][0]
    assert live["weight"] < 0.25                      # 111 days old: decayed
    assert out["low_80"] < out["p_win"] < out["high_80"]


def test_large_samples_are_capped_so_one_source_cannot_dominate():
    out = estimate_win_probability(live={"p_win": 90, "ev_n": 100000, "ev_last_outcome": "2026-11-10"},
                                   today=date(2026, 11, 11))
    assert out["sources"][0]["pseudo_n"] <= 80


def test_forward_paper_is_the_only_path_to_measured():
    out = estimate_win_probability(forward={"n": 40, "wins": 22})
    assert out["label"] == "MEASURED" and abs(out["p_win"] - 24 / 44) < 1e-3
    assert estimate_win_probability(forward={"n": 10, "wins": 9})["label"] == "THIN"
    assert estimate_win_probability(forward={"n": 10, "wins": 12})["label"] == "UNPROVEN"   # invalid


def test_overlay_joins_replay_evidence_by_symbol(monkeypatch):
    import product.decision_discovery_store as store
    import product.win_probability as wp

    wp._board_cache.update({"at": 0.0, "rows": {}})
    monkeypatch.setattr(store, "load_current", lambda: {"decisions": [
        {"symbol": "EMAMILTD", "evidence_intelligence": {"positive_rate": 0.37, "effective_n": 48.5}}]})
    payload = {"categories": [{"cards": [
        {"symbol": "emamiltd", "win_inputs": {}}, {"symbol": "RPEL", "win_probability": {"keep": 1}}]}]}
    out = overlay_cards(payload)
    cards = out["categories"][0]["cards"]
    assert "historical replay 37%" in cards[0]["win_probability"]["text"]
    assert cards[1]["win_probability"] == {"keep": 1}
