"""Home · BUY only (owner request 2026-11-15): every canonical BUY, nothing else."""
from __future__ import annotations

import json

import product.home_os as HO


def _index(**over):
    base = {"by_symbol": {}, "withheld": [], "withheld_count": 0, "scanner_buys": 0,
            "session": "2026-11-13", "scanned_at": "2026-11-15T17:23:00"}
    base.update(over)
    return base


def test_buy_rows_carry_levels_reward_and_the_measured_edge(monkeypatch):
    monkeypatch.setattr(HO, "_scan_evidence_index", lambda: _index(by_symbol={
        "INFY": {"edge_r": 0.12, "edge_lower_r": 0.03, "edge_source": "combination",
                 "signals": ["MOMENTUM", "VOL_SQUEEZE"], "score": 81, "verdict": "BUY"}}))
    out = HO.build_buy_only([{"symbol": "INFY", "decision": "BUY", "cmp": 1500, "entry": 1505,
                              "stop": 1450, "target": 1615}], verified=True)
    assert out["count"] == 1 and out["empty_reason"] == ""
    row = out["rows"][0]
    assert row["symbol"] == "INFY" and row["reward_r"] == 2.0
    assert row["edge_r"] == 0.12 and row["edge_lower_r"] == 0.03 and row["edge_source"] == "combination"
    assert "+0.05R" in out["rule"] and "Paper only" in out["rule"]


def test_a_committee_buy_the_scan_no_longer_certifies_is_not_listed(monkeypatch):
    """GPT cross-review C-8: load_scan() withdrew INFY's BUY (WATCH) after the board was built."""
    monkeypatch.setattr(HO, "_scan_evidence_index", lambda: _index(by_symbol={
        "INFY": {"edge_r": 0.12, "edge_lower_r": 0.03, "signals": ["MOMENTUM"], "score": 81, "verdict": "WATCH"}}))
    out = HO.build_buy_only([{"symbol": "INFY", "decision": "BUY", "cmp": 1500, "entry": 1505,
                              "stop": 1450, "target": 1615}], verified=True)
    assert out["rows"] == [] and out["admission_state"] == "NO_ELIGIBLE_BUY"


def test_an_empty_day_says_why_and_never_lists_near_misses(monkeypatch):
    monkeypatch.setattr(HO, "_scan_evidence_index", lambda: _index(
        withheld_count=19, withheld=[{"symbol": "APOLLO", "edge_r": 0.003, "score": 68.8}]))
    out = HO.build_buy_only([], verified=True)
    assert out["rows"] == [] and out["withheld_count"] == 19
    assert "19 setup(s)" in out["empty_reason"] and "APOLLO at +0.00R" in out["empty_reason"]
    unverified = HO.build_buy_only([], verified=False)
    assert unverified["empty_reason"].startswith("Verifying")


def test_buy_only_is_not_capped_at_homes_top_five(monkeypatch):
    import product.decision_simulation_gate as gate

    rows = [{"symbol": f"S{i}", "discovery_decision": "ENTER_NOW"} for i in range(8)]
    monkeypatch.setattr(gate, "status", lambda: {"scan_fresh": True, "discovery_ready": True,
                                                 "scan_scanned_at": "", "best_trades": rows})
    assert len(HO._canonical_best_trade_rows(history_current=True)) == 5
    assert len(HO._canonical_best_trade_rows(history_current=True, limit=HO.BUY_ONLY_LIMIT)) == 8


def test_scan_is_parsed_once_per_scan_file(tmp_path, monkeypatch):
    import product.scan_store as SS

    path = tmp_path / "scan.json"
    path.write_text("{}")
    calls = []
    payload = {"records": [
        {"symbol": "BUYME", "verdict": "BUY", "edge_r": 0.2, "score": 90},
        {"symbol": "HELD", "verdict": "WATCH", "edge_gated": True, "edge_r": -0.1, "score": 80},
        {"symbol": "PLAIN", "verdict": "WATCH", "score": 99},
    ], "provenance": {"market_session_date": "2026-11-13"}, "scanned_at": "x"}
    monkeypatch.setattr(SS, "DEFAULT_SCAN_PATH", path)
    monkeypatch.setattr(SS, "load_scan", lambda *a, **k: calls.append(1) or payload)
    monkeypatch.setattr(HO, "_SCAN_EVIDENCE_CACHE", {"key": None, "value": None})
    first = HO._scan_evidence_index()
    assert HO._scan_evidence_index() is first and calls == [1]
    assert first["scanner_buys"] == 1 and first["withheld_count"] == 1
    assert [w["symbol"] for w in first["withheld"]] == ["HELD"] and first["session"] == "2026-11-13"
    path.write_text(json.dumps({"changed": True}))
    HO._scan_evidence_index()
    assert calls == [1, 1]


def test_closest_withheld_means_best_measured_edge_not_best_score(tmp_path, monkeypatch):
    import product.scan_store as SS

    path = tmp_path / "scan.json"
    path.write_text("{}")
    payload = {"records": [
        {"symbol": "DCBBANK", "verdict": "WATCH", "edge_gated": True, "edge_r": -0.05, "score": 77.5},
        {"symbol": "APOLLO", "verdict": "WATCH", "edge_gated": True, "edge_r": 0.003, "score": 68.8},
        {"symbol": "NOEDGE", "verdict": "WATCH", "edge_gated": True, "edge_r": None, "score": 99},
    ], "provenance": {}, "scanned_at": "x"}
    monkeypatch.setattr(SS, "DEFAULT_SCAN_PATH", path)
    monkeypatch.setattr(SS, "load_scan", lambda *a, **k: payload)
    monkeypatch.setattr(HO, "_SCAN_EVIDENCE_CACHE", {"key": None, "value": None})
    assert [w["symbol"] for w in HO._scan_evidence_index()["withheld"]] == ["APOLLO", "DCBBANK", "NOEDGE"]


def test_no_eligible_buy_says_why_when_the_evidence_itself_is_the_blocker(monkeypatch):
    import product.home_os as H
    monkeypatch.setattr(H, "_scan_evidence_index", lambda: {
        "by_symbol": {}, "withheld": [], "withheld_count": 4, "scanner_buys": 0, "session": "2026-11-19",
        "scanned_at": "", "read_time_withheld": 0,
        "evidence_note": "the annual alpha budget was exhausted at look 53 of 52"})
    out = H.build_buy_only([], verified=True)
    assert out["admission_state"] == "NO_ELIGIBLE_BUY" and out["count"] == 0
    assert "alpha budget was exhausted" in out["empty_reason"] and "WATCH research only" in out["empty_reason"]
    assert "not a win probability" in out["rule"] and "family-wise" in out["rule"]


def test_the_home_evidence_cache_turns_over_when_the_authority_changes(tmp_path, monkeypatch):
    import product.home_os as H
    import product.scan_store as SS
    path = SS.save_scan({"schema_version": 2, "records": [], "evidence_identity": {"id": "x"}}, tmp_path / "s.json")
    monkeypatch.setattr(SS, "DEFAULT_SCAN_PATH", path)
    authority = {"k": 1}
    monkeypatch.setattr(SS, "evidence_authority_key", lambda: (("pointer", authority["k"]),))
    calls = {"n": 0}
    real = SS.load_scan

    def counting(*a, **k):
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(SS, "load_scan", counting)
    H._SCAN_EVIDENCE_CACHE.update(key=None, value=None)
    H._scan_evidence_index()
    H._scan_evidence_index()
    assert calls["n"] == 1                                             # cached while nothing changed
    authority["k"] = 2
    H._scan_evidence_index()
    assert calls["n"] == 2                                             # authority moved: re-read

