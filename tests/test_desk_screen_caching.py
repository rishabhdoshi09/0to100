"""The slow desk screens are served from the projection cache (2026-11-16)."""
from __future__ import annotations


def test_decision_board_builds_once_until_a_new_scan_lands(monkeypatch, tmp_path):
    import terminal_product_api as T
    import product.recommendations_store as RS

    reco = tmp_path / "latest_recommendations.json"
    reco.write_text("{}")
    monkeypatch.setattr(RS, "DEFAULT_RECO_PATH", reco)
    monkeypatch.setattr(T, "_market_regime", lambda: "WEAK")
    calls = []
    monkeypatch.setattr(T, "_build_decision_board", lambda limit: calls.append(limit) or {"decisions": [], "limit": limit})
    T.DECISION_BOARD_CACHE.invalidate()
    a = T.decisions_board(limit=40)
    b = T.decisions_board(limit=40)
    assert calls == [40] and a["projection_meta"]["status"] == "MISS" and b["projection_meta"]["status"] == "HIT"
    reco.write_text('{"scan_scanned_at": "new"}')
    stale = T.decisions_board(limit=40)
    assert stale["projection_meta"]["inputs_changed"] is True
    T.DECISION_BOARD_CACHE.invalidate()


def test_autonomous_learning_caches_the_dashboard_but_never_live_safety(monkeypatch):
    import api.runtime as R
    import product.autonomous_learning as AL

    calls = []
    monkeypatch.setattr(AL, "dashboard", lambda: calls.append(1) or {"mode": "PAPER"})
    live = {"n": 0}

    def safety():
        live["n"] += 1
        return {"live_money_locked": True, "safety_reads": live["n"]}

    monkeypatch.setattr(R, "live_safety_projection", safety)
    R.AUTONOMOUS_LEARNING_CACHE.invalidate()
    first = R.autonomous_learning_get()
    second = R.autonomous_learning_get()
    assert calls == [1]
    assert first["safety_reads"] == 1 and second["safety_reads"] == 2
    R.AUTONOMOUS_LEARNING_CACHE.invalidate()


def test_changing_an_autonomous_learning_control_is_visible_at_once(monkeypatch):
    import api.runtime as R
    import product.autonomous_learning as AL

    state = {"mode": "PAPER"}
    monkeypatch.setattr(AL, "dashboard", lambda: dict(state))
    monkeypatch.setattr(AL, "set_control", lambda **kw: state.update(mode=kw.get("mode", state["mode"])) or dict(state))
    monkeypatch.setattr(R, "live_safety_projection", lambda: {})
    R.AUTONOMOUS_LEARNING_CACHE.invalidate()
    assert R.autonomous_learning_get()["mode"] == "PAPER"
    R.autonomous_learning_set(mode="observe")
    assert R.autonomous_learning_get()["mode"] == "OBSERVE"
    R.AUTONOMOUS_LEARNING_CACHE.invalidate()


def test_typeahead_index_is_built_once_per_source_change(monkeypatch, tmp_path):
    import product.due_diligence.suggest as S
    import product.scan_store as SS
    import product.long_term_store as LT

    scan = tmp_path / "scan.json"
    scan.write_text("{}")
    monkeypatch.setattr(SS, "DEFAULT_SCAN_PATH", scan)
    monkeypatch.setattr(LT, "DEFAULT_LONG_TERM_PATH", tmp_path / "lt.json")
    builds = []
    monkeypatch.setattr(S, "_build_universe_names", lambda: builds.append(1) or {"ICICIBANK": "ICICI Bank Ltd"})
    monkeypatch.setattr(S, "_INDEX", {"key": None, "at": 0.0, "names": None})
    assert S.suggest_tickers("ICICI")[0]["symbol"] == "ICICIBANK"
    S.suggest_tickers("ICIC")
    S.suggest_tickers("ICI")
    assert builds == [1]
    scan.write_text('{"changed": 1}')
    S.suggest_tickers("ICICI")
    assert builds == [1, 1]
