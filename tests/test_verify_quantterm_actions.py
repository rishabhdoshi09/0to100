from __future__ import annotations

import scripts.verify_quantterm_actions as verify


def test_default_action_verifier_never_contains_live_money_controls():
    controls = {control for _label, control in verify.SAFE_CONTROLS}
    assert controls == {
        "RUN_SCAN_NOW",
        "REFRESH_LONG_TERM_NOW",
        "REFRESH_NEWS_NOW",
        "REFRESH_MARKET_REPORT_NOW",
    }
    assert all("BUY" not in control and "SELL" not in control and "UNLOCK" not in control for control in controls)


def test_watchlist_roundtrip_deletes_only_the_row_it_created(monkeypatch):
    calls: list[tuple[str, str, dict | None]] = []
    rows = [{"id": 7, "symbol": "EXISTING"}]

    def fake_request(url: str, *, method: str = "GET", timeout: float = 10.0, body=None):
        calls.append((url, method, body))
        if method == "POST":
            rows.append({"id": 99, "symbol": body["symbol"]})
            return {"accepted": True, "item": {"id": 99, "symbol": body["symbol"]}}
        if method == "DELETE":
            rows[:] = [row for row in rows if row["id"] != 99]
            return {"accepted": True, "removed_id": 99}
        return {"items": list(rows), "count": len(rows)}

    monkeypatch.setattr(verify, "_request_json", fake_request)
    verify._verify_watchlist_roundtrip("http://127.0.0.1:8765", "TCS", 1.0)

    assert rows == [{"id": 7, "symbol": "EXISTING"}]
    assert any(method == "POST" and body and body["symbol"] == "TCS" for _url, method, body in calls)
    assert any(method == "DELETE" and url.endswith("/api/watchlist/99") for url, method, _body in calls)


def test_action_verifier_can_skip_optional_broker_and_data_refreshes():
    parser = verify.build_parser()
    args = parser.parse_args([])
    assert args.include_fno is False
    assert args.include_data_refresh is False
    assert args.skip_watchlist_write is False
