"""A down provider must not freeze the desk or erase saved research."""
from __future__ import annotations

from pathlib import Path

from product.due_diligence import acquire as AQ


def test_provider_failure_keeps_cached_facts(tmp_path: Path, monkeypatch):
    previous = {
        "symbol": "INFY",
        "kpis": {"roe": {"current": 18.0, "source": "cache"}},
        "dataset_meta": {"quarterly_results": {"status": "current"}},
    }
    monkeypatch.setattr(AQ, "load_autonomy_facts", lambda symbol: dict(previous))
    saved: dict = {}

    def save(symbol, payload):
        saved.update(payload)
        return tmp_path / f"{symbol}.json"

    monkeypatch.setattr(AQ, "save_autonomy_facts", save)
    monkeypatch.setattr(
        "reporting.evidence_intake.load_raw_fundamentals",
        lambda symbol: {"data": {}, "fetched_at": ""},
    )
    monkeypatch.setattr(AQ, "extract_kpis_from_raw", lambda raw: {})
    monkeypatch.setattr(AQ, "extract_from_uploads", lambda symbol: {"kpis": {}})
    monkeypatch.setattr(AQ, "_framework_for", lambda symbol, raw: {"lending": False, "kpis": []})
    monkeypatch.setattr(AQ, "plan_acquire", lambda symbol, **k: {
        "to_fetch": ["exchange_filings"],
        "lanes": {"nse_filings": True, "nse_annual": False, "option_chain": False, "screener": False},
        "coverage": {},
        "force": False,
    })

    def boom_session():
        raise RuntimeError("nse down")

    monkeypatch.setattr(AQ, "_nse_session", boom_session)
    monkeypatch.setattr(AQ, "_framework_id_for", lambda symbol, raw: "generic")
    out = AQ.acquire_symbol("INFY")
    assert out["kpis"]["roe"]["current"] == 18.0
    assert saved["kpis"]["roe"]["current"] == 18.0
