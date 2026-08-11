"""Simulations for the yielding multi-source fundamentals resolver.

These run WITHOUT the network. Injected providers fail/succeed in controlled
order so we prove: every step yields, Screener→Yahoo fallback works, and
exhaustion stays honest (no invented numbers).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import fundamentals.cache as cache_mod
import fundamentals.resolver as R


def _rich(**overrides):
    base = {
        "about": "Industrial systems company",
        "key_ratios": [{"name": "P/E", "value": "22"}, {"name": "ROE", "value": "18"}],
        "profit_loss": [{"": "Sales", "Mar 2025": 100, "Mar 2026": 120}],
        "quarterly_results": [{"": "Sales", "Jun 2025": 30, "Sep 2025": 32}],
        "balance_sheet": [{"": "Debt", "Mar 2026": 40}],
        "cash_flow": [{"": "CFO", "Mar 2026": 25}],
        "shareholding": [{"": "FIIs", "Mar 2026": 9.4}],
        "peer_comparison": [],
        "_source": "sim",
    }
    base.update(overrides)
    return base


@pytest.fixture()
def isolated_cache(tmp_path: Path, monkeypatch):
    db = tmp_path / "fundamentals.db"
    monkeypatch.setattr(cache_mod, "_DB_PATH", db)
    return cache_mod.FundamentalsCache()


def test_simulation_every_step_yields_on_screener_success(isolated_cache):
    def boom(_):
        raise AssertionError("yahoo must not run when screener succeeds")

    providers = {
        "screener_in": lambda s: _rich(_source="screener_in"),
        "yahoo_finance": boom,
        "user_uploads": boom,
    }
    steps = list(R.iter_resolve("SIMCO", force_refresh=True, write_cache=True, providers=providers))
    assert steps[0].status == "TRYING" and steps[0].source == "screener_in"
    assert any(s.source == "screener_in" and s.status == "OK" for s in steps)
    assert all(s.source != "yahoo_finance" for s in steps)
    assert any(s.status == "TRYING" for s in steps)


def test_simulation_screener_fail_falls_back_to_yahoo(isolated_cache):
    calls = []

    def screener(_):
        calls.append("screener")
        raise RuntimeError("403 blocked")

    def yahoo(_):
        calls.append("yahoo")
        return _rich(_source="yahoo_finance")

    providers = {
        "screener_in": screener,
        "yahoo_finance": yahoo,
        "user_uploads": lambda s: (_ for _ in ()).throw(RuntimeError("no uploads")),
    }
    data, steps = R.resolve("SIMCO", force_refresh=True, write_cache=True, providers=providers)
    assert calls == ["screener", "yahoo"]
    assert data is not None
    assert data["_source"] == "yahoo_finance"
    statuses = [(s["source"], s["status"]) for s in steps]
    assert ("screener_in", "TRYING") in statuses
    assert ("screener_in", "ERROR") in statuses
    assert ("yahoo_finance", "TRYING") in statuses
    assert ("yahoo_finance", "OK") in statuses


def test_simulation_all_remote_fail_uses_stale_cache(isolated_cache, monkeypatch):
    isolated_cache.set("SIMCO", _rich(_source="stale_seed"))
    monkeypatch.setattr(isolated_cache, "get", lambda symbol: None)

    # iter_resolve constructs its own FundamentalsCache(); patch the class methods
    monkeypatch.setattr(cache_mod.FundamentalsCache, "get", lambda self, symbol: None)
    monkeypatch.setattr(
        cache_mod.FundamentalsCache,
        "get_any",
        lambda self, symbol: _rich(_source="stale_seed"),
    )

    providers = {
        "screener_in": lambda s: (_ for _ in ()).throw(RuntimeError("down")),
        "yahoo_finance": lambda s: (_ for _ in ()).throw(RuntimeError("down")),
        "user_uploads": lambda s: (_ for _ in ()).throw(RuntimeError("none")),
    }
    data, steps = R.resolve("SIMCO", force_refresh=False, write_cache=False, providers=providers)
    assert data is not None
    assert any(s["source"] == "local_cache_stale" and s["status"] == "OK" for s in steps)


def test_simulation_exhausted_yields_honest_missing(isolated_cache):
    providers = {
        "screener_in": lambda s: (_ for _ in ()).throw(RuntimeError("403")),
        "yahoo_finance": lambda s: (_ for _ in ()).throw(RuntimeError("timeout")),
        "user_uploads": lambda s: (_ for _ in ()).throw(RuntimeError("none")),
    }
    data, steps = R.resolve("NOSUCH", force_refresh=True, write_cache=False, providers=providers)
    assert data is None
    assert steps[-1]["status"] == "EXHAUSTED"
    by_source: dict[str, list[str]] = {}
    for s in steps:
        by_source.setdefault(s["source"], []).append(s["status"])
    for source in ("screener_in", "yahoo_finance", "user_uploads"):
        assert "TRYING" in by_source[source]


def test_simulation_user_uploads_rescue_after_remote_fail(tmp_path, monkeypatch, isolated_cache):
    import reporting.evidence_intake as EI

    monkeypatch.setattr(EI, "ROOT", tmp_path)
    monkeypatch.setattr(EI, "EVIDENCE_ROOT", tmp_path / "evidence")
    EI.install_worked_example("SIMCO")

    providers = {
        "screener_in": lambda s: (_ for _ in ()).throw(RuntimeError("403")),
        "yahoo_finance": lambda s: (_ for _ in ()).throw(RuntimeError("timeout")),
        "user_uploads": R.fetch_user_uploads,
    }
    data, steps = R.resolve("SIMCO", force_refresh=True, write_cache=False, providers=providers)
    assert data is not None
    assert data["_source"] == "user_structured_upload"
    assert any(s["source"] == "user_uploads" and s["status"] in {"OK", "PARTIAL"} for s in steps)
    assert R.coverage_score(data) > 0


def test_next_actions_always_include_official_and_reputed():
    actions = R.next_actions("INFY")
    kinds = {a["kind"] for a in actions}
    assert "official" in kinds and "reputed" in kinds
    assert any("screener.in" in a["url"] for a in actions)
    assert any("yahoo.com" in a["url"] for a in actions)
