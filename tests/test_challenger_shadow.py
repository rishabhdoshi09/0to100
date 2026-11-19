"""Prospective shadow test: frozen before outcomes, settled only when matured."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from product import challenger_shadow as CS
from product import learning_scorecard as LS


def _frame(seed, n=320, end="2026-11-18", vol=1.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01 * vol, n))
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6},
                        index=idx)


def test_record_is_frozen_once_and_needs_the_sessions_own_bar(tmp_path):
    frames = {f"S{i}": _frame(i) for i in range(6)}
    frames["NOBAR"] = _frame(99, end="2026-11-17")                   # missing the session bar
    cands = [{"symbol": s, "score": 50 + i} for i, s in enumerate(frames)]
    path = tmp_path / "shadow.jsonl"
    first = CS.record_session("2026-11-18", cands, frames.get, path=path)
    again = CS.record_session("2026-11-18", cands, frames.get, path=path)
    assert first["recorded"] == 6 and again["reason"] == "already_recorded"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {r["symbol"] for r in rows} == {f"S{i}" for i in range(6)}
    assert all(r["hypothesis"] == CS.HYPOTHESIS and "composite" in r for r in rows)


def test_future_bars_do_not_change_frozen_features(tmp_path):
    base = {f"S{i}": _frame(i) for i in range(5)}
    later = {k: pd.concat([v, v.iloc[-5:].set_axis(pd.bdate_range("2026-11-19", periods=5)) * 3]) for k, v in base.items()}
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    CS.record_session("2026-11-18", [{"symbol": s} for s in base], base.get, path=a)
    CS.record_session("2026-11-18", [{"symbol": s} for s in later], later.get, path=b)
    strip = lambda p: [{k: v for k, v in json.loads(l).items() if k != "frozen_at"} for l in p.read_text().splitlines()]
    assert strip(a) == strip(b)


def test_only_matured_sessions_are_evaluated_and_thin_evidence_is_insufficient(tmp_path):
    frames = {f"S{i}": _frame(i, end="2026-12-31") for i in range(8)}
    path = tmp_path / "shadow.jsonl"
    cut = {k: v[v.index <= "2026-11-18"] for k, v in frames.items()}
    CS.record_session("2026-11-18", [{"symbol": s, "score": i} for i, s in enumerate(frames)], cut.get, path=path)
    rep = CS.evaluate(frames.get, maturity=lambda s: LS.IMMATURE, path=path, horizon=10)
    assert rep["sessions_recorded"] == 1 and rep["sessions_matured"] == 0 and rep["sessions_with_ic"] == 0
    rep = CS.evaluate(frames.get, maturity=lambda s: LS.MATURED, path=path, horizon=10,
                      session_index={"2026-11-18": 0})
    assert rep["sessions_with_ic"] == 1
    assert rep["results"]["composite"]["verdict"] == "INSUFFICIENT EVIDENCE"
    assert rep["status"] == "SHADOW_ONLY"


def test_shadow_writes_nothing_but_its_own_ledger(tmp_path, monkeypatch):
    from core.runtime_paths import logs_dir
    before = {p for p in logs_dir().rglob("*") if p.is_file()} if logs_dir().exists() else set()
    frames = {f"S{i}": _frame(i) for i in range(5)}
    path = tmp_path / "shadow.jsonl"
    CS.record_session("2026-11-18", [{"symbol": s} for s in frames], frames.get, path=path)
    CS.evaluate(frames.get, maturity=lambda s: LS.MATURED, path=path, horizon=10, session_index={})
    after = {p for p in logs_dir().rglob("*") if p.is_file()} if logs_dir().exists() else set()
    assert after == before
