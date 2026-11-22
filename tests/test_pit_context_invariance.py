"""Point-in-time market context reaches the learning record and nothing a decision reads.

product.pit_features fills delivery %, India VIX, Nifty trend and breadth into the
frozen DECISION observation. These tests pin the claim that this changes the
research corpus only:

* the live analog evidence of a decision is identical whether the stored corpus
  carries the filled context or not (the query never carries it);
* enrich() returns the same decision payload with or without context; only the
  frozen observation differs, and only in the context features;
* an un-promoted challenger never scores, so paper selection is unaffected;
* only the learning-record builder (and the challenger, for the producer
  version) imports product.pit_features.
"""

from __future__ import annotations

import ast
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from product import evidence_intelligence as EI
from product import pit_features as PF
from product.decision import Decision
from research import feature_store as FS

REPO = Path(__file__).resolve().parents[1]
CONTEXT = ("delivery_pct", "vix", "index_trend", "breadth_pct_above_50dma")
_CTX_VALUES = {"vix": 13.7, "index_trend": "UP", "breadth_pct_above_50dma": 58.0}


def _decision(i: int, when: datetime, *, setup: str = "VCP_BREAKOUT", state: str = "WAIT") -> Decision:
    return Decision(
        symbol=f"T{i:03d}",
        state=state,
        setup=setup,
        generated_at=when.isoformat(),
        score=70.0 + (i % 7),
        market_state="TRENDING_BULL",
        sector_state="STRONG",
        technical_evidence={
            "rsi": 50.0 + (i % 11),
            "atr_pct": 1.5 + (i % 5) * 0.4,
            "quality_score": 60.0 + (i % 9),
            "volume_ratio": 1.0 + (i % 4) * 0.3,
            "reward_risk": 1.5 + (i % 3) * 0.5,
        },
        entry=100.0,
        stop=95.0,
        target=110.0,
        evidence_class="PAPER_FORWARD",
    )


def _seed_corpus(with_context: bool, n: int = 60) -> None:
    base = datetime(2026, 6, 1, 10, 30, tzinfo=timezone.utc)
    for i in range(n):
        d = _decision(i, base + timedelta(days=i), setup="VCP_BREAKOUT" if i % 3 else "PULLBACK")
        feats = EI.decision_features(d)
        if with_context:
            feats.update({"delivery_pct": 30.0 + i % 40, "vix": 11.0 + i % 9,
                          "index_trend": ("UP", "FLAT", "DOWN")[i % 3],
                          "breadth_pct_above_50dma": 35.0 + i % 40})
        oid = f"decision::seed{i:03d}"
        assert FS.snapshot(oid, d.symbol, "DECISION", feats, ts=d.generated_at,
                           meta={"setup": d.setup, "market_state": d.market_state,
                                 "sector_state": d.sector_state})["status"] == "frozen"
        FS.set_outcome(oid, ((i * 37) % 11 - 5) / 4.0, outcome_meta={"evidence_class": "PAPER_FORWARD"})


def _isolated_store(tmp_path, monkeypatch, name: str) -> Path:
    path = tmp_path / name
    monkeypatch.setattr(FS, "_DB_PATH", path)
    return path


def test_live_analog_evidence_ignores_context_stored_in_the_corpus(tmp_path, monkeypatch):
    query = _decision(500, datetime(2026, 9, 15, 10, 30, tzinfo=timezone.utc))

    _isolated_store(tmp_path, monkeypatch, "without.db")
    _seed_corpus(with_context=False)
    without = EI.evidence_read(query)

    _isolated_store(tmp_path, monkeypatch, "with.db")
    _seed_corpus(with_context=True)
    with_ctx = EI.evidence_read(query)

    assert without["nearest_analogs"], "the corpus must actually produce analogs"
    assert json.dumps(with_ctx, sort_keys=True, default=str) == json.dumps(without, sort_keys=True, default=str)


def test_enrich_returns_the_same_decision_and_freezes_context_only(tmp_path, monkeypatch):
    when = datetime(2026, 9, 15, 11, 0, tzinfo=timezone.utc)   # 16:30 IST
    frozen = {}
    for label, ctx in (("empty", {}), ("filled", dict(_CTX_VALUES))):
        path = _isolated_store(tmp_path, monkeypatch, f"{label}.db")
        _seed_corpus(with_context=False)
        monkeypatch.setattr(PF, "context_features", lambda decided_at, _c=ctx: dict(_c))
        monkeypatch.setattr("data.bhavcopy_store.get_ohlcv", lambda s: None)
        out = EI.enrich(_decision(700, when))
        con = sqlite3.connect(path)
        row = con.execute("SELECT features FROM observations WHERE observation_id=?",
                          (out.provenance["learning_observation_id"],)).fetchone()
        con.close()
        frozen[label] = (out, json.loads(row[0]))

    (a, fa), (b, fb) = frozen["empty"], frozen["filled"]
    assert a.state == b.state and a.score == b.score and a.entry == b.entry and a.stop == b.stop
    assert a.decision_id == b.decision_id
    assert json.dumps(a.historical_evidence, sort_keys=True, default=str) == \
        json.dumps(b.historical_evidence, sort_keys=True, default=str)
    changed = {k for k in fa if fa[k] != fb[k]}
    assert changed == {"vix", "index_trend", "breadth_pct_above_50dma"}
    assert fb["vix"] == 13.7 and fb["index_trend"] == "UP" and fb["breadth_pct_above_50dma"] == 58.0
    assert fa["index_trend"] == "UNKNOWN" and fa["vix"] is None
    assert fa["regime"] == fb["regime"]


@pytest.mark.parametrize("status", ["OBSERVING", "REJECTED", "SHADOW_CANDIDATE"])
def test_an_unpromoted_challenger_cannot_move_paper_selection(tmp_path, status):
    from product import challenger_learning as CL

    path = tmp_path / "challenger.json"
    model = {"status": status, "model_version": "clf_test", "coef": [0.5] * len(CL.MODEL_FEATURES),
             "medians": [0.0] * len(CL.MODEL_FEATURES), "scales": [1.0] * len(CL.MODEL_FEATURES),
             "intercept": 0.0}
    path.write_text(json.dumps({"current": model}))
    card = {"symbol": "TCS", "reco_tier": "good_setup", "rsi": 55, "vix": 30, "delivery_pct": 70,
            "breadth_pct_above_50dma": 80, "entry": 100, "stop": 95, "target": 110,
            "scan_scanned_at": "2026-09-15T16:30:00+05:30"}
    adj = CL.paper_selection_adjustment(card, path=path)
    assert adj["adjustment"] == 0.0 and adj["affects_selection"] is False


def test_only_the_learning_record_builder_imports_pit_features():
    importers = []
    for py in REPO.rglob("*.py"):
        rel = py.relative_to(REPO).as_posix()
        if rel.startswith(("tests/", "venv/", "node_modules/", "frontend/", ".claude/")) or "/site-packages/" in rel:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if any(n == "product.pit_features" or (n == "product.pit_features") for n in names) or \
                    (isinstance(node, ast.ImportFrom) and node.module == "product"
                     and any(a.name == "pit_features" for a in node.names)):
                importers.append(rel)
    # evidence_intelligence builds the learning record; challenger_learning only
    # reads the producer version a model was trained under.
    # research/pit_context_coverage.py is an offline measurement over a COPY.
    # product/learning_observation_correction.py re-derives frozen fills on
    # COPIES to find defective records; it is a CLI that nothing imports
    # (test_the_correction_tool_is_never_imported_by_the_desk).
    assert sorted(set(importers)) == ["product/challenger_learning.py", "product/evidence_intelligence.py",
                                      "product/learning_observation_correction.py",
                                      "research/pit_context_coverage.py"]


def test_the_correction_tool_is_never_imported_by_the_desk():
    importers = []
    for py in REPO.rglob("*.py"):
        rel = py.relative_to(REPO).as_posix()
        if rel.startswith(("tests/", "venv/", "node_modules/", "frontend/", ".claude/")) or "/site-packages/" in rel:
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        if rel != "product/learning_observation_correction.py" and "learning_observation_correction" in text:
            importers.append(rel)
    assert importers == []
