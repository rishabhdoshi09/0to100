"""Two independent equity PAPER engines share ONE real account.

execution.autopilot (legacy, trades.db) and product.paper_autopilot (modern,
intel_book.json -- the documented sole new-entry authority) each used to
gate new entries from a view of ONLY their own book. The account-level risk
*display* (risk.portfolio_risk.portfolio_risk_report) already combined both
books, but the *execution* gates did not -- a trade the display would mark
DANGER could still be approved by an engine that could not see what the
other had already committed.

risk.portfolio_risk.account_exposure_gate() is the ONE shared check both
engines now call before mutating, built on the exact same combined read
(_combined_open_rows) the display uses, so the two can never disagree.
account_exposure_gate() + account_mutation_lock() close:
  - duplicate symbol, either direction
  - combined position cap
  - combined sector concentration
  - combined open-risk cap (no double-use of the same account's capital)
  - the race between two processes each passing a check and then mutating

These tests exercise the REAL production functions (execution.autopilot.
consider(), product.paper_autopilot.run_reco_paper_cycle(), trade_executor.
place_trade()) against real on-disk state (trades.db / intel_book.json),
never a hand-built "pretend it was rejected" dict.
"""
from __future__ import annotations

import contextlib
import json
import threading
import time
from datetime import datetime, timezone

from product.paper_autopilot import (
    DUPLICATE_POSITION,
    MAX_POSITIONS,
    SECTOR_CAP,
    run_reco_paper_cycle,
)
from research.auto_research.paper_book import PaperBook


# ── shared fixtures (mirrors tests/test_paper_autopilot_money_path.py) ──────

def _now():
    return datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)


def _eligible_card(symbol="TCS", **over):
    card = {
        "symbol": symbol,
        "reco_tier": "high_conviction",
        "reco_tier_label": "High Conviction",
        "entry_state": "ready",
        "entry": 100.0,
        "stop": 94.0,
        "target": 115.0,
        "cmp": 100.0,
        "chase_risk": False,
        "volume_ratio": 1.4,
        "sector": "Technology",
        "family_confirms": 3,
        "score": 82,
        "primary_thesis": "VCP + quality",
        "setup_label": "VCP",
        "allows_recommend": True,
        "methods": [
            {"id": "tape", "status": "pass", "points": 90, "detail": "clean"},
            {"id": "sepa", "status": "pass", "points": 70, "detail": "base"},
            {"id": "funds", "status": "pass", "points": 80, "detail": "quality"},
            {"id": "trend", "status": "pass", "points": 85, "detail": "up"},
            {"id": "rs", "status": "pass", "points": 75, "detail": "leader"},
            {"id": "ev", "status": "unknown", "points": None, "detail": "n<30"},
            {"id": "conviction", "status": "pass", "points": 80, "detail": "ready"},
            {"id": "case", "status": "unknown", "points": None, "detail": "n<30"},
            {"id": "sector", "status": "pass", "points": 80, "detail": "leader"},
        ],
    }
    card.update(over)
    return card


def _workspace(cards, *, generated_at=None):
    stamp = generated_at or _now().isoformat()
    return {
        "schema_version": 4,
        "generated_at": stamp,
        "scan_scanned_at": stamp,
        "categories": [{"id": "wealth_builders", "count": len(cards), "cards": cards}],
    }


def _cycle(book, cards, **kwargs):
    kwargs.setdefault("now", _now())
    kwargs.setdefault("as_of", "2026-09-01")
    kwargs.setdefault("entries_allowed", True)
    kwargs.setdefault("paper_enabled", True)
    kwargs.setdefault("persist_journal", True)
    kwargs.setdefault("workspace", _workspace(cards))
    kwargs.setdefault("enforce_history", False)
    return run_reco_paper_cycle(book=book, cards=cards, **kwargs)


def _arm_legacy(tmp_path, monkeypatch, *, sector="Technology"):
    """Real legacy engine, wired exactly like tests/test_money_paths.py's
    TestAutopilot._setup -- not a stand-in, the actual production module."""
    import execution.autopilot as ap
    import execution.trade_executor as te
    import scan.sector_heat as sh

    monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "autopilot.json")
    monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
    monkeypatch.setattr(te, "kite_ready", lambda: False)
    monkeypatch.setattr(ap, "_state", {}, raising=False)
    ap._state = {}
    monkeypatch.setattr(ap, "_notify", lambda msg: None)
    monkeypatch.setattr(ap, "_broker_cash", lambda: 500_000.0)
    monkeypatch.setattr(ap, "_market_regime", lambda: "TRENDING_BULL")
    monkeypatch.setattr(ap, "_brain_posture", lambda: ("NORMAL", ""))
    monkeypatch.setattr(ap, "start_book_monitor", lambda: None)
    monkeypatch.setattr(ap, "_serial_losers_cached", lambda: set())
    monkeypatch.setattr(ap, "_anchor_live", lambda sym, entry, stop, mc: (entry, ""))
    monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
    monkeypatch.setattr(sh, "sector_performance", lambda min_members=3: [
        {"sector": sector, "chg_1d": 1.5, "chg_5d": 3.0, "members": 4},
    ])
    ap.set_config(allocation=100000, mode="PAPER")
    ap.arm()
    return ap, te


def _write_modern_open(symbol, *, qty=10, entry=100.0, stop=95.0):
    """Real on-disk modern-engine state -- the same file
    product.paper_status.modern_engine_open_positions() reads for real."""
    from core.runtime_paths import logs_path
    target = logs_path("intelligence", "intel_book.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"open": [
        {"symbol": symbol, "qty": qty, "entry_price": entry, "stop_price": stop},
    ]}), encoding="utf-8")


# ── 1. legacy opens A -> modern tries A -> rejected ─────────────────────────

def test_legacy_opens_symbol_then_modern_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    ap, te = _arm_legacy(tmp_path, monkeypatch)
    assert ap.consider("RELIANCE", 2500, 2400, 85, 0.3, "Technology", "test") is True

    book = PaperBook(capital=100_000)
    out = _cycle(book, [_eligible_card("RELIANCE")])
    assert out["rejections"][0]["reason_code"] == DUPLICATE_POSITION
    assert not book.open


# ── 2. modern opens A -> legacy tries A -> rejected ─────────────────────────

def test_modern_opens_symbol_then_legacy_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    _write_modern_open("RELIANCE")
    ap, te = _arm_legacy(tmp_path, monkeypatch)

    assert ap.consider("RELIANCE", 2500, 2400, 85, 0.3, "Technology", "test") is False
    f = ap.reject_funnel()
    assert f["rejects"].get("duplicate with primary paper engine") == 1
    # a symbol neither engine holds still trades normally
    assert ap.consider("INFY", 1500, 1450, 85, 0.3, "Technology", "test") is True


# ── 3. combined open risk + a new trade over the 5% cap -> rejected ─────────

def test_combined_open_risk_cap_blocks_a_new_trade_from_either_engine(tmp_path, monkeypatch):
    import execution.trade_executor as te
    import risk.portfolio_risk as prm

    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
    monkeypatch.setattr(te, "kite_ready", lambda: False)
    monkeypatch.setattr(prm, "_capital", lambda: 100_000.0)

    # Legacy book: 2000 rupees of open risk (100 qty x 20 rupee stop width).
    te.place_trade("AAA", 100, "LIMIT", 100.0, 80.0, 130.0)
    # Modern book: another 2000 rupees of open risk.
    _write_modern_open("BBB", qty=100, entry=200.0, stop=180.0)
    # Combined existing risk: 4000 (4% of 100,000) -- well under the 5% cap
    # when read from EITHER book alone, but a new 1500-rupee-risk trade pushes
    # the COMBINED total to 5500 (5.5%), over the ceiling.
    gate = prm.account_exposure_gate("CCC", qty=100, entry=300.0, stop=285.0, sector="Metals")
    assert gate["ok"] is False
    assert gate["reason_code"] == prm.GATE_MAX_PORTFOLIO_RISK
    assert gate["report"]["open_risk_pct"] > 5

    # The same candidate at a size that keeps the combined total under the
    # cap is approved.
    small = prm.account_exposure_gate("CCC", qty=10, entry=300.0, stop=285.0, sector="Metals")
    assert small["ok"] is True


# ── 4. combined position cap -> no new entry from either engine ────────────

def test_combined_position_cap_blocks_new_entry(tmp_path, monkeypatch):
    import execution.trade_executor as te

    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
    monkeypatch.setattr(te, "kite_ready", lambda: False)

    # Legacy alone already holds the account's max_open_positions (5).
    for i, sym in enumerate(["P1", "P2", "P3", "P4", "P5"]):
        te.place_trade(sym, 10, "LIMIT", 100.0 + i, 95.0 + i, 110.0 + i)

    book = PaperBook(capital=100_000)
    out = _cycle(book, [_eligible_card("P6")])
    assert out["rejections"][0]["reason_code"] == MAX_POSITIONS
    assert not book.open

    # The reciprocal direction: modern alone at the cap blocks a legacy entry.
    import execution.trade_executor as te2
    te2._DB.unlink(missing_ok=True)  # clear the legacy book for this half
    from core.runtime_paths import logs_path
    target = logs_path("intelligence", "intel_book.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"open": [
        {"symbol": f"M{i}", "qty": 10, "entry_price": 100.0, "stop_price": 95.0}
        for i in range(5)
    ]}), encoding="utf-8")
    ap, _te = _arm_legacy(tmp_path, monkeypatch)
    assert ap.consider("P7", 100, 95, 85, 0.3, "Technology", "test") is False
    f = ap.reject_funnel()
    assert f["rejects"].get("position limit full (combined account)") == 1


# ── 5. combined sector concentration affects the real entry gate ───────────

def test_combined_sector_concentration_blocks_new_entry(tmp_path, monkeypatch):
    import execution.trade_executor as te
    import scan.sector_heat as sh

    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
    monkeypatch.setattr(te, "kite_ready", lambda: False)
    monkeypatch.setattr(sh, "sector_of", lambda sym: "Technology")

    # Legacy holds 2 Technology names; modern holds a 3rd -- combined
    # concentration in ONE sector that neither book's own view reaches alone.
    te.place_trade("T1", 10, "LIMIT", 100.0, 95.0, 110.0)
    te.place_trade("T2", 10, "LIMIT", 100.0, 95.0, 110.0)
    _write_modern_open("T3")

    book = PaperBook(capital=100_000)
    out = _cycle(book, [_eligible_card("T4", sector="Technology")])
    assert out["rejections"][0]["reason_code"] == SECTOR_CAP
    assert not book.open


# ── 6. restart/reload preserves the combined exposure truth ────────────────

def test_restart_reload_preserves_combined_exposure_truth(tmp_path, monkeypatch):
    import importlib
    import execution.trade_executor as te

    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
    monkeypatch.setattr(te, "kite_ready", lambda: False)
    te.place_trade("R1", 10, "LIMIT", 100.0, 95.0, 110.0)
    _write_modern_open("R2")

    import risk.portfolio_risk as prm
    before = prm.account_exposure_gate("R3")
    # 2 real positions (R1 legacy + R2 modern) + the R3 candidate itself.
    assert before["report"]["n_positions"] == 3

    # Nothing in this gate is cached in module state -- it is derived purely
    # from durable on-disk state (trades.db + intel_book.json), so a full
    # module reload (the closest in-process proxy for "the process
    # restarted") must see the IDENTICAL combined truth, not an empty or
    # stale one.
    importlib.reload(prm)
    after = prm.account_exposure_gate("R3")
    assert after["report"]["n_positions"] == before["report"]["n_positions"]
    assert after["ok"] == before["ok"]


# ── 7. no race: the shared lock gives true cross-thread mutual exclusion ───

def test_account_mutation_lock_prevents_concurrent_overlap(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    from risk.portfolio_risk import account_mutation_lock

    events: list[tuple[str, str, float]] = []
    ev_lock = threading.Lock()

    def worker(name: str) -> None:
        with account_mutation_lock():
            with ev_lock:
                events.append((name, "enter", time.monotonic()))
            time.sleep(0.2)
            with ev_lock:
                events.append((name, "exit", time.monotonic()))

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))
    t1.start()
    time.sleep(0.05)
    t2.start()
    t1.join(5)
    t2.join(5)

    windows: dict[str, dict[str, float]] = {}
    for name, kind, ts in events:
        windows.setdefault(name, {})[kind] = ts
    assert set(windows) == {"A", "B"}
    a, b = windows["A"], windows["B"]
    # one engine's whole check-and-mutate window must finish before the
    # other's starts -- no interleave is possible while holding the lock.
    assert a["exit"] <= b["enter"] or b["exit"] <= a["enter"]


def test_both_engines_acquire_the_shared_lock_on_a_real_entry(tmp_path, monkeypatch):
    """Not just that the lock primitive works -- that BOTH production entry
    paths actually go through it on a real successful trade."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    import risk.portfolio_risk as prm

    calls: list[str] = []

    @contextlib.contextmanager
    def spy_lock():
        calls.append("acquire")
        yield
        calls.append("release")

    monkeypatch.setattr(prm, "account_mutation_lock", spy_lock)

    ap, te = _arm_legacy(tmp_path, monkeypatch)
    assert ap.consider("WIPRO", 400, 380, 85, 0.3, "Technology", "test") is True
    assert calls == ["acquire", "release"], "legacy engine did not use the shared lock"

    calls.clear()
    book = PaperBook(capital=100_000)
    out = _cycle(book, [_eligible_card("HCLTECH")])
    assert out["taken"], out
    assert calls == ["acquire", "release"], "modern engine did not use the shared lock"


# ── 8. live money stays locked throughout ───────────────────────────────────

def test_live_money_stays_locked_throughout(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    from product.live_execution_interlock import get_live_execution_state

    before = get_live_execution_state()
    assert before.locked is True
    assert before.authorized is False

    ap, te = _arm_legacy(tmp_path, monkeypatch)
    ap.consider("SBIN", 600, 580, 85, 0.3, "Technology", "test")
    book = PaperBook(capital=100_000)
    _cycle(book, [_eligible_card("AXISBANK")])

    after = get_live_execution_state()
    assert after.locked is True
    assert after.authorized is False
