"""
Final PAPER_AUTO activation-closeout tests: production NSE session freshness, strict
point-in-time separation for the in-sample bootstrap (no leakage), no-trade as a valid outcome,
the bhav→snapshot ingestion bridge, and a bounded headless smoke test.
"""
from __future__ import annotations

import dataclasses
import threading
from datetime import datetime, date

import pytest

from research.intelligence.data import nse_calendar as CAL
from research.intelligence.data.snapshot_store import SnapshotStore
from research.intelligence.data.provider import SnapshotBarProvider
from research.intelligence.registry import StrategyRegistry
from research.intelligence import evidence_brain as EB
from research.intelligence import decoder_registry as DREG
from research.auto_research.scheduler import AutoResearchBrain
from research.strategy_studio import discovery as DISC

_FWD = {"adjustment_consistent": True, "has_universe_history": True,
        "corporate_action_coverage": 1.0, "missing_session_rate": 0.0, "validation_errors": 0}


def _spec(sid, family, hold=5):
    return dataclasses.replace(DISC.generate(DISC.DiscoveryBudget())[0],
                               strategy_id=sid, family=family, max_holding_days=hold)


def _eq_rows(symbol, closes):
    rows, prev = [], closes[0]
    for i, c in enumerate(closes):
        rows.append((symbol, f"d{i:03d}", prev, c + 1.5, c - 1.5, c, 1000, "EQ"))
        prev = c
    return rows


def _universe(n):
    return (_eq_rows("WIN", [100 + 0.5 * i for i in range(n)])
            + _eq_rows("FLAT", [100 for _ in range(n)])
            + _eq_rows("WEAK", [100 - 0.2 * i for i in range(n)]))


# ── 1. production NSE session freshness ──────────────────────────────────────────

class TestFreshness:
    def test_friday_data_fresh_on_weekend(self):
        r = CAL.snapshot_freshness("2024-01-12", now=datetime(2024, 1, 13, 10, 0), holidays=set())
        assert r["fresh"]                                     # Sat: Friday still latest session

    def test_pre_close_prior_session_is_fresh(self):
        # Monday 11:00 (before publish cutoff) → the prior session (Fri) is what's required
        r = CAL.snapshot_freshness("2024-01-12", now=datetime(2024, 1, 15, 11, 0), holidays=set())
        assert r["fresh"] and r["required"] == "2024-01-12"

    def test_post_close_todays_session_fresh(self):
        r = CAL.snapshot_freshness("2024-01-15", now=datetime(2024, 1, 15, 18, 30), holidays=set())
        assert r["fresh"] and r["required"] == "2024-01-15"

    def test_publication_allowance_grace(self):
        # after cutoff, today's file not imported yet (gap 1) → within the 1-session allowance
        r = CAL.snapshot_freshness("2024-01-12", now=datetime(2024, 1, 15, 18, 30), holidays=set())
        assert r["fresh"] and r["sessions_behind"] == 1

    def test_delayed_file_beyond_allowance_is_stale(self):
        r = CAL.snapshot_freshness("2024-01-12", now=datetime(2024, 1, 16, 18, 30), holidays=set())
        assert not r["fresh"] and r["sessions_behind"] == 2

    def test_holiday_prior_session_fresh(self):
        r = CAL.snapshot_freshness("2024-01-12", now=datetime(2024, 1, 15, 12, 0),
                                   holidays={"2024-01-15"})
        assert r["fresh"] and r["required"] == "2024-01-12"   # Mon holiday → Fri required

    def test_missing_many_sessions_is_stale(self):
        r = CAL.snapshot_freshness("2024-01-02", now=datetime(2024, 1, 16, 18, 30), holidays=set())
        assert not r["fresh"] and r["sessions_behind"] >= 2

    def test_future_dated_bar_blocks(self):
        r = CAL.snapshot_freshness("2024-01-20", now=datetime(2024, 1, 15, 18, 30), holidays=set())
        assert not r["fresh"] and "future" in r["reason"]

    def test_duplicate_session_detected(self):
        assert CAL.has_duplicate_sessions([("A", "2024-01-01", 1, 1, 1, 1, 0, "EQ"),
                                           ("A", "2024-01-01", 1, 1, 1, 1, 0, "EQ")])
        assert not CAL.has_duplicate_sessions([("A", "2024-01-01", 1, 1, 1, 1, 0, "EQ"),
                                               ("A", "2024-01-02", 1, 1, 1, 1, 0, "EQ")])


# ── 2. strict point-in-time separation for _insample_evidence ────────────────────

class TestInSamplePIT:
    def _prov(self, tmp_path, rows, name):
        s = SnapshotStore(tmp_path / name)
        sid = s.commit_snapshot(rows, extra_manifest=_FWD)
        return SnapshotBarProvider(s.open_snapshot(sid))

    def test_future_bars_do_not_change_earlier_evidence(self, tmp_path):
        brain = AutoResearchBrain()
        spec = _spec("MOM", "cross_sectional_momentum")
        base = _universe(300)
        # snapshot B = base + 50 strongly profitable FUTURE bars appended after d298
        future = (_eq_rows("WIN", [250 + i for i in range(50)])[:0]  # keep types aligned
                  + [("WIN", f"d{300 + i:03d}", 250, 260, 240, 255, 1000, "EQ") for i in range(50)])
        pa = self._prov(tmp_path, base, "A")
        pb = self._prov(tmp_path, base + future, "B")
        ra = brain._insample_evidence(spec, pa, "d298")
        rb = brain._insample_evidence(spec, pb, "d298")
        assert ra == rb                                       # future bars excluded ⇒ identical
        # and the Evidence Card is byte-for-byte identical (same record id)
        sdef = DREG.decode("strategy", spec)[0]
        ca = EB.build_card(sdef, backtest_R=ra[0], forward_returns=[], in_sample_trades=ra[1])
        cb = EB.build_card(sdef, backtest_R=rb[0], forward_returns=[], in_sample_trades=rb[1])
        assert ca.record_id == cb.record_id and ca.as_dict() == cb.as_dict()

    def test_signal_bar_is_not_used_to_establish_its_own_evidence(self, tmp_path):
        brain = AutoResearchBrain()
        spec = _spec("MOM", "cross_sectional_momentum")
        base = _universe(300)
        # spike WIN's bar AT the decision session d298 — evidence for d298 must be unaffected
        spiked = [(r if not (r[0] == "WIN" and r[1] == "d298")
                   else ("WIN", "d298", 900, 999, 899, 950, 1, "EQ")) for r in base]
        r_norm = brain._insample_evidence(spec, self._prov(tmp_path, base, "N"), "d298")
        r_spk = brain._insample_evidence(spec, self._prov(tmp_path, spiked, "S"), "d298")
        assert r_norm == r_spk                                # d298 excluded from its own evidence

    def test_in_sample_card_is_not_forward_or_confirmed(self, tmp_path):
        brain = AutoResearchBrain()
        spec = _spec("MOM", "cross_sectional_momentum")
        r = brain._insample_evidence(spec, self._prov(tmp_path, _universe(300), "P"), "d298")
        sdef = DREG.decode("strategy", spec)[0]
        card = EB.build_card(sdef, backtest_R=r[0], forward_returns=[], in_sample_trades=r[1])
        assert card.forward_trades == 0                       # no forward evidence yet
        assert card.evidence_state not in ("CONFIRMED",)      # never forward-confirmed on in-sample


# ── 3. no-trade is a valid outcome ───────────────────────────────────────────────

class TestNoTrade:
    def _brain(self, tmp_path, n):
        s = SnapshotStore(tmp_path / "snaps")
        sid = s.commit_snapshot(_universe(n),
                                index_rows=[("NIFTY", f"d{i:03d}", 100, 101, 99, 100) for i in range(n)],
                                extra_manifest=_FWD)
        s.activate_snapshot(sid, actor="user")
        b = AutoResearchBrain(event_store_path=tmp_path / "e.jsonl",
                              runtime_state_path=tmp_path / "s.json",
                              intel_book_path=tmp_path / "b.json",
                              paper_config_path=tmp_path / "p.json",
                              regime_fn=lambda: "RISK_ON")
        b.snapshot_store = s
        b.strategy_registry = StrategyRegistry().build([_spec("MOM", "cross_sectional_momentum")])
        return b

    def test_insufficient_history_yields_no_eligible_trade(self, tmp_path):
        b = self._brain(tmp_path, 60)                         # < 121 lookback → no evidence
        out = b.run_intelligence_cycle_day(date="d059")
        assert out["positions_opened"] == [] and out["eligibility"] == "NO_ELIGIBLE_TRADE"

    def test_strategy_can_stay_inactive_without_error(self, tmp_path):
        b = self._brain(tmp_path, 60)
        for i in range(3):                                    # repeated no-trade cycles, no crash
            out = b.run_intelligence_cycle_day(date=f"d05{i}")
            assert out["status"] in ("OK", "NO_ACTION")
            assert not out["positions_opened"]
        assert not b.state.last_error


# ── 4. bhav → snapshot ingestion bridge (incl. nested .csv.zip) ───────────────────

class TestIngestionBridge:
    def test_nested_bhav_zip_to_active_snapshot(self, tmp_path):
        import io, zipfile
        from research.momentum_breakout import data_setup as D
        from research.intelligence.data.from_bhav import snapshot_from_bhav_dir
        bhav_csv = (b"TckrSymb,SctySrs,TradDt,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol\n"
                    b"RELIANCE,EQ,2024-01-01,100,110,95,105,1000\n"
                    b"TCS,EQ,2024-01-01,200,210,195,205,2000\n")
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as z:
            z.writestr("BhavCopy_NSE_CM_0_0_0_01012024_F_0000.csv", bhav_csv)
        outer = io.BytesIO()
        with zipfile.ZipFile(outer, "w") as z:                # zip-of-.csv.zip (real NSE case)
            z.writestr("Reports/BhavCopy_01012024.csv.zip", inner.getvalue())
        dest = tmp_path / "staged"
        rep = D.safe_extract_zip(io.BytesIO(outer.getvalue()), dest)
        assert rep.extracted                                  # nested zip ingested
        store = SnapshotStore(tmp_path / "snaps")
        sid, report = snapshot_from_bhav_dir(dest / "bhav", store, activate=True)
        assert sid and report["accepted"] == 2 and store.get_active_snapshot() == sid

    def test_defective_rows_quarantined_duplicates_rejected(self, tmp_path):
        from research.intelligence.data.from_bhav import bhav_to_rows
        d = tmp_path / "bhav"; d.mkdir(parents=True)
        (d / "01012024.csv").write_text(
            "SYMBOL,SERIES,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,TTL_TRD_QNTY\n"
            "GOOD,EQ,100,110,95,105,10\n"
            "BADOHLC,EQ,100,90,95,105,10\n"                   # high<low → quarantine
            "GOOD,EQ,100,110,95,105,10\n")                    # duplicate GOOD/date → reject
        rows, rep = bhav_to_rows(d)
        assert rep["accepted"] == 1 and rep["quarantined"] == 1 and rep["duplicates"] == 1


# ── 5. bounded headless smoke test (no Streamlit) ────────────────────────────────

class TestHeadlessSmoke:
    def test_worker_runs_a_cycle_and_survives_errors(self, tmp_path):
        s = SnapshotStore(tmp_path / "snaps")
        n = 300
        sid = s.commit_snapshot(_universe(n),
                                index_rows=[("NIFTY", f"d{i:03d}", 100, 101, 99, 100) for i in range(n)],
                                extra_manifest=_FWD)
        s.activate_snapshot(sid, actor="user")
        # a regime_fn that throws once proves a worker exception is caught + surfaced, not fatal
        brain = AutoResearchBrain(event_store_path=tmp_path / "e.jsonl",
                                  runtime_state_path=tmp_path / "s.json",
                                  intel_book_path=tmp_path / "b.json",
                                  paper_config_path=tmp_path / "p.json",
                                  interval_s=0.05, regime_fn=lambda: "RISK_ON")
        brain.snapshot_store = s
        brain.strategy_registry = StrategyRegistry().build([_spec("MOM", "cross_sectional_momentum")])
        assert brain.is_paper_auto_enabled()                  # persisted flag loaded
        brain.start()                                         # headless background worker
        try:
            ev = threading.Event()
            for _ in range(200):                              # bounded wait (~20s max)
                if brain.state.last_intel_cycle is not None:
                    break
                ev.wait(0.1)
            assert brain.state.last_intel_cycle is not None   # a scheduled cycle ran unattended
        finally:
            brain.stop()
        assert not brain.state.running

    def test_restart_restores_without_reapproval(self, tmp_path):
        s = SnapshotStore(tmp_path / "snaps")
        n = 300
        sid = s.commit_snapshot(_universe(n),
                                index_rows=[("NIFTY", f"d{i:03d}", 100, 101, 99, 100) for i in range(n)],
                                extra_manifest=_FWD)
        s.activate_snapshot(sid, actor="user")

        def mk():
            b = AutoResearchBrain(event_store_path=tmp_path / "e.jsonl",
                                  runtime_state_path=tmp_path / "s.json",
                                  intel_book_path=tmp_path / "b.json",
                                  paper_config_path=tmp_path / "p.json",
                                  regime_fn=lambda: "RISK_ON")
            b.snapshot_store = SnapshotStore(tmp_path / "snaps")
            b.strategy_registry = StrategyRegistry().build([_spec("MOM", "cross_sectional_momentum")])
            return b

        b1 = mk(); b1.run_intelligence_cycle_day(date="d298")
        opened = len(b1.intel_book.open)
        assert opened == 1
        b2 = mk()                                             # restart — no re-approval
        assert b2.is_paper_auto_enabled() and len(b2.intel_book.open) == opened
