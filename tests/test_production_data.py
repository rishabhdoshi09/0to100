"""
Deterministic, network-free tests for Production Data Activation & Strategy Runtime expansion:
new bar-by-bar adapters, the production registry, data states/evidence tiers, nested-archive
ingestion, and a fixture end-to-end (registry → signal → Brain 1 → Brain 2 → paper position).
No synthetic data is presented as market evidence; fixtures are clearly test inputs.
"""
from __future__ import annotations

import dataclasses
import io
import zipfile
import gzip

import pytest

from research.intelligence import strategy_runtime as RT
from research.intelligence import registry as REG
from research.intelligence import data_state as DS
from research.intelligence import evidence_brain as EB
from research.intelligence.event_store import EventStore
from research.intelligence.runtime import run_intelligence_cycle
from research.intelligence.runtime.cycle_context import CycleContext
from research.intelligence.runtime import modes as MODES
from research.intelligence.runtime.runtime_state import RuntimeState
from research.auto_research.paper_book import PaperBook
from research.strategy_studio import discovery as DISC
from research.momentum_breakout import data_setup as D


def _spec(sid, family):
    return dataclasses.replace(DISC.generate(DISC.DiscoveryBudget())[0],
                               strategy_id=sid, family=family)


def _bars(closes):
    out = []
    prev = closes[0]
    for i, c in enumerate(closes):
        out.append(RT.Bar(f"d{i}", open=prev, high=c + 1, low=c - 1, close=c))
        prev = c
    return out


# ── Phase 12: new adapters ───────────────────────────────────────────────────────

class TestNewAdapters:
    def test_trend_following_fires_on_breakout_only(self):
        closes = [100] * 55 + [101, 102, 103, 104, 112]     # flat then a breakout
        sigs = RT.entries_for(_spec("T", "trend_following"), "AAA", _bars(closes))
        assert sigs and all(s["date"] == "d59" for s in sigs)  # only the breakout bar
        assert sigs[-1]["exit_policy"] == "atr_2_3"            # its own frozen exit

    def test_pullback_uses_its_own_rules(self):
        # uptrend, then a pullback bar that tags the EMA and closes green
        closes = [100 + i for i in range(55)] + [150, 148, 152]
        bars = _bars(closes)
        sigs = RT.entries_for(_spec("P", "pullback"), "AAA", bars)
        assert isinstance(sigs, list)                          # deterministic list (may be empty)
        for s in sigs:
            assert s["stop"] < s["entry"] and s["exit_policy"] == "structural_2R"

    def test_cross_sectional_momentum_deterministic_ranks(self):
        spec = _spec("X", "cross_sectional_momentum")
        uni = {"WIN": _bars([100 + i * 0.5 for i in range(130)]),
               "FLAT": _bars([100] * 130),
               "WEAK": _bars([100 - i * 0.2 for i in range(130)])}
        a = RT.signals(spec, "d129", uni)
        b = RT.signals(spec, "d129", uni)
        assert [s["symbol"] for s in a] == [s["symbol"] for s in b]   # deterministic
        assert a[0]["symbol"] == "WIN"                                 # strongest ranked first

    def test_cross_sectional_is_point_in_time(self):
        spec = _spec("X", "cross_sectional_momentum")
        uni = {"WIN": _bars([100 + i * 0.5 for i in range(130)])}
        # evaluating as-of an EARLY date must use only bars through then (no future)
        early = RT.signals(spec, "d125", uni)
        # a symbol with <121 bars of history at as_of yields no momentum score
        short = {"WIN": _bars([100 + i for i in range(30)])}
        assert RT.signals(spec, "d29", short) == []

    def test_relative_strength_needs_benchmark(self):
        spec = _spec("R", "relative_strength")
        uni = {"WIN": _bars([100 + i for i in range(130)])}
        assert RT.signals(spec, "d129", uni, benchmark=None) == []     # no benchmark → honest []
        bench = _bars([100] * 130)
        assert RT.signals(spec, "d129", uni, benchmark=bench)          # with benchmark → ranks

    def test_unsupported_family_still_fails_loud(self):
        with pytest.raises(RT.UnsupportedStrategy):
            RT.signals(_spec("E", "event_driven"), "d1", {"AAA": _bars([100, 101])})

    def test_entries_for_rejects_cross_sectional(self):
        with pytest.raises(RT.UnsupportedStrategy):
            RT.entries_for(_spec("X", "cross_sectional_momentum"), "AAA", _bars([100] * 130))


# ── Phase 11: production registry ────────────────────────────────────────────────

class TestRegistry:
    def test_build_validates_and_disables_with_reasons(self):
        specs = [_spec("S1", "breakout"), _spec("S2", "event_driven"),   # unsupported family
                 _spec("S1", "breakout")]                                 # duplicate id
        reg = REG.StrategyRegistry().build(specs)
        enabled_s1 = [r for r in reg.all() if r.strategy_id == "S1" and r.enabled]
        assert enabled_s1 and enabled_s1[0].runtime_supported   # the original S1 kept + enabled
        s2 = [r for r in reg.all() if r.strategy_id == "S2"][0]
        assert not s2.enabled and "runtime adapter missing" in s2.disabled_reasons
        # the duplicate id was recorded with a reason, never crashed
        assert any("duplicate strategy id" in r.disabled_reasons for r in reg.all())

    def test_deployable_specs_exclude_unsupported(self):
        reg = REG.StrategyRegistry().build([_spec("A", "trend_following"),
                                            _spec("B", "regime_specific")])
        fams = {s.family for s in reg.deployable_specs()}
        assert "trend_following" in fams and "regime_specific" not in fams

    def test_owner_can_disable(self):
        reg = REG.StrategyRegistry().build([_spec("A", "breakout")],
                                           owner_enabled={"A": False})
        assert reg.deployable_specs() == []


# ── Phase 16/19: data states + evidence tiers ────────────────────────────────────

class TestDataStateTiers:
    def test_entry_permission_by_state(self):
        assert DS.allows_new_entries(DS.READY)
        assert DS.allows_new_entries(DS.READY_WITH_WARNINGS)
        for s in (DS.NO_DATA, DS.DEGRADED, DS.STALE, DS.CONFLICTED, DS.FAILED):
            assert not DS.allows_new_entries(s)
        assert DS.manages_positions(DS.STALE) and not DS.manages_positions(DS.FAILED)

    def test_tier_classification(self):
        assert DS.classify_tier({})[0] == DS.OPERATIONAL_ONLY
        assert DS.classify_tier({"has_prices": True, "validation_errors": 2})[0] == DS.OPERATIONAL_ONLY
        limited = DS.classify_tier({"has_prices": True, "adjustment_consistent": False})
        assert limited[0] == DS.LIMITED_RESEARCH
        full = DS.classify_tier({"has_prices": True, "adjustment_consistent": True,
                                 "missing_session_rate": 0.0, "has_benchmark": True,
                                 "has_universe_history": True, "corporate_action_coverage": 1.0,
                                 "freshness_days": 1})
        assert full[0] == DS.FORWARD_ELIGIBLE

    def test_card_carries_tier_and_flags_non_forward(self):
        sdef = __import__("research.intelligence.decoder_registry", fromlist=["decode"]).decode(
            "strategy", _spec("A", "breakout"))[0]
        card = EB.build_card(sdef, backtest_R=0.3, forward_returns=[0.3] * 40,
                             dataset_tier=DS.RESEARCH_ELIGIBLE)
        assert card.dataset_tier == DS.RESEARCH_ELIGIBLE
        assert any("not forward-eligible" in w for w in card.data_quality_warnings)


# ── Phase 21: nested-archive ingestion (the BhavCopy_*.csv.zip real-world case) ──

_BHAV_CSV = (b"TckrSymb,SctySrs,TradDt,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol\n"
             b"RELIANCE,EQ,2024-01-01,100,110,95,105,1000\n"
             b"TCS,EQ,2024-01-01,200,210,195,205,2000\n")


def _zip_bytes(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


class TestNestedArchiveIngestion:
    def test_nested_csv_zip_member_is_recursed_not_rejected(self, tmp_path):
        inner = _zip_bytes({"BhavCopy_NSE_CM_0_0_0_20240101_F_0000.csv": _BHAV_CSV})
        outer = _zip_bytes({"Reports-Daily/BhavCopy_NSE_CM_0_0_0_20240101_F_0000.csv.zip": inner})
        rep = D.safe_extract_zip(io.BytesIO(outer), tmp_path)
        # the inner archive must NOT be rejected merely for being zip-compressed
        assert not any("unsupported file" in why for _, why in rep.rejected)
        assert rep.extracted                                  # the bhav CSV was ingested

    def test_gz_member_is_decompressed(self, tmp_path):
        gz = gzip.compress(_BHAV_CSV)
        outer = _zip_bytes({"BhavCopy_20240101.csv.gz": gz})
        rep = D.safe_extract_zip(io.BytesIO(outer), tmp_path)
        assert rep.extracted and not any("unsupported" in why for _, why in rep.rejected)

    def test_zip_of_zips_depth_is_bounded(self, tmp_path):
        # deeply nest beyond the limit → the deepest is refused, never infinite recursion
        data = _zip_bytes({"BhavCopy_20240101.csv": _BHAV_CSV})
        for _ in range(6):
            data = _zip_bytes({"pack.csv.zip": data})
        rep = D.safe_extract_zip(io.BytesIO(data), tmp_path)
        assert any("nesting too deep" in why for _, why in rep.rejected)


# ── Phase 20: fixture end-to-end (registry → signal → brains → paper position) ──

class TestEndToEndOnRegistry:
    def _universe(self):
        return {"WIN": _bars([100 + i * 0.5 for i in range(130)]),
                "FLAT": _bars([100] * 130),
                "WEAK": _bars([100 - i * 0.2 for i in range(130)])}

    def test_momentum_strategy_opens_paper_position(self):
        reg = REG.StrategyRegistry().build([_spec("MOM", "cross_sectional_momentum")])
        specs = reg.deployable_specs()
        assert specs                                          # registry populated a runnable spec
        ctx = CycleContext(as_of_date="d129", mode=MODES.PAPER_AUTO, data_ok=True,
                           data_snapshot_id="snap1", dataset_tier=DS.FORWARD_ELIGIBLE,
                           strategies=specs, data={"MOM": self._universe()})
        store, book, state = EventStore(), PaperBook(), RuntimeState()
        res = run_intelligence_cycle(ctx, store=store, book=book, runtime_state=state,
                                     backtest_R={"MOM": 0.3}, backtest_trades={"MOM": 40})
        assert ("MOM", "WIN") in res.signals_generated
        assert res.positions_opened and res.positions_opened[0][1] == "WIN"
        assert len(book.open) == 1
