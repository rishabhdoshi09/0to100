"""
Governance layer — the committee's core ask: how the system LOSES trust.

Kill conditions must HALT, rollback triggers must DE_RISK, a human kill-switch is
absolute, and consecutive failures must accumulate to a halt. Data-integrity must
catch un-adjusted corporate-action gaps. All network-free.
"""
import numpy as np
import pytest

from core import governance as G
from core import data_integrity as DI
from core import evidence_levels as EL


class TestGovernanceDecision:
    def test_clean_state_is_normal(self):
        g = G.evaluate_state({"drawdown_pct": 3, "calibration_ece": 0.03})
        assert g["state"] == G.NORMAL and g["size_multiplier"] == 1.0

    def test_kill_conditions_halt(self):
        for cond in ("manual_halt", "data_corrupt", "ca_mismatch",
                     "reconciliation_mismatch"):
            g = G.evaluate_state({cond: True})
            assert g["state"] == G.HALT and g["size_multiplier"] == 0.0
            assert g["kill_reasons"]

    def test_drawdown_hard_kill_vs_soft_derisk(self):
        assert G.evaluate_state({"drawdown_pct": 26})["state"] == G.HALT
        soft = G.evaluate_state({"drawdown_pct": 14})
        assert soft["state"] == G.DE_RISK and soft["size_multiplier"] == 0.5

    def test_consecutive_failures_halt(self):
        assert G.evaluate_state({"consecutive_order_failures": 3})["state"] == G.HALT
        assert G.evaluate_state({"consecutive_gtt_failures": 3})["state"] == G.HALT
        assert G.evaluate_state({"consecutive_order_failures": 1})["state"] == G.NORMAL

    def test_rollback_triggers_derisk(self):
        assert G.evaluate_state({"calibration_ece": 0.15})["state"] == G.DE_RISK
        assert G.evaluate_state({"slippage_ratio": 2.5})["state"] == G.DE_RISK
        assert G.evaluate_state({"regime_adverse": True})["state"] == G.DE_RISK

    def test_extreme_vix_halts(self):
        assert G.evaluate_state({"vix": 50})["state"] == G.HALT
        assert G.evaluate_state({"vix": 20})["state"] == G.NORMAL

    def test_kill_beats_derisk(self):
        # a HALT condition present alongside a DE_RISK one → HALT wins
        g = G.evaluate_state({"data_corrupt": True, "drawdown_pct": 14})
        assert g["state"] == G.HALT


class TestGovernanceGate:
    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "_STATE_FILE", tmp_path / "gov.json")
        G._cache.update(ts=0.0, data=None)      # bust the cache between tests

    def test_manual_kill_switch_blocks_orders(self, monkeypatch):
        # isolate from live collectors → only the manual switch matters
        monkeypatch.setattr(G, "_collect_signals",
                            lambda: dict(G._read_state()))
        G.set_manual_halt(True)
        G._cache.update(ts=0.0, data=None)
        allowed, mult, reason = G.can_place_order()
        assert allowed is False and mult == 0.0 and "HALT" in reason
        G.set_manual_halt(False)
        G._cache.update(ts=0.0, data=None)
        assert G.can_place_order()[0] is True

    def test_consecutive_failures_accumulate_then_halt(self, monkeypatch):
        monkeypatch.setattr(G, "_collect_signals",
                            lambda: dict(G._read_state()))
        for _ in range(3):
            G.record_order_result(ok=False)
        G._cache.update(ts=0.0, data=None)
        assert G.can_place_order()[0] is False          # 3 fails → HALT
        G.record_order_result(ok=True)                  # a success resets
        G._cache.update(ts=0.0, data=None)
        assert G.can_place_order()[0] is True

    def test_assess_is_fail_open(self, monkeypatch):
        def _boom():
            raise RuntimeError("collector down")
        monkeypatch.setattr(G, "_collect_signals", _boom)
        G._cache.update(ts=0.0, data=None)
        g = G.assess(force=True)
        assert g["state"] == G.NORMAL                   # broken sentinel ≠ block all


class TestDataIntegrity:
    def test_phantom_gap_catches_unadjusted_split(self):
        # a 1:1 bonus: price halves overnight → a phantom −50% gap
        closes = [100, 101, 102, 51, 52, 53]            # split between bar 2 and 3
        gaps = DI.phantom_gaps(closes)
        assert len(gaps) == 1 and gaps[0]["pct"] < -35

    def test_clean_series_has_no_phantom_gaps(self):
        rng = np.random.default_rng(0)
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.015, 300))   # normal walk
        assert DI.phantom_gaps(closes) == []

    def test_report_fail_open(self):
        # no store in the test env → graceful, never raises
        r = DI.integrity_report()
        assert r["checked"] == 0 and r["ca_mismatch"] is False


class TestCorporateActions:
    """Phase-1 data integrity: back-adjustment must turn a phantom split-gap into
    a continuous series, and must NEVER invent an adjustment when no table exists."""

    def _bonus_frame(self):
        import pandas as pd
        idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03",
                              "2024-01-04", "2024-01-05", "2024-01-06"])
        # 1:1 bonus ex-date 2024-01-04 → price halves overnight (a phantom −50%)
        return pd.DataFrame({"open": [100, 101, 102, 51, 52, 53],
                             "high": [101, 102, 103, 52, 53, 54],
                             "low":  [99, 100, 101, 50, 51, 52],
                             "close": [100, 101, 102, 51, 52, 53],
                             "volume": [1000, 1000, 1000, 2000, 2000, 2000]},
                            index=idx)

    def test_raw_frame_has_a_phantom_gap(self):
        from core import data_integrity as DI
        df = self._bonus_frame()
        assert len(DI.phantom_gaps(df["close"].to_numpy(dtype=float))) == 1

    def test_adjustment_makes_it_continuous(self):
        import pandas as pd
        from data import corporate_actions as CA
        df = self._bonus_frame()
        events = [{"ex_date": pd.Timestamp("2024-01-04"), "factor": 2.0,
                   "type": "bonus"}]
        adj = CA.adjust_frame(df, events)
        assert CA.is_continuous(adj) is True                 # gap removed
        # pre-ex prices halved, volume doubled; post-ex bars untouched
        assert adj["close"].iloc[2] == pytest.approx(51.0)
        assert adj["close"].iloc[3] == pytest.approx(51.0)
        assert adj["volume"].iloc[0] == pytest.approx(2000.0)
        assert adj["volume"].iloc[5] == pytest.approx(2000.0)

    def test_no_events_is_a_noop_not_a_guess(self):
        from data import corporate_actions as CA
        df = self._bonus_frame()
        same = CA.adjust_frame(df, [])
        assert same["close"].tolist() == df["close"].tolist()   # unchanged

    def test_load_events_absent_file_is_empty(self, tmp_path):
        from data import corporate_actions as CA
        assert CA.load_events(tmp_path / "nope.json") == {}     # no fake table

    def test_load_events_parses_and_drops_junk(self, tmp_path):
        import json
        from data import corporate_actions as CA
        f = tmp_path / "ca.json"
        f.write_text(json.dumps([
            {"symbol": "reliance", "ex_date": "2024-01-04", "factor": 2.0, "type": "bonus"},
            {"symbol": "BADFACTOR", "ex_date": "2024-01-04", "factor": 1.0},   # no-op → dropped
            {"symbol": "NODATE", "factor": 3.0},                                # junk → dropped
        ]))
        ev = CA.load_events(f)
        assert set(ev) == {"RELIANCE"} and ev["RELIANCE"][0]["factor"] == 2.0

    def test_get_ohlcv_applies_adjustment_end_to_end(self, tmp_path, monkeypatch):
        import json
        import pandas as pd
        from data import bhavcopy_store as BS
        from data import corporate_actions as CA
        f = tmp_path / "ca.json"
        f.write_text(json.dumps([{"symbol": "TESTCO", "ex_date": "2024-01-04",
                                  "factor": 2.0, "type": "bonus"}]))
        monkeypatch.setenv("QT_CA_EVENTS_FILE", str(f))
        monkeypatch.setattr(BS, "_store", {"TESTCO": self._bonus_frame()}, raising=False)
        BS.reload_corporate_actions()
        out = BS.get_ohlcv("TESTCO")
        assert CA.is_continuous(out) is True                 # adjust-on-read worked

    def test_verify_ca_adjustment_fails_closed_without_a_table(self, monkeypatch):
        from core import data_integrity as DI
        from data import corporate_actions as CA
        monkeypatch.setattr(CA, "load_events", lambda *a, **k: {})
        assert DI.verify_ca_adjustment()["passed"] is False   # no table ⇒ never PASS


class TestSurvivorship:
    """A historical study must see stocks that later delisted; today's survivors
    alone are biased. And with no history on file, the code must SAY so, not fake."""

    def test_point_in_time_includes_then_delisted_names(self, tmp_path):
        import json
        from data import nse_universe as U
        f = tmp_path / "hist.json"
        f.write_text(json.dumps([
            {"symbol": "SURVIVOR", "listed": "2010-01-01"},
            {"symbol": "DELISTED", "listed": "2010-01-01", "delisted": "2020-06-01"},
            {"symbol": "FUTURE", "listed": "2023-01-01"},
        ]))
        # as of 2018 both SURVIVOR and DELISTED traded; FUTURE not yet listed
        r = U.point_in_time_universe("2018-01-01", path=f)
        assert r["survivorship_complete"] is True
        assert set(r["symbols"]) == {"SURVIVOR", "DELISTED"}
        # as of 2021 the delisted name is gone and FUTURE (lists 2023) not yet in
        r2 = U.point_in_time_universe("2021-01-01", path=f)
        assert set(r2["symbols"]) == {"SURVIVOR"}
        # as of 2024 SURVIVOR + FUTURE trade, DELISTED stays gone
        r3 = U.point_in_time_universe("2024-01-01", path=f)
        assert set(r3["symbols"]) == {"SURVIVOR", "FUTURE"}

    def test_no_history_flags_bias_not_fakes_it(self, tmp_path):
        from data import nse_universe as U
        r = U.point_in_time_universe("2020-01-01", path=tmp_path / "nope.json")
        assert r["survivorship_complete"] is False and "biased" in r["note"]


class TestEvidenceLevels:
    def test_alpha_is_e0_infra_is_e2(self):
        assert EL.level_of("Strategy alpha (the edge)")[0] == EL.E0
        assert EL.level_of("Execution engine")[0] == EL.E2

    def test_report_sorts_least_trusted_first(self):
        rows = EL.report()
        assert rows[0]["level"] <= rows[-1]["level"]
        assert any("alpha" in r["capability"].lower() for r in rows)
        assert "unproven" in EL.headline()

    def test_promotion_requires_objective_evidence_no_belief(self):
        # a capability at E0 cannot jump to E3 on developer say-so; each step is
        # gated by an OBJECTIVE artifact and only ONE level at a time.
        cap = "Strategy alpha (the edge)"
        assert EL.level_of(cap)[0] == EL.E0
        # empty / hand-wave artifact → refused, with the exact unmet criteria
        r = EL.promote(cap, {})
        assert r["promoted"] is False and r["unmet"]         # E0→E1 needs review
        # E0→E1 only with a review record
        assert EL.promote(cap, {"reviewed": True})["promoted"] is True
        # E1→E2 only with passing tests
        assert EL.promote(cap, {"reviewed": True})["promoted"] is False   # wrong gate
        assert EL.promote(cap, {"tests_passed": 5, "all_pass": True})["promoted"] is True
        # E2→E3 needs the FULL historical gauntlet — a partial artifact is refused
        weak = {"dsr": 0.99, "reality_check_p": 0.01}         # missing power, regimes…
        assert EL.promote(cap, weak)["promoted"] is False
        strong = {"data_clean": True, "dsr": 0.97, "reality_check_p": 0.01,
                  "fdr_corrected": True, "power": 0.85, "net_expectancy_r": 0.22,
                  "profit_factor": 1.5, "regimes_positive": 2,
                  "beats_benchmark": True, "block_ci_lower": 0.08,
                  "source": "harness"}
        assert EL.promote(cap, strong)["promoted"] is True
        assert EL.level_of(cap)[0] == EL.E3
        EL.demote(cap, EL.E0, "reset for other tests")       # withdraw is always allowed

    def test_next_gate_is_transparent(self):
        g = EL.next_gate("Cost model (values)")              # at E0
        assert g["next"] == "E1 code-reviewed" and g["requires"]

    def test_demote_is_always_allowed(self):
        cap = "Execution engine"                             # E2
        assert EL.level_of(cap)[0] == EL.E2
        EL.demote(cap, EL.E1, "regression found")
        assert EL.level_of(cap)[0] == EL.E1
        EL.promote(cap, {"tests_passed": 1, "all_pass": True})   # restore E2
        assert EL.level_of(cap)[0] == EL.E2
