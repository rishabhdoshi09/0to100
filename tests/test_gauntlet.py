"""
The Historical Gauntlet (E1–E6) — network-free, synthetic-data tests.

These prove the PIPELINE is correct on controlled inputs: a planted edge PASSes,
noise does not, a long-beta stream is denied by the alpha gate, the dataset gate
ABORTS when data is missing, records are immutable, experiments are reproducible,
and a mutated knob breaks the freeze. The real market run happens where the data
lives; here we verify the machinery that will judge it.
"""
import dataclasses

import numpy as np
import pytest

from scan import signal_backtest as BT
from gauntlet import ledger as L
from gauntlet import runner as R
from gauntlet import validator as V
from gauntlet import registry as REG
from gauntlet import freeze as FZ


# ── E1: simulation timing + ledger ────────────────────────────────────────────

class TestSimulateTiming:
    def test_simulate_contract_is_unchanged(self):
        # delegation must not change the (outcome, r) that 180+ money tests pin
        h = np.array([101.0, 103.0]); l = np.array([100.0, 101.0])
        c = np.array([100.5, 102.0])
        # risk = entry-stop = 2; target +2 → 1.0R (delegation must preserve this)
        assert BT._simulate(100.0, 98.0, 102.0, h, l, c) == ("WIN", 1.0)

    def test_timed_returns_fill_and_exit_offsets(self):
        # fills bar 0, target 102 hit bar 1
        h = np.array([101.0, 103.0]); l = np.array([100.0, 101.0])
        c = np.array([100.5, 102.0])
        o, r, e, x = BT._simulate_timed(100.0, 98.0, 102.0, h, l, c)
        assert o == "WIN" and e == 0 and x == 1

    def test_no_fill_has_negative_offsets(self):
        h = np.array([99.0, 99.5]); l = np.array([98.0, 98.5])
        c = np.array([98.5, 99.0])
        o, r, e, x = BT._simulate_timed(100.0, 98.0, 102.0, h, l, c)
        assert o == "NO_FILL" and e == -1 and x == -1


def _rec(strategy, net_R, bench_pct, regime="BULL", i=0):
    import pandas as pd
    dt = pd.Timestamp("2023-01-01") + pd.Timedelta(days=3 * i)
    return L.TradeRecord(
        symbol=f"S{i%20}", signal_id=strategy, signals=(strategy,),
        entry_datetime=str(dt), exit_datetime=str(dt + pd.Timedelta(days=5)),
        entry_price=100.0, exit_price=100.0 + net_R * 4.0, holding_period=5,
        stop_price=96.0, target_price=108.0, gross_R=net_R + 0.02, net_R=net_R,
        costs=0.02, slippage_used=0.10, exit_reason="WIN" if net_R > 0 else "LOSS",
        regime=regime, confidence=0.6, calibration_version="test",
        benchmark_return_during_trade=bench_pct, factor_returns_during_trade={})


class TestLedger:
    def test_trade_record_is_immutable(self):
        r = _rec("edge", 0.3, 0.005)
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.net_R = 9.9

    def test_builder_enriches_benchmark_over_window(self):
        import pandas as pd
        idx = pd.date_range("2023-01-01", periods=20, freq="D")
        close = pd.Series(np.linspace(100, 110, 20), index=idx)   # +10% over window
        b = L.LedgerBuilder()
        b({"symbol": "X", "signals": ("edge",), "signal_id": "edge",
           "entry_datetime": idx[0], "exit_datetime": idx[10],
           "entry_price": 100.0, "exit_price": 104.0, "holding_period": 10,
           "stop_price": 96.0, "target_price": 108.0, "gross_R": 1.0, "net_R": 0.98,
           "costs": 0.02, "slippage_used": 0.10, "exit_reason": "WIN",
           "regime": "BULL", "confidence": 0.5, "calibration_version": "t"})
        led = b.finalize(index_close=close)
        assert led[0].benchmark_return_during_trade == pytest.approx(
            close.iloc[10] / close.iloc[0] - 1, rel=1e-6)

    def test_per_strategy_groups_by_signal(self):
        led = [_rec("a", 0.1, 0.0, i=0), _rec("b", 0.2, 0.0, i=1)]
        grouped = L.per_strategy(led)
        assert set(grouped) == {"a", "b"}


# ── E2: the battery + verdict mapping ─────────────────────────────────────────

class TestEvaluateStrategy:
    def test_planted_edge_promotes_with_alpha(self):
        rng = np.random.default_rng(1)
        recs = [_rec("edge", float(rng.normal(0.35, 1.0)),
                     float(rng.normal(0.0, 0.02)), i=i) for i in range(300)]
        res = R.evaluate_strategy(recs, n_trials=1, seed=1)
        assert res["harness_verdict"] == "PROMOTE"
        assert res["beats_benchmark"] is True and res["benchmark_tested"] is True

    def test_pure_noise_does_not_promote(self):
        rng = np.random.default_rng(2)
        recs = [_rec("noise", float(rng.normal(0.0, 1.0)),
                     float(rng.normal(0.0, 0.02)), i=i) for i in range(300)]
        assert R.evaluate_strategy(recs, n_trials=1, seed=1)["harness_verdict"] != "PROMOTE"

    def test_long_beta_is_denied_by_alpha_gate(self):
        # deterministic long-beta stream (same construction the harness proves on):
        # returns are entirely explained by the benchmark → α≈0, profitable but no skill
        rng = np.random.default_rng(53)
        _mkt = rng.normal(0.05, 1.0, 500)                  # advance RNG as in harness test
        mom = rng.normal(0.30, 1.0, 500)
        strat = 1.0 * mom + rng.normal(0.0, 0.3, 500)
        # bench_R = benchmark_return_during_trade / risk_frac(0.04); make it == mom
        recs = [_rec("beta", float(strat[i]), float(mom[i] * 0.04), i=i)
                for i in range(500)]
        res = R.evaluate_strategy(recs, n_trials=1, seed=1)
        # profitable but no alpha → harness REJECTs as exposure, not skill
        assert res["harness_verdict"] == "REJECT"
        assert res["beats_benchmark"] is False


# ── E2 end-to-end: run_gauntlet with an injected ledger ───────────────────────

class TestRunGauntlet:
    def _mixed_ledger(self):
        rng = np.random.default_rng(7)
        led = []
        for i in range(300):
            led.append(_rec("edge", float(rng.normal(0.35, 1.0)),
                            float(rng.normal(0.0, 0.02)), i=i))
        for i in range(300):
            led.append(_rec("noise", float(rng.normal(0.0, 1.0)),
                            float(rng.normal(0.0, 0.02)), i=i))
        return led

    def test_edge_passes_noise_does_not(self, tmp_path, monkeypatch):
        monkeypatch.setattr(REG, "_REG_FILE", tmp_path / "exp.jsonl")
        monkeypatch.setattr(FZ, "_FREEZE_FILE", tmp_path / "freeze.json")
        raw = R.run_gauntlet(ledger=self._mixed_ledger(), skip_validation=True, seed=1)
        assert raw["aborted"] is False
        assert raw["strategies"]["edge"]["verdict"] == "PASS"
        assert raw["strategies"]["noise"]["verdict"] in ("FAIL", "INCONCLUSIVE")
        assert raw["experiment"]["experiment_id"]                # E5 stamped

    def test_verdict_is_one_of_three_words(self):
        raw = R.run_gauntlet(ledger=self._mixed_ledger(), skip_validation=True, seed=1)
        for s in raw["strategies"].values():
            assert s["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")


# ── E4: the abort-on-fail dataset gate ────────────────────────────────────────

class TestValidator:
    def test_missing_data_aborts(self):
        # no CA table, no index store, empty bhav store in the test env → must fail
        v = V.validate()
        assert v["ok"] is False
        assert "corporate_actions_loaded" in v["failed"]
        assert "index_history_present" in v["failed"]

    def test_gauntlet_aborts_when_validation_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(REG, "_REG_FILE", tmp_path / "exp.jsonl")
        raw = R.run_gauntlet(skip_validation=False)
        assert raw["aborted"] is True and raw["strategies"] == {}


# ── E5: reproducible experiment stamps ────────────────────────────────────────

class TestRegistry:
    def test_dataset_hash_is_deterministic(self):
        fp = {"n_trades": 100, "strategies": ["a", "b"]}
        assert REG.dataset_hash(fp) == REG.dataset_hash(dict(fp))

    def test_register_records_all_fields(self, tmp_path, monkeypatch):
        monkeypatch.setattr(REG, "_REG_FILE", tmp_path / "exp.jsonl")
        rec = REG.register("cfg123", {"n_trades": 10}, seed=42)
        for k in ("experiment_id", "git_commit", "dataset_hash", "config_hash",
                  "seed", "timestamp"):
            assert k in rec
        assert rec["seed"] == 42


# ── E6: evidence freeze ───────────────────────────────────────────────────────

class TestFreeze:
    def test_verify_detects_a_changed_knob(self, monkeypatch, tmp_path):
        monkeypatch.setattr(FZ, "_FREEZE_FILE", tmp_path / "freeze.json")
        fr = FZ.freeze()
        assert FZ.verify_unchanged(fr["hash"])["ok"] is True
        # mutate a result-determining threshold → the freeze must break
        import research.harness as HH
        monkeypatch.setattr(HH, "_PROMOTE_P", 0.80)
        assert FZ.verify_unchanged(fr["hash"])["ok"] is False


# ── E3: the report ────────────────────────────────────────────────────────────

class TestReport:
    def test_report_has_committee_sections(self, tmp_path, monkeypatch):
        from gauntlet import report as RP
        monkeypatch.setattr(REG, "_REG_FILE", tmp_path / "exp.jsonl")
        monkeypatch.setattr(FZ, "_FREEZE_FILE", tmp_path / "freeze.json")
        rng = np.random.default_rng(9)
        led = [_rec("edge", float(rng.normal(0.35, 1.0)),
                    float(rng.normal(0.0, 0.02)), i=i) for i in range(300)]
        raw = R.run_gauntlet(ledger=led, skip_validation=True, seed=1)
        rep = RP.build_report(raw)
        assert rep["status"] == "COMPLETE"
        assert rep["assumptions"] and rep["known_limitations"]
        assert "PASS" in rep["verdict_tally"]
        md = RP.to_markdown(rep)
        assert "Evidence Report" in md and "Assumptions" in md
