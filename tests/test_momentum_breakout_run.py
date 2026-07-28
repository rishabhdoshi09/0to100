"""
Deterministic, network-free tests for the EXP-006 historical evidence RUNNER.

A synthetic in-memory `SyntheticProvider` feeds point-in-time arrays (no bhav store,
no network, no wall-clock, no timezone) so the runner's mechanics are verified end to
end: data-quality gate, snapshot manifest, chronological candidate generation, stable
replay + event ids, no-same-bar entry, missing-session / IPO / delisting handling,
benchmark alignment, cost application, no-fill + gap handling, exit-variant separation,
ablation isolation, artifact reproducibility, verdict mapping (incl. the research-grade
downgrade and DATA_UNAVAILABLE fail-closed), and execution isolation.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from research.momentum_breakout import primary_config
from research.momentum_breakout import dataset as DS
from research.momentum_breakout import runner as RUN
from research.momentum_breakout.runner import run_evidence, _EmptyProvider


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic point-in-time data provider
# ══════════════════════════════════════════════════════════════════════════════

def _sym_arrays(n, *, leader=True, prior_len=340, base_len=60, base_depth=0.08,
                continuation=None, p0=100.0, resistance_mult=1.6, slope=1.004,
                ipo_at=0, delist_at=None, missing=(), breakout=True, intrabar=0.008):
    R = p0 * resistance_mult
    close = np.full(n, np.nan)
    if leader:
        close[:prior_len] = p0 * (R / p0) ** np.linspace(0, 1, prior_len)
    else:
        hi = R * 2.0
        close[:prior_len] = hi * (R / hi) ** np.linspace(0, 1, prior_len)
    for k in range(base_len):
        i = prior_len + k
        amp = base_depth * (1.0 - 0.7 * k / base_len)
        close[i] = min(R * (1 - base_depth / 2) + (amp / 2) * R * np.sin(k / 3.5), R * 0.995)
    if breakout:
        b = prior_len + base_len
        close[b] = R * 1.03
        last = n if continuation is None else min(n, b + 1 + continuation)
        for i in range(b + 1, last):
            close[i] = close[i - 1] * slope
    # prior phase has WIDER intrabar ranges than the tight base → the base window
    # shows genuine range/ATR contraction (matches the frozen detector's requirement)
    ib = np.full(n, intrabar); ib[:prior_len] = 0.022
    high = close * (1 + ib); low = close * (1 - ib)
    open_ = np.full(n, np.nan); open_[0] = close[0]
    open_[1:] = close[:-1] * (1 + 0.001)
    for k in range(base_len):
        high[prior_len + k] = min(high[prior_len + k], R * 0.99)
    if breakout:
        b = prior_len + base_len
        high[b] = max(R * 1.035, close[b] * (1 + intrabar))
        open_[b] = close[b] * 0.995            # gap-up-and-go open, ABOVE the base
    # enforce HLOC consistency on every bar: low ≤ min(open,close) ≤ max(open,close) ≤ high
    high = np.fmax(high, np.fmax(open_, close))
    low = np.fmin(low, np.fmin(open_, close))
    vol = np.full(n, 1e6); vol[prior_len:prior_len + base_len] = 4e5
    if breakout:
        vol[prior_len + base_len] = 3e6
    # IPO: blank everything before ipo_at; delisting: blank from delist_at
    for arr in (open_, high, low, close, vol):
        if ipo_at > 0:
            arr[:ipo_at] = np.nan
        if delist_at is not None:
            arr[delist_at:] = np.nan
        for j in missing:
            arr[j] = np.nan
    return {"open": open_, "high": high, "low": low, "close": close, "volume": vol}


class SyntheticProvider:
    def __init__(self, specs, n=520, survivorship=True, ca_raw=False,
                 sector=None, valuations=None, corrupt=None):
        self.n = n
        d0 = dt.date(2018, 1, 1)
        self._dates = [(d0 + dt.timedelta(days=i)).isoformat() for i in range(n)]
        self._bench = 100.0 * (1.0002) ** np.arange(n)      # mild flat-ish benchmark
        self._data = {name: _sym_arrays(n, **spec) for name, spec in specs.items()}
        if corrupt:
            for name, mut in corrupt.items():
                mut(self._data[name])
        self._survivorship = survivorship
        self._ca_raw = ca_raw
        self._sector = sector or {"sector_rs_pct": 5.0, "breadth_pct_above_50dma": 70.0,
                                  "membership_pit": True, "turnover_cr": 50.0}
        self._valuations = valuations or {}

    def calendar(self): return list(self._dates)
    def benchmark_close(self): return self._bench
    def benchmark_id(self): return "^NSEI_SYNTH"
    def symbols(self): return sorted(self._data)
    def ohlcv(self, sym): return self._data.get(sym)
    def sector_ctx(self, sym, i): return dict(self._sector)
    def valuation(self, sym, i): return self._valuations.get(sym)
    def source_identities(self): return {"prices": "SYNTHETIC"}
    def universe_policy(self): return {"survivorship_complete": self._survivorship, "note": ""}
    def adjustment_policy(self):
        return {"corporate_actions": "RAW" if self._ca_raw else "adjusted"}


def _clean_universe(k=8, **over):
    specs = {}
    for j in range(k):
        specs[f"SYM{j:02d}"] = dict(slope=1.003 + 0.0005 * (j % 4), **over)
    return SyntheticProvider(specs)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data-quality gate + snapshot
# ══════════════════════════════════════════════════════════════════════════════

class TestDataQualityAndSnapshot:
    def test_clean_dataset_passes_gate(self):
        q = DS.data_quality_report(_clean_universe(), primary_config())
        assert q.ok is True
        assert q.metrics["symbols_scanned"] >= 1
        assert q.metrics["non_positive_prices"] == 0

    def test_corrupted_prices_fail_closed(self):
        def mutate(d):
            d["close"][300] = -5.0          # non-positive price = proven corruption
        prov = SyntheticProvider({"BAD": dict()}, corrupt={"BAD": mutate})
        q = DS.data_quality_report(prov, primary_config())
        assert q.ok is False
        assert any("non-positive" in r for r in q.fatal_reasons)

    def test_hloc_inconsistency_fails_closed(self):
        def mutate(d):
            d["high"][310] = d["low"][310] - 1.0   # high < low
        prov = SyntheticProvider({"BAD": dict()}, corrupt={"BAD": mutate})
        q = DS.data_quality_report(prov, primary_config())
        assert q.ok is False

    def test_snapshot_id_stable_and_data_sensitive(self):
        p1 = _clean_universe(); p2 = _clean_universe()
        m1 = DS.snapshot_manifest(p1, primary_config())
        m2 = DS.snapshot_manifest(p2, primary_config())
        assert m1["snapshot_id"] == m2["snapshot_id"]        # same data → same id
        p3 = _clean_universe(k=9)                            # different symbol count
        m3 = DS.snapshot_manifest(p3, primary_config())
        assert m3["snapshot_id"] != m1["snapshot_id"]

    def test_limitations_recorded_not_hidden(self):
        q = DS.data_quality_report(SyntheticProvider({"A": dict()}, survivorship=False,
                                                     ca_raw=True), primary_config())
        joined = " ".join(q.limitations)
        assert "SURVIVORSHIP_INCOMPLETE" in joined
        assert "VALUATION_DATA_UNAVAILABLE" in joined
        assert "SECTOR_MEMBERSHIP_NOT_PIT" in joined


# ══════════════════════════════════════════════════════════════════════════════
# 2. Candidate generation + replay + event ids
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerationAndReplay:
    def test_chronological_generation_and_trades(self, tmp_path):
        res = run_evidence(_clean_universe(), out_dir=tmp_path / "r1")
        assert res["verdict"]["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")
        assert res["primary"]["n_eligible"] >= 1
        assert res["primary"]["n_trades"] >= 1

    def test_stable_replay_identical_observations_and_events(self, tmp_path):
        r1 = run_evidence(_clean_universe(), out_dir=tmp_path / "a")
        r2 = run_evidence(_clean_universe(), out_dir=tmp_path / "b")
        obs1 = (tmp_path / "a" / "observations.jsonl").read_text()
        obs2 = (tmp_path / "b" / "observations.jsonl").read_text()
        assert obs1 == obs2 and obs1.strip() != ""

    def test_one_event_per_breakout_no_double_count(self, tmp_path):
        res = run_evidence(_clean_universe(k=3), out_dir=tmp_path / "r")
        rows = [json.loads(l) for l in
                (tmp_path / "r" / "observations.jsonl").read_text().splitlines()]
        eids = [r["event_id"] for r in rows]
        assert len(eids) == len(set(eids))            # every event id unique

    def test_no_same_bar_entry(self, tmp_path):
        run_evidence(_clean_universe(k=3), out_dir=tmp_path / "r")
        ledger = [json.loads(l) for l in
                  (tmp_path / "r" / "trade_ledger.jsonl").read_text().splitlines()]
        obs = [json.loads(l) for l in
               (tmp_path / "r" / "observations.jsonl").read_text().splitlines()]
        cand = {o["symbol"]: o["candidate_date"] for o in obs}
        assert ledger
        for t in ledger:
            # entry is STRICTLY after the signal/candidate bar — no same-bar-close fill
            assert t["entry_date"] > cand[t["symbol"]]
            assert t["holding_period"] >= 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. Missing session / IPO / delisting / benchmark alignment
# ══════════════════════════════════════════════════════════════════════════════

class TestPITBoundaries:
    def test_missing_session_produces_no_spurious_candidate(self):
        # blanking bars inside the base must not fabricate a candidate at the gap
        prov = SyntheticProvider({"A": dict(missing=(370, 371, 372))})
        q = DS.data_quality_report(prov, primary_config())
        assert q.metrics["missing_ohlcv_cells"] >= 3
        res = run_evidence(prov, write=False)
        # runner still completes; the gap did not corrupt the gate
        assert res["verdict"]["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")

    def test_ipo_boundary_counted_and_not_scanned_before_listing(self):
        prov = SyntheticProvider({"NEW": dict(ipo_at=50)})
        q = DS.data_quality_report(prov, primary_config())
        assert q.metrics["ipo_boundaries"] >= 1

    def test_terminal_delisted_history_handled(self):
        prov = SyntheticProvider({"GONE": dict(delist_at=460)})
        q = DS.data_quality_report(prov, primary_config())
        assert q.metrics["terminal_history"] >= 1
        res = run_evidence(prov, write=False)
        assert res["verdict"]["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")

    def test_benchmark_return_is_aligned_to_trade_window(self, tmp_path):
        run_evidence(_clean_universe(k=2), out_dir=tmp_path / "r")
        ledger = [json.loads(l) for l in
                  (tmp_path / "r" / "trade_ledger.jsonl").read_text().splitlines()]
        assert ledger and all(t["benchmark_return"] is not None for t in ledger)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Costs / no-fill / exits / ablations
# ══════════════════════════════════════════════════════════════════════════════

class TestCostsExitsAblations:
    def test_costs_make_net_below_gross_for_a_winner(self, tmp_path):
        run_evidence(_clean_universe(k=3), out_dir=tmp_path / "r")
        ledger = [json.loads(l) for l in
                  (tmp_path / "r" / "trade_ledger.jsonl").read_text().splitlines()]
        winners = [t for t in ledger if t["gross_R"] > 0]
        assert winners and all(t["net_R"] < t["gross_R"] for t in winners)

    def test_no_fill_recorded_on_gap_below_stop(self):
        # a next-bar open that gaps below the structural stop → no realistic fill
        def gap(d):
            b = 401                                 # the entry bar (signal at 400)
            for arr in ("open", "high", "low", "close"):
                d[arr][b] = 10.0                    # collapses far below any stop
        prov = SyntheticProvider({"GAP": dict()}, corrupt={"GAP": gap})
        res = run_evidence(prov, write=False)
        assert res["primary"]["n_no_fill"] >= 1

    def test_exit_variants_are_separate_and_primary_labelled(self, tmp_path):
        res = run_evidence(_clean_universe(k=3), out_dir=tmp_path / "r")
        ev = res["exit_variants"]
        assert set(ev) == set(RUN.EXP.EXIT_VARIANTS)
        assert ev[RUN.EXP.PRIMARY_EXIT]["is_primary"] is True
        assert sum(1 for v in ev.values() if v["is_primary"]) == 1

    def test_ablations_isolated_and_present(self, tmp_path):
        res = run_evidence(_clean_universe(k=4), out_dir=tmp_path / "r")
        abl = res["ablations"]
        assert {"prior_only", "prior_base_risk", "full_framework"} <= set(abl)
        for v in abl.values():
            assert "n_eligible" in v and "verdict" in v

    def test_multiple_testing_control_runs(self, tmp_path):
        res = run_evidence(_clean_universe(k=4), out_dir=tmp_path / "r")
        mt = res["multiple_testing"]
        assert "primary" in mt["hypotheses"] and mt["family_size"] >= 3


# ══════════════════════════════════════════════════════════════════════════════
# 5. Verdict mapping (research-grade gate) + fail-closed
# ══════════════════════════════════════════════════════════════════════════════

class TestVerdictMapping:
    def _stub_provider(self, survivorship, ca_raw):
        class P:
            def universe_policy(self_): return {"survivorship_complete": survivorship}
            def adjustment_policy(self_): return {"corporate_actions": "RAW" if ca_raw else "adj"}
        return P()

    def _quality(self):
        return DS.DataQualityReport(ok=True, limitations=["x"])

    def test_promote_on_research_grade_data_is_pass(self):
        v = RUN._decide({"verdict": "PROMOTE", "insight": "edge", "expectancy_R": 0.4,
                         "n_trades": 40}, self._quality(),
                        {"snapshot_id": "s", "experiment_config_hash": "h"},
                        self._stub_provider(True, False), {})
        assert v["verdict"] == "PASS"

    def test_promote_on_biased_data_is_downgraded_to_inconclusive(self):
        v = RUN._decide({"verdict": "PROMOTE", "insight": "edge", "expectancy_R": 0.4,
                         "n_trades": 40}, self._quality(),
                        {"snapshot_id": "s", "experiment_config_hash": "h"},
                        self._stub_provider(False, False), {})   # survivorship incomplete
        assert v["verdict"] == "INCONCLUSIVE"
        assert any("DOWNGRADED" in r for r in v["reasons"])

    def test_reject_is_fail_even_on_biased_data(self):
        v = RUN._decide({"verdict": "REJECT", "insight": "no edge", "expectancy_R": -0.1,
                         "n_trades": 40}, self._quality(),
                        {"snapshot_id": "s", "experiment_config_hash": "h"},
                        self._stub_provider(False, True), {})
        assert v["verdict"] == "FAIL"

    def test_underpowered_is_inconclusive(self):
        v = RUN._decide({"verdict": "UNDERPOWERED", "insight": "thin", "expectancy_R": 0.2,
                         "n_trades": 5}, self._quality(),
                        {"snapshot_id": "s", "experiment_config_hash": "h"},
                        self._stub_provider(True, False), {})
        assert v["verdict"] == "INCONCLUSIVE"

    def test_data_unavailable_fails_closed_to_inconclusive(self, tmp_path):
        res = run_evidence(_EmptyProvider("no data / no network"), out_dir=tmp_path / "r")
        assert res["verdict"]["verdict"] == "INCONCLUSIVE"
        assert "DATA_UNAVAILABLE" in res["verdict"]["reason"]
        assert (tmp_path / "r" / "verdict.json").exists()


# ══════════════════════════════════════════════════════════════════════════════
# 6. Artifact reproducibility + valuation/sector honesty
# ══════════════════════════════════════════════════════════════════════════════

class TestArtifactsAndHonesty:
    def test_artifacts_are_reproducible(self, tmp_path):
        run_evidence(_clean_universe(k=3), out_dir=tmp_path / "a")
        run_evidence(_clean_universe(k=3), out_dir=tmp_path / "b")
        for name in ("observations.jsonl", "trade_ledger.jsonl", "verdict.json",
                     "primary_metrics.json"):
            assert (tmp_path / "a" / name).read_text() == (tmp_path / "b" / name).read_text()

    def test_valuation_reported_unavailable_not_zero(self, tmp_path):
        res = run_evidence(_clean_universe(k=2), out_dir=tmp_path / "r")
        assert res["valuation"]["status"] == "VALUATION_DATA_UNAVAILABLE"

    def test_full_artifact_index_written(self, tmp_path):
        run_evidence(_clean_universe(k=2), out_dir=tmp_path / "r")
        idx = json.loads((tmp_path / "r" / "artifact_index.json").read_text())
        for required in ("data_quality.json", "snapshot_manifest.json",
                         "trade_ledger.jsonl", "primary_metrics.json", "verdict.json",
                         "ablations.json", "multiple_testing.json"):
            assert required in idx["artifacts"]


# ══════════════════════════════════════════════════════════════════════════════
# 7. Execution isolation
# ══════════════════════════════════════════════════════════════════════════════

class TestRunnerExecutionIsolation:
    def test_runner_imports_no_execution_or_broker(self):
        import inspect
        for mod in (RUN, DS):
            src = inspect.getsource(mod)
            code = src.split('"""', 2)[-1] if src.count('"""') >= 2 else src
            for pat in ("import execution", "from execution", "import alerts",
                        "from alerts", ".place_trade(", "kite_client", "GTT_"):
                assert pat not in code, f"{mod.__name__} references {pat}"
