"""
Cross-sectional momentum engine (EXP-003) — network-free synthetic tests.

Prove the PURE logic: a planted momentum edge is captured (winners keep winning →
positive strategy stream), pure noise is not, the liquidity filter drops
untradeable names, and the annualiser matches a known stream.
"""
import numpy as np

from scan import momentum as MOM


def _series(fn, T):
    return np.array([fn(t) for t in range(T)], dtype=float)


class TestMomentumEngine:
    def test_planted_momentum_is_captured(self):
        T = 400
        closes, volumes = {}, {}
        # 20 persistent up-trenders (high momentum, keep rising) …
        for i in range(20):
            closes[f"UP{i}"] = _series(lambda t: 100.0 * (1.003 ** t), T)
            volumes[f"UP{i}"] = np.full(T, 2_000_000.0)
        # … and 20 persistent down-trenders (low momentum)
        for i in range(20):
            closes[f"DN{i}"] = _series(lambda t: 100.0 * (0.998 ** t), T)
            volumes[f"DN{i}"] = np.full(T, 2_000_000.0)
        bench = _series(lambda t: 100.0 * (1.0005 ** t), T)
        out = MOM.build_momentum_series(
            closes, volumes, bench, top_n=10, lookback=100, skip=5,
            rebalance=10, min_turnover_cr=5.0)
        assert out["n_rebalances"] > 5
        assert float(np.mean(out["strat_returns"])) > 0     # momentum earns here
        # it should be holding the up-trenders → beat the benchmark on average
        assert float(np.mean(out["strat_returns"])) > float(np.mean(out["bench_returns"]))

    def test_pure_noise_has_no_positive_edge(self):
        rng = np.random.default_rng(0)
        T = 400
        closes, volumes = {}, {}
        for i in range(40):
            steps = rng.normal(0, 0.02, T)
            closes[f"S{i}"] = 100.0 * np.cumprod(1 + steps)
            volumes[f"S{i}"] = np.full(T, 2_000_000.0)
        bench = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, T))
        out = MOM.build_momentum_series(closes, volumes, bench, top_n=10,
                                        lookback=100, skip=5, rebalance=10)
        # random data → mean strategy return should be small (no real edge)
        assert abs(float(np.mean(out["strat_returns"]))) < 0.03

    def test_liquidity_filter_drops_illiquid(self):
        T = 200
        closes = {"LIQ": _series(lambda t: 100.0 * (1.001 ** t), T),
                  "ILLIQ": _series(lambda t: 100.0 * (1.001 ** t), T)}
        volumes = {"LIQ": np.full(T, 5_000_000.0),      # ₹50cr/day → passes
                   "ILLIQ": np.full(T, 100.0)}          # tiny → dropped
        keep = MOM._liquid_symbols(closes, volumes, min_turnover_cr=5.0)
        assert "LIQ" in keep and "ILLIQ" not in keep

    def test_annualise_known_stream(self):
        # 12 months of +1% each → ~12.7% CAGR, positive Sharpe, no drawdown
        a = MOM.annualise([0.01] * 12, periods_per_year=12)
        assert 12.0 < a["cagr_pct"] < 13.5
        assert a["max_dd_pct"] == 0.0 and a["sharpe"] > 0

    def test_too_little_history_is_empty(self):
        closes = {"A": np.full(50, 100.0)}
        volumes = {"A": np.full(50, 2_000_000.0)}
        out = MOM.build_momentum_series(closes, volumes, np.full(50, 100.0),
                                        lookback=252)
        assert out["n_rebalances"] == 0     # not enough data → no claim, no crash
