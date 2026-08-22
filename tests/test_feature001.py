"""FEATURE-001: Trend/RS features, retired Core SEPA, no misleading licence copy."""
from __future__ import annotations

import pandas as pd

from product.sepa_setup import score_sepa
from research.feature001.analyze import classify_family_feature
from research.feature001.constants import TREND_VERSION
from research.feature001.forward_spec import forward_record_template
from research.feature001.trend_features import compute_trend_features
from research.sepa.status import CORE_SEPA_STATUS


def _uptrend(periods: int = 280) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=periods, freq="B")
    close = pd.Series([80 + i * 0.6 for i in range(periods)], index=index)
    return pd.DataFrame(
        {
            "open": close - 0.3,
            "high": close + 1.0,
            "low": close - 0.8,
            "close": close,
            "volume": [150000] * periods,
        },
        index=index,
    )


def test_core_sepa_is_retired_benchmark():
    assert CORE_SEPA_STATUS == "RETIRED_RESEARCH_BENCHMARK"
    from research.sepa import CORE_SEPA_STATUS as exported
    assert exported == "RETIRED_RESEARCH_BENCHMARK"


def test_trend_features_are_not_a_single_flag():
    vec = compute_trend_features(_uptrend())
    assert vec["version"] == TREND_VERSION
    assert vec["available"] is True
    for key in (
        "price_gt_sma50", "price_gt_sma150", "price_gt_sma200",
        "sma50_gt_sma150", "sma50_gt_sma200", "sma150_gt_sma200",
        "sma200_rising", "dist_above_52w_low_pct", "dist_from_52w_high_pct",
        "structure_pass", "n_structure_passed", "pct_above_sma200",
        "sma200_slope_pct", "ma_spread_50_200_pct",
    ):
        assert key in vec
        assert vec[key] is not None
    assert vec["structure_pass"] is True
    assert vec["n_structure_passed"] == 7


def test_trend_features_are_point_in_time():
    hist = _uptrend(280)
    future = pd.DataFrame(
        {
            "open": [hist["close"].iloc[-1] + 5],
            "high": [hist["close"].iloc[-1] + 8],
            "low": [hist["close"].iloc[-1] + 4],
            "close": [hist["close"].iloc[-1] + 7],
            "volume": [200000],
        },
        index=pd.date_range(hist.index[-1] + pd.Timedelta(days=1), periods=1, freq="B"),
    )
    a = compute_trend_features(hist)
    leaked = compute_trend_features(pd.concat([hist, future]))
    asof = compute_trend_features(pd.concat([hist, future]).iloc[:-1])
    assert a["n_structure_passed"] == asof["n_structure_passed"]
    assert a["pct_above_sma200"] == asof["pct_above_sma200"]
    # A later bar may change the live vector — that is expected.
    assert leaked["price"] != a["price"]


def test_ideas_headline_is_not_a_sepa_licence():
    sepa = score_sepa(_uptrend())
    assert sepa["verdict"] == "STRONG"
    assert "MEETS SEPA" not in sepa["headline"]
    assert "TREND QUALITY" in sepa["headline"]
    assert "not Core SEPA" in sepa["disclaimer"]


def test_forward_spec_is_shadow_only():
    spec = forward_record_template()
    assert spec["execution"] is False
    assert spec["paper"] is False
    assert spec["autopilot"] is False
    assert spec["activation"] == "documented_only"


def test_classification_vocab_is_prespecified():
    assert classify_family_feature(
        n=10, year_deltas={}, overall_delta=0.2, residual_rho=0.2,
        residual_p=0.01, tail_improved=True, rank_spread=0.1,
    ) == "INSUFFICIENT_DATA"
    cls = classify_family_feature(
        n=80,
        year_deltas={str(y): 0.1 for y in range(2020, 2026)},
        overall_delta=0.12,
        residual_rho=0.15,
        residual_p=0.01,
        tail_improved=False,
        rank_spread=0.1,
    )
    assert cls == "POSITIVE_RANK_FEATURE"
    neg = classify_family_feature(
        n=80,
        year_deltas={str(y): -0.1 for y in range(2020, 2026)},
        overall_delta=-0.12,
        residual_rho=-0.1,
        residual_p=0.01,
        tail_improved=False,
        rank_spread=-0.1,
    )
    assert neg == "NEGATIVE"
