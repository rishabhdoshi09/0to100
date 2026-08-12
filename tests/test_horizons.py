"""Phase A / A2 — multi-horizon target framework tests (network-free)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.harness import purged_kfold_indices
from research.horizons import (
    CAPABILITY_HORIZONS,
    LEGACY_MH_TARGETS,
    TargetSpec,
    assert_no_label_leakage,
    build_forward_return_labels,
    capability_names,
    get_horizon,
    get_legacy_mh_target,
    purged_splits_for_target,
)
from research.horizons.catalog import absolute_return_target
from research.horizons.labels import classification_from_returns, horizon_agreement
from research.horizons.splits import train_val_test_embargo_slices
from research.horizons.spec import OverlapPolicy


def _close(n=40, start=100.0, step=0.5):
    return pd.Series([start + i * step for i in range(n)])


# ── catalog / specs ──────────────────────────────────────────────────────────

def test_capability_horizons_include_required_set():
    names = set(capability_names())
    for required in ("5d", "10d", "20d", "22d", "60d", "66d", "120d", "130d",
                     "200d", "252d"):
        assert required in names
    assert CAPABILITY_HORIZONS["22d"].bars == 22
    assert CAPABILITY_HORIZONS["252d"].bars == 252


def test_adding_horizon_does_not_require_harness_change():
    # New TargetSpec consumes harness via purged_splits_for_target only.
    t = absolute_return_target("66d")
    splits = purged_splits_for_target(300, t, k=5)
    assert len(splits) == 5
    # Same underlying function the harness exposes:
    direct = purged_kfold_indices(300, k=5, embargo=t.effective_embargo_bars,
                                  label_horizon=t.effective_purge_bars)
    assert len(direct) == len(splits)


def test_invalid_horizon_bars_rejected():
    from research.horizons.spec import HorizonSpec
    with pytest.raises(ValueError):
        HorizonSpec(name="bad", bars=0)
    with pytest.raises(KeyError):
        get_horizon("not_a_horizon")


# ── label leakage / temporal order ───────────────────────────────────────────

class TestLabelConstruction:
    def test_target_timestamps_never_precede_feature_timestamps(self):
        close = _close(30)
        dates = [f"d{i:03d}" for i in range(30)]
        lf = build_forward_return_labels(
            close, absolute_return_target("5d"), dates=dates
        )
        assert not lf.empty
        for _, row in lf.frame.iterrows():
            assert row["target_pos"] > row["feature_pos"]
            assert str(row["target_ts"]) > str(row["feature_ts"]) or (
                row["target_pos"] > row["feature_pos"]
            )

    def test_feature_rows_cannot_include_future_exit_without_drop(self):
        close = _close(20)
        lf = build_forward_return_labels(close, absolute_return_target("10d"))
        # Last 10 bars lack exits and must be absent.
        assert lf.frame["feature_pos"].max() == 20 - 10 - 1
        assert len(lf.frame) == 20 - 10

    def test_raw_return_matches_manual_forward(self):
        close = _close(25, start=100.0, step=1.0)
        lf = build_forward_return_labels(close, absolute_return_target("5d"))
        i = int(lf.frame["feature_pos"].iloc[0])
        expected = float(close.iloc[i + 5] / close.iloc[i] - 1.0)
        assert lf.frame["raw_return"].iloc[0] == pytest.approx(expected)

    def test_costs_reduce_net_return(self):
        close = _close(25)
        t = TargetSpec(
            horizon=get_horizon("5d"),
            kind="absolute_return",
            cost_pct_roundtrip=0.01,
        )
        lf = build_forward_return_labels(close, t)
        assert (lf.frame["net_return"] == lf.frame["raw_return"] - 0.01).all()


# ── overlap / embargo ────────────────────────────────────────────────────────

class TestOverlapAndEmbargo:
    def test_purged_splits_match_harness_policy(self):
        t = absolute_return_target("10d")
        splits = purged_splits_for_target(100, t, k=4)
        for train, test in splits:
            assert_no_label_leakage(
                train, test, label_horizon=t.bars, mode="harness"
            )

    def test_walk_forward_embargo_strict(self):
        t = absolute_return_target("5d")
        parts = train_val_test_embargo_slices(80, t)
        assert parts["train"].size and parts["val"].size and parts["test"].size
        assert_no_label_leakage(
            parts["train"], parts["test"], label_horizon=t.bars, mode="strict"
        )
        assert_no_label_leakage(
            parts["val"], parts["test"], label_horizon=t.bars, mode="strict"
        )
        assert_no_label_leakage(
            parts["train"], parts["val"], label_horizon=t.bars, mode="strict"
        )

    def test_allow_with_warning_still_returns_splits(self):
        t = TargetSpec(
            horizon=get_horizon("5d"),
            overlap_policy=OverlapPolicy.ALLOW_WITH_WARNING,
        )
        assert purged_splits_for_target(50, t, k=3)


# ── legacy 1/5/10d reproducibility ───────────────────────────────────────────

class TestLegacyMultiHorizonRepro:
    def test_legacy_thresholds_match_ml_module(self):
        # Import the live module's private table — research must stay in sync.
        from ml.multi_horizon import _HORIZONS

        for name, cfg in _HORIZONS.items():
            tgt = get_legacy_mh_target(name)
            assert tgt.bars == cfg["shift"]
            assert tgt.buy_thresh == cfg["buy_thresh"]
            assert tgt.sell_thresh == cfg["sell_thresh"]

    def test_classification_labels_match_multi_horizon_formula(self):
        from ml.multi_horizon import _HORIZONS, _LABEL_BUY, _LABEL_SELL, _LABEL_HOLD

        close = _close(60, start=100.0, step=0.8)
        for name, cfg in _HORIZONS.items():
            shift = cfg["shift"]
            fwd = close.pct_change(shift).shift(-shift)
            legacy = pd.Series(np.nan, index=close.index, dtype=float)
            legacy[fwd > cfg["buy_thresh"]] = _LABEL_BUY
            legacy[fwd < cfg["sell_thresh"]] = _LABEL_SELL
            legacy[(fwd >= cfg["sell_thresh"]) & (fwd <= cfg["buy_thresh"])] = _LABEL_HOLD

            tgt = get_legacy_mh_target(name)
            lf = build_forward_return_labels(close, tgt)
            # Compare on overlapping valid index
            for idx, row in lf.frame.iterrows():
                assert row["label"] == legacy.loc[idx]

    def test_legacy_targets_registered(self):
        assert set(LEGACY_MH_TARGETS) == {"1d", "5d", "10d"}


def test_horizon_agreement_helper():
    summary = horizon_agreement({"5d": "BUY", "10d": "BUY", "22d": "HOLD"})
    assert summary["action"] == "BUY"
    assert summary["agreement"] == 2
    assert summary["dispersion"] == pytest.approx(1 / 3, abs=1e-3)


def test_classification_helper_thresholds():
    labels = classification_from_returns(
        [0.02, -0.02, 0.0], buy_thresh=0.01, sell_thresh=-0.01
    )
    assert list(labels) == [1.0, -1.0, 0.0]
