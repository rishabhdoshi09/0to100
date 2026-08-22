"""SEPA-003 PIT regime / sector / protocol tests. No live orders."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.sepa003.constants import FORBIDDEN_LABELS, R2_DIR
from research.sepa003.regime import append_future_invariant, classify_regime_level, nifty50_equal_weight_proxy
from research.sepa003.sector import load_sector_map_v1, sector_of
from tests.test_sepa_001r2 import _liq_frame


def test_prior_r2_files_untouched_by_import():
    assert (R2_DIR / "SEPA_001R2_DECISION.md").exists()
    text = (R2_DIR / "SEPA_001R2_DECISION.md").read_text()
    assert "KEEP RESEARCH-ONLY" in text


def test_regime_append_future_invariance():
    idx = pd.bdate_range("2019-01-02", periods=400)
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 400))), index=idx)
    assert append_future_invariant(close, 250) is True
    assert append_future_invariant(close, 300) is True
    # Many cuts
    for cut in (220, 260, 320, 360):
        assert append_future_invariant(close, cut) is True


def test_regime_does_not_use_future_return_in_current_label():
    idx = pd.bdate_range("2018-01-02", periods=260)
    close = pd.Series(np.linspace(100, 160, 260), index=idx)
    table = classify_regime_level(close)
    last = table.iloc[-1]
    # A crash appended later must not rewrite yesterday
    crashed = pd.concat([close, pd.Series([80.0], index=[idx[-1] + pd.Timedelta(days=3)])])
    full = classify_regime_level(crashed)
    assert str(table.iloc[-1]["regime"]) == str(full.loc[table.index[-1], "regime"])
    assert last["regime"] in {"STRONG_BULL", "BULL", "SIDEWAYS", "CORRECTION", "BEAR", "UNKNOWN"}


def test_sector_unknown_not_fabricated():
    pack = load_sector_map_v1()
    assert pack["never_infers_from_price"] is True
    assert pack["sector_identity_pit"] is False
    assert sector_of("RELIANCE", pack["map"]) != "UNKNOWN"
    assert sector_of("THIS_SYMBOL_DOES_NOT_EXIST_ZZZ", pack["map"]) == "UNKNOWN"


def test_proxy_uses_only_member_bars():
    frames = {
        "RELIANCE": _liq_frame(300, start=1000.0, step=0.4, volume=200_000),
        "TCS": _liq_frame(300, start=2000.0, step=0.5, volume=200_000),
        "OTHER": _liq_frame(300, start=50.0, step=-0.2, volume=200_000),
    }
    # Force names onto NIFTY50-like keys already in overlay
    lvl = nifty50_equal_weight_proxy(frames)
    assert len(lvl) > 50
    # OTHER is not NIFTY50 so a dump later should not change historical proxy if we only use members
    frames2 = dict(frames)
    extra = _liq_frame(300, start=10.0, step=3.0, volume=200_000)
    frames2["FAKE"] = extra
    lvl2 = nifty50_equal_weight_proxy(frames2)
    # FAKE is not in NIFTY50 list
    common = lvl.index.intersection(lvl2.index)
    assert np.allclose(lvl.loc[common].values, lvl2.loc[common].values, rtol=1e-6)


def test_protocol_forbids_oos_relabel():
    proto = (R2_DIR.parent / "SEPA-003" / "SEPA_003_RESEARCH_PROTOCOL.md").read_text()
    assert "NEW_HYPOTHESIS" in proto
    assert "VALIDATED_EDGE" in proto
    for lab in FORBIDDEN_LABELS:
        assert lab in proto
    hyp = json.loads((R2_DIR.parent / "SEPA-003" / "sepa_003_hypotheses.json").read_text())
    assert hyp["declared_before_results"] is True
    assert [h["id"] for h in hyp["primary"]] == [f"H{i}" for i in range(1, 9)]
