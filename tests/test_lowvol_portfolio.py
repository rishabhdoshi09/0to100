"""PR-2026-11-21-LOWVOL: frozen spec, point-in-time score, single-look held-out year, NW alpha (network-free)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research import lowvol_portfolio as LV
from research import momentum_portfolio as MP


def _frames(n_sym=120, n_days=330, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    out = {}
    for k in range(n_sym):
        vol = 0.005 + 0.0003 * k                      # a clean volatility ladder
        c = 100 * np.exp(np.cumsum(rng.normal(0.0002, vol, n_days)))
        out[f"S{k:03d}"] = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                                         "volume": np.full(n_days, 2e6)}, index=idx)
    return out


def _member(frames):
    return {s: ("2000-01-01", None) for s in frames}


def test_the_spec_is_frozen_and_cannot_create_a_buy():
    assert LV.spec_sha256() == LV.FROZEN_SHA256
    spec = LV.load_spec()
    assert spec["can_create_buy"] is False and spec["can_place_orders"] is False
    assert "alpha" in spec["primary_outcome"]["statistic"]


def test_scores_use_only_bars_up_to_the_cutoff_and_rank_low_vol_first():
    frames = _frames()
    spec = LV.load_spec()
    panel = MP.build_panel(frames)
    i = MP.rebalance_indices(panel["dates"])[-2]
    before = LV.eligible_scores(panel, i, spec, _member(frames))
    poisoned = {s: df.copy() for s, df in frames.items()}
    cut = pd.Timestamp(panel["dates"][i - 1])
    for df in poisoned.values():
        mask = df.index > cut
        df.loc[mask, ["open", "high", "low", "close"]] = (
            df.loc[mask, ["open", "high", "low", "close"]].mul(np.linspace(1, 5, int(mask.sum())), axis=0))
    after = LV.eligible_scores(MP.build_panel(poisoned), i, spec, _member(frames))
    assert before and before == after
    ranked = sorted(before, key=lambda j: -before[j])
    assert panel["symbols"][ranked[0]] < panel["symbols"][ranked[-1]]     # S000 (least volatile) first


def test_the_held_out_year_needs_an_explicit_single_look():
    frames = _frames()
    with pytest.raises(PermissionError):
        LV.evaluate(MP.build_panel(frames), LV.load_spec(), member=_member(frames),
                    coverage=("2000-01-01", None), period="held_out")


def test_newey_west_alpha_recovers_a_known_intercept():
    rng = np.random.default_rng(5)
    x = rng.normal(0.01, 0.04, 400)
    y = 0.004 + 0.7 * x + rng.normal(0, 0.01, 400)
    reg = LV.newey_west_alpha(y, x)
    assert reg["alpha"] == pytest.approx(0.004, abs=0.0015) and reg["beta"] == pytest.approx(0.7, abs=0.05)
    assert reg["t"] > 3


def test_a_spec_edit_refuses_to_evaluate(monkeypatch, tmp_path):
    edited = tmp_path / "spec.json"
    edited.write_text(LV.SPEC_PATH.read_text().replace('"min_names": 20', '"min_names": 10'))
    assert LV.spec_sha256(edited) != LV.FROZEN_SHA256
    monkeypatch.setattr(LV, "SPEC_PATH", edited)
    monkeypatch.setattr(LV, "spec_sha256", lambda path=edited: LV.hashlib.sha256(edited.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="changed after freezing"):
        LV.run(tmp_path)
