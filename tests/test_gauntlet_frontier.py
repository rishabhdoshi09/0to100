import json

import pandas as pd
import pytest

from gauntlet.frontier import observations
from gauntlet.validator import scoped_equity_frames


@pytest.mark.parametrize(("factor", "price", "bridged"), [(2, 50, 1), (3, 50, 0), (1.1, 100, 0)])
def test_ordinary_reference_and_ohlc_corroborate_only_known_share_actions(factor, price, bridged):
    days = pd.bdate_range("2025-04-01", periods=3)
    header = "SYMBOL,SERIES,DATE1,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,TTL_TRD_QNTY,PREV_CLOSE\n"

    def payload(i, price, previous):
        return (header + f"TCS,EQ,{days[i].date()},{price},{price},{price},{price},10000000,{previous}\n").encode()

    frames, report = scoped_equity_frames(
        zip(days.date, [payload(0, 100, 100), payload(1, price, 100), payload(2, price+1, price)]),
        ["TCS"], {"TCS": [{"ex_date": days[1], "factor": factor, "type": "split"}]},
        bridge_share_actions=True,
    )
    assert report["bridged_share_actions"] == bridged
    assert (days[1] in frames["TCS"].index) is bool(bridged)
    if bridged:
        assert frames["TCS"].loc[days[1], ["open", "share_factor"]].tolist() == [50, 2]


def test_frontier_payoff_preserves_share_units_and_exact_exchange_sessions():
    dates = pd.bdate_range("2025-04-01", periods=44)
    days = dates.strftime("%Y-%m-%d").tolist()
    frame = pd.DataFrame({"open": 100., "close": 100., "volume": 2e6,
                          "deliv_per": 40., "share_factor": 1.}, index=dates)
    frame.loc[dates[22]:, ["open", "close"]] = 50.
    frame.loc[dates[22], "share_factor"] = 2.
    frame.loc[dates[16]:dates[20], "deliv_per"] = 55.
    (_, features, outcomes), = observations(frame, days, [days[20]])
    assert json.loads(features)["delivery_change"] == 15
    assert json.loads(features)["momentum"] == 0
    assert json.loads(outcomes) == {"3": 0, "20": 0}
    (_, _, no_gap_fill), = observations(frame.drop(dates[22]), days, [days[20]])
    assert json.loads(no_gap_fill) == {}
    (_, buy_on_ex_day, ex_day_payoff), = observations(frame, days, [days[21]])
    assert json.loads(ex_day_payoff)["3"] == 0
    assert json.loads(buy_on_ex_day)["momentum"] == 0
    (_, _, reserved), = observations(frame, days, [days[40]])
    assert json.loads(reserved) == {}



def test_frontier_target_cannot_be_the_base_checkpoint(monkeypatch):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from gauntlet.frontier import build
    from research import registry

    with TemporaryDirectory(dir="/tmp") as scratch:
        root = Path(scratch).resolve()
        monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
        monkeypatch.setattr(registry, "_DB_PATH", root / "experiments.sqlite")
        source = root / "gauntlet-frontier.sqlite"
        source.write_bytes(b"original")
        with pytest.raises(ValueError, match="aliases"):
            build(source, root)
        assert source.read_bytes() == b"original"


def test_frontier_failure_is_spent_and_next_slot_remains_available(monkeypatch):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from research import registry
    from research.autonomy import frontier_discovery as director

    with TemporaryDirectory(dir="/tmp") as scratch:
        root = Path(scratch).resolve()
        monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
        monkeypatch.setattr(registry, "_DB_PATH", root / "experiments.sqlite")
        source = root / "base.sqlite"
        source.write_bytes(b"original")
        digest = director.base._digest(source, director.base._identity(source))
        monkeypatch.setattr(director.previous, "rows", lambda: [
            ({"result": json.dumps({"source_sha256": digest, "status": "REJECTED"})}, {"family": family})
            for family in ("continuation", "liquidity", "overnight")
        ])
        target = root / "frontier.sqlite"
        target.write_bytes(b"")
        days = ["2025-04-01", director.base.END]
        monkeypatch.setattr(director.frontier, "build", lambda *_: (target, days, {"bridged_share_actions": 0}))

        def fail(*_):
            raise ValueError("test measurement error")

        monkeypatch.setattr(director, "_measure", fail)
        result = director.run(source, root, 1)
        assert result["decision"] == "DISCOVERY_FAILED"
        assert result["promotion_eligible"] is False
        assert result["status"] == "INCONCLUSIVE"
        assert director.scheduler_key().endswith(":2")
        assert director.run(source, root, 1)["decision"] == "DISCOVERY_ALREADY_COMPLETED"
